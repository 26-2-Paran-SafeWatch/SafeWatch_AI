"""SafeWatch AI 파이프라인 진입점.

현재 연결된 구간 — INPUT · DETECTION · LANE · TRACKING · METRICS · RISK ·
EVENT(메타데이터 + 로컬 큐 적재). 이벤트 발생 시 JSON 메타데이터를 만들어
로컬 SQLite 큐(`src/event/queue.py`)에 저장하고 로그로도 남긴다. **영상
클립 추출은 아직 없다** — 링 버퍼 요청 인터페이스가 HW 파트와 협의 중이라
`NullClipProvider`(`src/event/clip.py`)가 항상 클립 없음을 반환한다
(docs/system-architecture.md "아직 남은 것"). 실제 인터페이스가 확정되면
이 자리에서 `ClipProvider` 구현체만 교체하면 된다. 각 스프린트 진행에
따라 파이프라인을 채운다 (docs/pipeline-architecture.md 참고).

**GPS 속도** — RISK의 `lane_departure`는 자차 속도 게이팅(LDWS 60km/h
근거, risk-criteria.md 1.3)이 필요하다. `src/input/sensors.py`로 입력
경로를 만들었지만, 실제 ESP32 UART 프로토콜은 HW 파트와 아직 협의되지
않아 `SerialGPSIMUSource`는 미구현이다 — `configs/pi5.yaml`은 당분간
`sensors.source: unavailable`(default.yaml 기본값)을 그대로 써서 항상
`ego_speed_kmh=None`을 넘기고, 그 결과 `lane_departure`는 지금도 절대
충족되지 않는다. `configs/dev.yaml`만 `sensors.source: dummy`로 override해
PC에서 이 경로 자체는 테스트할 수 있게 해 뒀다.

사용법
    python -m src.main --config configs/dev.yaml --source data/raw/sample.mp4
    python -m src.main --config configs/pi5.yaml --source camera
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.detection.detector import VehicleDetector
from src.event.builder import EventIdGenerator, build_event_metadata
from src.event.clip import NullClipProvider
from src.event.queue import EventQueue
from src.input.sensors import create_sensor_source
from src.input.source import create_source
from src.lane.lane_detector import LaneDetector
from src.metrics.offset import OffsetCalculator
from src.risk.frame_quality import FrameQualityChecker
from src.risk.scorer import RiskScorer
from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeWatch AI 파이프라인")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument(
        "--source", required=True, help="영상 파일 경로 또는 'camera' (Pi5)"
    )
    return parser.parse_args()


def run(config_path: str, source: str) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)
    logger.info("설정 로드 완료: %s", config_path)

    resolution = tuple(cfg.input.resolution)
    frame_source = create_source(source, resolution=resolution)
    logger.info("영상 소스 시작: %s", source)

    detector = VehicleDetector(cfg)
    detection_interval = cfg.detection.interval
    lane_detector = LaneDetector(cfg)
    tracker = VehicleTracker(cfg)
    # METRICS는 차선 모델과 같은 BEV 좌표계에서 차량 위치를 봐야 하므로
    # LANE이 쓰는 perspective 변환기를 공유한다.
    offset_calculator = OffsetCalculator(cfg, lane_detector.perspective)
    risk_scorer = RiskScorer(cfg)
    frame_quality_checker = FrameQualityChecker(cfg)
    event_id_generator = EventIdGenerator(cfg.event.device_id)
    clip_provider = NullClipProvider()  # HW 링 버퍼 인터페이스 확정 전까지 항상 클립 없음
    event_queue = EventQueue(Path(cfg.event.queue_dir) / "events.db")
    sensor_source = create_sensor_source(cfg)

    try:
        for frame in frame_source.frames():
            with profiler.stage("input"):
                pass  # 프레임은 이미 획득됨. 측정 대상은 향후 전처리 단계.

            # 검출 주기 분리 — N프레임마다 1회 실행 (pipeline-architecture.md 3.2).
            # ⚠️ interval > 1은 아직 실사용 대상이 아니다. interpolate()가 위치를
            #    전진시키지 않아 횡방향 속도가 왜곡된다 (S6 항목, tracker.py 참고).
            if frame.frame_id % detection_interval == 0:
                boxes = detector.detect(frame.image)
                tracks = tracker.update(boxes)
            else:
                tracks = tracker.interpolate()

            frame_quality_ok = frame_quality_checker.check(frame.image)

            # LANE을 먼저 돌려야 perspective 행렬이 만들어져 METRICS가 좌표를
            # 변환할 수 있다.
            lane = lane_detector.detect(frame.image)
            if not lane.valid and frame.frame_id % 30 == 0:
                logger.debug(
                    "frame_id=%d 차선 신뢰도 미달(%.2f, 잔차 %s) — offset 지표 제외",
                    frame.frame_id, lane.confidence, lane.fit_residual_ratio,
                )

            series = offset_calculator.update(tracks, lane, frame.timestamp)
            sensor_reading = sensor_source.read(frame.timestamp)

            risk_scorer.sync({t.track_id for t in tracks})
            for track in tracks:
                ts = series.get(track.track_id)
                if ts is None:
                    continue
                assessment = risk_scorer.assess(
                    track, ts, lane_valid=lane.valid, timestamp=frame.timestamp,
                    ego_speed_kmh=sensor_reading.gps_speed_kmh, frame_quality_ok=frame_quality_ok,
                )
                if assessment.should_emit_event:
                    clip_result = clip_provider.request_clip(
                        frame.timestamp, cfg.event.clip_pre_sec, cfg.event.clip_post_sec
                    )
                    event = build_event_metadata(
                        event_id=event_id_generator.next(frame.timestamp),
                        device_id=cfg.event.device_id,
                        timestamp=frame.timestamp,
                        assessment=assessment,
                        series=ts,
                        track=track,
                        cfg=cfg,
                        model_version=cfg.event.model_version,
                        rule_version=cfg.event.rule_version,
                        # location은 GPS 미연결로 아직 없음 (build_event_metadata docstring)
                        clip=clip_result.to_schema_dict() if clip_result else None,
                    )
                    event_queue.enqueue(
                        event, clip_path=clip_result.local_path if clip_result else None
                    )
                    logger.info(
                        "frame_id=%d 음주운전 의심 거동 감지 — %s",
                        frame.frame_id, json.dumps(event, ensure_ascii=False),
                    )
                    # TODO(S4): 실제 전송(HTTP 업로드) — event_queue.pending()에서 꺼내 전송 후 mark_sent()
                elif frame.frame_id % 30 == 0:
                    logger.debug(
                        "frame_id=%d track=%d offset=%.2fm(%.1f%%) v_lat=%.2fm/s score=%d",
                        frame.frame_id, track.track_id, ts.latest.offset_m,
                        ts.latest.offset_ratio * 100, ts.latest.lateral_velocity_mps,
                        assessment.score,
                    )
    except KeyboardInterrupt:
        logger.info("중단 요청 수신, 종료합니다.")
    finally:
        frame_source.close()
        sensor_source.close()
        event_queue.close()
        summary = profiler.summary()
        if summary:
            logger.info("단계별 평균 처리 시간(ms): %s", summary)


def main() -> None:
    args = parse_args()
    run(config_path=args.config, source=args.source)


if __name__ == "__main__":
    main()
