"""SafeWatch AI 파이프라인 진입점.

연결된 구간 — INPUT · DETECTION · LANE · TRACKING · METRICS · RISK · EVENT(JSON 생성).

**역할 나누기 (2026-11-15 HW 파트와 합의)** — AI는 이벤트 JSON을 만들어 HW에 넘기는
데까지만 한다. 이벤트가 확정되면 envelope(event / trigger / target)를 만들어
`EventSink.publish()`에 넘긴다 (docs/event-schema.md "AI → HW 전달 형식(envelope)").
클립 자르기·location·큐·재시도·서버 전송은 HW 파트가 자기 sink 안에서 한다.

**실기기 연결** — Pi 카메라는 한 프로세스만 열 수 있어서, HW 실행 스크립트가 카메라를 한 번
열어 H.264는 링버퍼로, RGB 프레임은 AI로 나눠 준다. 그래서 HW 스크립트가 `run()`을 import해
자기 FrameSource(`frames()`가 `Frame`을 내고 `close()`가 있음)와 EventSink를 넘긴다.
넘기지 않으면 `create_source()`와 `FileEventSink(event.sink_dir)`를 쓴다 (PC 개발용).

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

from src.detection.detector import VehicleDetector
from src.event.builder import EventIdGenerator, build_event_envelope, build_event_metadata
from src.event.sink import EventSink, FileEventSink
from src.input.sensors import create_sensor_source
from src.input.source import FrameSource, create_source
from src.lane import create_lane_detector
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


def run(
    config_path: str,
    source: str,
    frame_source: FrameSource | None = None,
    event_sink: EventSink | None = None,
) -> None:
    """파이프라인 실행. HW 실행 스크립트는 자기 카메라(`frame_source`)와 sink(`event_sink`)를 넘긴다.

    `source`는 넘긴 frame_source가 있어도 envelope의 trigger.source에 쓰인다 — 카메라면 "camera".
    넘긴 frame_source·event_sink도 끝날 때 이 함수가 `close()`한다.
    """
    cfg = load_config(config_path)
    setup_logging(cfg)
    logger.info("설정 로드 완료: %s", config_path)

    if frame_source is None:
        frame_source = create_source(source, resolution=tuple(cfg.input.resolution))
    if event_sink is None:
        event_sink = FileEventSink(cfg.event.sink_dir)
    logger.info("영상 소스 시작: %s", source)

    detector = VehicleDetector(cfg)
    detection_interval = cfg.detection.interval
    lane_detector = create_lane_detector(cfg)   # lane.method: opencv | ufld
    tracker = VehicleTracker(cfg)
    # opencv 경로 — METRICS는 차선 모델과 같은 BEV 좌표계에서 차량 위치를 봐야 하므로
    # LANE이 쓰는 perspective 변환기를 공유한다. ufld 경로는 원본 화면 기준이라 None이다.
    offset_calculator = OffsetCalculator(cfg, lane_detector.perspective)
    risk_scorer = RiskScorer(cfg)
    frame_quality_checker = FrameQualityChecker(cfg)
    event_id_generator = EventIdGenerator(cfg.event.device_id)
    sensor_source = create_sensor_source(cfg)
    # 차선 검출의 차량 영역 제외용 — 검출 주기(interval) 사이 프레임은 직전 검출 박스를
    # 그대로 쓴다. 몇 프레임 동안 차가 움직이는 거리는 박스 여유(pad_ratio)보다 작다.
    last_boxes: list = []

    try:
        for frame in frame_source.frames():
            with profiler.stage("input"):
                pass  # 프레임은 이미 획득됨. 측정 대상은 향후 전처리 단계.

            # 검출 주기 분리 — N프레임마다 1회 실행 (pipeline-architecture.md 3.2).
            # ⚠️ interval > 1은 아직 실사용 대상이 아니다. interpolate()가 위치를
            #    전진시키지 않아 횡방향 속도가 왜곡된다 (S6 항목, tracker.py 참고).
            if frame.frame_id % detection_interval == 0:
                boxes = detector.detect(frame.image)
                last_boxes = boxes
                tracks = tracker.update(boxes)
            else:
                tracks = tracker.interpolate()

            frame_quality_ok = frame_quality_checker.check(frame.image)

            # LANE을 먼저 돌려야 perspective 행렬이 만들어져 METRICS가 좌표를
            # 변환할 수 있다 (opencv 경로).
            lane = lane_detector.detect(
                frame.image, vehicle_boxes=last_boxes, timestamp=frame.timestamp
            )
            if not lane.valid and frame.frame_id % 30 == 0:
                logger.debug(
                    "frame_id=%d 차선 보류(신뢰도 %.2f, 잔차 %s, 이유 %s) — offset 지표 제외",
                    frame.frame_id, lane.confidence, lane.fit_residual_ratio, lane.reject_reason,
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
                    event = build_event_metadata(
                        event_id=event_id_generator.next(frame.timestamp),
                        device_id=cfg.event.device_id,
                        timestamp=frame.timestamp,
                        assessment=assessment,
                        cfg=cfg,
                        model_version=cfg.event.model_version,
                        rule_version=cfg.event.rule_version,
                    )
                    # 이 줄 형식은 HW 테스트 도구가 파싱한다 — 바꾸지 말 것
                    logger.info(
                        "frame_id=%d 음주운전 의심 거동 감지 — %s",
                        frame.frame_id, json.dumps(event, ensure_ascii=False),
                    )
                    _publish(event_sink, build_event_envelope(event, frame, track, ts, source))
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
        lane_detector.close()   # NPU(lane.ufld.backend: hailo) 자원 해제
        event_sink.close()
        summary = profiler.summary()
        if summary:
            logger.info("단계별 평균 처리 시간(ms): %s", summary)


def _publish(sink: EventSink, envelope: dict) -> None:
    """HW sink가 예외를 내도 파이프라인은 계속 돈다."""
    try:
        sink.publish(envelope)
    except Exception:  # noqa: BLE001
        logger.exception("이벤트 전달 실패 — event_id=%s", envelope["event"]["event_id"])


def main() -> None:
    args = parse_args()
    run(config_path=args.config, source=args.source)


if __name__ == "__main__":
    main()
