"""SafeWatch AI 파이프라인 진입점.

현재 연결된 구간 — INPUT · DETECTION · LANE · TRACKING · METRICS · RISK ·
EVENT(메타데이터만). 이벤트 발생 시 JSON 메타데이터는 만들어 로그로
남기지만, **영상 클립 추출·전송은 아직 없다** — 링 버퍼 요청 인터페이스가
HW 파트와 협의 중이라 미구현이다 (docs/system-architecture.md "아직
남은 것"). 각 스프린트 진행에 따라 파이프라인을 채운다
(docs/pipeline-architecture.md 참고).

**GPS 속도 미연결** — RISK의 `lane_departure`는 자차 속도 게이팅(LDWS
60km/h 근거, risk-criteria.md 1.3)이 필요한데 GPS 입력 자체가 아직
없다(HW 파트 확인 대기, pipeline-architecture.md 미결 사항). 지금은 항상
`ego_speed_kmh=None`을 넘겨 안전하게 판단 보류시킨다 — 즉 lane_departure는
GPS가 실제로 연결되기 전까지 절대 충족되지 않는다.

사용법
    python -m src.main --config configs/dev.yaml --source data/raw/sample.mp4
    python -m src.main --config configs/pi5.yaml --source camera
"""

from __future__ import annotations

import argparse
import json
import logging

from src.detection.detector import VehicleDetector
from src.event.builder import EventIdGenerator, build_event_metadata
from src.input.source import create_source
from src.lane.lane_detector import LaneDetector
from src.metrics.offset import OffsetCalculator
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
    event_id_generator = EventIdGenerator(cfg.event.device_id)

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

            # LANE을 먼저 돌려야 perspective 행렬이 만들어져 METRICS가 좌표를
            # 변환할 수 있다.
            lane = lane_detector.detect(frame.image)
            if not lane.valid and frame.frame_id % 30 == 0:
                logger.debug(
                    "frame_id=%d 차선 신뢰도 미달(%.2f, 잔차 %s) — offset 지표 제외",
                    frame.frame_id, lane.confidence, lane.fit_residual_ratio,
                )

            series = offset_calculator.update(tracks, lane, frame.timestamp)

            risk_scorer.sync({t.track_id for t in tracks})
            for track in tracks:
                ts = series.get(track.track_id)
                if ts is None:
                    continue
                # GPS 미연결 상태 — 위 모듈 docstring 참고. lane_departure는 항상 판단 보류된다.
                assessment = risk_scorer.assess(
                    track, ts, lane_valid=lane.valid, timestamp=frame.timestamp, ego_speed_kmh=None
                )
                if assessment.should_emit_event:
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
                        # location·clip은 GPS·링 버퍼 미연결로 아직 없음 (build_event_metadata docstring)
                    )
                    logger.info(
                        "frame_id=%d 음주운전 의심 거동 감지 — %s",
                        frame.frame_id, json.dumps(event, ensure_ascii=False),
                    )
                    # TODO(S4): 클립 요청(링 버퍼)·전송 큐 적재 — HW 인터페이스 확정 후
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
        summary = profiler.summary()
        if summary:
            logger.info("단계별 평균 처리 시간(ms): %s", summary)


def main() -> None:
    args = parse_args()
    run(config_path=args.config, source=args.source)


if __name__ == "__main__":
    main()
