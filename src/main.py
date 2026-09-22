"""SafeWatch AI 파이프라인 진입점.

현재 연결된 구간 — INPUT · DETECTION · LANE · TRACKING · METRICS.
RISK 이후(판정·이벤트 생성)는 판단 기준 확정 대기 중이라 미구현이다
(docs/risk-criteria.md 1.4, 지도교수 확인 후 S4). 각 스프린트 진행에 따라
파이프라인을 채운다 (docs/pipeline-architecture.md 참고).

사용법
    python -m src.main --config configs/dev.yaml --source data/raw/sample.mp4
    python -m src.main --config configs/pi5.yaml --source camera
"""

from __future__ import annotations

import argparse
import logging

from src.detection.detector import VehicleDetector
from src.input.source import create_source
from src.lane.lane_detector import LaneDetector
from src.metrics.offset import OffsetCalculator
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

            # TODO(S4): risk → event. 판단 기준 확정 후 착수 (risk-criteria.md 1.4)
            if frame.frame_id % 30 == 0 and series:
                for track_id, ts in series.items():
                    latest = ts.latest
                    if latest is None:
                        continue
                    logger.debug(
                        "frame_id=%d track=%d offset=%.2fm(%.1f%%) v_lat=%.2fm/s n=%d",
                        frame.frame_id, track_id, latest.offset_m,
                        latest.offset_ratio * 100, latest.lateral_velocity_mps, len(ts),
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
