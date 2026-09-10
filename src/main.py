"""SafeWatch AI 파이프라인 진입점.

현재는 스캐폴딩 단계 — INPUT 모듈만 연결되어 있고, DETECTION/LANE 이후
단계는 아직 구현되지 않았다. 각 스프린트 진행에 따라 파이프라인을 채운다
(docs/pipeline-architecture.md 참고).

사용법
    python -m src.main --config configs/dev.yaml --source data/raw/sample.mp4
    python -m src.main --config configs/pi5.yaml --source camera
"""

from __future__ import annotations

import argparse
import logging

from src.input.source import create_source
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

    try:
        for frame in frame_source.frames():
            with profiler.stage("input"):
                pass  # 프레임은 이미 획득됨. 측정 대상은 향후 전처리 단계.

            # TODO(S1~S3): detection → lane → tracking → metrics → risk → event
            if frame.frame_id % 30 == 0:
                logger.debug(
                    "frame_id=%d timestamp=%.3f", frame.frame_id, frame.timestamp
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
