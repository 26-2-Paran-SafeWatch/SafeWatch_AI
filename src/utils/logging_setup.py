"""로깅 설정.

AGENTS.md 규칙 — `print()` 대신 `logging`을 사용하고, 디버그 로그는 레벨로
제어한다 (Pi5 장시간 구동 시 과도한 로그가 성능에 영향을 줄 수 있음).
"""

from __future__ import annotations

import logging

from src.utils.config import Config


def setup_logging(cfg: Config) -> None:
    level_name = cfg.logging.level if "logging" in cfg else "INFO"
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
