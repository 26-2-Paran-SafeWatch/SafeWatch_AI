"""단계별 처리 시간 측정.

사용 예 (AGENTS.md 참고):

    with profiler.stage("detection"):
        boxes = detector.detect(frame)

Pi5는 성능 예산이 빠듯하므로(총 100ms/프레임 목표, docs/pipeline-architecture.md
4장), 무거운 연산을 추가할 때는 반드시 이 컨텍스트 매니저로 감싼다.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("safewatch.profiler")


class Profiler:
    def __init__(self) -> None:
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._totals[name] += elapsed_ms
            self._counts[name] += 1
            logger.debug("[stage:%s] %.2fms", name, elapsed_ms)

    def summary(self) -> dict[str, float]:
        """단계별 평균 처리 시간(ms)."""
        return {
            name: self._totals[name] / self._counts[name]
            for name in self._totals
            if self._counts[name] > 0
        }

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()


# 모듈 전역에서 공유하는 기본 프로파일러 인스턴스.
profiler = Profiler()
