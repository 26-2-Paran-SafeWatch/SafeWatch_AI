"""위험 판단 결과 데이터 구조. docs/pipeline-architecture.md 3.6 참고."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskAssessment:
    track_id: int
    score: int                          # 0 ~ 100
    level: str                          # low / medium / high
    types: list[str]                    # 충족된 단서 목록 (risk-criteria.md 1.4)
    indicators: dict[str, float] = field(default_factory=dict)  # 지표별 측정값
    should_emit_event: bool = False     # True는 판정 전이(비활성→활성) 시점에만 — 매 프레임 True가 아니다
