"""차선 모델 데이터 구조. docs/pipeline-architecture.md 3.3 참고."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LaneModel:
    left_fit: Optional[np.poly1d]      # 2차 다항식 계수 (bird's eye view 좌표계)
    right_fit: Optional[np.poly1d]
    lane_width_px: Optional[float]     # bird's eye view 기준 하단 차로 폭(px)
    curvature_radius: Optional[float]  # 미터. TODO: 필요 시 산출 로직 추가
    confidence: float                  # 0.0 ~ 1.0
    valid: bool                        # False면 상위 로직은 offset 지표를 제외한다 (판단 보류)
    fit_residual_ratio: Optional[float] = None
    """RMS 피팅 잔차 / 차로 폭. 신뢰도 산출 근거이자 튜닝용 관측값 —
    피팅 자체가 실패한 경우에만 None이다. 값이 클수록 선이 아니라
    노면 텍스처를 피팅했을 가능성이 높다 (lane.confidence.max_residual_ratio)."""
