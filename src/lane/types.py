"""차선 모델 데이터 구조. docs/pipeline-architecture.md 3.3 참고."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class LaneModel:
    left_fit: Optional[np.poly1d]      # 2차 다항식 계수 (bird's eye view 좌표계) — opencv 경로만
    right_fit: Optional[np.poly1d]
    lane_width_px: Optional[float]     # 하단 차로 폭(px). opencv = 조감도 기준, ufld = 원본 화면 기준
    curvature_radius: Optional[float]  # 미터. TODO: 필요 시 산출 로직 추가
    confidence: float                  # 0.0 ~ 1.0
    valid: bool                        # False면 상위 로직은 offset 지표를 제외한다 (판단 보류)
    fit_residual_ratio: Optional[float] = None
    """RMS 피팅 잔차 / 차로 폭. 신뢰도 산출 근거이자 튜닝용 관측값 —
    피팅 자체가 실패한 경우에만 None이다. 값이 클수록 선이 아니라
    노면 텍스처를 피팅했을 가능성이 높다 (lane.confidence.max_residual_ratio)."""

    space: Literal["bev", "image"] = "bev"
    """차선 좌표계. "bev" = 조감도 다항식(opencv 경로), "image" = 원본 화면 점(ufld 경로,
    docs/lane-detection-dl-plan.md D3). METRICS는 이 값으로 offset 계산 방식을 고른다."""
    left_points: Optional[np.ndarray] = None
    """image 좌표계 자차 왼쪽 차선 점 (N, 2) [x, y], y 오름차순. 원본 프레임 픽셀."""
    right_points: Optional[np.ndarray] = None
    measure_y_range: Optional[tuple[float, float]] = None
    """image 좌표계에서 offset을 잴 수 있는 y 구간 (위, 아래) 픽셀. 차량 바닥이 이 밖이면 재지 않는다
    (lane.ufld.measure_range_ratio). 조감도 경로의 '보정 구간 밖 제외'와 같은 역할.
    안전장치 판정 구간(lane.ufld.guard_range_ratio)과는 따로다 — 측정 구간이 더 멀리까지 간다."""
    reject_reason: Optional[str] = None
    """valid=False인 이유 (로그·튜닝용). ufld 경로의 안전장치가 채운다."""
