"""원본 화면(image 좌표계) 차선 점에서 특정 행의 x를 읽는 공용 함수.

ufld 경로의 안전장치(`src/lane/ufld_guard.py`)와 offset 계산(`src/metrics/offset.py`)이
같은 방식으로 차선 위치를 읽어야 하므로 한곳에 둔다. 노트북
`safewatch_lane_calib_v2.ipynb` 20번 셀 `x_bottom`과 같은 동작이다.
"""

from __future__ import annotations

import numpy as np


def x_at_row(points: np.ndarray | None, y: float) -> float | None:
    """y 행에서 차선의 x. 점 범위 안이면 선형 보간, 밖이면 가까운 쪽 절반 점으로 직선 연장.

    UFLD 행 앵커는 화면 아래 끝까지 오지 않는 경우가 많아(앞차 가림·보닛) 차량 바닥 높이가
    점 범위 밖일 수 있다. 원근 화면에서 차선은 짧은 구간에서 직선에 가까우므로, 가까운 쪽 절반
    점으로 직선을 맞춰 연장한다. 점이 2개 미만이거나 세로 폭이 1px 미만이면 None.
    """
    if points is None or len(points) < 2:
        return None
    p = points[np.argsort(points[:, 1])]
    ys, xs = p[:, 1], p[:, 0]
    if ys[0] <= y <= ys[-1]:
        return float(np.interp(y, ys, xs))
    half = len(p) // 2
    seg = p[half:] if y > ys[-1] else p[: max(half, 2)]
    if np.ptp(seg[:, 1]) < 1:
        seg = p
        if np.ptp(seg[:, 1]) < 1:
            return None
    slope, intercept = np.polyfit(seg[:, 1], seg[:, 0], 1)
    return float(slope * y + intercept)


def lane_at_row(
    left: np.ndarray | None, right: np.ndarray | None, y: float
) -> tuple[float, float] | None:
    """y 행의 (왼쪽 x, 오른쪽 x). 하나라도 못 읽거나 폭이 1px 이하면 None."""
    a, b = x_at_row(left, y), x_at_row(right, y)
    if a is None or b is None or b - a <= 1:
        return None
    return a, b
