"""bbox 크기 변화율 기반 상대속도(팽창률) 추정.

`docs/risk-criteria.md` 1.4 "이유 없는 가감속 / 속도 불규칙" (검토 후보)의
실현 가능성을 확인하기 위한 모듈. NHTSA 속도·제동 문제 범주는 판단
대상(전방 차량)의 절대 속도가 아니라 **거동**을 본다 — 본 시스템은 GPS로
자차 속도는 알지만 전방 차량의 절대 속도는 모른다. 단안 카메라로 직접
측정 가능한 것은 상대 거리의 변화율(팽창률)뿐이다.

**원리 (단안 looming)** — bbox 높이 h는 거리 Z에 반比례한다(h ∝ 1/Z,
핀홀 카메라 모델). 로그를 취해 미분하면:

    e(t) = d(ln h)/dt = -Z'(t)/Z(t) = v_rel / Z(t) = 1 / TTC(t)

e(t)는 스케일(실제 차량 크기)에 무관한 **팽창률**이며, 그 역수가
time-to-collision이다. e > 0이면 거리가 좁혀지는 중(추돌 방향),
e < 0이면 멀어지는 중이다. 절대 상대속도(m/s)가 아니라 **거리 대비
상대속도 비율(1/s)**이라는 점에 유의 — 이것으로 충분한 이유는
"급감속"은 관측 시점의 거리와 무관하게 팽창률 자체가 비정상적으로
커지는 사건이기 때문이다.

**한계 — 스케일 비결정성** — 단안 카메라로는 실제 차간거리(Z)나 상대
속도(v_rel)를 절대값으로 복원할 수 없다. e(t)만 알 수 있다. IMU/GPS로
자차 속도를 더하면 Z를 추정할 수 있지만(자차속도 - 상대속도 = 전방
차량 속도), 이는 별도 확장이며 현재 범위 밖이다.
"""

from __future__ import annotations

import numpy as np


def expansion_rate(
    timestamps: np.ndarray, heights: np.ndarray
) -> float | None:
    """구간 내 팽창률 e(t) = d(ln h)/dt를 최소자승 직선 기울기로 추정한다.

    ln(h)에 대한 선형 회귀를 쓰는 이유는 lane_detector의 2차 다항식
    피팅과 같다 — 개별 샘플의 검출 잡음에 덜 민감하고, 구간 평균적인
    변화 추세만 뽑아낸다.

    Returns:
        기울기(1/s). 샘플이 2개 미만이거나 높이에 0 이하 값이 있으면 None.
    """
    if len(timestamps) < 2 or np.any(heights <= 0):
        return None
    log_h = np.log(heights)
    # np.polyfit(x, y, 1) → [기울기, 절편]
    slope, _ = np.polyfit(timestamps, log_h, 1)
    return float(slope)


def rolling_expansion_rate(
    timestamps: np.ndarray, heights: np.ndarray, window: int
) -> np.ndarray:
    """길이 `window`인 슬라이딩 윈도우로 팽창률 시계열을 산출한다.

    앞쪽 `window - 1`개는 윈도우를 채우지 못해 NaN이다.
    """
    n = len(timestamps)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        e = expansion_rate(timestamps[i - window + 1 : i + 1], heights[i - window + 1 : i + 1])
        result[i] = e if e is not None else np.nan
    return result
