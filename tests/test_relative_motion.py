"""bbox 팽창률(상대속도 대용 지표) 실현 가능성 검증.

`docs/risk-criteria.md` 1.4 "이유 없는 가감속 / 속도 불규칙" (검토 후보)의
S3 실현 가능성 확인 항목. 실제 대시캠 영상이 없어 합성 데이터로
신호 대 잡음비를 정량 확인한다 — lane_detector의 보도블록 오탐을
합성 이미지로 재현했던 것과 같은 접근이다.

**시나리오** 4초 평시 주행(e=0) 후 1.5초간 전방 차량이 급감속(closing,
e=event_e 지속) → 2초 회복. bbox 높이에 가우시안 픽셀 잡음을 더해
`rolling_expansion_rate`로 복원했을 때, 잡음만 있는 구간의 최댓값과
실제 이벤트 구간의 평균값이 충분히 분리되는지 확인한다.

**픽셀 잡음 가정(1.5~3.0px)은 실측치가 아니라 가정치다** — YOLOv8n이
Pi5 해상도에서 내는 실제 bbox 경계 잡음은 대시캠 영상 확보 후
재검증해야 한다.

**결론 (2026-09-26 기준)** — 창 1초(15샘플) 기준, e ≥ 0.2/s(= TTC ≤ 5초)
지속 이벤트는 가정한 잡음 범위 전체에서 잡음 최댓값 대비 2배 이상
여유로 분리된다. e < 0.1/s(TTC > 10초, 완만한 변화)은 잡음과 구분이
어렵다. 즉 **"완만한 속도 변화"는 못 잡지만 "급격한 접근"은 잡을 수
있다** — 이는 NHTSA 단서가 원래 "이유 없는 급가감속"을 가리키는 것과
방향이 맞다. **결론: 조건부 채택 권장** — 이벤트 스키마·threshold
반영은 별도 결정 필요 (사용자 확인 후 risk-criteria.md 1.4/3.1 갱신).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.relative_motion import expansion_rate, rolling_expansion_rate


def _simulate(
    h0: float,
    px_noise: float,
    event_e: float,
    event_sec: float,
    window: int,
    fps: float = 15.0,
    seed: int = 0,
    base1_sec: float = 4.0,
    base2_sec: float = 2.0,
):
    """평시(e=0) → 이벤트(e=event_e) → 회복(e=0) 시나리오를 생성해 추정한다.

    Returns (baseline_worst_abs_e, event_mean_e) — 이벤트/평시 구간과 겹치는
    윈도우는 양쪽 계산에서 제외한다 (경계 오염 방지).
    """
    dt = 1.0 / fps
    n_base1 = int(base1_sec * fps)
    n_event = int(event_sec * fps)
    n_base2 = int(base2_sec * fps)
    n = n_base1 + n_event + n_base2

    e_true = np.zeros(n)
    e_true[n_base1 : n_base1 + n_event] = event_e
    t = np.arange(n) * dt
    h_true = h0 * np.exp(np.cumsum(e_true) * dt)

    rng = np.random.default_rng(seed)
    h_obs = h_true + rng.normal(0, px_noise, n)
    e_est = rolling_expansion_rate(t, h_obs, window)

    pre = e_est[window - 1 : n_base1]
    post_start = n_base1 + n_event + window - 1
    post = e_est[post_start:] if post_start < n else np.array([])
    baseline_worst = np.nanmax(np.abs(np.concatenate([pre, post])))

    lo, hi = n_base1 + window - 1, n_base1 + n_event
    event_vals = e_est[lo:hi] if hi > lo else np.array([])
    event_mean = float(np.nanmean(event_vals)) if event_vals.size else float("nan")

    return float(baseline_worst), event_mean


def test_expansion_rate_recovers_constant_slope_noiseless():
    """잡음 없는 이상적인 경우 — 이론값을 정확히 복원해야 한다."""
    t = np.linspace(0, 1, 15)
    true_e = 0.4
    h = 60.0 * np.exp(true_e * t)
    assert expansion_rate(t, h) == pytest.approx(true_e, abs=1e-9)


def test_expansion_rate_none_on_insufficient_or_invalid_input():
    assert expansion_rate(np.array([0.0]), np.array([60.0])) is None
    assert expansion_rate(np.array([0.0, 1.0]), np.array([60.0, -1.0])) is None


@pytest.mark.parametrize("h0", [60.0, 150.0])
@pytest.mark.parametrize("px_noise", [1.5, 2.0, 3.0])
def test_sudden_event_separated_from_noise_floor(h0: float, px_noise: float):
    """급감속형 이벤트(e=0.3/s, TTC≈3.3초)는 가정 잡음 범위 전체에서 검출 가능해야 한다.

    window=15(1초), event_sec=1.5초 — sprint-plan.md S3 "실현 가능성 확인" 결론.
    """
    baseline_worst, event_mean = _simulate(
        h0=h0, px_noise=px_noise, event_e=0.3, event_sec=1.5, window=15
    )
    assert event_mean > baseline_worst * 2, (
        f"h0={h0} noise={px_noise}: baseline_worst={baseline_worst:.3f} "
        f"event_mean={event_mean:.3f} — 분리 부족"
    )


def test_mild_event_not_reliably_separable_from_noise():
    """완만한 변화(e=0.05/s, TTC=20초)는 잡음과 구분되지 않는다 — 한계를 명시적으로 고정.

    이 테스트가 실패하게 되면(=분리가 잘 된다면) 잡음 가정이 바뀐 것이니
    docstring의 "완만한 변화는 못 잡는다"는 결론도 함께 재검토해야 한다.
    """
    baseline_worst, event_mean = _simulate(
        h0=60.0, px_noise=2.0, event_e=0.05, event_sec=1.5, window=15
    )
    assert event_mean < baseline_worst * 2


def test_short_event_needs_shorter_window():
    """0.7초의 짧은 이벤트는 1초 창(window=15)보다 짧은 창(window=8)이 필요하다."""
    long_window_worst, long_window_event = _simulate(
        h0=60.0, px_noise=1.5, event_e=0.6, event_sec=0.7, window=15
    )
    short_window_worst, short_window_event = _simulate(
        h0=60.0, px_noise=1.5, event_e=0.6, event_sec=0.7, window=8
    )
    # window=15는 0.7초짜리 이벤트 전체를 담지 못해 이벤트 전용 구간이 없다(NaN).
    assert np.isnan(long_window_event)
    # window=8이면 이벤트 구간을 온전히 담는 윈도우가 존재해 분리가 가능하다.
    assert short_window_event > short_window_worst * 2
