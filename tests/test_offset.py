"""OffsetCalculator 테스트.

lane offset은 채택 단서 4개 전부의 입력이므로(risk-criteria.md 1.4), 부호
규약과 정규화가 틀리면 판정 전체가 틀어진다. 합성 차선 모델 위에 차량 위치를
놓고 산출값을 직접 검증한다.

실제 영상이 없으므로 perspective 변환은 항등에 가깝게 두고(아래 fixture),
offset 산출 로직 자체를 확인하는 데 집중한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.types import VehicleBox
from src.lane.types import LaneModel
from src.metrics.offset import OffsetCalculator
from src.tracking.types import TrackedVehicle
from src.utils.config import load_config

LANE_LEFT_X = 150.0
LANE_RIGHT_X = 490.0
LANE_CENTER_X = (LANE_LEFT_X + LANE_RIGHT_X) / 2   # 320
LANE_WIDTH_PX = LANE_RIGHT_X - LANE_LEFT_X          # 340
LANE_WIDTH_M = 3.5


class _IdentityPerspective:
    """원본 좌표를 그대로 BEV 좌표로 쓰는 가짜 변환기.

    실제 `PerspectiveTransformer`는 카메라 캘리브레이션(`src_points_ratio`)에
    의존하는데 그 값이 아직 placeholder라, 변환 품질과 offset 산출 로직을
    분리해서 보기 위해 항등 변환을 쓴다.
    """

    warp_size = (640, 720)

    def warp_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)


@pytest.fixture
def calculator() -> OffsetCalculator:
    return OffsetCalculator(load_config("configs/dev.yaml"), _IdentityPerspective())


def _lane(confidence: float = 0.9, valid: bool = True) -> LaneModel:
    """좌우가 수직인 합성 차선. 어느 y에서도 폭과 중심이 같다."""
    return LaneModel(
        left_fit=np.poly1d([0.0, 0.0, LANE_LEFT_X]),
        right_fit=np.poly1d([0.0, 0.0, LANE_RIGHT_X]),
        lane_width_px=LANE_WIDTH_PX,
        curvature_radius=None,
        confidence=confidence,
        valid=valid,
        fit_residual_ratio=0.01,
    )


def _vehicle(center_x: float, track_id: int = 1, bottom_y: float = 600,
             tracked_frames: int = 30, interpolated: bool = False) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        box=VehicleBox(
            x1=center_x - 40, y1=bottom_y - 60,
            x2=center_x + 40, y2=bottom_y,
            confidence=0.9, class_id=2,
        ),
        tracked_frames=tracked_frames,
        is_interpolated=interpolated,
    )


def test_vehicle_at_lane_center_has_zero_offset(calculator: OffsetCalculator):
    series = calculator.update([_vehicle(LANE_CENTER_X)], _lane(), timestamp=0.0)
    assert series[1].latest.offset_m == pytest.approx(0.0)
    assert series[1].latest.offset_ratio == pytest.approx(0.0)


def test_offset_sign_is_positive_to_the_right(calculator: OffsetCalculator):
    """부호 규약 — 우측이 양수 (risk-criteria.md 2.1)."""
    right = calculator.update([_vehicle(LANE_CENTER_X + 85)], _lane(), 0.0)[1].latest
    assert right.offset_m > 0

    calculator._forget_disappeared(set())  # 버퍼 초기화
    left = calculator.update([_vehicle(LANE_CENTER_X - 85)], _lane(), 0.0)[1].latest
    assert left.offset_m < 0


def test_offset_ratio_normalized_by_lane_width(calculator: OffsetCalculator):
    """차로 폭의 25% 지점에 놓으면 비율이 0.25, 미터값은 3.5m의 25%."""
    offset_px = LANE_WIDTH_PX * 0.25
    sample = calculator.update([_vehicle(LANE_CENTER_X + offset_px)], _lane(), 0.0)[1].latest
    assert sample.offset_ratio == pytest.approx(0.25, abs=1e-6)
    assert sample.offset_m == pytest.approx(0.25 * LANE_WIDTH_M, abs=1e-6)


def test_invalid_lane_appends_no_sample(calculator: OffsetCalculator):
    """차선 인식 실패 구간은 판정 보류 — 샘플을 쌓지 않는다 (risk-criteria.md 4.1)."""
    series = calculator.update([_vehicle(LANE_CENTER_X + 50)], _lane(valid=False), 0.0)
    assert series == {}


def test_vehicle_outside_warp_area_is_skipped(calculator: OffsetCalculator):
    """BEV 변환 영역을 벗어난 차량은 차선과 비교할 수 없어 제외한다."""
    series = calculator.update([_vehicle(LANE_CENTER_X, bottom_y=5000)], _lane(), 0.0)
    assert series == {}


def test_buffer_drops_samples_older_than_observation_window(calculator: OffsetCalculator):
    window = load_config("configs/dev.yaml").metrics.observation_window_sec
    for i in range(40):
        calculator.update([_vehicle(LANE_CENTER_X)], _lane(), timestamp=i * 0.5)

    series = calculator.series[1]
    assert series.observed_duration_sec <= window
    assert series.offsets[0].timestamp >= 40 * 0.5 - window - 0.5


def test_disappeared_vehicle_buffer_is_released(calculator: OffsetCalculator):
    calculator.update([_vehicle(LANE_CENTER_X)], _lane(), 0.0)
    assert 1 in calculator.series

    calculator.update([], _lane(), 0.1)  # 차량이 사라진 프레임
    assert calculator.series == {}


def test_lateral_velocity_positive_when_drifting_right(calculator: OffsetCalculator):
    """표류 — 한 방향으로 꾸준히 이동하면 횡방향 속도에 부호가 유지된다."""
    for i in range(20):
        calculator.update([_vehicle(LANE_CENTER_X + i * 4)], _lane(), timestamp=i * 0.1)

    latest = calculator.series[1].latest
    assert latest.lateral_velocity_mps > 0
    assert calculator.series[1].peak_lateral_velocity_mps > 0


def test_lateral_velocity_near_zero_when_centered(calculator: OffsetCalculator):
    for i in range(20):
        calculator.update([_vehicle(LANE_CENTER_X)], _lane(), timestamp=i * 0.1)
    assert calculator.series[1].latest.lateral_velocity_mps == pytest.approx(0.0, abs=1e-6)


def test_smoothing_suppresses_single_frame_outlier(calculator: OffsetCalculator):
    """단일 프레임 이상치가 그대로 반영되지 않는다 (pipeline-architecture.md 3.5)."""
    for i in range(10):
        calculator.update([_vehicle(LANE_CENTER_X)], _lane(), timestamp=i * 0.1)

    outlier_px = LANE_WIDTH_PX * 0.5
    calculator.update([_vehicle(LANE_CENTER_X + outlier_px)], _lane(), timestamp=1.0)

    latest = calculator.series[1].latest
    # 평활화 창(5) 안에서 1개 값만 튀었으므로 반영 폭이 1/5 수준으로 줄어든다
    assert 0 < latest.offset_ratio < 0.5 * 0.5


def test_direction_changes_counts_weaving_not_micro_jitter():
    """사행 — 진폭이 충분한 좌우 전환만 센다 (risk-criteria.md 2.2)."""
    from src.metrics.types import OffsetSample, VehicleTimeSeries

    def series_of(ratios):
        s = VehicleTimeSeries(track_id=1)
        for i, r in enumerate(ratios):
            s.offsets.append(OffsetSample(i * 0.1, r * LANE_WIDTH_M, r, 0.0, 0.9, False))
        return s

    weaving = series_of([0.3, -0.3, 0.3, -0.3])
    jitter = series_of([0.01, -0.01, 0.01, -0.01])

    assert weaving.direction_changes(min_amplitude_ratio=0.1) == 3
    assert jitter.direction_changes(min_amplitude_ratio=0.1) == 0


def test_departure_duration_measures_longest_continuous_span():
    """차선 걸침 — 끊긴 구간이 아니라 연속으로 유지된 최대 시간을 잰다."""
    from src.metrics.types import OffsetSample, VehicleTimeSeries

    s = VehicleTimeSeries(track_id=1)
    # 0.3 이상이 0.0~0.2초(짧게), 0.5~1.1초(길게) 두 번 나타난다
    pattern = [0.4, 0.4, 0.4, 0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    for i, r in enumerate(pattern):
        s.offsets.append(OffsetSample(i * 0.1, r * LANE_WIDTH_M, r, 0.0, 0.9, False))

    assert s.departure_duration_sec(offset_ratio_threshold=0.3) == pytest.approx(0.6, abs=1e-6)
