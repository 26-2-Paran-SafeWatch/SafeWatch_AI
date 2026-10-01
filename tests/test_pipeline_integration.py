"""LANE → METRICS 좌표계 연결 통합 테스트.

`test_offset.py`는 perspective 변환을 항등으로 두고 offset 산출 **공식**만
검증한다. 이 파일은 반대로 **실제 `PerspectiveTransformer`를 끼워** 좌표계
연결을 검증한다 — 차선 모델은 bird's-eye-view 좌표계, 검출 박스는 원본 프레임
좌표계라 둘을 잇는 변환이 틀리면 offset 전체가 조용히 틀어진다. 단위 테스트가
전부 통과해도 잡히지 않는 종류의 버그라 별도로 둔다.

실제 대시캠 영상이 없으므로, BEV에서 평행한 차선을 정의한 뒤 역변환해
원근감 있는 합성 도로 이미지를 만들어 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.detection.types import VehicleBox
from src.lane.lane_detector import LaneDetector
from src.metrics.offset import OffsetCalculator
from src.tracking.types import TrackedVehicle
from src.utils.config import load_config

FRAME_H, FRAME_W = 720, 1280
BEV_LEFT_X, BEV_RIGHT_X = 150.0, 490.0
BEV_CENTER_X = (BEV_LEFT_X + BEV_RIGHT_X) / 2      # 320
BEV_LANE_WIDTH = BEV_RIGHT_X - BEV_LEFT_X          # 340
LANE_WIDTH_M = 3.5


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/dev.yaml")


@pytest.fixture(scope="module")
def detector(cfg) -> LaneDetector:
    return LaneDetector(cfg)


@pytest.fixture(scope="module")
def road_image(detector: LaneDetector) -> np.ndarray:
    """BEV의 평행 차선을 원본 프레임으로 역변환해 그린 합성 도로."""
    detector.perspective.warp(np.zeros((FRAME_H, FRAME_W, 3), np.uint8))  # 변환 행렬 생성
    _, bev_h = detector.perspective.warp_size

    image = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
    ys = np.linspace(0, bev_h - 1, 100)
    for bev_x in (BEV_LEFT_X, BEV_RIGHT_X):
        pts = detector.perspective.unwarp_points(
            np.stack([np.full_like(ys, bev_x), ys], axis=1)
        )
        cv2.polylines(image, [pts.astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 6)
    return image


def _vehicle_at_bev(detector: LaneDetector, bev_x: float, bev_y: float = 500.0,
                    track_id: int = 1) -> TrackedVehicle:
    """BEV 좌표로 지정한 위치에 차량을 놓는다 (박스는 원본 프레임 좌표계)."""
    origin = detector.perspective.unwarp_points(np.array([[bev_x, bev_y]], np.float32))[0]
    center_x, bottom_y = float(origin[0]), float(origin[1])
    return TrackedVehicle(
        track_id=track_id,
        box=VehicleBox(center_x - 40, bottom_y - 60, center_x + 40, bottom_y, 0.9, 2),
        tracked_frames=30,
        is_interpolated=False,
    )


def test_warp_round_trip_is_lossless(detector: LaneDetector, road_image):
    """warp_points ↔ unwarp_points가 서로의 역변환이다."""
    probe = np.array([[640.0, 700.0], [300.0, 650.0], [900.0, 600.0]], np.float32)
    restored = detector.perspective.unwarp_points(detector.perspective.warp_points(probe))
    assert np.abs(restored - probe).max() < 0.01


def test_synthetic_road_is_recognized_as_valid_lane(detector: LaneDetector, road_image):
    """합성 도로에서 차선이 원래 정의한 BEV 위치로 복원된다."""
    lane = detector.detect(road_image)
    assert lane.valid is True
    assert np.polyval(lane.left_fit, 360) == pytest.approx(BEV_LEFT_X, abs=8)
    assert np.polyval(lane.right_fit, 360) == pytest.approx(BEV_RIGHT_X, abs=8)


@pytest.mark.parametrize(
    "bev_x, expected_ratio",
    [
        (BEV_CENTER_X, 0.0),                            # 차로 정중앙
        (BEV_CENTER_X + BEV_LANE_WIDTH * 0.25, 0.25),   # 우측 25%
        (BEV_CENTER_X - BEV_LANE_WIDTH * 0.25, -0.25),  # 좌측 25%
    ],
)
def test_offset_matches_vehicle_position_through_real_perspective(
    cfg, detector: LaneDetector, road_image, bev_x, expected_ratio
):
    """원본 프레임에 놓인 차량의 offset이 의도한 BEV 위치와 일치한다.

    부호 규약(우측 양수, risk-criteria.md 2.1)과 차로 폭 정규화가 실제 변환을
    거친 뒤에도 유지되는지 확인한다.
    """
    lane = detector.detect(road_image)
    calculator = OffsetCalculator(cfg, detector.perspective)
    vehicle = _vehicle_at_bev(detector, bev_x)

    sample = calculator.update([vehicle], lane, timestamp=0.0)[1].latest

    assert sample.offset_ratio == pytest.approx(expected_ratio, abs=0.02)
    assert sample.offset_m == pytest.approx(expected_ratio * LANE_WIDTH_M, abs=0.07)


def test_vehicle_drifting_right_produces_positive_lateral_velocity(
    cfg, detector: LaneDetector, road_image
):
    """표류 시나리오 — 차량이 차로 안에서 우측으로 천천히 밀리는 경우.

    스코프 축소 후 채택된 단서 중 '표류'에 해당한다 (risk-criteria.md 1.4).
    """
    lane = detector.detect(road_image)
    calculator = OffsetCalculator(cfg, detector.perspective)

    for i in range(20):
        bev_x = BEV_CENTER_X + i * 3.0
        calculator.update([_vehicle_at_bev(detector, bev_x)], lane, timestamp=i * 0.1)

    series = calculator.series[1]
    assert series.latest.offset_ratio > 0
    assert series.latest.lateral_velocity_mps > 0
    assert series.max_offset_ratio > 0.1
