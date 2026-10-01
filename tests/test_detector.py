"""VehicleDetector 테스트.

ultralytics 패키지에 포함된 샘플 이미지(bus.jpg)로 검증한다. 이 이미지는
대시캠 촬영본이 아니라 일반 스냅샷이라 ROI(하늘 영역 제외) 가정이 맞지
않으므로, ROI 관련 테스트는 roi_top_ratio를 0으로 낮춰 검출 자체와 좌표
역변환, 차량 클래스 필터링만 검증한다. 실제 대시캠 영상 기준 ROI 검증은
샘플 주행 영상 확보 후 진행한다 (sprint-plan.md S1 "데이터 사전 준비" 참고).
"""

from pathlib import Path

import cv2
import pytest

from src.detection.detector import VehicleDetector
from src.utils.config import load_config

try:
    import ultralytics

    BUS_JPG = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
except ImportError:
    BUS_JPG = Path("__missing__")

pytestmark = pytest.mark.skipif(
    not BUS_JPG.exists(), reason="ultralytics 샘플 이미지를 찾을 수 없음"
)


@pytest.fixture(scope="module")
def detector() -> VehicleDetector:
    cfg = load_config("configs/dev.yaml")
    return VehicleDetector(cfg)


def test_detect_returns_list_without_crashing(detector: VehicleDetector):
    image = cv2.imread(str(BUS_JPG))
    boxes = detector.detect(image)
    assert isinstance(boxes, list)


def test_detect_finds_bus_with_coordinates_in_frame_bounds(detector: VehicleDetector):
    image = cv2.imread(str(BUS_JPG))
    height, width = image.shape[:2]

    # 이 이미지는 대시캠 시점이 아니므로 ROI를 끄고 검출 자체를 검증한다.
    detector._roi_top_ratio = 0.0
    boxes = detector.detect(image)

    assert len(boxes) >= 1
    bus = boxes[0]
    assert bus.class_id == 5  # COCO: bus
    assert 0 <= bus.x1 < bus.x2 <= width
    assert 0 <= bus.y1 < bus.y2 <= height


def test_forward_vehicle_filter_excludes_off_center_boxes(detector: VehicleDetector):
    from src.detection.types import VehicleBox

    frame_width = 1000
    detector._center_region_ratio = 0.5  # 중앙 50%만 허용 → 250~750
    boxes = [
        VehicleBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9, class_id=2),  # center_x=25, 제외
        VehicleBox(x1=450, y1=0, x2=550, y2=50, confidence=0.9, class_id=2),  # center_x=500, 포함
    ]
    result = detector._filter_forward_vehicles(boxes, frame_width)
    assert len(result) == 1
    assert result[0].center_x == 500
