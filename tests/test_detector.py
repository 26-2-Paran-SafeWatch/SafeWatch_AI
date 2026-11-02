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


class _FakeModel:
    def __init__(self, names: dict[int, str]):
        self.names = names


def _resolve(names: dict[int, str], wanted: list[str]) -> frozenset[int]:
    detector = VehicleDetector.__new__(VehicleDetector)   # 가중치 로드 없이 클래스 선택만 시험
    detector._model = _FakeModel(names)
    return detector._resolve_vehicle_classes(wanted)


WANTED = ["car", "motorcycle", "bus", "truck", "vehicle"]   # configs/default.yaml과 같은 목록


def test_coco_weights_select_vehicle_classes_by_name():
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert _resolve(names, WANTED) == {2, 3, 5, 7}


def test_finetuned_single_class_weights_select_vehicle():
    """파인튜닝 가중치(vehicle=0 하나)로 바꿔도 config 수정 없이 차량이 잡힌다.

    id를 하드코딩했을 때는 0이 COCO의 person이라 전부 버려져, 에러 없이 차량이
    하나도 검출되지 않았다.
    """
    assert _resolve({0: "vehicle"}, WANTED) == {0}


def test_weights_without_matching_class_fail_at_startup():
    with pytest.raises(ValueError, match="vehicle_class_names"):
        _resolve({0: "person", 1: "dog"}, WANTED)
