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
    """COCO 사전학습 가중치로 고정한다 — bus.jpg는 대시캠 시점이 아닌 일반 사진이라
    파인튜닝 가중치(기본값)로는 검출 신뢰도가 낮고, 여기서 보려는 것은 검출 경로
    자체(좌표 역변환·클래스 필터)이기 때문이다."""
    cfg = load_config("configs/dev.yaml")
    cfg.to_dict()["detection"]["weights"] = "data/models/yolov8n.pt"
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


# ── 먼 영역 한 번 더 검출 (detection.far_region) ──────────────────────────────
from types import SimpleNamespace

import numpy as np

from src.detection.detector import merge_far_boxes
from src.detection.types import VehicleBox


def _box(x1, y1, x2, y2, conf):
    return VehicleBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=0)


def test_far_box_not_overlapping_is_added_and_full_boxes_are_kept():
    full = [_box(0, 0, 100, 100, 0.9), _box(60, 0, 160, 100, 0.8)]   # 붙어 있는 두 차 — 다시 거르지 않는다
    far = [_box(500, 300, 520, 315, 0.6)]
    out = merge_far_boxes(full, far, merge_iou=0.5, high_thresh=0.5)
    assert len(out) == 3


def test_far_box_overlapping_full_box_is_dropped():
    full = [_box(500, 300, 520, 315, 0.7)]
    far = [_box(501, 300, 521, 315, 0.8)]
    assert merge_far_boxes(full, far, 0.5, 0.5) == full


def test_confident_far_box_replaces_unconfident_full_box():
    """작게 봐서 새 트랙을 못 만들던 차(0.2)를 확대해서 확신(0.7)했으면 먼 영역 박스로 바꾼다"""
    full = [_box(500, 300, 520, 315, 0.2), _box(0, 0, 100, 100, 0.9)]
    far = [_box(501, 300, 521, 315, 0.7)]
    out = merge_far_boxes(full, far, 0.5, 0.5)
    assert sorted(b.confidence for b in out) == [0.7, 0.9]


class _Pred:
    def __init__(self, rows):
        self.boxes = [
            SimpleNamespace(cls=[0], xyxy=[r[:4]], conf=[r[4]]) for r in rows
        ]


class _ShapeModel:
    """입력 크기로 전체 화면/먼 영역을 구분해 정해진 박스를 돌려준다 (좌표는 받은 이미지 기준)"""
    names = {0: "vehicle"}

    def __init__(self, full_rows, far_rows):
        self.full_rows, self.far_rows = full_rows, far_rows

    def predict(self, image, **_):
        return [_Pred(self.full_rows if image.shape[1] == 1280 else self.far_rows)]


def _far_detector(full_rows, far_rows, enabled=True):
    d = VehicleDetector.__new__(VehicleDetector)
    d._model = _ShapeModel(full_rows, far_rows)
    d._vehicle_class_ids = frozenset({0})
    d._conf_threshold, d._iou_threshold = 0.1, 0.45
    d._roi_top_ratio, d._center_region_ratio = 0.0, 0.7
    d._far_enabled = enabled
    d._far_x, d._far_y = (0.25, 0.75), (0.30, 0.65)
    d._far_edge_ratio, d._far_merge_iou, d._track_high_thresh = 0.01, 0.5, 0.5
    return d


def test_far_region_boxes_return_to_frame_coordinates():
    # 1280×720 → 먼 영역 x 320~960, y 216~468. 먼 영역 안 (300,100)은 원래 화면 (620,316)
    d = _far_detector(full_rows=[], far_rows=[(300, 100, 320, 115, 0.8)])
    boxes = d.detect(np.zeros((720, 1280, 3), np.uint8))
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (620, 316, 640, 331)


def test_far_region_drops_boxes_touching_crop_edge_and_can_be_disabled():
    edge = (0, 100, 30, 115, 0.8)                     # 먼 영역 왼쪽 경계에 붙음 → 몸통이 잘린 차
    inside = (300, 100, 320, 115, 0.8)
    d = _far_detector(full_rows=[], far_rows=[edge, inside])
    assert len(d.detect(np.zeros((720, 1280, 3), np.uint8))) == 1
    d = _far_detector(full_rows=[], far_rows=[edge, inside], enabled=False)
    assert d.detect(np.zeros((720, 1280, 3), np.uint8)) == []


def test_far_region_config_loads_per_platform():
    from src.utils.config import load_config

    assert load_config("configs/default.yaml").detection.far_region.enabled is True
    assert load_config("configs/pi5.yaml").detection.far_region.enabled is False     # 320×320 입력
    assert load_config("configs/pi5_ufld.yaml").detection.far_region.enabled is True  # 1280×720 입력
