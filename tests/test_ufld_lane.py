"""ufld(딥러닝) 차선 경로 테스트 — 전처리·출력 해석·안전장치·원본 화면 offset·ONNX 변환.

실제 가중치(수백 MB)는 저장소에 없으므로, 모델 출력은 '원하는 차선 위치'에서 거꾸로 만든
가짜 백엔드로 흉내 낸다. 출력 해석 식은 노트북 15번 셀(원 저장소 demo와 같은 결과 확인)과
같아야 하므로 식 자체를 숫자로 검증한다. ONNX 변환은 작은 형상의 무작위 가중치로 PyTorch와
비교한다(torch가 없는 환경에서는 건너뜀).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.detection.types import VehicleBox
from src.lane import create_lane_detector
from src.lane.geometry import lane_at_row, x_at_row
from src.lane.types import LaneModel
from src.lane.ufld import UFLDSpec, decode_rows, preprocess
from src.lane.ufld_detector import UFLDLaneDetector
from src.metrics.offset import OffsetCalculator
from src.tracking.types import TrackedVehicle
from src.utils.config import Config, load_config

IMG_W, IMG_H = 1280, 720


# ── 공용 도우미 ─────────────────────────────────────────────────────────────

def _cfg(**guard_overrides) -> Config:
    data = copy.deepcopy(load_config("configs/default.yaml").to_dict())
    data["lane"]["method"] = "ufld"
    guard = data["lane"]["ufld"]["guard"]
    for key, value in guard_overrides.items():
        section, _, name = key.partition("__")
        if name:
            guard[section][name] = value
        else:
            guard[section] = value
    return Config(data)


class FakeBackend:
    """차선별 x 위치 함수(화면 비율 y → 화면 비율 x, None = 그 행에 차선 없음)로 모델 출력을 만든다."""

    def __init__(self, spec: UFLDSpec, lanes: list):
        self.spec = spec
        self.lanes = lanes
        self.calls = 0

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.spec
        self.calls += 1
        loc = np.full((s.grid_num, s.num_row, s.num_lanes), -10.0, dtype=np.float32)
        exist = np.zeros((2, s.num_row, s.num_lanes), dtype=np.float32)
        exist[0] = 1.0                                         # 기본은 '없음'
        for li, fn in enumerate(self.lanes):
            if fn is None:
                continue
            for k, ya in enumerate(s.row_anchor):
                xr = fn(float(ya))
                if xr is None:
                    continue
                cell = int(round(xr * (s.grid_num - 1) - 0.5))
                if 0 <= cell < s.grid_num:
                    loc[cell, k, li] = 10.0
                    exist[:, k, li] = (0.0, 1.0)
        return loc, exist


def _straight(x_bottom: float, x_top: float, y_top: float = 0.42, y_bottom: float = 1.0):
    """화면 비율 좌표의 직선 차선 (원근으로 위로 갈수록 가운데로 모임)."""
    def fn(y: float) -> float:
        t = (y - y_top) / (y_bottom - y_top)
        return x_top + (x_bottom - x_top) * t
    return fn


def _road(left_bottom: float = 0.25, right_bottom: float = 0.75):
    """소실점 (0.5, 0.42)로 모이는 4차선 도로. 차선 1·2가 자차 차로."""
    w = right_bottom - left_bottom
    xs = [left_bottom - w, left_bottom, right_bottom, right_bottom + w]
    return [_straight(x, 0.5) for x in xs]


def _detector(lanes, **guard_overrides) -> tuple[UFLDLaneDetector, FakeBackend]:
    cfg = _cfg(**guard_overrides)
    backend = FakeBackend(UFLDSpec.from_config(cfg.lane.ufld), lanes)
    return UFLDLaneDetector(cfg, backend=backend), backend


IMAGE = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


def _run(det: UFLDLaneDetector, n: int, t0: float = 0.0, fps: float = 15.0) -> list[LaneModel]:
    return [det.detect(IMAGE, timestamp=t0 + i / fps) for i in range(n)]


# ── 전처리 · 출력 해석 ─────────────────────────────────────────────────────

def test_spec_from_default_config_matches_culane_weights():
    spec = UFLDSpec.from_config(load_config("configs/default.yaml").lane.ufld)
    assert (spec.input_w, spec.input_h, spec.grid_num, spec.num_row, spec.num_lanes) == (1600, 320, 200, 72, 4)
    assert spec.row_anchor[0] == pytest.approx(0.42) and spec.row_anchor[-1] == pytest.approx(1.0)


def test_preprocess_shape_rgb_order_and_normalization():
    spec = UFLDSpec.from_config(_cfg().lane.ufld)
    bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    bgr[..., 2] = 255                                          # BGR의 빨강
    x = preprocess(bgr, spec)
    assert x.shape == (1, 3, 320, 1600) and x.dtype == np.float32
    assert x[0, 0].mean() == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)   # R 채널이 첫 번째
    assert x[0, 2].mean() == pytest.approx((0.0 - 0.406) / 0.225, abs=1e-4)


def test_decode_uses_three_cell_soft_argmax_like_the_notebook():
    spec = UFLDSpec(input_w=1600, input_h=320, crop_ratio=0.6, grid_num=200, num_row=3,
                    row_anchor_start=0.5, row_anchor_end=1.0, num_lanes=2)
    loc = np.zeros((200, 3, 2), dtype=np.float32)
    exist = np.zeros((2, 3, 2), dtype=np.float32)
    exist[1, 0, 0] = exist[1, 2, 0] = 1.0                      # 차선 0: 행 0·2만 존재
    loc[50, 0, 0] = 5.0                                        # 좌우 이웃 같음 → 평균 칸 = 50
    loc[[99, 100, 101], 2, 0] = [0.0, 1.0, 0.0]                # 비대칭 아님 → 100
    loc[[0, 1], 1, 1] = [3.0, 1.0]                             # 차선 1은 존재 없음 → 결과 비어야 함
    lanes = decode_rows(loc, exist, spec, IMG_W, IMG_H)

    assert lanes[1].shape == (0, 2)
    assert lanes[0].shape == (2, 2)
    # 행 0: 칸 49·50·51 softmax 가중 평균 = 50 (이웃 둘 다 0) → x = (50 + 0.5) / 199 * W
    w = np.exp([0.0, 5.0, 0.0]); w /= w.sum()
    expected0 = (float((w * np.array([49, 50, 51])).sum()) + 0.5) / 199 * IMG_W
    assert lanes[0][0, 0] == pytest.approx(expected0)
    assert lanes[0][0, 1] == pytest.approx(0.5 * IMG_H)
    assert lanes[0][1, 0] == pytest.approx((100 + 0.5) / 199 * IMG_W)
    assert lanes[0][1, 1] == pytest.approx(1.0 * IMG_H)


def test_decode_at_grid_edge_uses_only_existing_neighbors():
    spec = UFLDSpec(1600, 320, 0.6, 10, 1, 1.0, 1.0, 1)
    loc = np.zeros((10, 1, 1), dtype=np.float32)
    loc[0, 0, 0], loc[1, 0, 0] = 2.0, 0.0
    exist = np.array([[[0.0]], [[1.0]]], dtype=np.float32)
    (lane,) = decode_rows(loc, exist, spec, 900, 100)
    w = np.exp([2.0, 0.0]); w /= w.sum()
    assert lane[0, 0] == pytest.approx((float(w[1]) + 0.5) / 9 * 900)


# ── 탐지기 · 안전장치 ─────────────────────────────────────────────────────

def test_centered_road_becomes_valid_after_width_calibration():
    det, _ = _detector(_road())
    out = _run(det, 20)
    n_cal = det._guard._width_min_samples  # noqa: SLF001
    assert all(m.reject_reason == "width_calibrating" for m in out[:n_cal])
    assert all(m.valid for m in out[n_cal:])
    lane = out[-1]
    assert lane.space == "image" and lane.confidence == pytest.approx(1.0)
    y = 0.9 * IMG_H
    left, right = lane_at_row(lane.left_points, lane.right_points, y)
    assert left == pytest.approx(_road()[1](0.9) * IMG_W, abs=IMG_W / 199)
    assert right == pytest.approx(_road()[2](0.9) * IMG_W, abs=IMG_W / 199)
    assert lane.measure_y_range == pytest.approx((0.68 * IMG_H, 0.98 * IMG_H))


def test_low_existence_in_measure_range_is_held():
    lanes = _road()
    short = [lambda y, f=f: f(y) if y < 0.6 else None for f in lanes]   # 측정 구간(0.68~) 위에서만 보임
    det, _ = _detector(short)
    lane = _run(det, 1)[0]
    assert not lane.valid and lane.reject_reason == "existence"


def test_camera_outside_predicted_lane_is_held_as_straddle():
    # 예측 자차 차로가 화면 가운데(카메라)를 포함하지 않음 — 한 칸 옆 차로를 자차로 잡은 경우
    det, _ = _detector(_road(left_bottom=0.55, right_bottom=1.05), width__enabled=False)
    lane = _run(det, 1)[0]
    assert not lane.valid and lane.reject_reason == "straddle"


def test_lane_skip_error_is_held_by_width_check_and_recovers():
    normal = _road(0.35, 0.65)                                  # 바깥 선이 화면 안에 보이는 좁은 화각
    skipped = list(normal)
    skipped[1] = normal[0]                                      # 가까운 왼쪽 선을 놓치고 바깥 선을 잡음 (폭 약 2배)
    det, backend = _detector(normal)
    _run(det, 30)
    backend.lanes = skipped
    bad = _run(det, 3, t0=2.0)
    assert all(not m.valid and m.reject_reason == "width" for m in bad)
    backend.lanes = normal
    assert _run(det, 1, t0=2.3)[0].valid


def test_width_reference_follows_a_real_lane_width_change():
    det, backend = _detector(_road(0.25, 0.75), width__window_sec=2.0, width__min_samples=5)
    _run(det, 30)
    backend.lanes = _road(0.15, 0.85)                           # 폭 +40% (넓은 도로로 진입)
    out = _run(det, 60, t0=2.0)
    assert not out[0].valid and out[0].reject_reason == "width"
    assert out[-1].valid                                        # 창(2초)이 지나면 새 폭을 기준으로 삼음


def test_temporal_check_holds_sudden_jump_when_enabled():
    det, backend = _detector(_road(), temporal__enabled=True, width__enabled=False)
    assert _run(det, 3)[-1].valid
    backend.lanes = _road(0.35, 0.85)                          # 중심이 폭의 0.2만큼 순간 이동
    assert _run(det, 1, t0=0.3)[0].reject_reason == "temporal"


def test_backend_failure_holds_frame_without_raising():
    det, backend = _detector(_road())

    def boom(_x):
        raise RuntimeError("npu timeout")

    backend.infer = boom
    lane = det.detect(IMAGE, timestamp=0.0)
    assert not lane.valid and lane.reject_reason == "error"


def test_wrong_backend_output_shape_fails_at_startup():
    cfg = _cfg()
    wrong = UFLDSpec.from_config(cfg.lane.ufld)
    wrong = UFLDSpec(**{**wrong.__dict__, "num_row": 56})
    with pytest.raises(ValueError, match="출력 형상"):
        UFLDLaneDetector(cfg, backend=FakeBackend(wrong, _road()))


def test_factory_selects_method():
    from src.lane.lane_detector import LaneDetector

    assert isinstance(create_lane_detector(load_config("configs/default.yaml")), LaneDetector)
    cfg = _cfg()
    cfg.to_dict()["lane"]["ufld"]["model_path"] = "data/models/__없는_파일__.onnx"
    with pytest.raises(FileNotFoundError):
        create_lane_detector(cfg)
    cfg.to_dict()["lane"]["method"] = "lanenet"
    with pytest.raises(ValueError):
        create_lane_detector(cfg)


# ── 원본 화면 offset ───────────────────────────────────────────────────────

def _image_lane(valid: bool = True) -> LaneModel:
    ys = np.linspace(300, 700, 9)
    return LaneModel(
        left_fit=None, right_fit=None, lane_width_px=340.0, curvature_radius=None, confidence=0.9,
        valid=valid, space="image",
        left_points=np.c_[np.full_like(ys, 150.0), ys], right_points=np.c_[np.full_like(ys, 490.0), ys],
        measure_y_range=(400.0, 700.0),
    )


def _track(cx: float, bottom: float, tid: int = 1) -> TrackedVehicle:
    box = VehicleBox(x1=cx - 50, y1=bottom - 80, x2=cx + 50, y2=bottom, confidence=0.9, class_id=0)
    return TrackedVehicle(track_id=tid, box=box, tracked_frames=40, is_interpolated=False)


@pytest.fixture
def calculator() -> OffsetCalculator:
    return OffsetCalculator(load_config("configs/dev.yaml"), None)


@pytest.mark.parametrize("cx, ratio", [(320.0, 0.0), (405.0, 0.25), (235.0, -0.25)])
def test_image_space_offset_ratio(calculator, cx, ratio):
    sample = calculator.update([_track(cx, 600.0)], _image_lane(), timestamp=0.0)[1].latest
    assert sample.offset_ratio == pytest.approx(ratio)
    assert sample.offset_m == pytest.approx(ratio * 3.5)


def test_image_space_adjacent_lane_vehicle_gets_its_own_lane(calculator):
    sample = calculator.update([_track(660.0, 600.0)], _image_lane(), timestamp=0.0)[1].latest
    assert sample.lane_index == 1 and sample.offset_ratio == pytest.approx(0.0)


def test_image_space_vehicle_outside_measure_range_is_not_measured(calculator):
    series = calculator.update([_track(320.0, 380.0)], _image_lane(), timestamp=0.0)
    assert 1 not in series


def test_image_space_invalid_lane_adds_nothing(calculator):
    series = calculator.update([_track(320.0, 600.0)], _image_lane(valid=False), timestamp=0.0)
    assert 1 not in series


# ── 기하 도우미 ───────────────────────────────────────────────────────────

def test_x_at_row_interpolates_and_extends_with_nearest_half():
    pts = np.array([[100.0, 400.0], [120.0, 500.0], [140.0, 600.0], [160.0, 700.0]])
    assert x_at_row(pts, 550.0) == pytest.approx(130.0)
    assert x_at_row(pts, 800.0) == pytest.approx(180.0)        # 아래쪽 직선 연장
    assert x_at_row(pts, 300.0) == pytest.approx(80.0)         # 위쪽 직선 연장
    assert x_at_row(pts[:1], 500.0) is None
    assert lane_at_row(pts, pts - [50.0, 0.0], 500.0) is None  # 오른쪽이 왼쪽보다 왼쪽


# ── ONNX 변환 (작은 형상, 무작위 가중치) ─────────────────────────────────────

def test_export_rows_only_matches_full_model_and_runs_in_detector(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from scripts.export_ufld_onnx import UFLDv2RowNet, export, verify
    from src.lane.ufld import OnnxUFLDBackend

    spec = UFLDSpec(input_w=320, input_h=64, crop_ratio=0.6, grid_num=20, num_row=6,
                    row_anchor_start=0.42, row_anchor_end=1.0, num_lanes=4)
    torch.manual_seed(0)
    net = UFLDv2RowNet(spec, grid_col=10, num_col=5, fc_norm=True).eval()
    x = torch.randn(1, 3, 64, 320)
    with torch.no_grad():
        full = net(x)
        net.keep_rows_only()
        rows = net(x)
    for a, b in zip(full, rows):
        assert torch.allclose(a, b, atol=1e-6)

    path = tmp_path / "ufld_small.onnx"
    export(net, path)
    assert verify(net, path) < 1e-4

    cfg = _cfg()
    u = cfg.to_dict()["lane"]["ufld"]
    u.update(model_path=str(path), input_size=[320, 64], grid_num=20, num_row=6)
    det = UFLDLaneDetector(cfg, backend=OnnxUFLDBackend(path))
    lane = det.detect(IMAGE, timestamp=0.0)
    assert lane.space == "image" and lane.reject_reason is not None   # 무작위 가중치 — 보류되면 정상


def test_pi5_ufld_config_selects_ufld_with_16_9_input():
    """Pi 시험용 설정 — pi5.yaml을 상속하되 차선은 UFLD, 입력은 학습 화면과 같은 16:9 (hw-handoff-guide 2.5)."""
    from pathlib import Path

    from src.utils.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "pi5_ufld.yaml")
    assert cfg.lane.method == "ufld"
    assert cfg.lane.ufld.model_path == "data/models/ufld_aihub_aug.onnx"
    w, h = cfg.input.resolution
    assert abs(w / h - 16 / 9) < 1e-6
    assert cfg.detection.weights == load_config(Path(__file__).resolve().parents[1] / "configs" / "pi5.yaml").detection.weights
