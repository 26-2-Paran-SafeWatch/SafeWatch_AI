"""YOLO Hailo 경로 — letterbox·NMS 출력 해석·내보내기 폴더 읽기·공유 장치·detector 분기.
HailoRT 없이 순수 함수와 가짜 모듈로 본다. 실제 NPU 실행은 Pi에서 확인한다."""

from __future__ import annotations

import copy
import json
import sys
import types

import numpy as np
import pytest

from src.detection import hailo_yolo
from src.detection.detector import VehicleDetector
from src.detection.hailo_yolo import decode_nms, letterbox, read_export
from src.utils import hailo_device
from src.utils.config import Config, load_config


def test_letterbox_matches_ultralytics_shape_and_padding():
    img = np.full((720, 1280, 3), (10, 20, 30), np.uint8)       # BGR
    x, r, (left, top) = letterbox(img, (320, 320))
    assert x.shape == (320, 320, 3) and x.dtype == np.uint8
    assert r == pytest.approx(0.25) and left == 0 and top == 70  # 1280×720 → 320×180, 위아래 70씩
    assert tuple(x[0, 0]) == (114, 114, 114)                     # 여백
    assert tuple(x[160, 160]) == (30, 20, 10)                    # RGB로 뒤집힘


def test_decode_nms_maps_back_to_frame_and_filters_conf():
    frame_w, frame_h = 1280, 720
    _, r, pad = letterbox(np.zeros((frame_h, frame_w, 3), np.uint8), (320, 320))
    want = (400.0, 300.0, 520.0, 380.0)                          # 원본 좌표
    to_in = lambda v, axis: (v * r + pad[axis]) / 320            # noqa: E731 — 원본 → 입력 0~1
    det = [to_in(want[1], 1), to_in(want[0], 0), to_in(want[3], 1), to_in(want[2], 0)]
    per_class = [np.array([det + [0.9], det + [0.05]], np.float32)]
    out = decode_nms(per_class, (320, 320), r, pad, (frame_w, frame_h), conf=0.1)
    assert len(out) == 1
    x1, y1, x2, y2, score, cls = out[0]
    assert (x1, y1, x2, y2) == pytest.approx(want, abs=1e-3) and score == pytest.approx(0.9) and cls == 0


def test_decode_nms_empty_class():
    assert decode_nms([np.zeros((0, 5))], (320, 320), 0.25, (0, 70), (1280, 720), 0.1) == []


def _export(tmp_path, nms=True, th=0.25):
    d = tmp_path / "m_hailo_model"
    d.mkdir()
    (d / "m.hef").write_bytes(b"")
    (d / "metadata.yaml").write_text(f"names:\n  0: vehicle\nnms: {str(nms).lower()}\n", encoding="utf-8")
    (d / "nms_config.json").write_text(json.dumps({"nms_scores_th": th, "nms_iou_th": 0.45}), encoding="utf-8")
    return d


def test_read_export(tmp_path):
    hef, names, nms = read_export(_export(tmp_path))
    assert hef.name == "m.hef" and names == {0: "vehicle"} and nms["nms_scores_th"] == 0.25


def test_read_export_requires_nms(tmp_path):
    with pytest.raises(ValueError, match="NMS"):
        read_export(_export(tmp_path, nms=False))


@pytest.fixture
def fake_hailo(monkeypatch):
    """hailo_platform 가짜 — VDevice 생성·해제 횟수만 센다."""
    log = {"created": 0, "released": 0}

    class VDevice:
        @staticmethod
        def create_params():
            return types.SimpleNamespace(scheduling_algorithm=None)

        def __init__(self, params):
            assert params.scheduling_algorithm == "RR"
            log["created"] += 1

        def release(self):
            log["released"] += 1

    mod = types.SimpleNamespace(VDevice=VDevice, HailoSchedulingAlgorithm=types.SimpleNamespace(ROUND_ROBIN="RR"))
    monkeypatch.setitem(sys.modules, "hailo_platform", mod)
    monkeypatch.setattr(hailo_device, "_vdevice", None)
    monkeypatch.setattr(hailo_device, "_refs", 0)
    return log


def test_shared_device_opened_once_and_closed_by_last_user(fake_hailo):
    a, b = hailo_device.acquire(), hailo_device.acquire()
    assert a is b and fake_hailo["created"] == 1
    hailo_device.release()
    assert fake_hailo["released"] == 0
    hailo_device.release()
    assert fake_hailo["released"] == 1
    hailo_device.release()                                      # 더 놓아도 무해
    assert fake_hailo["released"] == 1


def _cfg(**det) -> Config:
    data = copy.deepcopy(load_config("configs/default.yaml").to_dict())
    data["detection"].update(det)
    return Config(data)


def test_detector_hailo_backend(monkeypatch):
    calls = {"closed": 0}

    class FakeHailoYolo:
        names = {0: "vehicle"}

        def __init__(self, path, conf):
            assert conf == 0.1

        def predict(self, image, conf):
            return [(10.0, 20.0, 50.0, 60.0, 0.8, 0), (0.0, 0.0, 5.0, 5.0, 0.7, 3)]   # 클래스 3은 차량 아님

        def close(self):
            calls["closed"] += 1

    monkeypatch.setattr(hailo_yolo, "HailoYolo", FakeHailoYolo)
    det = VehicleDetector(_cfg(backend="hailo", weights="x_hailo_model"))
    boxes = det._predict(np.zeros((100, 100, 3), np.uint8), 5, 7)
    assert len(boxes) == 1 and (boxes[0].x1, boxes[0].y1) == (15.0, 27.0)
    det.close()
    det.close()
    assert calls["closed"] == 1


def test_detector_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        VehicleDetector(_cfg(backend="tpu"))


def test_hailo_config():
    cfg = load_config("configs/pi5_ufld_hailo.yaml")
    assert cfg.detection.backend == "hailo" and cfg.detection.weights.endswith("_hailo_model")
    assert load_config("configs/default.yaml").detection.backend == "ultralytics"
