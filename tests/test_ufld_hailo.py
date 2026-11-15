"""UFLD 백엔드 분기 테스트 — hailo_platform 없이 가짜 백엔드로 전처리 분기, 출력 형상 검사,
설정 분기, 자원 해제를 본다. 실제 NPU 실행은 Pi에서 scripts/check_hailo_ufld.py로 확인한다."""

from __future__ import annotations

import builtins
import copy

import cv2
import numpy as np
import pytest

from src.lane.ufld import HailoUFLDBackend, UFLDSpec, preprocess, preprocess_for, preprocess_uint8
from src.lane.ufld_detector import UFLDLaneDetector, _make_backend
from src.utils.config import Config, load_config


def _cfg() -> Config:
    data = copy.deepcopy(load_config("configs/default.yaml").to_dict())
    data["lane"]["method"] = "ufld"
    return Config(data)


class RecordingBackend:
    """받은 입력을 기록하고 설정 형상의 빈 출력을 돌려준다."""

    def __init__(self, spec: UFLDSpec, input_format: str | None, bad_shape: bool = False):
        if input_format is not None:
            self.input_format = input_format
        self.spec, self.bad, self.inputs, self.closed = spec, bad_shape, [], 0

    def infer(self, x):
        self.inputs.append(x)
        s = self.spec
        g = s.grid_num - 1 if self.bad else s.grid_num
        return np.zeros((g, s.num_row, s.num_lanes), np.float32), np.zeros((2, s.num_row, s.num_lanes), np.float32)

    def close(self):
        self.closed += 1


SPEC = UFLDSpec.from_config(_cfg().lane.ufld)
IMG = np.random.default_rng(0).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)


def test_preprocess_uint8_is_crop_and_rgb_only():
    x = preprocess_uint8(IMG, SPEC)
    assert x.shape == (SPEC.input_h, SPEC.input_w, 3) and x.dtype == np.uint8 and x.flags.c_contiguous
    resized = cv2.resize(IMG, (SPEC.input_w, int(SPEC.input_h / SPEC.crop_ratio)))
    np.testing.assert_array_equal(x, resized[-SPEC.input_h:, :, ::-1])
    # 정규화만 빠진 같은 화면이어야 한다 — float 전처리를 되돌리면 같은 화소
    f = preprocess(IMG, SPEC)[0].transpose(1, 2, 0)
    back = (f * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    assert np.abs(back - x).max() < 0.01


def test_preprocess_for_unknown_format():
    with pytest.raises(ValueError):
        preprocess_for(IMG, SPEC, "bgr")


@pytest.mark.parametrize("fmt, shape, dtype", [
    ("uint8_hwc", (SPEC.input_h, SPEC.input_w, 3), np.uint8),
    ("float_nchw", (1, 3, SPEC.input_h, SPEC.input_w), np.float32),
    (None, (1, 3, SPEC.input_h, SPEC.input_w), np.float32),        # input_format 없으면 기존 동작
])
def test_detector_feeds_backend_its_format(fmt, shape, dtype):
    be = RecordingBackend(SPEC, fmt)
    det = UFLDLaneDetector(_cfg(), backend=be)
    det.detect(IMG)
    assert len(be.inputs) == 2                                      # 시작 확인용 빈 입력 + 실제 프레임
    for x in be.inputs:
        assert x.shape == shape and x.dtype == dtype


def test_bad_output_shape_errors_and_releases():
    be = RecordingBackend(SPEC, "uint8_hwc", bad_shape=True)
    with pytest.raises(ValueError, match="출력 형상"):
        UFLDLaneDetector(_cfg(), backend=be)
    assert be.closed == 1


def test_close_delegates_to_backend():
    be = RecordingBackend(SPEC, "uint8_hwc")
    UFLDLaneDetector(_cfg(), backend=be).close()
    assert be.closed == 1


def test_config_branch():
    assert load_config("configs/default.yaml").lane.ufld.backend == "onnx"
    h = load_config("configs/pi5_ufld_hailo.yaml")
    assert h.lane.method == "ufld" and h.lane.ufld.backend == "hailo"
    assert h.lane.ufld.hef_path.endswith(".hef") and h.lane.ufld.tail_path.endswith(".onnx")
    assert tuple(h.input.resolution) == (1280, 720)                  # pi5_ufld.yaml 상속
    u = copy.deepcopy(_cfg().to_dict())
    u["lane"]["ufld"]["backend"] = "tpu"
    with pytest.raises(ValueError, match="backend"):
        _make_backend(Config(u).lane.ufld)


def test_hailo_backend_errors_without_files_or_runtime(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        HailoUFLDBackend(tmp_path / "x.hef", tmp_path / "t.onnx")
    (tmp_path / "x.hef").write_bytes(b"")
    (tmp_path / "t.onnx").write_bytes(b"")
    real_import = builtins.__import__

    def no_hailo(name, *a, **k):
        if name == "hailo_platform":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_hailo)
    with pytest.raises(RuntimeError, match="hailo_platform"):
        HailoUFLDBackend(tmp_path / "x.hef", tmp_path / "t.onnx")
