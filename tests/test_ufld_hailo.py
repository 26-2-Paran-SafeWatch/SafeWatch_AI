"""Hailo 백엔드의 입출력 변환 테스트 — HailoRT 없이 순수 함수만 본다 (실제 NPU 실행은 Pi에서
scripts/check_hailo_ufld.py로 확인)."""

from __future__ import annotations

import numpy as np
import pytest

from src.lane.ufld import UFLDSpec, preprocess
from src.lane.ufld_hailo import split_outputs, to_nhwc
from src.utils.config import load_config

SPEC = UFLDSpec(input_w=64, input_h=32, crop_ratio=0.6, grid_num=10, num_row=6,
                row_anchor_start=0.42, row_anchor_end=1.0, num_lanes=4)


def _ref():
    rng = np.random.default_rng(0)
    loc = rng.normal(size=(SPEC.grid_num, SPEC.num_row, SPEC.num_lanes)).astype(np.float32)
    exist = rng.normal(size=(2, SPEC.num_row, SPEC.num_lanes)).astype(np.float32)
    return loc, exist


def test_split_single_fc_output():
    loc, exist = _ref()
    flat = np.concatenate([loc.ravel(), exist.ravel()])[None, None, :]   # (1, 1, N) — 완전연결층 통째
    got_loc, got_exist = split_outputs({"fc": flat}, SPEC)
    np.testing.assert_array_equal(got_loc, loc)
    np.testing.assert_array_equal(got_exist, exist)


def test_split_two_outputs_nhwc():
    """Hailo는 (C, H, W) 텐서를 (1, H, W, C)로 돌려준다 — 축을 되돌려야 한다."""
    loc, exist = _ref()
    outs = {"a": exist.transpose(1, 2, 0)[None], "b": loc.transpose(1, 2, 0)[None]}
    got_loc, got_exist = split_outputs(outs, SPEC)
    np.testing.assert_array_equal(got_loc, loc)
    np.testing.assert_array_equal(got_exist, exist)


def test_split_two_outputs_flat_and_chw():
    loc, exist = _ref()
    got_loc, got_exist = split_outputs({"l": loc.reshape(1, -1), "e": exist[None]}, SPEC)
    np.testing.assert_array_equal(got_loc, loc)
    np.testing.assert_array_equal(got_exist, exist)


def test_split_unknown_shape_errors():
    with pytest.raises(ValueError, match="나눌 수 없습니다"):
        split_outputs({"x": np.zeros(7)}, SPEC)


def test_to_nhwc_modes():
    img = np.random.default_rng(1).integers(0, 256, size=(72, 128, 3), dtype=np.uint8)
    x = preprocess(img, SPEC)
    norm = to_nhwc(x, "normalized")
    assert norm.shape == (1, SPEC.input_h, SPEC.input_w, 3) and norm.dtype == np.float32
    np.testing.assert_array_equal(norm[0, :, :, 0], x[0, 0])
    rgb = to_nhwc(x, "rgb255")
    assert rgb.dtype == np.uint8
    # 정규화를 되돌리면 전처리 직전 RGB 화소(리사이즈·크롭 결과)와 같아야 한다
    import cv2
    resized = cv2.resize(img, (SPEC.input_w, int(SPEC.input_h / SPEC.crop_ratio)))[-SPEC.input_h:, :, ::-1]
    assert np.abs(rgb[0].astype(int) - resized.astype(int)).max() <= 1
    with pytest.raises(ValueError):
        to_nhwc(x, "bgr")


def test_hailo_config_loads():
    cfg = load_config("configs/pi5_hailo.yaml")
    assert cfg.lane.method == "ufld" and cfg.lane.ufld.backend == "hailo"
    assert cfg.lane.ufld.hailo.input in ("normalized", "rgb255")
    assert load_config("configs/default.yaml").lane.ufld.backend == "onnx"
