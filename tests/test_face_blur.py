"""FaceBlurrer 테스트. docs/pipeline-architecture.md 3.8 참고.

Haar cascade는 실제 얼굴 패턴에만 반응하므로 합성 이미지로는 검출(detect)
자체를 재현하기 어렵다 — `test_lane_detector.py`가 차선 "인식 실패" 쪽만
합성 이미지로 쉽게 재현할 수 있었던 것과 같은 제약이다. 그래서 여기서는
(1) detect()가 얼굴이 없는 이미지에서 크래시 없이 빈 목록을 내는지,
(2) 블러 적용 로직(blur_boxes) 자체가 주어진 좌표에 실제로 블러를
거는지를 분리해서 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.privacy.face_blur import FaceBlurrer
from src.utils.config import Config, load_config


@pytest.fixture
def blurrer() -> FaceBlurrer:
    return FaceBlurrer(load_config("configs/dev.yaml"))


def _noisy_image(width: int = 200, height: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_even_kernel_rejected():
    cfg = Config({"privacy": {"blur_faces": True, "blur_kernel": 30}})
    with pytest.raises(ValueError):
        FaceBlurrer(cfg)


def test_detect_returns_no_faces_on_blank_image(blurrer: FaceBlurrer):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert blurrer.detect(image) == []


def test_blur_boxes_reduces_local_variance(blurrer: FaceBlurrer):
    """박스 영역에 실제로 블러가 적용됐는지 — 고주파 잡음의 분산이 줄어드는 것으로 확인."""
    image = _noisy_image()
    box = (50, 50, 60, 60)
    x, y, w, h = box

    blurred = blurrer.blur_boxes(image, [box])

    original_var = image[y : y + h, x : x + w].astype(float).var()
    blurred_var = blurred[y : y + h, x : x + w].astype(float).var()
    assert blurred_var < original_var * 0.5

    # 박스 밖은 손대지 않는다
    assert np.array_equal(blurred[:40, :40], image[:40, :40])


def test_blur_boxes_ignores_zero_area_box(blurrer: FaceBlurrer):
    image = _noisy_image()
    result = blurrer.blur_boxes(image, [(10, 10, 0, 0)])
    assert np.array_equal(result, image)


def test_anonymize_noop_when_disabled():
    cfg = Config({"privacy": {"blur_faces": False, "blur_kernel": 31}})
    blurrer = FaceBlurrer(cfg)
    image = _noisy_image()
    assert np.array_equal(blurrer.anonymize(image), image)


def test_anonymize_runs_end_to_end_without_crashing(blurrer: FaceBlurrer):
    """실제 얼굴이 없는 이미지 — 검출 0건이라 블러 없이 원본과 동일해야 한다."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = blurrer.anonymize(image)
    assert np.array_equal(result, image)
