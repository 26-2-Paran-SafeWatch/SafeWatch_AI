"""FrameQualityChecker 테스트.

실제 대시캠 영상이 없어, 블러·밝기 급변 각각을 합성 이미지로 독립
검증한다 — test_lane_detector.py가 합성 BEV 이미지로 차선 알고리즘을
검증하는 것과 같은 방식이다.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from src.risk.frame_quality import FrameQualityChecker
from src.utils.config import load_config


@pytest.fixture
def checker() -> FrameQualityChecker:
    return FrameQualityChecker(load_config("configs/dev.yaml"))


def _sharp_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """고주파 성분이 풍부한 체커보드 — 블러 검사를 통과해야 한다."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    block = 10
    for y in range(0, height, block):
        for x in range(0, width, block):
            if (x // block + y // block) % 2 == 0:
                image[y : y + block, x : x + block] = 255
    return image


def _flat_frame(width: int = 640, height: int = 480, value: int = 128) -> np.ndarray:
    """에지가 전혀 없는 단색 프레임 — 블러 검사를 통과하지 못해야 한다."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_sharp_frame_passes_blur_check(checker: FrameQualityChecker):
    assert checker.check(_sharp_frame()) is True


def test_flat_frame_fails_blur_check(checker: FrameQualityChecker):
    assert checker.check(_flat_frame()) is False


def test_stable_brightness_across_frames_passes(checker: FrameQualityChecker):
    frame = _sharp_frame()
    assert checker.check(frame) is True
    assert checker.check(frame) is True  # 동일 프레임 반복 — 밝기 변화 0


def test_sudden_brightness_jump_fails(checker: FrameQualityChecker):
    assert checker.check(_sharp_frame()) is True
    # 직전 프레임은 선명했으니 이번 프레임이 실패하는 이유는 오직 밝기 급변이어야 한다.
    bright_sharp = cv2.convertScaleAbs(_sharp_frame(), alpha=1.0, beta=200)
    assert checker.check(bright_sharp) is False


def test_gradual_brightness_change_within_threshold_passes(checker: FrameQualityChecker):
    cfg = load_config("configs/dev.yaml")
    small_step = int(cfg.risk.frame_quality.max_brightness_jump) - 5
    assert checker.check(_sharp_frame()) is True
    slightly_brighter = cv2.convertScaleAbs(_sharp_frame(), alpha=1.0, beta=small_step)
    assert checker.check(slightly_brighter) is True
