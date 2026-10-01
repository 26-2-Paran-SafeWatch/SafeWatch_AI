"""LaneDetector 테스트.

실제 대시캠 영상이 아직 없어(sprint-plan.md S1 "데이터 사전 준비" 미완),
전체 detect() 파이프라인은 blank 이미지로 "크래시 없이 valid=False를
반환하는지"만 검증한다. 알고리즘 자체(이진화·sliding window·다항식
피팅·신뢰도 산출)는 합성 bird's-eye-view 이미지로 직접 검증한다 — 이
합성 이미지는 perspective 변환을 거치지 않은, 이미 변환된 것으로
가정한 좌표계다.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from src.lane.lane_detector import LaneDetector
from src.utils.config import load_config


@pytest.fixture(scope="module")
def detector() -> LaneDetector:
    return LaneDetector(load_config("configs/dev.yaml"))


def _synthetic_bev(width: int = 640, height: int = 720, left_x: int = 150, right_x: int = 490):
    """두 개의 평행한 흰색 수직선이 그려진 합성 bird's-eye-view 이미지."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)  # 어두운 도로면
    cv2.line(image, (left_x, 0), (left_x, height), (255, 255, 255), 15)
    cv2.line(image, (right_x, 0), (right_x, height), (255, 255, 255), 15)
    return image


def _textured_band_bev(width: int = 640, height: int = 720):
    """차선 대신 보도블록처럼 넓은 텍스처 띠가 양쪽에 깔린 합성 BEV 이미지.

    한 줄의 가는 선이 아니라 폭 120px의 격자 무늬라, 다항식 피팅은 띠의
    한가운데를 지나가고 픽셀은 그 주위로 넓게 흩어진다.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    for center_x in (150, 490):
        x0, x1 = center_x - 60, center_x + 60
        cv2.rectangle(image, (x0, 0), (x1, height), (180, 180, 180), -1)
        for x in range(x0, x1 + 1, 20):       # 블록 경계 — 세로 줄눈
            cv2.line(image, (x, 0), (x, height), (255, 255, 255), 3)
        for y in range(0, height, 20):        # 블록 경계 — 가로 줄눈
            cv2.line(image, (x0, y), (x1, y), (255, 255, 255), 3)
    return image


def test_detect_returns_invalid_on_blank_frame(detector: LaneDetector):
    """실제 도로 영상이 아닌 임의 프레임 — 차선이 없으니 valid=False가 정상."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert result.valid is False


def test_binarize_detects_white_lines(detector: LaneDetector):
    bev = _synthetic_bev()
    binary = detector._binarize(bev)
    assert binary.shape == bev.shape[:2]
    assert binary.sum() > 0  # 흰 선 부분이 1로 검출되어야 함


def test_sliding_window_and_polyfit_recover_parallel_lines(detector: LaneDetector):
    bev = _synthetic_bev(left_x=150, right_x=490)
    binary = detector._binarize(bev)
    leftx, lefty, rightx, righty = detector._sliding_window(binary)

    left_fit = detector._safe_polyfit(lefty, leftx)
    right_fit = detector._safe_polyfit(righty, rightx)
    assert left_fit is not None
    assert right_fit is not None

    # 직선이므로 하단(y=719)에서의 x가 그린 위치 근처여야 한다
    bottom_y = bev.shape[0] - 1
    assert abs(np.polyval(left_fit, bottom_y) - 150) < 10
    assert abs(np.polyval(right_fit, bottom_y) - 490) < 10


def _confidence_of(detector: LaneDetector, bev: np.ndarray) -> tuple[float, float | None]:
    """합성 BEV 이미지를 신뢰도 산출까지 통과시킨다."""
    binary = detector._binarize(bev)
    leftx, lefty, rightx, righty = detector._sliding_window(binary)
    left_fit = detector._safe_polyfit(lefty, leftx)
    right_fit = detector._safe_polyfit(righty, rightx)
    assert left_fit is not None and right_fit is not None
    return detector._compute_confidence(
        binary.shape, left_fit, right_fit, leftx, lefty, rightx, righty
    )


def test_confidence_high_for_parallel_lines(detector: LaneDetector):
    confidence, residual_ratio = _confidence_of(detector, _synthetic_bev(left_x=150, right_x=490))
    assert confidence >= detector._min_confidence
    # 선을 제대로 피팅했다면 잔차는 차로 폭 대비 아주 작다
    assert residual_ratio < detector._max_residual_ratio


def test_confidence_zero_when_lines_cross(detector: LaneDetector):
    height, width = 720, 640
    # 좌우가 뒤바뀐(교차하는) 가짜 피팅 — 명백한 오검출 케이스
    left_fit = np.array([0.0, 0.0, 500.0])   # x = 500 (오른쪽)
    right_fit = np.array([0.0, 0.0, 100.0])  # x = 100 (왼쪽) — 좌우 역전
    zeros = np.zeros(100)
    confidence, residual_ratio = detector._compute_confidence(
        (height, width), left_fit, right_fit, zeros, zeros, zeros, zeros
    )
    assert confidence == 0.0
    assert residual_ratio is None  # 차로 폭이 음수라 비율을 정의할 수 없다


def test_confidence_zero_for_textured_band(detector: LaneDetector):
    """보도블록 오탐 재현 — 선이 아니라 '면'을 피팅한 경우.

    S2 실행에서 인도 보도블록을 차선으로 오인하며 신뢰도 0.78이 나왔다
    (`docs/sprint-plan.md` S2). 평행도·픽셀 수만으로는 걸러지지 않고
    피팅 잔차에서만 드러나는 유형이라, 잔차 반영 후 회귀 방지용으로 남긴다.
    """
    bev = _textured_band_bev()
    confidence, residual_ratio = _confidence_of(detector, bev)
    # 넓은 띠를 피팅하면 잔차가 차로 폭 대비 상한을 넘어 명백한 오검출로 처리된다
    assert residual_ratio > detector._max_residual_ratio
    assert confidence == 0.0
    assert detector.detect_bev(bev).valid is False


def test_lane_model_carries_residual_ratio(detector: LaneDetector):
    """튜닝·디버깅에 쓰도록 잔차 비율이 LaneModel에 실려 나온다."""
    lane = detector.detect_bev(_synthetic_bev())
    assert lane.valid is True
    assert lane.fit_residual_ratio is not None
    assert 0.0 <= lane.fit_residual_ratio < detector._max_residual_ratio
