"""LaneDetector 테스트.

전체 detect() 파이프라인은 blank 이미지로 "크래시 없이 valid=False를
반환하는지"만 검증하고, 알고리즘(마스크·안쪽 쌍 선택·평행 제약 sliding
window·공유 형상 피팅·신뢰도)은 합성 bird's-eye-view 이미지로 직접 검증한다.
이 합성 이미지는 perspective 변환을 거치지 않은, 이미 변환된 것으로 가정한
좌표계다 (configs의 warp_size 256×384, margin_ratio 0.18 → 기대 차선 x≈46, 210).

실패 재현 케이스는 `notebook/safewatch_lane_calib.ipynb`로 AI Hub 정답 라벨을
검증하며 실제로 나온 오검출을 단순화한 것이다 (pipeline-architecture.md 3.3).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.lane.lane_detector import LaneDetector
from src.utils.config import load_config

BEV_W, BEV_H = 256, 384
LEFT_X, RIGHT_X = 46, 210          # margin_ratio 0.18 → 0.18·256, 0.82·256
ROAD = (60, 60, 60)
WHITE = (235, 235, 235)


@pytest.fixture(scope="module")
def detector() -> LaneDetector:
    return LaneDetector(load_config("configs/dev.yaml"))


def _road() -> np.ndarray:
    image = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    image[:] = ROAD
    return image


def _line(image: np.ndarray, x: float, color=WHITE, width: int = 5,
          curve: float = 0.0, dash: int = 0) -> None:
    """세로 차선. curve는 화면 위쪽으로 갈수록 휘는 양(px), dash>0이면 점선 주기."""
    for y in range(BEV_H):
        if dash and (y // dash) % 2:
            continue
        off = curve * ((BEV_H - y) / BEV_H) ** 2
        x0 = int(round(x + off))
        image[y, max(x0, 0):max(x0 + width, 0)] = color


def _lanes(**kwargs) -> np.ndarray:
    image = _road()
    _line(image, LEFT_X, **kwargs)
    _line(image, RIGHT_X, **kwargs)
    return image


def _center_error(lane, left_x: float = LEFT_X + 2, right_x: float = RIGHT_X + 2,
                  curve: float = 0.0) -> float:
    """차로 중심 오차 최대값 (차로 폭 대비) — 노트북 8번 셀과 같은 정의."""
    ys = np.linspace(0, BEV_H - 1, 9)
    off = curve * ((BEV_H - ys) / BEV_H) ** 2
    fit_c = (lane.left_fit(ys) + lane.right_fit(ys)) / 2
    gt_c = (left_x + right_x) / 2 + off
    return float(np.max(np.abs(fit_c - gt_c)) / (right_x - left_x))


def test_detect_returns_invalid_on_blank_frame(detector: LaneDetector):
    """실제 도로 영상이 아닌 임의 프레임 — 차선이 없으니 valid=False가 정상."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.detect(image).valid is False


def test_parallel_lines_recovered_with_high_confidence(detector: LaneDetector):
    lane = detector.detect_bev(_lanes())
    assert lane.valid is True
    assert lane.confidence > 0.9
    assert _center_error(lane) < 0.02
    assert lane.left_fit(BEV_H - 1) == pytest.approx(LEFT_X + 2, abs=2)
    assert lane.right_fit(BEV_H - 1) == pytest.approx(RIGHT_X + 2, abs=2)


def test_curved_and_dashed_lines(detector: LaneDetector):
    image = _road()
    _line(image, LEFT_X, curve=30)
    _line(image, RIGHT_X, curve=30, dash=48)       # 세로 성분 필터(높이 25px)를 넘는 점선
    lane = detector.detect_bev(image)
    assert lane.valid is True
    assert _center_error(lane, curve=30) < 0.03


def test_yellow_line_is_detected(detector: LaneDetector):
    image = _road()
    _line(image, LEFT_X, color=(0, 200, 230))      # BGR 황색 중앙선
    _line(image, RIGHT_X)
    lane = detector.detect_bev(image)
    assert lane.valid is True
    assert _center_error(lane) < 0.02


def test_outer_railing_does_not_win_over_inner_line(detector: LaneDetector):
    """교량 난간·터널 벽 — 오른쪽 차선 바깥에 세로 줄무늬가 빽빽한 경우.

    이전 방식은 픽셀이 가장 많은 열(난간)을 오른쪽 차선으로 잡고도 신뢰도
    0.87을 냈다. 가장 안쪽 쌍을 고르면 진짜 차선을 잡는다.
    """
    image = _lanes()
    for x in range(222, 254, 6):                    # 진짜 선(210)보다 바깥, 훨씬 촘촘
        _line(image, x, width=3)
    lane = detector.detect_bev(image)
    assert lane.valid is True
    assert _center_error(lane) < 0.03


def _guardrail_mask(detector: LaneDetector, posts: bool) -> np.ndarray:
    """왼쪽 선 바로 바깥(10~35px), 조감도 위쪽 절반에 가드레일 잡음을 얹은 마스크.

    마스크 단계를 건너뛰고 피팅 단계만 시험한다 — 실제 영상에서 가드레일은
    top-hat·세로 성분 필터를 일부 통과한 조각들로 남는다.
    """
    mask = detector._mask(_lanes())
    for y in range(0, BEV_H // 2, 4):              # 가로 레일 조각
        mask[y:y + 3, LEFT_X - 35:LEFT_X - 10] = 255
    if posts:                                      # 차선만큼 촘촘한 세로 기둥
        for x in range(LEFT_X - 34, LEFT_X - 10, 6):
            mask[: BEV_H // 2, x:x + 2] = 255
    return mask


def _fit_mask(detector: LaneDetector, mask: np.ndarray):
    """마스크에서 바로 LaneModel을 만든다 (detect_bev에서 마스크 단계만 뺀 것)."""
    detector_mask = detector._mask
    detector._mask = lambda _image: mask
    try:
        return detector.detect_bev(_road())
    finally:
        detector._mask = detector_mask


def test_guardrail_clutter_does_not_drag_the_window(detector: LaneDetector):
    """중앙분리대 옆 1차로 — 먼 구간(조감도 위쪽)에서 왼쪽 선 바로 바깥에 가드레일 잡음.

    이전 방식은 아래쪽은 맞게 시작했다가 위쪽에서 윈도가 잡음 쪽으로 끌려갔다
    (AI Hub 2_20201019_175300, 중심 오차 0.24, 신뢰도 0.80). 좌우 윈도를 같이
    옮기고 가장 촘촘한 열을 쓰면 진짜 선에 머문다.
    """
    lane = _fit_mask(detector, _guardrail_mask(detector, posts=False))
    assert lane.valid is True
    assert _center_error(lane) < 0.03


def test_dense_guardrail_posts_are_withheld_not_misfit(detector: LaneDetector):
    """기둥이 차선만큼 촘촘하면 어느 쪽이 차선인지 알 수 없다 — 이때는 틀린 차선을
    확신하는 대신 보류해야 한다 (정밀도 우선, README "성능 목표")."""
    lane = _fit_mask(detector, _guardrail_mask(detector, posts=True))
    assert not (lane.valid and _center_error(lane) > 0.10)


def test_pink_guide_line_is_excluded(detector: LaneDetector):
    """차로 가운데 분홍 색깔 유도선 — 흰 차선이 아니므로 마스크에서 빠진다."""
    image = _lanes()
    _line(image, (LEFT_X + RIGHT_X) / 2 - 8, color=(190, 120, 235), width=14)
    mask = detector._mask(image)
    assert mask[:, LEFT_X + 30:RIGHT_X - 30].sum() == 0
    lane = detector.detect_bev(image)
    assert lane.valid is True
    assert _center_error(lane) < 0.02


def test_narrow_lane_from_miscalibration_is_withheld(detector: LaneDetector):
    """보정이 틀어진 카메라 — 차로가 기대 폭의 0.6배로 보이면 판정 보류.

    AI Hub에서 장착이 다른 차량의 시퀀스가 정확히 이렇게 보였고(정답 차로 폭
    0.56~0.66배), 이전 방식은 바깥 구조물까지를 차로로 잡아 폭을 맞춰 버렸다.
    """
    image = _road()
    width = int(0.6 * (RIGHT_X - LEFT_X))
    _line(image, LEFT_X + 20)
    _line(image, LEFT_X + 20 + width)
    lane = detector.detect_bev(image)
    assert lane.valid is False


@pytest.mark.xfail(strict=True, reason=(
    "알려진 회귀 (2026-11-02) — 격자 띠의 세로 줄눈이 top-hat·세로 성분 필터를 통과해 "
    "차선 쌍으로 잡힌다. 이전 방식은 RMS 잔차 8% 게이트로 걸렀다. 실제 영상에서 얼마나 "
    "나오는지 확인 후 대응 (pipeline-architecture.md 3.3)"))
def test_textured_band_is_not_a_lane(detector: LaneDetector):
    """보도블록 오탐 회귀 — 차선 대신 폭 넓은 격자 무늬 띠가 깔린 경우.

    S2에서 이전 방식이 보도블록을 차선으로 잡고 신뢰도 0.78을 냈고, 잔차 상한으로
    걸렀다. 지금 방식은 줄눈 하나하나가 깨끗한 세로선이라 잔차가 작아 통과한다.
    """
    image = _road()
    for center_x in (LEFT_X, RIGHT_X):
        x0, x1 = center_x - 30, center_x + 30
        cv2.rectangle(image, (x0, 0), (x1, BEV_H), (180, 180, 180), -1)
        for x in range(x0, x1 + 1, 10):            # 세로 줄눈
            cv2.line(image, (x, 0), (x, BEV_H), (255, 255, 255), 2)
        for y in range(0, BEV_H, 10):              # 가로 줄눈
            cv2.line(image, (x0, y), (x1, y), (255, 255, 255), 2)
    assert detector.detect_bev(image).valid is False


def test_lane_model_carries_residual_ratio(detector: LaneDetector):
    """튜닝·디버깅에 쓰도록 잔차 비율이 LaneModel에 실려 나온다."""
    lane = detector.detect_bev(_lanes())
    assert lane.fit_residual_ratio is not None
    assert 0.0 <= lane.fit_residual_ratio < 0.02
    assert lane.lane_width_px == pytest.approx(RIGHT_X - LEFT_X, abs=3)
