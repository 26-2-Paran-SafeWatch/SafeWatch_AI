"""OpenCV 기반 차선 인식. docs/pipeline-architecture.md 3.3 참고.

딥러닝 모델을 사용하지 않는다 — 성능 제약 및 AGENTS.md 원칙("사용하지
않는 것: 딥러닝 기반 차선 인식 모델").

처리 순서
1. perspective transform — bird's eye view 변환 (PerspectiveTransformer)
2. 색상(HLS S/L 채널) + 엣지(Canny) 기반 차선 후보 픽셀 추출
3. sliding window로 좌/우 차선 픽셀 탐색
4. 2차 다항식 피팅
5. 신뢰도 산출 — 좌우 평행도 + 검출 픽셀 수

폴백 — 신뢰도가 `lane.min_confidence` 미만이거나 피팅 자체가 실패하면
`valid=False`를 반환한다. 이 경우 직전 프레임 값을 유지하지 않고 상위
로직(METRICS/RISK)이 해당 구간의 offset 기반 지표를 제외하도록 한다
(판단 보류 — pipeline-architecture.md 3.3 "폴백 경로"에서 이미 결정된 방식).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.lane.perspective import PerspectiveTransformer
from src.lane.types import LaneModel
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.lane")

_INVALID = LaneModel(
    left_fit=None, right_fit=None, lane_width_px=None,
    curvature_radius=None, confidence=0.0, valid=False,
)


class LaneDetector:
    def __init__(self, cfg: Config):
        lane_cfg = cfg.lane
        self._canny_low: int = lane_cfg.canny_low
        self._canny_high: int = lane_cfg.canny_high
        self._min_confidence: float = lane_cfg.min_confidence

        color_cfg = lane_cfg.color_threshold
        self._s_thresh = tuple(color_cfg.s_channel_thresh)
        self._l_thresh = tuple(color_cfg.l_channel_thresh)

        window_cfg = lane_cfg.sliding_window
        self._n_windows: int = window_cfg.n_windows
        self._margin: int = window_cfg.margin
        self._min_pixels: int = window_cfg.min_pixels

        self._perspective = PerspectiveTransformer(cfg)

    def detect(self, image: np.ndarray) -> LaneModel:
        with profiler.stage("lane"):
            warped = self._perspective.warp(image)
            binary = self._binarize(warped)
            leftx, lefty, rightx, righty = self._sliding_window(binary)

            left_fit = self._safe_polyfit(lefty, leftx)
            right_fit = self._safe_polyfit(righty, rightx)
            if left_fit is None or right_fit is None:
                return _INVALID

            confidence = self._compute_confidence(
                binary.shape, left_fit, right_fit, leftx, rightx
            )
            if confidence < self._min_confidence:
                return LaneModel(
                    left_fit=np.poly1d(left_fit),
                    right_fit=np.poly1d(right_fit),
                    lane_width_px=None,
                    curvature_radius=None,
                    confidence=confidence,
                    valid=False,
                )

            lane_width_px = self._lane_width_px(binary.shape[0], left_fit, right_fit)
            return LaneModel(
                left_fit=np.poly1d(left_fit),
                right_fit=np.poly1d(right_fit),
                lane_width_px=lane_width_px,
                curvature_radius=None,  # TODO: RISK에서 필요해지면 산출
                confidence=confidence,
                valid=True,
            )

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel, s_channel = hls[:, :, 1], hls[:, :, 2]

        s_binary = (s_channel >= self._s_thresh[0]) & (s_channel <= self._s_thresh[1])
        l_binary = (l_channel >= self._l_thresh[0]) & (l_channel <= self._l_thresh[1])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self._canny_low, self._canny_high) > 0

        combined = np.zeros(s_channel.shape, dtype=np.uint8)
        combined[s_binary | l_binary | edges] = 1
        return combined

    def _sliding_window(
        self, binary: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width = binary.shape
        histogram = np.sum(binary[height // 2 :, :], axis=0)
        midpoint = width // 2
        leftx_current = int(np.argmax(histogram[:midpoint]))
        rightx_current = int(np.argmax(histogram[midpoint:]) + midpoint)

        window_height = height // self._n_windows
        nonzero_y, nonzero_x = binary.nonzero()

        left_lane_inds: list[np.ndarray] = []
        right_lane_inds: list[np.ndarray] = []

        for window in range(self._n_windows):
            y_low = height - (window + 1) * window_height
            y_high = height - window * window_height

            good_left = np.where(
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= leftx_current - self._margin)
                & (nonzero_x < leftx_current + self._margin)
            )[0]
            good_right = np.where(
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= rightx_current - self._margin)
                & (nonzero_x < rightx_current + self._margin)
            )[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > self._min_pixels:
                leftx_current = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > self._min_pixels:
                rightx_current = int(np.mean(nonzero_x[good_right]))

        left_idx = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
        right_idx = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        return (
            nonzero_x[left_idx], nonzero_y[left_idx],
            nonzero_x[right_idx], nonzero_y[right_idx],
        )

    def _safe_polyfit(self, y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
        if len(y) < self._min_pixels:
            return None
        try:
            return np.polyfit(y, x, 2)
        except (np.linalg.LinAlgError, TypeError, ValueError):
            return None

    def _compute_confidence(
        self,
        shape: tuple[int, int],
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        leftx: np.ndarray,
        rightx: np.ndarray,
    ) -> float:
        height, _ = shape
        y_eval = np.array([0, height // 2, height - 1])
        widths = np.polyval(right_fit, y_eval) - np.polyval(left_fit, y_eval)

        if np.any(widths <= 0):
            return 0.0  # 좌우 차선이 교차/역전 — 명백한 오검출

        # 평행도 — 세 지점에서의 폭 변동 계수(CV)가 작을수록 두 차선이 평행에 가깝다
        width_cv = float(np.std(widths) / np.mean(widths))
        parallel_score = max(0.0, 1.0 - width_cv * 2)

        # 픽셀 수 — min_pixels의 6배에서 만점 포화
        pixel_score = min(1.0, (len(leftx) + len(rightx)) / (self._min_pixels * 6))

        return round(0.5 * parallel_score + 0.5 * pixel_score, 3)

    def _lane_width_px(
        self, height: int, left_fit: np.ndarray, right_fit: np.ndarray
    ) -> float:
        y_bottom = height - 1
        return float(np.polyval(right_fit, y_bottom) - np.polyval(left_fit, y_bottom))
