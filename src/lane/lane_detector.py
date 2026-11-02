"""OpenCV 기반 차선 인식. docs/pipeline-architecture.md 3.3 참고.

딥러닝 모델을 사용하지 않는다 — 성능 제약 및 AGENTS.md 원칙("사용하지
않는 것: 딥러닝 기반 차선 인식 모델").

처리 순서 (`notebook/safewatch_lane_calib.ipynb`에서 AI Hub 정답 라벨로 검증한 방식)
1. perspective transform — bird's eye view 변환 (PerspectiveTransformer)
2. 조감도 안에서 차선 후보 마스크 — 밝기 top-hat + 무채색 흰색, 황색, 세로 성분만
3. 시작 위치 — 하단 히스토그램 피크 중 허용 폭 안의 가장 안쪽 쌍
4. 평행 제약 sliding window — 좌우 윈도를 같은 만큼만 이동, 윈도 안에선 가장 촘촘한 열
5. 공유 형상 피팅 — 두 선을 같은 2차 모양에 위치만 다르게 함께 피팅, 잡음 점 한 번 제거
6. 신뢰도 — 윈도 적중률 × 피팅 잔차 × 폭 타당성

이전 방식(좌우 절반 최강 피크 + 독립 피팅 + 가중 합산 신뢰도)은 교량 난간·
터널 벽·가드레일을 차선으로 잡고도 신뢰도 0.55~0.87을 냈다. 위 3~5단계가
각각 그 실패를 막는다 (pipeline-architecture.md 3.3 "이 방식으로 바꾼 이유").

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

_MIN_POINTS_PER_SIDE = 4   # 2차 공유 형상(미지수 4개)을 안정적으로 풀기 위한 한쪽 최소 점 수


class LaneDetector:
    def __init__(self, cfg: Config):
        lane_cfg = cfg.lane
        self._min_confidence: float = lane_cfg.min_confidence

        mask_cfg = lane_cfg.mask
        self._tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_cfg.tophat_width, 1))
        self._min_tophat_threshold: int = mask_cfg.min_tophat_threshold
        self._white_s_max: int = mask_cfg.white_s_max
        self._yellow_low = np.array(mask_cfg.yellow_hls_low, dtype=np.uint8)
        self._yellow_high = np.array(mask_cfg.yellow_hls_high, dtype=np.uint8)
        self._min_component_height: int = mask_cfg.min_component_height
        self._min_component_aspect: float = mask_cfg.min_component_aspect
        self._min_component_area: int = mask_cfg.min_component_area

        peak_cfg = lane_cfg.peaks
        self._peak_min_height: float = peak_cfg.min_height
        self._peak_nms_px: int = peak_cfg.nms_px
        self._width_range: tuple[float, float] = tuple(peak_cfg.width_range)

        window_cfg = lane_cfg.sliding_window
        self._n_windows: int = window_cfg.n_windows
        self._margin: int = window_cfg.margin
        self._min_pixels: int = window_cfg.min_pixels

        conf_cfg = lane_cfg.confidence
        self._max_rms_ratio: float = conf_cfg.max_rms_ratio
        self._width_tolerance: float = conf_cfg.width_tolerance
        self._outlier_k: float = conf_cfg.outlier_k
        self._outlier_min_px: float = conf_cfg.outlier_min_px

        self._perspective = PerspectiveTransformer(cfg)
        self._expected_width: float = self._perspective.expected_lane_width_px

    @property
    def perspective(self) -> PerspectiveTransformer:
        """시각화 등에서 bird's-eye-view ↔ 원본 좌표 변환이 필요할 때 사용."""
        return self._perspective

    def detect(self, image: np.ndarray) -> LaneModel:
        """원본 프레임을 bird's eye view로 변환한 뒤 차선을 인식한다."""
        with profiler.stage("lane"):
            return self.detect_bev(self._perspective.warp(image))

    def detect_bev(self, warped: np.ndarray) -> LaneModel:
        """이미 bird's eye view로 변환된 이미지에서 차선을 인식한다.

        perspective 변환을 분리해 둔 것은 변환 품질과 무관하게 인식 알고리즘
        자체를 합성 조감도로 검증할 수 있게 하기 위해서다.
        """
        mask = self._mask(warped)
        hist = mask[mask.shape[0] // 2:].sum(axis=0) / 255.0
        pair = self._pick_pair(self._find_peaks(hist))
        if pair is None:
            return _INVALID

        left_pts, right_pts = self._sliding_window(mask, pair)
        fit = self._fit_shared_shape(left_pts, right_pts)
        if fit is None:
            return _INVALID
        (a, b, c_left, c_right), rms, n_points = fit

        width = c_right - c_left
        confidence = self._confidence(n_points, rms, width)
        residual_ratio = round(rms / width, 4) if width > 0 else None
        return LaneModel(
            left_fit=np.poly1d([a, b, c_left]),
            right_fit=np.poly1d([a, b, c_right]),
            lane_width_px=float(width) if confidence >= self._min_confidence else None,
            curvature_radius=None,  # TODO: RISK에서 필요해지면 산출
            confidence=confidence,
            valid=confidence >= self._min_confidence,
            fit_residual_ratio=residual_ratio,
        )

    def _mask(self, bev: np.ndarray) -> np.ndarray:
        """조감도에서 차선 후보 마스크(0/255).

        조감도로 먼저 변환한 뒤 임계값을 잡으므로 하늘·건물이 밝기 통계에 들어오지
        않는다. 수평 top-hat은 커널보다 폭 좁은 밝은 줄만 남겨 야간·안개에도 버티고,
        보도블록·차체처럼 넓은 밝은 면은 지운다.
        """
        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
        lightness, saturation = hls[:, :, 1], hls[:, :, 2]

        tophat = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, self._tophat_kernel)
        otsu, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, white = cv2.threshold(tophat, max(int(otsu), self._min_tophat_threshold), 255,
                                 cv2.THRESH_BINARY)
        white[saturation > self._white_s_max] = 0       # 유채색 제외 — 분홍·초록 색깔 유도선

        mask = cv2.bitwise_or(white, cv2.inRange(hls, self._yellow_low, self._yellow_high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        heights = stats[:, cv2.CC_STAT_HEIGHT]
        keep = ((heights >= self._min_component_height)
                & (heights >= self._min_component_aspect * stats[:, cv2.CC_STAT_WIDTH])
                & (stats[:, cv2.CC_STAT_AREA] >= self._min_component_area))
        keep[0] = False                                 # 배경
        return np.where(keep[labels], 255, 0).astype(np.uint8)

    def _find_peaks(self, hist: np.ndarray) -> list[int]:
        """히스토그램 국소 최대 — 강한 것부터 남기고 `nms_px` 안의 약한 것은 버린다."""
        smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
        idx = [i for i in range(1, len(smooth) - 1)
               if smooth[i] >= self._peak_min_height
               and smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]]
        kept: list[int] = []
        for i in sorted(idx, key=lambda i: -smooth[i]):
            if all(abs(i - j) > self._peak_nms_px for j in kept):
                kept.append(i)
        return sorted(kept)

    def _pick_pair(self, peaks: list[int]) -> tuple[int, int] | None:
        """중앙을 사이에 둔 쌍 중 폭이 허용 범위 안인 가장 좁은(= 가장 안쪽) 쌍.

        난간·가드레일·터널 벽은 항상 차선보다 바깥에 있으므로, 안쪽 선이
        보이면 그쪽을 고른다. 픽셀이 가장 많은 피크를 고르면 세로 줄무늬가
        빽빽한 이런 구조물이 이긴다.
        """
        mid = self._perspective.warp_size[0] // 2
        low, high = (r * self._expected_width for r in self._width_range)
        pairs = [(l, r) for l in peaks if l < mid for r in peaks if r >= mid and low <= r - l <= high]
        return min(pairs, key=lambda p: p[1] - p[0]) if pairs else None

    def _sliding_window(
        self, mask: np.ndarray, pair: tuple[int, int]
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """평행 제약 sliding window — 윈도마다 좌/우 차선 점 (x, y).

        조감도에서 두 차선은 평행하므로 좌우 윈도를 같은 만큼만 옮긴다. 한쪽
        윈도가 옆 구조물 잡음에 끌려가도 다른 쪽이 붙잡는다. 윈도 안 위치는
        픽셀 평균이 아니라 가장 촘촘한 세로 열로 잡는다 — 차선은 폭 좁은 연속
        줄이라 열 밀도가 가장 높고, 넓게 퍼진 잡음은 평균만 끌어당긴다.
        """
        height = mask.shape[0]
        window_height = height // self._n_windows
        nonzero_y, nonzero_x = mask.nonzero()
        centers = [float(pair[0]), float(pair[1])]
        points: tuple[list, list] = ([], [])

        for window in range(self._n_windows):
            y_low = height - (window + 1) * window_height
            y_high = height - window * window_height
            in_band = (nonzero_y >= y_low) & (nonzero_y < y_high)
            shifts = []
            for side in (0, 1):
                cx = centers[side]
                sel = in_band & (nonzero_x >= cx - self._margin) & (nonzero_x < cx + self._margin)
                if sel.sum() <= self._min_pixels:
                    continue
                base = int(cx - self._margin)
                columns = np.bincount(nonzero_x[sel] - base, minlength=2 * self._margin + 2)
                peak_x = base + int(np.argmax(np.convolve(columns, np.ones(3), mode="same")))
                near = sel & (np.abs(nonzero_x - peak_x) <= 3)   # 그 열 ±3px 평균 (소수점 위치)
                x = float(nonzero_x[near].mean())
                points[side].append((x, (y_low + y_high) / 2))
                shifts.append(x - cx)
            if shifts:
                shift = float(np.mean(shifts))
                centers = [centers[0] + shift, centers[1] + shift]

        return points

    def _fit_shared_shape(
        self, left: list[tuple[float, float]], right: list[tuple[float, float]]
    ) -> tuple[np.ndarray, float, int] | None:
        """x = a·y² + b·y + c_side 를 좌우 함께 최소제곱으로 푼다.

        잔차가 큰 점(1차 피팅 잔차 중앙값의 `outlier_k`배 초과)을 한 번 버리고
        다시 푼다. 반환 — (a, b, c_left, c_right), RMS 잔차(px), 남은 점 수.
        """
        if len(left) < _MIN_POINTS_PER_SIDE or len(right) < _MIN_POINTS_PER_SIDE:
            return None
        try:
            coef, residuals = self._solve(left, right)
            limit = max(self._outlier_min_px, self._outlier_k * float(np.median(np.abs(residuals))))
            keep = np.abs(residuals) <= limit
            left = [p for p, k in zip(left, keep[:len(left)]) if k]
            right = [p for p, k in zip(right, keep[len(keep) - len(right):]) if k]
            if len(left) < _MIN_POINTS_PER_SIDE or len(right) < _MIN_POINTS_PER_SIDE:
                return None
            coef, residuals = self._solve(left, right)
        except (np.linalg.LinAlgError, ValueError):
            return None
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        return coef, rms, len(left) + len(right)

    @staticmethod
    def _solve(
        left: list[tuple[float, float]], right: list[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = left + right
        y = np.array([p[1] for p in pts])
        x = np.array([p[0] for p in pts])
        is_left = np.r_[np.ones(len(left)), np.zeros(len(right))]
        design = np.c_[y ** 2, y, is_left, 1 - is_left]
        coef = np.linalg.lstsq(design, x, rcond=None)[0]
        return coef, x - design @ coef

    def _confidence(self, n_points: int, rms: float, width: float) -> float:
        """윈도 적중률 × 피팅 잔차 × 폭 타당성 (pipeline-architecture.md 3.3).

        곱이라 하나만 무너져도 판정이 보류된다. 폭 타당성은 보정이 틀어졌을 때
        (장착 변경·충격) 차로가 기대 폭과 달라 보이는 것을 잡는다.
        """
        hit_rate = n_points / (2 * self._n_windows)
        fit_quality = 1.0 - min(1.0, rms / (self._max_rms_ratio * self._expected_width))
        plausibility = float(np.clip(
            1.0 - abs(width / self._expected_width - 1.0) / self._width_tolerance, 0.0, 1.0
        ))
        return round(float(hit_rate * fit_quality * plausibility), 3)
