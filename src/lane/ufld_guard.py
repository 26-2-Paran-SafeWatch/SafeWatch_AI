"""ufld 경로 안전장치 — 모델이 낸 자차 차선을 믿을지 정한다.

원칙(docs/lane-detection-dl-plan.md D4): "선은 딥러닝이 찾고, 믿을지는 규칙과 검증이 정한다."
아래 검사를 순서대로 하고 하나라도 걸리면 보류(valid=False)한다. 기준값은 전부
`lane.ufld.guard` 설정에 있고, 근거는 노트북 `safewatch_lane_calib_v2.ipynb` 19~22번 셀이다.

1. 존재 비율 — 판정 구간 안 행 앵커 중 자차 좌·우 차선이 '존재'로 나온 비율의 작은 값.
   이 값이 그대로 LaneModel.confidence가 된다 (노트북 `ex_rng`).
2. 형상 — 판정 구간 위·아래 끝 행에서 좌우 차선 x를 읽을 수 있고 왼쪽 < 오른쪽.
3. 걸침 — 화면 가운데(카메라 위치)가 예측 차로 안에 있어야 한다. 자차가 그 차로에 있지 않다면
   그 '자차 차로'는 틀린 것이다 (노트북 20번 셀 `straddle`, 22번 그림 F-1의 1050 사례).
4. 차로 폭 — 위·아래 두 행의 폭이 최근 window_sec 동안의 폭 중앙값에서 (1+tolerance)배 넘게
   벗어나면 보류. 가까운 선을 놓치고 한 칸 바깥 선을 잡는 오류(폭 약 2배)를 거른다 (노트북 22번 셀).
   기준 폭은 정답 없이 모델 자신의 예측에서 낸다. 검사 3까지 통과한 프레임의 폭을 모두 기록하므로
   짧은 오류는 중앙값을 못 움직이고, 실제 차로 폭이 바뀌면 창 절반 정도 뒤에 따라간다.
   ⚠️ 한 주행 내내 같은 오류가 이어지면 기준도 같이 틀린다 — 알려진 약점.
5. 프레임 간 일관성(선택) — 직전 통과 프레임 대비 차로 중심·폭이 갑자기 튀면 보류.
   연속 영상으로 검증하기 전까지 기본은 끔 (lane.ufld.guard.temporal.enabled).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.lane.geometry import lane_at_row
from src.utils.config import Config


@dataclass
class GuardResult:
    valid: bool
    confidence: float
    reason: str | None           # 보류 이유 — None이면 통과
    width_px: float | None       # 판정 구간 아래 끝 행의 차로 폭(px)


class LaneGuard:
    """프레임마다 check()를 부른다. 차로 폭 기록과 직전 통과 값을 상태로 갖는다."""

    def __init__(self, guard_cfg: Config, row_anchor: np.ndarray, guard_range_ratio: tuple[float, float]):
        self._row_anchor = np.asarray(row_anchor, dtype=np.float64)
        self._top_ratio, self._bottom_ratio = (float(v) for v in guard_range_ratio)
        if not 0.0 <= self._top_ratio < self._bottom_ratio <= 1.0:
            raise ValueError(f"lane.ufld.guard_range_ratio가 잘못됐습니다: {guard_range_ratio}")

        self._min_existence: float = guard_cfg.min_existence
        self._min_straddle_margin: float = guard_cfg.min_straddle_margin

        w = guard_cfg.width
        self._width_enabled: bool = w.enabled
        self._width_log_tol: float = float(np.log1p(w.tolerance))
        self._width_window_sec: float = w.window_sec
        self._width_min_samples: int = w.min_samples
        self._widths: deque[tuple[float, float, float]] = deque()   # (timestamp, 아래 폭/화면 폭, 위 폭/화면 폭)

        t = guard_cfg.temporal
        self._temporal_enabled: bool = t.enabled
        self._max_center_jump: float = t.max_center_jump
        self._max_width_log_jump: float = float(np.log1p(t.max_width_jump))
        self._max_gap_sec: float = t.max_gap_sec
        self._last: tuple[float, float, float] | None = None       # (timestamp, 중심 x, 폭) — 아래 끝 행, px

    def reset(self) -> None:
        self._widths.clear()
        self._last = None

    def check(
        self, left: np.ndarray, right: np.ndarray, image_w: int, image_h: int, timestamp: float
    ) -> GuardResult:
        if self._widths and timestamp < self._widths[-1][0]:
            self.reset()                                            # 영상이 바뀌었거나 시간이 되돌아감

        y_top, y_bottom = self._top_ratio * image_h, self._bottom_ratio * image_h
        existence = self._existence(left, right, image_h, y_top, y_bottom)
        if existence < self._min_existence:
            return GuardResult(False, existence, "existence", None)

        bottom = lane_at_row(left, right, y_bottom)
        top = lane_at_row(left, right, y_top)
        if bottom is None or top is None:
            return GuardResult(False, existence, "geometry", None)
        width_b, width_t = bottom[1] - bottom[0], top[1] - top[0]

        cx = image_w / 2.0
        margin = min(cx - bottom[0], bottom[1] - cx) / width_b
        if margin < self._min_straddle_margin:
            return GuardResult(False, existence, "straddle", width_b)

        if self._width_enabled:
            reason = self._check_width(width_b / image_w, width_t / image_w, timestamp)
            if reason is not None:
                return GuardResult(False, existence, reason, width_b)

        center_b = (bottom[0] + bottom[1]) / 2.0
        if self._temporal_enabled and self._last is not None:
            ts0, c0, w0 = self._last
            if timestamp - ts0 <= self._max_gap_sec:
                if (abs(center_b - c0) / w0 > self._max_center_jump
                        or abs(np.log(width_b / w0)) > self._max_width_log_jump):
                    return GuardResult(False, existence, "temporal", width_b)
        self._last = (timestamp, center_b, width_b)
        return GuardResult(True, existence, None, width_b)

    def _existence(
        self, left: np.ndarray, right: np.ndarray, image_h: int, y_top: float, y_bottom: float
    ) -> float:
        anchors = self._row_anchor * image_h
        n = int(((anchors >= y_top) & (anchors <= y_bottom)).sum())
        if n == 0:
            return 0.0

        def ratio(points: np.ndarray) -> float:
            if points is None or len(points) == 0:
                return 0.0
            ys = points[:, 1]
            return float(((ys >= y_top) & (ys <= y_bottom)).sum()) / n

        return min(ratio(left), ratio(right))

    def _check_width(self, w_bottom: float, w_top: float, timestamp: float) -> str | None:
        """기록과 비교한 뒤 이번 폭도 기록에 넣는다. 기록이 모자라면 보류(보정 중)."""
        cutoff = timestamp - self._width_window_sec
        while self._widths and self._widths[0][0] < cutoff:
            self._widths.popleft()
        history = np.array([(b, t) for _, b, t in self._widths]) if self._widths else None
        self._widths.append((timestamp, w_bottom, w_top))
        if history is None or len(history) < self._width_min_samples:
            return "width_calibrating"
        ref = np.median(history, axis=0)
        dev = np.abs(np.log(np.array([w_bottom, w_top]) / ref))
        return "width" if float(dev.max()) > self._width_log_tol else None
