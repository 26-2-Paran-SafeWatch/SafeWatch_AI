"""lane offset 산출과 시계열 축적. docs/pipeline-architecture.md 3.5 참고.

차량 위치와 차선의 관계를 수치화한다. 채택된 판별 단서 4개(사행·차선 걸침·
스웨빙·표류)가 **모두 이 offset 시계열에서 파생**되므로, 이 모듈의 출력이
곧 판정의 입력 전부다 (risk-criteria.md 1.4).

좌표계 주의 — 차선 모델(`LaneModel`)은 bird's-eye-view 좌표계에 있고 검출·
추적 박스는 원본 프레임 좌표계에 있다. 차량 위치를 BEV로 옮긴 뒤 비교한다.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from src.lane.perspective import PerspectiveTransformer
from src.lane.types import LaneModel
from src.metrics.types import OffsetSample, VehicleTimeSeries
from src.tracking.types import TrackedVehicle
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.metrics")


class OffsetCalculator:
    """차량별 lane offset 시계열을 유지한다."""

    def __init__(self, cfg: Config, perspective: PerspectiveTransformer):
        self._perspective = perspective
        self._lane_width_m: float = cfg.lane.lane_width_m

        m_cfg = cfg.metrics
        self._window_sec: float = m_cfg.observation_window_sec
        self._smoothing: str = m_cfg.smoothing
        self._smoothing_window: int = m_cfg.smoothing_window

        self._series: dict[int, VehicleTimeSeries] = {}
        # 평활화 전 원시 offset — 평활화 결과만 저장하면 창 크기를 바꿔 재계산할 수 없다
        self._raw_offsets: dict[int, deque[tuple[float, float]]] = {}

    @property
    def series(self) -> dict[int, VehicleTimeSeries]:
        return self._series

    def update(
        self,
        tracks: list[TrackedVehicle],
        lane: LaneModel,
        timestamp: float,
    ) -> dict[int, VehicleTimeSeries]:
        """이번 프레임의 offset을 계산해 각 차량의 시계열에 누적한다.

        차선 인식이 무효한 프레임에서는 **아무 샘플도 추가하지 않는다** — 잘못된
        offset을 넣는 것보다 판정을 보류하는 편이 낫다는 결정에 따른다
        (risk-criteria.md 4.1, pipeline-architecture.md 3.3 폴백 경로).
        """
        with profiler.stage("metrics"):
            self._forget_disappeared({t.track_id for t in tracks})

            if not lane.valid or lane.left_fit is None or lane.right_fit is None:
                return self._series

            for track in tracks:
                offset_m = self._measure(track, lane)
                if offset_m is None:
                    continue
                self._append(track, offset_m, lane.confidence, timestamp)

            return self._series

    def _measure(self, track: TrackedVehicle, lane: LaneModel) -> float | None:
        """차량 한 대의 offset(미터, 우측 양수)을 산출한다. 측정 불가하면 None."""
        try:
            bev = self._perspective.warp_points(
                np.array([[track.center_x, track.bottom_y]], dtype=np.float32)
            )
        except RuntimeError:
            # warp()가 아직 한 번도 호출되지 않음 — 차선 인식보다 먼저 불린 경우
            logger.debug("perspective 행렬 미생성 — offset 산출 건너뜀")
            return None

        x_bev, y_bev = float(bev[0][0]), float(bev[0][1])

        warp_width, warp_height = self._perspective.warp_size
        if not (0 <= y_bev < warp_height and 0 <= x_bev < warp_width):
            # 변환 영역 밖 — 전방 차로 ROI를 벗어난 차량이라 차선과 비교할 수 없다
            return None

        left_x = float(np.polyval(lane.left_fit, y_bev))
        right_x = float(np.polyval(lane.right_fit, y_bev))
        lane_width_px = right_x - left_x
        if lane_width_px <= 0:
            return None

        lane_center_x = (left_x + right_x) / 2
        offset_ratio = (x_bev - lane_center_x) / lane_width_px  # 우측 양수
        return offset_ratio * self._lane_width_m

    def _append(
        self,
        track: TrackedVehicle,
        offset_m: float,
        lane_confidence: float,
        timestamp: float,
    ) -> None:
        raw = self._raw_offsets.setdefault(track.track_id, deque())
        raw.append((timestamp, offset_m))
        self._trim(raw, timestamp)

        smoothed_m = self._smooth(raw)
        # offset_m = offset_ratio * lane_width_m 이므로 비율은 평활화된 미터값을
        # 되나누기만 하면 된다. 두 값을 따로 평활화하면 서로 어긋날 수 있다.
        smoothed_ratio = smoothed_m / self._lane_width_m

        series = self._series.setdefault(
            track.track_id, VehicleTimeSeries(track_id=track.track_id)
        )
        series.offsets.append(
            OffsetSample(
                timestamp=timestamp,
                offset_m=smoothed_m,
                offset_ratio=smoothed_ratio,
                lateral_velocity_mps=self._lateral_velocity(series, smoothed_m, timestamp),
                lane_confidence=lane_confidence,
                is_interpolated=track.is_interpolated,
                bbox_height=track.box.y2 - track.box.y1,
            )
        )
        self._trim(series.offsets, timestamp)

    def _smooth(self, raw: deque[tuple[float, float]]) -> float:
        """최근 offset을 평활화한 값. 단일 프레임 이상치를 제거한다."""
        if self._smoothing != "moving_average":
            # TODO(S4): 칼만 필터 옵션. 현재는 이동평균만 구현되어 있다.
            logger.warning(
                "metrics.smoothing='%s'는 아직 구현되지 않아 이동평균으로 대체합니다.",
                self._smoothing,
            )
        window = list(raw)[-self._smoothing_window :]
        return float(np.mean([value for _, value in window]))

    def _lateral_velocity(
        self, series: VehicleTimeSeries, smoothed_m: float, timestamp: float
    ) -> float:
        """평활화된 offset의 시간 미분 (m/s, 우측 양수).

        인접 두 프레임의 차분은 잡음이 그대로 증폭되므로(risk-criteria.md 2.6),
        평활화 창과 같은 길이만큼 뒤의 샘플을 기준점으로 삼아 기울기를 낸다.
        """
        if not series.offsets:
            return 0.0

        index = max(0, len(series.offsets) - self._smoothing_window)
        reference = series.offsets[index]
        dt = timestamp - reference.timestamp
        if dt <= 0:
            return 0.0
        return (smoothed_m - reference.offset_m) / dt

    def _trim(self, buffer: deque, now: float) -> None:
        """관측 구간(기본 10초)을 벗어난 오래된 샘플을 버린다."""
        cutoff = now - self._window_sec
        while buffer:
            head = buffer[0]
            ts = head[0] if isinstance(head, tuple) else head.timestamp
            if ts < cutoff:
                buffer.popleft()
            else:
                break

    def _forget_disappeared(self, alive: set[int]) -> None:
        """화면에서 사라진 차량의 버퍼를 해제한다 (pipeline-architecture.md 3.5)."""
        for track_id in list(self._series):
            if track_id not in alive:
                del self._series[track_id]
                self._raw_offsets.pop(track_id, None)
