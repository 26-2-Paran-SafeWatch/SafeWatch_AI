"""lane offset 산출과 시계열 축적. docs/pipeline-architecture.md 3.5 참고.

차량 위치와 차선의 관계를 수치화한다. 채택된 판별 단서 4개(사행·차선 걸침·
스웨빙·표류)가 **모두 이 offset 시계열에서 파생**되므로, 이 모듈의 출력이
곧 판정의 입력 전부다 (risk-criteria.md 1.4).

좌표계 주의 — 차선 모델(`LaneModel.space`)에 따라 재는 방식이 다르다.
- "bev"(opencv 경로): 차선은 bird's-eye-view 다항식, 검출·추적 박스는 원본 프레임
  좌표계다. 차량 위치를 BEV로 옮긴 뒤 비교한다.
- "image"(ufld 경로, docs/lane-detection-dl-plan.md D3): 차선도 원본 화면 점이다.
  차량 바닥 높이 행에서 좌우 차선 x를 읽어 바로 비교한다. 평평한 도로를 정면으로 보는
  카메라(기울기 없음)에서 화면의 한 행은 카메라에서 같은 거리의 노면이라, 그 행 안에서는
  가로 거리 비율이 보존된다 — 그래서 조감도 변환 없이 같은 offset이 나온다.
  측정은 `LaneModel.measure_y_range` 안에서만 한다(그 밖은 차선을 믿을 범위가 아님).

차로 배정 — offset은 자차 차로가 아니라 **그 차량이 달리는 차로** 기준이다
(risk-criteria.md 2.1). 자차 차로 기준으로만 재면 옆 차로를 정상 주행하는
차량이 offset ±1로 찍혀 걸침·표류를 항상 충족한다. 그래서
1. 자차 차로 기준 위치 r을 재고, `round(r)`이 `lane.max_lane_index`를 넘으면 버린다
2. 직전 값과 차로 1개 단위로 어긋나면 그만큼 보정해 **연속 횡위치**로 잇는다
   (자차가 차로를 바꿔 기준이 한 차로 밀려도 끊기지 않게)
3. 연속 횡위치를 평활화하고 횡방향 속도도 여기서 낸다
4. 그 차량의 차로를 빼 차로 내 offset을 낸다 — 차로 배정은 offset이
   `0.5 + metrics.lane_switch_hysteresis`를 넘을 때만 바꾼다. 선을 밟고 달리는
   차량의 offset이 매 프레임 +0.5/−0.5로 뒤집혀 사행으로 세지는 것을 막는다
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from src.lane.geometry import lane_at_row
from src.lane.perspective import PerspectiveTransformer
from src.lane.types import LaneModel
from src.metrics.types import OffsetSample, VehicleTimeSeries
from src.tracking.types import TrackedVehicle
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.metrics")


class OffsetCalculator:
    """차량별 lane offset 시계열을 유지한다."""

    def __init__(self, cfg: Config, perspective: PerspectiveTransformer | None):
        """perspective는 opencv(조감도) 경로에서만 쓴다. ufld 경로는 None을 넘긴다."""
        self._perspective = perspective
        self._lane_width_m: float = cfg.lane.lane_width_m
        self._max_lane_index: int = cfg.lane.max_lane_index

        m_cfg = cfg.metrics
        self._window_sec: float = m_cfg.observation_window_sec
        self._smoothing: str = m_cfg.smoothing
        self._smoothing_window: int = m_cfg.smoothing_window
        self._lane_switch_at: float = 0.5 + m_cfg.lane_switch_hysteresis

        self._series: dict[int, VehicleTimeSeries] = {}
        # 평활화 전 연속 횡위치(m) — 평활화 결과만 저장하면 창 크기를 바꿔 재계산할 수 없다
        self._raw_offsets: dict[int, deque[tuple[float, float]]] = {}
        # 차량별 배정 차로 (연속 횡위치 좌표계의 정수, 차로 폭 단위)
        self._assigned_lane: dict[int, int] = {}

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

            if not lane.valid:
                return self._series

            measure = self._measure_image if lane.space == "image" else self._measure
            for track in tracks:
                ratio = measure(track, lane)
                if ratio is None:
                    continue
                self._append(track, ratio, lane.confidence, timestamp)

            return self._series

    def _measure(self, track: TrackedVehicle, lane: LaneModel) -> float | None:
        """자차 차로 중심 기준 위치(차로 폭 단위, 우측 양수). 측정 불가하면 None.

        옆 차로 차량은 ±1 근처가 나온다. `lane.max_lane_index`보다 먼 차로는
        차선 다항식 외삽을 믿을 수 없어 버린다.
        """
        if lane.left_fit is None or lane.right_fit is None or self._perspective is None:
            return None
        try:
            bev = self._perspective.warp_points(
                np.array([[track.center_x, track.bottom_y]], dtype=np.float32)
            )
        except RuntimeError:
            # warp()가 아직 한 번도 호출되지 않음 — 차선 인식보다 먼저 불린 경우
            logger.debug("perspective 행렬 미생성 — offset 산출 건너뜀")
            return None

        x_bev, y_bev = float(bev[0][0]), float(bev[0][1])

        _, warp_height = self._perspective.warp_size
        if not (0 <= y_bev < warp_height):
            # 보정 구간(조감도 높이) 밖 — 그 거리에서는 차선을 관측하지 않았으므로
            # 다항식을 위아래로 외삽한 값을 믿을 수 없다
            return None
        # 가로 방향은 조감도 폭을 넘어도 잰다 — 옆 차로 중심은 조감도 밖에 있다.
        # 두 차선이 같은 모양으로 피팅되어 있어 옆으로의 외삽은 안정적이고,
        # 얼마나 멀리까지 믿을지는 아래 max_lane_index가 정한다

        left_x = float(np.polyval(lane.left_fit, y_bev))
        right_x = float(np.polyval(lane.right_fit, y_bev))
        lane_width_px = right_x - left_x
        if lane_width_px <= 0:
            return None

        lane_center_x = (left_x + right_x) / 2
        ratio = (x_bev - lane_center_x) / lane_width_px  # 우측 양수
        if abs(round(ratio)) > self._max_lane_index:
            return None
        return ratio

    def _measure_image(self, track: TrackedVehicle, lane: LaneModel) -> float | None:
        """원본 화면(ufld 경로) 기준 위치 — 차량 바닥 행에서 자차 차로 중심 대비, 차로 폭 단위(우측 양수).

        `_measure`와 같은 규약(옆 차로 ±1, max_lane_index 밖 제외)을 따른다.
        """
        if lane.measure_y_range is None:
            return None
        y = track.bottom_y
        y_top, y_bottom = lane.measure_y_range
        if not (y_top <= y <= y_bottom):
            # 측정 구간 밖 — 그 거리의 차선 위치를 믿을 근거가 없다 (조감도 경로의 '보정 구간 밖'과 같음)
            return None
        row = lane_at_row(lane.left_points, lane.right_points, y)
        if row is None:
            return None
        left_x, right_x = row
        ratio = (track.center_x - (left_x + right_x) / 2) / (right_x - left_x)
        if abs(round(ratio)) > self._max_lane_index:
            return None
        return ratio

    def _append(
        self,
        track: TrackedVehicle,
        ego_ratio: float,
        lane_confidence: float,
        timestamp: float,
    ) -> None:
        lane_width_m = self._lane_width_m
        raw = self._raw_offsets.setdefault(track.track_id, deque())
        # 연속 횡위치로 잇기 — 직전 위치와 차로 1개 단위로 어긋난 만큼을 정수로 보정한다.
        # 실제 횡이동은 한 프레임에 차로 반 개를 넘지 않으므로 이 보정은 기준 이동만 지운다.
        shift = round(raw[-1][1] / lane_width_m - ego_ratio) if raw else 0
        raw.append((timestamp, (ego_ratio + shift) * lane_width_m))
        self._trim(raw, timestamp)

        position_m = self._smooth(raw)
        position_ratio = position_m / lane_width_m

        lane = self._assigned_lane.get(track.track_id)
        if lane is None or abs(position_ratio - lane) > self._lane_switch_at:
            lane = round(position_ratio)
            self._assigned_lane[track.track_id] = lane
        # offset_m = offset_ratio * lane_width_m — 평활화된 위치에서 함께 내므로 서로 어긋나지 않는다
        smoothed_ratio = position_ratio - lane
        smoothed_m = smoothed_ratio * lane_width_m

        series = self._series.setdefault(
            track.track_id, VehicleTimeSeries(track_id=track.track_id)
        )
        series.offsets.append(
            OffsetSample(
                timestamp=timestamp,
                offset_m=smoothed_m,
                offset_ratio=smoothed_ratio,
                lateral_velocity_mps=self._lateral_velocity(series, position_m, timestamp),
                lane_confidence=lane_confidence,
                is_interpolated=track.is_interpolated,
                bbox_height=track.box.y2 - track.box.y1,
                lane_index=lane - shift,            # 연속 좌표의 차로 → 자차 차로 기준 번호
                position_m=position_m,
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
        self, series: VehicleTimeSeries, position_m: float, timestamp: float
    ) -> float:
        """평활화된 연속 횡위치의 시간 미분 (m/s, 우측 양수).

        차로 내 offset이 아니라 연속 횡위치를 쓰는 이유 — 차로 배정이 바뀌는 순간
        offset은 차로 폭만큼 점프하지만 실제 이동은 없다 (risk-criteria.md 2.1).

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
        return (position_m - reference.position_m) / dt

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
                self._assigned_lane.pop(track_id, None)
