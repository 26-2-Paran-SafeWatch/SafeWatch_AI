"""딥러닝(UFLDv2) 차선 인식 경로. docs/lane-detection-dl-plan.md 참고.

`lane.method: ufld`일 때 `create_lane_detector()`가 이 클래스를 만든다. 출력은 OpenCV 경로와
같은 `LaneModel`이지만 좌표계가 원본 화면(`space="image"`)이라 조감도 보정이 필요 없다
(결정 D3). METRICS는 `LaneModel.space`를 보고 offset 계산 방식을 고른다.

처리 순서
1. 전처리 → 추론(백엔드: ONNX Runtime, 향후 NPU) → 행 방향 출력 해석 (`src/lane/ufld.py`)
2. 자차 차로 = 차선 번호 `lane.ufld.ego_lane_ids` (CULane 가중치 1·2)
3. 안전장치 (`src/lane/ufld_guard.py`) — 존재 비율·형상·걸침·차로 폭·(선택)일관성.
   판정 구간(`guard_range_ratio`)과 offset 측정 구간(`measure_range_ratio`)은 따로다

실패 처리 — 추론이나 해석에서 예외가 나도 파이프라인을 멈추지 않고 그 프레임을 보류한다
(AGENTS.md "예외 처리"). 출력 형상이 설정과 다르면 시작할 때 바로 에러를 낸다(잘못된 가중치·설정).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.detection.types import VehicleBox
from src.lane.types import LaneModel
from src.lane.ufld import OnnxUFLDBackend, UFLDBackend, UFLDSpec, decode_rows, preprocess
from src.lane.ufld_guard import LaneGuard
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.lane")


class UFLDLaneDetector:
    # OpenCV 경로와 같은 모양을 맞추기 위한 속성 — 원본 화면 기준이라 조감도 변환기가 없다
    perspective = None

    def __init__(self, cfg: Config, backend: UFLDBackend | None = None):
        u = cfg.lane.ufld
        self._spec = UFLDSpec.from_config(u)
        self._ego = tuple(int(i) for i in u.ego_lane_ids)
        if len(self._ego) != 2 or not all(0 <= i < self._spec.num_lanes for i in self._ego):
            raise ValueError(f"lane.ufld.ego_lane_ids가 잘못됐습니다: {u.ego_lane_ids}")
        # 판정(안전장치)은 가까운 구간, 측정은 더 먼 곳까지 — 판정을 통과한 프레임이면 측정 구간 어느 행에서든
        # 예측 차선으로 바로 잰다 (노트북 v2 29번 셀 '① 직접')
        self._top_ratio, self._bottom_ratio = (float(v) for v in u.measure_range_ratio)
        if not float(u.row_anchor_range[0]) <= self._top_ratio < self._bottom_ratio <= 1.0:
            raise ValueError(
                f"lane.ufld.measure_range_ratio가 잘못됐습니다: {u.measure_range_ratio} "
                f"(행 앵커 위끝 {u.row_anchor_range[0]} 이상, 위 < 아래 ≤ 1)"
            )
        self._guard = LaneGuard(u.guard, self._spec.row_anchor, tuple(float(v) for v in u.guard_range_ratio))
        self._backend = backend if backend is not None else OnnxUFLDBackend(u.model_path, u.num_threads)
        self._verify_backend()

    def _verify_backend(self) -> None:
        """빈 입력 한 번으로 출력 형상을 확인한다 (첫 추론 지연을 미리 치르는 효과도 있다)."""
        s = self._spec
        x = np.zeros((1, 3, s.input_h, s.input_w), dtype=np.float32)
        loc, exist = self._backend.infer(x)
        want_loc, want_exist = (s.grid_num, s.num_row, s.num_lanes), (2, s.num_row, s.num_lanes)
        if tuple(loc.shape) != want_loc or tuple(exist.shape) != want_exist:
            raise ValueError(
                f"UFLD 출력 형상 {tuple(loc.shape)}, {tuple(exist.shape)}이 설정 {want_loc}, {want_exist}과 다릅니다 "
                "— 가중치와 lane.ufld 설정(grid_num·num_row·num_lanes)을 맞출 것"
            )

    def reset(self) -> None:
        """영상(주행)이 바뀔 때 부른다 — 차로 폭 기록을 비운다."""
        self._guard.reset()

    def detect(
        self,
        image: np.ndarray,
        vehicle_boxes: list[VehicleBox] | None = None,   # OpenCV 경로와 같은 호출 모양. 지금은 쓰지 않음
        timestamp: float | None = None,
    ) -> LaneModel:
        """timestamp(초)는 차로 폭 기록 창과 일관성 검사에 쓴다. 파이프라인은 프레임 시각을 넘기고,
        생략하면 호출 시각(time.monotonic)을 쓴다."""
        h, w = image.shape[:2]
        measure = (self._top_ratio * h, self._bottom_ratio * h)
        with profiler.stage("lane"):
            try:
                with profiler.stage("lane_infer"):
                    loc, exist = self._backend.infer(preprocess(image, self._spec))
                lanes = decode_rows(loc, exist, self._spec, w, h)
                left, right = lanes[self._ego[0]], lanes[self._ego[1]]
                ts = time.monotonic() if timestamp is None else timestamp
                result = self._guard.check(left, right, w, h, ts)
            except Exception:  # noqa: BLE001 — 프레임 하나의 실패가 파이프라인을 멈추면 안 된다
                logger.exception("UFLD 차선 인식 실패 — 이 프레임은 보류")
                return LaneModel(
                    left_fit=None, right_fit=None, lane_width_px=None, curvature_radius=None,
                    confidence=0.0, valid=False, space="image", measure_y_range=measure,
                    reject_reason="error",
                )
        return LaneModel(
            left_fit=None, right_fit=None, lane_width_px=result.width_px, curvature_radius=None,
            confidence=result.confidence, valid=result.valid, space="image",
            left_points=left, right_points=right, measure_y_range=measure,
            reject_reason=result.reason,
        )
