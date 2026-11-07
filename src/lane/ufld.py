"""UFLDv2 차선 모델 — 전처리, 추론 백엔드, 행 방향 출력 해석.

결정과 근거는 docs/lane-detection-dl-plan.md. 모델 구조·가중치 변환은
`scripts/export_ufld_onnx.py`에 있다 (PyTorch는 개발 PC에서만 쓰고, 실행 장치는 ONNX만 연다).

전처리와 해석은 노트북 `notebook/safewatch_lane_calib_v2.ipynb` 15번 셀 `ufld_lanes`
(원 저장소 demo.py `pred2coords`와 같은 결과임을 확인함)를 그대로 옮겼다. 노트북에서 정한
임계값(존재 비율 등)이 이 코드에서도 같은 뜻이 되도록, 식을 바꾸지 않는다.

- 전처리: 화면 전체를 (in_w, in_h / crop_ratio)로 늘린 뒤 아래 in_h 행만 남긴다(원 저장소 데모와 같은 크롭).
  RGB, ImageNet 평균·표준편차 정규화, NCHW float32.
- 해석: 차선마다 행 앵커별 '존재' 확률이 0.5를 넘는 행만 쓴다. 그 행의 x는 최댓값 칸과 좌우 1칸의
  softmax 가중 평균 칸 위치(+0.5)를 격자 수로 나눠 화면 폭에 곱한다. y는 행 앵커 비율 × 화면 높이.
- 차선 번호는 왼쪽부터 0..num_lanes-1. CULane 가중치는 자차 차로 = 1·2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from src.utils.config import Config

logger = logging.getLogger("safewatch.lane.ufld")

# 원 저장소 학습 시 정규화 값 (ImageNet). 튜닝 대상이 아니라 모델의 일부라 상수로 둔다.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class UFLDSpec:
    """모델 입출력 형상. 가중치(학습 설정)와 반드시 같아야 한다 — 원 저장소 configs/culane_res18.py."""

    input_w: int
    input_h: int
    crop_ratio: float
    grid_num: int
    num_row: int
    row_anchor_start: float
    row_anchor_end: float
    num_lanes: int

    @property
    def row_anchor(self) -> np.ndarray:
        return np.linspace(self.row_anchor_start, self.row_anchor_end, self.num_row)

    @classmethod
    def from_config(cls, ufld_cfg: Config) -> "UFLDSpec":
        w, h = ufld_cfg.input_size
        a0, a1 = ufld_cfg.row_anchor_range
        return cls(
            input_w=int(w), input_h=int(h), crop_ratio=float(ufld_cfg.crop_ratio),
            grid_num=int(ufld_cfg.grid_num), num_row=int(ufld_cfg.num_row),
            row_anchor_start=float(a0), row_anchor_end=float(a1), num_lanes=int(ufld_cfg.num_lanes),
        )


_SCALE = (1.0 / (255.0 * _STD)).reshape(1, 3, 1, 1)
_BIAS = (-_MEAN / _STD).reshape(1, 3, 1, 1)


def preprocess(bgr: np.ndarray, spec: UFLDSpec) -> np.ndarray:
    """BGR 프레임 → 모델 입력 (1, 3, input_h, input_w) float32.

    (x/255 - mean)/std 를 x·scale + bias 한 번으로 계산하고, 채널 순서 변환·NCHW 배치는
    cv2.dnn.blobFromImage에 맡긴다 — numpy로 단계별 계산하는 것과 결과가 같고(차이 < 1e-6)
    1280×720 입력에서 약 17ms → 1ms (개발 맥 기준, 2026-11-07). Pi5는 CPU에서 도는 부분이라 중요하다.
    """
    resized_h = int(spec.input_h / spec.crop_ratio)
    resized = cv2.resize(bgr, (spec.input_w, resized_h), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(resized[-spec.input_h:], 1.0, swapRB=True)   # RGB, NCHW float32
    blob *= _SCALE
    blob += _BIAS
    return blob


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def decode_rows(
    loc_row: np.ndarray, exist_row: np.ndarray, spec: UFLDSpec, image_w: int, image_h: int
) -> list[np.ndarray]:
    """행 방향 출력 → 차선별 원본 화면 점 목록.

    loc_row (grid_num, num_row, num_lanes), exist_row (2, num_row, num_lanes) — 배치 차원 없이.
    반환: num_lanes개의 (N, 2) [x, y] 배열 (y 오름차순, 존재 행만). 없으면 (0, 2).
    """
    g = spec.grid_num
    exist = exist_row[1] > exist_row[0]                      # softmax > 0.5 와 같음
    idx = loc_row.argmax(axis=0)                             # (num_row, num_lanes)
    ys = spec.row_anchor * image_h
    lanes: list[np.ndarray] = []
    for li in range(spec.num_lanes):
        pts = []
        for k in np.flatnonzero(exist[:, li]):
            i = int(idx[k, li])
            cells = np.arange(max(0, i - 1), min(g - 1, i + 1) + 1)
            w = _softmax(loc_row[cells, k, li], axis=0)
            xg = float((w * cells).sum()) + 0.5
            pts.append((xg / (g - 1) * image_w, ys[k]))
        lanes.append(np.array(pts, dtype=np.float64).reshape(-1, 2))
    return lanes


class UFLDBackend(Protocol):
    """추론 엔진 추상화 — ONNX Runtime, (향후) NPU 런타임, 테스트용 가짜 구현이 같은 모양을 따른다."""

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """입력 (1, 3, H, W) → (loc_row (grid, num_row, lanes), exist_row (2, num_row, lanes))."""
        ...


class OnnxUFLDBackend:
    """ONNX Runtime CPU 실행. 출력 이름은 export 스크립트가 붙인 'loc_row', 'exist_row'."""

    def __init__(self, model_path: str | Path, num_threads: int = 0):
        import onnxruntime as ort  # Pi5에서도 설치되는 패키지지만, 가짜 백엔드 테스트는 import 없이 돌게 한다

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"UFLD 모델 파일이 없습니다: {path} — scripts/export_ufld_onnx.py로 만들어 data/models/에 둘 것"
            )
        opts = ort.SessionOptions()
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
        self._session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        names = [o.name for o in self._session.get_outputs()]
        missing = {"loc_row", "exist_row"} - set(names)
        if missing:
            raise ValueError(f"UFLD ONNX 출력에 {sorted(missing)}가 없습니다 (출력: {names})")
        logger.info("UFLD ONNX 로드: %s", path)

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loc, exist = self._session.run(["loc_row", "exist_row"], {self._input: x})
        return loc[0], exist[0]
