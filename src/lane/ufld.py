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


def preprocess_uint8(bgr: np.ndarray, spec: UFLDSpec) -> np.ndarray:
    """BGR 프레임 → Hailo 입력 (input_h, input_w, 3) RGB uint8 HWC. 자르기는 `preprocess`와 같고,
    ImageNet 정규화는 HEF 안에 들어 있어 하지 않는다 (docs/ufld-hailo-conversion.md 모델 스크립트)."""
    resized_h = int(spec.input_h / spec.crop_ratio)
    resized = cv2.resize(bgr, (spec.input_w, resized_h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized[-spec.input_h:, :, ::-1])


INPUT_FORMATS = ("float_nchw", "uint8_hwc")


def preprocess_for(bgr: np.ndarray, spec: UFLDSpec, input_format: str) -> np.ndarray:
    """백엔드 입력 형식에 맞는 전처리. float_nchw = `preprocess`, uint8_hwc = `preprocess_uint8`."""
    if input_format == "float_nchw":
        return preprocess(bgr, spec)
    if input_format == "uint8_hwc":
        return preprocess_uint8(bgr, spec)
    raise ValueError(f"UFLD 백엔드 입력 형식 '{input_format}'을 모릅니다 — {INPUT_FORMATS} 중 하나")


def blank_input(spec: UFLDSpec, input_format: str) -> np.ndarray:
    """형식별 빈 입력 — 시작할 때 출력 형상 확인용."""
    if input_format == "uint8_hwc":
        return np.zeros((spec.input_h, spec.input_w, 3), dtype=np.uint8)
    return np.zeros((1, 3, spec.input_h, spec.input_w), dtype=np.float32)


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
    """추론 엔진 추상화 — ONNX Runtime(CPU), Hailo NPU, 테스트용 가짜 구현이 같은 모양을 따른다.

    `input_format`으로 받을 입력을 밝힌다 (`preprocess_for`). 없으면 float_nchw로 본다.
    자원을 쥐는 백엔드(NPU)는 `close()`를 둔다 — UFLDLaneDetector.close()가 부른다."""

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """입력 (float_nchw (1, 3, H, W) | uint8_hwc (H, W, 3)) →
        (loc_row (grid, num_row, lanes), exist_row (2, num_row, lanes))."""
        ...


class OnnxUFLDBackend:
    """ONNX Runtime CPU 실행. 출력 이름은 export 스크립트가 붙인 'loc_row', 'exist_row'."""

    input_format = "float_nchw"

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


class HailoUFLDBackend:
    """Hailo-8 NPU + CPU 분할 실행 (docs/ufld-hailo-conversion.md).

    - NPU(HEF): 입력 ~ `/pool/Conv` (백본, 계산량 대부분). Hailo 파서가 그 뒤 LayerNorm을 못 받고,
      마지막 완전연결층은 가중치가 커서(fp32 487MB) 칩에 안 들어간다.
    - CPU(`tail_path`, ONNX Runtime): 원본 ONNX에서 `/pool/Conv` 뒤를 떼어낸 모델. 출력 이름·형상이
      원본과 같다 (loc_row, exist_row).

    입력은 RGB uint8 HWC — ImageNet 정규화는 HEF 안에 있다. HEF 출력 (10, 50, 8) HWC는 FLOAT32로 받으면
    HailoRT가 양자화를 풀어 주고, (1, 8, 10, 50)로 돌려 tail에 넣는다.
    HW 파트가 Pi(HailoRT 4.23)에서 돌려 확인한 구현을 옮겼다.
    """

    input_format = "uint8_hwc"
    _TIMEOUT_MS = 2000

    def __init__(self, hef_path: str | Path, tail_path: str | Path, num_threads: int = 0):
        for p in (hef_path, tail_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"UFLD NPU 모델 파일이 없습니다: {p} — docs/ufld-hailo-conversion.md로 만들 것")
        try:
            # Pi에 apt(hailo-all)로 깔린다. 맥·PC·가짜 백엔드 테스트는 이 import 없이 돈다
            from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform을 import할 수 없습니다 — Pi에서 'sudo apt install hailo-all' 후 "
                "가상환경이 시스템 패키지를 보는지(--system-site-packages) 확인할 것"
            ) from exc
        import onnxruntime as ort

        params = VDevice.create_params()
        # 필수 — 빠뜨리면 HAILO_STREAM_NOT_ACTIVATED로 출력이 빈다 (hailo-apps 공식 코드와 같은 방식)
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)
        self._ctx = None
        try:
            model = self._vdevice.create_infer_model(str(hef_path))
            model.input().set_format_type(FormatType.UINT8)
            model.output().set_format_type(FormatType.FLOAT32)
            self._ctx = model.configure()
            self._cfg = self._ctx.__enter__()
            self._out = np.empty(model.output().shape, np.float32)      # (10, 50, 8) HWC
            self._bind = self._cfg.create_bindings()
            self._bind.output().set_buffer(self._out)

            opts = ort.SessionOptions()
            if num_threads > 0:
                opts.intra_op_num_threads = num_threads
            self._tail = ort.InferenceSession(str(tail_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self._tail_in = self._tail.get_inputs()[0].name
        except Exception:
            self.close()
            raise
        logger.info("UFLD NPU 로드: %s (+ CPU %s), HEF 출력 %s", hef_path, tail_path, self._out.shape)

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._bind.input().set_buffer(np.ascontiguousarray(x))
        self._cfg.run([self._bind], self._TIMEOUT_MS)
        feat = np.ascontiguousarray(self._out.transpose(2, 0, 1)[None])
        loc, exist = self._tail.run(["loc_row", "exist_row"], {self._tail_in: feat})
        return loc[0], exist[0]

    def close(self) -> None:
        """NPU 자원 해제. 여러 번 불러도 된다."""
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        if self._vdevice is not None:
            self._vdevice.release()
            self._vdevice = None
