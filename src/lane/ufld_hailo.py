"""UFLDv2 Hailo NPU 백엔드 — `.hef`를 HailoRT로 실행한다 (docs/lane-detection-dl-plan.md D7).

`OnnxUFLDBackend`와 같은 모양(`infer(x) -> (loc_row, exist_row)`)이라 전처리·해석·안전장치는 그대로 쓴다.

HEF는 변환 방법에 따라 입출력 모양이 달라서, 아래 두 가지를 실행할 때 맞춘다.

- **입력**: Hailo는 NHWC다. 전처리 결과(NCHW, ImageNet 정규화)를 NHWC로 바꿔 넣는다. 변환할 때
  정규화 층을 모델 안에 넣었으면(`normalization` 모델 스크립트) RGB 0~255를 넣어야 한다 —
  `lane.ufld.hailo.input: rgb255`. 어느 쪽인지는 `scripts/check_hailo_ufld.py`가 ONNX와 비교해 알려 준다.
- **출력**: 완전연결층 출력이 통째로 한 개(loc+exist 길이)로 나오거나, 두 개로 나뉘어 나올 수 있다.
  나뉜 경우 Hailo는 NCHW 텐서를 NHWC (num_row, lanes, C)로 돌려주므로 (C, num_row, lanes)로 되돌린다.
  `split_outputs`가 크기로 구분한다.

hailo_platform(HailoRT 파이썬 바인딩)은 Pi에 `sudo apt install hailo-all`로 시스템에 깔린다 —
가상환경은 `--system-site-packages`로 만들어야 보인다.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from src.lane.ufld import _MEAN, _STD, UFLDSpec

logger = logging.getLogger("safewatch.lane.ufld_hailo")

INPUT_MODES = ("normalized", "rgb255")


def to_nhwc(x: np.ndarray, mode: str) -> np.ndarray:
    """전처리 결과 (1, 3, H, W) 정규화 float32 → Hailo 입력 (1, H, W, 3).

    normalized = 그대로(float32, HailoRT가 양자화), rgb255 = 정규화를 되돌린 RGB 0~255 uint8.
    """
    nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    if mode == "normalized":
        return nhwc.astype(np.float32, copy=False)
    if mode == "rgb255":
        rgb = (nhwc * _STD + _MEAN) * 255.0
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    raise ValueError(f"lane.ufld.hailo.input '{mode}'를 모릅니다 — {INPUT_MODES} 중 하나")


def _as_chw(a: np.ndarray, c: int, spec: UFLDSpec) -> np.ndarray:
    """출력 하나 → (c, num_row, num_lanes). Hailo NHWC면 축을 되돌리고, 납작하면 C 순서로 접는다."""
    want = (c, spec.num_row, spec.num_lanes)
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.shape == want:
        return a
    if a.shape == (spec.num_row, spec.num_lanes, c):
        return np.ascontiguousarray(a.transpose(2, 0, 1))
    return a.reshape(want)


def split_outputs(outputs: dict[str, np.ndarray], spec: UFLDSpec) -> tuple[np.ndarray, np.ndarray]:
    """HEF 출력 → (loc_row (grid, num_row, lanes), exist_row (2, num_row, lanes)). 배치 1."""
    n_loc = spec.grid_num * spec.num_row * spec.num_lanes
    n_exist = 2 * spec.num_row * spec.num_lanes
    sizes = {name: int(np.asarray(v).size) for name, v in outputs.items()}

    whole = [n for n, s in sizes.items() if s == n_loc + n_exist]
    if len(outputs) == 1 and whole:                      # 완전연결층 출력 통째 — export 순서 loc, exist
        flat = np.asarray(outputs[whole[0]], dtype=np.float32).reshape(-1)
        return flat[:n_loc].reshape(spec.grid_num, spec.num_row, spec.num_lanes), \
            flat[n_loc:].reshape(2, spec.num_row, spec.num_lanes)

    loc = [n for n, s in sizes.items() if s == n_loc]
    exist = [n for n, s in sizes.items() if s == n_exist]
    if len(loc) == 1 and len(exist) == 1:
        return _as_chw(outputs[loc[0]], spec.grid_num, spec), _as_chw(outputs[exist[0]], 2, spec)

    shapes = {n: tuple(np.asarray(v).shape) for n, v in outputs.items()}
    raise ValueError(
        f"HEF 출력 {shapes}을 loc_row({n_loc}개)·exist_row({n_exist}개)로 나눌 수 없습니다 — "
        "lane.ufld 형상 설정이 변환한 모델과 같은지, 변환 때 출력 층을 어디로 잡았는지 확인할 것"
    )


class HailoUFLDBackend:
    """HailoRT로 HEF 실행. 장치·스트림은 생성할 때 한 번 열어 두고 `close()`까지 유지한다."""

    def __init__(self, hef_path: str | Path, spec: UFLDSpec, input_mode: str = "normalized"):
        if input_mode not in INPUT_MODES:
            raise ValueError(f"lane.ufld.hailo.input '{input_mode}'를 모릅니다 — {INPUT_MODES} 중 하나")
        path = Path(hef_path)
        if not path.exists():
            raise FileNotFoundError(f"UFLD HEF 파일이 없습니다: {path}")
        try:
            from hailo_platform import (  # type: ignore[import-not-found]
                HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                InputVStreamParams, OutputVStreamParams, VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform을 import할 수 없습니다. Pi에서 'sudo apt install hailo-all' 후, "
                "가상환경을 '--system-site-packages'로 만들었는지 확인하세요."
            ) from exc

        self._spec = spec
        self._mode = input_mode
        hef = HEF(str(path))
        self._in_info = hef.get_input_vstream_infos()[0]
        want = (spec.input_h, spec.input_w, 3)
        if tuple(self._in_info.shape) != want:
            raise ValueError(f"HEF 입력 {tuple(self._in_info.shape)}이 설정 (H, W, C) {want}과 다릅니다")

        self._stack = ExitStack()
        try:
            self._device = self._stack.enter_context(VDevice())
            params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
            self._group = self._device.configure(hef, params)[0]
            in_fmt = FormatType.UINT8 if input_mode == "rgb255" else FormatType.FLOAT32
            in_params = InputVStreamParams.make(self._group, format_type=in_fmt)
            out_params = OutputVStreamParams.make(self._group, format_type=FormatType.FLOAT32)
            self._pipe = self._stack.enter_context(InferVStreams(self._group, in_params, out_params))
            self._stack.enter_context(self._group.activate(self._group.create_params()))
        except Exception:
            self._stack.close()
            raise
        outs = {o.name: tuple(o.shape) for o in hef.get_output_vstream_infos()}
        logger.info("UFLD HEF 로드: %s (입력 %s %s, 출력 %s)", path, self._in_info.shape, input_mode, outs)

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self._pipe.infer({self._in_info.name: to_nhwc(x, self._mode)})
        return split_outputs(result, self._spec)

    def close(self) -> None:
        self._stack.close()
