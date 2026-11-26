"""YOLO Hailo NPU 실행 — Ultralytics `format=hailo` 내보내기 폴더(.hef + metadata.yaml + nms_config.json).

Ultralytics도 이 폴더를 직접 열 수 있지만, 스케줄러 없이 VDevice를 따로 열어서 차선 모델(UFLD)과 칩을 나눠
쓰지 못한다(`HAILO_OUT_OF_PHYSICAL_DEVICES`, 2026-11-15 Pi). 그래서 공유 VDevice(`src/utils/hailo_device.py`)
위에서 직접 실행한다. 전처리·출력 해석은 Ultralytics Hailo 백엔드(`ultralytics/nn/backends/hailo.py`)와 같게 했다.

- 입력: letterbox(비율 유지 축소 + 회색 114 여백), RGB uint8 HWC — Ultralytics 전처리와 같다.
- 출력: HEF 안에서 NMS까지 끝난 클래스별 박스 [y1, x1, y2, x2, score] (입력 크기 기준 0~1).
- NMS 점수·IoU 기준은 내보낼 때 `nms_config.json`에 박힌다. 실행 중 `detection.iou_threshold`는 쓰지 않고,
  `conf_threshold`는 그 위에서 한 번 더 거른다. 내보낼 때 점수 기준이 conf_threshold보다 높으면 낮은 박스는
  이미 빠져 있다 — 시작할 때 경고한다.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger("safewatch.detection.hailo")

PAD_VALUE = 114


def letterbox(bgr: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """BGR 프레임 → (RGB uint8 (h_in, w_in, 3), 배율 r, (왼쪽 여백, 위 여백)). Ultralytics LetterBox(center=True)와 같은 식."""
    w_in, h_in = size
    h, w = bgr.shape[:2]
    r = min(h_in / h, w_in / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (w_in - new_w) / 2, (h_in - new_h) / 2
    img = bgr if (w, h) == (new_w, new_h) else cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return np.ascontiguousarray(img[:, :, ::-1]), r, (left, top)


def decode_nms(
    per_class: list, input_size: tuple[int, int], r: float, pad: tuple[int, int],
    frame_size: tuple[int, int], conf: float,
) -> list[tuple[float, float, float, float, float, int]]:
    """HEF NMS 출력(배치 1장, 클래스별 [y1, x1, y2, x2, score] 0~1) → 원본 프레임 좌표 (x1, y1, x2, y2, score, class_id)."""
    w_in, h_in = input_size
    fw, fh = frame_size
    out = []
    for class_id, dets in enumerate(per_class):
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
        for y1, x1, y2, x2, score in dets:
            if score < conf:
                continue
            bx1 = (x1 * w_in - pad[0]) / r
            by1 = (y1 * h_in - pad[1]) / r
            bx2 = (x2 * w_in - pad[0]) / r
            by2 = (y2 * h_in - pad[1]) / r
            out.append((
                float(np.clip(bx1, 0, fw)), float(np.clip(by1, 0, fh)),
                float(np.clip(bx2, 0, fw)), float(np.clip(by2, 0, fh)), float(score), class_id,
            ))
    return out


def read_export(export_dir: str | Path) -> tuple[Path, dict[int, str], dict]:
    """내보내기 폴더 → (HEF 경로, 클래스 이름, nms_config). NMS가 들어간 detect HEF만 받는다."""
    d = Path(export_dir)
    hef = next(d.rglob("*.hef"), None)
    if hef is None:
        raise FileNotFoundError(f"HEF가 없습니다: {d} — Ultralytics format=hailo 내보내기 폴더를 지정할 것")
    meta_path = next(d.rglob("metadata.yaml"), None)
    if meta_path is None:
        raise FileNotFoundError(f"metadata.yaml이 없습니다: {d}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    if not meta.get("nms", False):
        raise ValueError(f"{hef.name}에 NMS가 없습니다 — detect 모델을 NMS 포함으로 내보낸 HEF만 지원")
    names = {int(k): str(v) for k, v in (meta.get("names") or {}).items()}
    nms_path = next(d.rglob("nms_config.json"), None)
    nms = json.loads(nms_path.read_text(encoding="utf-8")) if nms_path else {}
    return hef, names, nms


class HailoYolo:
    """공유 VDevice 위에서 YOLO HEF 실행. `predict(bgr, conf)` → 원본 좌표 박스 목록."""

    def __init__(self, export_dir: str | Path, conf_threshold: float):
        hef_path, self.names, nms = read_export(export_dir)
        th = (nms.get("nms_scores_th") if isinstance(nms, dict) else None)
        if th is not None and float(th) > conf_threshold:
            logger.warning(
                "HEF NMS 점수 기준 %.2f > detection.conf_threshold %.2f — 그 사이 박스는 칩에서 이미 빠진다 "
                "(낮은 박스로 트랙을 잇는 추적·먼 영역 검출에 영향). 내보낼 때 conf=%.2f로 다시 내보낼 것",
                float(th), conf_threshold, conf_threshold,
            )

        from src.utils import hailo_device

        self._vdevice = hailo_device.acquire()
        self._stack = ExitStack()
        try:
            from hailo_platform import (  # type: ignore[import-not-found]
                HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                InputVStreamParams, OutputVStreamParams,
            )

            hef = HEF(str(hef_path))
            self._in = hef.get_input_vstream_infos()[0]
            self._out = hef.get_output_vstream_infos()[0]
            params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            group = self._vdevice.configure(hef, params)[0]
            # 스케줄러가 켜진 장치라 activate하지 않는다 (src/utils/hailo_device.py)
            self._pipe = self._stack.enter_context(InferVStreams(
                group,
                InputVStreamParams.make(group, format_type=FormatType.UINT8),
                OutputVStreamParams.make(group, format_type=FormatType.FLOAT32),
            ))
        except Exception:
            self.close()
            raise
        h_in, w_in = self._in.shape[:2]
        self._size = (int(w_in), int(h_in))
        logger.info("YOLO HEF 로드: %s (입력 %s, 클래스 %s, NMS %s)", hef_path, self._size, self.names, nms)

    def predict(self, bgr: np.ndarray, conf: float) -> list[tuple[float, float, float, float, float, int]]:
        x, r, pad = letterbox(bgr, self._size)
        result = self._pipe.infer({self._in.name: x[None]})
        per_class = result[self._out.name][0]
        return decode_nms(per_class, self._size, r, pad, (bgr.shape[1], bgr.shape[0]), conf)

    def close(self) -> None:
        stack, self._stack = getattr(self, "_stack", None), None
        if stack is not None:
            stack.close()
        if getattr(self, "_vdevice", None) is not None:
            from src.utils import hailo_device

            hailo_device.release()
            self._vdevice = None
