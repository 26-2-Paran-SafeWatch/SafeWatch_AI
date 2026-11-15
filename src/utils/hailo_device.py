"""Hailo-8 장치(VDevice) 공유 — 차량 검출(YOLO)과 차선(UFLD)이 칩 하나를 같이 쓴다.

모델마다 VDevice를 따로 열면 두 번째가 `HAILO_OUT_OF_PHYSICAL_DEVICES`로 실패한다(2026-11-15 Pi 확인 —
Ultralytics Hailo 백엔드가 스케줄러 없이 장치를 따로 열어 차선 모델과 충돌). 그래서 프로세스에서 VDevice를
하나만 만들고, ROUND_ROBIN 스케줄러로 두 모델이 번갈아 쓰게 한다. 스케줄러가 켜져 있으면 네트워크 그룹을
직접 activate하지 않는다(스케줄러가 맡는다).

`acquire()`로 받고 다 쓰면 `release()` — 마지막 사용자가 놓을 때 장치를 닫는다.
hailo_platform은 함수 안에서 import한다 (맥·PC·테스트는 import 없이 돈다).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("safewatch.hailo")

_lock = threading.Lock()
_vdevice: Any = None
_refs = 0


def acquire() -> Any:
    """공유 VDevice를 돌려준다. 처음 부를 때 연다."""
    global _vdevice, _refs
    with _lock:
        if _vdevice is None:
            try:
                from hailo_platform import HailoSchedulingAlgorithm, VDevice  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "hailo_platform을 import할 수 없습니다 — Pi에서 'sudo apt install hailo-all' 후 "
                    "가상환경이 시스템 패키지를 보는지(--system-site-packages) 확인할 것"
                ) from exc
            params = VDevice.create_params()
            # 필수 — 여러 모델이 칩을 나눠 쓰고, 빠뜨리면 InferModel 쪽이 HAILO_STREAM_NOT_ACTIVATED로 출력이 빈다
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            _vdevice = VDevice(params)
            logger.info("Hailo VDevice 열림 (ROUND_ROBIN 스케줄러)")
        _refs += 1
        return _vdevice


def release() -> None:
    """`acquire()` 한 번에 한 번. 마지막이면 장치를 닫는다."""
    global _vdevice, _refs
    with _lock:
        if _refs <= 0:
            return
        _refs -= 1
        if _refs == 0 and _vdevice is not None:
            _vdevice.release()
            _vdevice = None
            logger.info("Hailo VDevice 닫힘")
