"""영상 입력 소스 추상화.

PC/Pi5 플랫폼 차이는 이 모듈 안에서만 흡수한다 (AGENTS.md "개발 환경 차이").
상위 로직(main.py 이후 파이프라인)은 어떤 플랫폼에서 실행 중인지 알 필요가 없다.

- source == "camera" → picamera2 (Pi5 전용, 카메라 하드웨어 필요)
- 그 외 (파일 경로)  → cv2.VideoCapture

TODO(S1): picamera2 실기기 연동 구현 및 실측. 현재는 인터페이스만 정의.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Iterator

import cv2

from src.input.frame import Frame


class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """프레임을 순서대로 yield한다."""

    @abstractmethod
    def close(self) -> None:
        ...


class VideoFileSource(FrameSource):
    """PC 개발용 — 영상 파일 또는 웹캠 인덱스.

    타임스탬프 = 영상을 연 시각(epoch) + **영상 안에서의 경과 시간**. 처리 시각(time.time())을 쓰면
    PC가 빨리 돌릴수록 시간이 압축되고 느리면 늘어나서, 초 단위 창(offset 버퍼·차로 폭 기록·위험 판단
    지속 시간)이 영상 속 실제 시간과 어긋난다. 경과 시간은 컨테이너의 재생 위치(ms)를 우선 쓰고,
    못 읽으면 frame_id / fps로 계산한다. 웹캠(숫자 인덱스)은 실시간이라 처리 시각을 그대로 쓴다.
    """

    _FALLBACK_FPS = 30.0

    def __init__(self, path: str):
        self._live = path.isdigit()
        self._cap = cv2.VideoCapture(int(path) if self._live else path)
        if not self._cap.isOpened():
            raise RuntimeError(f"영상 소스를 열 수 없습니다: {path}")
        self._frame_id = 0
        self._t0 = time.time()
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if 0.0 < fps < 1000.0 else self._FALLBACK_FPS

    def _video_time(self) -> float:
        """방금 읽은 프레임의 영상 내 시각(초)."""
        pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if pos_ms > 0.0 or self._frame_id == 0:
            return pos_ms / 1000.0
        return self._frame_id / self._fps

    def frames(self) -> Iterator[Frame]:
        last = -1.0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if self._live:
                ts = time.time()
            else:
                vt = self._video_time()
                if vt <= last:                      # 재생 위치가 안 움직이는 코덱 — 프레임 번호로 (역행 금지)
                    vt = max(self._frame_id / self._fps, last + 1e-6)
                last = vt
                ts = self._t0 + vt
            yield Frame(image=image, timestamp=ts, frame_id=self._frame_id)
            self._frame_id += 1

    def close(self) -> None:
        self._cap.release()


class PiCameraSource(FrameSource):
    """Pi5 전용 — picamera2. 시스템 패키지로 설치 필요 (README.md 참고)."""

    def __init__(self, resolution: tuple[int, int]):
        try:
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "picamera2를 import할 수 없습니다. Pi5에서 "
                "'sudo apt install -y python3-picamera2'로 설치했는지 확인하세요."
            ) from exc

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        self._frame_id = 0

    def frames(self) -> Iterator[Frame]:
        while True:
            image = self._picam2.capture_array()
            yield Frame(image=image, timestamp=time.time(), frame_id=self._frame_id)
            self._frame_id += 1

    def close(self) -> None:
        self._picam2.stop()


def create_source(source: str, resolution: tuple[int, int]) -> FrameSource:
    if source == "camera":
        return PiCameraSource(resolution=resolution)
    return VideoFileSource(path=source)
