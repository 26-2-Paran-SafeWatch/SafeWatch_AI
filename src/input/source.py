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
    """PC 개발용 — 영상 파일 또는 웹캠 인덱스."""

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"영상 소스를 열 수 없습니다: {path}")
        self._frame_id = 0

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(image=image, timestamp=time.time(), frame_id=self._frame_id)
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
