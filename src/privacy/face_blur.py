"""얼굴 비식별화 — Haar cascade 기반. docs/pipeline-architecture.md 3.8,
5장 "얼굴 블러 검출 방식" 미결 사항 참고.

**방식 선택 근거** — AGENTS.md는 YOLO 외 딥러닝 모델 추가를 금지한다
("사용하지 않는 것" 표). pipeline-architecture.md 5장은 이 제약과 충돌하지
않는 두 선택지로 "YOLO 클래스 확장"과 "Haar cascade(non-DL)"를 들었다.
YOLO 클래스 확장은 얼굴 라벨이 있는 파인튜닝 데이터가 있어야 하지만(S5,
AI Hub 데이터 확보 이후), Haar cascade는 OpenCV에 사전 포함된 비-딥러닝
분류기라 **데이터셋 없이 지금 완성할 수 있다.** 번호판 블러(YOLO 클래스
확장 전제, `pipeline-architecture.md` 3.8)와 달리 이 모듈은 데이터 확보를
기다리지 않는다.

⚠️ Haar cascade는 YOLO보다 부정확하다 — 정면·충분한 크기의 얼굴에서만
안정적으로 동작하고 측면·저조도·작은 얼굴은 놓치기 쉬우며 오탐도 있다.
"완벽한 비식별화"가 아니라 **번호판 YOLO 전환 전까지의 1차 방어선**으로
본다. 실차 영상으로 검출률을 실측하기 전까지는 보수적인 기본 파라미터
(`detectMultiScale`의 `scaleFactor`/`minNeighbors`)를 둔다.

**호출 시점** — pipeline-architecture.md 3.8대로 상시 프레임이 아니라
**이벤트 발생 시 클립에만** 적용한다. 링 버퍼(HW 파트)가 아직 연결되지
않아(`src/event/clip.py`) 실제로 호출할 클립 프레임이 없으므로, 이
모듈은 클립 파이프라인이 연결되면 바로 쓸 수 있는 독립 컴포넌트로 먼저
완성해 둔다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.utils.config import Config

logger = logging.getLogger("safewatch.privacy")

BoundingBox = tuple[int, int, int, int]  # (x, y, w, h), px


class FaceBlurrer:
    def __init__(self, cfg: Config):
        privacy_cfg = cfg.privacy
        self._enabled: bool = privacy_cfg.blur_faces
        self._kernel: int = privacy_cfg.blur_kernel
        if self._kernel % 2 == 0:
            raise ValueError("privacy.blur_kernel은 홀수여야 한다 (cv2.GaussianBlur 요구사항)")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Haar cascade 로드 실패: {cascade_path}")

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """얼굴 영역을 (x, y, w, h) 목록으로 반환한다."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def blur_boxes(self, image: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
        """주어진 영역에 가우시안 블러를 적용한 복사본을 반환한다.

        `detect()`와 분리해 둔 이유 — Haar cascade는 실제 얼굴 패턴에만
        반응해 합성 이미지로는 검출 자체를 테스트하기 어렵다. 블러 적용
        로직은 박스 좌표만 있으면 되므로 별도로 단위 테스트할 수 있다.
        """
        result = image.copy()
        for x, y, w, h in boxes:
            roi = result[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            result[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (self._kernel, self._kernel), 0)
        return result

    def anonymize(self, image: np.ndarray) -> np.ndarray:
        """얼굴을 검출해 블러까지 적용한다. `privacy.blur_faces=false`면 원본을 그대로 반환."""
        if not self._enabled:
            return image
        return self.blur_boxes(image, self.detect(image))
