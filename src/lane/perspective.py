"""Bird's eye view perspective 변환. docs/pipeline-architecture.md 3.3 참고.

⚠️ `configs/default.yaml`의 `lane.perspective.src_points_ratio`는 카메라
장착 위치·화각에 의존하는 placeholder 값이다. 실제 카메라가 확정되면
재캘리브레이션이 필요하다 (pipeline-architecture.md 5장 미결 사항).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.config import Config


class PerspectiveTransformer:
    def __init__(self, cfg: Config):
        p_cfg = cfg.lane.perspective
        self._warp_size = tuple(p_cfg.warp_size)  # (width, height)
        self._src_ratio = p_cfg.src_points_ratio

        w, h = self._warp_size
        self._dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        self._cached_frame_size: tuple[int, int] | None = None
        self._M: np.ndarray | None = None

    def _build_matrix(self, width: int, height: int) -> None:
        src = np.float32([[x * width, y * height] for x, y in self._src_ratio])
        self._M = cv2.getPerspectiveTransform(src, self._dst)
        self._cached_frame_size = (width, height)

    def warp(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if self._cached_frame_size != (width, height):
            self._build_matrix(width, height)
        return cv2.warpPerspective(image, self._M, self._warp_size)
