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
        self._Minv: np.ndarray | None = None

    def _build_matrix(self, width: int, height: int) -> None:
        src = np.float32([[x * width, y * height] for x, y in self._src_ratio])
        self._M = cv2.getPerspectiveTransform(src, self._dst)
        self._Minv = cv2.getPerspectiveTransform(self._dst, src)
        self._cached_frame_size = (width, height)

    def warp(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if self._cached_frame_size != (width, height):
            self._build_matrix(width, height)
        return cv2.warpPerspective(image, self._M, self._warp_size)

    def unwarp_points(self, points: np.ndarray) -> np.ndarray:
        """bird's-eye-view 좌표(Nx2)를 원본 프레임 좌표로 역변환한다.

        시각화 등 디버깅 목적 — LaneModel의 다항식 곡선을 원본 영상 위에
        그려서 보여줄 때 사용한다. `warp()`를 한 번 이상 호출해 `_Minv`가
        계산된 뒤에만 유효하다.
        """
        if self._Minv is None:
            raise RuntimeError("warp()를 먼저 호출해야 unwarp_points()를 쓸 수 있습니다.")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._Minv).reshape(-1, 2)
