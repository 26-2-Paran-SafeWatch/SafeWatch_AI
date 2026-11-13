"""Bird's eye view perspective 변환. docs/pipeline-architecture.md 3.3 참고.

⚠️ `configs/default.yaml`의 `lane.perspective.src_points_ratio`는 AI Hub
`60_전방` 카메라의 정답 라벨로 보정한 값이다(`notebook/safewatch_lane_calib.ipynb`).
우리 카메라 값이 아니므로 장착이 확정되면 재보정이 필요하다
(pipeline-architecture.md 5장 미결 사항).

사다리꼴은 출력 폭 양옆 `margin_ratio`만큼 안쪽으로 매핑한다. 자차 차로를
조감도 전체 폭에 꽉 채우면 옆 차로와 갓길이 잘려 나가, 차선 검출이 바깥
구조물과 진짜 차선을 구분할 수 없고 옆 차로 차량의 offset도 잴 수 없다.
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
        m = p_cfg.margin_ratio * w
        self._dst = np.float32([[m, 0], [w - m, 0], [w - m, h], [m, h]])
        self._expected_lane_width_px = float(w - 2 * m)

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

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        """단일 채널 마스크(원본 프레임 크기)를 조감도로 옮긴다. 값이 섞이지 않게 최근접 보간.

        `warp()`와 같은 프레임 크기여야 한다 — 행렬을 그 크기로 만든다.
        """
        height, width = mask.shape[:2]
        if self._cached_frame_size != (width, height):
            self._build_matrix(width, height)
        return cv2.warpPerspective(mask, self._M, self._warp_size, flags=cv2.INTER_NEAREST)

    def warp_points(self, points: np.ndarray) -> np.ndarray:
        """원본 프레임 좌표(Nx2)를 bird's-eye-view 좌표로 변환한다.

        차량 위치(검출 박스 하단 중심)를 차선 다항식과 같은 좌표계로 옮길 때
        쓴다 — 차선 모델은 BEV 좌표계에 있고 검출 박스는 원본 프레임
        좌표계에 있다 (METRICS, pipeline-architecture.md 3.5).
        """
        if self._M is None:
            raise RuntimeError("warp()를 먼저 호출해야 warp_points()를 쓸 수 있습니다.")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._M).reshape(-1, 2)

    @property
    def warp_size(self) -> tuple[int, int]:
        """(width, height) — BEV 좌표 범위 검사에 사용한다."""
        return self._warp_size

    @property
    def expected_lane_width_px(self) -> float:
        """보정이 맞을 때 자차 차로가 조감도에서 가질 폭(px). 신뢰도의 폭 타당성 기준."""
        return self._expected_lane_width_px

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
