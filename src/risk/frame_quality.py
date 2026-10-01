"""프레임 품질 게이팅 — 블러·밝기 급변 검사. docs/risk-criteria.md 4.1 참고.

RISK의 입력 신뢰도 게이팅 항목 중 "프레임 품질"에 해당한다. 검출
confidence·차선 신뢰도·관측 충분성·자차 속도 게이팅은 이미 `RiskScorer`에
구현되어 있었으나 이 항목만 비어 있었다 (`scorer.py` 모듈 docstring
"구현되지 않은 게이팅" 참고). 데이터셋 없이도 완성 가능한 일반적인
컴퓨터 비전 기법이라 지금 채운다.

**블러 — Laplacian 분산**
흐릴수록 고주파 성분(에지)이 줄어 Laplacian 분산이 작아진다는, 무참조
(reference-free) 블러 측정에 널리 쓰이는 휴리스틱이다. 장면·해상도·렌즈에
따라 절대값이 크게 달라져 보편적으로 맞는 수치는 없고, 실무에서도
"대략 100 전후"를 출발점으로 삼아 촬영 환경에 맞춰 재조정하는 것이
관례다. `lane.confidence.max_residual_ratio`가 이론값에서 출발해 실측으로
보정된 것과 같은 패턴을 따른다.

**밝기 급변 — 평균 밝기 변화량**
터널 진출입처럼 급격한 조도 변화 구간에서는 노출이 따라가지 못해 검출·
차선 인식 품질이 함께 떨어진다. 직전 프레임 대비 그레이스케일 평균
밝기의 변화량으로 급변 여부를 본다 — risk-criteria.md 4.1이 제안한
"히스토그램 변화율"의 단순화된 근사치다. 완전한 히스토그램 비교(예:
Bhattacharyya distance)는 계산 비용 대비 이득이 불분명해, 평균값 비교로
시작하고 필요해지면 교체한다.

⚠️ 아래 기본 임계값은 모두 **초안**이며 실차 영상 확보 후 재튜닝 대상이다
(sprint-plan.md S3 "신뢰도 지표 임계값 튜닝").
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.config import Config
from src.utils.profiler import profiler


class FrameQualityChecker:
    """프레임 단위 상태를 유지한다 — 밝기 급변 검사가 직전 프레임과의 비교이기 때문이다.

    차량별이 아니라 프레임 전체에 대한 판단이므로, 호출부(main.py)는
    프레임마다 한 번만 호출해 그 결과를 해당 프레임의 모든 차량 판정에
    동일하게 적용한다.
    """

    def __init__(self, cfg: Config):
        quality_cfg = cfg.risk.frame_quality
        self._min_blur_variance: float = quality_cfg.min_blur_variance
        self._max_brightness_jump: float = quality_cfg.max_brightness_jump
        self._prev_brightness: float | None = None

    def check(self, image: np.ndarray) -> bool:
        """이번 프레임이 판단에 쓸 만큼 품질이 충분하면 True."""
        with profiler.stage("frame_quality"):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_variance < self._min_blur_variance:
                return False

            brightness = float(gray.mean())
            brightness_jumped = (
                self._prev_brightness is not None
                and abs(brightness - self._prev_brightness) > self._max_brightness_jump
            )
            self._prev_brightness = brightness
            return not brightness_jumped
