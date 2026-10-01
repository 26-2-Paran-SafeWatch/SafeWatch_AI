"""지표 시계열 데이터 구조. docs/pipeline-architecture.md 3.5 참고.

부호 규약 — offset은 **우측이 양수**다 (risk-criteria.md 2.1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.metrics.relative_motion import rolling_expansion_rate


@dataclass
class OffsetSample:
    timestamp: float
    offset_m: float              # 차선 중심 대비 편차, 부호 있음
    offset_ratio: float          # 차로 폭 대비 비율
    lateral_velocity_mps: float  # 평활화된 offset의 시간 미분 (risk-criteria.md 2.6)
    lane_confidence: float
    is_interpolated: bool        # 검출 없이 추적만으로 채워진 프레임인지
    bbox_height: float | None = None
    """검출 박스 높이(px). "속도 불규칙" 단서(bbox 팽창률, risk-criteria.md 1.4.1)
    산출에 쓰인다.

    ⚠️ **알려진 한계** — 이 값은 lane offset과 무관한 신호라 원래는 차선 인식
    실패 구간에서도 계속 기록되어야 하지만(1.4 "독립 신호" 근거), 현재
    `OffsetCalculator`는 `lane.valid`일 때만 샘플을 추가하므로 이 필드도 그
    시점에만 채워진다. 즉 지금 구현에서는 차선 인식이 실패하면 속도 불규칙
    지표도 함께 멈춘다 — 독립성의 이점이 아직 완전히 살지 않는다. 차선과
    무관하게 항상 기록하도록 분리하는 것은 후속 작업이다."""


@dataclass
class VehicleTimeSeries:
    """차량 한 대의 최근 관측 구간 offset 이력.

    `pipeline-architecture.md` 3.5는 `direction_changes`·`max_offset_ratio`·
    `departure_duration_sec`을 필드로 적고 있으나, 모두 `offsets`에서 계산되는
    값이라 여기서는 프로퍼티로 둔다. 저장된 상태가 아니라 파생값이므로 버퍼와
    따로 관리하면 불일치가 생긴다.
    """

    track_id: int
    offsets: deque[OffsetSample] = field(default_factory=deque)

    def __len__(self) -> int:
        return len(self.offsets)

    @property
    def latest(self) -> OffsetSample | None:
        return self.offsets[-1] if self.offsets else None

    @property
    def max_offset_ratio(self) -> float:
        """관측 구간 내 최대 편차 비율 (절댓값 기준, 부호는 잃는다)."""
        if not self.offsets:
            return 0.0
        return max(abs(s.offset_ratio) for s in self.offsets)

    @property
    def peak_lateral_velocity_mps(self) -> float:
        """관측 구간 내 최대 횡방향 속도 (절댓값). 스웨빙 판정용 (risk-criteria.md 2.6)."""
        if not self.offsets:
            return 0.0
        return max(abs(s.lateral_velocity_mps) for s in self.offsets)

    def peak_expansion_rate(self, window: int) -> float | None:
        """길이 `window`인 슬라이딩 윈도우로 구한 bbox 팽창률의 최대 절댓값.

        "속도 불규칙" 단서 산출용 (risk-criteria.md 1.4.1). `bbox_height`가
        기록된 샘플이 `window`개 미만이면 판단할 수 없어 None을 반환한다
        (bbox_height의 알려진 한계는 `OffsetSample` docstring 참고).
        """
        samples = [s for s in self.offsets if s.bbox_height is not None]
        if len(samples) < window:
            return None

        timestamps = np.array([s.timestamp for s in samples])
        heights = np.array([s.bbox_height for s in samples])
        rates = rolling_expansion_rate(timestamps, heights, window)
        valid = rates[~np.isnan(rates)]
        return float(np.max(np.abs(valid))) if valid.size else None

    @property
    def observed_duration_sec(self) -> float:
        if len(self.offsets) < 2:
            return 0.0
        return self.offsets[-1].timestamp - self.offsets[0].timestamp

    def direction_changes(self, min_amplitude_ratio: float) -> int:
        """offset 부호가 전환된 횟수 (risk-criteria.md 2.2).

        부호가 바뀌어도 그 전후 진폭이 `min_amplitude_ratio`에 못 미치면 미세
        진동으로 보고 세지 않는다. 차로 중앙 근처에서 offset이 0을 오가기만 해도
        전환으로 집계되는 것을 막기 위함이다.
        """
        changes = 0
        last_sign = 0
        peak_since_change = 0.0

        for sample in self.offsets:
            ratio = sample.offset_ratio
            sign = 1 if ratio > 0 else (-1 if ratio < 0 else 0)
            if sign == 0:
                continue
            if last_sign == 0:
                last_sign = sign
                peak_since_change = abs(ratio)
                continue

            if sign != last_sign:
                # 직전 방향에서 충분한 진폭에 도달했던 경우만 전환으로 인정
                if peak_since_change >= min_amplitude_ratio:
                    changes += 1
                last_sign = sign
                peak_since_change = abs(ratio)
            else:
                peak_since_change = max(peak_since_change, abs(ratio))

        return changes

    def departure_duration_sec(self, offset_ratio_threshold: float) -> float:
        """차선 걸침이 연속으로 유지된 최대 시간 (risk-criteria.md 3.1).

        임계값을 넘는 구간이 여러 번 나타나면 그중 가장 긴 구간을 반환한다.
        """
        longest = 0.0
        start_ts: float | None = None

        for sample in self.offsets:
            if abs(sample.offset_ratio) >= offset_ratio_threshold:
                if start_ts is None:
                    start_ts = sample.timestamp
                longest = max(longest, sample.timestamp - start_ts)
            else:
                start_ts = None

        return longest
