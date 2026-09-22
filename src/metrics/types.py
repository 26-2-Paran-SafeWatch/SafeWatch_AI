"""지표 시계열 데이터 구조. docs/pipeline-architecture.md 3.5 참고.

부호 규약 — offset은 **우측이 양수**다 (risk-criteria.md 2.1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class OffsetSample:
    timestamp: float
    offset_m: float              # 차선 중심 대비 편차, 부호 있음
    offset_ratio: float          # 차로 폭 대비 비율
    lateral_velocity_mps: float  # 평활화된 offset의 시간 미분 (risk-criteria.md 2.6)
    lane_confidence: float
    is_interpolated: bool        # 검출 없이 추적만으로 채워진 프레임인지


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
