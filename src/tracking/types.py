"""추적 결과 데이터 구조. docs/pipeline-architecture.md 3.4 참고."""

from __future__ import annotations

from dataclasses import dataclass

from src.detection.types import VehicleBox


@dataclass
class TrackedVehicle:
    track_id: int
    box: VehicleBox
    tracked_frames: int   # 이 track이 관측된 누적 프레임 수
    is_interpolated: bool
    """검출 없이 추적만으로 위치가 채워진 프레임인지.

    True인 구간의 위치는 검출로 확인된 값이 아니므로, METRICS가 offset
    시계열에 반영할 때 신뢰도를 달리 볼 근거가 된다.
    """

    @property
    def center_x(self) -> float:
        return self.box.center_x

    @property
    def bottom_y(self) -> float:
        """lane offset 산출 시 차량 위치로 쓰는 하단 y좌표 (risk-criteria.md 2.1)."""
        return self.box.bottom_y
