"""검출 결과 데이터 구조. docs/pipeline-architecture.md 3.2 참고."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VehicleBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int  # COCO 클래스 id (car=2, motorcycle=3, bus=5, truck=7)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def bottom_y(self) -> float:
        """차량 하단 y좌표. lane offset 산출 시 차량 위치로 사용 (risk-criteria.md 2.1)."""
        return self.y2
