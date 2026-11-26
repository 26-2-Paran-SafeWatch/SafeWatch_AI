"""프레임 데이터 구조. docs/pipeline-architecture.md 3.1 참고."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR, (H, W, 3)
    timestamp: float  # Unix epoch, 초 단위 소수점
    frame_id: int  # 순차 증가. 드롭되어도 건너뛰지 않고 계속 증가시켜 누락 추적 가능하게 한다.
    media_sec: float | None = None
    """영상 파일 입력일 때 그 프레임의 영상 안 시각(초). 카메라·웹캠은 None.
    이벤트 envelope의 trigger.video_sec로 넘어가 HW가 파일에서 클립을 자르는 기준이 된다
    (docs/event-schema.md "AI → HW 전달 형식")."""
