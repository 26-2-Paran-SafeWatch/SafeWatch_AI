"""클립 요청 인터페이스. docs/pipeline-architecture.md 3.7,
docs/system-architecture.md "아직 남은 것" 참고.

링 버퍼는 HW 파트(강섬희)가 구현하며, 버퍼 크기도 HW 파트가 실측해서
정한다(확정 사항). 이 저장소(AI 파트)의 EVENT 모듈은 "이벤트 시점 + 전후
구간을 넘겨 클립을 요청하고 반환받는" 쪽이다. **요청 인터페이스의 구체
형태(함수 호출 / IPC / 파일)는 HW 파트와 아직 협의되지 않았다** — 그래서
전송 방식이 아니라 **이 모듈이 기대하는 타입 계약**만 지금 정의해 둔다.
HW 쪽 인터페이스가 확정되면 이 `ClipProvider`를 구현하는 클래스 하나만
새로 추가하면 되고, `main.py`/`build_event_metadata` 쪽은 바꿀 필요가 없다.

지금은 `NullClipProvider`(항상 클립 없음)로 동작한다 — 링 버퍼가 아직
연결되지 않았으므로 없는 클립을 지어내지 않고 정직하게 비운다
(`event/builder.py`의 `clip=None` 처리와 같은 원칙).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ClipResult:
    """event-schema.md 3.6 `clip` 필드와 1:1 대응한다. `local_path`만 로컬
    전송 큐(`src/event/queue.py`)에 넘기기 위해 스키마 밖에 추가로 둔다."""

    filename: str
    local_path: str
    duration_sec: int
    pre_event_sec: int
    post_event_sec: int
    resolution: str
    fps: int
    size_bytes: int
    blurred: bool

    def to_schema_dict(self) -> dict[str, object]:
        """`event-schema.md` 3.6의 `clip` 객체 그대로. `local_path`는 제외한다
        (서버로 나가는 필드가 아니라 로컬 전송 큐 내부 경로이기 때문)."""
        return {
            "filename": self.filename,
            "duration_sec": self.duration_sec,
            "pre_event_sec": self.pre_event_sec,
            "post_event_sec": self.post_event_sec,
            "resolution": self.resolution,
            "fps": self.fps,
            "size_bytes": self.size_bytes,
            "blurred": self.blurred,
        }


class ClipProvider(Protocol):
    """HW 링 버퍼 요청 인터페이스의 AI 파트 쪽 계약.

    구현체는 이벤트 시점 전후 `pre_sec`+`post_sec` 구간의 영상을 확보해
    `ClipResult`로 반환하거나, 확보할 수 없으면 None을 반환한다(예: 버퍼
    용량 밖 구간, 링 버퍼 미연결).
    """

    def request_clip(
        self, event_timestamp: float, pre_sec: int, post_sec: int
    ) -> ClipResult | None: ...


class NullClipProvider:
    """링 버퍼가 아직 연결되지 않은 현재 상태의 기본 구현. 항상 None을 반환한다."""

    def request_clip(
        self, event_timestamp: float, pre_sec: int, post_sec: int
    ) -> ClipResult | None:
        return None
