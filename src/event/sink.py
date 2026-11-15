"""이벤트를 HW 파트에 넘기는 출구 (docs/event-schema.md "AI → HW 전달 형식(envelope)").

AI는 이벤트 envelope를 만들어 `EventSink.publish()`에 넘기는 데까지만 한다. 클립 자르기·
location 채우기·큐·재시도·서버 전송은 HW 파트(SafeWatch_HW)가 자기 sink 안에서 한다.
실기기에서는 HW 실행 스크립트가 `src.main.run(..., event_sink=<HW sink>)`로 넘긴다.

`FileEventSink`는 기본 구현이다 — PC 개발·테스트와, HW sink 없이 돌릴 때 쓴다.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("safewatch.event.sink")


class EventSink(Protocol):
    def publish(self, envelope: dict[str, Any]) -> None:
        """envelope 하나를 넘긴다. 바로 돌아와야 한다 — 파이프라인(프레임 처리)을 막지 않게."""
        ...

    def close(self) -> None:
        """종료 때 한 번 부른다 (src/main.py finally)."""
        ...


class FileEventSink:
    """envelope를 `<dir>/<event_id>.json`으로 저장한다. PC 개발·테스트용.

    같은 폴더의 임시 파일에 쓴 뒤 `os.replace`로 이름을 바꾼다 — 읽는 쪽이 반쯤 쓰인 파일을
    보지 않게. 같은 event_id면 덮어쓴다. 실패해도 예외를 밖으로 내지 않고 로그만 남긴다
    (파이프라인이 멈추면 안 된다).
    """

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def publish(self, envelope: dict[str, Any]) -> None:
        tmp = None
        try:
            event_id = envelope["event"]["event_id"]
            fd, tmp = tempfile.mkstemp(prefix=f".{event_id}.", suffix=".tmp", dir=self._dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(envelope, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._dir / f"{event_id}.json")
            tmp = None
        except Exception:  # noqa: BLE001 — 이벤트 하나 저장 실패로 파이프라인을 멈추지 않는다
            logger.exception("이벤트 envelope 저장 실패")
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def close(self) -> None:
        """파일마다 바로 닫으므로 할 일 없음."""
