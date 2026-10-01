"""로컬 이벤트 큐 — Store-and-Forward. docs/pipeline-architecture.md 3.7 참고.

통신 단절에도 이벤트가 유실되지 않도록 이벤트 발생 즉시 로컬(SQLite)에
저장한다. **이 모듈이 하는 일은 저장·조회·전송 상태 갱신까지다 — 실제
네트워크 전송(서버 업로드)은 포함하지 않는다.** 업로드 큐의 "구현 주체"는
`system-architecture.md`가 여전히 미정으로 적어 두었지만, 로컬 저장소
자체는 이 저장소(AI 파트)의 EVENT 모듈 책임 범위이고(pipeline-architecture.md
3.7 "Store-and-Forward") 외부 팀 결정과 무관하게 데이터셋 없이 지금
완성할 수 있어 먼저 구현한다. 실제 전송 클라이언트가 정해지면
`pending(kind=...)`로 미전송 레코드를 꺼내 보내고 `mark_sent()`로
표시하면 된다.

메타데이터(JSON)와 영상 클립은 전송 경로가 다르므로(LTE 즉시 / WiFi
연결 시, event-schema.md 1장 C안) 전송 상태를 **따로** 추적한다.

**재전송 중복** — `event_id`가 PRIMARY KEY라 같은 event_id로 `enqueue()`를
다시 호출해도 레코드가 늘지 않는다(UPSERT). 서버 쪽 중복 제거
(event-schema.md 3.1)와는 별개로, 로컬 큐 자체에서도 중복 적재를 막는다.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    clip_path TEXT,
    metadata_sent INTEGER NOT NULL DEFAULT 0,
    clip_sent INTEGER NOT NULL DEFAULT 0
);
"""

Kind = Literal["metadata", "clip"]
_SENT_COLUMN: dict[Kind, str] = {"metadata": "metadata_sent", "clip": "clip_sent"}


class EventQueue:
    """SQLite 기반 로컬 이벤트 큐. 디바이스 재부팅 후에도 같은 `db_path`로
    다시 열면 미전송 레코드가 그대로 남아 있다(파일 기반 영속성)."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def enqueue(self, event: dict[str, Any], clip_path: str | None = None) -> None:
        """이벤트를 큐에 적재한다. `event["event_id"]`가 PRIMARY KEY다."""
        self._conn.execute(
            """
            INSERT INTO events (event_id, created_at, metadata_json, clip_path)
            VALUES (:event_id, :created_at, :metadata_json, :clip_path)
            ON CONFLICT(event_id) DO UPDATE SET
                metadata_json = excluded.metadata_json,
                clip_path = excluded.clip_path
            """,
            {
                "event_id": event["event_id"],
                "created_at": time.time(),
                "metadata_json": json.dumps(event, ensure_ascii=False),
                "clip_path": clip_path,
            },
        )
        self._conn.commit()

    def attach_clip(self, event_id: str, clip_path: str) -> None:
        """메타데이터 전송 뒤 클립이 나중에 준비되는 경우(C안) 경로를 갱신한다.

        클립 전송 상태는 다시 미전송으로 되돌린다 — 새 클립이 아직 안
        나갔기 때문이다.
        """
        self._conn.execute(
            "UPDATE events SET clip_path = ?, clip_sent = 0 WHERE event_id = ?",
            (clip_path, event_id),
        )
        self._conn.commit()

    def pending(self, kind: Kind) -> list[dict[str, Any]]:
        """미전송 레코드를 생성 순서대로 꺼낸다.

        `kind="clip"`은 클립 경로가 아직 없는(링 버퍼 미연동 등) 레코드는
        제외한다 — 보낼 파일 자체가 없기 때문이다.
        """
        column = _SENT_COLUMN[kind]
        clip_filter = "AND clip_path IS NOT NULL" if kind == "clip" else ""
        rows = self._conn.execute(
            f"SELECT event_id, metadata_json, clip_path FROM events "
            f"WHERE {column} = 0 {clip_filter} ORDER BY created_at"
        ).fetchall()
        return [
            {"event_id": r[0], "metadata": json.loads(r[1]), "clip_path": r[2]}
            for r in rows
        ]

    def mark_sent(self, event_id: str, kind: Kind) -> None:
        column = _SENT_COLUMN[kind]
        self._conn.execute(f"UPDATE events SET {column} = 1 WHERE event_id = ?", (event_id,))
        self._conn.commit()

    def purge_fully_sent(self) -> int:
        """메타·클립 모두 전송 완료된 레코드를 삭제하고 삭제 건수를 반환한다.

        ⚠️ 클립이 아예 요청되지 않은(`clip_path IS NULL`) 이벤트는 여기서
        지우지 않는다 — 링 버퍼가 아직 연결되지 않아 클립이 "영영 안 옴"과
        "아직 안 옴"을 이 모듈만으로는 구분할 수 없기 때문이다. 링 버퍼
        연동(HW 인터페이스 확정) 이후 재검토 대상 (`sprint-plan.md` S4).
        """
        cur = self._conn.execute(
            "DELETE FROM events WHERE metadata_sent = 1 AND clip_sent = 1"
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()
