"""EventQueue(로컬 Store-and-Forward) 테스트. docs/pipeline-architecture.md 3.7 참고."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.event.queue import EventQueue


def _event(event_id: str = "evt_20261001_120000_001") -> dict:
    return {"event_id": event_id, "risk": {"score": 80, "level": "high", "types": ["weaving"]}}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "events.db"


def test_enqueued_event_is_pending_for_both_kinds(db_path: Path):
    queue = EventQueue(db_path)
    queue.enqueue(_event())
    assert len(queue.pending("metadata")) == 1
    assert queue.pending("clip") == []  # 클립 경로가 없으면 clip 큐에는 안 잡힘
    queue.close()


def test_mark_sent_removes_from_pending(db_path: Path):
    queue = EventQueue(db_path)
    queue.enqueue(_event())
    queue.mark_sent("evt_20261001_120000_001", "metadata")
    assert queue.pending("metadata") == []
    queue.close()


def test_clip_tracked_separately_from_metadata(db_path: Path):
    queue = EventQueue(db_path)
    queue.enqueue(_event(), clip_path="/tmp/evt.mp4")
    assert len(queue.pending("clip")) == 1

    queue.mark_sent("evt_20261001_120000_001", "clip")
    assert queue.pending("clip") == []
    assert len(queue.pending("metadata")) == 1  # 메타는 여전히 미전송
    queue.close()


def test_attach_clip_later_marks_clip_unsent(db_path: Path):
    """C안(메타 LTE 즉시 / 클립 WiFi 지연) — 클립이 메타보다 한참 뒤에 도착하는 경우."""
    queue = EventQueue(db_path)
    queue.enqueue(_event())
    queue.mark_sent("evt_20261001_120000_001", "metadata")
    assert queue.pending("clip") == []  # 클립 경로가 아직 없어 대상 아님

    queue.attach_clip("evt_20261001_120000_001", "/tmp/evt.mp4")
    assert len(queue.pending("clip")) == 1
    queue.close()


def test_duplicate_enqueue_does_not_duplicate_record(db_path: Path):
    queue = EventQueue(db_path)
    queue.enqueue(_event())
    queue.enqueue(_event())  # 재전송 시나리오 — 같은 event_id
    assert len(queue.pending("metadata")) == 1
    queue.close()


def test_purge_fully_sent_removes_only_fully_sent(db_path: Path):
    queue = EventQueue(db_path)
    queue.enqueue(_event("evt_a"), clip_path="/tmp/a.mp4")
    queue.enqueue(_event("evt_b"), clip_path="/tmp/b.mp4")

    queue.mark_sent("evt_a", "metadata")
    queue.mark_sent("evt_a", "clip")
    queue.mark_sent("evt_b", "metadata")  # evt_b는 클립 미전송 상태로 남김

    removed = queue.purge_fully_sent()
    assert removed == 1
    assert queue.pending("clip") == [
        {"event_id": "evt_b", "metadata": _event("evt_b"), "clip_path": "/tmp/b.mp4"}
    ]
    queue.close()


def test_queue_survives_reopen(db_path: Path):
    """디바이스 재부팅 시나리오 — 같은 db_path로 다시 열어도 미전송 레코드가 남아 있어야 한다."""
    queue = EventQueue(db_path)
    queue.enqueue(_event())
    queue.close()

    reopened = EventQueue(db_path)
    assert len(reopened.pending("metadata")) == 1
    reopened.close()
