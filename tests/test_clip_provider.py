"""ClipProvider 인터페이스 테스트. docs/pipeline-architecture.md 3.7 참고."""

from __future__ import annotations

from src.event.clip import ClipResult, NullClipProvider


def test_null_clip_provider_always_returns_none():
    provider = NullClipProvider()
    assert provider.request_clip(event_timestamp=100.0, pre_sec=10, post_sec=5) is None


def test_clip_result_to_schema_dict_matches_event_schema_fields():
    """event-schema.md 3.6 clip 필드와 키 집합이 정확히 일치해야 한다."""
    result = ClipResult(
        filename="evt_20261001_120000_001.mp4",
        local_path="/data/processed/events/evt_20261001_120000_001.mp4",
        duration_sec=15,
        pre_event_sec=10,
        post_event_sec=5,
        resolution="1280x720",
        fps=15,
        size_bytes=3821004,
        blurred=True,
    )
    schema_dict = result.to_schema_dict()

    assert set(schema_dict.keys()) == {
        "filename", "duration_sec", "pre_event_sec", "post_event_sec",
        "resolution", "fps", "size_bytes", "blurred",
    }
    assert "local_path" not in schema_dict  # 서버로 나가는 필드가 아니다
    assert schema_dict["filename"] == "evt_20261001_120000_001.mp4"
    assert schema_dict["duration_sec"] == 15
