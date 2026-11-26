"""이벤트 메타데이터 생성 테스트. docs/event-schema.md 참고."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import pytest

from src.detection.types import VehicleBox
from src.event.builder import EventIdGenerator, build_event_envelope, build_event_metadata
from src.input.frame import Frame
from src.metrics.types import VehicleTimeSeries
from src.risk.types import RiskAssessment
from src.tracking.types import TrackedVehicle
from src.utils.config import load_config

EVENT_ID_PATTERN = re.compile(r"^evt_\d{8}_\d{6}_\d{3}$")


@pytest.fixture
def cfg():
    return load_config("configs/dev.yaml")


def _assessment() -> RiskAssessment:
    return RiskAssessment(
        track_id=17,
        score=60,
        level="medium",
        types=["lane_departure", "speed_irregular"],
        indicators={
            "lane_offset_max_ratio": 0.34,
            "lane_departure_duration_sec": 2.8,
            "expansion_rate_peak_per_sec": 0.31,
        },
        should_emit_event=True,
    )


def _series() -> VehicleTimeSeries:
    series = VehicleTimeSeries(track_id=17)
    series.offsets.extend([])  # observed_duration_sec은 샘플이 있어야 의미 있으나 여기선 구조만 확인
    return series


def _track() -> TrackedVehicle:
    return TrackedVehicle(
        track_id=17,
        box=VehicleBox(100, 100, 180, 220, confidence=0.87, class_id=2),
        tracked_frames=200,
        is_interpolated=False,
    )


class TestEventIdGenerator:
    def test_format_matches_schema(self):
        gen = EventIdGenerator("safewatch-001")
        event_id = gen.next(1_700_000_000.0)
        assert EVENT_ID_PATTERN.match(event_id)

    def test_sequence_increments_within_same_second(self):
        gen = EventIdGenerator("safewatch-001")
        first = gen.next(1_700_000_000.0)
        second = gen.next(1_700_000_000.4)  # 같은 초(정수부 기준)
        assert first.rsplit("_", 1)[0] == second.rsplit("_", 1)[0]
        assert first.endswith("_001")
        assert second.endswith("_002")

    def test_sequence_resets_on_new_second(self):
        gen = EventIdGenerator("safewatch-001")
        gen.next(1_700_000_000.0)
        later = gen.next(1_700_000_001.0)
        assert later.endswith("_001")


def test_build_event_metadata_has_required_top_level_fields(cfg):
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        cfg=cfg,
        model_version="yolov8n-safewatch-v0.1",
        rule_version="risk-v0.1-dui-scope",
    )
    for key in ("event_id", "device_id", "timestamp", "risk", "indicators", "meta"):
        assert key in event


def test_event_has_no_clip_location_or_target_vehicle(cfg):
    """clip·location은 HW가 붙이고, target_vehicle은 envelope의 target으로 옮겼다 (2026-11-15)."""
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        cfg=cfg,
        model_version="v1",
        rule_version="v1",
    )
    for key in ("location", "clip", "target_vehicle"):
        assert key not in event


def test_indicators_include_observation_window_and_assessment_values(cfg):
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        cfg=cfg,
        model_version="v1",
        rule_version="v1",
    )
    assert event["indicators"]["observation_window_sec"] == cfg.risk.observation_window_sec
    assert event["indicators"]["lane_offset_max_ratio"] == 0.34
    assert event["indicators"]["expansion_rate_peak_per_sec"] == 0.31


def test_risk_fields(cfg):
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        cfg=cfg,
        model_version="v1",
        rule_version="v1",
    )
    assert event["risk"] == {
        "score": 60,
        "level": "medium",
        "types": ["lane_departure", "speed_irregular"],
    }


def _event(cfg):
    return build_event_metadata(
        event_id="evt_20260927_120000_001", device_id="safewatch-001", timestamp=1_700_000_000.0,
        assessment=_assessment(), cfg=cfg, model_version="v1", rule_version="v1",
    )


def _frame(media_sec=None):
    return Frame(image=np.zeros((720, 1280, 3), np.uint8), timestamp=1_700_000_000.25, frame_id=1234,
                 media_sec=media_sec)


def test_envelope_camera(cfg):
    event = _event(cfg)
    env = build_event_envelope(event, _frame(), _track(), _series(), "camera")
    assert set(env) == {"event", "trigger", "target"}
    assert env["event"] is event
    for key in ("clip", "location", "target_vehicle"):
        assert key not in env["event"]
    assert env["trigger"] == {"epoch": 1_700_000_000.25, "frame_id": 1234, "source": "camera", "video_sec": None}
    t = env["target"]
    assert t["track_id"] == 17 and t["box"] == [100, 100, 180, 220] and t["frame_size"] == [1280, 720]
    assert t["avg_detection_confidence"] == pytest.approx(0.87)
    assert "tracked_duration_sec" in t


def test_envelope_video_file_uses_absolute_path_and_media_sec(cfg):
    env = build_event_envelope(_event(cfg), _frame(media_sec=62.4), _track(), _series(), "data/raw/x.mp4")
    assert env["trigger"]["source"] == str(Path("data/raw/x.mp4").resolve())
    assert Path(env["trigger"]["source"]).is_absolute()
    assert env["trigger"]["video_sec"] == pytest.approx(62.4)
