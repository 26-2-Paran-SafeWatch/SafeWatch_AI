"""이벤트 메타데이터 생성 테스트. docs/event-schema.md 참고."""

from __future__ import annotations

import re

import pytest

from src.detection.types import VehicleBox
from src.event.builder import EventIdGenerator, build_event_metadata
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
        series=_series(),
        track=_track(),
        cfg=cfg,
        model_version="yolov8n-safewatch-v0.1",
        rule_version="risk-v0.1-dui-scope",
    )
    for key in ("event_id", "device_id", "timestamp", "risk", "indicators", "target_vehicle", "meta"):
        assert key in event


def test_location_and_clip_omitted_when_not_provided(cfg):
    """GPS·링버퍼 미연결 — 스키마의 'GPS 미수신 시 객체 생략' 방식을 따른다."""
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        series=_series(),
        track=_track(),
        cfg=cfg,
        model_version="v1",
        rule_version="v1",
    )
    assert "location" not in event
    assert "clip" not in event


def test_location_and_clip_included_when_provided(cfg):
    location = {"latitude": 37.32, "longitude": 126.83, "speed_kmh": 62.5, "heading": 271.3}
    clip = {"filename": "evt_x.mp4", "duration_sec": 15, "blurred": True}
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        series=_series(),
        track=_track(),
        cfg=cfg,
        model_version="v1",
        rule_version="v1",
        location=location,
        clip=clip,
    )
    assert event["location"] == location
    assert event["clip"] == clip


def test_indicators_include_observation_window_and_assessment_values(cfg):
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        series=_series(),
        track=_track(),
        cfg=cfg,
        model_version="v1",
        rule_version="v1",
    )
    assert event["indicators"]["observation_window_sec"] == cfg.risk.observation_window_sec
    assert event["indicators"]["lane_offset_max_ratio"] == 0.34
    assert event["indicators"]["expansion_rate_peak_per_sec"] == 0.31


def test_risk_and_target_vehicle_fields(cfg):
    event = build_event_metadata(
        event_id="evt_20260927_120000_001",
        device_id="safewatch-001",
        timestamp=1_700_000_000.0,
        assessment=_assessment(),
        series=_series(),
        track=_track(),
        cfg=cfg,
        model_version="v1",
        rule_version="v1",
    )
    assert event["risk"] == {
        "score": 60,
        "level": "medium",
        "types": ["lane_departure", "speed_irregular"],
    }
    assert event["target_vehicle"]["track_id"] == 17
    assert event["target_vehicle"]["avg_detection_confidence"] == pytest.approx(0.87)
