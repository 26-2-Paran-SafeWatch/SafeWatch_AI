"""AI → HW 이벤트 전달 — FileEventSink와, 주입한 frame_source·event_sink로 run()이 끝까지 도는지.

판단 쪽(검출·차선·추적·offset·위험)은 가짜로 바꿔 '세 번째 프레임에서 이벤트 1건'을 만든다.
판단 로직 자체는 test_risk_scorer.py 등이 따로 본다."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

import src.main as main_mod
from src.detection.types import VehicleBox
from src.event.sink import FileEventSink
from src.input.frame import Frame
from src.lane.types import LaneModel
from src.metrics.types import OffsetSample, VehicleTimeSeries
from src.risk.types import RiskAssessment
from src.tracking.types import TrackedVehicle


def _env(event_id: str, score: int = 60) -> dict:
    return {"event": {"event_id": event_id, "risk": {"score": score}}, "trigger": {}, "target": {}}


# ── FileEventSink ───────────────────────────────────────────────────────────

def test_file_sink_writes_event_id_json_without_temp_files(tmp_path):
    sink = FileEventSink(tmp_path / "outbox")
    sink.publish(_env("evt_20261115_143022_001"))
    sink.close()
    files = list((tmp_path / "outbox").iterdir())
    assert [f.name for f in files] == ["evt_20261115_143022_001.json"]
    assert json.loads(files[0].read_text(encoding="utf-8"))["event"]["event_id"] == "evt_20261115_143022_001"


def test_file_sink_overwrites_same_event_id(tmp_path):
    sink = FileEventSink(tmp_path)
    sink.publish(_env("evt_x", 60))
    sink.publish(_env("evt_x", 80))
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["event"]["risk"]["score"] == 80


def test_file_sink_failure_does_not_raise_or_leave_temp(tmp_path, caplog):
    sink = FileEventSink(tmp_path)
    with caplog.at_level(logging.ERROR):
        sink.publish({"event": {"event_id": "evt_bad"}, "x": object()})   # JSON으로 못 바꾸는 값
    assert list(tmp_path.iterdir()) == []
    assert "저장 실패" in caplog.text


# ── run() 주입 ──────────────────────────────────────────────────────────────

BOX = VehicleBox(252, 164, 348, 238, confidence=0.81, class_id=2)
TRACK = TrackedVehicle(track_id=3, box=BOX, tracked_frames=50, is_interpolated=False)
EMIT_AT = 2


class FakeFrames:
    def __init__(self, n: int = 5):
        self.n, self.closed = n, False

    def frames(self):
        for i in range(self.n):
            yield Frame(image=np.zeros((720, 1280, 3), np.uint8), timestamp=1_794_725_420.0 + i * 0.5, frame_id=i)

    def close(self):
        self.closed = True


class ListSink:
    def __init__(self, fail: bool = False):
        self.got, self.closed, self.fail = [], False, fail

    def publish(self, envelope):
        if self.fail:
            raise RuntimeError("HW sink 고장")
        self.got.append(envelope)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_pipeline(monkeypatch):
    lane = LaneModel(left_fit=None, right_fit=None, lane_width_px=500, curvature_radius=None,
                     confidence=0.9, valid=True, space="image")

    class Scorer:
        def __init__(self, cfg): ...
        def sync(self, ids): ...

        def assess(self, track, ts, *, timestamp, **kw):
            fid = round((timestamp - 1_794_725_420.0) / 0.5)
            return RiskAssessment(track_id=track.track_id, score=60, level="medium", types=["weaving", "swerving"],
                                  indicators={"direction_changes_count": 3}, should_emit_event=fid == EMIT_AT)

    monkeypatch.setattr(main_mod, "VehicleDetector", lambda cfg: SimpleNamespace(detect=lambda img: [BOX]))
    monkeypatch.setattr(main_mod, "create_lane_detector", lambda cfg: SimpleNamespace(
        detect=lambda img, **kw: lane, perspective=None, close=lambda: None))
    monkeypatch.setattr(main_mod, "VehicleTracker", lambda cfg: SimpleNamespace(
        update=lambda boxes: [TRACK], interpolate=lambda: [TRACK]))
    series = VehicleTimeSeries(track_id=3)
    series.offsets.append(OffsetSample(timestamp=0.0, offset_m=0.3, offset_ratio=0.1, lateral_velocity_mps=0.2,
                                       lane_confidence=0.9, is_interpolated=False))
    monkeypatch.setattr(main_mod, "OffsetCalculator", lambda cfg, p: SimpleNamespace(
        update=lambda tracks, lane, ts: {3: series}))
    monkeypatch.setattr(main_mod, "RiskScorer", Scorer)
    monkeypatch.setattr(main_mod, "FrameQualityChecker", lambda cfg: SimpleNamespace(check=lambda img: True))
    monkeypatch.setattr(main_mod, "create_sensor_source", lambda cfg: SimpleNamespace(
        read=lambda ts: SimpleNamespace(gps_speed_kmh=None), close=lambda: None))


def test_run_with_injected_source_and_sink(fake_pipeline, caplog):
    frames, sink = FakeFrames(), ListSink()
    with caplog.at_level(logging.INFO, logger="safewatch.main"):
        main_mod.run("configs/dev.yaml", "camera", frame_source=frames, event_sink=sink)

    assert frames.closed and sink.closed
    assert len(sink.got) == 1
    env = sink.got[0]
    assert set(env) == {"event", "trigger", "target"}
    assert env["event"]["risk"]["types"] == ["weaving", "swerving"]
    assert env["trigger"] == {"epoch": 1_794_725_421.0, "frame_id": EMIT_AT, "source": "camera", "video_sec": None}
    assert env["target"]["box"] == [252, 164, 348, 238] and env["target"]["frame_size"] == [1280, 720]
    # HW 테스트 도구가 파싱하는 로그 줄 — 형식 유지, 뒤에 event JSON
    line = next(r.getMessage() for r in caplog.records if "음주운전 의심 거동 감지" in r.getMessage())
    prefix = f"frame_id={EMIT_AT} 음주운전 의심 거동 감지 — "
    assert line.startswith(prefix)
    assert json.loads(line[len(prefix):]) == env["event"]


def test_run_survives_sink_failure(fake_pipeline):
    frames, sink = FakeFrames(), ListSink(fail=True)
    main_mod.run("configs/dev.yaml", "camera", frame_source=frames, event_sink=sink)
    assert frames.closed and sink.closed        # 예외로 끊기지 않고 끝까지 돌았다
