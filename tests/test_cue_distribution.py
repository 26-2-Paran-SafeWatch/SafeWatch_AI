"""단서 분포 스크립트 — 차량별 최댓값 모으기, 게이팅 프레임 제외, 요약 표·파일 출력."""

from __future__ import annotations

import csv

import numpy as np

from scripts.cue_distribution import Collector, summarize, write_outputs
from src.detection.types import VehicleBox
from src.input.frame import Frame
from src.metrics.types import VehicleTimeSeries
from src.risk.types import RiskAssessment
from src.tracking.types import TrackedVehicle
from src.utils.config import load_config

CFG = load_config("configs/default.yaml")


def _track(tid):
    return TrackedVehicle(track_id=tid, box=VehicleBox(10, 10, 60, 50, 0.9, 0), tracked_frames=50, is_interpolated=False)


def _feed(col, tid, t, changes, vel, score=0, types=(), gated=False):
    ind = {} if gated else {"direction_changes_count": changes, "lateral_velocity_peak_mps": vel, "drift_duration_sec": 0.0}
    a = RiskAssessment(track_id=tid, score=score, level="low", types=list(types), indicators=ind)
    frame = Frame(image=np.zeros((72, 128, 3), np.uint8), timestamp=t, frame_id=int(t * 10), media_sec=t)
    col.hook(frame, _track(tid), VehicleTimeSeries(track_id=tid), a)


def test_collects_per_track_max_and_skips_gated_frames():
    col = Collector(CFG, snapshots=True)
    col.video = "v.mp4"
    _feed(col, 1, 0.0, 1, 0.4)
    _feed(col, 1, 2.0, 4, 0.2, score=40, types=("weaving", "swerving"))
    _feed(col, 1, 4.0, 2, 1.3)
    _feed(col, 1, 9.0, 99, 9.9, gated=True)            # 게이팅된 프레임 — 무시
    r = col.records[("v.mp4", 1)]
    assert r.peaks["direction_changes"] == 4 and r.peaks["lateral_velocity_peak"] == 1.3
    assert r.judged_sec == 4.0 and r.n_judged == 3
    assert r.max_score == 40 and r.cues_seen == {"weaving", "swerving"}
    assert r.best_frame_id == 20 and r.best_cues == ["weaving", "swerving"] and r.snapshot


def test_summary_and_files(tmp_path):
    col = Collector(CFG, snapshots=True)
    col.video = "v.mp4"
    for tid in range(1, 11):                           # 10대 중 2대만 사행 기준(3회) 넘음
        changes = 5 if tid <= 2 else 1
        _feed(col, tid, 0.0, changes, 0.1 * tid, score=20 if tid <= 2 else 0, types=("weaving",) if tid <= 2 else ())
        _feed(col, tid, 5.0, changes, 0.1 * tid)
    _feed(col, 99, 0.0, 9, 9.0)                        # 짧게 보인 차 — 요약에서 빠짐
    recs = list(col.records.values())
    text = summarize(recs, col.metrics, min_sec=3.0)
    assert "11대 중" in text and "**10대**" in text
    assert "20.0% (2대)" in text                         # 사행 기준 넘은 차량
    path = write_outputs(tmp_path, recs, col.metrics, 3.0, 5, "# t")
    assert path.exists() and list((tmp_path / "snapshots").glob("*.jpg"))
    rows = list(csv.DictReader(open(tmp_path / "tracks.csv", encoding="utf-8")))
    assert len(rows) == 11 and rows[0]["direction_changes"]


def test_empty_summary():
    assert "집계할 차량이 없다" in summarize([], Collector(CFG, False).metrics, 3.0)


def test_compare_two_datasets(tmp_path):
    from scripts.cue_distribution import compare, load_tracks

    paths = []
    for name, scale in (("a", 1.0), ("b", 2.0)):       # B는 옆 이동 속도가 A의 두 배
        col = Collector(CFG, snapshots=False)
        col.video = f"{name}.mp4"
        for tid in range(1, 101):
            _feed(col, tid, 0.0, 1, scale * tid / 100)
            _feed(col, tid, 5.0, 1, scale * tid / 100)
        write_outputs(tmp_path / name, list(col.records.values()), col.metrics, 3.0, 1, "# x")
        paths.append(tmp_path / name / "tracks.csv")
    metrics = Collector(CFG, False).metrics
    a, b = (load_tracks(p, metrics, 3.0) for p in paths)
    assert len(a["lateral_velocity_peak"]) == 100
    text = compare(a, b, metrics, ("a", "b"))
    row = next(line for line in text.splitlines() if line.startswith("| 스웨빙"))
    # A 95% 값(≈0.95)을 기준으로 삼으면 B(두 배)에선 절반 넘게 걸린다
    assert float(row.split("|")[6].strip().split("%")[0]) > 50
