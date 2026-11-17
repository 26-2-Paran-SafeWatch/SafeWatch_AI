"""단서별 측정값 분포 — 판정 기준(임계값)을 데이터로 정하기 위한 분석 도구.

정상 주행 영상을 넣으면 **차량마다** 단서 측정값의 최댓값을 모아, 단서별로
- 분포(중앙값·90·95·99 백분위·최대)
- "기준을 X로 두면 차량 몇 %가 걸리는지" (지금 설정값과 백분위 후보)
- 점수 분포(단서 몇 개까지 동시에 켜졌는지)
- 점수가 높게 나온 차량 목록과 그 순간 캡처(사람이 직접 보고 원인 확인용)
을 낸다. 판단 로직은 건드리지 않는다 — `src.main.run()`을 그대로 돌리며 판단 결과만 모은다(이벤트 파일도 안 만든다).

정상 주행에서 차량 몇 %가 넘는지 = 그 기준의 오탐률 추정. 음주처럼 움직이는 영상(시뮬레이션)을 넣으면
같은 표가 '놓치는 비율'을 보여 준다. 어느 백분위를 기준으로 삼을지는 데이터가 아니라 선택이다
(docs/risk-criteria.md 3.1 — NHTSA는 수치 기준을 공개하지 않는다).

차량 단위로 세는 이유 — 프레임 단위는 같은 차가 수백 번 세져 서로 독립이 아니다. 한 차가 지나가는 동안
한 번이라도 기준을 넘으면 그 차는 '걸린 차'다(이벤트도 차 단위로 난다).

사용법 (Pi, src.* import 때문에 -m)
    python -m scripts.cue_distribution --config configs/pi5_ufld_hailo.yaml \\
        --source ~/models/normal/ --output data/processed/cue_dist/normal --snapshots

    # 여러 파일
    python -m scripts.cue_distribution --config configs/pi5_ufld_hailo.yaml \\
        --source a.mp4 b.mp4 --output data/processed/cue_dist/ab

    # 두 데이터셋 비교 (파이프라인은 안 돌림) — 예: BDD100K로 잡은 기준이 한국 영상에도 맞는지
    python -m scripts.cue_distribution --config configs/pi5_ufld_hailo.yaml \
        --compare data/processed/cue_dist/bdd/tracks.csv data/processed/cue_dist/korea/tracks.csv \
        --output data/processed/cue_dist/compare

결과: <output>/summary.md (표), tracks.csv (차량별 값), snapshots/*.jpg (--snapshots, 점수 상위 차량).
비교 모드는 <output>/compare.md — 단서별 백분위를 나란히, 그리고 A의 백분위를 기준으로 삼았을 때 B에서 걸리는 비율.
⚠️ 영상 원본 프레임을 저장하므로(--snapshots) 밖으로 공유할 때는 비식별화 원칙(AGENTS.md 5)을 따른다.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
logger = logging.getLogger("safewatch.cue_distribution")


@dataclass(frozen=True)
class Metric:
    key: str          # 차량 기록의 열 이름
    cue: str          # 단서 이름 (risk.types)
    label: str
    unit: str
    threshold: float  # 지금 설정값 — 이 이상이면 단서 충족(측정값 기준)


def metrics_from_config(cfg) -> list[Metric]:
    r = cfg.risk
    return [
        Metric("direction_changes", "weaving", "사행 — 좌우 방향 전환 횟수", "회", float(r.weaving.min_direction_changes)),
        Metric("lateral_velocity_peak", "swerving", "스웨빙 — 최대 옆 이동 속도", "m/s",
               float(r.swerving.lateral_velocity_threshold_mps)),
        Metric("drift_duration", "drifting", "표류 — 한쪽 느린 이동 지속", "초", float(r.drifting.min_duration_sec)),
        Metric("departure_duration", "lane_departure", "차선 걸침 — 치우침 지속 (GPS 무관하게 계산)", "초",
               float(r.lane_departure.min_duration_sec)),
        Metric("expansion_rate_peak", "speed_irregular", "속도 불규칙 — 박스 팽창률", "/s",
               float(r.speed_irregular.expansion_rate_threshold_per_sec)),
    ]


@dataclass
class TrackRecord:
    video: str
    track_id: int
    first_ts: float | None = None
    last_ts: float | None = None
    n_judged: int = 0                                  # 판단 대상이 된(게이팅 통과) 프레임 수
    peaks: dict[str, float] = field(default_factory=dict)
    max_score: int = 0
    cues_seen: set[str] = field(default_factory=set)
    best_key: tuple[int, float] = (-1, -1.0)
    best_frame_id: int | None = None
    best_media_sec: float | None = None
    best_cues: list[str] = field(default_factory=list)
    snapshot: bytes | None = None

    @property
    def judged_sec(self) -> float:
        return 0.0 if self.first_ts is None else self.last_ts - self.first_ts


class Collector:
    """`src.main.run(on_assessment=...)`에 넣어 판단 결과를 차량별로 모은다."""

    def __init__(self, cfg, snapshots: bool):
        self.metrics = metrics_from_config(cfg)
        self._dep_th = float(cfg.risk.lane_departure.offset_ratio_threshold)
        self._snap = snapshots
        self.records: dict[tuple[str, int], TrackRecord] = {}
        self.video = ""

    def hook(self, frame, track, series, assessment) -> None:
        ind = assessment.indicators
        if "direction_changes_count" not in ind:     # 게이팅에서 걸러진 프레임(추적 부족·차선 보류·화질) — 세지 않음
            return
        rec = self.records.setdefault((self.video, track.track_id), TrackRecord(self.video, track.track_id))
        rec.first_ts = frame.timestamp if rec.first_ts is None else rec.first_ts
        rec.last_ts = frame.timestamp
        rec.n_judged += 1
        values = {
            "direction_changes": float(ind["direction_changes_count"]),
            "lateral_velocity_peak": float(ind.get("lateral_velocity_peak_mps", 0.0)),
            "drift_duration": float(ind.get("drift_duration_sec", 0.0)),
            "departure_duration": float(series.departure_duration_sec(self._dep_th)),
        }
        if "expansion_rate_peak_per_sec" in ind:
            values["expansion_rate_peak"] = float(ind["expansion_rate_peak_per_sec"])
        for k, v in values.items():
            rec.peaks[k] = max(rec.peaks.get(k, v), v)
        rec.max_score = max(rec.max_score, assessment.score)
        rec.cues_seen.update(assessment.types)

        # 이 차에서 가장 '음주 같아 보인' 순간 — 점수, 같으면 기준 대비 가장 큰 측정값
        ratio = max((values[m.key] / m.threshold for m in self.metrics if m.key in values and m.threshold > 0), default=0.0)
        key = (assessment.score, ratio)
        if key > rec.best_key:
            rec.best_key = key
            rec.best_frame_id = frame.frame_id
            rec.best_media_sec = frame.media_sec
            rec.best_cues = list(assessment.types)
            if self._snap:
                rec.snapshot = _snapshot(frame.image, track, assessment)


def _snapshot(image: np.ndarray, track, assessment) -> bytes:
    img = image.copy()
    b = track.box
    cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 0, 255), 2)
    text = f"track {track.track_id} score {assessment.score} {','.join(assessment.types) or '-'}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    if img.shape[1] > 960:
        img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


# ── 요약 ────────────────────────────────────────────────────────────────────

PCTS = (50, 90, 95, 99)


def summarize(records: list[TrackRecord], metrics: list[Metric], min_sec: float) -> str:
    kept = [r for r in records if r.judged_sec >= min_sec]
    lines = [
        f"차량 {len(records)}대 중 판단 구간 {min_sec:g}초 이상 **{len(kept)}대**로 집계 "
        f"(짧게 보인 차는 측정값이 작게 나와 분포를 낮춘다 — `--min-sec`).",
        "",
    ]
    if not kept:
        return "\n".join(lines + ["집계할 차량이 없다."])

    lines += ["## 단서별 분포 (차량별 최댓값)", "",
              "| 단서 | 대수 | 중앙값 | 90% | 95% | 99% | 최대 | 지금 기준 | 기준 넘은 차량 |",
              "|---|---|---|---|---|---|---|---|---|"]
    tables = []
    for m in metrics:
        v = np.array([r.peaks[m.key] for r in kept if m.key in r.peaks], dtype=float)
        if len(v) == 0:
            lines.append(f"| {m.label} | 0 | – | – | – | – | – | {m.threshold:g}{m.unit} | – |")
            continue
        p = np.percentile(v, PCTS)
        over = (v >= m.threshold).mean()
        lines.append(f"| {m.label} | {len(v)} | " + " | ".join(f"{x:.2f}" for x in p)
                     + f" | {v.max():.2f} | {m.threshold:g}{m.unit} | {over:.1%} ({int((v >= m.threshold).sum())}대) |")
        cands = sorted({m.threshold, *[float(x) for x in np.percentile(v, (90, 95, 99))]})
        rows = [f"| {c:.3g}{m.unit}{' (지금)' if c == m.threshold else ''} | {(v >= c).mean():.1%} ({int((v >= c).sum())}대) |"
                for c in cands]
        tables += [f"### {m.label}", "", "| 기준 | 걸리는 차량 |", "|---|---|", *rows, ""]
    lines += ["", "## 기준을 바꾸면 (후보 = 지금 설정 + 90·95·99 백분위)", "", *tables]

    scores = np.array([r.max_score for r in kept])
    lines += ["## 점수 분포 (차량별 최고 점수 = 동시에 켜진 단서 수 × 20)", "", "| 최고 점수 이상 | 차량 |", "|---|---|"]
    for s in (20, 40, 60, 80):
        lines.append(f"| {s}점 (단서 {s // 20}개) | {(scores >= s).mean():.1%} ({int((scores >= s).sum())}대) |")
    cue_counts = {m.cue: sum(m.cue in r.cues_seen for r in kept) for m in metrics}
    lines += ["", "단서별로 한 번이라도 켜진 차량: "
              + ", ".join(f"{c} {n}대" for c, n in cue_counts.items()), ""]
    return "\n".join(lines)


def top_cases(records: list[TrackRecord], n: int) -> list[TrackRecord]:
    return sorted(records, key=lambda r: r.best_key, reverse=True)[:n]


def write_outputs(out: Path, records: list[TrackRecord], metrics: list[Metric], min_sec: float, n_top: int,
                  header: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    keys = [m.key for m in metrics]
    with open(out / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track_id", "judged_sec", "n_judged", *keys, "max_score", "cues_seen",
                    "best_frame_id", "best_video_sec", "best_cues"])
        for r in records:
            w.writerow([r.video, r.track_id, f"{r.judged_sec:.2f}", r.n_judged,
                        *[f"{r.peaks[k]:.4f}" if k in r.peaks else "" for k in keys], r.max_score,
                        "|".join(sorted(r.cues_seen)), r.best_frame_id,
                        "" if r.best_media_sec is None else f"{r.best_media_sec:.2f}", "|".join(r.best_cues)])

    top = top_cases([r for r in records if r.judged_sec >= min_sec], n_top)
    snap_dir = out / "snapshots"
    lines = [header, "", summarize(records, metrics, min_sec), "## 점수 상위 차량 — 직접 보고 원인 확인", "",
             "| 영상 | 차량 | 최고 점수 | 그 순간 단서 | 영상 시각(초) | 프레임 | 캡처 |", "|---|---|---|---|---|---|---|"]
    for r in top:
        snap = ""
        if r.snapshot is not None:
            snap_dir.mkdir(exist_ok=True)
            name = f"{Path(r.video).stem}_t{r.track_id}_s{r.best_key[0]}.jpg"
            (snap_dir / name).write_bytes(r.snapshot)
            snap = f"snapshots/{name}"
        sec = "" if r.best_media_sec is None else f"{r.best_media_sec:.1f}"
        lines.append(f"| {Path(r.video).name} | {r.track_id} | {r.best_key[0]} | {', '.join(r.best_cues) or '-'} "
                     f"| {sec} | {r.best_frame_id} | {snap} |")
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── 두 데이터셋 비교 ──────────────────────────────────────────────────────────

def load_tracks(path: str | Path, metrics: list[Metric], min_sec: float) -> dict[str, np.ndarray]:
    """tracks.csv → 단서별 차량 값 (판단 구간 min_sec 이상만)."""
    out: dict[str, list[float]] = {m.key: [] for m in metrics}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["judged_sec"]) < min_sec:
                continue
            for m in metrics:
                if row.get(m.key):
                    out[m.key].append(float(row[m.key]))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray], metrics: list[Metric],
            names: tuple[str, str]) -> str:
    """A(기준을 잡는 쪽, 예: BDD100K)와 B(확인하는 쪽, 예: 한국 영상)의 분포 비교.

    'A 95% 기준 → B에서 걸리는 비율'이 5% 근처면 A로 잡은 기준을 B에 써도 오탐률이 비슷하다는 뜻이다.
    크게 높으면 B(한국)에서 측정값이 더 크게 나온다 — 기준을 B 쪽으로 다시 잡아야 한다.
    """
    na, nb = names
    lines = [f"# 분포 비교 — A `{na}` vs B `{nb}`", "",
             "| 단서 | A 대수 | B 대수 | A 50/90/95/99% | B 50/90/95/99% | A 95% 기준 → B 걸림 | A 99% 기준 → B 걸림 |",
             "|---|---|---|---|---|---|---|"]
    for m in metrics:
        va, vb = a.get(m.key, np.array([])), b.get(m.key, np.array([]))
        if len(va) == 0 or len(vb) == 0:
            lines.append(f"| {m.label} | {len(va)} | {len(vb)} | – | – | – | – |")
            continue
        pa, pb = np.percentile(va, PCTS), np.percentile(vb, PCTS)
        fmt = lambda p: " / ".join(f"{x:.2f}" for x in p)   # noqa: E731
        lines.append(f"| {m.label} | {len(va)} | {len(vb)} | {fmt(pa)} | {fmt(pb)} "
                     f"| {(vb >= pa[2]).mean():.1%} | {(vb >= pa[3]).mean():.1%} |")
    lines += ["", "읽는 법 — 'A 95% 기준 → B 걸림'이 약 5%, 'A 99% 기준 → B 걸림'이 약 1%면 두 데이터의 분포가 비슷하다. "
              "B 대수가 적으면(수십 대) 1~2대 차이로 비율이 크게 흔들리니 대수와 같이 본다."]
    return "\n".join(lines)


# ── 실행 ────────────────────────────────────────────────────────────────────

class _NoEvents:
    """이벤트 파일을 만들지 않는 sink — 분석만 한다."""

    def publish(self, envelope) -> None: ...
    def close(self) -> None: ...


def find_videos(sources: list[str]) -> list[Path]:
    out = []
    for s in sources:
        p = Path(s).expanduser()
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_SUFFIXES)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(s)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="단서별 측정값 분포 (차량 단위)")
    p.add_argument("--config", required=True)
    p.add_argument("--source", nargs="+", help="영상 파일들 또는 폴더")
    p.add_argument("--compare", nargs=2, metavar=("A_CSV", "B_CSV"),
                   help="파이프라인 대신 두 tracks.csv 비교 (A = 기준을 잡는 쪽, B = 확인하는 쪽)")
    p.add_argument("--output", required=True, help="결과 폴더")
    p.add_argument("--min-sec", type=float, default=3.0, help="판단 구간이 이보다 짧게 보인 차량은 요약에서 뺀다")
    p.add_argument("--top", type=int, default=30, help="점수 상위 차량 목록 길이")
    p.add_argument("--snapshots", action="store_true", help="상위 차량의 그 순간을 jpg로 저장")
    return p.parse_args()


def main() -> None:
    from src import main as pipeline
    from src.utils.config import load_config

    args = parse_args()
    cfg = load_config(args.config)
    if args.compare:
        metrics = metrics_from_config(cfg)
        a, b = (load_tracks(p, metrics, args.min_sec) for p in args.compare)
        text = compare(a, b, metrics, (args.compare[0], args.compare[1]))
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "compare.md").write_text(text + "\n", encoding="utf-8")
        print(text + f"\n\n저장됨: {out / 'compare.md'}")
        return
    if not args.source:
        raise SystemExit("--source나 --compare 중 하나가 필요합니다")
    videos = find_videos(args.source)
    if not videos:
        raise SystemExit("영상이 없습니다")
    if int(cfg.detection.interval) > 1:
        print(f"⚠️ detection.interval={cfg.detection.interval} — 검출을 건너뛴 프레임은 위치를 그대로 두어 옆 이동 속도가 "
              "틀어진다(사행·스웨빙 값에 영향). NPU면 interval 1 설정으로 재는 것을 권한다.")

    col = Collector(cfg, args.snapshots)
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v}")
        col.video = str(v)
        pipeline.run(args.config, str(v), event_sink=_NoEvents(), on_assessment=col.hook)

    header = (f"# 단서 분포 — {len(videos)}개 영상, 설정 `{args.config}`\n\n"
              f"영상: {', '.join(v.name for v in videos[:10])}{' …' if len(videos) > 10 else ''}")
    path = write_outputs(Path(args.output), list(col.records.values()), col.metrics, args.min_sec, args.top, header)
    print(path.read_text(encoding="utf-8"))
    print(f"\n저장됨: {path}")


if __name__ == "__main__":
    main()
