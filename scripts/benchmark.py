"""성능 벤치마크. docs/sprint-plan.md S6, docs/pipeline-architecture.md 4장 참고.

INPUT~RISK(EVENT 제외 — pipeline-architecture.md 3.7 "이벤트 발생 시에만
소요"라 상시 성능 예산에 안 들어간다)를 실제로 돌리면서 단계별 처리
시간(`src/utils/profiler.py` 집계)·전체 fps·(Pi5에서는) CPU 온도를 재서
콘솔에 출력하고 `results/benchmark/`에 JSON으로 남긴다.

**CPU 온도 측정은 Pi5 전용**이다. `/sys/class/thermal/thermal_zone0/temp`
(Pi5 경로)가 없는 환경(개발 PC)에서는 자동으로 건너뛰고 기록하지 않는다
— 플랫폼 분기를 이 스크립트 안에서 흡수해 Pi5/PC 양쪽에서 코드 수정 없이
동작하게 한다(AGENTS.md "플랫폼 차이의 국소화"와 같은 원칙).

**이 결과를 Pi5 성능으로 단정하지 않는다** — 개발 PC에서 돌리면 PC
성능이 나올 뿐이다(AGENTS.md "불확실할 때" 원칙). S6 완료 조건(Pi5에서
10fps 이상, 30분 연속 구동 시 fps 저하 20% 이내)을 판정하려면 실제로
Pi5에서 이 스크립트를 돌려야 한다.

사용법 — `src.*`를 import하므로 반드시 `-m`으로 실행한다(`src/main.py`와 같은 이유,
직접 `python scripts/benchmark.py`로 실행하면 `ModuleNotFoundError: No module named 'src'`).
    # 영상 파일 전체를 한 번 처리
    python -m scripts.benchmark --config configs/pi5.yaml --source data/raw/sample.mp4

    # 카메라로 60초간 연속 구동 측정 (Pi5)
    python -m scripts.benchmark --config configs/pi5.yaml --source camera --duration-sec 60
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.detection.detector import VehicleDetector
from src.input.source import create_source
from src.lane.lane_detector import LaneDetector
from src.metrics.offset import OffsetCalculator
from src.risk.frame_quality import FrameQualityChecker
from src.risk.scorer import RiskScorer
from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.benchmark")

_PI5_THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
_TARGET_FPS = 10  # sprint-plan.md S6 완료 조건 (Pi5 기준)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeWatch AI 성능 벤치마크")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument("--source", required=True, help="영상 파일 경로 또는 'camera'(Pi5)")
    parser.add_argument(
        "--duration-sec", type=float, default=None,
        help="이 시간(초)만큼만 처리하고 종료. 생략하면 영상 파일은 끝까지, "
             "camera는 Ctrl+C로 중단할 때까지 처리한다",
    )
    parser.add_argument(
        "--temp-interval-sec", type=float, default=5.0,
        help="CPU 온도 샘플링 주기(Pi5 전용). 매 프레임 읽으면 그 자체가 오버헤드라 간격을 둔다",
    )
    parser.add_argument(
        "--output-dir", default="results/benchmark", help="결과 JSON을 저장할 디렉터리",
    )
    return parser.parse_args()


def read_cpu_temp_c() -> float | None:
    """Pi5의 `/sys/class/thermal/thermal_zone0/temp`(밀리섭씨 단위 텍스트)를 읽는다.

    이 경로가 없는 환경(개발 PC)에서는 조용히 None을 반환한다 — 온도
    미측정이 벤치마크 자체의 실패 이유가 되면 안 된다(AGENTS.md "실패
    격리"와 같은 정신. 장시간 구동 중 온도 센서 접근이 일시적으로
    실패하는 경우도 같은 이유로 조용히 넘어간다).
    """
    try:
        millidegrees = int(_PI5_THERMAL_PATH.read_text().strip())
        return millidegrees / 1000.0
    except (FileNotFoundError, ValueError, PermissionError, OSError):
        return None


@dataclass
class BenchmarkResult:
    config_path: str
    source: str
    started_at: str                      # ISO 8601 UTC
    elapsed_sec: float
    frame_count: int
    fps: float
    meets_target_fps: bool               # fps >= _TARGET_FPS (Pi5 기준일 때만 의미 있음)
    stage_avg_ms: dict[str, float] = field(default_factory=dict)
    temp_samples_c: list[float] = field(default_factory=list)
    temp_note: str = ""


def run_benchmark(
    config_path: str, source: str, duration_sec: float | None, temp_interval_sec: float,
) -> BenchmarkResult:
    cfg = load_config(config_path)
    setup_logging(cfg)
    profiler.reset()  # 이전 실행 잔여 통계가 섞이지 않도록

    resolution = tuple(cfg.input.resolution)
    frame_source = create_source(source, resolution=resolution)
    logger.info("벤치마크 시작: config=%s source=%s", config_path, source)

    detector = VehicleDetector(cfg)
    detection_interval = cfg.detection.interval
    lane_detector = LaneDetector(cfg)
    tracker = VehicleTracker(cfg)
    offset_calculator = OffsetCalculator(cfg, lane_detector.perspective)
    risk_scorer = RiskScorer(cfg)
    frame_quality_checker = FrameQualityChecker(cfg)

    started_wall = datetime.now(timezone.utc)
    start = time.perf_counter()
    frame_count = 0
    temp_samples: list[float] = []
    last_temp_at = 0.0

    try:
        for frame in frame_source.frames():
            elapsed = time.perf_counter() - start
            if duration_sec is not None and elapsed >= duration_sec:
                break

            if frame.frame_id % detection_interval == 0:
                boxes = detector.detect(frame.image)
                tracks = tracker.update(boxes)
            else:
                tracks = tracker.interpolate()

            lane = lane_detector.detect(frame.image)
            frame_quality_ok = frame_quality_checker.check(frame.image)
            series = offset_calculator.update(tracks, lane, frame.timestamp)

            risk_scorer.sync({t.track_id for t in tracks})
            for track in tracks:
                ts = series.get(track.track_id)
                if ts is None:
                    continue
                risk_scorer.assess(
                    track, ts, lane_valid=lane.valid, timestamp=frame.timestamp,
                    ego_speed_kmh=None, frame_quality_ok=frame_quality_ok,
                )

            frame_count += 1
            if elapsed - last_temp_at >= temp_interval_sec:
                temp = read_cpu_temp_c()
                if temp is not None:
                    temp_samples.append(temp)
                last_temp_at = elapsed
    except KeyboardInterrupt:
        logger.info("중단 요청 수신, 지금까지 처리한 구간으로 결과를 낸다.")
    finally:
        frame_source.close()

    elapsed_sec = time.perf_counter() - start
    fps = frame_count / elapsed_sec if elapsed_sec > 0 else 0.0

    if temp_samples:
        temp_note = f"샘플 {len(temp_samples)}개"
    elif _PI5_THERMAL_PATH.exists():
        temp_note = "온도 경로는 있으나 측정 구간 내 샘플링 전에 종료됨"
    else:
        temp_note = f"미측정 — Pi5 전용 경로({_PI5_THERMAL_PATH}) 없음, 개발 PC로 추정"

    return BenchmarkResult(
        config_path=config_path,
        source=source,
        started_at=started_wall.isoformat(timespec="seconds"),
        elapsed_sec=round(elapsed_sec, 2),
        frame_count=frame_count,
        fps=round(fps, 2),
        meets_target_fps=fps >= _TARGET_FPS,
        stage_avg_ms={k: round(v, 3) for k, v in profiler.summary().items()},
        temp_samples_c=temp_samples,
        temp_note=temp_note,
    )


_COLD_START_WARNING_THRESHOLD = 30  # 이 미만이면 최초 추론 콜드스타트가 평균을 크게 왜곡할 수 있음


def print_summary(result: BenchmarkResult) -> None:
    print(f"\nconfig={result.config_path}  source={result.source}")
    print(f"frames={result.frame_count}  elapsed={result.elapsed_sec}s  fps={result.fps}")
    print(f"목표 {_TARGET_FPS}fps 이상(Pi5 기준, sprint-plan.md S6) — "
          f"{'달성' if result.meets_target_fps else '미달'}")
    if result.frame_count < _COLD_START_WARNING_THRESHOLD:
        print(f"⚠️ 처리 프레임이 {result.frame_count}개로 적다 — 최초 추론 콜드스타트(모델 "
              f"워밍업) 1회가 평균을 크게 부풀렸을 수 있다. 더 긴 영상이나 "
              f"--duration-sec을 늘려 재측정을 권장한다")

    print("\n단계별 평균 처리 시간(ms):")
    for stage, ms in result.stage_avg_ms.items():
        print(f"  {stage:>15}: {ms:>8.3f}")
    total = sum(result.stage_avg_ms.values())
    print(f"  {'합계(참고)':>15}: {total:>8.3f}  (검출 미수행 프레임 포함 평균이라 "
          f"pipeline-architecture.md 4장 단일 예산표와 직접 비교 불가)")

    print(f"\nCPU 온도: {result.temp_note}")
    if result.temp_samples_c:
        print(f"  평균 {statistics.mean(result.temp_samples_c):.1f}°C, "
              f"최대 {max(result.temp_samples_c):.1f}°C")


def save_result(result: BenchmarkResult, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_stem = Path(result.config_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{config_stem}_{timestamp}.json"
    out_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        config_path=args.config, source=args.source,
        duration_sec=args.duration_sec, temp_interval_sec=args.temp_interval_sec,
    )
    print_summary(result)
    out_path = save_result(result, args.output_dir)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
