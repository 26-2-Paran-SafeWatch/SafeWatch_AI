"""더미 이벤트 생성기.

앱·서버 파트가 실제 AI 파이프라인 완성 전에 이벤트 수신·표시 로직을 먼저
개발할 수 있도록, docs/event-schema.md 스키마를 따르는 샘플 JSON을 생성한다.
클립 파일은 만들지 않고 메타데이터만 생성한다 (파일명만 스키마에 채움).

주의 — 이 스키마는 협의 중(v0.1)이다. 필드를 임의로 바꾸지 않는다
(AGENTS.md 원칙 4). 확정 전까지는 참고용으로만 사용한다.

사용법
    python scripts/generate_dummy_event.py --count 10 --output samples/
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))

RISK_TYPES = [
    "lane_departure",
    "weaving",
    "sudden_decel",
    "sudden_accel",
    "abrupt_lane_change",
]


def _level_for(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def make_event(index: int, device_id: str) -> dict:
    ts = datetime.now(KST) - timedelta(seconds=random.randint(0, 3600))
    event_id = f"evt_{ts.strftime('%Y%m%d_%H%M%S')}_{index:03d}"
    score = random.randint(40, 95)
    types = random.sample(RISK_TYPES, k=random.choice([2, 2, 3]))

    return {
        "event_id": event_id,
        "device_id": device_id,
        "timestamp": ts.isoformat(timespec="milliseconds"),
        "risk": {
            "score": score,
            "level": _level_for(score),
            "types": types,
        },
        "indicators": {
            "lane_offset_max_ratio": round(random.uniform(0.2, 0.45), 2),
            "lane_departure_duration_sec": round(random.uniform(1.0, 4.0), 1),
            "direction_changes_count": random.randint(2, 6),
            "observation_window_sec": 10.0,
            "longitudinal_accel_peak_g": round(random.uniform(-0.6, -0.2), 2),
            "heading_change_deg": round(random.uniform(5.0, 20.0), 1),
        },
        "target_vehicle": {
            "track_id": random.randint(1, 200),
            "tracked_duration_sec": round(random.uniform(8.0, 20.0), 1),
            "avg_detection_confidence": round(random.uniform(0.75, 0.98), 2),
        },
        "location": {
            "latitude": round(37.3219 + random.uniform(-0.01, 0.01), 6),
            "longitude": round(126.8309 + random.uniform(-0.01, 0.01), 6),
            "speed_kmh": round(random.uniform(40.0, 90.0), 1),
            "heading": round(random.uniform(0, 360), 1),
        },
        "clip": {
            "filename": f"{event_id}.mp4",
            "duration_sec": 15,
            "pre_event_sec": 10,
            "post_event_sec": 5,
            "resolution": "1280x720",
            "fps": 15,
            "size_bytes": random.randint(2_000_000, 5_000_000),
            "blurred": True,
        },
        "meta": {
            "model_version": "yolov8n-safewatch-dummy",
            "rule_version": "risk-v0.1-dummy",
            "confidence_gate_passed": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="더미 이벤트 JSON 생성")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", default="samples/")
    parser.add_argument("--device-id", default="safewatch-dummy-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.count + 1):
        event = make_event(i, args.device_id)
        out_path = out_dir / f"{event['event_id']}.json"
        out_path.write_text(
            json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"생성됨: {out_path}")


if __name__ == "__main__":
    main()
