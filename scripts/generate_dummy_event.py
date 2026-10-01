"""더미 이벤트 생성기.

앱·서버 파트가 실제 AI 파이프라인 완성 전에 이벤트 수신·표시 로직을 먼저
개발할 수 있도록, docs/event-schema.md 스키마를 따르는 샘플 JSON을 생성한다.
클립 파일은 만들지 않고 메타데이터만 생성한다 (파일명만 스키마에 채움).

⚠️ 이 스크립트는 **AI 파트가 확정한 스키마 제안**(event-schema.md 5.1,
2026-09-26 스코프 변경 반영)을 따른다 — `risk.types`에 swerving·drifting·
speed_irregular가 추가되고 sudden_decel·sudden_accel·abrupt_lane_change는
빠졌다. `docs/event-schema.md` 2·3장의 예시는 서버 파트(주민규) 최종 합의
전까지 스코프 변경 이전 필드를 그대로 보존해두고 있어, 이 스크립트의
출력과 문서 본문이 당장은 서로 다른 스냅샷을 보여준다 — 이 스크립트가
"AI 파트가 실제로 만들 형태"이고, 서버 합의가 끝나면 문서를 맞춰 갱신한다.

**지표 생략 규칙도 실제 스키마와 동일하게 흉내낸다** — 이벤트마다 무작위로
선택된 `types`에 관련된 지표만 채우고 나머지는 생략한다(`observation_window_sec`
제외). 서버·앱 파트가 "이 지표는 항상 있는 게 아니다"라는 실제 동작을
개발 단계에서부터 반영할 수 있게 하기 위함이다.

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

# risk.types 허용값 (event-schema.md 5.1). 타입별로 채울 indicators 필드를 함께 정의한다.
TYPE_INDICATORS: dict[str, dict[str, tuple[float, float]]] = {
    "lane_departure": {
        "lane_offset_max_ratio": (0.2, 0.45),
        "lane_departure_duration_sec": (1.0, 4.0),
    },
    "weaving": {
        "direction_changes_count": (2, 6),  # int로 반올림해서 씀
    },
    "swerving": {
        "lateral_velocity_peak_mps": (0.8, 2.0),
    },
    "drifting": {
        "drift_duration_sec": (2.5, 6.0),
    },
    "speed_irregular": {
        "expansion_rate_peak_per_sec": (0.2, 0.6),
    },
}
RISK_TYPES = list(TYPE_INDICATORS)


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

    indicators: dict[str, float] = {"observation_window_sec": 10.0}
    for t in types:
        for field, (lo, hi) in TYPE_INDICATORS[t].items():
            value = random.uniform(lo, hi)
            indicators[field] = int(value) if field.endswith("_count") else round(value, 2)

    return {
        "event_id": event_id,
        "device_id": device_id,
        "timestamp": ts.isoformat(timespec="milliseconds"),
        "risk": {
            "score": score,
            "level": _level_for(score),
            "types": types,
        },
        "indicators": indicators,
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
            "rule_version": "risk-v0.10-dui-scope-dummy",
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
