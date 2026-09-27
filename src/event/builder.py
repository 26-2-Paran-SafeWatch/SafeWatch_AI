"""이벤트 메타데이터(JSON) 생성. docs/event-schema.md 참고.

⚠️ 이 모듈은 **AI 파트가 확정한 스키마 제안**(event-schema.md 5.1,
`risk.types`에 swerving/drifting/speed_irregular 반영)을 구현한다.
문서 2·3장의 예시는 스코프 변경 전 필드(`sudden_decel` 등)를 그대로
보여주고 있는데, 이는 **서버 파트(주민규) 최종 합의 전까지 의도적으로
보존**해둔 것이다(event-schema.md 5.1 "영향 받는 산출물") — 이 모듈과
`scripts/generate_dummy_event.py`가 실제로 만드는 JSON은 5.1의 새
필드를 쓰므로, 문서 2·3장과 코드가 당장은 서로 다른 스냅샷을 보여준다.
서버 합의가 끝나면 문서 본문을 갱신해 일치시킨다.

**이 모듈이 만들지 않는 것** — 영상 클립 추출·비식별화·전송 큐 적재는
EVENT 모듈의 나머지 책임이지만, 링 버퍼 요청 인터페이스가 HW 파트와
아직 협의되지 않아 미구현이다 (system-architecture.md "아직 남은 것").
그래서 `clip`·`location`은 호출부가 값을 못 주면 그대로 비워 둔다 —
스키마에서는 필수(`✓`)지만, 없는 값을 지어내는 것보다 정직하게 비우는
쪽을 택했다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.metrics.types import VehicleTimeSeries
from src.risk.types import RiskAssessment
from src.tracking.types import TrackedVehicle
from src.utils.config import Config

KST = timezone(timedelta(hours=9))


class EventIdGenerator:
    """`evt_YYYYMMDD_HHMMSS_NNN` 형식의 event_id를 발급한다 (event-schema.md 3.1).

    같은 초에 여러 이벤트가 발생하면 NNN을 증가시켜 구분한다. 디바이스
    하나당 하나씩 두고 재사용한다 — 새로 만들면 순번이 초기화되어 같은
    초에 겹칠 수 있다.
    """

    def __init__(self, device_id: str):
        self._device_id = device_id
        self._last_second_key: str | None = None
        self._seq = 0

    def next(self, timestamp: float) -> str:
        second_key = datetime.fromtimestamp(timestamp, tz=KST).strftime("%Y%m%d_%H%M%S")
        if second_key == self._last_second_key:
            self._seq += 1
        else:
            self._last_second_key = second_key
            self._seq = 1
        return f"evt_{second_key}_{self._seq:03d}"


def build_event_metadata(
    *,
    event_id: str,
    device_id: str,
    timestamp: float,
    assessment: RiskAssessment,
    series: VehicleTimeSeries,
    track: TrackedVehicle,
    cfg: Config,
    model_version: str,
    rule_version: str,
    location: dict[str, float] | None = None,
    clip: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """이벤트 JSON 메타데이터를 만든다. `docs/event-schema.md` 2장 구조를 따른다.

    Args:
        location: GPS 연결 전까지는 None — 스키마는 "GPS 미수신 시 객체
            전체를 생략"하는 방안을 검토 중이므로(3.5) 그 방식을 따른다.
        clip: 링 버퍼 연동 전까지는 None. 채워지면 그대로 통과시킨다.
    """
    indicators = {
        "observation_window_sec": cfg.risk.observation_window_sec,
        **assessment.indicators,
    }

    event: dict[str, Any] = {
        "event_id": event_id,
        "device_id": device_id,
        "timestamp": datetime.fromtimestamp(timestamp, tz=KST).isoformat(timespec="milliseconds"),
        "risk": {
            "score": assessment.score,
            "level": assessment.level,
            "types": assessment.types,
        },
        "indicators": indicators,
        "target_vehicle": {
            "track_id": track.track_id,
            "tracked_duration_sec": series.observed_duration_sec,
            # TODO: 평균 검출 신뢰도의 누적 이력이 아직 없다 — 최신 프레임 값으로 대체
            "avg_detection_confidence": track.box.confidence,
        },
        "meta": {
            "model_version": model_version,
            "rule_version": rule_version,
            "confidence_gate_passed": True,  # RISK 게이팅을 통과했을 때만 이 함수가 호출된다는 전제
        },
    }

    if location is not None:
        event["location"] = location
    if clip is not None:
        event["clip"] = clip

    return event
