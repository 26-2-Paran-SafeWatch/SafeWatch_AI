"""이벤트 메타데이터(JSON) 생성. docs/event-schema.md 참고.

⚠️ 이 모듈은 **AI 파트가 확정한 스키마 제안**(event-schema.md 5.1,
`risk.types`에 swerving/drifting/speed_irregular 반영)을 구현한다.
문서 2·3장의 예시는 스코프 변경 전 필드(`sudden_decel` 등)를 그대로
보여주고 있는데, 이는 **서버 파트(주민규) 최종 합의 전까지 의도적으로
보존**해둔 것이다(event-schema.md 5.1 "영향 받는 산출물") — 이 모듈과
`scripts/generate_dummy_event.py`가 실제로 만드는 JSON은 5.1의 새
필드를 쓰므로, 문서 2·3장과 코드가 당장은 서로 다른 스냅샷을 보여준다.
서버 합의가 끝나면 문서 본문을 갱신해 일치시킨다.

**역할 나누기 (2026-11-15 HW 파트와 합의)** — AI는 이벤트 JSON을 만들어 HW에
넘기는 데까지만 한다. 영상 클립 자르기, `clip`·`location` 채우기, 큐, 재시도,
서버 전송은 HW 파트(SafeWatch_HW)가 맡는다. 그래서 `build_event_metadata`는
`clip`·`location`을 넣지 않는다 — 최종 서버 JSON = event + location + clip이고,
HW는 event 안의 값을 고치지 않는다.

`build_event_envelope`가 HW에 넘기는 묶음(event / trigger / target)을 만든다.
형식은 docs/event-schema.md "AI → HW 전달 형식(envelope)".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.input.frame import Frame
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
    cfg: Config,
    model_version: str,
    rule_version: str,
) -> dict[str, Any]:
    """서버로 나가는 이벤트 JSON의 AI 몫(event). `docs/event-schema.md` 2장 구조에서
    `clip`·`location`(HW가 붙임)과 `target_vehicle`(envelope의 target으로 옮김 — 서버로
    보낼지는 서버 파트 결정 전까지 TBD)을 뺀 것이다."""
    indicators = {
        "observation_window_sec": cfg.risk.observation_window_sec,
        **assessment.indicators,
    }
    return {
        "event_id": event_id,
        "device_id": device_id,
        "timestamp": datetime.fromtimestamp(timestamp, tz=KST).isoformat(timespec="milliseconds"),
        "risk": {
            "score": assessment.score,
            "level": assessment.level,
            "types": assessment.types,
        },
        "indicators": indicators,
        "meta": {
            "model_version": model_version,
            "rule_version": rule_version,
            "confidence_gate_passed": True,  # RISK 게이팅을 통과했을 때만 이 함수가 호출된다는 전제
        },
    }


def build_event_target(track: TrackedVehicle, series: VehicleTimeSeries, frame_size: tuple[int, int]) -> dict[str, Any]:
    """envelope의 target — 서버에 안 보내는 참고 정보. HW가 대상 차량 표시·번호판 블러 위치에 쓴다."""
    b = track.box
    return {
        "track_id": track.track_id,
        "box": [round(b.x1), round(b.y1), round(b.x2), round(b.y2)],   # 원본 프레임 좌표 [x1, y1, x2, y2]
        "frame_size": [int(frame_size[0]), int(frame_size[1])],      # [width, height]
        "tracked_duration_sec": series.observed_duration_sec,
        # TODO: 평균 검출 신뢰도의 누적 이력이 아직 없다 — 최신 프레임 값으로 대체
        "avg_detection_confidence": track.box.confidence,
    }


def trigger_source(source: str) -> str:
    """envelope trigger.source — 카메라면 "camera", 영상 파일이면 절대 경로."""
    return source if source == "camera" else str(Path(source).resolve())


def build_event_envelope(
    event: dict[str, Any],
    frame: Frame,
    track: TrackedVehicle,
    series: VehicleTimeSeries,
    source: str,
) -> dict[str, Any]:
    """AI → HW 전달 묶음. event = 서버로 나가는 부분, trigger = HW가 클립을 자를 기준,
    target = 서버에 안 보내는 참고 정보. 클립 전·후 길이는 HW config가 정한다."""
    h, w = frame.image.shape[:2]
    return {
        "event": event,
        "trigger": {
            "epoch": frame.timestamp,          # 판정이 켜진 프레임 시각. 카메라면 HW 링버퍼와 같은 시계(time.time())
            "frame_id": frame.frame_id,
            "source": trigger_source(source),
            "video_sec": frame.media_sec,      # 영상 파일 입력일 때만 영상 안 시각, 카메라는 None
        },
        "target": build_event_target(track, series, (w, h)),
    }
