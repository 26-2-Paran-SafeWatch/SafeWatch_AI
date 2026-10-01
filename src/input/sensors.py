"""GPS/IMU 센서 입력 추상화. docs/pipeline-architecture.md 5장 "GPS/위치 데이터
입력 경로", docs/risk-criteria.md 4.1 "자차 속도" 게이팅 참고.

PC/Pi5 플랫폼 차이는 `src/input/source.py`(영상)와 같은 원칙으로 이 모듈
안에서만 흡수한다 (AGENTS.md "개발 환경 차이" 표 — "IMU: 더미 데이터(PC) /
ESP32 UART 수신(Pi5)").

**지금 채우는 것과 못 채우는 것**

`risk-criteria.md` 4.1의 "자차 속도 ≥ 60km/h"(LDWS 근거) 게이팅은
`RiskScorer`에 이미 구현돼 있었지만, `main.py`가 항상 `ego_speed_kmh=None`을
하드코딩해서 넘겨 `lane_departure` 단서가 **절대 충족될 수 없는 상태**였다
(GPS 입력 경로 자체가 없었음). 이 모듈로 그 입력 경로를 만들어 PC에서는
더미 속도로, 실제로 `lane_departure`가 발동하는 것까지 `main.py` 기준으로
테스트할 수 있게 한다.

**Pi5(ESP32 UART) 쪽은 아직 못 채운다** — `pipeline-architecture.md` 5장이
명시하듯 "ESP32 경유 여부 등 구체 프로토콜, 타임스탬프 동기화 방식"이
HW 파트와 아직 협의되지 않았다. 와이어 포맷을 추측해서 구현하면 실제
ESP32 출력과 맞지 않을 가능성이 높고, 나중에 통째로 다시 짜야 할 수
있다 — 그래서 `SerialGPSIMUSource`는 인터페이스 계약만 만족하는 스텁으로
남겨 두고, 프로토콜이 확정되면 그 자리만 채우면 되게 했다
(`src/event/clip.py`의 `ClipProvider`/`NullClipProvider`와 같은 패턴).

IMU(yaw rate 등) 자체는 아직 어떤 코드도 소비하지 않는다 — 자차 거동
보상 공식이 근거 없이 미구현 상태로 남아 있기 때문이다(`risk-criteria.md`
2.1, AGENTS.md 원칙 3). 그래서 `SensorReading`은 지금 당장 쓰이는
`gps_speed_kmh`만 담고, IMU 필드는 보상 공식이 정해질 때 추가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.utils.config import Config


@dataclass
class SensorReading:
    timestamp: float
    gps_speed_kmh: float | None  # None이면 미수신 — RiskScorer가 안전하게 판단 보류 (risk-criteria.md 4.1)


class SensorSource(Protocol):
    def read(self, timestamp: float) -> SensorReading: ...

    def close(self) -> None: ...


class DummySensorSource:
    """PC 개발용 — config에 고정된 속도를 그대로 반환한다.

    실제 GPS 신호를 흉내 내려는 게 아니라, `lane_departure`의 속도
    게이팅 경로 자체를 `main.py`로 끝까지 통과시켜 볼 수 있게 하는 용도다
    (AGENTS.md "개발 환경 차이" 표의 "IMU: 더미 데이터" 칸과 같은 역할).
    **PC 개발 전용** — 실제 배포 환경(Pi5)에서는 쓰지 않는다. 존재하지도
    않는 GPS 신호를 고정값으로 지어내는 것이라, 실기기에서 쓰면 속도
    게이팅이 실제로는 미수신인데도 항상 통과한 것처럼 보이게 된다.
    """

    def __init__(self, speed_kmh: float | None):
        self._speed_kmh = speed_kmh

    def read(self, timestamp: float) -> SensorReading:
        return SensorReading(timestamp=timestamp, gps_speed_kmh=self._speed_kmh)

    def close(self) -> None:
        pass


class UnavailableSensorSource:
    """GPS/IMU가 아직 연결되지 않은 상태의 정직한 기본값 — 항상 미수신(None)을 반환한다.

    `main.py`가 지금까지 하드코딩해 온 `ego_speed_kmh=None`과 정확히 같은
    동작이다(`RiskScorer`가 None을 안전하게 판단 보류로 처리 —
    risk-criteria.md 4.1). `SerialGPSIMUSource`가 아직 미구현이라
    `pi5.yaml`의 실제 기본값은 이 클래스다 — ESP32 프로토콜이 확정되기
    전까지는 "크래시" 대신 "정직하게 미수신 처리"가 더 안전하다.
    """

    def read(self, timestamp: float) -> SensorReading:
        return SensorReading(timestamp=timestamp, gps_speed_kmh=None)

    def close(self) -> None:
        pass


class SerialGPSIMUSource:
    """Pi5 전용 — ESP32가 UART로 보내는 GPS/IMU 프레임을 수신한다.

    **아직 구현되지 않았다.** 와이어 포맷(필드 구성·구분자·체크섬 유무),
    보율(baud rate), 타임스탬프 동기화 방식이 HW 파트(강섬희)와 확정되지
    않아 추측으로 구현하면 실제 장치와 맞지 않을 수 있다 — 확정되는 대로
    `read()`만 채우면 된다(생성자에서 시리얼 포트를 열어 두는 구조까지는
    미리 잡아 뒀다).
    """

    def __init__(self, port: str, baudrate: int):
        self._port = port
        self._baudrate = baudrate
        raise NotImplementedError(
            "TODO: ESP32 UART 프레임 포맷 확정 후 구현 "
            "(pipeline-architecture.md 5장 'GPS/위치 데이터 입력 경로' 참고). "
            f"port={port} baudrate={baudrate}"
        )

    def read(self, timestamp: float) -> SensorReading:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def create_sensor_source(cfg: Config) -> SensorSource:
    sensors_cfg = cfg.sensors
    source = sensors_cfg.source
    if source == "unavailable":
        return UnavailableSensorSource()
    if source == "dummy":
        return DummySensorSource(speed_kmh=sensors_cfg.dummy_gps_speed_kmh)
    if source == "serial":
        return SerialGPSIMUSource(port=sensors_cfg.serial_port, baudrate=sensors_cfg.serial_baudrate)
    raise ValueError(f"알 수 없는 sensors.source: {source!r} (unavailable | dummy | serial)")
