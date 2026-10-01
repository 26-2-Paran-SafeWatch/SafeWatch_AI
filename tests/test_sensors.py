"""GPS/IMU 센서 입력 테스트. docs/pipeline-architecture.md 5장 참고."""

from __future__ import annotations

import pytest

from src.input.sensors import (
    DummySensorSource,
    SerialGPSIMUSource,
    UnavailableSensorSource,
    create_sensor_source,
)
from src.utils.config import Config, load_config


def test_unavailable_source_always_reports_no_speed():
    source = UnavailableSensorSource()
    reading = source.read(timestamp=100.0)
    assert reading.gps_speed_kmh is None
    assert reading.timestamp == 100.0


def test_dummy_source_returns_configured_speed():
    source = DummySensorSource(speed_kmh=80.0)
    assert source.read(timestamp=1.0).gps_speed_kmh == 80.0
    assert source.read(timestamp=2.0).gps_speed_kmh == 80.0  # 매번 동일 — 고정값


def test_serial_source_not_yet_implemented():
    """ESP32 UART 프로토콜 미확정 — 추측으로 구현하지 않고 명확히 실패한다."""
    with pytest.raises(NotImplementedError):
        SerialGPSIMUSource(port="/dev/ttyUSB0", baudrate=115200)


def test_create_sensor_source_dispatches_on_config():
    unavailable_cfg = Config({"sensors": {"source": "unavailable"}})
    assert isinstance(create_sensor_source(unavailable_cfg), UnavailableSensorSource)

    dummy_cfg = Config({"sensors": {"source": "dummy", "dummy_gps_speed_kmh": 72.0}})
    dummy_source = create_sensor_source(dummy_cfg)
    assert isinstance(dummy_source, DummySensorSource)
    assert dummy_source.read(0.0).gps_speed_kmh == 72.0


def test_create_sensor_source_rejects_unknown_source():
    cfg = Config({"sensors": {"source": "bluetooth"}})
    with pytest.raises(ValueError):
        create_sensor_source(cfg)


def test_dev_config_uses_dummy_and_pi5_uses_unavailable():
    """실제 배포 config가 의도한 안전 기본값을 쓰는지 — pi5.yaml이 실수로
    dummy(고정값 지어내기)를 쓰면 실제 GPS 미수신 상황에서도 속도 게이팅이
    항상 통과한 것처럼 동작해 위험하다."""
    dev_cfg = load_config("configs/dev.yaml")
    pi5_cfg = load_config("configs/pi5.yaml")

    assert dev_cfg.sensors.source == "dummy"
    assert pi5_cfg.sensors.source == "unavailable"
