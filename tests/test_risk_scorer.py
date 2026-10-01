"""RiskScorer 테스트.

실제 파이프라인(검출→차선→추적→offset) 없이, `VehicleTimeSeries`를 직접
구성해 판정 로직만 검증한다. 좌표 변환·차선 인식 품질은 `test_offset.py`,
`test_pipeline_integration.py`가 이미 다룬다 — 여기서는 "이 시계열이 주어지면
어떤 단서가 충족되는가"만 본다.
"""

from __future__ import annotations

import pytest

from src.detection.types import VehicleBox
from src.metrics.types import OffsetSample, VehicleTimeSeries
from src.risk.scorer import RiskScorer
from src.tracking.types import TrackedVehicle
from src.utils.config import load_config


@pytest.fixture
def cfg():
    return load_config("configs/dev.yaml")


@pytest.fixture
def scorer(cfg) -> RiskScorer:
    return RiskScorer(cfg)


def _track(track_id: int = 1, tracked_frames: int = 30) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        box=VehicleBox(100, 100, 180, 220, confidence=0.9, class_id=2),
        tracked_frames=tracked_frames,
        is_interpolated=False,
    )


def _series(track_id: int, samples: list[OffsetSample]) -> VehicleTimeSeries:
    series = VehicleTimeSeries(track_id=track_id)
    series.offsets.extend(samples)
    return series


def _flat_samples(n: int, offset_ratio: float, dt: float = 1 / 15, bbox_height: float = 60.0):
    """일정한 offset·속도 0·팽창 없음 — "아무 일도 없는" 기준선."""
    return [
        OffsetSample(
            timestamp=i * dt,
            offset_m=offset_ratio * 3.5,
            offset_ratio=offset_ratio,
            lateral_velocity_mps=0.0,
            lane_confidence=0.9,
            is_interpolated=False,
            bbox_height=bbox_height,
        )
        for i in range(n)
    ]


def test_insufficient_tracking_yields_no_indicators(scorer: RiskScorer):
    track = _track(tracked_frames=5)  # min_tracked_frames(30) 미달
    series = _series(1, _flat_samples(40, offset_ratio=0.4))  # 지표 조건은 다 충족하는 데이터인데도
    result = scorer.assess(track, series, lane_valid=True, timestamp=100.0, ego_speed_kmh=80)
    assert result.types == []
    assert result.score == 0
    assert result.should_emit_event is False


def test_quiet_driving_triggers_nothing(scorer: RiskScorer):
    track = _track()
    series = _series(1, _flat_samples(80, offset_ratio=0.02))  # 차로 중앙 근처, 변화 없음
    result = scorer.assess(track, series, lane_valid=True, timestamp=100.0, ego_speed_kmh=80)
    assert result.types == []
    assert result.level == "low"


def test_lane_departure_requires_min_speed_gating(scorer: RiskScorer, cfg):
    """LDWS 근거 게이팅 — 속도 미상/저속이면 offset이 충분해도 lane_departure 미충족."""
    threshold = cfg.risk.lane_departure.offset_ratio_threshold
    min_duration = cfg.risk.lane_departure.min_duration_sec
    n = int(min_duration * 15) + 10
    series = _series(1, _flat_samples(n, offset_ratio=threshold + 0.1))
    track = _track()

    below_speed = scorer.assess(track, series, lane_valid=True, timestamp=100.0, ego_speed_kmh=40)
    assert "lane_departure" not in below_speed.types

    unknown_speed = scorer.assess(track, series, lane_valid=True, timestamp=100.0, ego_speed_kmh=None)
    assert "lane_departure" not in unknown_speed.types

    above_speed = scorer.assess(track, series, lane_valid=True, timestamp=100.0, ego_speed_kmh=80)
    assert "lane_departure" in above_speed.types


def test_weaving_detected_from_alternating_offset(scorer: RiskScorer, cfg):
    """좌우로 반복 전환하는 offset — 사행 판정."""
    dt = 1 / 15
    amplitude = cfg.risk.weaving.min_amplitude_ratio + 0.1
    samples = []
    sign = 1
    t = 0.0
    for cycle in range(cfg.risk.weaving.min_direction_changes + 2):
        for _ in range(5):
            samples.append(
                OffsetSample(t, sign * amplitude * 3.5, sign * amplitude, 0.0, 0.9, False)
            )
            t += dt
        sign *= -1

    track = _track()
    result = scorer.assess(track, _series(1, samples), lane_valid=True, timestamp=t, ego_speed_kmh=80)
    assert "weaving" in result.types
    assert result.indicators["direction_changes_count"] >= cfg.risk.weaving.min_direction_changes


def test_lane_based_indicators_skipped_when_lane_invalid_this_frame(scorer: RiskScorer, cfg):
    """이번 프레임 차선 인식이 무효면, 과거 이력이 있어도 lane 기반 지표는 판단 보류.

    speed_irregular는 lane offset과 독립적인 신호라 계속 평가된다 —
    risk-criteria.md 1.4 "속도 불규칙은 독립 신호" 근거.
    """
    threshold = cfg.risk.lane_departure.offset_ratio_threshold
    n = int(cfg.risk.lane_departure.min_duration_sec * 15) + 10
    samples = _flat_samples(n, offset_ratio=threshold + 0.1, bbox_height=60.0)
    # 마지막 구간에 급접근(팽창) 이벤트를 심는다.
    dt = 1 / 15
    for i in range(20):
        samples.append(
            OffsetSample(
                timestamp=samples[-1].timestamp + dt,
                offset_m=(threshold + 0.1) * 3.5,
                offset_ratio=threshold + 0.1,
                lateral_velocity_mps=0.0,
                lane_confidence=0.9,
                is_interpolated=False,
                bbox_height=samples[-1].bbox_height * 1.03,  # 프레임마다 3%씩 커짐 → 뚜렷한 팽창
            )
        )

    track = _track()
    series = _series(1, samples)

    result = scorer.assess(track, series, lane_valid=False, timestamp=samples[-1].timestamp, ego_speed_kmh=80)
    assert "lane_departure" not in result.types
    assert "weaving" not in result.types
    assert "speed_irregular" in result.types


def test_frame_quality_failure_suppresses_all_indicators(scorer: RiskScorer, cfg):
    """프레임 품질 미달(블러·밝기 급변) — 다른 조건을 다 충족해도 전부 판단 보류.

    risk-criteria.md 4.1 "프레임 품질" 게이팅. FrameQualityChecker 자체의
    블러·밝기 판정은 tests/test_frame_quality.py가 다루고, 여기서는
    "frame_quality_ok=False가 RiskScorer에 실제로 반영되는가"만 본다.
    """
    threshold = cfg.risk.lane_departure.offset_ratio_threshold
    n = int(cfg.risk.lane_departure.min_duration_sec * 15) + 10
    samples = _flat_samples(n, offset_ratio=threshold + 0.1, bbox_height=60.0)
    dt = 1 / 15
    for i in range(20):
        samples.append(
            OffsetSample(
                timestamp=samples[-1].timestamp + dt,
                offset_m=(threshold + 0.1) * 3.5,
                offset_ratio=threshold + 0.1,
                lateral_velocity_mps=0.0,
                lane_confidence=0.9,
                is_interpolated=False,
                bbox_height=samples[-1].bbox_height * 1.03,
            )
        )
    track = _track()
    series = _series(1, samples)
    t = samples[-1].timestamp

    result = scorer.assess(
        track, series, lane_valid=True, timestamp=t, ego_speed_kmh=80, frame_quality_ok=False
    )
    assert result.types == []
    assert result.score == 0
    assert result.should_emit_event is False


def test_event_fires_once_then_cooldown_suppresses_repeat(scorer: RiskScorer, cfg):
    """3개 지표를 동시 충족(가중치 0.2×3=score 60=start_score)시켜 이벤트 발생을 확인.

    쿨다운 동안 동일 차량의 재발생은 억제된다 (risk-criteria.md 4.3).
    """
    dt = 1 / 15
    threshold = cfg.risk.lane_departure.offset_ratio_threshold
    n = int(cfg.risk.lane_departure.min_duration_sec * 15) + 10
    samples = _flat_samples(n, offset_ratio=threshold + 0.1, bbox_height=60.0)
    for i in range(20):
        samples.append(
            OffsetSample(
                timestamp=samples[-1].timestamp + dt,
                offset_m=(threshold + 0.1) * 3.5,
                offset_ratio=threshold + 0.1,
                lateral_velocity_mps=0.0,
                lane_confidence=0.9,
                is_interpolated=False,
                bbox_height=samples[-1].bbox_height * 1.03,
            )
        )
    track = _track()
    series = _series(1, samples)
    t = samples[-1].timestamp

    first = scorer.assess(track, series, lane_valid=True, timestamp=t, ego_speed_kmh=80)
    assert "lane_departure" in first.types
    assert "speed_irregular" in first.types
    assert first.score >= cfg.risk.hysteresis.start_score
    assert first.should_emit_event is True

    # 활성 상태 유지 중 동일 조건 재평가 — 이미 활성화되어 있으므로 새 이벤트 아님
    second = scorer.assess(track, series, lane_valid=True, timestamp=t + 1.0, ego_speed_kmh=80)
    assert second.should_emit_event is False

    # 쿨다운 중 다른 차량 track_id는 영향받지 않는다
    other_track = _track(track_id=2)
    other = scorer.assess(other_track, series, lane_valid=True, timestamp=t + 1.0, ego_speed_kmh=80)
    assert other.should_emit_event is True


def test_forget_clears_hysteresis_state(scorer: RiskScorer, cfg):
    dt = 1 / 15
    threshold = cfg.risk.lane_departure.offset_ratio_threshold
    n = int(cfg.risk.lane_departure.min_duration_sec * 15) + 10
    samples = _flat_samples(n, offset_ratio=threshold + 0.1, bbox_height=60.0)
    for i in range(20):
        samples.append(
            OffsetSample(
                timestamp=samples[-1].timestamp + dt,
                offset_m=(threshold + 0.1) * 3.5,
                offset_ratio=threshold + 0.1,
                lateral_velocity_mps=0.0,
                lane_confidence=0.9,
                is_interpolated=False,
                bbox_height=samples[-1].bbox_height * 1.03,
            )
        )
    track = _track()
    series = _series(1, samples)
    t = samples[-1].timestamp

    first = scorer.assess(track, series, lane_valid=True, timestamp=t, ego_speed_kmh=80)
    assert first.should_emit_event is True

    scorer.forget(track.track_id)

    # forget 이후에는 다시 "비활성"으로 취급되어, 조건이 같아도 새 이벤트로 잡힌다
    again = scorer.assess(track, series, lane_valid=True, timestamp=t + 1.0, ego_speed_kmh=80)
    assert again.should_emit_event is True
