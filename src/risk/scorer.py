"""음주운전 의심 점수 산출. docs/pipeline-architecture.md 3.6, docs/risk-criteria.md 참고.

**채택 단서 5종** (risk-criteria.md 1.4) — 차선 걸침·사행·스웨빙·표류(모두
lane offset 시계열 기반)와 속도 불규칙(bbox 팽창률 기반, 독립 신호).

**처리 순서**
1. 입력 신뢰도 게이팅 — 구현된 것만 (아래 "구현되지 않은 게이팅" 참고)
2. 지표별 충족 여부 판정
3. `min_indicators` 결합 판정 + 가중 합산 점수
4. 히스테리시스 + 쿨다운으로 이벤트 발생 여부 결정

**구현되지 않은 게이팅 (risk-criteria.md 4.1)** — 프레임 품질(블러·밝기 급변)
검사와 유효 관측 거리(bbox 면적) 검사는 이 모듈에 아직 없다. 둘 다
`docs/risk-criteria.md`에서도 여전히 `[TBD]`라 임의로 수치를 만들어 넣지
않았다 — 근거 없는 임계값을 코드에 박아넣지 않는다는 원칙(AGENTS.md 3)에
따른 것이다.

**알려진 단순화**
- 스웨빙·표류 판정은 [TBD] 임계값을 전제로 한 1차 근사 로직이다. 정확한
  패턴 정의는 실차 데이터 확보 후 재검토 대상이다 (아래 각 함수 docstring).
- `speed_irregular`는 현재 `lane.valid`일 때만 기록되는 bbox_height에
  의존해, 설계 의도(차선 실패와 무관하게 동작)가 완전히 실현되지 않는다
  (`src/metrics/types.py`의 `OffsetSample.bbox_height` docstring 참고).
"""

from __future__ import annotations

import logging

from src.metrics.types import VehicleTimeSeries
from src.risk.types import RiskAssessment
from src.tracking.types import TrackedVehicle
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.risk")


class RiskScorer:
    def __init__(self, cfg: Config):
        self._cfg = cfg.risk
        self._min_tracked_frames: int = cfg.tracking.min_tracked_frames
        self._fps: float = cfg.input.fps

        self._weights: dict[str, float] = {
            "lane_departure": self._cfg.weights.lane_departure,
            "weaving": self._cfg.weights.weaving,
            "swerving": self._cfg.weights.swerving,
            "drifting": self._cfg.weights.drifting,
            "speed_irregular": self._cfg.weights.speed_irregular,
        }

        # 차량별 히스테리시스 활성 상태 + 마지막 이벤트 시각(쿨다운용)
        self._active: dict[int, bool] = {}
        self._last_event_ts: dict[int, float] = {}

    def assess(
        self,
        track: TrackedVehicle,
        series: VehicleTimeSeries,
        lane_valid: bool,
        timestamp: float,
        ego_speed_kmh: float | None = None,
    ) -> RiskAssessment:
        """차량 한 대의 이번 프레임 위험 판단 결과를 낸다.

        Args:
            lane_valid: 이번 프레임의 차선 인식 유효 여부. False면 lane
                offset 기반 4개 단서(차선걸침·사행·스웨빙·표류)를 전부
                판단 보류한다 (risk-criteria.md 3.3 폴백 경로).
            ego_speed_kmh: 자차 GPS 속도. None(미수신)이면 속도 게이팅이
                필요한 지표(차선 걸침)는 안전하게 판단 보류한다 — 신뢰할
                수 없는 입력으로 판단하지 않는다는 원칙(risk-criteria.md 4.1).
        """
        with profiler.stage("risk"):
            indicators: dict[str, float] = {}
            matched: list[str] = []

            # --- 1. 입력 신뢰도 게이팅 (구현된 부분만) ---
            sufficiently_observed = track.tracked_frames >= self._min_tracked_frames

            if sufficiently_observed and lane_valid:
                if self._check_lane_departure(series, ego_speed_kmh, indicators):
                    matched.append("lane_departure")
                if self._check_weaving(series, indicators):
                    matched.append("weaving")
                if self._check_swerving(series, indicators):
                    matched.append("swerving")
                if self._check_drifting(series, indicators):
                    matched.append("drifting")

            if sufficiently_observed and self._check_speed_irregular(series, indicators):
                matched.append("speed_irregular")

            # --- 2~3. 결합 판정 + 가중 합산 ---
            score = round(100 * sum(self._weights[t] for t in matched))
            should_emit = self._update_hysteresis(
                track.track_id, score, len(matched) >= self._cfg.min_indicators, timestamp
            )

        return RiskAssessment(
            track_id=track.track_id,
            score=score,
            level=self._level(score),
            types=matched,
            indicators=indicators,
            should_emit_event=should_emit,
        )

    def forget(self, track_id: int) -> None:
        """화면에서 사라진 차량의 히스테리시스 상태를 해제한다."""
        self._active.pop(track_id, None)
        self._last_event_ts.pop(track_id, None)

    def sync(self, alive_track_ids: set[int]) -> None:
        """이번 프레임에 살아있는 track_id 집합을 기준으로 사라진 차량 상태를 정리한다.

        `OffsetCalculator._forget_disappeared`와 같은 목적 — 호출부(main.py)가
        매 프레임 살아있는 track_id 집합을 넘기면 된다.
        """
        tracked_ids = set(self._active) | set(self._last_event_ts)
        for track_id in tracked_ids - alive_track_ids:
            self.forget(track_id)

    # --- 지표별 판정 ---

    def _check_lane_departure(
        self, series: VehicleTimeSeries, ego_speed_kmh: float | None, indicators: dict
    ) -> bool:
        cfg = self._cfg.lane_departure
        if ego_speed_kmh is None or ego_speed_kmh < cfg.min_speed_kmh:
            # LDWS 규정 근거 게이팅 (risk-criteria.md 1.3, 4.1). 속도를 모르면
            # 게이팅 조건 충족 여부를 확인할 수 없으므로 안전하게 미충족 처리한다.
            return False

        indicators["lane_offset_max_ratio"] = series.max_offset_ratio
        duration = series.departure_duration_sec(cfg.offset_ratio_threshold)
        indicators["lane_departure_duration_sec"] = duration
        return duration >= cfg.min_duration_sec

    def _check_weaving(self, series: VehicleTimeSeries, indicators: dict) -> bool:
        cfg = self._cfg.weaving
        count = series.direction_changes(cfg.min_amplitude_ratio)
        indicators["direction_changes_count"] = count
        return count >= cfg.min_direction_changes

    def _check_swerving(self, series: VehicleTimeSeries, indicators: dict) -> bool:
        """급격한 횡이동 후 원위치 복귀 (risk-criteria.md 1.4 swerving).

        **1차 근사** — 관측 구간 내 횡방향 속도 최댓값이 임계값을 넘었고,
        그 시점으로부터 `recovery_window_sec` 이내인 최신 샘플에서 횡방향
        속도가 다시 임계값 아래로 내려왔으면 스웨빙으로 본다. "복귀"를
        엄밀하게 판정하려면(예: offset이 원래 위치로 돌아왔는지) 실차
        데이터로 패턴을 더 봐야 한다 — 지금은 속도만 본다.
        """
        cfg = self._cfg.swerving
        indicators["lateral_velocity_peak_mps"] = series.peak_lateral_velocity_mps

        if not series.offsets or series.peak_lateral_velocity_mps < cfg.lateral_velocity_threshold_mps:
            return False

        peak_sample = max(series.offsets, key=lambda s: abs(s.lateral_velocity_mps))
        latest = series.offsets[-1]
        if peak_sample is latest:
            return False  # 아직 정점 — 복귀 여부를 판단할 다음 샘플이 없다

        recovered = abs(latest.lateral_velocity_mps) < cfg.lateral_velocity_threshold_mps
        within_window = (latest.timestamp - peak_sample.timestamp) <= cfg.recovery_window_sec
        return recovered and within_window

    def _check_drifting(self, series: VehicleTimeSeries, indicators: dict) -> bool:
        """한 방향으로 느린 횡이동 지속 (risk-criteria.md 1.4 drifting).

        **1차 근사** — 최근 `min_duration_sec` 구간 전체에서 횡방향 속도가
        `lateral_velocity_max_mps` 이하(= "느림")이고 offset 부호가 유지된
        채로 `offset_ratio_threshold`에 도달했으면 표류로 본다.
        """
        cfg = self._cfg.drifting
        if not series.offsets:
            return False

        latest = series.offsets[-1]
        indicators["drift_duration_sec"] = 0.0
        if abs(latest.offset_ratio) < cfg.offset_ratio_threshold:
            return False

        cutoff = latest.timestamp - cfg.min_duration_sec
        recent = [s for s in series.offsets if s.timestamp >= cutoff]
        if len(recent) < 2 or (recent[-1].timestamp - recent[0].timestamp) < cfg.min_duration_sec * 0.8:
            return False  # 관측 구간이 아직 min_duration_sec만큼 안 쌓임

        sign = 1 if latest.offset_ratio > 0 else -1
        same_direction = all(s.offset_ratio * sign >= 0 for s in recent)
        slow = all(abs(s.lateral_velocity_mps) <= cfg.lateral_velocity_max_mps for s in recent)

        indicators["drift_duration_sec"] = recent[-1].timestamp - recent[0].timestamp
        return same_direction and slow

    def _check_speed_irregular(self, series: VehicleTimeSeries, indicators: dict) -> bool:
        cfg = self._cfg.speed_irregular
        window = max(2, round(cfg.window_sec * self._fps))
        rate = series.peak_expansion_rate(window)
        if rate is None:
            return False
        indicators["expansion_rate_peak_per_sec"] = rate
        return rate >= cfg.expansion_rate_threshold_per_sec

    # --- 결합·안정화 ---

    def _level(self, score: int) -> str:
        thresholds = self._cfg.level_thresholds
        if score >= thresholds.high:
            return "high"
        if score >= thresholds.medium:
            return "medium"
        return "low"

    def _update_hysteresis(
        self, track_id: int, score: int, meets_min_indicators: bool, timestamp: float
    ) -> bool:
        """히스테리시스 + 쿨다운을 적용해 "이번이 이벤트 발생 시점인지"를 낸다.

        이벤트는 비활성→활성 전이 시점에만 True다 — 활성 상태가 계속
        유지되는 동안 매 프레임 이벤트를 내지 않는다 (쿨다운의 취지와
        같다: 동일 상황에 대한 중복 이벤트 억제, risk-criteria.md 4.3).
        """
        cfg = self._cfg.hysteresis
        was_active = self._active.get(track_id, False)

        if was_active and score < cfg.end_score:
            self._active[track_id] = False
            return False

        if not was_active and meets_min_indicators and score >= cfg.start_score:
            last_ts = self._last_event_ts.get(track_id)
            in_cooldown = last_ts is not None and (timestamp - last_ts) < self._cfg.cooldown_sec
            if in_cooldown:
                return False
            self._active[track_id] = True
            self._last_event_ts[track_id] = timestamp
            return True

        return False
