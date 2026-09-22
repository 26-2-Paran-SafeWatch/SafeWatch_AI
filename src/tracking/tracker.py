"""ByteTrack 기반 차량 추적. docs/pipeline-architecture.md 3.4 참고.

ultralytics 내장 `BYTETracker`를 그대로 쓰되, `model.track()`이 아니라
검출 결과(`VehicleBox[]`)를 받아 추적하는 형태로 감싼다. 이렇게 하는 이유는
두 가지다.

1. **모듈 경계** — `model.track()`은 검출과 추적을 한 번에 수행해 DETECTION과
   TRACKING의 경계를 없앤다 (pipeline-architecture.md 1.1).
2. **검출 주기 분리** — S6에서 N프레임마다 검출하고 사이를 추적으로 메우려면
   검출과 추적을 따로 호출할 수 있어야 한다 (sprint-plan.md S6).

ByteTrack은 딥러닝 모델이 아니라 IoU·칼만 필터 기반 매칭이라 추가 추론 비용이
없다 (AGENTS.md "딥러닝 모델 추가 금지" 원칙과 무관).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from src.detection.types import VehicleBox
from src.tracking.types import TrackedVehicle
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.tracking")


class _DetectionResults:
    """`BYTETracker.update()`가 기대하는 Results-like 어댑터.

    ultralytics는 자사 `Boxes` 객체를 전제로 `conf` / `xywh` / `cls` 속성과
    불리언 마스크 인덱싱을 사용한다. `VehicleBox` 목록을 그 형태로 맞춘다.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_boxes(cls, boxes: list[VehicleBox]) -> "_DetectionResults":
        if not boxes:
            return cls(np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.float32))
        xywh = np.array(
            [
                [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x2 - b.x1, b.y2 - b.y1]
                for b in boxes
            ],
            dtype=np.float32,
        )
        conf = np.array([b.confidence for b in boxes], dtype=np.float32)
        class_ids = np.array([b.class_id for b in boxes], dtype=np.float32)
        return cls(xywh, conf, class_ids)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_DetectionResults":
        return _DetectionResults(self.xywh[mask], self.conf[mask], self.cls[mask])


class VehicleTracker:
    def __init__(self, cfg: Config):
        # ultralytics import 비용이 커서 생성 시점에만 로드한다 (detector.py와 동일).
        from ultralytics.trackers.byte_tracker import BYTETracker

        track_cfg = cfg.tracking
        self._min_tracked_frames: int = track_cfg.min_tracked_frames

        # BYTETracker는 설정을 속성 접근으로 읽으므로 네임스페이스로 넘긴다.
        self._tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=track_cfg.track_high_thresh,
                track_low_thresh=track_cfg.track_low_thresh,
                new_track_thresh=track_cfg.new_track_thresh,
                track_buffer=track_cfg.track_buffer,
                match_thresh=track_cfg.match_thresh,
                fuse_score=track_cfg.fuse_score,
            )
        )
        self._tracked_frames: dict[int, int] = {}
        self._last_result: list[TrackedVehicle] = []

    @property
    def min_tracked_frames(self) -> int:
        """관측 충분성 기준. 게이팅 자체는 RISK가 수행한다 (risk-criteria.md 4.1)."""
        return self._min_tracked_frames

    def update(self, boxes: list[VehicleBox]) -> list[TrackedVehicle]:
        """검출 결과에 track_id를 부여한다. 검출을 수행한 프레임에서만 호출한다."""
        with profiler.stage("tracking"):
            tracks = self._tracker.update(_DetectionResults.from_boxes(boxes))
            result: list[TrackedVehicle] = []
            seen: set[int] = set()

            for row in tracks:
                # row = [x1, y1, x2, y2, track_id, score, cls, det_idx]
                x1, y1, x2, y2 = (float(v) for v in row[:4])
                track_id = int(row[4])
                seen.add(track_id)
                self._tracked_frames[track_id] = self._tracked_frames.get(track_id, 0) + 1

                result.append(
                    TrackedVehicle(
                        track_id=track_id,
                        # 좌표는 칼만 필터로 평활화된 추적 위치다. 원본 검출 박스보다
                        # 프레임 간 흔들림이 적어 lane offset 시계열에 유리하다.
                        box=VehicleBox(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=float(row[5]),
                            class_id=int(row[6]),
                        ),
                        tracked_frames=self._tracked_frames[track_id],
                        is_interpolated=False,
                    )
                )

            self._forget_disappeared(seen)
            self._last_result = result
            return result

    def interpolate(self) -> list[TrackedVehicle]:
        """검출을 건너뛴 프레임의 추적 결과.

        **현재는 직전 위치를 그대로 유지한다** — 칼만 예측으로 위치를 전진시키지
        않는다. 검출 주기 분리는 S6 항목이고(sprint-plan.md S6), 그 전까지
        `detection.interval`은 dev 기준 1이라 이 경로를 타지 않는다.

        ⚠️ S6에서 이 메서드를 실제로 쓸 때 해결해야 할 문제 — ByteTrack의 칼만
        필터는 `update()` 1회당 1프레임 경과를 가정한다. N프레임마다 검출하면
        실제로는 N프레임이 지났는데 1프레임만 예측하게 되어 속도 추정이 1/N로
        축소된다. 횡방향 속도(risk-criteria.md 2.6)가 이 값에 의존하므로 그냥
        둘 수 없다. 자세한 내용은 pipeline-architecture.md 5장 미결 사항 참고.
        """
        return [
            TrackedVehicle(
                track_id=tv.track_id,
                box=tv.box,
                tracked_frames=tv.tracked_frames,
                is_interpolated=True,
            )
            for tv in self._last_result
        ]

    def _forget_disappeared(self, seen: set[int]) -> None:
        """화면에서 사라진 track의 누적 카운터를 정리한다.

        ByteTrack은 `track_buffer` 프레임 동안 lost track을 살려두고 재매칭하므로,
        그 사이에 돌아온 차량은 같은 track_id로 이어진다. 버퍼를 넘겨 완전히
        사라진 track만 정리 대상이다.
        """
        alive = seen | {t.track_id for t in self._tracker.lost_stracks}
        for track_id in list(self._tracked_frames):
            if track_id not in alive:
                del self._tracked_frames[track_id]
