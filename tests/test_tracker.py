"""VehicleTracker 테스트.

실제 대시캠 영상이 없어(sprint-plan.md S1 데이터 준비 미완) 합성 검출 박스로
검증한다. ByteTrack 자체의 성능이 아니라 **래퍼의 계약**을 확인하는 것이
목적이다 — track_id 유지, 누적 프레임 수, 사라진 차량 정리, 보간 플래그.

ID switching 빈도의 정량 측정은 BDD100K MOT 서브셋으로 별도 수행한다
(sprint-plan.md S3).
"""

from __future__ import annotations

import pytest

from src.detection.types import VehicleBox
from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config


@pytest.fixture
def tracker() -> VehicleTracker:
    # 추적기는 내부에 상태를 쌓으므로 테스트마다 새로 만든다
    return VehicleTracker(load_config("configs/dev.yaml"))


def _box(center_x: float, center_y: float = 500, w: float = 80, h: float = 60,
         conf: float = 0.9) -> VehicleBox:
    return VehicleBox(
        x1=center_x - w / 2, y1=center_y - h / 2,
        x2=center_x + w / 2, y2=center_y + h / 2,
        confidence=conf, class_id=2,
    )


def _run(tracker: VehicleTracker, positions, frames: int):
    """positions(frame_index -> box 목록)로 여러 프레임을 돌린다."""
    last = []
    for i in range(frames):
        last = tracker.update(positions(i))
    return last


def test_track_id_persists_while_vehicle_visible(tracker: VehicleTracker):
    """차량이 화면에 있는 동안 같은 track_id가 유지된다 (S3 완료 조건)."""
    ids = set()
    for i in range(15):
        for tv in tracker.update([_box(300 + i * 12)]):
            ids.add(tv.track_id)
    assert len(ids) == 1, f"track_id가 바뀌었다: {ids}"


def test_tracked_frames_accumulates(tracker: VehicleTracker):
    """관측 충분성 판단의 근거가 되는 누적 프레임 수가 증가한다."""
    counts = []
    for i in range(12):
        tracks = tracker.update([_box(300 + i * 10)])
        if tracks:
            counts.append(tracks[0].tracked_frames)
    assert counts == sorted(counts)
    assert counts[-1] >= 10


def test_two_vehicles_get_distinct_ids(tracker: VehicleTracker):
    last = _run(tracker, lambda i: [_box(200 + i * 8), _box(500 - i * 8)], frames=12)
    assert len({tv.track_id for tv in last}) == 2


def test_disappeared_vehicle_is_forgotten(tracker: VehicleTracker):
    """사라진 차량의 누적 카운터가 정리되어 메모리가 무한정 늘지 않는다.

    ByteTrack은 `track_buffer` 프레임 동안 lost track을 살려두므로, 그 기간을
    넘길 만큼 충분히 돌린 뒤 확인한다.
    """
    _run(tracker, lambda i: [_box(300 + i * 10)], frames=10)
    assert len(tracker._tracked_frames) == 1

    buffer_frames = load_config("configs/dev.yaml").tracking.track_buffer
    for _ in range(buffer_frames + 5):
        tracker.update([])
    assert tracker._tracked_frames == {}


def test_interpolate_holds_last_position_and_flags_it(tracker: VehicleTracker):
    """검출을 건너뛴 프레임은 위치를 유지하되 보간 플래그가 선다.

    ⚠️ 위치를 칼만 예측으로 전진시키지 않는 것은 현재 구현의 한계다 —
    검출 주기 분리는 S6 항목이며, 그때 이 테스트도 함께 바뀌어야 한다
    (`VehicleTracker.interpolate` docstring 참고).
    """
    tracked = _run(tracker, lambda i: [_box(300 + i * 10)], frames=8)
    interpolated = tracker.interpolate()

    assert [tv.track_id for tv in interpolated] == [tv.track_id for tv in tracked]
    assert all(tv.is_interpolated for tv in interpolated)
    assert interpolated[0].center_x == pytest.approx(tracked[0].center_x)
    assert not any(tv.is_interpolated for tv in tracked)


def test_empty_detections_produce_no_tracks(tracker: VehicleTracker):
    assert tracker.update([]) == []
