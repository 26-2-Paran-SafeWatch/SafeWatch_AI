"""영상 입력 타임스탬프 — 파일은 영상 속 시간을 따라야 한다 (처리 속도와 무관)."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from src.input.source import VideoFileSource


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8, np.uint8))
    w.release()
    return str(path)


def test_file_timestamps_follow_video_time_not_processing_time(video):
    src = VideoFileSource(video)
    ts = []
    for f in src.frames():
        ts.append(f.timestamp)
        if f.frame_id < 3:
            time.sleep(0.2)                      # 처리가 느려도 영상 시간은 그대로
    src.close()
    assert len(ts) == 30
    assert np.allclose(np.diff(ts), 1 / 15, atol=1e-3)
    assert abs(ts[0] - time.time()) < 5          # epoch 기준 (이벤트 ID·시각에 쓰임)


def test_stuck_position_falls_back_to_frame_count(video):
    src = VideoFileSource(video)
    real = src._cap

    class StuckPosition:                         # 재생 위치를 못 주는 코덱 흉내
        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_POS_MSEC else real.get(prop)

        def __getattr__(self, name):
            return getattr(real, name)

    src._cap = StuckPosition()
    ts = [f.timestamp for f in src.frames()]
    src.close()
    assert np.all(np.diff(ts) > 0)
    assert ts[-1] - ts[0] == pytest.approx(29 / 15, abs=1e-3)
