"""차선 인식 모듈. `lane.method`로 OpenCV(조감도) / UFLD(딥러닝, 원본 화면) 경로를 고른다."""

from __future__ import annotations

from src.utils.config import Config

LANE_METHODS = ("opencv", "ufld")


def create_lane_detector(cfg: Config):
    """설정에 맞는 차선 인식기. 둘 다 `detect(image, vehicle_boxes=, timestamp=) -> LaneModel`과
    `perspective`(opencv = 조감도 변환기, ufld = None) 속성을 갖는다."""
    method = cfg.lane.method if "method" in cfg.lane else "opencv"
    if method == "opencv":
        from src.lane.lane_detector import LaneDetector

        return LaneDetector(cfg)
    if method == "ufld":
        from src.lane.ufld_detector import UFLDLaneDetector

        return UFLDLaneDetector(cfg)
    raise ValueError(f"lane.method '{method}'를 모릅니다 — {LANE_METHODS} 중 하나")
