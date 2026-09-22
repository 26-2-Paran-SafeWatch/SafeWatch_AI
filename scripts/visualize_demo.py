"""검출·차선 인식 결과를 영상/이미지 위에 그려서 확인한다.

디버깅·시연용 스크립트다. 파이프라인 자체(src/main.py)는 시각화를 하지
않고 로그만 남기지만, 이 스크립트는 결과를 눈으로 확인할 수 있게
bounding box와 차선 곡선을 원본 프레임에 그려 저장한다.

사용법
    python scripts/visualize_demo.py --config configs/dev.yaml \
        --source data/raw/sample.mp4 --output data/processed/demo.mp4

    # 이미지 한 장만 확인할 때
    python scripts/visualize_demo.py --config configs/dev.yaml \
        --source data/raw/sample.jpg --output data/processed/demo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import VehicleDetector
from src.detection.types import VehicleBox
from src.lane.lane_detector import LaneDetector
from src.lane.types import LaneModel
from src.utils.config import load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="검출·차선 인식 결과 시각화")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument("--source", required=True, help="영상 또는 이미지 파일 경로")
    parser.add_argument("--output", required=True, help="결과 저장 경로")
    return parser.parse_args()


def draw_vehicle_boxes(image: np.ndarray, boxes: list[VehicleBox]) -> None:
    for box in boxes:
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        label = f"{box.confidence:.2f}"
        cv2.putText(
            image, label, (p1[0], max(0, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )


def draw_lane(image: np.ndarray, lane: LaneModel, lane_detector: LaneDetector) -> None:
    """LaneModel의 bird's-eye-view 다항식 곡선을 원본 프레임에 역변환해 그린다."""
    if lane.left_fit is None or lane.right_fit is None:
        return

    warp_height = lane_detector.perspective._warp_size[1]  # noqa: SLF001 — 시각화 전용 접근
    ys = np.linspace(0, warp_height - 1, 30)

    color = (0, 255, 255) if lane.valid else (0, 0, 255)  # 저신뢰/판단보류는 빨간색으로 구분

    for fit in (lane.left_fit, lane.right_fit):
        xs = np.polyval(fit, ys)
        bev_points = np.stack([xs, ys], axis=1)
        try:
            orig_points = lane_detector.perspective.unwarp_points(bev_points)
        except RuntimeError:
            return
        pts = orig_points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=4)

    status = f"lane valid={lane.valid} conf={lane.confidence:.2f}"
    cv2.putText(image, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 잔차 비율 — 신뢰도가 왜 그렇게 나왔는지 바로 보이게 함께 표시한다.
    # 상한을 넘으면 선이 아니라 노면 텍스처를 피팅한 것이다 (lane.confidence.max_residual_ratio)
    if lane.fit_residual_ratio is not None:
        residual = f"fit residual={lane.fit_residual_ratio:.3f} (of lane width)"
        cv2.putText(image, residual, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def process_frame(
    image: np.ndarray, detector: VehicleDetector, lane_detector: LaneDetector
) -> np.ndarray:
    annotated = image.copy()
    boxes = detector.detect(image)
    lane = lane_detector.detect(image)
    draw_vehicle_boxes(annotated, boxes)
    draw_lane(annotated, lane, lane_detector)
    return annotated


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    detector = VehicleDetector(cfg)
    lane_detector = LaneDetector(cfg)

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {source_path}")
        annotated = process_frame(image, detector, lane_detector)
        cv2.imwrite(str(output_path), annotated)
        print(f"저장됨: {output_path}")
        return

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(process_frame(frame, detector, lane_detector))
        frame_count += 1

    cap.release()
    writer.release()
    print(f"저장됨: {output_path} ({frame_count} 프레임)")


if __name__ == "__main__":
    main()
