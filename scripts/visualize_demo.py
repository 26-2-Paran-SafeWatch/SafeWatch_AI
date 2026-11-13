"""검출·차선 인식 결과를 영상/이미지 위에 그려서 확인한다.

디버깅·시연용 스크립트다. 파이프라인 자체(src/main.py)는 시각화를 하지
않고 로그만 남기지만, 이 스크립트는 결과를 눈으로 확인할 수 있게
bounding box와 차선 곡선을 원본 프레임에 그려 저장한다.

사용법 — `src.*`를 import하므로 반드시 `-m`으로 실행한다(2026-10-01 확인:
직접 `python scripts/visualize_demo.py`로 실행하면 `ModuleNotFoundError: No
module named 'src'`가 난다 — `src/main.py`·`scripts/benchmark.py`와 같은 이유).
    python -m scripts.visualize_demo --config configs/dev.yaml \
        --source data/raw/sample.mp4 --output data/processed/demo.mp4

    # 이미지 한 장만 확인할 때
    python -m scripts.visualize_demo --config configs/dev.yaml \
        --source data/raw/sample.jpg --output data/processed/demo.jpg
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import VehicleDetector
from src.detection.types import VehicleBox
from src.lane import create_lane_detector
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


def draw_lane(image: np.ndarray, lane: LaneModel, lane_detector) -> None:
    """차선을 원본 프레임에 그린다. opencv 경로는 조감도 다항식을 역변환하고,
    ufld 경로(space="image")는 원본 화면 점과 측정 구간을 그대로 그린다."""
    if lane.space == "image":
        _draw_lane_image(image, lane)
        return
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
    # 좌우 선 모양이 서로 다를수록 커진다 — 기대 폭의 lane.confidence.max_rms_ratio에서 잔차 항이 0
    if lane.fit_residual_ratio is not None:
        residual = f"fit residual={lane.fit_residual_ratio:.3f} (of lane width)"
        cv2.putText(image, residual, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_lane_image(image: np.ndarray, lane: LaneModel) -> None:
    color = (0, 255, 255) if lane.valid else (0, 0, 255)
    for points in (lane.left_points, lane.right_points):
        if points is not None and len(points) >= 2:
            cv2.polylines(image, [points.astype(np.int32).reshape(-1, 1, 2)], False, color, 4)
    if lane.measure_y_range is not None:
        for y in lane.measure_y_range:      # offset을 재는 구간 (lane.ufld.measure_range_ratio)
            cv2.line(image, (0, int(y)), (image.shape[1], int(y)), (255, 200, 0), 1)
    status = f"lane valid={lane.valid} conf={lane.confidence:.2f} reason={lane.reject_reason}"
    cv2.putText(image, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def process_frame(
    image: np.ndarray, detector: VehicleDetector, lane_detector, timestamp: float | None = None
) -> tuple[np.ndarray, object]:
    annotated = image.copy()
    boxes = detector.detect(image)
    lane = lane_detector.detect(image, vehicle_boxes=boxes, timestamp=timestamp)
    draw_vehicle_boxes(annotated, boxes)
    draw_lane(annotated, lane, lane_detector)
    return annotated, lane


# 보류 이유 설명 (opencv 경로의 이유 문자열은 그대로 출력)
REASON_TEXT = {
    "existence": "차선이 측정 구간에 충분히 안 나옴",
    "geometry": "차선 모양 이상(교차·순서 뒤바뀜)",
    "straddle": "화면 가운데가 예측 차로 밖",
    "width_calibrating": "차로 폭 기록 모으는 중(시작 직후 정상)",
    "width": "차로 폭이 최근 기준보다 급변",
    "temporal": "직전 프레임 대비 급변",
}


def print_lane_summary(lanes: list, width: int, height: int, method: str) -> None:
    """영상 전체의 차선 통과·보류 요약 — 모델이 일을 하는지 숫자로 본다."""
    n = len(lanes)
    if n == 0:
        return
    valid = [i for i, ln in enumerate(lanes) if ln.valid]
    reasons = Counter(ln.reject_reason or "기타" for ln in lanes if not ln.valid)
    calib = reasons.get("width_calibrating", 0)
    print(f"\n── 차선 요약 ({width}×{height}, {n}프레임) ──")
    if method == "ufld" and abs(width / max(height, 1) - 16 / 9) > 0.05:
        print(f"⚠️ 16:9가 아님 — UFLD는 16:9 블랙박스 화면으로 학습했다. 늘어난 화면이라 결과를 믿기 어렵다")
    print(f"  통과(차선 잡음): {len(valid)}/{n} ({len(valid) / n:.0%})")
    if 0 < calib < n:
        print(f"  폭 보정 끝난 뒤 통과: {len(valid)}/{n - calib} ({len(valid) / (n - calib):.0%})"
              f"  ← 시작 직후 폭 기록 모으는 {calib}프레임 제외")
    for r, c in reasons.most_common():
        print(f"  보류 {r:<18} {c:>4}  {REASON_TEXT.get(r, '')}")
    confs = [ln.confidence for ln in lanes]
    print(f"  신뢰도 중앙값 {float(np.median(confs)):.2f} · 통과 프레임 번호 {valid[:5]}{' …' if len(valid) > 5 else ''}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    detector = VehicleDetector(cfg)
    lane_detector = create_lane_detector(cfg)   # lane.method: opencv | ufld

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {source_path}")
        annotated, lane = process_frame(image, detector, lane_detector)
        print(f"lane valid={lane.valid} conf={lane.confidence:.2f} reason={lane.reject_reason}")
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
    lanes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, lane = process_frame(frame, detector, lane_detector, timestamp=frame_count / fps)
        writer.write(annotated)
        lanes.append(lane)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"저장됨: {output_path} ({frame_count} 프레임)")
    print_lane_summary(lanes, width, height, cfg.lane.method)


if __name__ == "__main__":
    main()
