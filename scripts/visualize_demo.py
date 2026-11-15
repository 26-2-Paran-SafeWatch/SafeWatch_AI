"""검출·차선 인식 결과를 화면 위에 그려서 눈으로 확인한다 (디버깅·실차 확인용).

파이프라인 자체(src/main.py)는 시각화를 하지 않고 로그만 남긴다. 이 스크립트는
차량 박스와 차선을 프레임에 그려 **창에 띄우거나(--show) 영상으로 저장(--output)**한다.

그리는 것
- 차량: 판정 대상(화면 중앙 영역 = detection.center_region_ratio) 박스는 자홍색, 신뢰도 표시.
  세로 점선 두 줄이 그 중앙 영역 경계다. 경계 밖 차량은 파이프라인이 버리므로 그리지 않는다.
- 차선(ufld): 자차 왼쪽 선 = 빨간 점, 오른쪽 선 = 초록 점. 판단 보류(valid=False)면 회색 점.
  하늘색 가로줄 두 개 = offset을 재는 구간(lane.ufld.measure_range_ratio).
- 차선(opencv): 조감도 다항식을 원본 화면으로 되돌려 선으로 그린다.
- 위쪽 상태 줄: 처리 fps, 단계별 시간(ms), 차량 수, 차선 상태·보류 이유.

`src.*`를 import하므로 반드시 `-m`으로 실행한다(직접 실행하면 ModuleNotFoundError).

    # Pi5 실차 — 카메라 화면을 실시간으로 띄우고, 그린 영상과 원본을 함께 저장
    #   (창은 Pi에 모니터·VNC가 있어야 뜬다. SSH만이면 --show 빼고 저장만)
    python -m scripts.visualize_demo --config configs/pi5_ufld.yaml --source camera \
        --show --output data/processed/drive_demo.mp4 --save-raw data/raw/drive_raw.mp4

    # 녹화해 둔 영상 / 이미지 한 장
    python -m scripts.visualize_demo --config configs/pi5_ufld.yaml \
        --source data/raw/cam_test.mp4 --output data/processed/cam_test_demo.mp4
    python -m scripts.visualize_demo --config configs/pi5_ufld.yaml \
        --source data/raw/sample.jpg --output data/processed/demo.jpg

창에서 q 또는 Esc로 끝낸다. Ctrl+C로 끊어도 저장 중인 영상은 닫고 끝낸다.
카메라 입력일 때 저장 영상은 실제로 처리한 프레임만 담는다(CPU면 초당 1~2장) — 재생 속도는
`--out-fps`로 정한다. 원본(--save-raw)도 처리한 프레임만 저장되며, 차량 안 PC 밖으로 내보낼 때는
비식별화 원칙(AGENTS.md 5)을 따른다.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import VehicleDetector
from src.detection.types import VehicleBox
from src.input.source import create_source
from src.lane import create_lane_detector
from src.lane.types import LaneModel
from src.utils.config import load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
WINDOW = "SafeWatch demo"

# BGR
RED, GREEN, GRAY, YELLOW, CYAN, WHITE = (0, 0, 255), (0, 220, 0), (150, 150, 150), (0, 255, 255), (255, 200, 0), (255, 255, 255)
MAGENTA = (255, 0, 255)   # 차량 박스 — 차선 빨강·초록과 구분


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="검출·차선 인식 결과 시각화")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument("--source", required=True, help='영상·이미지 파일 경로, 웹캠 번호, 또는 "camera"(Pi 카메라)')
    parser.add_argument("--output", help="그린 결과 저장 경로 (.mp4 / .jpg)")
    parser.add_argument("--show", action="store_true", help="결과를 창에 띄운다 (모니터·VNC 필요)")
    parser.add_argument("--save-raw", help="원본 프레임 영상 저장 경로 — 나중에 다시 돌려보거나 학습 데이터로 쓰기 위함")
    parser.add_argument("--out-fps", type=float, help="저장 영상 재생 fps (기본: 파일 입력은 원본 fps, 카메라는 5)")
    parser.add_argument("--max-frames", type=int, default=0, help="이만큼 처리하고 끝낸다 (0 = 끝까지)")
    return parser.parse_args()


def draw_vehicle_boxes(image: np.ndarray, boxes: list[VehicleBox]) -> None:
    scale = max(0.6, image.shape[1] / 1600)
    for box in boxes:
        p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        cv2.rectangle(image, p1, p2, MAGENTA, 2)
        _text(image, f"{box.confidence:.2f}", (p1[0], max(20, p1[1] - 6)), scale, MAGENTA)


def draw_center_region(image: np.ndarray, ratio: float) -> None:
    """판정 대상 중앙 영역 경계 (detector._filter_forward_vehicles와 같은 식)."""
    h, w = image.shape[:2]
    margin = w * (1 - ratio) / 2
    for x in (int(margin), int(w - margin)):
        for y in range(0, h, 20):
            cv2.line(image, (x, y), (x, min(h, y + 10)), WHITE, 1)


def draw_lane(image: np.ndarray, lane: LaneModel, lane_detector) -> None:
    """ufld 경로(space="image")는 원본 화면 점을, opencv 경로는 조감도 다항식을 역변환해 그린다."""
    if lane.space == "image":
        _draw_lane_image(image, lane)
    else:
        _draw_lane_bev(image, lane, lane_detector)


def _draw_lane_image(image: np.ndarray, lane: LaneModel) -> None:
    if lane.measure_y_range is not None:
        for y in lane.measure_y_range:      # offset을 재는 구간 (lane.ufld.measure_range_ratio)
            cv2.line(image, (0, int(y)), (image.shape[1], int(y)), CYAN, 1)
    for points, color in ((lane.left_points, RED), (lane.right_points, GREEN)):
        if points is None or len(points) == 0:
            continue
        c = color if lane.valid else GRAY
        pts = points.astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(image, [pts.reshape(-1, 1, 2)], False, c, 1)
        for x, y in pts:
            cv2.circle(image, (int(x), int(y)), 4, c, -1)


def _draw_lane_bev(image: np.ndarray, lane: LaneModel, lane_detector) -> None:
    if lane.left_fit is None or lane.right_fit is None:
        return
    warp_height = lane_detector.perspective._warp_size[1]  # noqa: SLF001 — 시각화 전용 접근
    ys = np.linspace(0, warp_height - 1, 30)
    for fit, color in ((lane.left_fit, RED), (lane.right_fit, GREEN)):
        bev_points = np.stack([np.polyval(fit, ys), ys], axis=1)
        try:
            orig_points = lane_detector.perspective.unwarp_points(bev_points)
        except RuntimeError:
            return
        cv2.polylines(image, [orig_points.astype(np.int32).reshape(-1, 1, 2)], False,
                      color if lane.valid else GRAY, 4)


def _text(image: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    """검은 테두리 글자 — 밝은 하늘·도로 위에서도 읽히게."""
    cv2.putText(image, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_status(image: np.ndarray, lane: LaneModel, n_boxes: int, fps: float | None, stage_ms: dict[str, float]) -> None:
    scale = max(0.5, image.shape[1] / 1600)
    line_h = int(32 * scale / 0.8)
    head = f"{fps:.1f} fps  " if fps else ""
    head += "  ".join(f"{k} {v:.0f}ms" for k, v in stage_ms.items())
    _text(image, head, (12, line_h), scale, WHITE)
    lane_s = f"lane {'OK' if lane.valid else 'HOLD'} conf={lane.confidence:.2f}"
    if not lane.valid and lane.reject_reason:
        lane_s += f" ({lane.reject_reason})"
    _text(image, f"cars {n_boxes}   {lane_s}", (12, line_h * 2), scale, YELLOW if lane.valid else RED)


def process_frame(
    image: np.ndarray, detector: VehicleDetector, lane_detector, timestamp: float | None = None
) -> tuple[np.ndarray, LaneModel, list[VehicleBox], dict[str, float]]:
    """그린 프레임, 차선, 차량 박스, 단계별 시간(ms)."""
    annotated = image.copy()
    t0 = time.perf_counter()
    boxes = detector.detect(image)
    t1 = time.perf_counter()
    lane = lane_detector.detect(image, vehicle_boxes=boxes, timestamp=timestamp)
    t2 = time.perf_counter()
    ms = {"det": (t1 - t0) * 1000, "lane": (t2 - t1) * 1000}
    draw_center_region(annotated, detector._center_region_ratio)  # noqa: SLF001 — 시각화 전용 접근
    draw_lane(annotated, lane, lane_detector)
    draw_vehicle_boxes(annotated, boxes)
    return annotated, lane, boxes, ms


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
        print("⚠️ 16:9가 아님 — UFLD는 16:9 블랙박스 화면으로 학습했다. 늘어난 화면이라 결과를 믿기 어렵다")
    print(f"  통과(차선 잡음): {len(valid)}/{n} ({len(valid) / n:.0%})")
    if 0 < calib < n:
        print(f"  폭 보정 끝난 뒤 통과: {len(valid)}/{n - calib} ({len(valid) / (n - calib):.0%})"
              f"  ← 시작 직후 폭 기록 모으는 {calib}프레임 제외")
    for r, c in reasons.most_common():
        print(f"  보류 {r:<18} {c:>4}  {REASON_TEXT.get(r, '')}")
    confs = [ln.confidence for ln in lanes]
    print(f"  신뢰도 중앙값 {float(np.median(confs)):.2f} · 통과 프레임 번호 {valid[:5]}{' …' if len(valid) > 5 else ''}")


def run_image(args: argparse.Namespace, detector: VehicleDetector, lane_detector) -> None:
    image = cv2.imread(args.source)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {args.source}")
    annotated, lane, boxes, ms = process_frame(image, detector, lane_detector)
    draw_status(annotated, lane, len(boxes), None, ms)
    print(f"lane valid={lane.valid} conf={lane.confidence:.2f} reason={lane.reject_reason} · 차량 {len(boxes)}대")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, annotated)
        print(f"저장됨: {args.output}")
    if args.show:
        cv2.imshow(WINDOW, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _writer(path: str | None, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not path:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def run_stream(args: argparse.Namespace, cfg, detector: VehicleDetector, lane_detector) -> None:
    live = args.source == "camera" or args.source.isdigit()
    if not live:
        cap = cv2.VideoCapture(args.source)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15
        cap.release()
    out_fps = args.out_fps or (5.0 if live else src_fps)

    source = create_source(args.source, resolution=tuple(cfg.input.resolution))
    out = raw = None
    lanes: list[LaneModel] = []
    size = (0, 0)
    t_prev = None
    fps_ema = None
    try:
        for frame in source.frames():
            image = frame.image
            if not lanes and size == (0, 0):          # 첫 프레임 크기로 저장 영상을 연다
                size = (image.shape[1], image.shape[0])
                out = _writer(args.output, out_fps, size)
                raw = _writer(args.save_raw, out_fps, size)
            try:
                annotated, lane, boxes, ms = process_frame(image, detector, lane_detector, timestamp=frame.timestamp)
            except Exception as exc:  # noqa: BLE001 — 실차에서 한 프레임 실패로 끝나지 않게
                print(f"프레임 {frame.frame_id} 처리 실패: {exc}")
                continue

            now = time.perf_counter()
            if t_prev is not None:
                inst = 1.0 / max(now - t_prev, 1e-6)
                fps_ema = inst if fps_ema is None else 0.8 * fps_ema + 0.2 * inst
            t_prev = now
            draw_status(annotated, lane, len(boxes), fps_ema, ms)
            lanes.append(lane)

            if out is not None:
                out.write(annotated)
            if raw is not None:
                raw.write(image)
            if args.show:
                cv2.imshow(WINDOW, annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and len(lanes) >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        source.close()
        for w in (out, raw):
            if w is not None:
                w.release()
        if args.show:
            cv2.destroyAllWindows()

    for path in (args.output, args.save_raw):
        if path:
            print(f"저장됨: {path} ({len(lanes)} 프레임, 재생 {out_fps:g} fps)")
    if fps_ema:
        print(f"처리 속도 약 {fps_ema:.1f} fps")
    print_lane_summary(lanes, size[0], size[1], cfg.lane.method)


def main() -> None:
    args = parse_args()
    if not (args.output or args.show):
        raise SystemExit("--output이나 --show 중 하나는 있어야 결과를 볼 수 있습니다")
    cfg = load_config(args.config)
    detector = VehicleDetector(cfg)
    lane_detector = create_lane_detector(cfg)   # lane.method: opencv | ufld

    if Path(args.source).suffix.lower() in IMAGE_SUFFIXES:
        run_image(args, detector, lane_detector)
    else:
        run_stream(args, cfg, detector, lane_detector)


if __name__ == "__main__":
    main()
