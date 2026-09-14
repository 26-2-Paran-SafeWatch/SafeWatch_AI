"""해상도별 YOLOv8n 추론 처리 시간 비교 (sprint-plan.md S2 산출물).

640 / 416 / 320 세 해상도로 리사이즈한 동일 이미지에 대해 추론 시간을
반복 측정하여 평균·표준편차를 출력한다.

주의 — 이 스크립트는 개발 PC에서의 상대 비교용이다. Pi5 실기기 성능은
scripts/benchmark.py(추후 구현)로 별도 측정해야 하며, 이 결과를 Pi5
성능으로 단정하지 않는다 (AGENTS.md "불확실할 때" 원칙).

사용법
    python scripts/compare_resolutions.py --image data/raw/sample.jpg --weights data/models/yolov8n.pt
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="해상도별 YOLOv8n 추론 시간 비교")
    parser.add_argument("--image", required=True, help="테스트용 이미지 경로")
    parser.add_argument("--weights", default="data/models/yolov8n.pt")
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=[640, 416, 320]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO  # 지연 임포트 — CLI 파싱 실패 시 로드 비용 방지

    model = YOLO(args.weights)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {args.image}")

    print(f"원본 해상도: {image.shape[1]}x{image.shape[0]}")
    print(f"{'해상도':>8} | {'평균(ms)':>10} | {'표준편차(ms)':>12} | {'fps':>6}")

    for size in args.resolutions:
        resized = cv2.resize(image, (size, size))

        for _ in range(args.warmup):
            model.predict(resized, verbose=False)

        elapsed_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model.predict(resized, verbose=False)
            elapsed_ms.append((time.perf_counter() - start) * 1000)

        mean = statistics.mean(elapsed_ms)
        stdev = statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0
        fps = 1000 / mean
        print(f"{size:>8} | {mean:>10.2f} | {stdev:>12.2f} | {fps:>6.1f}")


if __name__ == "__main__":
    main()
