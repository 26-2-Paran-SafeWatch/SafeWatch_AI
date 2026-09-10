"""모델 변환 — PyTorch(.pt) → ONNX / NCNN (README.md, AGENTS.md 참고).

추론 엔진 최종 선택은 Pi5 실측 후 확정 (미결 사항,
docs/pipeline-architecture.md 5장).

사용법
    python scripts/export_model.py --format onnx
    python scripts/export_model.py --format ncnn
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8n 모델 변환")
    parser.add_argument("--format", choices=["onnx", "ncnn"], required=True)
    parser.add_argument(
        "--weights", default="data/models/yolov8n.pt", help="원본 PyTorch 가중치 경로"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "TODO(S6): ultralytics YOLO.export()로 변환 및 Pi5 실측 벤치마크 연동. "
        f"format={args.format} weights={args.weights}"
    )


if __name__ == "__main__":
    main()
