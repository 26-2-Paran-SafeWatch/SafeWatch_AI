"""모델 변환 — PyTorch(.pt) → ONNX / NCNN (README.md, AGENTS.md 참고).

추론 엔진 최종 선택(ONNX vs NCNN)은 Pi5 실측 후 확정한다 (미결 사항,
docs/pipeline-architecture.md 5장). 변환 자체는 파인튜닝 여부와 무관하게
먼저 해볼 수 있어, 사전학습 가중치(`data/models/yolov8n.pt`)로 ONNX export
경로부터 완성한다 — 파인튜닝(S5) 후에는 같은 명령을 그 가중치에 다시
돌리면 된다.

**ONNX만 구현** — NCNN은 `onnx2ncnn` 변환 도구에 추가 시스템 의존성이
있어(`ncnn` 파이썬 패키지, ARM 빌드가 플랫폼마다 다름) 별도로 확인 후
이어서 구현한다. 지금은 "ONNX export부터" 요청에 맞춰 그 경로만 완성했다.

사용법
    # 개발 PC 검증용 (640, dev.yaml과 동일 해상도)
    python scripts/export_model.py --format onnx

    # Pi5 배포용 (320, pi5.yaml과 동일 해상도)
    python scripts/export_model.py --format onnx --imgsz 320

    # 파인튜닝 후 (S5 이후)
    python scripts/export_model.py --format onnx --weights data/models/yolov8n-safewatch.pt --imgsz 320
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger("safewatch.export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8n 모델 변환")
    parser.add_argument("--format", choices=["onnx", "ncnn"], required=True)
    parser.add_argument(
        "--weights", default="data/models/yolov8n.pt", help="원본 PyTorch 가중치 경로"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="변환 시 고정할 입력 해상도(정사각형). "
             "개발 PC 검증은 640(configs/dev.yaml), Pi5 배포는 320(configs/pi5.yaml)",
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset 버전. requirements.txt의 onnxruntime>=1.17과 호환되는 범위로 고정",
    )
    parser.add_argument(
        "--no-simplify", action="store_true",
        help="onnxslim으로 그래프 단순화를 건너뛴다 (기본은 단순화 적용)",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="변환 직후 onnxruntime으로 로드해 더미 입력을 돌려보는 검증을 건너뛴다",
    )
    return parser.parse_args()


def export_onnx(weights: str, imgsz: int, opset: int, simplify: bool) -> Path:
    """`YOLO.export()`로 PyTorch(.pt) 가중치를 ONNX로 변환한다.

    `imgsz`를 고정하면(동적 입력 크기 미사용, `dynamic=False`) Pi5에서 매
    프레임 셰이프를 다시 추론할 필요가 없어 추론 속도에 유리하다 — 어차피
    이 파이프라인은 config로 해상도를 고정해서 쓰므로(AGENTS.md "설정값을
    하드코딩하지 않는다"와는 별개로, 입력 해상도 자체는 실행 중 안 바뀜)
    동적 입력 지원을 둘 이유가 없다.
    """
    from ultralytics import YOLO

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"가중치 파일이 없습니다: {weights_path}")

    logger.info(
        "ONNX 변환 시작: weights=%s imgsz=%d opset=%d simplify=%s",
        weights_path, imgsz, opset, simplify,
    )
    model = YOLO(str(weights_path))
    exported = model.export(
        format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False,
    )
    return Path(exported)


def verify_onnx(onnx_path: Path, imgsz: int) -> None:
    """onnxruntime으로 변환된 모델을 실제로 로드·추론해 본다.

    변환 자체는 성공해도 onnxruntime이 지원하지 않는 연산자가 섞여 있으면
    런타임에서야 실패하는 경우가 있어("측정을 습관화한다", AGENTS.md 부록 C),
    export 직후 바로 확인한다. Pi5 실측(scripts/benchmark.py, S6)과는 별개로
    "일단 로드는 되는가"만 보는 빠른 검증이다.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    dummy = np.random.default_rng(0).random(input_meta.shape, dtype=np.float32)

    start = time.perf_counter()
    outputs = session.run(None, {input_meta.name: dummy})
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "onnxruntime 검증 통과 — 입력 %s, 출력 %d개(shape 0: %s), 1회 추론 %.1fms "
        "(CPUExecutionProvider, 워밍업 없는 최초 1회라 실제 처리 시간보다 느릴 수 있음)",
        input_meta.shape, len(outputs), outputs[0].shape, elapsed_ms,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.format == "ncnn":
        raise NotImplementedError(
            "TODO(S6): NCNN 변환 — onnx2ncnn 추가 의존성 확인 후 이어서 구현. "
            "우선 '--format onnx'로 ONNX export 경로부터 완성했다."
        )

    onnx_path = export_onnx(
        weights=args.weights, imgsz=args.imgsz, opset=args.opset,
        simplify=not args.no_simplify,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX 변환 완료: %s (%.1fMB)", onnx_path, size_mb)

    if not args.no_verify:
        verify_onnx(onnx_path, args.imgsz)


if __name__ == "__main__":
    main()
