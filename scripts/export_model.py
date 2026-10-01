"""모델 변환 — PyTorch(.pt) → ONNX / NCNN (README.md, AGENTS.md 참고).

추론 엔진 최종 선택(ONNX vs NCNN)은 Pi5 실측 후 확정한다 (미결 사항,
docs/pipeline-architecture.md 5장). 변환 자체는 파인튜닝 여부와 무관하게
먼저 해볼 수 있어, 사전학습 가중치(`data/models/yolov8n.pt`)로 두 경로
모두 완성해 뒀다 — 파인튜닝(S5) 후에는 같은 명령을 그 가중치에 다시
돌리면 된다.

**NCNN은 ONNX를 거치지 않는다** — 예상과 달리 `onnx2ncnn`이 아니라
PNNX(https://github.com/pnnx/pnnx)가 PyTorch 그래프를 직접 추적(trace)해
NCNN으로 변환한다(ultralytics 8.4 기준, `ultralytics.utils.export.ncnn`).
그래서 ONNX export(`export_onnx`)와 NCNN export(`export_ncnn`)는 서로
독립된 별도 경로이며, 하나가 안 된다고 다른 하나도 막히지 않는다. 추가
의존성은 `ncnn`(런타임)과 `pnnx`(변환 도구) 두 파이썬 패키지뿐이라
우려했던 것보다 가벼웠다.

사용법
    # 개발 PC 검증용 (640, dev.yaml과 동일 해상도)
    python scripts/export_model.py --format onnx
    python scripts/export_model.py --format ncnn

    # Pi5 배포용 (320, pi5.yaml과 동일 해상도)
    python scripts/export_model.py --format onnx --imgsz 320
    python scripts/export_model.py --format ncnn --imgsz 320

    # 파인튜닝 후 (S5 이후)
    python scripts/export_model.py --format ncnn --weights data/models/yolov8n-safewatch.pt --imgsz 320
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
        help="(ONNX 전용) onnxslim으로 그래프 단순화를 건너뛴다 (기본은 단순화 적용)",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="(NCNN 전용) FP16으로 저장한다. 기본은 FP32 — 양자화는 정확도 손실을 실측하기 "
             "전까지 이득이 확인된 게 아니므로 기본값으로 켜지 않는다 "
             "(AGENTS.md 부록 C, sprint-plan.md S6 '양자화가 실제로 이득인지 실측 확인')",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="변환 직후 실제로 로드해 더미 입력을 돌려보는 검증을 건너뛴다",
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


def export_ncnn(weights: str, imgsz: int, fp16: bool) -> Path:
    """`YOLO.export()`로 PyTorch(.pt) 가중치를 NCNN으로 변환한다.

    출력은 ONNX처럼 파일 하나가 아니라 `model.ncnn.param`+`model.ncnn.bin`이
    담긴 디렉터리(`<weights 이름>_ncnn_model/`)다 — `VehicleDetector`가
    `YOLO(weights)`에 이 디렉터리 경로를 그대로 넘기면 ultralytics가 내부적으로
    두 파일을 읽는다.

    ONNX export의 `opset`/`simplify`에 대응하는 개념이 NCNN에는 없다 — PNNX가
    변환까지 전부 처리해 조정할 그래프 최적화 옵션 자체가 노출되지 않는다.
    `quantize=16`(FP16)만 켤 수 있다.
    """
    from ultralytics import YOLO

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"가중치 파일이 없습니다: {weights_path}")

    logger.info("NCNN 변환 시작: weights=%s imgsz=%d fp16=%s", weights_path, imgsz, fp16)
    model = YOLO(str(weights_path))
    exported = model.export(format="ncnn", imgsz=imgsz, quantize=16 if fp16 else None)
    return Path(exported)


def verify_ncnn(ncnn_dir: Path, imgsz: int) -> None:
    """변환된 NCNN 모델을 `VehicleDetector`와 같은 방식(`YOLO(path).predict()`)으로
    실제로 로드·추론해 본다.

    NCNN은 전용 블롭 이름(`in0`/`out0` 등)을 알아야 `ncnn` 패키지로 직접 돌릴 수
    있어, 대신 이 파이프라인이 실제로 쓰는 경로 그대로(ultralytics의 NCNN
    AutoBackend) 검증한다 — ONNX의 `onnxruntime.InferenceSession` 직접 호출보다
    한 단계 더 통합에 가까운 검증이다.
    """
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(ncnn_dir), task="detect")
    dummy = np.random.default_rng(0).integers(0, 256, size=(imgsz, imgsz, 3), dtype=np.uint8)

    start = time.perf_counter()
    results = model.predict(dummy, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "NCNN 로드·추론 검증 통과 — 입력 %dx%d, 검출 %d건, 1회 추론 %.1fms "
        "(워밍업 없는 최초 1회라 실제 처리 시간보다 느릴 수 있음)",
        imgsz, imgsz, len(results[0].boxes), elapsed_ms,
    )


def _path_size_mb(path: Path) -> float:
    """파일 하나(ONNX) 또는 디렉터리(NCNN) 모두에 대응하는 크기 계산."""
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
    return path.stat().st_size / (1024 * 1024)


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

    if args.format == "onnx":
        output_path = export_onnx(
            weights=args.weights, imgsz=args.imgsz, opset=args.opset,
            simplify=not args.no_simplify,
        )
    else:
        output_path = export_ncnn(weights=args.weights, imgsz=args.imgsz, fp16=args.fp16)

    logger.info(
        "%s 변환 완료: %s (%.1fMB)",
        args.format.upper(), output_path, _path_size_mb(output_path),
    )

    if not args.no_verify:
        if args.format == "onnx":
            verify_onnx(output_path, args.imgsz)
        else:
            verify_ncnn(output_path, args.imgsz)


if __name__ == "__main__":
    main()
