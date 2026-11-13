"""YOLOv8n 파인튜닝 (S5).

사전학습 가중치를 국내 도로 데이터(AI Hub)로 미세조정한다. 학습이 끝나면
가중치를 `data/models/yolov8n-safewatch.pt`로 복사하므로, 이후
`scripts/export_model.py --weights data/models/yolov8n-safewatch.pt`를 그대로
이어 돌리면 된다.

**모델 크기는 yolov8n으로 고정한다** — AGENTS.md "YOLOv8n 이상 크기 모델 사용
금지". Pi5에서 s 이상은 3~5fps로 실시간이 불가능하다. `--model`로 다른 걸
넘기면 경고하고 확인을 요구한다.

사용법
    # 0) 먼저 스모크 테스트 — 설정이 맞는지 2 epoch로 빠르게 확인 (몇 분)
    python scripts/train.py --data data/processed/yolo_dataset/data.yaml --epochs 2 --name smoke

    # 1) 본 학습
    python scripts/train.py --data data/processed/yolo_dataset/data.yaml --epochs 100

    # 중단됐을 때 이어서
    python scripts/train.py --data ... --resume

해상도에 대해 — 기본 640으로 학습한다. Pi5 추론은 320이지만(`configs/pi5.yaml`),
작은 해상도로 학습하면 원거리 차량 특징을 덜 배운다. 640으로 학습한 뒤 320으로
export하는 쪽이 일반적이다. 320 학습과 비교해보고 싶으면 `--imgsz 320`으로
한 번 더 돌려 mAP를 비교하면 된다 (결과는 서로 다른 `--name`에 쌓인다).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("safewatch.train")

ALLOWED_MODEL = "yolov8n.pt"
OUTPUT_WEIGHTS = Path("data/models/yolov8n-safewatch.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8n 파인튜닝")
    parser.add_argument("--data", required=True, help="data.yaml 경로 (prepare_yolo_dataset.py가 생성)")
    parser.add_argument("--model", default=ALLOWED_MODEL, help="시작 가중치 (기본 yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=200, help="기본 200 — patience로 알아서 조기 종료된다")
    parser.add_argument("--imgsz", type=int, default=640, help="학습 해상도 (기본 640)")
    parser.add_argument(
        "--batch", type=int, default=-1,
        help="배치 크기. 기본 -1은 ultralytics AutoBatch(GPU 메모리에 맞춰 자동 결정)",
    )
    parser.add_argument("--device", default="0", help="GPU 번호. CPU면 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="데이터 로더 워커 수")
    parser.add_argument("--name", default="safewatch", help="결과가 쌓일 실행 이름")
    parser.add_argument(
        "--project", default="results/train",
        help="결과가 쌓일 루트 디렉터리",
    )
    parser.add_argument("--patience", type=int, default=50, help="이 epoch 동안 개선 없으면 조기 종료")
    parser.add_argument("--resume", action="store_true", help="같은 --name의 마지막 체크포인트에서 이어서")
    parser.add_argument("--yes", action="store_true", help="모델 크기 경고를 자동 승인")

    perf = parser.add_argument_group(
        "성능 관련",
        "정확도를 끌어올리는 쪽으로 기본값을 잡아뒀다. 특별한 이유가 없으면 건드리지 않아도 된다.",
    )
    perf.add_argument(
        "--cache", default="none", choices=["ram", "disk", "none"],
        help=(
            "이미지 캐싱. 학습 속도에만 영향을 주고 정확도와는 무관하다. "
            "RAM 여유가 데이터셋보다 크면 ram이 가장 빠르다 (기본 none — 안전한 쪽)"
        ),
    )
    perf.add_argument(
        "--close-mosaic", type=int, default=15,
        help="마지막 N epoch 동안 mosaic 증강을 끈다. 실제 분포로 마무리 학습해 정확도가 오른다 (기본 15)",
    )
    perf.add_argument("--mosaic", type=float, default=1.0, help="mosaic 증강 확률")
    perf.add_argument(
        "--mixup", type=float, default=0.15,
        help="mixup 증강 확률. 가려진 차량·겹친 차량에 강해진다 (기본 0.15)",
    )
    perf.add_argument("--scale", type=float, default=0.5, help="스케일 증강 폭 — 원거리/근거리 차량 대응")
    perf.add_argument("--degrees", type=float, default=5.0, help="회전 증강 각도. 대시캠은 수평이라 작게")
    perf.add_argument("--fliplr", type=float, default=0.5, help="좌우 반전 확률")
    perf.add_argument(
        "--no-cos-lr", action="store_true",
        help="코사인 학습률 스케줄을 끈다 (기본은 켜짐 — 후반 수렴이 안정적)",
    )
    return parser.parse_args()


def check_model_size(model: str, auto_yes: bool) -> None:
    if model == ALLOWED_MODEL or Path(model).name.startswith("yolov8n"):
        return
    logger.warning(
        "⚠️ '%s'는 yolov8n이 아니다. AGENTS.md는 n보다 큰 모델을 금지한다 — "
        "Pi5에서 3~5fps로 떨어져 실시간(10fps 목표)이 불가능하다.",
        model,
    )
    if auto_yes:
        logger.warning("   --yes 가 지정되어 그대로 진행한다.")
        return
    if input("   그래도 진행할까? [y/N] ").strip().lower() != "y":
        sys.exit("중단했다.")


def check_dataset(data_yaml: Path) -> None:
    if not data_yaml.is_file():
        sys.exit(
            f"data.yaml을 찾지 못했다: {data_yaml}\n"
            "  scripts/prepare_yolo_dataset.py를 먼저 돌려 데이터셋을 구성할 것."
        )
    text = data_yaml.read_text(encoding="utf-8")
    if "⚠️ 임시" in text:
        logger.warning(
            "⚠️ data.yaml의 train이 val을 가리키고 있다(임시 설정). 스모크 테스트는 가능하지만, "
            "본 학습 전에 반드시 --split train 으로 학습 세트를 구성할 것 — "
            "같은 데이터로 학습하고 평가하면 성능 수치가 의미 없다."
        )


def summarize(metrics, weights_path: Path) -> None:
    """학습 결과 요약을 사람이 읽을 형태로 출력한다."""
    box = getattr(metrics, "box", None)
    logger.info("\n%s\n학습 완료\n%s", "=" * 56, "=" * 56)
    if box is not None:
        logger.info("  mAP50-95 : %.4f   ← 종합 정확도 (높을수록 좋음)", box.map)
        logger.info("  mAP50    : %.4f", box.map50)
        logger.info("  Precision: %.4f   ← 오탐이 적을수록 높음", box.mp)
        logger.info("  Recall   : %.4f   ← 놓친 차량이 적을수록 높음", box.mr)
    logger.info("  가중치   : %s", weights_path)
    logger.info(
        "\n다음 단계 — Pi5 배포용 변환:\n"
        "  python scripts/export_model.py --format onnx --weights %s --imgsz 320\n"
        "  python scripts/export_model.py --format ncnn --weights %s --imgsz 320",
        weights_path, weights_path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    check_dataset(Path(args.data))
    check_model_size(args.model, args.yes)

    from ultralytics import YOLO  # import 비용이 커서 인자 검증 후에 로드한다

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        exist_ok=True,
        # --- 성능 관련 ---
        cache=False if args.cache == "none" else args.cache,
        cos_lr=not args.no_cos_lr,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=args.mixup,
        scale=args.scale,
        degrees=args.degrees,
        fliplr=args.fliplr,
        plots=True,          # 학습 곡선·PR 커브 이미지 저장 — 결과 보고에 그대로 쓴다
        save_period=10,      # 10 epoch마다 체크포인트. 중간에 멈춰도 되돌아갈 지점이 남는다
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        sys.exit(f"학습은 끝났으나 best.pt를 찾지 못했다: {best}")

    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, OUTPUT_WEIGHTS)

    summarize(model.metrics, OUTPUT_WEIGHTS)


if __name__ == "__main__":
    main()
