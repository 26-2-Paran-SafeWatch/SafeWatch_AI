"""YOLO 학습용 데이터셋 레이아웃 구성.

`scripts/voc_to_yolo.py`는 XML을 YOLO `.txt` 라벨로 바꿔주기만 한다. ultralytics
학습기는 그와 별개로 아래 디렉터리 구조와 `data.yaml`을 요구한다.

    <output>/images/train/*.jpg   <output>/labels/train/*.txt
    <output>/images/val/*.jpg     <output>/labels/val/*.txt
    <output>/data.yaml

이 스크립트가 그 구조를 만든다. AI Hub 원천데이터(이미지)와 라벨이 서로 다른
zip으로 내려오므로 **파일 이름(확장자 제외)으로 짝을 맞춘다.**

기본은 **심볼릭 링크**다 — 데스크탑 저장공간이 2TB인데 데이터셋이 수백 GB라
이미지를 복사하면 같은 파일이 두 벌 생긴다. 다른 디스크로 옮겨야 하거나 링크가
문제되는 환경이면 `--copy`를 쓴다.

사용법
    # Validation 먼저 (학습 전 소량 검증용)
    python scripts/prepare_yolo_dataset.py \\
        --images-dir data/raw/aihub/016.도로주행영상/2.Validation/원천데이터_0107/extracted \\
        --labels-dir data/processed/yolo_labels/val \\
        --output-dir data/processed/yolo_dataset \\
        --split val

    # Training
    python scripts/prepare_yolo_dataset.py \\
        --images-dir data/raw/aihub/016.도로주행영상/1.Training/원천데이터_0107/extracted \\
        --labels-dir data/processed/yolo_labels/train \\
        --output-dir data/processed/yolo_dataset \\
        --split train
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("safewatch.prepare_dataset")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# voc_to_yolo.py의 VEHICLE_CLASS_INDEX(0)와 반드시 일치해야 한다.
CLASS_NAMES = ["vehicle"]


def index_by_stem(directory: Path, suffixes: set[str] | None) -> dict[str, Path]:
    """디렉터리를 재귀 탐색해 {파일이름(확장자 제외): 경로}로 만든다."""
    found: dict[str, Path] = {}
    duplicates = 0
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        if path.stem in found:
            duplicates += 1
            continue
        found[path.stem] = path

    if duplicates:
        logger.warning(
            "이름이 겹치는 파일 %d개를 건너뜀 (%s) — 먼저 찾은 것만 사용한다",
            duplicates, directory,
        )
    return found


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        # 상대 경로 심볼릭 링크 — 데이터셋 폴더를 통째로 옮겨도 깨지지 않는다
        os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)


def build_split(
    images_dir: Path, labels_dir: Path, output_dir: Path, split: str, copy: bool
) -> int:
    images = index_by_stem(images_dir, IMAGE_SUFFIXES)
    labels = index_by_stem(labels_dir, {".txt"})
    labels.pop("classes", None)  # voc_to_yolo가 남기는 classes.txt는 라벨이 아니다

    logger.info("이미지 %d개, 라벨 %d개 발견", len(images), len(labels))
    if not images:
        raise SystemExit(f"이미지를 찾지 못했다: {images_dir}\n  압축 해제가 끝났는지 확인할 것.")
    if not labels:
        raise SystemExit(
            f"라벨(.txt)을 찾지 못했다: {labels_dir}\n"
            "  scripts/voc_to_yolo.py를 먼저 돌려 XML을 YOLO 포맷으로 바꿔야 한다."
        )

    paired = sorted(set(images) & set(labels))
    only_images = len(images) - len(paired)
    only_labels = len(labels) - len(paired)

    if not paired:
        raise SystemExit(
            "이미지와 라벨의 파일 이름이 하나도 안 맞는다.\n"
            f"  이미지 예시: {sorted(images)[:3]}\n"
            f"  라벨 예시:   {sorted(labels)[:3]}\n"
            "  같은 날짜/부위(bb) 묶음을 받았는지 확인할 것."
        )

    image_out = output_dir / "images" / split
    label_out = output_dir / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for stem in paired:
        src_image = images[stem]
        link_or_copy(src_image, image_out / f"{stem}{src_image.suffix}", copy)
        link_or_copy(labels[stem], label_out / f"{stem}.txt", copy)

    logger.info(
        "[%s] 짝 맞은 %d쌍 구성 완료 (%s)", split, len(paired),
        "복사" if copy else "심볼릭 링크",
    )
    if only_images or only_labels:
        logger.warning(
            "[%s] 짝이 없어 제외됨 — 이미지만 있는 것 %d개, 라벨만 있는 것 %d개. "
            "원천데이터와 라벨링데이터를 같은 묶음으로 받았는지 확인할 것.",
            split, only_images, only_labels,
        )
    return len(paired)


def write_data_yaml(output_dir: Path) -> Path:
    """ultralytics가 읽는 data.yaml. 이미 있으면 split 정보를 보존하며 덮어쓴다."""
    has_train = (output_dir / "images" / "train").is_dir()
    has_val = (output_dir / "images" / "val").is_dir()

    lines = [
        "# scripts/prepare_yolo_dataset.py가 생성함. 직접 수정하지 말 것.",
        f"path: {output_dir.resolve()}",
    ]
    if has_train:
        lines.append("train: images/train")
    if has_val:
        lines.append("val: images/val")
        if not has_train:
            # 학습 세트가 아직 없으면 ultralytics가 train 키를 요구하므로 val로 대신 채운다
            # (소량 스모크 테스트용 — 실제 학습 전에 train split을 반드시 구성할 것)
            lines.append("train: images/val  # ⚠️ 임시 — train split 구성 후 재실행하면 교체된다")

    lines.append("names:")
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}")

    path = output_dir / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("data.yaml 작성: %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 학습용 데이터셋 레이아웃 구성")
    parser.add_argument("--images-dir", required=True, help="압축 해제된 원천데이터(이미지) 디렉터리")
    parser.add_argument("--labels-dir", required=True, help="voc_to_yolo.py가 만든 YOLO .txt 라벨 디렉터리")
    parser.add_argument("--output-dir", required=True, help="데이터셋을 구성할 위치")
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument(
        "--copy", action="store_true",
        help="심볼릭 링크 대신 파일을 복사한다 (저장공간을 두 배로 쓴다)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)

    build_split(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=output_dir,
        split=args.split,
        copy=args.copy,
    )
    write_data_yaml(output_dir)


if __name__ == "__main__":
    main()
