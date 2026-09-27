"""AI Hub 도로주행영상류 Pascal VOC XML 라벨 → YOLO 포맷 변환기.

`bb`(bounding box) 카테고리 XML만 대상으로 한다. `pg`(폴리곤)·`sl`(차선)·
`fs`(자유공간)는 bbox 형식이 아니라 별도 변환이 필요해 이 스크립트의
대상이 아니다.

차종 구분 없이 단일 `vehicle` 클래스로 병합한다 — src/detection/detector.py의
`VEHICLE_CLASS_IDS`가 이미 차종을 구분하지 않고, class_id가 RISK·METRICS·
EVENT 어디에서도 쓰이지 않기 때문이다(docs/pipeline-architecture.md 3.2
"파인튜닝 라벨 클래스 매핑" 참고). `Vehicle_Unknown`도 bbox 크기 분포상
제외할 근거가 없어 포함한다.

사용법
    python scripts/voc_to_yolo.py \
        --input-dir data/raw/aihub/016.도로주행영상/2.Validation/라벨링데이터_0107/extracted/bb \
        --output-dir data/processed/yolo_labels
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger("safewatch.voc_to_yolo")

# 파인튜닝 시 이 이름들을 전부 단일 vehicle 클래스로 병합한다.
VEHICLE_LABELS = {"Vehicle_Car", "Vehicle_Bus", "Vehicle_Motorcycle", "Vehicle_Unknown"}
VEHICLE_CLASS_INDEX = 0


def convert_one(xml_path: Path) -> tuple[list[str], int, int]:
    """XML 하나를 YOLO 라벨 라인 목록으로 변환한다.

    반환값: (YOLO 라인 목록, 변환된 vehicle 인스턴스 수, 스킵한 비차량 라벨 수)
    """
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    lines: list[str] = []
    skipped = 0
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in VEHICLE_LABELS:
            skipped += 1
            continue

        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{VEHICLE_CLASS_INDEX} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines, len(lines), skipped


def convert_dir(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.txt").write_text("vehicle\n", encoding="utf-8")

    xml_files = sorted(input_dir.rglob("*.xml"))
    if not xml_files:
        logger.warning("XML 파일을 찾지 못함: %s", input_dir)
        return

    total_vehicles = 0
    total_skipped = 0
    failed = 0
    for xml_path in xml_files:
        try:
            lines, n_vehicles, n_skipped = convert_one(xml_path)
        except Exception as exc:  # noqa: BLE001 — 라벨 하나 실패로 전체 변환이 멈추면 안 됨
            logger.warning("변환 실패, 건너뜀: %s (%s)", xml_path, exc)
            failed += 1
            continue

        total_vehicles += n_vehicles
        total_skipped += n_skipped
        out_path = output_dir / f"{xml_path.stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    logger.info(
        "변환 완료 — XML %d개(실패 %d개), vehicle 인스턴스 %d개, 비차량 라벨 스킵 %d개 → %s",
        len(xml_files), failed, total_vehicles, total_skipped, output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOC XML(bb 카테고리) → YOLO 라벨 변환")
    parser.add_argument("--input-dir", required=True, help="XML 라벨이 있는 디렉터리(재귀 탐색)")
    parser.add_argument("--output-dir", required=True, help="YOLO .txt 라벨 출력 디렉터리")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    convert_dir(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
