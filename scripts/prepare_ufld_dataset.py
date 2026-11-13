"""AI Hub 도로주행영상(차선 sl) zip → UFLDv2 학습 데이터 (개발 PC 전용).

사용법 (docs/ufld-training-guide.md)
    python -m scripts.prepare_ufld_dataset \\
        --label-zips "data/raw/aihub/016.도로주행영상/1.Training/라벨링데이터_0107/sl/도심로.zip" \\
                     "data/raw/aihub/016.도로주행영상/1.Training/라벨링데이터_0107/sl/자동차전용도로.zip" \\
        --image-zips "data/raw/aihub/016.도로주행영상/1.Training/원천데이터_0107/sl/도심로.zip" \\
                     "data/raw/aihub/016.도로주행영상/1.Training/원천데이터_0107/sl/자동차전용도로.zip" \\
        --exclude-label-zip "data/raw/aihub/016.도로주행영상/2.Validation/라벨링데이터_0107/라벨링데이터.zip" \\
        --out data/processed/ufld_aihub

하는 일
1. 검증(Validation) 라벨 zip에 나오는 시퀀스를 전부 학습에서 뺀다. **AI Hub 학습·검증 데이터는 같은 시퀀스,
   심지어 같은 프레임을 공유한다**(60_전방 기준 검증 시퀀스 98개 중 44개, 같은 프레임 1,968장 — 2026-11-07 확인).
   노트북의 검증·틀어짐 평가가 이 시퀀스들이라 빼지 않으면 평가 점수가 부풀려진다.
   `configs/ufld_train.yaml`의 data.exclude_dates 날짜도 통째로 뺀다.
2. 라벨 XML(설정 화각만, 같은 프레임의 여러 버전은 이름순 마지막 것)을 정답 v3로 바꾼다 (scripts/ufld_labels.py).
   자차 차로 좌·우가 없거나 폭이 이상한 프레임은 뺀다 — '차선 없음'을 잘못 배우지 않게(노트북 18·19번과 같음).
   옆 차선(0·3번)은 옆 차로 폭이 자차 차로 폭의 0.6~1.6배 밖이면 그 선만 뺀다 — 이중선·도류대·갓길 선을
   옆 차선으로 배우지 않게 (2026-11-09, 노트북 27번 v2 / `drop_false_adjacent`).
3. 이미지를 실행 경로와 같은 방식(src/lane/ufld.py preprocess)으로 모델 입력 폭에 맞춰 줄이되 **자르지 않고**
   (1600×533) JPEG로 저장한다. 학습 때 이동·회전·확대(기하 증강)를 한 뒤 아래 320행을 자른다 — 미리 자르면
   위로 밀거나 돌릴 때 잘려 나간 부분을 되살릴 수 없다 (2026-11-08, 형식 2).
4. 시퀀스 단위로 학습/선택용(val)을 나눠 `train.npz`·`val.npz`(행 정답, 뒤집은 정답, 기하 증강용 차선 선)와
   `manifest.json`을 쓴다.

이미 저장된 이미지는 다시 만들지 않는다 — 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
import unicodedata
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts.ufld_labels import (LabelRules, drop_false_adjacent, ego_lanes, frame_key, lane_polylines, parse_lanes,
                                 row_labels, sequence_of)
from src.lane.ufld import UFLDSpec
from src.utils.config import load_config

logger = logging.getLogger("safewatch.prepare_ufld")

_FOV = re.compile(r"\d+_(전방|측방)")
FORMAT = 2              # 1 = 잘린 1600×320 사진 (기하 증강 불가) / 2 = 자르기 전 1600×533 + 차선 선
MAX_PTS = 64            # 차선 하나의 점 수 상한 (Training 60_전방 6,000장 실측 최대 44)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def fov_of(path: str) -> str | None:
    m = _FOV.search(_nfc(path))
    return m.group(0) if m else None


def excluded_sequences(label_zip: Path) -> set[str]:
    """검증 라벨 zip의 모든 XML 시퀀스 (화각 무관 — 같은 주행은 통째로 뺀다)."""
    with zipfile.ZipFile(label_zip) as zf:
        return {sequence_of(frame_key(_nfc(n))) for n in zf.namelist() if n.lower().endswith(".xml")}


def to_train_frame(bgr: np.ndarray, spec: UFLDSpec) -> np.ndarray:
    """src/lane/ufld.py preprocess의 크기 변환 부분, 자르기 전 — BGR uint8 (input_h/crop_ratio, input_w, 3)."""
    resized_h = int(spec.input_h / spec.crop_ratio)
    return cv2.resize(bgr, (spec.input_w, resized_h), interpolation=cv2.INTER_LINEAR)


def _write_images(task: tuple[str, list[tuple[str, str]], dict, int]) -> tuple[int, int]:
    """한 이미지 zip에서 (zip 안 이름, 저장 경로) 목록을 처리 — 프로세스마다 zip을 따로 연다."""
    zip_path, items, spec_d, quality = task
    spec = UFLDSpec(**spec_d)
    done = failed = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member, out in items:
            outp = Path(out)
            if outp.exists():
                done += 1
                continue
            bgr = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                failed += 1
                continue
            outp.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", to_train_frame(bgr, spec), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                failed += 1
                continue
            tmp = outp.with_suffix(".tmp")
            tmp.write_bytes(buf.tobytes())
            tmp.replace(outp)                               # 끊겨도 반쯤 쓴 파일이 남지 않게
            done += 1
    return done, failed


def main() -> None:
    p = argparse.ArgumentParser(description="AI Hub 차선 라벨 → UFLDv2 학습 데이터")
    p.add_argument("--label-zips", nargs="+", required=True, help="Training 라벨링데이터 sl zip들")
    p.add_argument("--image-zips", nargs="+", required=True, help="Training 원천데이터 sl zip들")
    p.add_argument("--exclude-label-zip", required=True, help="Validation 라벨 zip — 여기 나오는 시퀀스는 학습에서 뺀다")
    p.add_argument("--out", required=True, help="출력 폴더 (data/processed/ 아래 — 커밋 금지)")
    p.add_argument("--config", default="configs/default.yaml", help="lane.ufld 형상")
    p.add_argument("--train-config", default="configs/ufld_train.yaml")
    p.add_argument("--workers", type=int, default=4, help="이미지 변환 프로세스 수")
    p.add_argument("--limit", type=int, default=0, help="시험용 — 프레임 수 상한 (0 = 전부)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    spec = UFLDSpec.from_config(load_config(args.config).lane.ufld)
    tcfg = load_config(args.train_config)
    rules = LabelRules(**tcfg.to_dict()["labels"])
    fov, ex_dates = tcfg.data.fov, set(map(str, tcfg.data.exclude_dates))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. 제외 시퀀스 ──
    ex_seqs = excluded_sequences(Path(args.exclude_label_zip))
    logger.info("검증 zip 시퀀스 %d개 → 학습에서 제외, 제외 날짜 %s", len(ex_seqs), sorted(ex_dates))

    # ── 2. 라벨 ──
    skip: Counter = Counter()
    latest: dict[str, tuple[str, str]] = {}                 # 프레임 키 → (zip, 이름) — 버전 중 이름순 마지막
    for lz in args.label_zips:
        with zipfile.ZipFile(lz) as zf:
            for n in sorted(zf.namelist(), key=_nfc):
                c = _nfc(n)
                if not c.lower().endswith(".xml") or fov_of(c) != fov:
                    continue
                latest[frame_key(c)] = (lz, n)
    logger.info("%s 라벨 %d장", fov, len(latest))

    anchors = spec.row_anchor
    recs = []
    by_zip: dict[str, list[tuple[str, str]]] = {}
    for key, (lz, n) in sorted(latest.items()):
        seq = sequence_of(key)
        if seq in ex_seqs:
            skip["검증 시퀀스와 겹침"] += 1
            continue
        if seq[:8] in ex_dates:
            skip["제외 날짜"] += 1
            continue
        by_zip.setdefault(lz, []).append((key, n))
    zips = {lz: zipfile.ZipFile(lz) for lz in by_zip}
    for lz, items in by_zip.items():
        for key, n in items:
            try:
                W, H, lanes = parse_lanes(zips[lz].read(n))
            except Exception:
                skip["XML 깨짐"] += 1
                continue
            l4 = ego_lanes(lanes, W, H, rules) if lanes else None
            if l4 is None:
                skip["자차 쌍 없음/폭 이상"] += 1
                continue
            l4, nd = drop_false_adjacent(l4, W, H, rules)
            skip["옆 차선 폭 이상으로 뺀 옆 차선(프레임은 유지)"] += nd
            recs.append(dict(key=key, seq=sequence_of(key),
                             rows=row_labels(l4, W, H, spec.grid_num, anchors).astype(np.int16),
                             rows_flip=row_labels(l4, W, H, spec.grid_num, anchors, flip=True).astype(np.int16),
                             polys=lane_polylines(l4, W, H, MAX_PTS),
                             n_lanes=sum(q is not None for q in l4)))
    for z in zips.values():
        z.close()
    logger.info("정답 있는 프레임 %d장 · 제외 %s", len(recs), dict(skip))

    # ── 3. 이미지 찾기 ──
    img_of: dict[str, tuple[str, str]] = {}
    for iz in args.image_zips:
        with zipfile.ZipFile(iz) as zf:
            for n in zf.namelist():
                c = _nfc(n)
                if c.lower().endswith((".jpg", ".png")) and fov_of(c) == fov:
                    img_of[frame_key(c)] = (iz, n)
    have = [r for r in recs if r["key"] in img_of]
    skip["이미지 없음"] += len(recs) - len(have)
    recs = have
    if args.limit:
        recs = recs[: args.limit]

    # ── 4. 시퀀스 단위 나누기 ──
    seqs = sorted({r["seq"] for r in recs})
    random.Random(int(tcfg.data.seed)).shuffle(seqs)
    val_seqs = set(seqs[: max(1, round(len(seqs) * float(tcfg.data.val_frac)))])
    for r in recs:
        r["split"] = "val" if r["seq"] in val_seqs else "train"
        r["path"] = f"img{FORMAT}/{r['split']}/{r['seq']}/{r['key']}.jpg"

    # ── 5. 이미지 저장 ──
    tasks: dict[str, list[tuple[str, str]]] = {}
    for r in recs:
        iz, n = img_of[r["key"]]
        tasks.setdefault(iz, []).append((n, str(out / r["path"])))
    spec_d = dict(spec.__dict__)
    chunks = []
    for iz, items in tasks.items():
        step = max(1, len(items) // max(1, args.workers * 4))
        chunks += [(iz, items[i:i + step], spec_d, int(tcfg.data.jpeg_quality)) for i in range(0, len(items), step)]
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (d, f) in enumerate(ex.map(_write_images, chunks), 1):
            done += d
            failed += f
            if i % 10 == 0 or i == len(chunks):
                logger.info("이미지 %d / %d (실패 %d)", done + failed, len(recs), failed)
    bad = {r["key"] for r in recs if not (out / r["path"]).exists()}
    skip["이미지 디코딩 실패"] += len(bad)
    recs = [r for r in recs if r["key"] not in bad]

    # ── 6. 정답 파일 ──
    summary = {}
    for split in ("train", "val"):
        rs = [r for r in recs if r["split"] == split]
        np.savez_compressed(
            out / f"{split}.npz",
            paths=np.array([r["path"] for r in rs]),
            seqs=np.array([r["seq"] for r in rs]),
            rows=np.stack([r["rows"] for r in rs]) if rs else np.zeros((0, spec.num_row, 4), np.int16),
            rows_flip=np.stack([r["rows_flip"] for r in rs]) if rs else np.zeros((0, spec.num_row, 4), np.int16),
            polys=np.stack([r["polys"] for r in rs]) if rs else np.zeros((0, 4, MAX_PTS, 2), np.float32),
        )
        summary[split] = dict(frames=len(rs), sequences=len({r["seq"] for r in rs}),
                              lanes_per_frame=round(float(np.mean([r["n_lanes"] for r in rs])), 2) if rs else 0)
    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), format=FORMAT, fov=fov, spec=spec_d,
        frame_size=[spec.input_w, int(spec.input_h / spec.crop_ratio)],
        label_rules=rules.__dict__, excluded_sequences=len(ex_seqs), excluded_dates=sorted(ex_dates),
        skipped=dict(skip), splits=summary,
        label_zips=[Path(z).name for z in args.label_zips], image_zips=[Path(z).name for z in args.image_zips],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info("완료 %.0f분 — 학습 %s · 선택용 %s · 제외 %s",
                (time.time() - t0) / 60, summary["train"], summary["val"], dict(skip))


if __name__ == "__main__":
    main()
