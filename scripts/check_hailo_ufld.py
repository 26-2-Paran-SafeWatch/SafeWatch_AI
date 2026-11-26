"""UFLD NPU 변환 검증 — 같은 프레임을 원본 ONNX(CPU 전체)와 Hailo(NPU 백본 + CPU 뒷부분)로 돌려 비교한다.

재학습·재변환 뒤 결과가 원본과 같은지 확인하는 용도다 (docs/ufld-hailo-conversion.md 5장).
양자화(INT8) 때문에 완전히 같지는 않다. 자차 좌우 차선의 '있음' 행 일치율과 위치 차이(격자 칸)로 본다.
HW 파트 첫 확인은 쉬운 구간 10프레임이었다 — 차선 변경·가림 구간을 포함해 넓게 돌릴 것.

Pi에서 (src.* import 때문에 -m으로 실행)
    python -m scripts.check_hailo_ufld --config configs/pi5_ufld_hailo.yaml \
        --source data/raw/drive.mp4 --frames 100 --step 10
    python -m scripts.check_hailo_ufld --config configs/pi5_ufld_hailo.yaml --source data/raw/road.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.lane.ufld import HailoUFLDBackend, OnnxUFLDBackend, UFLDSpec, preprocess_for
from src.utils.config import load_config

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UFLD 원본 ONNX vs Hailo 비교")
    p.add_argument("--config", default="configs/pi5_ufld_hailo.yaml")
    p.add_argument("--source", required=True, help="영상 또는 이미지 파일")
    p.add_argument("--start", type=float, default=0.0, help="영상 시작 위치(초)")
    p.add_argument("--frames", type=int, default=50, help="비교할 프레임 수")
    p.add_argument("--step", type=int, default=5, help="몇 프레임마다 하나씩 볼지")
    return p.parse_args()


def frames(args: argparse.Namespace):
    if args.source.lower().endswith(IMAGE_SUFFIXES):
        img = cv2.imread(args.source)
        if img is None:
            raise FileNotFoundError(args.source)
        yield img
        return
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(args.source)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    n = i = 0
    while n < args.frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % args.step == 0:
            yield img
            n += 1
        i += 1
    cap.release()


def compare(ref, got, ego: list[int]) -> tuple[float, list[float], int]:
    """(자차 차선 '있음' 일치율, 둘 다 있는 행의 위치 차이(격자 칸) 목록, 원본 '있음' 행 수)."""
    (loc_r, ex_r), (loc_g, ex_g) = ref, got
    er, eg = (ex_r[1] > ex_r[0])[:, ego], (ex_g[1] > ex_g[0])[:, ego]
    both = er & eg
    diff = np.abs(loc_r.argmax(0)[:, ego] - loc_g.argmax(0)[:, ego])[both]
    return float((er == eg).mean()), diff.astype(float).tolist(), int(er.sum())


def main() -> None:
    args = parse_args()
    u = load_config(args.config).lane.ufld
    spec = UFLDSpec.from_config(u)
    ego = [int(i) for i in u.ego_lane_ids]

    cpu = OnnxUFLDBackend(u.model_path, u.num_threads)
    npu = HailoUFLDBackend(u.hef_path, u.tail_path, u.num_threads)
    agree, diffs, rows, t_cpu, t_npu, n = [], [], 0, [], [], 0
    try:
        for img in frames(args):
            if abs(img.shape[1] / img.shape[0] - 16 / 9) > 0.05 and n == 0:
                print("⚠️ 16:9가 아님 — 차선 결과 자체를 믿기 어렵다")
            t0 = time.perf_counter()
            ref = cpu.infer(preprocess_for(img, spec, cpu.input_format))
            t1 = time.perf_counter()
            got = npu.infer(preprocess_for(img, spec, npu.input_format))
            t2 = time.perf_counter()
            a, d, r = compare(ref, got, ego)
            agree.append(a); diffs += d; rows += r
            t_cpu.append((t1 - t0) * 1000); t_npu.append((t2 - t1) * 1000)
            n += 1
    finally:
        npu.close()

    if n == 0:
        raise SystemExit("읽은 프레임이 없습니다")
    d = np.array(diffs) if diffs else np.array([np.nan])
    print(f"프레임 {n}개 · 원본 '있음' 자차 차선 행 {rows}개")
    print(f"  '있음' 일치율   평균 {np.mean(agree):.1%} · 최저 {np.min(agree):.1%}")
    print(f"  위치 차이(칸/{spec.grid_num})  평균 {np.nanmean(d):.2f} · 최대 {np.nanmax(d):.0f}")
    print(f"  추론 시간(ms, 전처리 포함)  CPU 전체 {np.median(t_cpu):.0f} → NPU+뒷부분 {np.median(t_npu):.0f} (중앙값)")
    if rows == 0:
        print("⚠️ 원본도 차선을 못 찾았다 — 도로 영상으로 다시 볼 것 (비교가 의미 없음)")


if __name__ == "__main__":
    main()
