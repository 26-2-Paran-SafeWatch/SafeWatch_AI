"""UFLD 차선 모델 NPU(.hef) 동작 확인 — 같은 화면을 ONNX(CPU)와 HEF(Hailo)로 돌려 비교한다.

확인하는 것
1. HEF가 NPU에 올라가 추론이 되는지
2. 결과가 ONNX와 같은지 — 양자화(8비트) 때문에 완전히 같지는 않다. 자차 좌우 차선 점의 x 차이(px)와
   차선 '있음' 판정 일치율로 본다
3. 입력 방식 — 변환 때 정규화 층을 넣었는지에 따라 normalized / rgb255 중 하나만 ONNX와 맞는다.
   둘 다 돌려 보고 맞는 쪽을 알려 준다 → configs/pi5_hailo.yaml lane.ufld.hailo.input에 적는다
4. 속도 — 한 장 추론 시간(ms), ONNX vs HEF

Pi에서 (src.* import 때문에 -m으로 실행)
    python -m scripts.check_hailo_ufld --config configs/pi5_hailo.yaml --source camera
    python -m scripts.check_hailo_ufld --config configs/pi5_hailo.yaml --source data/raw/road.jpg \
        --output data/processed/hailo_check.jpg

카메라는 도로가 보이게(유튜브 주행 영상을 비추는 것도 됨) 두고 실행한다. 도로가 아니면 차선이 없어
'있음' 비교가 의미 없다.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.lane.ufld import OnnxUFLDBackend, UFLDSpec, decode_rows, preprocess
from src.lane.ufld_hailo import INPUT_MODES, HailoUFLDBackend
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UFLD HEF(NPU) vs ONNX(CPU) 비교")
    p.add_argument("--config", default="configs/pi5_hailo.yaml")
    p.add_argument("--source", required=True, help='이미지 파일 경로 또는 "camera"')
    p.add_argument("--output", help="두 결과를 겹쳐 그린 이미지 저장 경로")
    p.add_argument("--runs", type=int, default=20, help="속도 측정 반복 횟수")
    return p.parse_args()


def grab(source: str, resolution: tuple[int, int]) -> np.ndarray:
    if source != "camera":
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {source}")
        return img
    from src.input.source import create_source

    cam = create_source("camera", resolution=resolution)
    try:
        for i, frame in enumerate(cam.frames()):
            if i >= 10:                      # 노출·화이트밸런스가 자리 잡을 때까지 몇 장 버린다
                return frame.image
    finally:
        cam.close()
    raise RuntimeError("카메라 프레임을 받지 못했습니다")


def timed(backend, x: np.ndarray, runs: int) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    out = backend.infer(x)                   # 첫 실행은 준비 비용이 섞이니 빼고 잰다
    t0 = time.perf_counter()
    for _ in range(runs):
        backend.infer(x)
    return out, (time.perf_counter() - t0) / runs * 1000


def compare(ref, got, spec: UFLDSpec, ego: tuple[int, int], w: int, h: int) -> dict:
    (loc_r, ex_r), (loc_g, ex_g) = ref, got
    exist_r, exist_g = ex_r[1] > ex_r[0], ex_g[1] > ex_g[0]
    lanes_r, lanes_g = decode_rows(loc_r, ex_r, spec, w, h), decode_rows(loc_g, ex_g, spec, w, h)
    dx = []
    for li in ego:
        r = {round(y): x for x, y in lanes_r[li]}
        dx += [abs(x - r[round(y)]) for x, y in lanes_g[li] if round(y) in r]
    return {
        "exist_agree": float((exist_r[:, list(ego)] == exist_g[:, list(ego)]).mean()),
        "n_rows_ref": int(exist_r[:, list(ego)].sum()),
        "n_rows_got": int(exist_g[:, list(ego)].sum()),
        "dx_median": float(np.median(dx)) if dx else float("nan"),
        "dx_max": float(np.max(dx)) if dx else float("nan"),
        "loc_corr": float(np.corrcoef(loc_r.ravel(), loc_g.ravel())[0, 1]),
        "lanes": lanes_g,
    }


def draw(image: np.ndarray, lanes_ref, lanes_got, ego) -> np.ndarray:
    out = image.copy()
    for li in ego:
        for x, y in lanes_ref[li]:
            cv2.circle(out, (int(x), int(y)), 8, (255, 200, 0), 2)       # 하늘색 원 = ONNX(CPU)
        for x, y in lanes_got[li]:
            cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)         # 빨간 점 = HEF(NPU)
    cv2.putText(out, "cyan ring = ONNX(CPU)   red dot = HEF(NPU)", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    u = cfg.lane.ufld
    spec = UFLDSpec.from_config(u)
    ego = tuple(int(i) for i in u.ego_lane_ids)

    image = grab(args.source, tuple(cfg.input.resolution))
    h, w = image.shape[:2]
    print(f"입력 {w}×{h}" + ("  ⚠️ 16:9 아님 — 차선 결과를 믿기 어렵다" if abs(w / h - 16 / 9) > 0.05 else ""))
    x = preprocess(image, spec)

    onnx = OnnxUFLDBackend(u.model_path, u.num_threads)
    ref, onnx_ms = timed(onnx, x, args.runs)
    lanes_ref = decode_rows(*ref, spec, w, h)
    print(f"ONNX(CPU) {u.model_path}: {onnx_ms:.1f} ms/장, 자차 차선 '있음' 행 {int((ref[1][1] > ref[1][0])[:, list(ego)].sum())}개")

    results = {}
    for mode in INPUT_MODES:
        try:
            hailo = HailoUFLDBackend(u.hailo.model_path, spec, mode)
        except Exception as exc:  # noqa: BLE001 — 한 방식이 실패해도 다른 방식은 본다
            print(f"HEF [{mode}] 열기 실패: {exc}")
            continue
        try:
            got, hailo_ms = timed(hailo, x, args.runs)
        finally:
            hailo.close()
        c = compare(ref, got, spec, ego, w, h)
        results[mode] = (c, hailo_ms)
        print(f"HEF(NPU) [{mode:10}] {hailo_ms:6.1f} ms/장 | '있음' 일치 {c['exist_agree']:.0%} "
              f"(ONNX {c['n_rows_ref']}행 / HEF {c['n_rows_got']}행) | 차선 x 차이 중앙 {c['dx_median']:.1f}px "
              f"최대 {c['dx_max']:.1f}px | loc 상관 {c['loc_corr']:.3f}")

    if not results:
        raise SystemExit("HEF를 한 번도 실행하지 못했습니다 — 위 에러 확인")
    best = max(results, key=lambda m: (results[m][0]["exist_agree"], results[m][0]["loc_corr"]))
    c, hailo_ms = results[best]
    print(f"\n→ ONNX와 더 맞는 입력 방식: {best}  (configs/pi5_hailo.yaml lane.ufld.hailo.input)")
    print(f"→ 속도: CPU {onnx_ms:.1f} ms → NPU {hailo_ms:.1f} ms (모델 추론만, 전처리·해석 제외)")
    if c["exist_agree"] < 0.9 or not c["dx_median"] < 10:
        print("⚠️ 두 방식 모두 ONNX와 차이가 크다 — 도로 화면이 맞는지, 변환(양자화 보정 이미지·출력 층)을 확인할 것")
    if c["n_rows_ref"] == 0:
        print("⚠️ ONNX도 차선을 못 찾았다 — 도로가 보이는 화면으로 다시 확인할 것 (비교가 의미 없음)")

    if args.output:
        cv2.imwrite(args.output, draw(image, lanes_ref, c["lanes"], ego))
        print(f"저장됨: {args.output}")


if __name__ == "__main__":
    main()
