"""UFLDv2 차선 모델 파인튜닝 — AI Hub 정답 v3 (개발 PC GPU 전용).

사용법 (docs/ufld-training-guide.md)
    python -m scripts.train_ufld --data data/processed/ufld_aihub \\
        --weights data/models/culane_res18.pth --out data/models/ufld_aihub_ft.pth

- 시작점: CULane 사전학습 가중치(원 저장소 culane_res18.pth). 모델 구조는 scripts/export_ufld_onnx.py
  `UFLDv2RowNet`(원 저장소와 키 이름 같음)을 그대로 쓴다.
- 손실: 노트북 18번 셀 `loss18`과 같다 — 행 위치 focal CE + 존재 CE + 평균 위치 smooth L1 × mean_loss_weight.
  열(col) 출력은 실행 경로에서 쓰지 않으므로 학습하지 않는다(가중치는 그대로 남는다).
- 증강 (2026-11-08): 원 저장소처럼 사진을 이동·회전·확대(기하 증강)하고 차선 정답도 같이 변환한다
  (scripts/ufld_labels.py polyline_rows). 1차 재학습(뒤집기·밝기만)은 학습 카메라 계열(검증)은 좋아졌지만
  장착이 다른 카메라(노트북 '틀어짐')에서 나빠졌다 — 차선이 화면 어디쯤 나오는지를 외운 것으로 본다.
- 에폭 선택: 학습 시퀀스와 겹치지 않는 선택용(val) 시퀀스의 손실. 기본은 **기하 증강을 고정 난수로 건 선택용 손실**
  (train.select_by: aug) — 선택용 시퀀스도 학습과 같은 카메라 계열이라, 그대로 재면 카메라가 바뀌어도 버티는지가
  반영되지 않는다. 증강 없는 손실도 함께 기록한다.
  최종 판정은 이 스크립트가 아니라 노트북의 검증·틀어짐 평가(학습에서 뺀 시퀀스)로 한다.
- 먼 행 가중 (2026-11-10, train.far_rows): 화면 위쪽(먼 곳) 행의 위치 손실에 가중치. 노트북 29번에서 차 바닥 0.58보다
  먼 차의 offset 오차가 커서(0.10 초과 7~20%) 넣었다. 선택용 손실도 같은 가중치로 잰다(에폭 선택 = 학습 목표와 같게).
  기본 1.0 = 끔 → 2차와 같은 학습.
- 저장: `--out`에 최고 에폭 가중치를 원 저장소와 같은 형식({'model': state_dict})으로 저장한다 →
  노트북 `load_ufld`·`scripts/export_ufld_onnx.py --weights`가 그대로 연다.
  `--out`과 같은 폴더에 `*_last.pth`(이어 하기용), `*_history.json`(에폭별 손실)도 남긴다.
- 끊겼으면 같은 명령에 `--resume`을 붙이면 마지막 에폭 다음부터 한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.export_ufld_onnx import UFLDv2RowNet, load_weights
from scripts.ufld_labels import polyline_rows
from src.lane.ufld import UFLDSpec
from src.utils.config import Config, load_config

logger = logging.getLogger("safewatch.train_ufld")

_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def random_affine(rng: np.random.Generator, geo: dict, w: int, h: int) -> np.ndarray | None:
    """원 저장소 증강(회전 ±rotate_deg, 확대 scale, 이동 ±shift_x·w / ±shift_y·h) → cv2.warpAffine용 2×3 행렬.

    회전·확대는 화면 가운데 기준. prob 확률로만 건다(나머지는 None = 변환 없음).
    """
    if rng.random() >= float(geo["prob"]):
        return None
    lo, hi = (float(v) for v in geo["scale"])
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-1, 1) * float(geo["rotate_deg"])), float(rng.uniform(lo, hi)))
    m[0, 2] += rng.uniform(-1, 1) * float(geo["shift_x"]) * w
    m[1, 2] += rng.uniform(-1, 1) * float(geo["shift_y"]) * h
    return m


class UFLDDataset(Dataset):
    """prepare_ufld_dataset.py 출력 하나(train 또는 val). 이미지는 자르기 전 모델 입력 크기(BGR JPEG, 형식 2).

    aug = train.augment 설정 (None = 증강 없음). fixed_seed가 있으면 i번 사진의 난수를 (fixed_seed, i)로 고정 —
    에폭마다 같은 변환이 걸려 선택용 손실을 에폭끼리 비교할 수 있다. color=False면 뒤집기·밝기 없이 기하 증강만.
    """

    def __init__(self, root: Path, split: str, spec: UFLDSpec, aug: Config | None,
                 fixed_seed: int | None = None, color: bool = True):
        d = np.load(root / f"{split}.npz")
        self.root, self.spec, self.fixed_seed, self.color = root, spec, fixed_seed, color
        # 평범한 dict로 들고 있는다 — Config는 피클이 안 돼 DataLoader 작업 프로세스(spawn)로 못 넘긴다
        self.aug = aug.to_dict() if isinstance(aug, Config) else aug
        self.paths = d["paths"]
        self.rows = d["rows"].astype(np.int64)
        self.rows_flip = d["rows_flip"].astype(np.int64)
        self.polys = d["polys"] if "polys" in d.files else None
        self.full_h = int(spec.input_h / spec.crop_ratio)
        self.anchors = spec.row_anchor
        if len(self.paths) and self.rows.shape[1:] != (spec.num_row, spec.num_lanes):
            raise ValueError(f"{split}.npz 정답 형상 {self.rows.shape[1:]}가 설정과 다릅니다 — 데이터를 다시 만들 것")
        if len(self.paths) and self.polys is None:
            raise ValueError(f"{root}는 옛 형식(잘린 사진·차선 선 없음) — prepare_ufld_dataset.py로 다시 만들 것")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        bgr = cv2.imread(str(self.root / str(self.paths[i])), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / str(self.paths[i]))
        if bgr.shape[:2] != (self.full_h, self.spec.input_w):
            raise ValueError(f"이미지 크기 {bgr.shape[:2]} ≠ ({self.full_h}, {self.spec.input_w}): {self.paths[i]}"
                             " — 옛 형식(잘린 사진)이면 데이터를 다시 만들 것")
        y, flip, m = self.rows[i], False, None
        if self.aug is not None:
            rng = np.random.default_rng(None if self.fixed_seed is None else (self.fixed_seed, i))
            if "geometric" in self.aug:
                m = random_affine(rng, self.aug["geometric"], self.spec.input_w, self.full_h)
            if self.color:
                flip = rng.random() < float(self.aug["hflip"])
            if m is not None:
                bgr = cv2.warpAffine(bgr, m, (self.spec.input_w, self.full_h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                y = polyline_rows(self.polys[i], self.anchors, self.spec.grid_num, self.spec.input_w, self.full_h,
                                  m, flip)
            elif flip:
                y = self.rows_flip[i]
        x = bgr[-self.spec.input_h:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0   # 아래 자르기, RGB CHW 0~1
        if flip:
            x = x[:, :, ::-1]
        if self.aug is not None and self.color:
            a = 1.0 + rng.uniform(-1, 1) * float(self.aug["brightness"])
            b = rng.uniform(-1, 1) * float(self.aug["contrast_shift"])
            x = np.clip(x * a + b, 0.0, 1.0)
        x = (x - _MEAN) / _STD
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(y))


def ufld_loss(loc: torch.Tensor, exist: torch.Tensor, lab: torch.Tensor, mean_w: float,
              row_w: torch.Tensor | None = None) -> torch.Tensor:
    """노트북 loss18. loc (B,G,n,L), exist (B,2,n,L), lab (B,n,L) — -1 = 차선 없음.

    row_w (n,) — 행마다 위치 손실(focal CE·평균 위치) 가중치. None이면 전부 1(노트북과 같음).
    가중 평균이라 전부 같은 값이면 None과 같은 결과다. 존재 손실에는 걸지 않는다.
    """
    loc, exist = loc.float(), exist.float()
    valid = lab >= 0
    l_ext = F.cross_entropy(exist, valid.long())
    if not valid.any():
        return l_ext
    logp = F.log_softmax(loc, 1)
    focal = (1 - logp.exp()) ** 2 * logp
    cls = -(focal.gather(1, lab.clamp(min=0)[:, None]).squeeze(1))[valid]
    grid = torch.arange(loc.shape[1], device=loc.device).view(1, -1, 1, 1)
    mean = F.smooth_l1_loss((loc.softmax(1) * grid).sum(1)[valid], lab[valid].float(), reduction="none")
    if row_w is None:
        return cls.mean() + l_ext + mean_w * mean.mean()
    w = row_w.to(loc.device, torch.float32).view(1, -1, 1).expand_as(lab)[valid]
    return (cls * w).sum() / w.sum() + l_ext + mean_w * (mean * w).sum() / w.sum()


def far_row_weights(anchors: np.ndarray, far_cfg: Config | None) -> torch.Tensor | None:
    """train.far_rows → 행 가중치. 행 앵커(화면 높이 비율)가 below보다 위(먼 쪽)인 행은 weight배. 1이면 None(끔)."""
    if far_cfg is None or float(far_cfg.weight) == 1.0:
        return None
    if float(far_cfg.weight) <= 0:
        raise SystemExit(f"train.far_rows.weight는 0보다 커야 합니다: {far_cfg.weight}")
    a = np.asarray(anchors, np.float64)
    return torch.from_numpy(np.where(a < float(far_cfg.below), float(far_cfg.weight), 1.0).astype(np.float32))


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(net: UFLDv2RowNet, loader: DataLoader, dev: str, mean_w: float, amp: bool,
             row_w: torch.Tensor | None = None) -> float:
    net.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loc, ex = net(x)
        tot += float(ufld_loss(loc, ex, y, mean_w, row_w)) * len(x)
        n += len(x)
    net.train()
    return tot / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="UFLDv2 파인튜닝 (AI Hub 정답 v3)")
    p.add_argument("--data", required=True, help="prepare_ufld_dataset.py 출력 폴더")
    p.add_argument("--weights", required=True, help="시작 가중치 (culane_res18.pth)")
    p.add_argument("--out", required=True, help="최고 에폭 가중치 저장 경로 (.pth, data/models/ 아래)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--train-config", default="configs/ufld_train.yaml")
    p.add_argument("--resume", action="store_true", help="*_last.pth에서 이어 하기")
    p.add_argument("--epochs", type=int, default=0, help="설정값 대신 쓸 에폭 수 (0 = 설정값)")
    p.add_argument("--max-steps", type=int, default=0, help="시험용 — 에폭당 스텝 상한 (0 = 전부)")
    p.add_argument("--grid-col", type=int, default=100)
    p.add_argument("--num-col", type=int, default=81)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    spec = UFLDSpec.from_config(load_config(args.config).lane.ufld)
    tc = load_config(args.train_config).train
    epochs = args.epochs or int(tc.epochs)
    dev = pick_device()
    amp = bool(tc.amp) and dev == "cuda"
    mean_w = float(tc.mean_loss_weight)
    row_w = far_row_weights(spec.row_anchor, tc.far_rows if "far_rows" in tc else None)
    torch.manual_seed(0)
    np.random.seed(0)

    root = Path(args.data)
    ds_tr = UFLDDataset(root, "train", spec, tc.augment)
    ds_va = UFLDDataset(root, "val", spec, None)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise SystemExit(f"데이터가 비었습니다 — 학습 {len(ds_tr)}장, 선택용 {len(ds_va)}장")
    select_by = str(tc.select_by)
    if select_by not in ("plain", "aug"):
        raise SystemExit(f"train.select_by는 plain 또는 aug: {select_by}")
    has_geo = "geometric" in tc.augment
    if select_by == "aug" and not has_geo:
        raise SystemExit("select_by: aug인데 train.augment.geometric 설정이 없습니다")
    nw = int(tc.num_workers)
    dl_tr = DataLoader(ds_tr, batch_size=int(tc.batch_size), shuffle=True, num_workers=nw,
                       pin_memory=dev == "cuda", drop_last=True, persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=int(tc.batch_size), shuffle=False, num_workers=nw,
                       pin_memory=dev == "cuda", persistent_workers=nw > 0)
    dl_va_aug = None
    if has_geo:                                              # 기하 증강만, 사진마다 고정 난수 → 에폭끼리 비교 가능
        ds_va_aug = UFLDDataset(root, "val", spec, tc.augment, fixed_seed=int(tc.select_seed), color=False)
        dl_va_aug = DataLoader(ds_va_aug, batch_size=int(tc.batch_size), shuffle=False, num_workers=nw,
                               pin_memory=dev == "cuda", persistent_workers=nw > 0)
    logger.info("장치 %s (fp16 %s) · 학습 %d장 · 선택용 %d장 · 배치 %d · 에폭 %d · 기하 증강 %s · 에폭 선택 %s · 먼 행 가중 %s",
                dev, amp, len(ds_tr), len(ds_va), int(tc.batch_size), epochs,
                "켬" if has_geo else "끔", "증강 건 선택용 손실" if select_by == "aug" else "선택용 손실",
                "끔" if row_w is None else f"화면 높이 {float(tc.far_rows.below)} 위 {int((row_w != 1).sum())}행 × {float(tc.far_rows.weight)}")

    def val_losses() -> tuple[float, float | None]:
        v = evaluate(net, dl_va, dev, mean_w, amp, row_w)
        va = evaluate(net, dl_va_aug, dev, mean_w, amp, row_w) if dl_va_aug is not None else None
        return v, va

    def score(v: float, va: float | None) -> float:
        return va if select_by == "aug" else v

    net = UFLDv2RowNet(spec, args.grid_col, args.num_col, fc_norm=True)
    load_weights(net, Path(args.weights))
    net.to(dev).train()

    steps_per_ep = len(dl_tr) if not args.max_steps else min(len(dl_tr), args.max_steps)
    total = epochs * steps_per_ep
    warm = min(int(tc.warmup_steps), max(1, total // 10))
    opt = torch.optim.AdamW(net.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    last_path = out.with_name(out.stem + "_last.pth")
    hist_path = out.with_name(out.stem + "_history.json")
    history: list[dict] = []
    start_ep, best = 1, math.inf
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, best = int(ck["epoch"]) + 1, float(ck["best"])
        history = json.loads(hist_path.read_text()) if hist_path.exists() else []
        logger.info("이어 하기 — 에폭 %d부터 (지금까지 최저 선택 기준 손실 %.4f)", start_ep, best)
    else:
        v0, va0 = val_losses()
        history.append(dict(epoch=0, train_loss=None, val_loss=v0, val_aug_loss=va0))
        best = score(v0, va0)                                # 사전학습보다 나아진 에폭만 저장한다
        logger.info("학습 전 선택용 손실 %.4f%s (CULane 사전학습 그대로)", v0,
                    "" if va0 is None else f" · 증강 건 선택용 손실 {va0:.4f}")

    for ep in range(start_ep, epochs + 1):
        t0, tot, nb = time.time(), 0.0, 0
        for step, (x, y) in enumerate(dl_tr, 1):
            if args.max_steps and step > args.max_steps:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loc, ex = net(x)
            loss = ufld_loss(loc, ex, y, mean_w, row_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1
            if step % int(tc.log_every) == 0:
                logger.info("  에폭 %d 스텝 %d/%d 손실 %.4f lr %.2e", ep, step, steps_per_ep, tot / nb, sched.get_last_lr()[0])
        vl, vla = val_losses()
        mark = ""
        if score(vl, vla) < best:
            best = score(vl, vla)
            torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, "epoch": ep, "val_loss": vl,
                        "val_aug_loss": vla, "select_by": select_by,
                        "spec": dict(spec.__dict__), "data": str(root)}, out)
            mark = "  ← 최저, 저장"
        history.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_loss=vl, val_aug_loss=vla,
                            sec=round(time.time() - t0)))
        hist_path.write_text(json.dumps(history, indent=1))
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, last_path)
        logger.info("에폭 %2d  학습 손실 %.4f  선택용 손실 %.4f%s  (%.0f분)%s",
                    ep, tot / max(nb, 1), vl, "" if vla is None else f"  증강 건 선택용 {vla:.4f}",
                    (time.time() - t0) / 60, mark)
    if out.exists():
        logger.info("끝 — 최저 선택 기준 손실 %.4f → %s", best, out)
    else:
        logger.warning("끝 — 어느 에폭도 사전학습보다 선택용 손실이 낮지 않아 저장하지 않았습니다")


if __name__ == "__main__":
    main()
