"""UFLDv2 차선 모델 파인튜닝 — AI Hub 정답 v3 (개발 PC GPU 전용).

사용법 (docs/ufld-training-guide.md)
    python -m scripts.train_ufld --data data/processed/ufld_aihub \\
        --weights data/models/culane_res18.pth --out data/models/ufld_aihub_ft.pth

- 시작점: CULane 사전학습 가중치(원 저장소 culane_res18.pth). 모델 구조는 scripts/export_ufld_onnx.py
  `UFLDv2RowNet`(원 저장소와 키 이름 같음)을 그대로 쓴다.
- 손실: 노트북 18번 셀 `loss18`과 같다 — 행 위치 focal CE + 존재 CE + 평균 위치 smooth L1 × mean_loss_weight.
  열(col) 출력은 실행 경로에서 쓰지 않으므로 학습하지 않는다(가중치는 그대로 남는다).
- 에폭 선택: 학습 시퀀스와 겹치지 않는 선택용(val) 시퀀스의 손실이 가장 낮은 에폭 (노트북 19번과 같음).
  최종 판정은 이 스크립트가 아니라 노트북의 검증·틀어짐 평가(학습에서 뺀 시퀀스)로 한다.
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
from src.lane.ufld import UFLDSpec
from src.utils.config import Config, load_config

logger = logging.getLogger("safewatch.train_ufld")

_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


class UFLDDataset(Dataset):
    """prepare_ufld_dataset.py 출력 하나(train 또는 val). 이미지는 이미 모델 입력 크기(BGR JPEG)."""

    def __init__(self, root: Path, split: str, spec: UFLDSpec, aug: Config | None):
        d = np.load(root / f"{split}.npz")
        self.root, self.spec, self.aug = root, spec, aug
        self.paths = d["paths"]
        self.rows = d["rows"].astype(np.int64)
        self.rows_flip = d["rows_flip"].astype(np.int64)
        if len(self.paths) and self.rows.shape[1:] != (spec.num_row, spec.num_lanes):
            raise ValueError(f"{split}.npz 정답 형상 {self.rows.shape[1:]}가 설정과 다릅니다 — 데이터를 다시 만들 것")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        bgr = cv2.imread(str(self.root / str(self.paths[i])), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / str(self.paths[i]))
        if bgr.shape[:2] != (self.spec.input_h, self.spec.input_w):
            raise ValueError(f"이미지 크기 {bgr.shape[:2]}가 설정과 다릅니다: {self.paths[i]}")
        x = bgr[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0          # RGB CHW 0~1
        y = self.rows[i]
        if self.aug is not None:
            rng = np.random.default_rng()
            if rng.random() < float(self.aug.hflip):
                x = x[:, :, ::-1]
                y = self.rows_flip[i]
            a = 1.0 + rng.uniform(-1, 1) * float(self.aug.brightness)
            b = rng.uniform(-1, 1) * float(self.aug.contrast_shift)
            x = np.clip(x * a + b, 0.0, 1.0)
        x = (x - _MEAN) / _STD
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(y))


def ufld_loss(loc: torch.Tensor, exist: torch.Tensor, lab: torch.Tensor, mean_w: float) -> torch.Tensor:
    """노트북 loss18. loc (B,G,n,L), exist (B,2,n,L), lab (B,n,L) — -1 = 차선 없음."""
    loc, exist = loc.float(), exist.float()
    valid = lab >= 0
    l_ext = F.cross_entropy(exist, valid.long())
    if not valid.any():
        return l_ext
    logp = F.log_softmax(loc, 1)
    focal = (1 - logp.exp()) ** 2 * logp
    l_cls = -(focal.gather(1, lab.clamp(min=0)[:, None]).squeeze(1))[valid].mean()
    grid = torch.arange(loc.shape[1], device=loc.device).view(1, -1, 1, 1)
    l_mean = F.smooth_l1_loss((loc.softmax(1) * grid).sum(1)[valid], lab[valid].float())
    return l_cls + l_ext + mean_w * l_mean


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(net: UFLDv2RowNet, loader: DataLoader, dev: str, mean_w: float, amp: bool) -> float:
    net.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loc, ex = net(x)
        tot += float(ufld_loss(loc, ex, y, mean_w)) * len(x)
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
    torch.manual_seed(0)
    np.random.seed(0)

    root = Path(args.data)
    ds_tr = UFLDDataset(root, "train", spec, tc.augment)
    ds_va = UFLDDataset(root, "val", spec, None)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise SystemExit(f"데이터가 비었습니다 — 학습 {len(ds_tr)}장, 선택용 {len(ds_va)}장")
    nw = int(tc.num_workers)
    dl_tr = DataLoader(ds_tr, batch_size=int(tc.batch_size), shuffle=True, num_workers=nw,
                       pin_memory=dev == "cuda", drop_last=True, persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=int(tc.batch_size), shuffle=False, num_workers=nw,
                       pin_memory=dev == "cuda", persistent_workers=nw > 0)
    logger.info("장치 %s (fp16 %s) · 학습 %d장 · 선택용 %d장 · 배치 %d · 에폭 %d",
                dev, amp, len(ds_tr), len(ds_va), int(tc.batch_size), epochs)

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
    mean_w = float(tc.mean_loss_weight)

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
        logger.info("이어 하기 — 에폭 %d부터 (지금까지 최저 선택용 손실 %.4f)", start_ep, best)
    else:
        v0 = evaluate(net, dl_va, dev, mean_w, amp)
        history.append(dict(epoch=0, train_loss=None, val_loss=v0))
        best = v0                                            # 사전학습보다 나아진 에폭만 저장한다
        logger.info("학습 전 선택용 손실 %.4f (CULane 사전학습 그대로)", v0)

    for ep in range(start_ep, epochs + 1):
        t0, tot, nb = time.time(), 0.0, 0
        for step, (x, y) in enumerate(dl_tr, 1):
            if args.max_steps and step > args.max_steps:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loc, ex = net(x)
            loss = ufld_loss(loc, ex, y, mean_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1
            if step % int(tc.log_every) == 0:
                logger.info("  에폭 %d 스텝 %d/%d 손실 %.4f lr %.2e", ep, step, steps_per_ep, tot / nb, sched.get_last_lr()[0])
        vl = evaluate(net, dl_va, dev, mean_w, amp)
        mark = ""
        if vl < best:
            best = vl
            torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, "epoch": ep, "val_loss": vl,
                        "spec": dict(spec.__dict__), "data": str(root)}, out)
            mark = "  ← 최저, 저장"
        history.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_loss=vl, sec=round(time.time() - t0)))
        hist_path.write_text(json.dumps(history, indent=1))
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best}, last_path)
        logger.info("에폭 %2d  학습 손실 %.4f  선택용 손실 %.4f  (%.0f분)%s",
                    ep, tot / max(nb, 1), vl, (time.time() - t0) / 60, mark)
    if out.exists():
        logger.info("끝 — 최저 선택용 손실 %.4f → %s", best, out)
    else:
        logger.warning("끝 — 어느 에폭도 사전학습보다 선택용 손실이 낮지 않아 저장하지 않았습니다")


if __name__ == "__main__":
    main()
