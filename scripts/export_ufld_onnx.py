"""UFLDv2 PyTorch 가중치 → ONNX 변환 (개발 PC 전용 — Pi5는 결과 .onnx만 실행한다).

사용법
    python -m scripts.export_ufld_onnx --weights data/models/culane_res18.pth \\
        --output data/models/ufld_culane_res18.onnx --config configs/default.yaml --verify

- 모델 구조는 원 저장소 model/model_culane.py parsingNet(use_aux=False)과 같다 — 노트북
  `safewatch_lane_calib_v2.ipynb` 15번 셀 `UFLDv2`를 옮긴 것이고 가중치 키 이름도 같다.
  (원 저장소 코드는 NVIDIA DALI를 import해 바로 열리지 않아 구조만 다시 썼다.)
- 형상 값(input_size·grid_num·num_row·num_lanes)은 `--config`의 `lane.ufld`에서 읽는다.
  열 방향(col) 형상은 실행 경로에서 쓰지 않아 설정에 없으므로 인자로 받는다(CULane 기본값).
- **열 방향 출력은 잘라낸다.** 실행 경로(src/lane/ufld.py)는 행 방향 출력(loc_row·exist_row)만
  쓴다. 마지막 완전연결층에서 그 출력에 해당하는 행만 남기면 결과는 똑같고 그 층이 약 35% 작아진다
  (CULane: 2048×91,224 → 2048×58,176). NPU 메모리 부담을 줄이는 쪽이다 (lane-detection-dl-plan.md 6장).
- 가중치는 `torch.load(weights_only=True)`로만 연다 (pickle 실행 방지).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from src.lane.ufld import UFLDSpec
from src.utils.config import load_config

logger = logging.getLogger("safewatch.export_ufld")


class UFLDv2RowNet(nn.Module):
    """parsingNet(use_aux=False). 처음엔 원 저장소와 같은 전체 출력으로 만들어 가중치를 그대로 받고,
    `keep_rows_only()`로 행 방향 출력만 남긴다."""

    def __init__(self, spec: UFLDSpec, grid_col: int, num_col: int, fc_norm: bool):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        self.model = nn.Module()
        for k in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self.model, k, getattr(r, k))
        lanes = spec.num_lanes
        self.spec = spec
        self.dims = (
            spec.grid_num * spec.num_row * lanes,   # loc_row
            grid_col * num_col * lanes,             # loc_col
            2 * spec.num_row * lanes,               # exist_row
            2 * num_col * lanes,                    # exist_col
        )
        self.input_dim = spec.input_h // 32 * spec.input_w // 32 * 8
        self.pool = nn.Conv2d(512, 8, 1)
        self.cls = nn.Sequential(
            nn.LayerNorm(self.input_dim) if fc_norm else nn.Identity(),   # TuSimple판은 LayerNorm 없음
            nn.Linear(self.input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, sum(self.dims)),
        )
        self._rows_only = False

    @torch.no_grad()
    def keep_rows_only(self) -> None:
        d1, d2, d3, _ = self.dims
        last: nn.Linear = self.cls[3]
        keep = torch.cat([torch.arange(0, d1), torch.arange(d1 + d2, d1 + d2 + d3)])
        new = nn.Linear(last.in_features, len(keep))
        new.weight.copy_(last.weight[keep])
        new.bias.copy_(last.bias[keep])
        self.cls[3] = new
        self._rows_only = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        out = self.cls(self.pool(x).reshape(-1, self.input_dim))
        s, (d1, d2, d3, _) = self.spec, self.dims
        exist_from = d1 if self._rows_only else d1 + d2
        loc_row = out[:, :d1].reshape(-1, s.grid_num, s.num_row, s.num_lanes)
        exist_row = out[:, exist_from:exist_from + d3].reshape(-1, 2, s.num_row, s.num_lanes)
        return loc_row, exist_row


def load_weights(net: UFLDv2RowNet, path: Path) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck.get("model", ck).items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("seg_head")}   # 학습 보조 출력 — 추론에 안 씀
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"가중치 키 불일치 — 없음 {missing[:5]} / 남음 {unexpected[:5]}")


def export(net: UFLDv2RowNet, output: Path, opset: int = 17) -> None:
    s = net.spec
    dummy = torch.zeros(1, 3, s.input_h, s.input_w)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net.eval(), (dummy,), str(output), input_names=["input"], output_names=["loc_row", "exist_row"],
        opset_version=opset, dynamo=False,
    )


def verify(net: UFLDv2RowNet, output: Path, seed: int = 0) -> float:
    """무작위 입력으로 PyTorch와 ONNX Runtime 출력의 최대 절대 차이를 낸다."""
    import onnxruntime as ort

    s = net.spec
    x = np.random.default_rng(seed).standard_normal((1, 3, s.input_h, s.input_w)).astype(np.float32)
    with torch.no_grad():
        ref = [t.numpy() for t in net.eval()(torch.from_numpy(x))]
    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    got = sess.run(["loc_row", "exist_row"], {"input": x})
    return float(max(np.abs(a - b).max() for a, b in zip(ref, got)))


def main() -> None:
    p = argparse.ArgumentParser(description="UFLDv2 → ONNX (행 방향 출력만)")
    p.add_argument("--weights", required=True, help="원 저장소 가중치 .pth (예: culane_res18.pth)")
    p.add_argument("--output", required=True, help="저장할 .onnx 경로 (data/models/ 아래 — 커밋 금지)")
    p.add_argument("--config", default="configs/default.yaml", help="lane.ufld 형상을 읽을 설정")
    p.add_argument("--grid-col", type=int, default=100, help="열 방향 격자 수 (CULane 100)")
    p.add_argument("--num-col", type=int, default=81, help="열 앵커 수 (CULane 81)")
    p.add_argument("--no-fc-norm", action="store_true", help="TuSimple 가중치처럼 LayerNorm이 없는 경우")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify", action="store_true", help="변환 후 PyTorch와 출력 비교")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec = UFLDSpec.from_config(load_config(args.config).lane.ufld)
    net = UFLDv2RowNet(spec, args.grid_col, args.num_col, fc_norm=not args.no_fc_norm)
    load_weights(net, Path(args.weights))
    before = sum(p.numel() for p in net.parameters())
    net.keep_rows_only()
    after = sum(p.numel() for p in net.parameters())
    logger.info("파라미터 %.0fM → %.0fM (열 방향 출력 제거, fp32 %.0fMB)", before / 1e6, after / 1e6, after * 4 / 1e6)
    export(net, Path(args.output), args.opset)
    logger.info("저장: %s", args.output)
    if args.verify:
        diff = verify(net, Path(args.output))
        logger.info("PyTorch ↔ ONNX 최대 차이 %.2e", diff)
        if diff > 1e-3:
            raise SystemExit("출력 차이가 큽니다 — 변환 결과를 쓰지 말 것")


if __name__ == "__main__":
    main()
