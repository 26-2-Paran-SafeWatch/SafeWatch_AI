"""UFLDv2 학습 준비·학습 스크립트 테스트 — 가짜 AI Hub zip으로 끝까지 돌려 본다.

정답 규칙(scripts/ufld_labels.py)이 노트북과 같은지는 실제 AI Hub 검증 라벨 2,766장으로 따로 대조했다
(2026-11-07, 노트북 함수 원문과 행 정답이 전부 일치). 여기서는 규칙의 핵심 동작과 스크립트 연결을 본다.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.ufld_labels import LabelRules, ego_lanes, frame_key, parse_lanes, row_labels, sequence_of

W, H = 1920, 1080


def _xml(lines: list[tuple[str, list[tuple[float, float]]]]) -> bytes:
    body = "".join(
        f"<line><name>{nm}</name><controlPt>" + "".join(f"<x>{x}</x><y>{y}</y>" for x, y in pts)
        + "</controlPt></line>" for nm, pts in lines)
    return f"<annotation><size><width>{W}</width><height>{H}</height></size>{body}</annotation>".encode()


def _road(cx: float = 960.0, half: float = 400.0, dashed_left: bool = True, double_right: bool = False):
    """소실점(960, 500)으로 모이는 직선 차로 4선. 왼쪽 자차 경계는 점선 조각 여러 개."""
    def line(xb):
        return lambda y: 960 + (xb - 960) * (y - 500) / (1060 - 500)
    out = []
    for xb, nm in ((cx - 3 * half, "Lane_White_Solid"), (cx + 3 * half, "Lane_White_Solid")):
        f = line(xb)
        out.append((nm, [(f(y), y) for y in (560, 800, 1060)]))
    f = line(cx - half)
    if dashed_left:
        for y0 in range(560, 1060, 100):
            out.append(("Lane_White_Dash", [(f(y0), y0), (f(y0 + 60), y0 + 60)]))
    else:
        out.append(("Lane_White_Solid", [(f(y), y) for y in (560, 800, 1060)]))
    f = line(cx + half)
    out.append(("Lane_Yellow_Solid", [(f(y), y) for y in (560, 800, 1060)]))
    if double_right:
        g = line(cx + half + 40)
        out.append(("Lane_Yellow_Solid", [(g(y), y) for y in (560, 800, 1060)]))
    return out


ANCH = np.linspace(0.42, 1.0, 72)


def test_dashed_ego_boundary_is_merged_and_picked():
    W_, H_, lanes = parse_lanes(_xml(_road()))
    l4 = ego_lanes(lanes, W_, H_)
    assert l4 is not None and all(q is not None for q in l4[1:3])
    assert l4[1][0] == pytest.approx(560, abs=2) and l4[2][0] == pytest.approx(1360, abs=2)
    assert "×" in l4[1][2]                                   # 점선 조각이 이어 붙음


def test_double_line_becomes_one_boundary():
    _, _, lanes = parse_lanes(_xml(_road(double_right=True)))
    l4 = ego_lanes(lanes, W, H)
    assert l4[2][2] == "겹선" and l4[2][0] == pytest.approx(1380, abs=2)   # 두 선 가운데
    assert l4[3] is not None and l4[3][0] > 2000                           # 바깥 옆 선은 그대로


def test_narrow_or_one_sided_frames_have_no_label():
    _, _, lanes = parse_lanes(_xml(_road(half=150)))              # 폭 300px = 화면의 16% < 25%
    assert ego_lanes(lanes, W, H) is None
    _, _, lanes = parse_lanes(_xml([l for l in _road() if l[1][-1][0] < 960]))
    assert ego_lanes(lanes, W, H) is None


def test_row_labels_flip_mirrors_lanes():
    _, _, lanes = parse_lanes(_xml(_road(dashed_left=False)))
    l4 = ego_lanes(lanes, W, H)
    a = row_labels(l4, W, H, 200, ANCH)
    b = row_labels(l4, W, H, 200, ANCH, flip=True)
    ok = (a[:, 1] >= 0) & (b[:, 2] >= 0)
    assert ok.sum() > 30
    assert np.abs(a[ok, 1] + b[ok, 2] - 199).max() <= 1          # 칸 번호 거울상 (내림 오차 1칸)
    assert (a[ANCH * H > 1060, 1] >= 0).all()                     # 라벨 아래끝 밑으로 직선 연장


def test_frame_key_and_sequence():
    k = frame_key("a/b/2_20200916_085235_I1000225_v001_1.xml")
    assert k == "2_20200916_085235_I1000225" and sequence_of(k) == "20200916_085235"


# ── 데이터 준비 + 학습 (가짜 zip) ────────────────────────────────────────────

def _small_cfg(tmp: Path) -> tuple[Path, Path]:
    cfg = tmp / "small.yaml"
    cfg.write_text(yaml.safe_dump({"lane": {"ufld": dict(
        input_size=[320, 64], crop_ratio=0.6, grid_num=20, num_row=8, row_anchor_range=[0.42, 1.0], num_lanes=4)}}))
    tc = yaml.safe_load(Path("configs/ufld_train.yaml").read_text())
    tc["data"].update(val_frac=0.34)
    tc["train"].update(batch_size=2, num_workers=0, warmup_steps=1, log_every=1)
    tcfg = tmp / "train.yaml"
    tcfg.write_text(yaml.safe_dump(tc, allow_unicode=True))
    return cfg, tcfg


def _make_zips(tmp: Path) -> dict[str, Path]:
    img = np.full((H, W, 3), 90, np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    seqs = ["20200101_000001", "20200101_000002", "20200101_000003", "20200916_000004", "20200202_999999"]
    lab, src, val = tmp / "lab.zip", tmp / "src.zip", tmp / "val.zip"
    with zipfile.ZipFile(lab, "w") as zl, zipfile.ZipFile(src, "w") as zs:
        for s in seqs:
            for f in range(3):
                stem = f"2_{s}_{f * 30:06d}"
                d = f"도심로/주간일출/맑음/60_전방/{s}"
                zl.writestr(f"{d}/2_annotations_v001_1/{stem}_v001_1.xml", _xml(_road()))
                zs.writestr(f"{d}/{stem}.jpg", jpg.tobytes())
            zl.writestr(f"도심로/주간일출/맑음/30_전방/{s}/x/2_{s}_000000_v001_1.xml", _xml(_road()))  # 다른 화각
        zl.writestr("도심로/주간일출/맑음/60_전방/20200101_000001/x/2_20200101_000001_000090_v001_1.xml", b"<broken>")
    with zipfile.ZipFile(val, "w") as zv:
        zv.writestr("60_전방/2_20200202_999999_000000_v001_1.xml", _xml(_road()))      # 학습과 겹치는 시퀀스
    return dict(lab=lab, src=src, val=val)


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    module.main()


def test_prepare_then_train_end_to_end(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from scripts import prepare_ufld_dataset as prep
    from scripts import train_ufld as tr
    from scripts.export_ufld_onnx import UFLDv2RowNet, load_weights
    from src.lane.ufld import UFLDSpec
    from src.utils.config import load_config

    cfg, tcfg = _small_cfg(tmp_path)
    z = _make_zips(tmp_path)
    out = tmp_path / "ds"
    args = ["--label-zips", str(z["lab"]), "--image-zips", str(z["src"]), "--exclude-label-zip", str(z["val"]),
            "--out", str(out), "--config", str(cfg), "--train-config", str(tcfg), "--workers", "1"]
    _run(prep, args, monkeypatch)

    man = json.loads((out / "manifest.json").read_text())
    assert man["skipped"]["검증 시퀀스와 겹침"] == 3
    assert man["skipped"]["제외 날짜"] == 3                    # 20200916
    assert man["skipped"]["XML 깨짐"] == 1
    assert man["splits"]["train"]["frames"] + man["splits"]["val"]["frames"] == 9
    assert man["splits"]["val"]["sequences"] == 1
    tr_npz, va_npz = np.load(out / "train.npz"), np.load(out / "val.npz")
    assert not set(tr_npz["seqs"]) & set(va_npz["seqs"])        # 시퀀스 단위 분리
    im = cv2.imread(str(out / tr_npz["paths"][0]))
    assert im.shape == (64, 320, 3)
    assert tr_npz["rows"].shape[1:] == (8, 4) and (tr_npz["rows"][:, :, 1:3] >= 0).any()

    _run(prep, args, monkeypatch)                               # 다시 실행 — 이미지는 재사용, 결과 같음
    assert json.loads((out / "manifest.json").read_text())["splits"] == man["splits"]

    spec = UFLDSpec.from_config(load_config(cfg).lane.ufld)
    torch.manual_seed(0)
    init = tmp_path / "init.pth"
    torch.save({"model": UFLDv2RowNet(spec, 10, 6, fc_norm=True).state_dict()}, init)
    w_out = tmp_path / "ft.pth"
    targs = ["--data", str(out), "--weights", str(init), "--out", str(w_out), "--config", str(cfg),
             "--train-config", str(tcfg), "--epochs", "2", "--grid-col", "10", "--num-col", "6"]
    _run(tr, targs, monkeypatch)
    hist = json.loads((tmp_path / "ft_history.json").read_text())
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    assert (tmp_path / "ft_last.pth").exists()
    if w_out.exists():                                          # 무작위 시작이라 거의 항상 나아짐 — 저장본은 그대로 열려야 함
        load_weights(UFLDv2RowNet(spec, 10, 6, fc_norm=True), w_out)

    _run(tr, targs[:-6] + ["--epochs", "3", "--grid-col", "10", "--num-col", "6", "--resume"], monkeypatch)
    assert [h["epoch"] for h in json.loads((tmp_path / "ft_history.json").read_text())] == [0, 1, 2, 3]


def test_loss_matches_notebook_definition():
    torch = pytest.importorskip("torch")
    from scripts.train_ufld import ufld_loss

    torch.manual_seed(0)
    loc, ex = torch.randn(2, 20, 8, 4), torch.randn(2, 2, 8, 4)
    lab = torch.randint(-1, 20, (2, 8, 4))
    F = torch.nn.functional
    valid = lab >= 0
    logp = F.log_softmax(loc, 1)
    focal = (1 - logp.exp()) ** 2 * logp
    ref = (-(focal.gather(1, lab.clamp(min=0)[:, None]).squeeze(1))[valid].mean()
           + F.cross_entropy(ex, valid.long())
           + 0.05 * F.smooth_l1_loss((loc.softmax(1) * torch.arange(20).view(1, -1, 1, 1)).sum(1)[valid],
                                     lab[valid].float()))
    assert torch.allclose(ufld_loss(loc, ex, lab, 0.05), ref)
    assert torch.isfinite(ufld_loss(loc, ex, torch.full_like(lab, -1), 0.05))   # 차선 없는 배치
