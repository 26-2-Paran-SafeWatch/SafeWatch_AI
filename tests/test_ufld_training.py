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

from scripts.ufld_labels import (LabelRules, ego_lanes, frame_key, lane_polylines, parse_lanes, polyline_rows,
                                 row_labels, sequence_of)

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


def test_polyline_rows_without_transform_equals_row_labels():
    """기하 증강용 정답(차선 선 → 행 칸)이 변환 없을 때 기존 정답과 같다 (실제 Training 6,000장: 346만 칸 중 10칸, 1칸 차)."""
    for road in (_road(), _road(dashed_left=False, double_right=True), _road(cx=900, half=500)):
        _, _, lanes = parse_lanes(_xml(road))
        l4 = ego_lanes(lanes, W, H)
        pl = lane_polylines(l4, W, H, 64)
        for flip in (False, True):
            a = row_labels(l4, W, H, 200, ANCH, flip=flip)
            b = polyline_rows(pl, ANCH, 200, 1600, 533, flip=flip)
            assert (a != b).sum() <= 2 and np.abs(a - b).max() <= 1


def test_polyline_rows_follow_affine_transform():
    _, _, lanes = parse_lanes(_xml(_road(dashed_left=False)))
    l4 = ego_lanes(lanes, W, H)
    pl = lane_polylines(l4, W, H, 64)
    w, h = 1600, 533
    base = polyline_rows(pl, ANCH, 200, w, h)
    shift = np.array([[1, 0, 80.0], [0, 1, 0]])                  # 오른쪽으로 80px = 199칸 기준 약 10칸
    moved = polyline_rows(pl, ANCH, 200, w, h, shift)
    ok = (base[:, 1] >= 0) & (moved[:, 1] >= 0)
    d = moved[ok, 1] - base[ok, 1]
    assert ok.sum() > 30 and d.min() >= 9 and d.max() <= 11
    up = polyline_rows(pl, ANCH, 200, w, h, np.array([[1, 0, 0], [0, 1, -100.0]]))   # 위로 밀면 아래쪽 행은 정답 없음
    assert (up[ANCH * h > h - 100 + 1, 1] == -1).all() and (up[:, 1] >= 0).sum() > 10


def _line_dataset(tmp: Path, geo: dict | None):
    """흰 직선 하나(자차 좌)를 그린 사진 + 그 정답으로 만든 형식 2 데이터 — 증강 뒤 사진과 정답이 맞는지 본다."""
    from src.lane.ufld import UFLDSpec
    spec = UFLDSpec(input_w=320, input_h=64, crop_ratio=0.6, grid_num=40, num_row=8,
                    row_anchor_start=0.42, row_anchor_end=1.0, num_lanes=4)
    fw, fh = 320, 106
    img = np.zeros((fh, fw, 3), np.uint8)
    p0, p1 = (60.0, 106.0), (150.0, 30.0)                         # 화면 좌표 (x, y)
    cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), (255, 255, 255), 3)
    (tmp / "img").mkdir(parents=True)
    cv2.imwrite(str(tmp / "img" / "a.png"), img)
    pl = np.full((4, 8, 2), np.nan, np.float32)
    pl[1, :2] = [[p1[0] / fw, p1[1] / fh], [p0[0] / fw, p0[1] / fh]]
    rows = polyline_rows(pl, spec.row_anchor, spec.grid_num, fw, fh)
    rows_f = polyline_rows(pl, spec.row_anchor, spec.grid_num, fw, fh, flip=True)
    np.savez(tmp / "train.npz", paths=np.array(["img/a.png"]), seqs=np.array(["s"]), rows=rows[None].astype(np.int16),
             rows_flip=rows_f[None].astype(np.int16), polys=pl[None])
    aug = dict(hflip=0.0, brightness=0.0, contrast_shift=0.0)
    if geo is not None:
        aug["geometric"] = geo
    return spec, aug


def test_augmented_image_and_label_stay_aligned(tmp_path):
    pytest.importorskip("torch")
    from scripts.train_ufld import UFLDDataset, _MEAN, _STD
    geo = dict(prob=1.0, rotate_deg=6.0, scale=[0.8, 1.2], shift_x=0.12, shift_y=0.17)
    spec, aug = _line_dataset(tmp_path, geo)
    checked = 0
    for seed in range(20):
        ds = UFLDDataset(tmp_path, "train", spec, aug, fixed_seed=seed)
        x, y = ds[0]
        img = (x.numpy() * _STD + _MEAN)[0]                       # (64, 320) 0~1, 아래 잘린 사진
        full_h = 106
        for r, a in enumerate(spec.row_anchor):
            if y[r, 1] < 0:
                continue
            row = int(round(a * full_h)) - (full_h - 64)          # 잘린 사진에서의 행
            if not 1 <= row < 63:
                continue
            bright = np.where(img[row] > 0.5)[0]
            if len(bright) == 0:
                continue
            gx = float(y[r, 1]) / (spec.grid_num - 1) * 320
            assert abs(bright.mean() - gx) <= 320 / (spec.grid_num - 1) + 4, (seed, r)
            checked += 1
    assert checked > 40


def test_fixed_seed_is_repeatable_and_old_format_rejected(tmp_path):
    pytest.importorskip("torch")
    from scripts.train_ufld import UFLDDataset
    geo = dict(prob=1.0, rotate_deg=6.0, scale=[0.8, 1.2], shift_x=0.12, shift_y=0.17)
    spec, aug = _line_dataset(tmp_path / "d", geo)
    a, b = UFLDDataset(tmp_path / "d", "train", spec, aug, fixed_seed=3), UFLDDataset(tmp_path / "d", "train", spec, aug, fixed_seed=3)
    assert (a[0][1] == b[0][1]).all() and (a[0][0] == b[0][0]).all()
    plain = UFLDDataset(tmp_path / "d", "train", spec, None)
    assert (plain[0][1].numpy() == np.load(tmp_path / "d" / "train.npz")["rows"][0]).all()
    d = dict(np.load(tmp_path / "d" / "train.npz"))
    del d["polys"]
    (tmp_path / "old").mkdir()
    np.savez(tmp_path / "old" / "train.npz", **d)
    with pytest.raises(ValueError, match="옛 형식"):
        UFLDDataset(tmp_path / "old", "train", spec, aug)


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
    assert im.shape == (106, 320, 3)                            # 자르기 전 (64 / 0.6) — 기하 증강용 (형식 2)
    assert man["format"] == 2 and man["frame_size"] == [320, 106]
    assert tr_npz["rows"].shape[1:] == (8, 4) and (tr_npz["rows"][:, :, 1:3] >= 0).any()
    assert tr_npz["polys"].shape[1:] == (4, 64, 2)

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


def test_drop_false_adjacent_removes_narrow_outer_lane_only():
    """옆 차로 폭이 자차의 0.6~1.6배 밖인 옆 차선만 뺀다 — 자차 차선과 정상 옆 차선은 그대로 (27번 v2)."""
    from scripts.ufld_labels import drop_false_adjacent

    l4 = ego_lanes(_road(), W, H)
    kept, n = drop_false_adjacent(l4, W, H)
    assert n == 0 and kept == l4                                   # 정상 도로: 옆 폭 = 자차 폭

    def line(xb):
        return lambda y: 960 + (xb - 960) * (y - 500) / (1060 - 500)
    near = line(960 + 400 + 220)                                    # 자차 오른선 바깥 220px — 겹선 기준(154px)보다 멀고 차로로는 좁음
    road = [ln for ln in _road() if not (ln[0] == "Lane_White_Solid" and ln[1][-1][0] > 1500)]
    road.append(("Lane_White_Solid", [(near(y), y) for y in (560, 800, 1060)]))
    l4 = ego_lanes(road, W, H)
    assert l4[3] is not None
    out, n = drop_false_adjacent(l4, W, H)
    assert n == 1 and out[3] is None
    assert out[:3] == l4[:3]

    loose = LabelRules(adj_ratio_min=0.1)                           # 규칙 값을 바꾸면 남는다
    assert drop_false_adjacent(l4, W, H, loose)[1] == 0
