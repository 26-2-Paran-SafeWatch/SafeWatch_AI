"""AI Hub 차선 라벨(XML) → UFLDv2 학습 정답 — '정답 v3'.

노트북 `notebook/safewatch_lane_calib_v2.ipynb`에서 단계적으로 고친 정답 규칙을 그대로 옮겼다.
노트북 평가 숫자와 학습 정답이 같은 뜻이 되도록 식을 바꾸지 않는다 (대조: tests/test_ufld_training.py).

- 2번 셀 `parse_lanes`      : XML의 line/controlPt → (이름, [(x, y), ...])
- 17번 셀 `merge_dashes`    : 점선이 칸마다 따로 라벨된 것을 같은 색·같은 직선끼리 이어 붙인다 (정답 v2)
- 18번 셀 `cands`           : 판정 구간 세로 길이의 COVER_MIN 이상을 덮는 선만, 아래끝 행 x를 구한다
- 19번 셀 `clusters`        : 아래끝 x가 가까운 선(겹선)은 한 경계로 합친다 (정답 v3)
- 19번 셀 `ego3`            : 화면 가운데 기준 좌·우 가장 가까운 경계 = 자차 차로. 폭이 화면 폭의 WMIN~WMAX 밖이면 정답 없음
- 18번 셀 `row_labels`      : 행 앵커마다 차선 4개(옆·자차좌·자차우·옆)의 격자 칸 번호 (-1 = 없음)

좌표는 전부 라벨 XML의 원본 화면 좌표(픽셀). 학습 입력 크기와 무관하다(x/W 비율만 쓴다).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np

Line = tuple[str, list[tuple[float, float]]]
# (아래끝 x, 위끝 x 또는 None, 이름, 점 목록) — 노트북 cands 출력 형식
Cand = tuple[float, "float | None", str, list[tuple[float, float]]]


@dataclass(frozen=True)
class LabelRules:
    """정답 규칙 값 — 노트북 셀의 상수와 같다. 바꾸면 노트북 평가와 뜻이 달라진다."""

    band_bottom: float = 0.98   # 판정 구간 아래끝 (화면 높이 비율) — 4번 셀 REF['60_전방'][0]
    band_top: float = 0.56      # 판정 구간 위끝 — 4번 셀 REF['60_전방'][1] (SPAN_MAX 0.42)
    gap_max: float = 0.15       # 17번 GAP_MAX  점선 조각 사이 세로 간격 한도 (화면 높이 대비)
    join_tol: float = 0.02      # 17번 JOIN_TOL 이어 붙일 두 조각의 중간 높이 x 차이 한도 (화면 폭 대비)
    cover_min: float = 0.3      # 17번 COVER_MIN 후보 선이 덮어야 할 구간 세로 길이 비율
    double: float = 0.08        # 19번 DOUBLE19 겹선 — 아래끝 x 간격이 화면 폭의 이 비율 미만이면 한 경계
    width_min: float = 0.25     # 19번 WMIN19 자차 차로 폭 (아래끝 행, 화면 폭 대비)
    width_max: float = 0.95     # 19번 WMAX19


_VERSION = re.compile(r"_v\d+_\d+$")


def frame_key(name: str) -> str:
    """라벨·이미지 파일 이름 → 프레임 키. 라벨 버전 접미사(_v001_1 등)를 뗀다 (노트북 `norm`)."""
    return _VERSION.sub("", PurePosixPath(name).stem)


def sequence_of(key: str) -> str:
    """'2_20200916_085235_I1000225' → '20200916_085235' (같은 주행 = 같은 시퀀스)."""
    parts = key.split("_")
    return "_".join(parts[1:3])


def parse_lanes(xml_bytes: bytes) -> tuple[float, float, list[Line]]:
    r = ET.fromstring(xml_bytes)
    W, H = float(r.findtext("size/width")), float(r.findtext("size/height"))
    out: list[Line] = []
    for ln in r.iter("line"):
        cp = ln.find("controlPt")
        if cp is None:
            continue
        xs = [float(e.text) for e in cp.findall("x")]
        ys = [float(e.text) for e in cp.findall("y")]
        if len(xs) >= 2 and len(xs) == len(ys):
            out.append(((ln.findtext("name") or "").strip(), list(zip(xs, ys))))
    return W, H, out


def x_at(pts, y: float) -> float | None:
    """폴리라인에서 y의 x. 구간 안 선형 보간만 — 범위 밖은 None."""
    p = sorted(pts, key=lambda q: q[1])
    ys = np.array([q[1] for q in p])
    xs = np.array([q[0] for q in p])
    if y < ys[0] or y > ys[-1]:
        return None
    return float(np.interp(y, ys, xs))


def _color(nm: str) -> str:
    return "Y" if "Yellow" in nm else ("W" if "White" in nm else nm)


def merge_dashes(lanes: list[Line], W: float, H: float, rules: LabelRules = LabelRules()) -> list[Line]:
    segs = [(nm, sorted(p, key=lambda q: q[1])) for nm, p in lanes if len(p) >= 2]
    parent = list(range(len(segs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def lin_x(p, y):
        ys, xs = np.array([q[1] for q in p]), np.array([q[0] for q in p])
        return float(xs.mean()) if ys[-1] - ys[0] < 1 else float(np.polyval(np.polyfit(ys, xs, 1), y))

    for i, (na, a) in enumerate(segs):
        for j, (nb, b) in enumerate(segs):
            if i == j or _color(na) != _color(nb):
                continue
            gap = a[0][1] - b[-1][1]                      # a 위끝 - b 아래끝 (b가 a보다 위 = 먼 쪽)
            if not (-0.01 * H <= gap <= rules.gap_max * H):
                continue
            ym = (a[0][1] + b[-1][1]) / 2
            if abs(lin_x(a, ym) - lin_x(b, ym)) <= rules.join_tol * W:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(segs)):
        groups[find(i)].append(i)
    out: list[Line] = []
    for gi in groups.values():
        pts = sorted([q for i in gi for q in segs[i][1]], key=lambda q: q[1])
        out.append((segs[gi[0]][0] + (f" ×{len(gi)}" if len(gi) > 1 else ""), pts))
    return out


def cands(lines: list[Line], W: float, H: float, rules: LabelRules = LabelRules()) -> list[Cand]:
    yb, yt = rules.band_bottom, rules.band_top
    c: list[Cand] = []
    for nm, p in lines:
        ys = np.array([q[1] for q in p])
        xs = np.array([q[0] for q in p])
        cov = min(ys.max(), yb * H) - max(ys.min(), yt * H)
        if cov < rules.cover_min * (yb - yt) * H or ys.max() - ys.min() < 1:
            continue
        xb = x_at(p, yb * H)
        if xb is None:
            near = ys >= np.percentile(ys, 50)            # 아래쪽 절반 점으로 직선 연장
            f = np.polyfit(ys[near], xs[near], 1) if np.ptp(ys[near]) > 1 else np.polyfit(ys, xs, 1)
            xb = float(np.polyval(f, yb * H))
        c.append((xb, x_at(p, yt * H), nm, p))
    return c


def clusters(c: list[Cand], W: float, H: float, rules: LabelRules = LabelRules()) -> list[Cand]:
    """겹선끼리 묶어 한 선으로. 묶인 선은 겹치는 행에서 x 평균."""
    c = sorted(c, key=lambda q: q[0])
    groups: list[list[Cand]] = []
    for q in c:
        if groups and q[0] - groups[-1][-1][0] < rules.double * W:
            groups[-1].append(q)
        else:
            groups.append([q])
    out: list[Cand] = []
    for gm in groups:
        if len(gm) == 1:
            out.append(gm[0])
            continue
        ps = [m[3] for m in gm]
        ys = sorted({q[1] for p in ps for q in p})
        av = [(float(np.mean([x_at(p, y) for p in ps])), y) for y in ys if all(x_at(p, y) is not None for p in ps)]
        if len(av) < 2:
            out.append(max(gm, key=lambda m: np.ptp([q[1] for q in m[3]])))
            continue
        out.append((float(np.mean([m[0] for m in gm])), x_at(av, rules.band_top * H), "겹선", av))
    return out


def ego_lanes(lanes: list[Line], W: float, H: float, rules: LabelRules = LabelRules()) -> list[Cand | None] | None:
    """차선 4개 [왼쪽 옆, 자차 좌, 자차 우, 오른쪽 옆] (옆은 없으면 None). 자차 쌍이 없거나 폭 이상이면 None.

    노트북 19번 `ego3` + 학습 데이터 만들 때의 l4 구성과 같다.
    """
    cl = clusters(cands(merge_dashes(lanes, W, H, rules), W, H, rules), W, H, rules)
    li = [i for i, q in enumerate(cl) if q[0] < W / 2]
    ri = [i for i, q in enumerate(cl) if q[0] >= W / 2]
    if not li or not ri:
        return None
    l, r = li[-1], ri[0]
    if not (rules.width_min <= (cl[r][0] - cl[l][0]) / W <= rules.width_max):
        return None
    return [cl[l - 1] if l > 0 else None, cl[l], cl[r], cl[r + 1] if r + 1 < len(cl) else None]


def row_labels(l4: list[Cand | None], W: float, H: float, grid: int, anchors: np.ndarray,
               flip: bool = False) -> np.ndarray:
    """행 앵커마다 차선 4개의 격자 칸 번호 (n_row, 4), -1 = 없음.

    원 저장소처럼 라벨 아래쪽 끝은 화면 아래까지 직선 연장한다. flip=True면 좌우 뒤집은 화면의 정답
    (차선 번호도 0↔3, 1↔2로 바뀐다).
    """
    n = len(anchors)
    lab = np.full((n, 4), -1, np.int64)
    ya = np.asarray(anchors) * H
    for li, q in enumerate(l4):
        if q is None:
            continue
        ps = sorted(q[3], key=lambda t: t[1])
        ys = np.array([t[1] for t in ps])
        xs = np.array([t[0] for t in ps])
        x = np.full(n, np.nan)
        ins = (ya >= ys[0]) & (ya <= ys[-1])
        x[ins] = np.interp(ya[ins], ys, xs)
        below = ya > ys[-1]
        if below.any() and np.ptp(ys) > 1:
            half = ys >= np.median(ys)
            f = np.polyfit(ys[half], xs[half], 1) if np.ptp(ys[half]) > 1 else np.polyfit(ys, xs, 1)
            x[below] = np.polyval(f, ya[below])
        if flip:
            x = W - x
        ok = np.isfinite(x) & (x >= 0) & (x <= W)
        lab[ok, 3 - li if flip else li] = np.minimum((x[ok] / W * (grid - 1)).astype(np.int64), grid - 1)
    return lab
