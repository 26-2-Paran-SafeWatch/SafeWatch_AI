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
    # 옆 차선 폭 검사 (2026-11-09, 노트북 27번 v2 ADJ_RATIO27) — 옆 차로 폭 ÷ 자차 차로 폭(같은 행 중앙값)이 이 범위
    # 밖이면 그 옆 차선은 이중선·도류대·갓길 선으로 보고 뺀다. 라벨 분포: Training 84%가 0.8~1.25, 12%가 0.6 미만
    adj_ratio_min: float = 0.6
    adj_ratio_max: float = 1.6


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


def drop_false_adjacent(l4: list[Cand | None], W: float, H: float,
                        rules: LabelRules = LabelRules()) -> tuple[list[Cand | None], int]:
    """옆 차선(0·3번) 중 옆 차로 폭이 자차 차로 폭과 너무 다른 것을 None으로 바꾼다 → (새 l4, 뺀 개수).

    라벨의 옆 차선은 '자차 차선 바로 바깥 선'이라 이중선 사이·도류대·갓길 선이 자주 잡힌다(옆 차로가 아주 좁음).
    판정 구간(band_top~band_bottom) 12행에서 옆 폭 ÷ 자차 폭의 중앙값으로 본다. 세 선이 함께 있는 행이
    3개 미만이면 판단하지 않고 둔다. 자차 차선(1·2번)은 건드리지 않는다.
    """
    out, dropped = list(l4), 0
    ys = np.linspace(rules.band_top * H, rules.band_bottom * H, 12)
    for outer, inner, other in ((0, 1, 2), (3, 2, 1)):
        if l4[outer] is None:
            continue
        rs = []
        for y in ys:
            a, b, c = x_at(l4[outer][3], y), x_at(l4[inner][3], y), x_at(l4[other][3], y)
            if a is not None and b is not None and c is not None and abs(b - c) > 1:
                rs.append(abs(a - b) / abs(b - c))
        if len(rs) >= 3 and not (rules.adj_ratio_min <= float(np.median(rs)) <= rules.adj_ratio_max):
            out[outer] = None
            dropped += 1
    return out, dropped


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


# ── 기하 증강용 정답 (2026-11-08) ─────────────────────────────────────────────
# 사진을 이동·회전·확대하면 행 앵커 위치의 차선 x가 바뀌므로, 칸 번호(row_labels)가 아니라 차선 선 자체를
# 저장해 두고 학습 때 같은 변환을 적용한 뒤 행 앵커에서 다시 읽는다 (원 저장소가 라벨 점을 같이 변환하는 것과 같은 방식).
# 변환이 없으면 row_labels와 같은 칸 번호가 나온다 (tests/test_ufld_training.py로 대조).

_EXT_EPS = 1e-3     # 아래쪽 직선 연장이 시작되는 지점 (픽셀) — row_labels와 같은 꺾임을 재현하려고 점을 하나 더 둔다


def lane_polylines(l4: list[Cand | None], W: float, H: float, max_pts: int) -> np.ndarray:
    """차선 4개를 (4, max_pts, 2) 배열(x/W, y/H — 원본 화면 비율, 빈 칸 NaN)로.

    row_labels처럼 라벨 아래끝 밑은 화면 아래(y = H)까지 직선 연장한 점을 붙인다.
    """
    out = np.full((4, max_pts, 2), np.nan, np.float32)
    for li, q in enumerate(l4):
        if q is None:
            continue
        ps = sorted(q[3], key=lambda t: t[1])
        ys = np.array([t[1] for t in ps], np.float64)
        xs = np.array([t[0] for t in ps], np.float64)
        pts = list(zip(xs, ys))
        if ys[-1] < H and np.ptp(ys) > 1:
            half = ys >= np.median(ys)
            f = np.polyfit(ys[half], xs[half], 1) if np.ptp(ys[half]) > 1 else np.polyfit(ys, xs, 1)
            y0 = min(ys[-1] + _EXT_EPS, H)
            pts += [(float(np.polyval(f, y0)), y0), (float(np.polyval(f, H)), float(H))]
        if len(pts) > max_pts:
            raise ValueError(f"차선 점 {len(pts)}개 > max_pts {max_pts}")
        a = np.array(pts, np.float64)
        out[li, :len(a)] = np.stack([a[:, 0] / W, a[:, 1] / H], 1)
    return out


def polyline_rows(polys: np.ndarray, anchors: np.ndarray, grid: int, img_w: int, img_h: int,
                  M: np.ndarray | None = None, flip: bool = False) -> np.ndarray:
    """lane_polylines 출력 → 행 앵커별 칸 번호 (n_row, 4), -1 = 없음.

    img_w·img_h = 변환을 적용하는 화면 크기(학습 사진 = 자르기 전 모델 입력 크기). M = 사진에 적용한 2×3 아핀 행렬
    (cv2.warpAffine과 같은 것), flip = 그 뒤 좌우 뒤집기. 변환 뒤 화면 밖으로 나간 행은 -1.
    """
    n = len(anchors)
    lab = np.full((n, 4), -1, np.int64)
    ya = np.asarray(anchors, np.float64) * img_h
    for li in range(4):
        p = polys[li]
        p = p[np.isfinite(p[:, 0])].astype(np.float64)
        if len(p) < 2:
            continue
        xy = p * (img_w, img_h)
        if M is not None:
            xy = xy @ np.asarray(M, np.float64)[:, :2].T + np.asarray(M, np.float64)[:, 2]
        if flip:
            xy[:, 0] = img_w - xy[:, 0]
        xy = xy[np.argsort(xy[:, 1], kind="stable")]
        x = np.full(n, np.nan)
        ins = (ya >= xy[0, 1]) & (ya <= xy[-1, 1])
        x[ins] = np.interp(ya[ins], xy[:, 1], xy[:, 0])
        ok = np.isfinite(x) & (x >= 0) & (x <= img_w)
        lab[ok, 3 - li if flip else li] = np.minimum((x[ok] / img_w * (grid - 1)).astype(np.int64), grid - 1)
    return lab
