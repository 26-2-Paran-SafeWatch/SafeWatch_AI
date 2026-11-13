
# ============================== 1번 셀 ==============================
from google.colab import drive
drive.mount('/content/drive')

import re, json, shutil, zipfile, random
import unicodedata as ud
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter

import cv2, numpy as np
import matplotlib.pyplot as plt

DRIVE_ROOT = Path('/content/drive/MyDrive/SafeWatch')
VALID_DIR  = DRIVE_ROOT / '도로주행영상' / 'Validation'
WEIGHTS    = DRIVE_ROOT / 'runs/vehicle_v1/weights/best.pt'

# 추출 조건 — 바꾸면 TAG가 바뀌어 새로 추출·백업한다 (이전 백업은 그대로 남는다)
ONLY_FOV = '60_전방'   # 이 화각만 사용. None이면 전 화각
N_LANE   = 2000        # zip 하나에서 추출할 최대 프레임 수
# TAG에 '60_전방' 형태를 넣지 않는다 — fov_of 정규식이 경로 속 이 문자열을 화각으로 읽어버린다
TAG      = f"fov{ONLY_FOV.split('_')[0] if ONLY_FOV else 'all'}_n{N_LANE}"

LANE      = Path(f'/content/lane_gt_{TAG}')          # 세션 로컬 — 끊기면 소실
BACKUP    = DRIVE_ROOT / f'lane_gt_{TAG}.tar'        # 드라이브 백업 — 생존
CALIB_OUT = DRIVE_ROOT / f'lane_calib_{TAG}.json'

N_MIN       = 20     # 화각별 최소 채택 수 (중앙값 안정성)
STRAIGHT_PX = 15     # 직선 판정 임계 (px, 1920 기준)
SYM_TOL     = 0.12   # 자차 차로 좌우 대칭 허용 오차

# 보정용 / 검증용 분리 — 시퀀스(연속 촬영) 단위
HOLDOUT     = 0.5              # 검증에만 쓰는 시퀀스 비율
EXCLUDE_CAL = ('2_20200916',)  # 카메라 장착이 다른 차량 — 보정에서 빼고 "보정 틀어짐" 시험에만 쓴다
TOP_MAX     = 0.60             # 보정 구간 윗변이 화면에서 이 높이(비율)까지는 올라와야 한다.
                               # 조감도는 이 구간만 보여주므로 낮을수록 먼 차량을 못 잰다. None이면 제한 없음
MAX_ERR     = 0.10   # 8번 셀 허용 중심 오차 (차로 폭 대비, 3.5m 차로에서 0.35m) — 잠정값

# 조감도
BEV_W, BEV_H, MARGIN = 256, 384, 0.18
TOPHAT_W, NWIN, MINPIX, WIN_M = 31, 10, 30, 35
W_EXP = (1 - 2 * MARGIN) * BEV_W     # 올바른 자차 차로가 조감도에서 가질 폭

# 차선 검출 (7번 셀) — 10번 진단 셀의 C 방식
WHITE_S_MAX = 70          # HLS S 상한 — 흰 차선은 무채색. 분홍·초록 색깔 유도선 제외
PEAK_MIN    = 10          # 히스토그램 피크 최소 높이
PEAK_NMS    = 12          # 이 거리(px) 안의 피크는 하나로 본다
W_RANGE     = (0.5, 1.5)  # 차선 쌍으로 인정할 폭 (W_EXP 배수). 최종 판단은 신뢰도의 폭 타당성 항

# 개발 기준 화각 — 4번 셀 결과를 보고 정한다
G_DEV    = '60_전방'
MIN_CONF = 0.50      # 2026-11-02 결정 — 보정 맞는 프레임에서 오차 초과 conf 최고 0.389 대비 여유 0.11
MAX_LANE_INDEX = 1   # 인접 차로까지만 판단

nfc    = lambda s: ud.normalize('NFC', s)                      # 드라이브 경유 자모 결합
norm   = lambda s: re.sub(r'_v\d+_\d+$', '', Path(s).stem)    # XML 버전 접미사 제거
fov_of = lambda p: (m.group(0) if (m := re.search(r'\d+_(전방|측방)', nfc(str(p)))) else '기타')
seq_of = lambda p: '_'.join(Path(p).stem.split('_')[:3])      # 예: 2_20200916_082638

print('VALID_DIR 존재:', VALID_DIR.exists(), '/ 가중치:', WEIGHTS.exists())
print('데이터셋 TAG:', TAG)
for z in sorted(VALID_DIR.rglob('*.zip')):
    print(f'  {z.name:<24}{z.stat().st_size / 2**30:>6.1f} GB')

# ============================== 2번 셀 ==============================
def extract_lane_set():
    shutil.rmtree(LANE, ignore_errors=True)
    (LANE / 'img').mkdir(parents=True)
    (LANE / 'xml').mkdir(parents=True)
    is_sl  = lambda c: c.startswith('sl/') or '/sl/' in c
    in_fov = lambda c: ONLY_FOV is None or fov_of(c) == ONLY_FOV

    want = {}
    for z in sorted(VALID_DIR.rglob('*.zip')):            # 이미지
        with zipfile.ZipFile(z) as zf:
            sl = [(o, nfc(o)) for o in zf.namelist()
                  if is_sl(nfc(o)) and in_fov(nfc(o))
                  and nfc(o).lower().endswith(('.jpg', '.png'))]
            if not sl:
                continue
            random.seed(42)
            take = random.sample(sl, min(N_LANE, len(sl)))
            for o, c in take:
                zf.extract(o, LANE / 'img')
                want[Path(c).stem] = o
            print(f'이미지 {len(take):>4}개 (후보 {len(sl)}개)  <- {z.name}')
    assert want, (f'{ONLY_FOV} 이미지를 찾지 못함 — zip 내부 경로에 화각 이름이 없을 수 있다. '
                  'ONLY_FOV = None으로 바꿔 전 화각을 추출할 것')

    got = 0
    for z in sorted(VALID_DIR.rglob('*.zip')):            # 라벨
        with zipfile.ZipFile(z) as zf:
            xs = [(o, nfc(o)) for o in zf.namelist()
                  if is_sl(nfc(o)) and nfc(o).lower().endswith('.xml')]
            if not xs:
                continue
            for o, c in xs:
                if norm(c) in want:
                    zf.extract(o, LANE / 'xml')
                    got += 1
            print(f'XML  {got:>4}개  <- {z.name}')


if LANE.exists() and any((LANE / 'xml').rglob('*.xml')):
    print('기존 데이터 사용')
elif BACKUP.exists():
    print('드라이브 백업에서 복원 중...')
    LANE.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(BACKUP), str(LANE), 'tar')
else:
    extract_lane_set()
    print('드라이브에 백업 저장 중...')
    shutil.make_archive(str(BACKUP.with_suffix('')), 'tar', str(LANE))
    print('저장 완료:', BACKUP)

imgs = {p.stem: p for p in (LANE / 'img').rglob('*.jpg')}
print('\n이미지', len(imgs), '/ XML', len(list((LANE / 'xml').rglob('*.xml'))))

# ============================== 3번 셀 ==============================
def parse_lanes(xp):
    r = ET.parse(xp).getroot()
    W, H = float(r.findtext('size/width')), float(r.findtext('size/height'))
    out = []
    for ln in r.iter('line'):
        cp = ln.find('controlPt')
        if cp is None:
            continue
        xs = [float(e.text) for e in cp.findall('x')]
        ys = [float(e.text) for e in cp.findall('y')]
        if len(xs) >= 2 and len(xs) == len(ys):
            out.append(((ln.findtext('name') or '').strip(), list(zip(xs, ys))))
    return W, H, out


def x_at(pts, y):
    """폴리라인에서 y의 x. 구간 내 선형보간만 — 범위 밖은 None (외삽 금지)."""
    p = sorted(pts, key=lambda q: q[1])
    ys = np.array([q[1] for q in p])
    xs = np.array([q[0] for q in p])
    if y < ys[0] or y > ys[-1]:
        return None
    return float(np.interp(y, ys, xs))


def straightness_band(pts, H, yt, yb, n=9):
    """구간을 균등 분할한 보간 샘플로 직선성 판정 — 제어점 개수에 무관."""
    ys = np.linspace(yt * H, yb * H, n)
    xs = [x_at(pts, y) for y in ys]
    if any(v is None for v in xs):
        return 1e9
    xs = np.array(xs)
    a, b = np.polyfit(ys, xs, 1)
    return float(np.abs(xs - (a * ys + b)).max())


names, npl = Counter(), []
for xp in (LANE / 'xml').rglob('*.xml'):
    _, _, ls = parse_lanes(xp)
    npl.append(len(ls))
    for nm, _ in ls:
        names[nm] += 1

print(f'프레임당 평균 차선 {sum(npl) / max(len(npl), 1):.1f}개\n')
print(f"{'차선 종류':<26}{'개수':>7}")
for k, v in names.most_common():
    print(f'{k:<26}{v:>7}')

# ============================== 4번 셀 ==============================
SPAN_MAX = 0.42   # 보정 구간 최대 폭 — 0.42면 0.98~0.56 (멀리까지), 0.34면 0.98~0.64 (가까이만)

FRAMES = {}
for xp in sorted((LANE / 'xml').rglob('*.xml')):
    g = fov_of(xp.relative_to(LANE))     # LANE 기준 상대 경로 — 로컬 폴더 이름이 화각으로 오인되지 않게
    if ONLY_FOV and g != ONLY_FOV:
        continue                         # 파일명이 같아 함께 추출된 다른 화각 라벨 제외
    W, H, lanes = parse_lanes(xp)
    if lanes:
        FRAMES[norm(xp.name)] = (xp, g, W, H, lanes)   # 같은 프레임의 라벨 버전(_v1_0, _v1_1…)은 마지막 것만
FRAMES = list(FRAMES.values())
print(f'프레임 {len(FRAMES)}개 로드\n')


def ego_pair(lanes, W, H, yb, yt):
    """구간 [yt, yb]를 온전히 덮는 선만 후보로 삼아 자차 차로 좌/우 경계를 고른다."""
    c = []
    for nm, p in lanes:
        xb, xt = x_at(p, yb * H), x_at(p, yt * H)
        if xb is None or xt is None:
            continue
        c.append((xb, xt, nm, p))
    L = [q for q in c if q[0] < W / 2]
    R = [q for q in c if q[0] >= W / 2]
    if not L or not R:
        return None
    return max(L, key=lambda q: q[0]), min(R, key=lambda q: q[0])


def try_band(frames, yb, yt):
    out = []
    for xp, g, W, H, lanes in frames:
        pair = ego_pair(lanes, W, H, yb, yt)
        if pair is None:
            continue
        (lx, ltx, lnm, lp), (rx, rtx, rnm, rp) = pair
        if max(straightness_band(lp, H, yt, yb),
               straightness_band(rp, H, yt, yb)) > STRAIGHT_PX:
            continue                                   # 곡선 구간 제외
        if not (0 < lx < rx < W and 0 < ltx < rtx < W):
            continue                                   # 좌우 뒤집힘
        if not (0.25 < (rx - lx) / W < 0.95):
            continue                                   # 차로 폭 이상치
        if abs((lx + rx) / 2 / W - 0.5) > SYM_TOL:
            continue                                   # 자차가 차로 중앙에서 이탈
        out.append((xp, [lx / W, rx / W, rtx / W, ltx / W],
                    (W, H, lanes, pair, yb, yt), lnm, rnm))
    return out


def gt_pair(frames, yb, yt):
    """검증용 정답 쌍. 보정 채택 조건(구간 전체를 덮는 직선·대칭)은 걸지 않는다.
    구간이 넓으면 위아래 끝까지 이어진 라벨이 드물어 검증 표본이 크게 줄기 때문에,
    구간 안에서 아래쪽부터 좌우 선이 모두 있는 첫 행에서 자차 차로 쌍을 고르고
    오차는 8번 셀에서 라벨이 있는 행만으로 잰다. full은 구간 전체를 덮는지."""
    out = []
    for xp, g, W, H, lanes in frames:
        pair = None
        for y in np.linspace(yb, yt, 7):               # 아래쪽 행부터
            pair = ego_pair(lanes, W, H, y, y)
            if pair is not None:
                break
        if pair is None:
            continue
        (lx, _, _, lp), (rx, _, _, rp) = pair
        if not (0 < lx < rx < W):
            continue
        full = all(x_at(p, y * H) is not None for p in (lp, rp) for y in (yb, yt))
        out.append((xp, g, (W, H, lanes, pair, yb, yt), full))
    return out


def split_by_seq(frames, rng):
    """시퀀스(연속 촬영) 단위로 보정용 / 검증용 / 보정 제외로 나눈다.
    같은 시퀀스의 거의 같은 프레임이 양쪽에 들어가면 검증이 낙관적이 된다."""
    seqs = sorted({seq_of(f[0]) for f in frames})
    excl = {s for s in seqs if s.startswith(EXCLUDE_CAL)}
    rest = [s for s in seqs if s not in excl]
    rng.shuffle(rest)
    n_hold = max(1, round(len(rest) * HOLDOUT))
    hold_s, cal_s = set(rest[:n_hold]), set(rest[n_hold:])
    pick = lambda S: [f for f in frames if seq_of(f[0]) in S]
    return pick(cal_s), pick(hold_s), pick(excl), (len(cal_s), len(hold_s), len(excl))


SPANS = [sp for sp in (0.42, 0.38, 0.34, 0.30, 0.26, 0.22, 0.18) if sp <= SPAN_MAX + 1e-9]
REF, acc, types, keep, CALIB, HOLD, MIS = {}, {}, defaultdict(Counter), [], {}, [], []
rng = random.Random(0)

for g in sorted({f[1] for f in FRAMES}):
    fr_all = [f for f in FRAMES if f[1] == g]
    if len(fr_all) < 10:
        print(f'{g:<12} 프레임 {len(fr_all)}개 — 건너뜀\n'); continue
    fr, hold_fr, mis_fr, (n_c, n_h, n_x) = split_by_seq(fr_all, rng)
    if len(fr) < 10:
        print(f'{g:<12} 보정용 {len(fr)}장 — 건너뜀\n'); continue
    depth = [max(q[1] for q in p) / f[3] for f in fr for nm, p in f[4]]
    # 아랫변 후보 — 라벨 끝 위치 분위수 + 고정 격자. 분위수만 쓰면 라벨이 대부분 화면 끝(0.98)까지
    # 내려올 때 아랫변이 0.98로만 잡혀 TOP_MAX를 만족하는 구간이 아예 안 생긴다
    YB = sorted({round(min(float(np.percentile(depth, pct)), 0.98), 3) for pct in range(10, 90, 5)}
                | {round(float(v), 2) for v in np.arange(0.70, 0.99, 0.02)})

    best, fallback = None, None
    for sp in SPANS:                                   # 넓은 구간부터
        cands = []
        for yb in YB:
            yt = round(yb - sp, 3)
            if yt < 0.22 or (TOP_MAX is not None and yt > TOP_MAX):
                continue                               # 너무 높거나, 먼 곳까지 못 덮는 구간
            cands.append((yb, yt, try_band(fr, yb, yt)))
        if not cands:
            continue
        yb, yt, res = max(cands, key=lambda c: len(c[2]))
        if fallback is None or len(res) > len(fallback[2]):
            fallback = (yb, yt, res, sp)
        if len(res) >= N_MIN:                          # 최소 표본 충족 → 확정
            best = (yb, yt, res, sp); break
    if best is None:
        best = fallback
    if best is None:
        print(f'{g:<12} 조건을 만족하는 구간 없음 — TOP_MAX를 완화할 것\n'); continue
    yb, yt, res, sp = best

    REF[g] = (yb, yt); acc[g] = [r[1] for r in res]
    for r in res:
        keep.append((r[0], g, r[2])); types[g][r[3]] += 1; types[g][r[4]] += 1
    hold, mis = gt_pair(hold_fr, yb, yt), gt_pair(mis_fr, yb, yt)
    HOLD.extend(hold); MIS.extend(mis)

    print(f'{g:<12} 보정용   {n_c:>3}시퀀스 {len(fr):>5}장  채택 {len(res):>4} '
          f'({len(res)/len(fr)*100:>3.0f}%)  구간 {yb} ~ {yt} (span {sp})')
    print(f'{"":12} 검증용   {n_h:>3}시퀀스 {len(hold_fr):>5}장  정답쌍 {len(hold):>4} '
          f'(구간 전체 {sum(h[3] for h in hold)})')
    print(f'{"":12} 보정제외 {n_x:>3}시퀀스 {len(mis_fr):>5}장  정답쌍 {len(mis):>4} '
          f'(구간 전체 {sum(h[3] for h in mis)})  {EXCLUDE_CAL}')
    if not res:
        print(); continue
    if len(res) < N_MIN:
        print(f'{"":12} 주의 — 표본 {len(res)}개로 최소 {N_MIN} 미달')
    a = np.median(np.array(acc[g]), axis=0)
    wb, wt = a[1] - a[0], a[2] - a[3]
    CALIB[g] = dict(n=len(res), bot_y=yb, top_y=yt,
                    src=[[round(float(a[0]), 4), yb], [round(float(a[1]), 4), yb],
                         [round(float(a[2]), 4), yt], [round(float(a[3]), 4), yt]])
    print(f'{"":12} src=[({a[0]:.3f},{yb}), ({a[1]:.3f},{yb}), '
          f'({a[2]:.3f},{yt}), ({a[3]:.3f},{yt})]')
    print(f'{"":12} 아랫변 {wb:.3f}  윗변 {wt:.3f}  비 {wt/wb:.2f}  '
          f'중심 {(a[0]+a[1])/2:.3f}')
    print(f'{"":12} 경계 차선: {dict(types[g])}\n')

print('표본 최다:', max(CALIB, key=lambda k: CALIB[k]['n']) if CALIB else '없음')
print(f'개발 기준 화각 G_DEV = {G_DEV}')

# ============================== 5번 셀 ==============================
COLOR = {'Lane_White_Solid':  (255, 255, 255), 'Lane_White_Dash':  (180, 180, 180),
         'Lane_Yellow_Solid': (0, 215, 255),   'Lane_Yellow_Dash': (0, 160, 200),
         'Lane_Blue_Solid':   (255, 120, 0),   'Lane_Blue_Dash':   (200, 90, 0)}

N_SHOW, shown = 6, 0
for xp, g, (W, H, lanes, pair, yb, yt) in keep:
    ip = imgs.get(norm(xp.name))
    if ip is None:
        continue
    im = cv2.imread(str(ip))
    if im is None:
        continue

    for nm, p in lanes:
        cv2.polylines(im, [np.int32(p)], False, COLOR.get(nm, (0, 0, 255)), 2)
    for (_, _, nm, p), c in zip(pair, [(0, 0, 255), (0, 255, 0)]):
        cv2.polylines(im, [np.int32(p)], False, c, 6)

    (lx, ltx, lnm, _), (rx, rtx, rnm, _) = pair
    cv2.polylines(im, [np.int32([(lx, yb * H), (rx, yb * H),
                                 (rtx, yt * H), (ltx, yt * H)])],
                  True, (0, 255, 255), 3)

    plt.figure(figsize=(10, 5.5))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title(f'{g}   좌:{lnm}  우:{rnm}   |   {ip.name}', fontsize=9)
    plt.show()

    shown += 1
    if shown == N_SHOW:
        break

# ============================== 6번 셀 ==============================
CALIB_OUT.write_text(json.dumps(CALIB, ensure_ascii=False, indent=2), encoding='utf-8')
print('저장:', CALIB_OUT)
print(json.dumps(CALIB, ensure_ascii=False, indent=2))

# ============================== 7번 셀 ==============================
!pip -q install ultralytics

# ============================== 8번 셀 ==============================
from ultralytics import YOLO

SRC = [tuple(p) for p in CALIB[G_DEV]['src']]
LBL = {0: '자차 차로', -1: '좌측 차로', 1: '우측 차로'}


def make_H(w, h, src):
    s = np.float32([(x * w, y * h) for x, y in src])
    m = MARGIN * BEV_W
    d = np.float32([(m, BEV_H), (BEV_W - m, BEV_H), (BEV_W - m, 0), (m, 0)])
    return cv2.getPerspectiveTransform(s, d)


def bev_mask(bgr, Hm):
    bev = cv2.warpPerspective(bgr, Hm, (BEV_W, BEV_H))
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    L, S = hls[:, :, 1], hls[:, :, 2]

    k  = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_W, 1))
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
    t, _ = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, white = cv2.threshold(th, max(int(t), 15), 255, cv2.THRESH_BINARY)
    white[S > WHITE_S_MAX] = 0                         # 유채색 제외 — 분홍·초록 색깔 유도선

    m = cv2.bitwise_or(white, cv2.inRange(hls, (15, 60, 60), (38, 255, 255)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))

    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):                              # 노면 화살표·문자 제거
        w_ = st[i, cv2.CC_STAT_WIDTH]
        h_ = st[i, cv2.CC_STAT_HEIGHT]
        a_ = st[i, cv2.CC_STAT_AREA]
        if h_ >= 25 and h_ >= 1.8 * w_ and a_ >= 40:
            out[lab == i] = 255
    return bev, out


def find_peaks(hist):
    """히스토그램 국소 최대 — 강한 것부터 남기고 PEAK_NMS 안의 약한 것은 버린다."""
    hs = np.convolve(hist, np.ones(5) / 5, mode='same')
    idx = [i for i in range(1, len(hs) - 1)
           if hs[i] >= PEAK_MIN and hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]]
    kept = []
    for i in sorted(idx, key=lambda i: -hs[i]):
        if all(abs(i - j) > PEAK_NMS for j in kept):
            kept.append(i)
    return sorted(kept)


def pick_pair(peaks):
    """중앙을 사이에 둔 쌍 중 폭이 W_RANGE 안에 드는 가장 좁은(= 가장 안쪽) 쌍.
    난간·가드레일·터널 벽은 바깥에 있으므로 안쪽 선이 있으면 그쪽을 고른다.
    가장 강한 피크를 고르던 이전 방식은 이런 구조물을 잡고도 신뢰도가 높게 나왔다."""
    mid = BEV_W // 2
    pairs = [(l, r) for l in peaks if l < mid for r in peaks if r >= mid
             and W_RANGE[0] * W_EXP <= r - l <= W_RANGE[1] * W_EXP]
    return min(pairs, key=lambda p: p[1] - p[0]) if pairs else None


def fit_lanes(m):
    hist = m[BEV_H // 2:].sum(axis=0) / 255.0
    pair = pick_pair(find_peaks(hist))
    if pair is None:
        return None, None, 0.0, None

    lx, rx = pair
    ys, xs = m.nonzero()
    wh, pts = BEV_H // NWIN, {'l': [], 'r': []}
    for i in range(NWIN):
        y0, y1 = BEV_H - (i + 1) * wh, BEV_H - i * wh
        for key in ('l', 'r'):
            cx = lx if key == 'l' else rx
            sel = (ys >= y0) & (ys < y1) & (xs >= cx - WIN_M) & (xs < cx + WIN_M)
            if sel.sum() > MINPIX:
                nx = int(xs[sel].mean())
                pts[key].append((nx, (y0 + y1) / 2))
                if key == 'l': lx = nx
                else:          rx = nx

    if len(pts['l']) < 4 or len(pts['r']) < 4:
        return None, None, 0.0, pair

    fl = np.polyfit([p[1] for p in pts['l']], [p[0] for p in pts['l']], 2)
    fr = np.polyfit([p[1] for p in pts['r']], [p[0] for p in pts['r']], 2)

    w  = np.array([np.polyval(fr, y) - np.polyval(fl, y)
                   for y in np.linspace(0, BEV_H, 8)])
    wm = float(np.median(w))
    cons  = 1.0 - min(1.0, float(np.std(w) / max(abs(wm), 1e-6)) * 3)  # 폭 일관성
    plaus = float(np.clip(1 - abs(wm / W_EXP - 1) / 0.40, 0, 1))       # 폭 타당성
    conf  = (len(pts['l']) + len(pts['r'])) / (2 * NWIN) * max(cons, 0.0) * plaus
    return fl, fr, round(float(conf), 3), pair


det = YOLO(str(WEIGHTS))
print(f'{G_DEV}  src={SRC}')
print(f'기대 피크 ≈ {int(MARGIN*BEV_W)}, {int((1-MARGIN)*BEV_W)}   '
      f'min_confidence={MIN_CONF}\n')


def analyze(bgr, Hm):
    bev, m = bev_mask(bgr, Hm)
    fl, fr, conf, pk = fit_lanes(m)
    if fl is None or conf < MIN_CONF:
        return bev, m, None, conf, pk, []              # 차선 불명 → 판단 보류

    r = det.predict(bgr, imgsz=320, conf=0.25, verbose=False)[0]
    out = []
    for b in r.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        gp = np.array([[[(x1 + x2) / 2, y2]]], np.float32)   # 노면 접지점
        bx, by = cv2.perspectiveTransform(gp, Hm)[0][0]
        if not (0 <= by < BEV_H):
            continue                                   # 유효 관측 구간 밖
        xl, xr = float(np.polyval(fl, by)), float(np.polyval(fr, by))
        c, w = (xl + xr) / 2, (xr - xl)
        if w <= 1:
            continue
        raw = (bx - c) / w
        idx = int(round(raw))
        if abs(idx) > MAX_LANE_INDEX:
            continue                                   # 두 차로 밖은 외삽 신뢰 불가
        out.append(dict(box=(x1, y1, x2, y2), bev=(bx, by),
                        lane_index=idx, lane_offset=round(raw - idx, 3)))
    return bev, m, (fl, fr), conf, pk, out


frames = [(xp, imgs[norm(xp.name)]) for xp, g, _ in keep
          if g == G_DEV and norm(xp.name) in imgs]

for xp, ip in frames[:4]:
    bgr = cv2.imread(str(ip))
    if bgr is None:
        continue
    h, w = bgr.shape[:2]
    Hm = make_H(w, h, SRC)
    bev, m, fit, conf, pk, veh = analyze(bgr, Hm)

    vis = bgr.copy()
    cv2.polylines(vis, [np.int32([(x*w, y*h) for x, y in SRC])], True, (0,255,255), 3)
    for v in veh:                                      # 박스·접지점만 OpenCV로
        x1, y1, x2, y2 = map(int, v['box'])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, ((x1 + x2) // 2, y2), 6, (0, 0, 255), -1)

    ov = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    if fit:
        yy = np.linspace(0, BEV_H - 1, 50)
        for f, c in zip(fit, [(0, 0, 255), (0, 255, 0)]):
            cv2.polylines(ov, [np.int32(np.c_[np.polyval(f, yy), yy])], False, c, 2)
    for v in veh:
        cv2.circle(ov, (int(v['bev'][0]), int(v['bev'][1])), 5, (255, 0, 255), -1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im_, t_ in zip(ax,
            [cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), cv2.cvtColor(bev, cv2.COLOR_BGR2RGB),
             cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)],
            ['검출 + 차로 판정', '조감도', f'차선 피팅  conf={conf}']):
        a.imshow(im_); a.set_title(t_, fontsize=9); a.axis('off')

    for v in veh:                                      # 한글 라벨은 matplotlib으로
        x1, y1, _, _ = v['box']
        ax[0].annotate(f"{LBL.get(v['lane_index'], v['lane_index'])}\n"
                       f"{v['lane_offset']:+.2f}",
                       xy=(x1, max(14, y1 - 6)), color='yellow', fontsize=8,
                       va='bottom', bbox=dict(fc='black', alpha=.6, pad=1, lw=0))

    plt.suptitle(f'{ip.name}   피크 x={pk}   차량 {len(veh)}대', fontsize=9)
    plt.tight_layout(); plt.show()

    for v in veh:
        print(f"   {LBL.get(v['lane_index'], str(v['lane_index'])):<10}"
              f"offset={v['lane_offset']:+.3f}")

# ============================== 9번 셀 ==============================
def fit_lanes_p(m):
    """평행 제약 버전 (8번 셀에서 fit_lanes와 비교).
    조감도에서 두 차선은 평행하므로
    1) 슬라이딩 윈도는 좌우가 같은 만큼만 움직이고 (한쪽이 난간 잡음에 끌려가지 않게)
       윈도 안 위치는 평균이 아니라 가장 촘촘한 세로 열로 잡으며
    2) 두 선을 같은 모양(a·y² + b·y)에 위치만 다른 곡선으로 함께 피팅하며
    3) 잔차가 큰 점을 한 번 걸러낸 뒤 다시 푼다.
    두 선의 폭이 항상 일정하므로 폭 일관성 항 대신 피팅 잔차 항을 쓴다."""
    hist = m[BEV_H // 2:].sum(axis=0) / 255.0
    pair = pick_pair(find_peaks(hist))
    if pair is None:
        return None, None, 0.0, None

    lx, rx = pair
    ys, xs = m.nonzero()
    wh, pts = BEV_H // NWIN, {'l': [], 'r': []}
    for i in range(NWIN):
        y0, y1 = BEV_H - (i + 1) * wh, BEV_H - i * wh
        shift = []
        for key, cx in (('l', lx), ('r', rx)):
            sel = (ys >= y0) & (ys < y1) & (xs >= cx - WIN_M) & (xs < cx + WIN_M)
            if sel.sum() > MINPIX:
                # 평균 대신 가장 촘촘한 세로 열 — 차선은 폭 좁은 연속 줄이라 열 밀도가 가장 높고,
                # 옆 난간·노면 잡음은 넓게 퍼져 평균만 끌어당긴다
                base = int(cx - WIN_M)
                col = np.bincount(xs[sel] - base, minlength=2 * WIN_M + 2)
                col = np.convolve(col, np.ones(3), mode='same')
                pk_x = base + int(np.argmax(col))
                near = sel & (np.abs(xs - pk_x) <= 3)              # 그 열 ±3px 평균 (소수점 위치)
                nx = float(xs[near].mean())
                pts[key].append((nx, (y0 + y1) / 2))
                shift.append(nx - cx)
        if shift:                                      # 좌우 같은 만큼 이동
            d = float(np.mean(shift))
            lx, rx = lx + d, rx + d

    def solve(pl, pr):
        Y = np.array([p[1] for p in pl + pr])
        X = np.array([p[0] for p in pl + pr])
        A = np.c_[Y ** 2, Y, [1] * len(pl) + [0] * len(pr), [0] * len(pl) + [1] * len(pr)]
        coef = np.linalg.lstsq(A, X, rcond=None)[0]
        return coef, X - A @ coef

    pl, pr = pts['l'], pts['r']
    if len(pl) < 4 or len(pr) < 4:
        return None, None, 0.0, pair
    coef, res = solve(pl, pr)
    keep_ = np.abs(res) <= max(3.0, 2.5 * float(np.median(np.abs(res))))   # 잡음 점 제거
    pl = [p for p, k in zip(pl, keep_[:len(pl)]) if k]
    pr = [p for p, k in zip(pr, keep_[len(pts['l']):]) if k]
    if len(pl) < 4 or len(pr) < 4:
        return None, None, 0.0, pair
    coef, res = solve(pl, pr)
    a, b, cl, cr = coef
    fl, fr = np.array([a, b, cl]), np.array([a, b, cr])

    rms   = float(np.sqrt(np.mean(res ** 2)))
    fitq  = 1.0 - min(1.0, rms / (0.05 * W_EXP))                       # 피팅 잔차
    plaus = float(np.clip(1 - abs((cr - cl) / W_EXP - 1) / 0.40, 0, 1))  # 폭 타당성
    conf  = (len(pl) + len(pr)) / (2 * NWIN) * fitq * plaus
    return fl, fr, round(float(conf), 3), pair

# ============================== 10번 셀 ==============================
PROC_WS = [None, 320]        # 평가 입력 폭 — None=원본, 320 = Pi5 처리 해상도 흉내 (종횡비 유지)
METHODS = {'C': fit_lanes, 'P': fit_lanes_p}   # C = 지금 방식, P = 평행 제약 (7번 셀)
N_SHOW  = 4                  # 시각화할 검증용 고신뢰 실패 수


def gt_bev(meta, Hm, s=1.0):
    """정답 자차 차로 좌/우 경계를 조감도 좌표로. 라벨이 있는 행만 쓴다 (9행 중 5행 이상).
    s = 처리 이미지 폭 / 라벨 기준 폭 — 축소 처리할 때 라벨 좌표도 같은 비율로 줄인다."""
    W_, H_, lanes, pair, yb, yt = meta
    lp, rp = pair[0][3], pair[1][3]
    rows = [(x_at(lp, y), x_at(rp, y), y) for y in np.linspace(yt * H_, yb * H_, 9)]
    rows = [(a, b, y) for a, b, y in rows if a is not None and b is not None]
    if len(rows) < 5:
        return None
    sp = np.float32([[a * s, y * s] for a, b, y in rows] +
                    [[b * s, y * s] for a, b, y in rows]).reshape(-1, 1, 2)
    bp = cv2.perspectiveTransform(sp, Hm).reshape(-1, 2)
    return bp[:len(rows)], bp[len(rows):]


def center_err(fl, fr, gt):
    """차로 중심 오차 (차로 폭 대비) — (중앙, 최대)."""
    gL, gR = gt
    gt_c = (gL[:, 0] + gR[:, 0]) / 2
    gt_y = (gL[:, 1] + gR[:, 1]) / 2
    gt_w = gR[:, 0] - gL[:, 0]
    ok = gt_w > 1
    if ok.sum() < 5:
        return None
    fit_c = (np.polyval(fl, gt_y) + np.polyval(fr, gt_y)) / 2
    e = np.abs(fit_c - gt_c)[ok] / gt_w[ok]
    return float(np.median(e)), float(np.max(e))


def load(ip, pw):
    bgr = cv2.imread(str(ip))
    if bgr is None or pw is None:
        return bgr
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (pw, round(h * pw / w)), interpolation=cv2.INTER_AREA)


def evaluate(entries, pw):
    """방법별 결과 목록. 마스크는 프레임당 한 번만 만들고, 저장하지 않는다 (메모리)."""
    rows = {k: [] for k in METHODS}
    for xp, g, meta, full in entries:
        if g != G_DEV:
            continue
        ip = imgs.get(norm(xp.name))
        if ip is None:
            continue
        bgr = load(ip, pw)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        Hm = make_H(w, h, SRC)
        gt = gt_bev(meta, Hm, s=w / meta[0])
        if gt is None:
            continue
        _, m = bev_mask(bgr, Hm)
        gtw = float(gt[1][-1, 0] - gt[0][-1, 0]) / W_EXP              # 정답 폭 (W_EXP 배수)
        for k, fn in METHODS.items():
            fl, fr, conf, pk = fn(m)
            rows[k].append(dict(ip=ip, meta=meta, seq=seq_of(ip), conf=conf, pk=pk, full=full, gtw=gtw,
                                err=center_err(fl, fr, gt) if fl is not None else None))
    return rows


def table(rows, title):
    n = len(rows)
    print(f'\n── {title}: {n}장 / {len({r["seq"] for r in rows})}시퀀스 '
          f'(구간 전체 {sum(r["full"] for r in rows)}, 피팅 실패 {sum(r["err"] is None for r in rows)}) ──')
    if not n:
        return
    w = np.array([r['gtw'] for r in rows])
    print(f'정답 차로 폭 (W_EXP 배수) p10 {np.percentile(w, 10):.2f} / '
          f'p50 {np.median(w):.2f} / p90 {np.percentile(w, 90):.2f}')
    print(f'{"임계":>6}{"통과":>11}{"가용률":>8}{"오차중앙":>9}{"오차최대":>9}{"초과":>6}')
    for th in (0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.00):
        s = [r for r in rows if r['err'] is not None and r['conf'] >= th]
        if not s:
            print(f'{th:>6.2f}{0:>6}/{n:<4}{0:>7.0f}%{"-":>9}{"-":>9}{0:>6}'); continue
        mark = '  <- 현재 설정' if abs(th - MIN_CONF) < 1e-9 else ''
        print(f'{th:>6.2f}{len(s):>6}/{n:<4}{len(s) / n * 100:>7.0f}%'
              f'{np.median([r["err"][0] for r in s]):>9.3f}{max(r["err"][1] for r in s):>9.3f}'
              f'{sum(r["err"][1] > MAX_ERR for r in s):>6}{mark}')


grid = np.round(np.arange(0.30, 0.81, 0.01), 2)
summary, hard_all = [], []
for pw in PROC_WS:
    res_tag = '원본' if pw is None else f'{pw}px'
    hold_all, mis_all = evaluate(HOLD, pw), evaluate(MIS, pw)
    for k in METHODS:
        tag = f'{res_tag}·{k}'
        hold_rows, mis_rows = hold_all[k], mis_all[k]
        print('\n' + '═' * 64 + f'\n입력 {res_tag} / 방법 {k}\n' + '═' * 64)
        table(hold_rows, '검증 — 보정에 쓰지 않은 시퀀스')
        table(mis_rows,  f'보정 틀어짐 시험 — {EXCLUDE_CAL}')

        ok_rows = [r for r in hold_rows if r['err'] is not None]
        bad_c  = [r['conf'] for r in ok_rows if r['err'][1] > MAX_ERR]
        good_c = [r['conf'] for r in ok_rows if r['err'][1] <= MAX_ERR]
        cands = [th for th in grid
                 if (s := [r for r in ok_rows if r['conf'] >= th]) and max(r['err'][1] for r in s) <= MAX_ERR]
        rec = float(min(cands)) if cands else None
        print(f'\n오차 초과(>{MAX_ERR}) {len(bad_c)}장 — conf 최고 ' + (f'{max(bad_c):.3f}' if bad_c else '-')
              + (f'   정상 conf p10 {np.percentile(good_c, 10):.3f} / p50 {np.median(good_c):.3f}' if good_c else ''))
        if bad_c:
            print('오차 기준 최저 임계 ' + (f'{rec:.2f}' if rec is not None else '없음')
                  + f'  (현재 MIN_CONF {MIN_CONF}, 여유 {MIN_CONF - max(bad_c):+.3f})')

        hs = [r for r in ok_rows if r['conf'] >= MIN_CONF]
        ms = [r for r in mis_rows if r['err'] is not None and r['conf'] >= MIN_CONF]
        summary.append((tag, len(hold_rows), len(hs), sum(r['err'][1] > MAX_ERR for r in hs),
                        np.median([r['err'][0] for r in hs]) if hs else float('nan'),
                        (rec if bad_c else '-'), len(ms), sum(r['err'][1] > MAX_ERR for r in ms)))
        hard_all += [(tag, pw, k, r) for r in hs if r['err'][1] > MAX_ERR]

print('\n' + '═' * 72 + f'\n요약 — MIN_CONF {MIN_CONF} 기준 (SPAN_MAX {SPAN_MAX}, 구간 {REF[G_DEV][0]} ~ {REF[G_DEV][1]})\n' + '═' * 72)
print('오답0 임계 = 통과분 오답이 0이 되는 가장 낮은 임계 ("-" = 오답 자체가 없음)')
print(f'{"입력·방법":<10}{"검증 통과":>12}{"가용률":>8}{"오답":>6}{"오차중앙":>9}{"오답0 임계":>10} │ {"틀어짐 통과":>10}{"오답":>6}')
for tag, n, p, b, med, rec, mp, mb in summary:
    print(f'{tag:<10}{p:>7}/{n:<4}{p / max(n, 1) * 100:>7.0f}%{b:>6}{med:>9.3f}'
          f'{(rec if rec == "-" else f"{rec:.2f}" if rec is not None else "없음"):>10} │ {mp:>10}{mb:>6}')

# 검증 시퀀스의 고신뢰 실패 시각화 — 같은 프레임은 한 번만
seen = set()
for tag, pw, k, r in sorted(hard_all, key=lambda t: -t[3]['conf']):
    if r['ip'] in seen:
        continue
    seen.add(r['ip'])
    if len(seen) > N_SHOW:
        break
    bgr = load(r['ip'], pw)
    h, w = bgr.shape[:2]
    Hm = make_H(w, h, SRC)
    bev, m = bev_mask(bgr, Hm)
    gt = gt_bev(r['meta'], Hm, s=w / r['meta'][0])
    panels = [(bgr, f'원본 ({tag})'), (bev, f"조감도 (하늘색 정답)  정답폭 {r['gtw']:.2f}")]
    for kk, fn in METHODS.items():                     # 두 방법을 나란히
        fl, fr, conf, pk = fn(m)
        ov = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        if fl is not None:
            yy = np.linspace(0, BEV_H - 1, 50)
            for f, c in zip((fl, fr), [(0, 0, 255), (0, 255, 0)]):
                cv2.polylines(ov, [np.int32(np.c_[np.polyval(f, yy), yy])], False, c, 2)
        for side in gt:
            cv2.polylines(ov, [np.int32(side)], False, (255, 255, 0), 1)
        e = center_err(fl, fr, gt) if fl is not None else None
        panels.append((ov, f'{kk} conf={conf:.3f} ' + (f'최대오차={e[1]:.3f}' if e else '피팅 실패')))
    for side in gt:
        cv2.polylines(bev, [np.int32(side)], False, (255, 255, 0), 2)
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for a, (im_, t_) in zip(ax, panels):
        a.imshow(cv2.cvtColor(im_, cv2.COLOR_BGR2RGB)); a.set_title(t_, fontsize=9); a.axis('off')
    plt.suptitle(r['ip'].name, fontsize=9)
    plt.tight_layout(); plt.show()

SRC_CFG = [[round(float(v), 4) for v in SRC[i]] for i in (3, 2, 1, 0)]  # 좌상→우상→우하→좌하
print('\n' + '─' * 58)
print('configs/default.yaml 에 반영할 값 (저장소 키 이름 기준)')
print('─' * 58)
print('lane:')
print(f'  min_confidence: {MIN_CONF}')
print('  perspective:')
print(f'    src_points_ratio: {SRC_CFG}   # {G_DEV} 기준(AI Hub), 실기기 재보정 필요')
print(f'    warp_size: [{BEV_W}, {BEV_H}]')
print(f'    margin: {MARGIN}            # 신규 키')
print(f'  tophat_width: {TOPHAT_W}          # 신규 키')
print(f'  white_s_max: {WHITE_S_MAX}           # 신규 키')
print(f'  peak_min: {PEAK_MIN}  peak_nms: {PEAK_NMS}  width_range: {list(W_RANGE)}   # 신규 키')
print(f'  max_lane_index: {MAX_LANE_INDEX}         # 신규 키')

# ============================== 11번 셀 ==============================
# 11. Pi5 시험용 주행 영상 만들기 — 보정이 맞는 60_전방 시퀀스 → mp4 (드라이브 pi_test/)
# 1·2번 셀(데이터 복원)만 실행돼 있으면 된다. 차선 보정값이 이 카메라 기준이라
# 차선 인식까지 포함한 파이프라인 전체를 Pi5에서 돌려볼 수 있는 유일한 입력이다.
import re
import cv2
from collections import defaultdict

N_SEQ    = 3            # 만들 영상 수 (프레임이 많은 시퀀스부터)
REPEAT   = 30           # 사진 1장을 몇 프레임 반복할지 — 원본이 30프레임 간격 사진이라 이어 붙이면
                        # 차가 순간이동한다. 30번 반복하면 추적이 최소 30프레임(min_tracked_frames)을 채운다
FPS      = 15           # configs input.fps
OUT_SIZE = (1280, 720)  # 카메라 main 스트림 해상도 (system-architecture.md)
SKIP     = ('2_20200916',)   # 카메라 장착이 다른 차량 — 보정이 안 맞는다 (4번 셀 EXCLUDE_CAL)

frame_no = lambda p: int(m.group(1)) if (m := re.search(r'(\d+)$', p.stem)) else -1
by = defaultdict(list)
for p in imgs.values():
    s = seq_of(p)
    if not s.startswith(SKIP):
        by[s].append(p)
picked = sorted(by.items(), key=lambda kv: -len(kv[1]))[:N_SEQ]

out_dir = DRIVE_ROOT / 'pi_test'
out_dir.mkdir(exist_ok=True)
for s, ps in picked:
    ps = sorted(ps, key=frame_no)
    out = out_dir / f'{s}.mp4'
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), FPS, OUT_SIZE)
    n = 0
    for p in ps:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, OUT_SIZE, interpolation=cv2.INTER_AREA)
        for _ in range(REPEAT):
            w.write(im)
        n += 1
    w.release()
    print(f'{s}: 사진 {n}장 → {n * REPEAT / FPS:.0f}초, {out.stat().st_size / 2**20:.0f}MB  {out}')

# ============================== 12번 셀 ==============================
# 12. 차선 가용률 개선 실험 — V: 차량 영역 제외 / T: 직전 사진 마스크 누적 / VT: 둘 다
# 필요: 1~4번, 7~10번 셀 실행 (imgs·HOLD·MIS·det·fit_lanes_p·gt_bev·center_err·load 사용)
# Pi 실행에서 차선이 거의 안 잡힌 원인 두 가지를 겨냥한다.
#   ① 점선 빈칸 + 앞차 가림 → 한쪽 윈도 적중이 4개 미만이라 피팅 자체가 안 됨
#   ② 앞차 옆면 세로 모서리를 '가장 안쪽 쌍'으로 골라 차선으로 오인
# 합격 기준 — 검증 시퀀스에서 MIN_CONF 통과분 오답(최대오차 > MAX_ERR) 0 유지 + 가용률 상승,
#            보정 틀어짐 시퀀스는 계속 거의 통과하지 않을 것 (폭 타당성 안전장치 유지)
# ⚠️ AI Hub 사진은 1초(30프레임) 간격이라 T는 Pi의 '최근 약 15프레임 누적'에 해당하는 근사다.
#    연속 영상에서의 효과는 13번 셀로 따로 본다.

PROC_W12 = [None, 320]   # None=원본, 320=Pi5 해상도 흉내
DET_CONF = 0.25          # 차량 제외용 검출 임계
OCC_PAD  = 0.05          # 박스 좌우를 폭의 이 비율만큼 넓혀 제외 (차체 모서리·그림자)
OCC_WIN  = 0.5           # 윈도 면적의 이 비율 이상이 차량이면 '가려진 윈도' — 적중률 분모에서 뺀다
MAX_GAP  = 45            # 직전 사진과의 원본 프레임 번호 차 상한 (사진 간격 30 = 1초)
N_SHOW12 = 3             # 시각화 — P 실패·VT 통과 예시 수 (VT 오답은 전부)
NAMES    = ['P', 'V', 'T', 'VT', 'VT3']   # VT3 = VT + 한쪽 최소 점 3개 (앞차가 먼 쪽을 가려 점이 모자란 경우)
GRID12   = np.round(np.arange(0.30, 0.81, 0.01), 2)


def frame_no(p):
    m_ = re.search(r'(\d+)$', norm(Path(p).name))
    return int(m_.group(1)) if m_ else None


by_seq = defaultdict(list)
for ip in imgs.values():
    by_seq[seq_of(ip)].append(ip)
PREV = {}
for lst in by_seq.values():
    lst.sort(key=lambda p: (frame_no(p) if frame_no(p) is not None else -1, p.name))
    for a_, b_ in zip(lst, lst[1:]):
        fa, fb = frame_no(a_), frame_no(b_)
        if fa is not None and fb is not None and 0 < fb - fa <= MAX_GAP:
            PREV[b_] = a_
ex = max(by_seq.values(), key=len)[:3] if by_seq else []
print('파일명 예:', [p.name for p in ex], ' → 프레임 번호', [frame_no(p) for p in ex])
print(f'직전 사진이 있는 프레임: {len(PREV)}/{len(imgs)} (번호 차 ≤ {MAX_GAP})')
if not PREV:
    print('⚠️ 직전 사진을 하나도 못 찾음 — T·VT는 P·V와 같게 나온다. 파일명 규칙을 확인할 것')


def occ_bev(bgr, Hm):
    """검출된 차량 박스를 조감도로 옮긴 마스크(0/255). 차체는 노면 위에 있어 조감도에서
    세로로 길게 늘어나는데, 박스 픽셀을 그대로 옮기므로 늘어난 부분까지 덮는다."""
    h, w = bgr.shape[:2]
    occ = np.zeros((h, w), np.uint8)
    r = det.predict(bgr, imgsz=320, conf=DET_CONF, verbose=False)[0]
    for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
        p = OCC_PAD * (x2 - x1)
        occ[max(0, int(y1)):int(y2) + 1, max(0, int(x1 - p)):int(x2 + p) + 1] = 255
    return cv2.warpPerspective(occ, Hm, (BEV_W, BEV_H), flags=cv2.INTER_NEAREST)


def fit_lanes_v(m, occ=None, min_side=4):
    """fit_lanes_p + 차량 영역 제외. occ=None이면 fit_lanes_p와 같다.
    차량에 가려진 윈도는 '못 찾은 것'이 아니라 '볼 수 없던 것'이므로 적중률 분모에서 뺀다.
    단 분모는 NWIN 아래로 내리지 않는다 — 절반 넘게 가려지면 적중률이 부풀지 않게."""
    if occ is not None:
        m = np.where(occ > 0, 0, m).astype(np.uint8)
    hist = m[BEV_H // 2:].sum(axis=0) / 255.0
    pair = pick_pair(find_peaks(hist))
    if pair is None:
        return None, None, 0.0, None

    lx, rx = pair
    ys, xs = m.nonzero()
    wh, pts, n_occ = BEV_H // NWIN, {'l': [], 'r': []}, 0
    for i in range(NWIN):
        y0, y1 = BEV_H - (i + 1) * wh, BEV_H - i * wh
        shift = []
        for key, cx in (('l', lx), ('r', rx)):
            if occ is not None:
                x0, x1_ = int(max(0, cx - WIN_M)), int(min(BEV_W, cx + WIN_M))
                if x1_ > x0 and (occ[y0:y1, x0:x1_] > 0).mean() >= OCC_WIN:
                    n_occ += 1
            sel = (ys >= y0) & (ys < y1) & (xs >= cx - WIN_M) & (xs < cx + WIN_M)
            if sel.sum() > MINPIX:
                base = int(cx - WIN_M)
                col = np.bincount(xs[sel] - base, minlength=2 * WIN_M + 2)
                col = np.convolve(col, np.ones(3), mode='same')
                pk_x = base + int(np.argmax(col))
                near = sel & (np.abs(xs - pk_x) <= 3)
                nx = float(xs[near].mean())
                pts[key].append((nx, (y0 + y1) / 2))
                shift.append(nx - cx)
        if shift:
            d = float(np.mean(shift))
            lx, rx = lx + d, rx + d

    def solve(pl, pr):
        Y = np.array([p[1] for p in pl + pr])
        X = np.array([p[0] for p in pl + pr])
        A = np.c_[Y ** 2, Y, [1] * len(pl) + [0] * len(pr), [0] * len(pl) + [1] * len(pr)]
        coef = np.linalg.lstsq(A, X, rcond=None)[0]
        return coef, X - A @ coef

    pl, pr = pts['l'], pts['r']
    if len(pl) < min_side or len(pr) < min_side:
        return None, None, 0.0, pair
    coef, res = solve(pl, pr)
    keep_ = np.abs(res) <= max(3.0, 2.5 * float(np.median(np.abs(res))))
    pl = [p for p, k in zip(pl, keep_[:len(pl)]) if k]
    pr = [p for p, k in zip(pr, keep_[len(pts['l']):]) if k]
    if len(pl) < min_side or len(pr) < min_side:
        return None, None, 0.0, pair
    coef, res = solve(pl, pr)
    a, b, cl, cr = coef
    fl, fr = np.array([a, b, cl]), np.array([a, b, cr])

    rms   = float(np.sqrt(np.mean(res ** 2)))
    fitq  = 1.0 - min(1.0, rms / (0.05 * W_EXP))
    plaus = float(np.clip(1 - abs((cr - cl) / W_EXP - 1) / 0.40, 0, 1))
    hit   = min(1.0, (len(pl) + len(pr)) / max(2 * NWIN - n_occ, NWIN))
    return fl, fr, round(float(hit * fitq * plaus), 3), pair


CACHE12 = {}
def prep(ip, pw):
    """(마스크, 차량 조감도 마스크, Hm, 처리 폭) — 프레임·해상도별 한 번만 계산."""
    key = (ip, pw)
    if key not in CACHE12:
        bgr = load(ip, pw)
        if bgr is None:
            CACHE12[key] = None
        else:
            h, w = bgr.shape[:2]
            Hm = make_H(w, h, SRC)
            _, m = bev_mask(bgr, Hm)
            CACHE12[key] = (m, occ_bev(bgr, Hm), Hm, w)
    return CACHE12[key]


def variants(ip, pw):
    """방법별 (피팅 결과, 사용한 마스크), Hm, 처리 폭."""
    cur = prep(ip, pw)
    if cur is None:
        return None, None, None
    m, occ, Hm, w = cur
    mv = np.where(occ > 0, 0, m).astype(np.uint8)
    out = {'P': (fit_lanes_p(m), m), 'V': (fit_lanes_v(m, occ), mv)}
    prv = prep(PREV[ip], pw) if ip in PREV else None
    if prv is not None and prv[0].shape == m.shape:
        m2, occ2 = prv[0], prv[1]
        mt  = cv2.bitwise_or(m, m2)
        mvt = cv2.bitwise_or(mv, np.where(occ2 > 0, 0, m2).astype(np.uint8))
        out['T']  = (fit_lanes_p(mt), mt)
        occ_t = cv2.bitwise_and(occ, occ2)                                 # 두 장 다 가려진 곳만 '가림'
        out['VT']  = (fit_lanes_v(mvt, occ_t), mvt)
        out['VT3'] = (fit_lanes_v(mvt, occ_t, min_side=3), mvt)
    else:
        out['T'], out['VT'] = out['P'], out['V']
        out['VT3'] = (fit_lanes_v(m, occ, min_side=3), mv)
    return out, Hm, w


def evaluate12(entries, pw):
    rows = {k: [] for k in NAMES}
    for xp, g, meta, full in entries:
        if g != G_DEV:
            continue
        ip = imgs.get(norm(xp.name))
        if ip is None:
            continue
        res, Hm, w = variants(ip, pw)
        if res is None:
            continue
        gt = gt_bev(meta, Hm, s=w / meta[0])
        if gt is None:
            continue
        gtw = float(gt[1][-1, 0] - gt[0][-1, 0]) / W_EXP
        for k in NAMES:
            (fl, fr, conf, pk), _ = res[k]
            rows[k].append(dict(ip=ip, meta=meta, seq=seq_of(ip), conf=conf, pk=pk, full=full, gtw=gtw,
                                prev=ip in PREV, err=center_err(fl, fr, gt) if fl is not None else None))
    return rows


def passed(r):
    return r['err'] is not None and r['conf'] >= MIN_CONF


hold_none = None
for pw in PROC_W12:
    tag = '원본' if pw is None else f'{pw}px'
    H12, M12 = evaluate12(HOLD, pw), evaluate12(MIS, pw)
    if pw is None:
        hold_none = H12
    n_prev = sum(r['prev'] for r in H12['P'])
    print('\n' + '═' * 96)
    print(f'입력 {tag} — MIN_CONF {MIN_CONF}, 오답 = 최대오차 > {MAX_ERR}   '
          f'(검증 {len(H12["P"])}장 중 직전 사진 있음 {n_prev}장)')
    print('═' * 96)
    print(f'{"방법":<5}{"검증 통과":>11}{"가용률":>7}{"오답":>5}{"오차중앙":>9}{"오답0 임계":>10}'
          f'{"오답 conf최고":>13}{"직전有 가용률":>13} │{"틀어짐 통과":>10}{"오답":>5}')
    for k in NAMES:
        hr = H12[k]
        ok = [r for r in hr if r['err'] is not None]
        hs = [r for r in ok if r['conf'] >= MIN_CONF]
        bad_c = [r['conf'] for r in ok if r['err'][1] > MAX_ERR]
        cands = [th for th in GRID12
                 if (s := [r for r in ok if r['conf'] >= th]) and max(r['err'][1] for r in s) <= MAX_ERR]
        rec = '-' if not bad_c else (f'{min(cands):.2f}' if cands else '없음')
        pr_ = [r for r in hr if r['prev']]
        ms = [r for r in M12[k] if passed(r)]
        print(f'{k:<5}{len(hs):>6}/{len(hr):<4}{len(hs) / max(len(hr), 1) * 100:>6.0f}%'
              f'{sum(r["err"][1] > MAX_ERR for r in hs):>5}'
              f'{(np.median([r["err"][0] for r in hs]) if hs else float("nan")):>9.3f}{rec:>10}'
              f'{(max(bad_c) if bad_c else float("nan")):>13.3f}'
              f'{sum(map(passed, pr_)) / max(len(pr_), 1) * 100:>12.0f}%'
              f' │{len(ms):>6}/{len(M12[k]):<4}{sum(r["err"][1] > MAX_ERR for r in ms):>5}')


# ── 시각화 (원본 해상도) — VT 오답 전부 + P는 실패했는데 VT는 통과한 예시 ──
def show12(r, title):
    ip, meta = r['ip'], r['meta']
    res, Hm, w = variants(ip, None)
    bgr = load(ip, None)
    gt = gt_bev(meta, Hm, s=w / meta[0])
    occ = prep(ip, None)[1]
    panels = [(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), '원본'),
              (occ, '차량 영역 (조감도)')]
    for k in ('P', 'VT'):
        (fl, fr, conf, pk), mk = res[k]
        ov = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
        if fl is not None:
            yy = np.linspace(0, BEV_H - 1, 50)
            for f, c in zip((fl, fr), [(0, 0, 255), (0, 255, 0)]):
                cv2.polylines(ov, [np.int32(np.c_[np.polyval(f, yy), yy])], False, c, 2)
        for side in gt:
            cv2.polylines(ov, [np.int32(side)], False, (255, 255, 0), 1)
        e = center_err(fl, fr, gt) if fl is not None else None
        panels.append((cv2.cvtColor(ov, cv2.COLOR_BGR2RGB),
                       f'{k} conf={conf:.3f} ' + (f'최대오차={e[1]:.3f}' if e else '피팅 실패')))
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, (im_, t_) in zip(ax, panels):
        a.imshow(im_, cmap='gray'); a.set_title(t_, fontsize=9); a.axis('off')
    plt.suptitle(f'{title} — {ip.name}', fontsize=9)
    plt.tight_layout(); plt.show()


P_by = {r['ip']: r for r in hold_none['P']}
vt_bad = [(k, r) for k in ('VT', 'VT3') for r in hold_none[k] if passed(r) and r['err'][1] > MAX_ERR]
gained = [r for r in hold_none['VT'] if passed(r) and r['err'][1] <= MAX_ERR and not passed(P_by[r['ip']])]
print(f'\nVT·VT3 오답 {len(vt_bad)}장 (전부 표시) / P 실패→VT 통과 {len(gained)}장 (중 {N_SHOW12}장 표시)')
for k, r in sorted(vt_bad, key=lambda t: -t[1]['conf']):
    show12(r, f'{k} 오답 (conf {r["conf"]:.3f})')
for r in gained[:N_SHOW12]:
    show12(r, 'P 실패 → VT 통과')

# ============================== 13번 셀 ==============================
# 13. 연속 영상 확인 — 12번 셀의 T(마스크 누적)를 실제 프레임 간격으로 본다
# 필요: 12번 셀 실행 (occ_bev·fit_lanes_v 사용). 정답 라벨이 없으므로 '맞았는지'는 직접 못 잰다.
# 대신 차선은 프레임 사이에 거의 움직이지 않는다는 점을 이용한다 — 연속된 유효 프레임 사이에서
# 차로 중심이 크게 튀면 둘 중 하나는 틀린 차선일 가능성이 높다 (자차 차로 변경 구간은 예외).
# 영상 — 강섬희 시험 영상(sim_*.mp4)이나 실제 블랙박스 영상을 드라이브에 올려 경로를 적는다.
# ⚠️ pi_test/의 AI Hub 영상은 사진 1장을 30번 반복한 것이라 누적 효과를 볼 수 없다.
from collections import deque

VIDEOS  = sorted((DRIVE_ROOT / 'sim').glob('*.mp4'))   # 또는 [DRIVE_ROOT / '파일.mp4']
PROC_W  = 640          # 처리 폭 (종횡비 유지)
FPS_PI  = 15           # Pi 처리 속도 흉내 — 원본 fps에서 이 간격으로 솎는다
KS      = [5, 10]      # 누적 프레임 수 (15fps에서 0.33초 / 0.67초)
MAX_SEC = 60           # 영상당 최대 처리 길이
JUMP    = 0.10         # 연속 유효 프레임 사이 차로 중심 이동이 기대 폭(W_EXP)의 이 비율 초과 → '튐'
N_SHOW13 = 3           # 영상당 시각화 프레임 수

METHODS13 = [('P', 1, False, 4), ('V', 1, True, 4)] + \
            [(f'{p}{k}', k, v, s) for k in KS for p, v, s in (('T', False, 4), ('VT', True, 4), ('VT3_', True, 3))]


def run_video(path):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / FPS_PI))
    total = min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1e9, MAX_SEC * src_fps) / step
    every = max(1, int(total // N_SHOW13))              # 시각화용 프레임만 남긴다 (메모리)
    hist_m, hist_o = deque(maxlen=max(KS)), deque(maxlen=max(KS))
    out = {name: [] for name, *_ in METHODS13}      # 프레임별 (유효, 차로 중심 x 또는 None, conf)
    samples, fi = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok or fi / src_fps > MAX_SEC:
            break
        fi += 1
        if (fi - 1) % step:
            continue
        h, w = bgr.shape[:2]
        if PROC_W and w != PROC_W:
            bgr = cv2.resize(bgr, (PROC_W, round(h * PROC_W / w)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        Hm = make_H(w, h, SRC)
        _, m = bev_mask(bgr, Hm)
        occ = occ_bev(bgr, Hm)
        hist_m.appendleft(m); hist_o.appendleft(occ)
        fits = {}
        for name, k, use_v, ms in METHODS13:
            ms_ = list(hist_m)[:k]; os_ = list(hist_o)[:k]
            if use_v:
                mk = np.zeros_like(m)
                for mm, oo in zip(ms_, os_):
                    mk = cv2.bitwise_or(mk, np.where(oo > 0, 0, mm).astype(np.uint8))
                oc = os_[0].copy()
                for oo in os_[1:]:
                    oc = cv2.bitwise_and(oc, oo)
                fl, fr, conf, pk = fit_lanes_v(mk, oc, min_side=ms)
            else:
                mk = ms_[0].copy()
                for mm in ms_[1:]:
                    mk = cv2.bitwise_or(mk, mm)
                fl, fr, conf, pk = fit_lanes_p(mk)
            valid = fl is not None and conf >= MIN_CONF
            cx = float(np.polyval(fl, BEV_H - 1) + np.polyval(fr, BEV_H - 1)) / 2 if valid else None
            out[name].append((valid, cx, conf))
            fits[name] = (fl, fr, conf, mk)
        if (len(out['P']) - 1) % every == 0 and len(samples) < N_SHOW13:
            samples.append((len(out['P']) - 1, bgr, fits))
    cap.release()
    return out, samples, src_fps, step


def report(out):
    print(f'{"방법":<8}{"가용률":>8}{"튐":>10}{"중심 이동 중앙":>14}{"conf p50":>10}')
    for name, *_ in METHODS13:
        rows = out[name]
        n = len(rows)
        val = [r for r in rows if r[0]]
        moves = [abs(b[1] - a[1]) / W_EXP for a, b in zip(rows, rows[1:]) if a[0] and b[0]]
        jumps = sum(mv > JUMP for mv in moves)
        print(f'{name:<8}{len(val) / max(n, 1) * 100:>7.0f}%{jumps:>5}/{len(moves):<4}'
              f'{(np.median(moves) if moves else float("nan")):>14.3f}'
              f'{np.median([r[2] for r in rows]) if rows else float("nan"):>10.3f}')


def show13(samples, out, best):
    for i, bgr, fits in samples:
        h, w = bgr.shape[:2]
        vis = bgr.copy()
        cv2.polylines(vis, [np.int32([(x * w, y * h) for x, y in SRC])], True, (0, 255, 255), 2)
        panels = [(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f'프레임 {i}')]
        for name in ('P', best):
            fl, fr, conf, mk = fits[name]
            ov = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
            if fl is not None:
                yy = np.linspace(0, BEV_H - 1, 50)
                for f, c in zip((fl, fr), [(0, 0, 255), (0, 255, 0)]):
                    cv2.polylines(ov, [np.int32(np.c_[np.polyval(f, yy), yy])], False, c, 2)
            panels.append((cv2.cvtColor(ov, cv2.COLOR_BGR2RGB),
                           f'{name} conf={conf:.3f} ' + ('유효' if out[name][i][0] else '보류')))
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for a, (im_, t_) in zip(ax, panels):
            a.imshow(im_); a.set_title(t_, fontsize=9); a.axis('off')
        plt.tight_layout(); plt.show()


if not VIDEOS:
    print('⚠️ 영상 없음 — VIDEOS 경로를 확인할 것')
for vp in VIDEOS:
    out, samples, fps, step = run_video(vp)
    print('\n' + '═' * 56 + f'\n{vp.name}  ({fps:.0f}fps → {step}프레임마다, {len(out["P"])}프레임 처리)\n' + '═' * 56)
    report(out)
    best = max((n for n, *_ in METHODS13 if n != 'P'), key=lambda n: sum(r[0] for r in out[n]))
    show13(samples, out, best)

# ============================== 14번 셀 ==============================
# 14. 판정 임계(MIN_CONF) 낮추기 실험 — 오답 0을 지키면서 가용률이 얼마나 오르나
# 필요: 12번 셀 실행 (evaluate12·CACHE12 사용, 캐시가 있어 다시 돌려도 빠름)
# 12번 결과에서 맞게 잡은 차선도 conf가 0.5에 못 미쳐 버려지는 경우가 많았다
# (예: 오차 0.012인데 conf 0.460). 점선은 윈도의 절반가량만 적중해서 적중률이 conf를 깎기 때문이다.
# 표 한 칸 = "가용률 / 검증 오답 / 틀어짐 통과(그중 오답)"
#   검증 오답이 0이고, 틀어짐 통과도 거의 0인 임계 중 가장 낮은 값이 후보다.
#   단, 표본이 247장뿐이라 '오답 conf최고' 바로 위 값은 우연일 수 있다 → 0.03 이상 여유를 둔다.
TH14 = [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]

for pw in PROC_W12:
    tag = '원본' if pw is None else f'{pw}px'
    H14, M14 = evaluate12(HOLD, pw), evaluate12(MIS, pw)
    print('\n' + '═' * 100)
    print(f'입력 {tag} — 칸 = 가용률/검증 오답/틀어짐 통과(오답)   (검증 {len(H14["P"])}장, 틀어짐 {len(M14["P"])}장)')
    print('═' * 100)
    print(f'{"방법":<5}' + ''.join(f'{th:>12.2f}' for th in TH14))
    for k in NAMES:
        cells = []
        for th in TH14:
            hs = [r for r in H14[k] if r['err'] is not None and r['conf'] >= th]
            ms = [r for r in M14[k] if r['err'] is not None and r['conf'] >= th]
            bad = sum(r['err'][1] > MAX_ERR for r in hs)
            mbad = sum(r['err'][1] > MAX_ERR for r in ms)
            cells.append(f'{len(hs) / max(len(H14[k]), 1) * 100:.0f}%/{bad}/{len(ms)}({mbad})')
        print(f'{k:<5}' + ''.join(f'{c:>12}' for c in cells))

# conf를 깎는 주범 확인 — 원본 해상도 P에서 '맞게 잡았는데 conf 0.5 미만'인 프레임의 세 요소
# fit_lanes_p는 요소를 따로 돌려주지 않으므로 conf와 같은 식으로 다시 계산한다.
def conf_parts(m):
    hist = m[BEV_H // 2:].sum(axis=0) / 255.0
    pair = pick_pair(find_peaks(hist))
    if pair is None:
        return None
    fl, fr, conf, _ = fit_lanes_v(m, None)
    if fl is None:
        return None
    w_ = (fr[2] - fl[2]) / W_EXP
    plaus = float(np.clip(1 - abs(w_ - 1) / 0.40, 0, 1))
    return conf, plaus

near = [r for r in hold_none['P'] if r['err'] is not None and r['err'][1] <= MAX_ERR and 0.3 <= r['conf'] < MIN_CONF]
print(f'\n원본 P — 정답 근처(최대오차 ≤ {MAX_ERR})인데 conf {MIN_CONF} 미만으로 버려진 프레임: {len(near)}장')
if near:
    pl_ = [conf_parts(prep(r['ip'], None)[0]) for r in near]
    pl_ = [p for p in pl_ if p]
    print(f'  그중 폭 타당성(plaus) 중앙값 {np.median([p[1] for p in pl_]):.2f} '
          f'→ 나머지(적중률×피팅 품질) 중앙값 {np.median([p[0] / max(p[1], 1e-6) for p in pl_]):.2f}')

# ============================== 15번 셀 ==============================
# 15. 딥러닝 차선 1단계 — UFLDv2 공개 가중치를 학습 없이 그대로 시험 (원본 화면 기준 평가)
# 필요: 1~4번, 7~10번, 12번 셀 실행 (imgs·HOLD·MIS·x_at·load·prep·fit_lanes_p·fit_lanes_v 사용)
# 런타임: GPU 권장 (CPU도 되지만 느림). 결정 근거·합격 기준 — docs/lane-detection-dl-plan.md
#
# 무엇을 보나
#   ① 같은 검증 247장에서 지금 방식(P·V)보다 가용률이 크게 높으면서 통과분 오답 0을 지키는가
#   ② 보정이 다른 카메라 196장('틀어짐')에서도 맞게 잡는가 — 지금 방식은 여기서 0장 통과(보류)였다.
#      UFLD는 사다리꼴 보정을 안 쓰므로 여기서 맞게 잡으면 '카메라마다 보정 불필요'의 직접 증거다
# 평가는 원본 화면 기준이다 — 정답 자차 차로 좌/우 경계와 예측 경계를 같은 행에서 비교해
#   차로 중심 오차 / 정답 차로 폭 (MAX_ERR 초과 = 오답). P·V도 조감도 피팅을 원본 화면으로 되돌려 같은 식으로 잰다.
#   (12·14번 셀은 조감도에서 쟀으므로 P·V 수치가 조금 다를 수 있다)
# UFLD 신뢰도 = 자차 좌·우 차선 각각 '존재'로 판정된 행 비율 중 작은 값 (원 저장소 데모는 0.5 초과면 차선 채택)
import sys, subprocess, time
import torch, torch.nn as nn, torchvision

UFLD_DIR = DRIVE_ROOT / 'ufld'          # 가중치 보관 (드라이브 — 세션이 끊겨도 남음)
UFLD_DIR.mkdir(parents=True, exist_ok=True)
# 원 저장소 configs/*_res18.py 값 그대로. gid = README의 Google Drive 파일 id
UFLD_CFG = {
    'culane':   dict(gid='1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W', in_w=1600, in_h=320, crop=0.6,
                     grid_row=200, n_row=72, grid_col=100, n_col=81, aspect=1640 / 590, fc_norm=True,
                     row_anchor=np.linspace(0.42, 1, 72)),
    'tusimple': dict(gid='1Clnj9-dLz81S3wXiYtlkc4HVusCb978t', in_w=800, in_h=320, crop=0.8,
                     grid_row=100, n_row=56, grid_col=100, n_col=41, aspect=1280 / 720, fc_norm=False,
                     row_anchor=np.linspace(160, 710, 56) / 720),
}
PREP15 = ['full', 'bottom']   # full = 화면 전체를 넣음 / bottom = 학습 데이터 화면 비율로 아래쪽만 잘라 넣음 (둘 다 보정 불필요)
TH15   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TRUST_CKPT = False            # True면 가중치 파일을 pickle 그대로 연다 (weights_only 실패 시에만, 출처 확인 후)
N_SHOW15 = 3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class UFLDv2(nn.Module):
    """원 저장소 model/model_culane.py parsingNet(use_aux=False)과 같은 구조 — 저장소 import 없이 재구성
    (원 저장소 utils.common이 NVIDIA DALI를 import해 Colab에서 바로 안 열린다). 키 이름도 같게 맞췄다."""
    def __init__(self, c):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        self.model = nn.Module()
        for k in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'):
            setattr(self.model, k, getattr(r, k))
        L = 4
        self.dims = (c['grid_row'] * c['n_row'] * L, c['grid_col'] * c['n_col'] * L, 2 * c['n_row'] * L, 2 * c['n_col'] * L)
        self.c, self.input_dim = c, c['in_h'] // 32 * c['in_w'] // 32 * 8
        self.pool = nn.Conv2d(512, 8, 1)
        self.cls = nn.Sequential(nn.LayerNorm(self.input_dim) if c['fc_norm'] else nn.Identity(),   # TuSimple판은 LayerNorm 없음
                                 nn.Linear(self.input_dim, 2048),
                                 nn.ReLU(), nn.Linear(2048, sum(self.dims)))

    def forward(self, x):
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        out = self.cls(self.pool(x).view(-1, self.input_dim))
        c, (d1, d2, d3, _) = self.c, self.dims
        return {'loc_row':   out[:, :d1].view(-1, c['grid_row'], c['n_row'], 4),
                'exist_row': out[:, d1 + d2:d1 + d2 + d3].view(-1, 2, c['n_row'], 4)}


def load_ufld(name):
    c = UFLD_CFG[name]
    path = UFLD_DIR / f'{name}_res18.pth'
    if not path.exists():
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'gdown'], check=True)
            import gdown
        gdown.download(id=c['gid'], output=str(path), quiet=False)
    if not path.exists() or path.stat().st_size < 1e6:
        raise FileNotFoundError(f'{path} 다운로드 실패 — 원 저장소 README의 Google Drive 링크에서 받아 이 경로에 올릴 것')
    try:
        ck = torch.load(path, map_location='cpu', weights_only=not TRUST_CKPT)
    except Exception as e:
        raise RuntimeError(f'{path.name} 안전 모드(weights_only)로 못 엶 — 출처 확인 후 TRUST_CKPT=True ({e})')
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ck.get('model', ck).items()}
    net = UFLDv2(c)
    miss, unexp = net.load_state_dict({k: v for k, v in sd.items() if not k.startswith('seg_head')}, strict=False)
    unexp = [k for k in unexp if not k.startswith(('loc_col', 'exist_col'))]
    if miss or unexp:
        raise RuntimeError(f'가중치 키 불일치 — 없음 {miss[:5]} / 남음 {unexp[:5]}')
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'{name}: 파라미터 {n_par:.0f}M (fp32 {n_par * 4:.0f}MB) — 마지막 FC가 대부분. NPU 메모리 판단에 쓸 것')
    return net.eval().to(DEV)


def canvas_of(bgr, c, mode):
    """모델에 넣을 화면과 원본 y 오프셋. bottom은 학습 데이터 화면 비율(가로:세로)에 맞춰 아래쪽만 자른다."""
    h, w = bgr.shape[:2]
    if mode == 'bottom':
        hc = min(h, round(w / c['aspect']))
        return bgr[h - hc:], h - hc
    return bgr, 0


@torch.no_grad()
def ufld_lanes(net, bgr, mode):
    """차선 4개(왼쪽부터 0~3, 자차 차로 = 1·2)의 원본 좌표 점과 존재 비율."""
    c = net.c
    cv_, y0 = canvas_of(bgr, c, mode)
    ch, cw = cv_.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(cv_, (c['in_w'], int(c['in_h'] / c['crop'])), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)[-c['in_h']:]                    # 원 저장소 데모와 같은 아래쪽 크롭
    x = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255 - torch.tensor([.485, .456, .406])[:, None, None]) \
        / torch.tensor([.229, .224, .225])[:, None, None]
    out = net(x[None].to(DEV))
    loc, ex = out['loc_row'][0].cpu(), out['exist_row'][0].cpu().softmax(0)[1]   # (grid,n_row,4), (n_row,4)
    idx, G = loc.argmax(0), c['grid_row']
    lanes = []
    for li in range(4):
        pts = []
        for k in range(c['n_row']):
            if ex[k, li] > 0.5:                                           # 데모의 exist argmax와 같음
                i = int(idx[k, li]); ind = torch.arange(max(0, i - 1), min(G - 1, i + 1) + 1)
                xg = float((loc[ind, k, li].softmax(0) * ind.float()).sum()) + 0.5
                pts.append((xg / (G - 1) * cw, c['row_anchor'][k] * ch + y0))
        lanes.append((pts, float((ex[:, li] > 0.5).float().mean())))
    return lanes


def gt_img(meta, s=1.0):
    """정답 자차 차로 좌/우 경계 — 원본 좌표 (y, x_left, x_right) 행 목록. gt_bev와 같은 9행."""
    W_, H_, lanes_, pair, yb, yt = meta
    lp, rp = pair[0][3], pair[1][3]
    rows = [(y * s, x_at(lp, y), x_at(rp, y)) for y in np.linspace(yt * H_, yb * H_, 9)]
    rows = [(y, a * s, b * s) for y, a, b in rows if a is not None and b is not None]
    return np.array(rows) if len(rows) >= 5 else None


def interp_x(pts, y):
    if len(pts) < 2:
        return None
    p = sorted(pts, key=lambda q: q[1])
    ys, xs = np.array([q[1] for q in p]), np.array([q[0] for q in p])
    return float(np.interp(y, ys, xs)) if ys[0] <= y <= ys[-1] else None   # 외삽 금지


def err_img(left_pts, right_pts, gt):
    e = []
    for y, gl, gr in gt:
        a, b = interp_x(left_pts, y), interp_x(right_pts, y)
        if a is None or b is None or gr - gl <= 1:
            continue
        e.append(abs((a + b) / 2 - (gl + gr) / 2) / (gr - gl))
    return (float(np.median(e)), float(np.max(e))) if len(e) >= 5 else None


def bev_to_img(f, Hinv):
    yy = np.linspace(0, BEV_H - 1, 60)
    p = np.float32(np.c_[np.polyval(f, yy), yy]).reshape(-1, 1, 2)
    return [tuple(q) for q in cv2.perspectiveTransform(p, Hinv).reshape(-1, 2)]


def eval15(entries, nets):
    rows = defaultdict(list)
    for xp, g, meta, full in entries:
        if g != G_DEV:
            continue
        ip = imgs.get(norm(xp.name))
        bgr = load(ip, None) if ip is not None else None
        if bgr is None:
            continue
        gt = gt_img(meta, s=bgr.shape[1] / meta[0])
        if gt is None:
            continue
        base = dict(ip=ip, meta=meta, seq=seq_of(ip))
        m, occ, Hm, _ = prep(ip, None)                                   # 12번 셀 캐시
        Hinv = np.linalg.inv(Hm)
        for k, (fl, fr, conf, _), th in (('P(0.50)', fit_lanes_p(m), 0.50), ('V(0.47 현재)', fit_lanes_v(m, occ), 0.47)):
            e = err_img(bev_to_img(fl, Hinv), bev_to_img(fr, Hinv), gt) if fl is not None else None
            rows[k].append(dict(base, conf=conf, th=th, err=e))
        for name, net in nets.items():
            for mode in PREP15:
                lanes = ufld_lanes(net, bgr, mode)
                (lp, lf), (rp, rf) = lanes[1], lanes[2]
                rows[f'UFLD-{name}-{mode}'].append(dict(base, conf=min(lf, rf), th=None, lanes=lanes, mode=mode,
                                                       net=name, err=err_img(lp, rp, gt)))
    return rows


nets15 = {n: load_ufld(n) for n in UFLD_CFG}
for n, net in nets15.items():                                            # GPU 처리 시간 (참고용 — Pi NPU와 무관)
    dummy = torch.zeros(1, 3, net.c['in_h'], net.c['in_w'], device=DEV)
    for _ in range(3): net(dummy)
    if DEV == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): net(dummy)
    if DEV == 'cuda': torch.cuda.synchronize()
    print(f'  {n} 추론 {(time.time() - t0) / 20 * 1000:.1f}ms/장 ({DEV})')

R_HOLD, R_MIS = eval15(HOLD, nets15), eval15(MIS, nets15)


def cell(rs, th):
    s = [r for r in rs if r['err'] is not None and r['conf'] >= th]
    return len(s), sum(r['err'][1] > MAX_ERR for r in s), (np.median([r['err'][0] for r in s]) if s else float('nan'))


nh, nm = len(R_HOLD['P(0.50)']), len(R_MIS['P(0.50)'])
print('\n' + '═' * 140)
print(f'원본 화면 기준 — 칸 = 가용률/검증 오답/틀어짐 가용률(오답)   검증 {nh}장 · 틀어짐 {nm}장 · 오답 = 최대오차 > {MAX_ERR}')
print('═' * 140)
print(f'{"방법":<22}' + ''.join(f'{("임계 " + str(t)):>15}' for t in TH15) + f'{"오차중앙":>9}')
for k in R_HOLD:
    hs, ms = R_HOLD[k], R_MIS[k]
    ths = [hs[0]['th']] if hs and hs[0]['th'] is not None else TH15
    cells = []
    for th in ths:
        p, b, _ = cell(hs, th); mp, mb, _ = cell(ms, th)
        cells.append(f'{p / max(nh, 1) * 100:.0f}%/{b}/{mp / max(nm, 1) * 100:.0f}%({mb})')
    med = cell(hs, ths[0] if len(ths) == 1 else 0.5)[2]           # 오차중앙 — P·V는 자기 임계, UFLD는 0.5 기준
    line = ''.join(f'{c:>15}' for c in cells)
    if len(ths) == 1:
        line = f'{cells[0]:>15}' + f'{"(자기 임계 " + str(ths[0]) + "에서만)":<{15 * (len(TH15) - 1)}}'
    print(f'{k:<22}{line}{med:>9.3f}')

# 합격 판정용 요약 — UFLD 각 조합에서 '검증 오답 0'인 가장 낮은 임계와 그때의 가용률
print('\n검증 오답 0을 지키는 가장 낮은 임계 (UFLD)')
for k in [k for k in R_HOLD if k.startswith('UFLD')]:
    ok = [th for th in np.round(np.arange(0.30, 0.96, 0.01), 2) if cell(R_HOLD[k], th)[1] == 0 and cell(R_HOLD[k], th)[0] > 0]
    if not ok:
        print(f'  {k:<22} 없음 — 통과분이 있으면서 오답 0인 임계가 0.95 이하에 없음'); continue
    th = min(ok); p, _, med = cell(R_HOLD[k], th); mp, mb, _ = cell(R_MIS[k], th)
    print(f'  {k:<22} 임계 {th:.2f} → 검증 가용률 {p / nh * 100:.0f}% (오차중앙 {med:.3f}) / 틀어짐 {mp / max(nm, 1) * 100:.0f}% 오답 {mb}')


# ── 시각화 — UFLD 오답(임계 0.5 통과분, 최대 6장) + V는 보류인데 UFLD는 맞게 잡은 예 ──
COL15 = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 128, 0)]

def show15(r, title):
    bgr = load(r['ip'], None)
    gt = gt_img(r['meta'], s=bgr.shape[1] / r['meta'][0])
    vis = bgr.copy()
    for y, a, b in gt:
        cv2.circle(vis, (int(a), int(y)), 7, (255, 255, 0), 2); cv2.circle(vis, (int(b), int(y)), 7, (255, 255, 0), 2)
    for li, (pts, frac) in enumerate(r['lanes']):
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 4, COL15[li], -1)
    if r['mode'] == 'bottom':
        y0 = canvas_of(bgr, UFLD_CFG[r['net']], 'bottom')[1]
        cv2.line(vis, (0, y0), (vis.shape[1], y0), (0, 255, 255), 2)
    plt.figure(figsize=(12, 6.5)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
    e = r['err']
    plt.title(f'{title} — {r["ip"].name}\nconf={r["conf"]:.2f} ' + (f'최대오차={e[1]:.3f}' if e else '평가 불가')
              + '   (하늘색 원 = 정답, 빨강·초록 = 자차 차로 예측, 분홍·주황 = 옆 차선)', fontsize=9)
    plt.show()

ukeys = [k for k in R_HOLD if k.startswith('UFLD')]
bad = [(k, r) for k in ukeys for r in R_HOLD[k] if r['err'] is not None and r['conf'] >= 0.5 and r['err'][1] > MAX_ERR]
print(f'\nUFLD 임계 0.5 통과분 오답 {len(bad)}건 (최대 6건 표시)')
for k, r in sorted(bad, key=lambda t: -t[1]['conf'])[:6]:
    show15(r, f'{k} 오답')
best = max(ukeys, key=lambda k: cell(R_HOLD[k], 0.5)[0])
v_by = {r['ip']: r for r in R_HOLD['V(0.47 현재)']}
gain = [r for r in R_HOLD[best] if r['err'] is not None and r['conf'] >= 0.5 and r['err'][1] <= MAX_ERR
        and not (v_by[r['ip']]['err'] is not None and v_by[r['ip']]['conf'] >= 0.47)]
print(f'{best}: V는 보류인데 UFLD는 맞게 잡은 프레임 {len(gain)}장 (중 {N_SHOW15}장 표시)')
for r in gain[:N_SHOW15]:
    show15(r, f'{best} — V 보류 → UFLD 통과')


# ============================== 16번 셀 ==============================
# 16. 딥러닝 차선 1.5단계 — 학습 없이 '안전 검사'로 틀린 차선을 걸러낼 수 있나 (UFLD CULane)
# 필요: 15번 셀 실행 (nets15·canvas_of·gt_img·interp_x·err_img·HOLD·MIS 사용). GPU 런타임 권장
#
# 15번 결과: CULane은 차선을 잘 찾지만(가용률 78~89%) 틀린 프레임도 conf=1.00이었다 —
#   지금 신뢰도(차선 '존재' 행 비율)로는 틀린 것만 골라 버릴 수 없다. 학습 없이 계산되는 검사 4개를 붙여 본다.
#   pk  위치 확신도 — 행마다 차선 위치 확률의 최댓값. 자차 좌·우 차선 각각의 중앙값 중 작은 값 (높을수록 좋음)
#   res 한 줄 검사 — 차선 점들을 2차 곡선에 맞춘 잔차(RMS) / 차로 폭. 점이 두 선을 오가면 커진다 (작을수록 좋음)
#   geo 차로 모양 검사 — 좌 < 우, 차로 폭이 아래로 갈수록 넓어짐, 맨 아래 폭이 화면 폭의 WMIN~WMAX
#   agr 두 입력 일치 — 같은 모델에 full / bottom으로 넣은 결과의 차로 중심 차이 / 차로 폭 (작을수록 좋음)
# ⚠️ 과적합 방지: 임계 조합은 검증 247장에서만 고른다(오답 0 중 가용률 최대).
#    합격 판정은 튜닝에 한 번도 안 쓴 '틀어짐' 196장(다른 카메라)에서 오답 0인지로 한다.
# 합격 기준(docs/lane-detection-dl-plan.md): 검증 오답 0 + 가용률 ≥ 40% + 틀어짐 오답 0
import itertools
import glob as _glob
import matplotlib.font_manager as fm

try:                                                             # 그림 제목 한글 (□ 깨짐 방지)
    if not any('Nanum' in f.name for f in fm.fontManager.ttflist):
        subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-nanum'], capture_output=True)
        for f in _glob.glob('/usr/share/fonts/truetype/nanum/*.ttf'):
            fm.fontManager.addfont(f)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print('한글 글꼴 설정 실패 (결과에는 영향 없음):', e)

NET16  = nets15['culane']     # TuSimple은 15번에서 탈락
MAIN16 = 'full'               # 차선 좌표는 full 입력 결과를 쓰고, bottom은 일치 검사에만 쓴다
WMIN, WMAX = 0.15, 1.0        # 맨 아래 공통 행의 차로 폭 / 화면 폭 허용 범위 (느슨하게)
MIN_PTS = 6                   # 차선 하나당 최소 점 수 (2차 곡선 맞춤용)
N_SHOW16 = 3


@torch.no_grad()
def ufld_detail(net, bgr, mode):
    """15번 ufld_lanes와 같은 디코딩 + 점마다 위치 확률 최댓값. 자차 차선(1·2)만 반환."""
    c = net.c
    cv_, y0 = canvas_of(bgr, c, mode)
    ch, cw = cv_.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(cv_, (c['in_w'], int(c['in_h'] / c['crop'])), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)[-c['in_h']:]
    x = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255 - torch.tensor([.485, .456, .406])[:, None, None]) \
        / torch.tensor([.229, .224, .225])[:, None, None]
    out = net(x[None].to(DEV))
    loc, ex = out['loc_row'][0].cpu(), out['exist_row'][0].cpu().softmax(0)[1]
    prob, idx, G = loc.softmax(0), loc.argmax(0), c['grid_row']
    res = []
    for li in (1, 2):
        pts, pk = [], []
        for k in range(c['n_row']):
            if ex[k, li] > 0.5:
                i = int(idx[k, li]); ind = torch.arange(max(0, i - 1), min(G - 1, i + 1) + 1)
                xg = float((loc[ind, k, li].softmax(0) * ind.float()).sum()) + 0.5
                pts.append((xg / (G - 1) * cw, c['row_anchor'][k] * ch + y0))
                pk.append(float(prob[i, k, li]))
        res.append(dict(pts=pts, pk=pk, ex=float((ex[:, li] > 0.5).float().mean())))
    return res


def xs_at(pts, ys):
    p = sorted(pts, key=lambda q: q[1])
    return np.interp(ys, [q[1] for q in p], [q[0] for q in p])


def lane_rms(pts):
    p = np.array(pts)
    f = np.polyfit(p[:, 1], p[:, 0], 2)
    return float(np.sqrt(np.mean((np.polyval(f, p[:, 1]) - p[:, 0]) ** 2)))


def feats(det, img_w):
    L, R = det
    f = dict(ex=min(L['ex'], R['ex']), pk=0.0, res=np.inf, geo=False, rows=None)
    if len(L['pts']) < MIN_PTS or len(R['pts']) < MIN_PTS:
        return f
    f['pk'] = float(min(np.median(L['pk']), np.median(R['pk'])))
    lo = max(min(q[1] for q in L['pts']), min(q[1] for q in R['pts']))
    hi = min(max(q[1] for q in L['pts']), max(q[1] for q in R['pts']))
    if hi - lo < 1:
        return f
    ys = np.linspace(lo, hi, 12)
    xl, xr = xs_at(L['pts'], ys), xs_at(R['pts'], ys)
    w = xr - xl
    f['rows'] = (ys, xl, xr)
    if w[-1] > 1:
        f['res'] = max(lane_rms(L['pts']), lane_rms(R['pts'])) / w[-1]
    f['geo'] = bool(np.all(w > 0) and np.polyfit(ys, w, 1)[0] > 0 and WMIN <= w[-1] / img_w <= WMAX)
    return f


def agree(fa, fb):
    if fa['rows'] is None or fb['rows'] is None:
        return np.inf
    (ya, la, ra), (yb_, lb, rb) = fa['rows'], fb['rows']
    lo, hi = max(ya[0], yb_[0]), min(ya[-1], yb_[-1])
    if hi - lo < 1:
        return np.inf
    ys = np.linspace(lo, hi, 10)
    ca = (np.interp(ys, ya, la) + np.interp(ys, ya, ra)) / 2
    cb = (np.interp(ys, yb_, lb) + np.interp(ys, yb_, rb)) / 2
    return float(np.max(np.abs(ca - cb) / np.maximum(np.interp(ys, ya, ra - la), 1)))


def eval16(entries):
    out = []
    for xp, g, meta, full in entries:
        if g != G_DEV:
            continue
        ip = imgs.get(norm(xp.name))
        bgr = load(ip, None) if ip is not None else None
        if bgr is None:
            continue
        gt = gt_img(meta, s=bgr.shape[1] / meta[0])
        if gt is None:
            continue
        dets = {m: ufld_detail(NET16, bgr, m) for m in PREP15}
        fs = {m: feats(dets[m], bgr.shape[1]) for m in PREP15}
        L, R = dets[MAIN16]
        out.append(dict(ip=ip, meta=meta, det=dets[MAIN16], **{k: fs[MAIN16][k] for k in ('ex', 'pk', 'res', 'geo')},
                        agr=agree(fs['full'], fs['bottom']), err=err_img(L['pts'], R['pts'], gt)))
    return out


E_HOLD, E_MIS = eval16(HOLD), eval16(MIS)
nh16, nm16 = len(E_HOLD), len(E_MIS)
is_good = lambda r: r['err'] is not None and r['err'][1] <= MAX_ERR
is_bad  = lambda r: r['err'] is not None and r['err'][1] > MAX_ERR

# ── ① 검사값이 맞은 프레임과 틀린 프레임을 가르는가 (검증, 존재 비율 0.5 이상만) ──
base = [r for r in E_HOLD if r['ex'] >= 0.5]
G16, B16 = [r for r in base if is_good(r)], [r for r in base if is_bad(r)]
print(f'검증 {nh16}장 중 존재 비율 ≥ 0.5: 맞음 {len(G16)} / 틀림 {len(B16)} / 평가불가 {len(base) - len(G16) - len(B16)}')
print(f'{"검사":<6}{"맞음 p10":>10}{"p50":>8}{"p90":>8}   {"틀림 p10":>10}{"p50":>8}{"p90":>8}')
for k in ('ex', 'pk', 'res', 'agr'):
    q = lambda rs: np.percentile([min(r[k], 9.99) for r in rs], [10, 50, 90]) if rs else [np.nan] * 3
    a, b = q(G16), q(B16)
    print(f'{k:<6}{a[0]:>10.3f}{a[1]:>8.3f}{a[2]:>8.3f}   {b[0]:>10.3f}{b[1]:>8.3f}{b[2]:>8.3f}')
print(f'geo   통과율 맞음 {np.mean([r["geo"] for r in G16]) * 100 if G16 else 0:.0f}% / '
      f'틀림 {np.mean([r["geo"] for r in B16]) * 100 if B16 else 0:.0f}%')

# ── ② 임계 조합 탐색 — 후보값은 '맞은 프레임' 분포의 백분위에서 뽑는다 (검증만 사용) ──
def pct(k, qs):
    v = [r[k] for r in G16 if np.isfinite(r[k])]
    return sorted(set(np.round(np.percentile(v, qs), 4))) if v else []

GRID = dict(ex=[0.5, 0.6, 0.7, 0.8],
            pk=[0.0] + pct('pk', [5, 10, 20, 30, 50]),
            res=[np.inf] + pct('res', [50, 70, 80, 90, 95, 99]),
            agr=[np.inf] + pct('agr', [50, 70, 80, 90, 95, 99]),
            geo=[False, True])


def gate(r, g):
    return (r['ex'] >= g['ex'] and r['pk'] >= g['pk'] and r['res'] <= g['res']
            and r['agr'] <= g['agr'] and (r['geo'] or not g['geo']))


def score(rs, g):
    s = [r for r in rs if gate(r, g)]
    good, bad = sum(map(is_good, s)), sum(map(is_bad, s))
    return good, bad, len(s) - good - bad                 # 맞음, 오답, 평가불가(정답 행과 안 겹침)


def used(g):
    return tuple(k for k, off in (('pk', 0.0), ('res', np.inf), ('agr', np.inf), ('geo', False)) if g[k] != off)


combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
scored = [(g, score(E_HOLD, g)) for g in combos]
zero = sorted([(g, s) for g, s in scored if s[1] == 0 and s[0] > 0], key=lambda t: (-t[1][0], len(used(t[0]))))

def fmt_g(g):
    return ' '.join([f'ex≥{g["ex"]}'] + [f'{k}{"≥" if k == "pk" else "≤"}{g[k]:.3f}' for k in ('pk', 'res', 'agr') if k in used(g)]
                    + (['geo'] if g['geo'] else []))

def row16(g):
    hg, hb, hu = score(E_HOLD, g); mg, mb, mu = score(E_MIS, g)
    return (f'{(hg + hb) / nh16 * 100:>5.0f}% 오답{hb:>3} 불가{hu:>3}  │ '
            f'{(mg + mb) / max(nm16, 1) * 100:>5.0f}% 오답{mb:>3} 불가{mu:>3}')

print('\n' + '═' * 118)
print(f'검증 오답 0을 지키는 조합 — 검사 종류별 최고 (가용률 = 평가 가능 통과 / 전체, 불가 = 통과했지만 정답 행과 안 겹쳐 못 잰 수)')
print(f'{"쓴 검사":<16}{"검증 (튜닝용)":>26}  │ {"틀어짐 (판정용)":>22}   조합')
print('═' * 118)
best_by = {}
for g, s in zero:
    best_by.setdefault(used(g), g)
for u in sorted(best_by, key=lambda u: (len(u), u)):
    g = best_by[u]
    print(f'{("+".join(u) or "존재 비율만"):<16}{row16(g)}   {fmt_g(g)}')

if zero:
    final = zero[0][0]
    hg, hb, _ = score(E_HOLD, final); mg, mb, mu = score(E_MIS, final)
    av = (hg + hb) / nh16
    print(f'\n최종 후보: {fmt_g(final)}')
    print(f'  검증 가용률 {av * 100:.0f}% 오답 {hb} / 틀어짐 가용률 {(mg + mb) / max(nm16, 1) * 100:.0f}% 오답 {mb} (평가불가 통과 {mu})')
    ok = hb == 0 and av >= 0.40 and mb == 0
    print('  → 합격 기준 ' + ('충족 ✅ (3단계 Hailo 변환 검토 가능)' if ok else '미달 ❌ — 2단계(파인튜닝)로'))
    top = zero[:10]
    print(f'  검증 상위 {len(top)}개 조합 중 틀어짐 오답 0인 것: {sum(score(E_MIS, g)[1] == 0 for g, _ in top)}개 '
          '(적으면 최종 후보가 운으로 고른 조합일 수 있음)')
else:
    final = None
    print('\n검증 오답 0인 조합 없음 — 학습 없는 검사로는 못 거른다. 2단계(파인튜닝)로')


# ── ③ 시각화 ──
COL16 = [(0, 0, 255), (0, 255, 0)]

def show16(r, title):
    bgr = load(r['ip'], None)
    vis = bgr.copy()
    for y, a, b in gt_img(r['meta'], s=bgr.shape[1] / r['meta'][0]):
        cv2.circle(vis, (int(a), int(y)), 7, (255, 255, 0), 2); cv2.circle(vis, (int(b), int(y)), 7, (255, 255, 0), 2)
    for li, ln in enumerate(r['det']):
        for x, y in ln['pts']:
            cv2.circle(vis, (int(x), int(y)), 4, COL16[li], -1)
    plt.figure(figsize=(12, 6.5)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
    e = r['err']
    plt.title(f'{title} — {r["ip"].name}\n' + (f'최대오차={e[1]:.3f}  ' if e else '평가 불가  ')
              + f'ex={r["ex"]:.2f} pk={r["pk"]:.2f} res={r["res"]:.3f} agr={r["agr"]:.3f} geo={r["geo"]}'
              + '   (하늘색 원 = 정답, 빨강·초록 = 예측)', fontsize=9)
    plt.show()

def why(r, g):
    return ', '.join(k for k, bad in (('ex', r['ex'] < g['ex']), ('pk', r['pk'] < g['pk']), ('res', r['res'] > g['res']),
                                      ('agr', r['agr'] > g['agr']), ('geo', g['geo'] and not r['geo'])) if bad)

if final is not None:
    mis_bad = [r for r in E_MIS if gate(r, final) and is_bad(r)]
    print(f'\n틀어짐에서 최종 후보를 통과한 오답 {len(mis_bad)}건 (최대 4건 표시)')
    for r in mis_bad[:4]:
        show16(r, '틀어짐 오답 (통과됨)')
    caught = [r for r in B16 if not gate(r, final)]
    print(f'검증에서 15번 기준(존재 0.5)으론 통과했던 오답 {len(B16)}건 중 {len(caught)}건을 걸러냄 ({N_SHOW16}건 표시)')
    for r in caught[:N_SHOW16]:
        show16(r, f'검증 오답 → 걸러짐 ({why(r, final)})')
    lost = [r for r in G16 if not gate(r, final)]
    print(f'대신 맞은 프레임 {len(G16)}장 중 {len(lost)}장도 버려짐 ({min(N_SHOW16, len(lost))}건 표시)')
    for r in lost[:N_SHOW16]:
        show16(r, f'검증 정답 → 버려짐 ({why(r, final)})')


# ============================== 17번 셀 ==============================
# 17. 정답 쌍 점검 — UFLD 오답이 '모델이 틀린 것'인가, '정답(자차 차로 쌍)을 잘못 고른 것'인가
# 필요: 16번 셀 실행 (E_HOLD·E_MIS — UFLD CULane full 결과 캐시) + 4번 셀(ego_pair·gt_pair)·12번 셀(prep·fit_lanes_p/v)
# 학습 없음. GPU 불필요(16번 캐시만 씀)
#
# 의심: 4번 셀 ego_pair는 '그 행에 라벨이 있는 선'만 후보로 본다. 점선이 칸마다 따로 라벨링돼 있으면
#   칸 사이 행에서 그 점선이 후보에서 빠져, 한 칸 너머 선을 자차 차로 경계로 잡는다 (16번 틀어짐 오답 그림).
# ① 라벨 조각화 통계 — 라벨 선 하나가 화면 세로로 얼마나 짧은가
# ② UFLD 오답 분류 (존재 비율 ≥ EX17)
#    (가) 다른 쌍   — 예측 좌·우 선 모두 실제 라벨 선 위에 있지만 정답 쌍과 다른 선 → 정답 쌍 선택 문제 의심
#    (나) 라벨 밖   — 예측 선 중 하나가 어떤 라벨 선과도 안 겹침 → 모델 오류 또는 라벨 누락 (그림으로 판단)
#    (다) 같은 선   — 정답 쌍과 같은 선을 잡았지만 일부 행(주로 먼 쪽)에서 오차 초과 → 진짜 정밀도 문제
# ③ 정답 v2 — 같은 색 점선 조각을 위아래로 이어 붙인 선으로 자차 쌍을 다시 골라 P·V·UFLD를 같은 기준으로 재평가
#    (정답만 바꾸고 방법별 임계는 그대로 — 모든 방법에 같은 정답을 쓰므로 공정 비교)
EX17     = 0.8     # 15·16번 최종 후보 (존재 비율 ≥ 0.8)
MATCH    = 0.08    # 예측 점이 라벨 선에서 차로 폭의 이 비율 이내면 '그 선 위'
ON_FRAC  = 0.6     # 예측 점 중 이 비율 이상이 어떤 라벨 선 위에 있어야 '라벨 위의 선'
GAP_MAX  = 0.15    # 점선 조각 이어 붙이기 — 위아래 조각 사이 세로 간격 한도 (화면 높이 대비)
COVER_MIN = 0.3   # 정답 v2 후보 선 — 구간 세로 길이의 이 비율 이상을 덮어야 함
JOIN_TOL = 0.02    # 이어 붙이기 — 두 조각을 직선으로 늘였을 때 중간 높이에서의 x 차이 한도 (화면 폭 대비)
N_SHOW17 = 3

# ── ① 라벨 조각화 통계 ──
ext = defaultdict(list)
for xp, g, (W, H, lanes, pair, yb, yt), full in HOLD + MIS:
    for nm, p in lanes:
        ys = [q[1] for q in p]
        ext[nm].append((max(ys) - min(ys)) / H)
print('라벨 선 이름별 세로 길이 (화면 높이 대비) — 점선이 칸마다 따로면 Dash 쪽 중앙값이 작다')
print(f'{"이름":<24}{"개수":>7}{"p10":>8}{"중앙":>8}{"p90":>8}{"<0.1 비율":>11}')
for nm, v in sorted(ext.items(), key=lambda t: -len(t[1]))[:10]:
    v = np.array(v)
    print(f'{nm[:23]:<24}{len(v):>7}{np.percentile(v, 10):>8.3f}{np.median(v):>8.3f}{np.percentile(v, 90):>8.3f}'
          f'{(v < 0.1).mean() * 100:>10.0f}%')


# ── 점선 조각 이어 붙이기 (정답 v2) ──
def color_of(nm):
    return 'Y' if 'Yellow' in nm else ('W' if 'White' in nm else nm)


def merge_dashes(lanes, W, H):
    segs = [(nm, sorted(p, key=lambda q: q[1])) for nm, p in lanes if len(p) >= 2]
    parent = list(range(len(segs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def lin_x(p, y):
        ys, xs = np.array([q[1] for q in p]), np.array([q[0] for q in p])
        return float(xs.mean()) if ys[-1] - ys[0] < 1 else float(np.polyval(np.polyfit(ys, xs, 1), y))

    for i, (na, a) in enumerate(segs):
        for j, (nb, b) in enumerate(segs):
            if i == j or color_of(na) != color_of(nb):
                continue
            gap = a[0][1] - b[-1][1]                     # a 위끝 - b 아래끝 (b가 a보다 위=먼 쪽)
            if not (-0.01 * H <= gap <= GAP_MAX * H):
                continue
            ym = (a[0][1] + b[-1][1]) / 2
            if abs(lin_x(a, ym) - lin_x(b, ym)) <= JOIN_TOL * W:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(segs)):
        groups[find(i)].append(i)
    out = []
    for gi in groups.values():
        pts = sorted([q for i in gi for q in segs[i][1]], key=lambda q: q[1])
        out.append((segs[gi[0]][0] + (f' ×{len(gi)}' if len(gi) > 1 else ''), pts))
    return out


def ego_pair_v2(lines, W, H, yb, yt):
    """구간 아래끝 행(yb)에서 자차 쌍을 고른다. 선이 그 행까지 안 내려오면 직선으로 늘여 x를 구한다
    (4번 셀은 '그 행에 라벨이 있는 선'만 봐서, 아래끝에 칸이 비는 점선을 건너뛰었다).
    구간 세로 길이의 COVER_MIN 이상을 덮는 선만 후보 (정지선 같은 가로선 제외)."""
    c = []
    for nm, p in lines:
        ys = np.array([q[1] for q in p]); xs = np.array([q[0] for q in p])
        cov = min(ys.max(), yb * H) - max(ys.min(), yt * H)
        if cov < COVER_MIN * (yb - yt) * H or ys.max() - ys.min() < 1:
            continue
        xb = x_at(p, yb * H)
        if xb is None:
            near = ys >= np.percentile(ys, 50)            # 아래쪽 절반 점으로 직선 연장
            f = np.polyfit(ys[near], xs[near], 1) if np.ptp(ys[near]) > 1 else np.polyfit(ys, xs, 1)
            xb = float(np.polyval(f, yb * H))
        c.append((xb, x_at(p, yt * H), nm, p))
    Lc = [q for q in c if q[0] < W / 2]
    Rc = [q for q in c if q[0] >= W / 2]
    if not Lc or not Rc:
        return None
    return max(Lc, key=lambda q: q[0]), min(Rc, key=lambda q: q[0])


def regt(entries):
    """정답 v2 — 점선 조각을 이어 붙이고(merge_dashes) 아래끝 행에서 자차 쌍을 고른다(ego_pair_v2).
    반환 {xp: (meta, 쌍 바뀜 여부)}"""
    out, n_join = {}, 0
    for xp, g, (W, H, lanes, pair, yb, yt), full in entries:
        merged = merge_dashes(lanes, W, H)
        n_join += len(lanes) - len(merged)
        pr = ego_pair_v2(merged, W, H, yb, yt)
        if pr is None or not (0 < pr[0][0] < pr[1][0] < W):
            continue
        changed = False
        for old, new in zip(pair, pr):
            ys_ = [y for y in np.linspace(yt * H, yb * H, 9) if x_at(old[3], y) is not None and x_at(new[3], y) is not None]
            if ys_:
                changed |= max(abs(x_at(old[3], y) - x_at(new[3], y)) for y in ys_) > 0.02 * W
            else:
                changed = True                            # 겹치는 행이 없으면 다른 선
        out[xp] = ((W, H, merged, pr, yb, yt), changed)
    return out, n_join


GT2_HOLD, nj_h = regt(HOLD)
GT2_MIS, nj_m = regt(MIS)
print(f'\n점선 조각 이어 붙임: 검증 {nj_h}회 / 틀어짐 {nj_m}회')
print(f'정답 쌍이 바뀐 프레임: 검증 {sum(c for _, c in GT2_HOLD.values())}/{len(GT2_HOLD)} · '
      f'틀어짐 {sum(c for _, c in GT2_MIS.values())}/{len(GT2_MIS)}')


# ── ② UFLD 오답 분류 ──
def lane_on(pred_pts, lines, s, gt_rows):
    """예측 선 점마다 가장 가까운 라벨 선(거리/차로 폭). 반환: (라벨 위 비율, 가장 많이 겹친 라벨 번호)"""
    gy, gw = gt_rows[:, 0], gt_rows[:, 2] - gt_rows[:, 1]
    hits, ids = 0, []
    n = 0
    for x, y in pred_pts:
        w = float(np.interp(y, gy, gw))
        best = None
        for li, (nm, p) in enumerate(lines):
            xl = x_at(p, y / s)
            if xl is None:
                continue
            d = abs(x - xl * s) / max(w, 1)
            if best is None or d < best[0]:
                best = (d, li)
        if best is None:
            continue                                      # 이 행엔 라벨이 하나도 없음
        n += 1
        if best[0] <= MATCH:
            hits += 1; ids.append(best[1])
    if n < 4:
        return 0.0, None
    return hits / n, (Counter(ids).most_common(1)[0][0] if ids else None)


def classify(r, meta, s):
    W, H, lanes, pair, yb, yt = meta
    lines = merge_dashes(lanes, W, H)                     # 조각 사이 행도 '라벨 위'로 보도록 이어 붙인 선 기준
    gt = gt_img(meta, s=s)
    if gt is None:
        return '나'                                       # 정답 행이 모자라 판단 불가 — 보수적으로 '라벨 밖'
    gl = [('gt', pair[0][3])]
    gr = [('gt', pair[1][3])]
    (L, R) = r['det']
    fl, _ = lane_on(L['pts'], lines, s, gt)
    fr, _ = lane_on(R['pts'], lines, s, gt)
    if fl < ON_FRAC or fr < ON_FRAC:
        return '나'
    sl, _ = lane_on(L['pts'], gl, s, gt)
    sr, _ = lane_on(R['pts'], gr, s, gt)
    return '다' if sl >= ON_FRAC and sr >= ON_FRAC else '가'


def wrongs(E, ex_min):
    return [r for r in E if r['ex'] >= ex_min and r['err'] is not None and r['err'][1] > MAX_ERR]


CLS = {'가': '다른 쌍 (정답 선택 의심)', '나': '라벨 밖 (모델 오류/라벨 누락)', '다': '같은 선, 정밀도'}
meta_of = {xp: meta for xp, g, meta, full in HOLD + MIS}
ip2xp = {imgs.get(norm(xp.name)): xp for xp, g, meta, full in HOLD + MIS}
cls_ex = {}
print('\n' + '═' * 84)
print('UFLD(CULane full) 오답 분류 — 정답 v1(지금 정답) 기준')
print('═' * 84)
for ex_min in (0.5, EX17):
    for tag, E in (('검증', E_HOLD), ('틀어짐', E_MIS)):
        ws = wrongs(E, ex_min)
        cs = [classify(r, r['meta'], load(r['ip'], None).shape[1] / r['meta'][0]) for r in ws]
        if ex_min == EX17:
            for r, c in zip(ws, cs):
                cls_ex.setdefault(c, []).append((tag, r))
        cnt = Counter(cs)
        print(f'존재 ≥ {ex_min}  {tag:<4} 오답 {len(ws):>3}  →  ' + '  '.join(f'({k}) {cnt.get(k, 0)}' for k in CLS))
print('  ' + ' / '.join(f'({k}) {v}' for k, v in CLS.items()))


# ── ③ 정답 v2로 재평가 (P·V는 자기 임계, UFLD는 존재 비율 스윕) ──
def reeval(E, entries, GT2):
    """v1 칸은 v1 정답이 있는 프레임, v2 칸은 v2 정답이 있는 프레임으로 각각 잰다.
    v2에서 새로 평가 가능해진 프레임은 16번 캐시에 UFLD 결과가 없으므로 여기서 추론한다(GPU)."""
    rows, n_new = defaultdict(list), 0
    by_ip = {r['ip']: r for r in E}
    for xp, g, meta, full in entries:
        ip = imgs.get(norm(xp.name))
        bgr = load(ip, None) if ip is not None else None
        if bgr is None:
            continue
        w = bgr.shape[1]
        gts = {'v1': gt_img(meta, s=w / meta[0]),
               'v2': gt_img(GT2[xp][0], s=w / meta[0]) if xp in GT2 else None}
        if gts['v1'] is None and gts['v2'] is None:
            continue
        if ip in by_ip:
            det, ex = by_ip[ip]['det'], by_ip[ip]['ex']
        else:
            det = ufld_detail(NET16, bgr, MAIN16); ex = min(det[0]['ex'], det[1]['ex']); n_new += 1
        m, occ, Hm, _ = prep(ip, None)
        Hinv = np.linalg.inv(Hm)
        preds = {}
        for k, (fl, fr, conf, _) in (('P', fit_lanes_p(m)), ('V', fit_lanes_v(m, occ))):
            preds[k] = ((bev_to_img(fl, Hinv), bev_to_img(fr, Hinv)) if fl is not None else None, conf)
        preds['U'] = ((det[0]['pts'], det[1]['pts']), ex)
        for v, gt in gts.items():
            if gt is None:
                continue
            for k, (lr, conf) in preds.items():
                rows[(v, k)].append(dict(conf=conf, err=err_img(*lr, gt) if lr is not None else None))
    return rows, n_new


(RH2, nn_h), (RM2, nn_m) = reeval(E_HOLD, HOLD, GT2_HOLD), reeval(E_MIS, MIS, GT2_MIS)
print(f'v2에서 새로 평가 가능해진 프레임: 검증 {nn_h} / 틀어짐 {nn_m}')


def cnt17(rs, th):
    s = [r for r in rs if r['err'] is not None and r['conf'] >= th]
    return len(s), sum(r['err'][1] > MAX_ERR for r in s)


TH17 = {'P': [0.50], 'V': [0.47], 'U': [0.5, 0.6, 0.7, 0.8, 0.9]}
NAME17 = {'P': 'P(0.50)', 'V': 'V(0.47 현재)', 'U': 'UFLD 존재≥'}
print('\n' + '═' * 96)
print('정답 v1(지금) vs v2(점선 이어 붙임) — 칸 = 가용률/오답   (각 정답이 있는 프레임 기준, 장수는 아래)')
print('═' * 96)
print(f'{"방법":<18}{"검증 v1":>14}{"검증 v2":>14}   │{"틀어짐 v1":>14}{"틀어짐 v2":>14}')
for k in ('P', 'V', 'U'):
    for th in TH17[k]:
        cells = []
        for R_ in (RH2, RM2):
            for v in ('v1', 'v2'):
                rs = R_[(v, k)]
                p, b = cnt17(rs, th)
                cells.append(f'{p / max(len(rs), 1) * 100:.0f}%/{b}')
        nm = NAME17[k] + (f'{th}' if k == 'U' else '')
        print(f'{nm:<18}{cells[0]:>14}{cells[1]:>14}   │{cells[2]:>14}{cells[3]:>14}')
print(f'(장수 — 검증 v1 {len(RH2[("v1", "P")])} / v2 {len(RH2[("v2", "P")])} · 틀어짐 v1 {len(RM2[("v1", "P")])} / v2 {len(RM2[("v2", "P")])})')
u_ok = [th for th in np.round(np.arange(0.30, 0.96, 0.01), 2) if cnt17(RH2[('v2', 'U')], th)[1] == 0 and cnt17(RH2[('v2', 'U')], th)[0] > 0]
if u_ok:
    th = min(u_ok); p, _ = cnt17(RH2[('v2', 'U')], th); mp, mb = cnt17(RM2[('v2', 'U')], th)
    nh2, nm2 = len(RH2[('v2', 'U')]), len(RM2[('v2', 'U')])
    print(f'\n정답 v2 기준 UFLD 검증 오답 0인 가장 낮은 존재 임계 {th:.2f} → 검증 가용률 {p / nh2 * 100:.0f}% / '
          f'틀어짐 {mp / max(nm2, 1) * 100:.0f}% 오답 {mb}')
    print('  → 합격 기준(검증 오답 0 + 가용률 ≥ 40% + 틀어짐 오답 0) '
          + ('충족 ✅' if p / nh2 >= 0.40 and mb == 0 else '미달 ❌'))
else:
    print('\n정답 v2 기준에서도 UFLD 검증 오답 0인 임계 없음')


# ── 시각화 — 라벨 선 전체(가는 선, 번호) + 정답 v1(하늘색) · v2(노랑, 다를 때만) + 예측(빨강·초록) ──
LCOL = {'Y': (0, 200, 255), 'W': (230, 230, 230)}

def show17(r, title):
    xp = ip2xp[r['ip']]
    meta = meta_of[xp]
    bgr = load(r['ip'], None)
    s = bgr.shape[1] / meta[0]
    vis = bgr.copy()
    for li, (nm, p) in enumerate(merge_dashes(meta[2], meta[0], meta[1])):
        q = np.int32([(x * s, y * s) for x, y in sorted(p, key=lambda t: t[1])])
        cv2.polylines(vis, [q], False, LCOL.get(color_of(nm), (160, 160, 160)), 2)
        cv2.putText(vis, str(li), tuple(int(v) for v in q[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(vis, str(li), tuple(int(v) for v in q[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    for y, a, b in gt_img(meta, s=s):
        cv2.circle(vis, (int(a), int(y)), 9, (255, 255, 0), 2); cv2.circle(vis, (int(b), int(y)), 9, (255, 255, 0), 2)
    g2 = (GT2_HOLD.get(xp) or GT2_MIS.get(xp))
    gt2 = gt_img(g2[0], s=s) if g2 and g2[1] else None      # v2 쌍이 바뀌었어도 겹치는 행이 5개 미만이면 None
    if gt2 is not None:
        for y, a, b in gt2:
            cv2.circle(vis, (int(a), int(y)), 14, (0, 255, 255), 2); cv2.circle(vis, (int(b), int(y)), 14, (0, 255, 255), 2)
    for li, ln in enumerate(r['det']):
        for x, y in ln['pts']:
            cv2.circle(vis, (int(x), int(y)), 4, [(0, 0, 255), (0, 255, 0)][li], -1)
    plt.figure(figsize=(12, 6.5)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title(f'{title} — {r["ip"].name}  최대오차(v1)={r["err"][1]:.3f}\n'
              '가는 선 = 라벨(노랑/흰, 번호 = 이어 붙인 선)  작은 하늘색 원 = 정답 v1  큰 노랑 원 = 정답 v2(바뀐 경우)  '
              '빨강·초록 = 예측', fontsize=9)
    plt.show()

for k in CLS:
    items = cls_ex.get(k, [])
    print(f'\n({k}) {CLS[k]} — {len(items)}건 (최대 {N_SHOW17}건 표시, 틀어짐 먼저)')
    for tag, r in sorted(items, key=lambda t: t[0] != '틀어짐')[:N_SHOW17]:
        show17(r, f'({k}) {tag}')


# ============================== 18번 셀 ==============================
# 18. 딥러닝 차선 2단계 — UFLDv2 CULane을 AI Hub '보정용' 시퀀스로 파인튜닝
# 필요: 17번 셀 실행 (merge_dashes·GT2_HOLD·GT2_MIS·RH2·RM2·show17 사용). GPU 런타임 필수
#
# 목표: 17번에서 남은 검증 오답의 대부분인 (다) '같은 선, 먼 행 오차'를 줄인다.
# 데이터 — 4번 셀과 같은 방식(같은 난수)으로 시퀀스를 다시 나눠 '보정용' 시퀀스만 학습에 쓴다.
#   검증(HOLD)·틀어짐(MIS) 시퀀스는 학습에 절대 안 들어간다(겹침 검사 출력).
#   학습 시퀀스 중 VAL_FRAC18만큼은 따로 떼어 '언제 멈출지'(최고 에폭 선택)에만 쓴다 — 검증 247장은 마지막 판정에만.
# 정답 — 17번 정답 v2 방식(점선 조각 이어 붙이기 + 아래끝 행에서 쌍 선택)으로 차선 4개(옆·자차좌·자차우·옆)를 만든다.
#   자차 좌·우 중 하나라도 없는 프레임은 뺀다('차선 없음'을 잘못 배우지 않게).
# 손실 — 원 저장소와 같은 구성(행 위치 focal CE + 존재 CE + 평균 위치 L1×0.05). 열(col) 출력은 쓰지 않으므로 학습 안 함.
# HEAD18 — 'orig'  원래 출력층 그대로 (2억 600만 파라미터)
#          'row100' 행 출력만 + 격자 200→100 (사전학습 가중치를 이웃 칸 평균으로 옮겨 시작). 파라미터 약 1/3 — NPU용 후보
# 결과는 드라이브 ufld/ft_{HEAD18}.pth에 저장, 다음엔 불러온다 (RETRAIN18=True면 다시 학습)
import copy, math
import torch.nn.functional as F

HEAD18     = 'orig'
EPOCHS18   = 15
BS18       = 16
LR18       = 1e-4
VAL_FRAC18 = 0.15      # 학습 시퀀스 중 에폭 선택용으로 떼어 둘 비율
MIN_GAP18  = 0.06      # 옆 차선 — 이미 고른 선과 아래끝 x 차이가 화면 폭의 이 비율 이하면 겹선으로 보고 건너뜀
RETRAIN18  = False
CKPT18     = UFLD_DIR / f'ft_{HEAD18}.pth'
torch.manual_seed(0); np.random.seed(0)

# ── 데이터 나누기 (4번 셀과 같은 순서·같은 난수) ──
rng18, CAL18 = random.Random(0), []
for g in sorted({f[1] for f in FRAMES}):
    fr_all = [f for f in FRAMES if f[1] == g]
    if len(fr_all) < 10:
        continue
    fr, _, _, _ = split_by_seq(fr_all, rng18)
    if g == G_DEV:
        CAL18 = fr
eval_seqs = {seq_of(xp) for xp, *_ in HOLD + MIS}
print(f'보정용 프레임 {len(CAL18)}장 — 검증·틀어짐 시퀀스와 겹침 {sum(seq_of(f[0]) in eval_seqs for f in CAL18)}장 (0이어야 함)')
CAL18 = [f for f in CAL18 if seq_of(f[0]) not in eval_seqs]
YB18, YT18 = REF[G_DEV]


# ── 정답: 차선 4개 ──
def cands(lines, W, H, yb, yt):
    """ego_pair_v2와 같은 후보·아래끝 x — (xb, xt, 이름, 점)"""
    c = []
    for nm, p in lines:
        ys = np.array([q[1] for q in p]); xs = np.array([q[0] for q in p])
        cov = min(ys.max(), yb * H) - max(ys.min(), yt * H)
        if cov < COVER_MIN * (yb - yt) * H or ys.max() - ys.min() < 1:
            continue
        xb = x_at(p, yb * H)
        if xb is None:
            near = ys >= np.percentile(ys, 50)
            f = np.polyfit(ys[near], xs[near], 1) if np.ptp(ys[near]) > 1 else np.polyfit(ys, xs, 1)
            xb = float(np.polyval(f, yb * H))
        c.append((xb, x_at(p, yt * H), nm, p))
    return c


def lanes4(lanes, W, H, yb, yt):
    c = cands(merge_dashes(lanes, W, H), W, H, yb, yt)
    Lc = sorted([q for q in c if q[0] < W / 2], key=lambda q: -q[0])
    Rc = sorted([q for q in c if q[0] >= W / 2], key=lambda q: q[0])
    if not Lc or not Rc or not (0 < Lc[0][0] < Rc[0][0] < W):
        return None
    nb = lambda side, ego: next((q for q in side[1:] if abs(q[0] - ego[0]) > MIN_GAP18 * W), None)
    return [nb(Lc, Lc[0]), Lc[0], Rc[0], nb(Rc, Rc[0])]


def row_labels(l4, W, H, G, anchors, flip=False):
    """행 앵커마다 차선 4개의 격자 칸 번호 (-1 = 없음). 원 저장소처럼 라벨 아래쪽 끝은 화면 아래까지 직선 연장."""
    n = len(anchors)
    lab = np.full((n, 4), -1, np.int64)
    ya = np.asarray(anchors) * H
    for li, q in enumerate(l4):
        if q is None:
            continue
        ps = sorted(q[3], key=lambda t: t[1])
        ys = np.array([t[1] for t in ps]); xs = np.array([t[0] for t in ps])
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
        lab[ok, 3 - li if flip else li] = np.minimum((x[ok] / W * (G - 1)).astype(np.int64), G - 1)
    return lab


def to_input(bgr, c):
    """15번 ufld_lanes와 같은 전처리 (full) — uint8 RGB (in_h, in_w, 3)"""
    return cv2.cvtColor(cv2.resize(bgr, (c['in_w'], int(c['in_h'] / c['crop'])), interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_BGR2RGB)[-c['in_h']:]


# ── 모델 ──
def make_net18():
    base = copy.deepcopy(nets15['culane'])
    if HEAD18 == 'orig':
        return base
    c = base.c
    c2 = dict(c, grid_row=c['grid_row'] // 2, n_col=0)            # 열 출력 제거 (d2 = d4 = 0)
    net = UFLDv2(c2)
    sd, own = base.state_dict(), net.state_dict()
    for k in own:
        if not k.startswith('cls.3'):
            own[k] = sd[k]
    W3, b3 = sd['cls.3.weight'], sd['cls.3.bias']
    G0, n, L = c['grid_row'], c['n_row'], 4
    d1, d2, d3 = G0 * n * L, c['grid_col'] * c['n_col'] * L, 2 * n * L
    lw = W3[:d1].view(G0 // 2, 2, n * L, -1).mean(1).reshape(-1, W3.shape[1])   # 이웃 두 칸 평균
    lb = b3[:d1].view(G0 // 2, 2, n * L).mean(1).reshape(-1)
    own['cls.3.weight'] = torch.cat([lw, W3[d1 + d2:d1 + d2 + d3]])
    own['cls.3.bias'] = torch.cat([lb, b3[d1 + d2:d1 + d2 + d3]])
    net.load_state_dict(own)
    del base
    return net.to(DEV)


net18 = make_net18()
C18 = net18.c
n_par18 = sum(p.numel() for p in net18.parameters()) / 1e6
print(f'HEAD18={HEAD18}: 파라미터 {n_par18:.0f}M (fp32 {n_par18 * 4:.0f}MB / int8 약 {n_par18:.0f}MB)')

# ── 학습 데이터 만들기 (전처리 결과를 메모리에 올림) ──
seqs = sorted({seq_of(f[0]) for f in CAL18})
random.Random(1).shuffle(seqs)
val_s = set(seqs[:max(1, round(len(seqs) * VAL_FRAC18))])
X18, Y18, YF18, VAL18, skip = [], [], [], [], Counter()
for xp, g, W, H, lanes in CAL18:
    l4 = lanes4(lanes, W, H, YB18, YT18)
    if l4 is None:
        skip['자차 좌우 없음'] += 1; continue
    ip = imgs.get(norm(xp.name))
    bgr = load(ip, None) if ip is not None else None
    if bgr is None:
        skip['이미지 없음'] += 1; continue
    if seq_of(xp) in val_s:
        meta = (W, H, merge_dashes(lanes, W, H), (l4[1], l4[2]), YB18, YT18)
        if gt_img(meta, s=bgr.shape[1] / W) is not None:
            VAL18.append((ip, meta))
        continue
    X18.append(to_input(bgr, C18))
    Y18.append(row_labels(l4, W, H, C18['grid_row'], C18['row_anchor']))
    YF18.append(row_labels(l4, W, H, C18['grid_row'], C18['row_anchor'], flip=True))
X18, Y18, YF18 = np.stack(X18), np.stack(Y18), np.stack(YF18)
print(f'학습 {len(X18)}장 ({len(seqs) - len(val_s)}시퀀스) · 에폭 선택용 {len(VAL18)}장 ({len(val_s)}시퀀스) · 제외 {dict(skip)}')
print(f'  프레임당 라벨 차선 {(Y18 >= 0).any(1).sum(1).mean():.2f}개 (최대 4) · 메모리 {X18.nbytes / 1e9:.1f}GB')

MEAN18 = torch.tensor([.485, .456, .406], device=DEV).view(1, 3, 1, 1)
STD18 = torch.tensor([.229, .224, .225], device=DEV).view(1, 3, 1, 1)


def batch18(idx, train=True):
    x = torch.from_numpy(X18[idx]).to(DEV).permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(Y18[idx]).to(DEV)
    if train:
        fl = torch.rand(len(idx), device=DEV) < 0.5                       # 좌우 뒤집기 (차선 번호도 뒤집힘)
        x[fl] = torch.flip(x[fl], dims=[3])
        y[fl] = torch.from_numpy(YF18[idx]).to(DEV)[fl]
        a = torch.empty(len(idx), 1, 1, 1, device=DEV).uniform_(0.7, 1.3)   # 밝기·대비
        b = torch.empty(len(idx), 1, 1, 1, device=DEV).uniform_(-0.1, 0.1)
        x = (x * a + b).clamp(0, 1)
    return ((x - MEAN18) / STD18).contiguous(), y          # permute로 생긴 channels_last 배치 → 모델의 view가 실패하므로 정렬


def loss18(out, lab):
    loc, ex = out['loc_row'].float(), out['exist_row'].float()             # (B,G,n,4), (B,2,n,4)
    valid = lab >= 0
    if not valid.any():
        return F.cross_entropy(ex, valid.long())
    logp = F.log_softmax(loc, 1)
    focal = (1 - logp.exp()) ** 2 * logp
    l_cls = -(focal.gather(1, lab.clamp(min=0)[:, None]).squeeze(1))[valid].mean()
    l_ext = F.cross_entropy(ex, valid.long())
    grid = torch.arange(loc.shape[1], device=loc.device).view(1, -1, 1, 1)
    l_mean = F.smooth_l1_loss((loc.softmax(1) * grid).sum(1)[valid], lab[valid].float())
    return l_cls + l_ext + 0.05 * l_mean


def score18(net, items, ex_min=0.5):
    """에폭 선택용 — 존재 비율 ≥ ex_min 통과분 중 맞음/오답 (정답 v2 방식, 원본 화면 기준)"""
    net.eval(); good = bad = 0
    for ip, meta in items:
        bgr = load(ip, None)
        L, R = ufld_detail(net, bgr, MAIN16)
        if min(L['ex'], R['ex']) < ex_min:
            continue
        e = err_img(L['pts'], R['pts'], gt_img(meta, s=bgr.shape[1] / meta[0]))
        if e is not None:
            good += e[1] <= MAX_ERR; bad += e[1] > MAX_ERR
    net.train()
    return good, bad


# ── 학습 ──
if CKPT18.exists() and not RETRAIN18:
    ck = torch.load(CKPT18, map_location='cpu', weights_only=True)
    net18.load_state_dict(ck['state_dict'])
    print(f'저장된 파인튜닝 가중치 불러옴: {CKPT18.name} (에폭 {ck["epoch"]}, 선택용 맞음 {ck["good"]} / 오답 {ck["bad"]})')
else:
    g0, b0 = score18(net18, VAL18)
    print(f'학습 전 선택용: 맞음 {g0} / 오답 {b0} (존재 ≥ 0.5)')
    opt = torch.optim.AdamW(net18.parameters(), lr=LR18, weight_decay=1e-4)
    steps = EPOCHS18 * math.ceil(len(X18) / BS18)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == 'cuda')
    best = (g0 - 5 * b0, 0, g0, b0, copy.deepcopy({k: v.cpu() for k, v in net18.state_dict().items()}))
    net18.train()
    for ep in range(1, EPOCHS18 + 1):
        t0, tot, nb = time.time(), 0.0, 0
        perm = np.random.permutation(len(X18))
        for i in range(0, len(perm), BS18):
            idx = np.sort(perm[i:i + BS18])
            x, y = batch18(idx)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
                out = net18(x)
            loss = loss18(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); nb += 1
        g_, b_ = score18(net18, VAL18)
        sc = g_ - 5 * b_
        mark = ''
        if sc > best[0]:
            best = (sc, ep, g_, b_, copy.deepcopy({k: v.cpu() for k, v in net18.state_dict().items()})); mark = '  ← 최고'
        print(f'에폭 {ep:>2}  손실 {tot / nb:.4f}  선택용 맞음 {g_:>3} / 오답 {b_:>2}  ({time.time() - t0:.0f}초){mark}')
    _, ep_b, g_b, b_b, sd_b = best
    net18.load_state_dict(sd_b)
    torch.save({'state_dict': sd_b, 'head': HEAD18, 'epoch': ep_b, 'good': g_b, 'bad': b_b}, CKPT18)
    print(f'최고 에폭 {ep_b} (선택용 맞음 {g_b} / 오답 {b_b}) → 저장 {CKPT18}')
net18.eval()

# ── 최종 평가 — 검증 247장·틀어짐 196장, 정답 v2 (17번과 같은 프레임·같은 기준) ──
def eval18(entries, GT2):
    rows = []
    for xp, g, meta, full in entries:
        ip = imgs.get(norm(xp.name))
        bgr = load(ip, None) if ip is not None else None
        if bgr is None or xp not in GT2:
            continue
        gt = gt_img(GT2[xp][0], s=bgr.shape[1] / meta[0])
        if gt is None:
            continue
        det = ufld_detail(net18, bgr, MAIN16)
        rows.append(dict(ip=ip, det=det, conf=min(det[0]['ex'], det[1]['ex']),
                         err=err_img(det[0]['pts'], det[1]['pts'], gt)))
    return rows


F_HOLD, F_MIS = eval18(HOLD, GT2_HOLD), eval18(MIS, GT2_MIS)
dummy = torch.zeros(1, 3, C18['in_h'], C18['in_w'], device=DEV)
with torch.no_grad():
    for _ in range(3): net18(dummy)
    if DEV == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): net18(dummy)
    if DEV == 'cuda': torch.cuda.synchronize()
print(f'\n추론 {(time.time() - t0) / 20 * 1000:.1f}ms/장 ({DEV}, 참고용)')


def med18(rs, th):
    e = [r['err'][0] for r in rs if r['err'] is not None and r['conf'] >= th]
    return np.median(e) if e else float('nan')


print('\n' + '═' * 100)
print('정답 v2 기준 — 칸 = 가용률/오답   (사전학습 = 17번 표의 UFLD 줄과 같은 값)')
print('═' * 100)
print(f'{"방법":<18}{"검증 사전학습":>14}{"검증 파인튜닝":>14}   │{"틀어짐 사전학습":>15}{"틀어짐 파인튜닝":>15}')
for k, th in (('P', 0.50), ('V', 0.47)):
    c_ = [f'{cnt17(R_[("v2", k)], th)[0] / max(len(R_[("v2", k)]), 1) * 100:.0f}%/{cnt17(R_[("v2", k)], th)[1]}' for R_ in (RH2, RM2)]
    print(f'{NAME17[k]:<18}{c_[0]:>14}{"":>14}   │{c_[1]:>15}')
for th in (0.5, 0.6, 0.7, 0.8, 0.9):
    cells = []
    for pre, ft in ((RH2[('v2', 'U')], F_HOLD), (RM2[('v2', 'U')], F_MIS)):
        for rs in (pre, ft):
            p, b = cnt17(rs, th)
            cells.append(f'{p / max(len(rs), 1) * 100:.0f}%/{b}')
    print(f'{"UFLD 존재≥" + str(th):<18}{cells[0]:>14}{cells[1]:>14}   │{cells[2]:>15}{cells[3]:>15}')
print(f'오차중앙(존재≥0.5) — 검증 {med18(F_HOLD, 0.5):.4f} · 틀어짐 {med18(F_MIS, 0.5):.4f}   '
      f'(장수 검증 {len(F_HOLD)} · 틀어짐 {len(F_MIS)})')

ok18 = [th for th in np.round(np.arange(0.30, 0.96, 0.01), 2) if cnt17(F_HOLD, th)[1] == 0 and cnt17(F_HOLD, th)[0] > 0]
TH18 = min(ok18) if ok18 else None
if TH18 is not None:
    p, _ = cnt17(F_HOLD, TH18); mp, mb = cnt17(F_MIS, TH18)
    av, mav = p / len(F_HOLD), mp / max(len(F_MIS), 1)
    print(f'\n파인튜닝 — 검증 오답 0인 가장 낮은 존재 임계 {TH18:.2f} → 검증 가용률 {av * 100:.0f}% / 틀어짐 {mav * 100:.0f}% 오답 {mb}')
    print('  → 합격 기준(검증 오답 0 + 가용률 ≥ 40% + 틀어짐 오답 0) '
          + ('충족 ✅' if av >= 0.40 and mb == 0 else '미달 ❌')
          + ('' if mb == 0 else '  — 틀어짐 오답은 라벨 누락일 수 있으니 아래 그림으로 직접 확인'))
else:
    print('\n파인튜닝 — 검증 오답 0인 임계 없음')

# ── 그림 — 파인튜닝 오답 (검증: 존재 ≥ 0.5 / 틀어짐: 위 임계 통과분) ──
for tag, rs, th in (('검증', F_HOLD, 0.5), ('틀어짐', F_MIS, TH18 if TH18 is not None else 0.5)):
    bad = sorted([r for r in rs if r['conf'] >= th and r['err'] is not None and r['err'][1] > MAX_ERR],
                 key=lambda r: -r['err'][1])
    print(f'\n파인튜닝 {tag} 오답 (존재 ≥ {th}) {len(bad)}건 — 최대 4건 표시')
    for r in bad[:4]:
        show17(r, f'파인튜닝 {tag} 오답')


# ============================== 19번 셀 ==============================
# 19. 정답 v3 + 재학습 + 측정 범위 분석 — 18번 실패 원인(에폭 선택 11장·겹선 정답)을 고치고, 어디까지 재야 하는지 본다
# 필요: 18번 셀 실행 (CAL18·YB18·YT18·net18·to_input·row_labels·loss18·make_net18·cands 사용). GPU 런타임 필수
#
# ① 정답 v3 — v2(점선 이어 붙이기 + 아래끝 행 쌍 선택)에
#    · 겹선 합치기: 아래끝 x가 화면 폭의 DOUBLE19 안쪽으로 붙은 선들은 한 경계로 보고 가운데 선으로 바꾼다
#      (18번 오답 1번 그림 — 겹선 두 줄이 자차 좌·우 경계로 잡혀 차로 폭 60px, 최대오차 3.27)
#    · 차로 폭 검사: 아래끝 행 자차 차로 폭이 화면 폭의 WMIN19~WMAX19 밖이면 정답 없음 (4번 셀 보정과 같은 범위)
#    학습 정답과 평가 정답 모두 v3로 만든다.
# ② 재학습 — 학습 시퀀스의 VAL_FRAC19를 떼어 '선택용 손실'이 가장 낮은 에폭을 고른다 (18번은 11장 맞음/오답 수로 골랐다)
# ③ 측정 범위 — 판정 구간 위쪽 끝을 CUTS19로 바꿔 가며 P·V·UFLD(사전학습·18번·19번)를 같은 정답으로 비교.
#    UFLD 신뢰도 = 그 범위 안 행 앵커 중 자차 좌·우 차선이 '존재'로 나온 비율의 작은 값
# ④ 차량 위치 — 같은 사진에서 YOLO(8번 셀 det)로 자차·옆 차로 차량의 바퀴 아래쪽 높이를 재서
#    범위마다 앞차의 몇 %를 잴 수 있는지 본다. 범위를 좁힐지(측정 불가 차량이 생김)는 이 표로 판단
import gc

WMIN19, WMAX19 = 0.25, 0.95
DOUBLE19   = 0.08      # 겹선 — 아래끝 x 간격이 화면 폭의 이 비율 미만이면 한 경계
CUTS19     = [0.56, 0.60, 0.64, 0.68]
HEAD19     = 'orig'
EPOCHS19   = 20
BS19       = 16
LR19       = 1e-4
VAL_FRAC19 = 0.3
RETRAIN19  = False
CKPT19     = UFLD_DIR / f'ft_{HEAD19}_v3.pth'
N_SHOW19   = 3
torch.manual_seed(0); np.random.seed(0)
for _n in ('X18', 'Y18', 'YF18'):                 # 18번 학습 배열은 더 안 씀 — 메모리 확보
    globals().pop(_n, None)
gc.collect()


# ── ① 정답 v3 ──
def clusters(c, W, H, yt):
    """후보 선(xb 기준)을 겹선끼리 묶어 한 선으로. 묶인 선은 겹치는 행에서 x 평균."""
    c = sorted(c, key=lambda q: q[0])
    groups = []
    for q in c:
        if groups and q[0] - groups[-1][-1][0] < DOUBLE19 * W:
            groups[-1].append(q)
        else:
            groups.append([q])
    out = []
    for gm in groups:
        if len(gm) == 1:
            out.append(gm[0]); continue
        ps = [m[3] for m in gm]
        ys = sorted({q[1] for p in ps for q in p})
        av = [(float(np.mean([x_at(p, y) for p in ps])), y) for y in ys if all(x_at(p, y) is not None for p in ps)]
        if len(av) < 2:
            out.append(max(gm, key=lambda m: np.ptp([q[1] for q in m[3]]))); continue
        out.append((float(np.mean([m[0] for m in gm])), x_at(av, yt * H), '겹선', av))
    return out


def ego3(lanes, W, H, yb, yt):
    """(묶은 선 목록, 자차 좌 번호, 자차 우 번호) 또는 None"""
    cl = clusters(cands(merge_dashes(lanes, W, H), W, H, yb, yt), W, H, yt)
    Li = [i for i, q in enumerate(cl) if q[0] < W / 2]
    Ri = [i for i, q in enumerate(cl) if q[0] >= W / 2]
    if not Li or not Ri:
        return None
    l, r = Li[-1], Ri[0]
    if not (WMIN19 <= (cl[r][0] - cl[l][0]) / W <= WMAX19):
        return None
    return cl, l, r


def regt3(entries):
    out = {}
    for xp, g, (W, H, lanes, pair, yb, yt), full in entries:
        e = ego3(lanes, W, H, yb, yt)
        if e is not None:
            cl, l, r = e
            out[xp] = (W, H, [(q[2], q[3]) for q in cl], (cl[l], cl[r]), yb, yt)
    return out


GT3_HOLD, GT3_MIS = regt3(HOLD), regt3(MIS)
print(f'정답 v3: 검증 {len(GT3_HOLD)}/{len(HOLD)} · 틀어짐 {len(GT3_MIS)}/{len(MIS)} 프레임에 자차 쌍 있음')


def gt_rows(meta, s, cut, n=12):
    """정답 자차 좌/우 경계 — 원본 좌표 (y, xl, xr), 화면 높이 cut ~ 아래끝 사이 n행 중 라벨이 있는 행 (5행 미만이면 None)"""
    W, H, lines, pair, yb, yt = meta
    rows = []
    for y in np.linspace(cut * H, yb * H, n):
        a, b = x_at(pair[0][3], y), x_at(pair[1][3], y)
        if a is not None and b is not None:
            rows.append((y * s, a * s, b * s))
    return np.array(rows) if len(rows) >= 5 else None


# ── ② 재학습 ──
_h = HEAD18; HEAD18 = HEAD19; net19 = make_net18(); HEAD18 = _h
C19 = net19.c
seqs19 = sorted({seq_of(f[0]) for f in CAL18})
random.Random(2).shuffle(seqs19)
val19 = set(seqs19[:max(2, round(len(seqs19) * VAL_FRAC19))])
XT, YT_, YFT, XV, YV, VAL19M, skip19 = [], [], [], [], [], [], Counter()
for xp, g, W, H, lanes in CAL18:
    e = ego3(lanes, W, H, YB18, YT18)
    if e is None:
        skip19['자차 쌍 없음/폭 이상'] += 1; continue
    cl, l, r = e
    l4 = [cl[l - 1] if l > 0 else None, cl[l], cl[r], cl[r + 1] if r + 1 < len(cl) else None]
    ip = imgs.get(norm(xp.name))
    bgr = load(ip, None) if ip is not None else None
    if bgr is None:
        skip19['이미지 없음'] += 1; continue
    x = to_input(bgr, C19)
    y = row_labels(l4, W, H, C19['grid_row'], C19['row_anchor'])
    if seq_of(xp) in val19:
        XV.append(x); YV.append(y)
        VAL19M.append((ip, (W, H, [], (cl[l], cl[r]), YB18, YT18)))
    else:
        XT.append(x); YT_.append(y); YFT.append(row_labels(l4, W, H, C19['grid_row'], C19['row_anchor'], flip=True))
XT, YT_, YFT, XV, YV = map(np.stack, (XT, YT_, YFT, XV, YV))
print(f'학습 {len(XT)}장 ({len(seqs19) - len(val19)}시퀀스) · 선택용 {len(XV)}장 ({len(val19)}시퀀스) · 제외 {dict(skip19)}')


def batch19(X, Y, YF, idx, train):
    x = torch.from_numpy(X[idx]).to(DEV).permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(Y[idx]).to(DEV)
    if train:
        fl = torch.rand(len(idx), device=DEV) < 0.5
        x[fl] = torch.flip(x[fl], dims=[3])
        y[fl] = torch.from_numpy(YF[idx]).to(DEV)[fl]
        a = torch.empty(len(idx), 1, 1, 1, device=DEV).uniform_(0.7, 1.3)
        b = torch.empty(len(idx), 1, 1, 1, device=DEV).uniform_(-0.1, 0.1)
        x = (x * a + b).clamp(0, 1)
    return ((x - MEAN18) / STD18).contiguous(), y


@torch.no_grad()
def vloss19(net):
    net.eval(); tot = 0.0
    for i in range(0, len(XV), BS19):
        idx = np.arange(i, min(i + BS19, len(XV)))
        x, y = batch19(XV, YV, None, idx, False)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
            out = net(x)
        tot += float(loss18(out, y)) * len(idx)
    net.train()
    return tot / len(XV)


def vscore19(net):
    """참고용 — 선택용 프레임에서 존재 ≥ 0.5 통과분 맞음/오답 (정답 v3, 범위 0.56)"""
    net.eval(); good = bad = 0
    for ip, meta in VAL19M:
        bgr = load(ip, None)
        gt = gt_rows(meta, bgr.shape[1] / meta[0], 0.56)
        if gt is None:
            continue
        L, R = ufld_detail(net, bgr, MAIN16)
        if min(L['ex'], R['ex']) < 0.5:
            continue
        e = err_img(L['pts'], R['pts'], gt)
        if e is not None:
            good += e[1] <= MAX_ERR; bad += e[1] > MAX_ERR
    net.train()
    return good, bad


if CKPT19.exists() and not RETRAIN19:
    ck = torch.load(CKPT19, map_location='cpu', weights_only=True)
    net19.load_state_dict(ck['state_dict'])
    print(f'저장된 가중치 불러옴: {CKPT19.name} (에폭 {ck["epoch"]}, 선택용 손실 {ck["vloss"]:.4f})')
else:
    v0 = vloss19(net19); g0, b0 = vscore19(net19)
    print(f'학습 전  선택용 손실 {v0:.4f}  (참고: 맞음 {g0} / 오답 {b0})')
    opt = torch.optim.AdamW(net19.parameters(), lr=LR19, weight_decay=1e-4)
    steps = EPOCHS19 * math.ceil(len(XT) / BS19)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler('cuda', enabled=DEV == 'cuda')
    best = (v0, 0, {k: v.cpu().clone() for k, v in net19.state_dict().items()})
    net19.train()
    for ep in range(1, EPOCHS19 + 1):
        t0, tot, nb = time.time(), 0.0, 0
        perm = np.random.permutation(len(XT))
        for i in range(0, len(perm), BS19):
            idx = np.sort(perm[i:i + BS19])
            x, y = batch19(XT, YT_, YFT, idx, True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
                out = net19(x)
            loss = loss18(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()); nb += 1
        vl = vloss19(net19); g_, b_ = vscore19(net19)
        mark = ''
        if vl < best[0]:
            best = (vl, ep, {k: v.cpu().clone() for k, v in net19.state_dict().items()}); mark = '  ← 최저'
        print(f'에폭 {ep:>2}  학습 손실 {tot / nb:.4f}  선택용 손실 {vl:.4f}  (참고: 맞음 {g_} / 오답 {b_})  '
              f'({time.time() - t0:.0f}초){mark}')
    vl_b, ep_b, sd_b = best
    net19.load_state_dict(sd_b)
    torch.save({'state_dict': sd_b, 'head': HEAD19, 'epoch': ep_b, 'vloss': vl_b}, CKPT19)
    print(f'선택 에폭 {ep_b} (선택용 손실 {vl_b:.4f}) → 저장 {CKPT19}')
net19.eval()
del XT, YT_, YFT; gc.collect()


# ── ③ 측정 범위별 비교 ──
NETS19 = {'UFLD 사전학습': nets15['culane'], 'UFLD 18번': net18, 'UFLD 19번': net19}
E_BY = {r['ip']: r for r in E_HOLD + E_MIS}


def collect19(entries, GT3):
    out = []
    for xp, g, meta, full in entries:
        if xp not in GT3:
            continue
        ip = imgs.get(norm(xp.name))
        bgr = load(ip, None) if ip is not None else None
        if bgr is None:
            continue
        m, occ, Hm, _ = prep(ip, None)
        Hinv = np.linalg.inv(Hm)
        preds = {}
        for k, (fl, fr, conf, _) in (('P(0.50)', fit_lanes_p(m)), ('V(0.47 현재)', fit_lanes_v(m, occ))):
            preds[k] = (bev_to_img(fl, Hinv), bev_to_img(fr, Hinv), conf) if fl is not None else None
        for k, net in NETS19.items():
            det = E_BY[ip]['det'] if k == 'UFLD 사전학습' and ip in E_BY else ufld_detail(net, bgr, MAIN16)
            preds[k] = (det[0]['pts'], det[1]['pts'], det)
        out.append(dict(ip=ip, s=bgr.shape[1] / meta[0], Himg=bgr.shape[0], meta=GT3[xp], preds=preds))
    return out


def ex_rng(det, cut, f):
    """범위 [cut, 아래끝] 안 행 앵커 중 자차 좌·우 각각 '존재' 비율의 작은 값"""
    yb, Hi = f['meta'][4], f['Himg']
    ya = np.asarray(C19['row_anchor']) * Hi
    n = int(((ya >= cut * Hi) & (ya <= yb * Hi)).sum())
    if n == 0:
        return 0.0
    return min(sum(cut * Hi <= y <= yb * Hi for _, y in ln['pts']) / n for ln in det)


def rows19(frames, name, cut):
    rs = []
    for f in frames:
        gt = gt_rows(f['meta'], f['s'], cut)
        if gt is None:
            continue
        pr = f['preds'][name]
        if pr is None:
            rs.append(dict(conf=0.0, err=None, f=f)); continue
        lp, rp, c = pr
        rs.append(dict(conf=ex_rng(c, cut, f) if isinstance(c, list) else float(c), err=err_img(lp, rp, gt), f=f, det=c))
    return rs


F19_HOLD, F19_MIS = collect19(HOLD, GT3_HOLD), collect19(MIS, GT3_MIS)
OWN19 = {'P(0.50)': 0.50, 'V(0.47 현재)': 0.47}
SEL19 = {}
print('\n' + '═' * 104)
print('측정 범위별 — 칸 = 가용률/오답. P·V는 자기 임계, UFLD는 검증 오답 0인 가장 낮은 범위-존재 임계(검증에서만 고름)')
print('합격 = 검증 오답 0 · 가용률 ≥ 40% · 틀어짐 오답 0  (틀어짐 오답은 라벨 누락일 수 있어 그림 확인 필요)')
print('═' * 104)
for cut in CUTS19:
    nh_ = len(rows19(F19_HOLD, 'P(0.50)', cut)); nm_ = len(rows19(F19_MIS, 'P(0.50)', cut))
    print(f'\n▶ 판정 구간 화면 높이 {cut:.2f} ~ 아래끝   (검증 {nh_}장 · 틀어짐 {nm_}장)')
    print(f'  {"방법":<16}{"임계":>6}{"검증":>12}{"틀어짐":>12}{"오차중앙":>10}  판정')
    for name in list(OWN19) + list(NETS19):
        rh, rm = rows19(F19_HOLD, name, cut), rows19(F19_MIS, name, cut)
        if name in OWN19:
            th = OWN19[name]
        else:
            ok = [t for t in np.round(np.arange(0.30, 0.96, 0.01), 2) if cnt17(rh, t)[1] == 0 and cnt17(rh, t)[0] > 0]
            th = min(ok) if ok else None
        if th is None:
            print(f'  {name:<16}{"없음":>6}   검증 오답 0인 임계 없음'); continue
        SEL19[(name, cut)] = th
        hp, hb = cnt17(rh, th); mp, mb = cnt17(rm, th)
        e = [r['err'][0] for r in rh if r['err'] is not None and r['conf'] >= th]
        av = hp / max(len(rh), 1)
        verdict = '✅' if hb == 0 and av >= 0.40 and mb == 0 else ''
        print(f'  {name:<16}{th:>6.2f}{f"{av * 100:.0f}%/{hb}":>12}{f"{mp / max(len(rm), 1) * 100:.0f}%/{mb}":>12}'
              f'{(np.median(e) if e else float("nan")):>10.4f}  {verdict}')


# ── ④ 앞차 위치 분포 ──
VEH19 = {'자차·옆 차로': [], '자차 차로': []}
for f in F19_HOLD + F19_MIS:
    W, H, lines, pair, yb, yt = f['meta']; s = f['s']
    fits = []
    for q in pair:
        ys = np.array([t[1] for t in q[3]]) * s; xs = np.array([t[0] for t in q[3]]) * s
        fits.append(np.polyfit(ys, xs, 1) if np.ptp(ys) > 1 else None)
    if fits[0] is None or fits[1] is None:
        continue
    r = det.predict(load(f['ip'], None), imgsz=320, conf=0.25, verbose=False)[0]
    for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
        cx, by = (x1 + x2) / 2, float(y2)
        xl, xr = np.polyval(fits[0], by), np.polyval(fits[1], by)
        w = xr - xl
        if w <= 0:
            continue
        if xl - w <= cx <= xr + w:
            VEH19['자차·옆 차로'].append(by / f['Himg'])
        if xl <= cx <= xr:
            VEH19['자차 차로'].append(by / f['Himg'])
print('\n' + '═' * 104)
print('차량 바퀴 아래쪽 높이 (화면 높이 대비, 검증+틀어짐 사진의 YOLO 검출) — 판정 구간 안에 있어야 차선 기준 위치를 잴 수 있다')
print('═' * 104)
print(f'  {"차량":<12}{"대수":>6}{"p10":>7}{"p50":>7}{"p90":>7}' + ''.join(f'{"≥" + str(c):>8}' for c in CUTS19)
      + f'{"<0.56":>8}')
for k, v in VEH19.items():
    if not v:
        print(f'  {k:<12}{0:>6}'); continue
    v = np.array(v)
    print(f'  {k:<12}{len(v):>6}{np.percentile(v, 10):>7.2f}{np.percentile(v, 50):>7.2f}{np.percentile(v, 90):>7.2f}'
          + ''.join(f'{(v >= c).mean() * 100:>7.0f}%' for c in CUTS19) + f'{(v < 0.56).mean() * 100:>7.0f}%')
print('  (≥c = 그 높이보다 아래(가까이)에 있어 범위 c에서 잴 수 있는 비율. 0.56보다 위는 지금 어느 방법으로도 못 잰다)')


# ── 그림 — 19번 모델 오답 (검증: 범위 0.56·존재 ≥ 0.5 / 틀어짐: 범위 0.64·선택 임계) ──
def show19(r, cut, title):
    f = r['f']; bgr = load(f['ip'], None); vis = bgr.copy()
    W, H, lines, pair, yb, yt = f['meta']; s = f['s']
    for nm, p in lines:
        q = np.int32([(x * s, y * s) for x, y in sorted(p, key=lambda t: t[1])])
        cv2.polylines(vis, [q], False, (200, 200, 200), 2)
    for y, a, b in gt_rows(f['meta'], s, cut):
        cv2.circle(vis, (int(a), int(y)), 9, (255, 255, 0), 2); cv2.circle(vis, (int(b), int(y)), 9, (255, 255, 0), 2)
    for li, ln in enumerate(r['det']):
        for x, y in ln['pts']:
            cv2.circle(vis, (int(x), int(y)), 4, [(0, 0, 255), (0, 255, 0)][li], -1)
    cv2.line(vis, (0, int(cut * f['Himg'])), (vis.shape[1], int(cut * f['Himg'])), (0, 255, 255), 2)
    plt.figure(figsize=(12, 6.5)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title(f'{title} — {f["ip"].name}  최대오차={r["err"][1]:.3f}  conf={r["conf"]:.2f}\n'
              '회색 = 라벨(겹선은 합친 선)  하늘색 원 = 정답 v3  노랑 가로선 = 판정 구간 위쪽 끝  빨강·초록 = 예측', fontsize=9)
    plt.show()

for tag, frames, cut, th in (('검증', F19_HOLD, 0.56, 0.5), ('틀어짐', F19_MIS, 0.64, SEL19.get(('UFLD 19번', 0.64), 0.5))):
    bad = sorted([r for r in rows19(frames, 'UFLD 19번', cut) if r['err'] is not None and r['conf'] >= th
                  and r['err'][1] > MAX_ERR], key=lambda r: -r['err'][1])
    print(f'\n19번 모델 {tag} 오답 (범위 {cut} · conf ≥ {th}) {len(bad)}건 — 최대 {N_SHOW19}건 표시')
    for r in bad[:N_SHOW19]:
        show19(r, cut, f'19번 {tag} 오답')


# ============================== 20번 셀 ==============================
# 20. 걸침 규칙 + 두 결과 일치 검사 — 학습 없이 오답을 걸러 범위 0.60~0.64에서도 오답 0을 지킬 수 있나
# 필요: 19번 셀 실행 (F19_HOLD·F19_MIS·gt_rows·ex_rng·show19·nets15 사용). 학습 없음, GPU 권장(bottom 입력 추론)
#
# 19번 오답 그림 중 일부는 자차가 차선을 걸친(차로 변경 중) 프레임 — 카메라 바로 아래에 차선이 있어
#   '자차 차로'가 하나로 정해지지 않는다. 실제 시스템에서도 이때는 앞차 위치를 잴 기준이 없으므로 보류하는 게 맞다.
#   (첫 실행 결과: 정답 기준 걸침 프레임은 검증의 3%뿐이었고 걸침 규칙은 어느 범위에서도 선택되지 않았다)
# ① 걸침 규칙(예측만 사용 — 실제 시스템에서 그대로 쓸 수 있는 규칙)
#    걸침 여유 = min(화면 가운데 - 자차 왼쪽 선, 자차 오른쪽 선 - 화면 가운데) / 차로 폭  (구간 아래끝 행 기준)
#    이 값이 STR 미만이면 보류. 0.25면 차가 차로 가운데에서 차로 폭의 1/4 이상 치우치면 보류.
#    음수 = 화면 가운데가 예측 차로 밖 → 항상 보류(STR을 켠 경우)
# ② 일치 검사 — 출력은 첫 번째 결과, 두 번째 결과와 차로 중심이 범위 안 모든 행에서 차로 폭의 AGR 이내일 때만 통과
#    '사전학습=19번' : 사전학습 모델과 19번 파인튜닝 모델 / '사전학습 full=bottom' : 같은 모델에 화면 전체·아래쪽만
# 임계 조합(걸침·일치·존재)은 검증에서만 고르고(오답 0 중 가용률 최대), 틀어짐 196장은 판정에만 쓴다.
STR20 = [None, 0.05, 0.10, 0.15, 0.20, 0.25]     # None = 걸침 규칙 끔
AGR20 = [None, 0.03, 0.05, 0.08]                 # None = 일치 검사 끔
TH20  = np.round(np.arange(0.30, 0.96, 0.01), 2)
M20 = {'사전학습': ('UFLD 사전학습', None), '19번': ('UFLD 19번', None),
       '사전학습=19번': ('UFLD 사전학습', 'UFLD 19번'), '사전학습 full=bottom': ('UFLD 사전학습', 'U0b')}
N_SHOW20 = 4

for f in F19_HOLD + F19_MIS:                     # 사전학습 모델 bottom 입력 결과 (일치 검사용)
    if 'U0b' not in f:
        f['U0b'] = ufld_detail(nets15['culane'], load(f['ip'], None), 'bottom')


def x_bottom(pts, y):
    """y에서의 x — 점 범위 밖이면 아래쪽 절반 점으로 직선 연장"""
    x = interp_x(pts, y)
    if x is not None or len(pts) < 2:
        return x
    p = np.array(sorted(pts, key=lambda q: q[1]))
    half = p[len(p) // 2:] if np.ptp(p[len(p) // 2:, 1]) >= 1 else p
    if np.ptp(half[:, 1]) < 1:
        return None
    return float(np.polyval(np.polyfit(half[:, 1], half[:, 0], 1), y))


def straddle(lp, rp, f):
    yb, cx = f['meta'][4] * f['Himg'], f['meta'][0] * f['s'] / 2
    a, b = x_bottom(lp, yb), x_bottom(rp, yb)
    if a is None or b is None or b - a <= 1:
        return -1.0
    return min(cx - a, b - cx) / (b - a)


def agree_rng(p1, p2, f, cut):
    Hi = f['Himg']; d = []
    for y in np.linspace(cut * Hi, f['meta'][4] * Hi, 10):
        v = [interp_x(q, y) for q in (*p1, *p2)]
        if None in v or v[1] - v[0] <= 1:
            continue
        d.append(abs((v[0] + v[1]) / 2 - (v[2] + v[3]) / 2) / (v[1] - v[0]))
    return max(d) if len(d) >= 3 else np.inf


def feats20(frames, out_name, chk, cut):
    R = []
    for f in frames:
        gt = gt_rows(f['meta'], f['s'], cut)
        if gt is None:
            continue
        lp, rp, det = f['preds'][out_name]
        if chk is None:
            ag = 0.0
        else:
            d2 = f['U0b'] if chk == 'U0b' else f['preds'][chk][2]
            ag = agree_rng((lp, rp), (d2[0]['pts'], d2[1]['pts']), f, cut)
        R.append(dict(f=f, det=det, err=err_img(lp, rp, gt), conf=ex_rng(det, cut, f), strad=straddle(lp, rp, f), agr=ag))
    return R


def arr20(R):
    g = np.array([r['err'] is not None and r['err'][1] <= MAX_ERR for r in R], bool)
    b = np.array([r['err'] is not None and r['err'][1] > MAX_ERR for r in R], bool)
    return dict(conf=np.array([r['conf'] for r in R]), strad=np.array([r['strad'] for r in R]),
                agr=np.array([r['agr'] for r in R]), good=g, bad=b, n=len(R))


def mask20(A, th, st, ag):
    m = A['conf'] >= th
    if st is not None:
        m &= A['strad'] >= st
    if ag is not None:
        m &= A['agr'] <= ag
    return m


def gt_strad(f):
    W, H, lines, pair, yb, yt = f['meta']
    xl, xr = pair[0][0], pair[1][0]
    return min(W / 2 - xl, xr - W / 2) / (xr - xl)


for tag, frames in (('검증', F19_HOLD), ('틀어짐', F19_MIS)):
    v = np.array([gt_strad(f) for f in frames])
    print(f'{tag}: 정답 기준 걸침 여유 < 0.10 {(v < 0.10).mean() * 100:.0f}% · < 0.15 {(v < 0.15).mean() * 100:.0f}% · '
          f'< 0.25 {(v < 0.25).mean() * 100:.0f}%  ({len(v)}장)')

BEST20 = {}
print('\n' + '═' * 112)
print('범위별 — 칸 = 가용률/오답. 걸침·일치·존재 임계는 검증에서만 고름(오답 0 중 가용률 최대). 견고 = 검증 상위 10개 조합 중 틀어짐 오답 0인 수')
print('합격 = 검증 오답 0 · 가용률 ≥ 40% · 틀어짐 오답 0  (틀어짐 오답은 라벨 누락일 수 있어 그림 확인)')
print('═' * 112)
for cut in CUTS19:
    print(f'\n▶ 판정 구간 화면 높이 {cut:.2f} ~ 아래끝')
    print(f'  {"방법":<20}{"걸침≥":>7}{"일치≤":>7}{"존재≥":>7}{"검증":>12}{"틀어짐":>12}{"견고":>6}  판정')
    for name, th in (('P(0.50)', 0.50), ('V(0.47 현재)', 0.47)):
        rh, rm = rows19(F19_HOLD, name, cut), rows19(F19_MIS, name, cut)
        hp, hb = cnt17(rh, th); mp, mb = cnt17(rm, th)
        print(f'  {name:<20}{"-":>7}{"-":>7}{th:>7.2f}{f"{hp / max(len(rh), 1) * 100:.0f}%/{hb}":>12}'
              f'{f"{mp / max(len(rm), 1) * 100:.0f}%/{mb}":>12}')
    for name, (out_name, chk) in M20.items():
        RH, RM = feats20(F19_HOLD, out_name, chk, cut), feats20(F19_MIS, out_name, chk, cut)
        AH, AM = arr20(RH), arr20(RM)
        combos = []
        for st in STR20:
            for ag in (AGR20 if chk else [None]):
                for th in TH20:
                    m = mask20(AH, th, st, ag)
                    if (AH['bad'] & m).sum() == 0 and (AH['good'] & m).sum() > 0:
                        combos.append(((AH['good'] & m).sum(), st, ag, th))
        if not combos:
            print(f'  {name:<20}   검증 오답 0인 조합 없음'); continue
        combos.sort(key=lambda t: (-t[0], t[1] is not None, t[2] is not None))
        _, st, ag, th = combos[0]
        mh, mm = mask20(AH, th, st, ag), mask20(AM, th, st, ag)
        hp, mp, mb = (AH['good'] & mh).sum(), ((AM['good'] | AM['bad']) & mm).sum(), (AM['bad'] & mm).sum()
        robust = sum((AM['bad'] & mask20(AM, t, s_, a_)).sum() == 0 for _, s_, a_, t in combos[:10])
        av = hp / AH['n']
        BEST20[(name, cut)] = (st, ag, th, RH, RM, av)
        fmt = lambda v: '끔' if v is None else f'{v:.2f}'
        print(f'  {name:<20}{fmt(st):>7}{fmt(ag):>7}{th:>7.2f}{f"{av * 100:.0f}%/0":>12}'
              f'{f"{mp / max(AM["n"], 1) * 100:.0f}%/{mb}":>12}{f"{robust}/{min(10, len(combos))}":>6}  '
              + ('✅' if av >= 0.40 and mb == 0 else ''))


# ── 그림 A — 검증 오답 0 조합이 있는 가장 넓은 범위에서 가용률 최고 방법: 틀어짐 통과 오답 (라벨 누락인지 눈으로 확인) ──
if BEST20:
    cut_a = min(k[1] for k in BEST20)
    name_a, (st, ag, th, RH, RM, av) = max(((k[0], v) for k, v in BEST20.items() if k[1] == cut_a), key=lambda t: t[1][5])
    AM = arr20(RM)
    bad = [r for r, ok in zip(RM, AM['bad'] & mask20(AM, th, st, ag)) if ok]
    print(f'\n[그림 A] 범위 {cut_a} · {name_a} — 틀어짐 통과 오답 {len(bad)}건 (최대 {N_SHOW20}건 표시)')
    for r in bad[:N_SHOW20]:
        show19(r, cut_a, f'[A] {name_a} 틀어짐 오답 (걸침 여유 {r["strad"]:.2f}, 일치 {min(r["agr"], 9.99):.3f})')
else:
    print('\n[그림 A] 어느 범위에서도 검증 오답 0인 조합이 없어 생략')

# ── 진단 B — 범위 0.60에서 왜 규칙들이 오답을 못 거르나: 사전학습 모델의 맞음/오답별 검사값 + 오답 그림 ──
CUT_DIAG = 0.60
RD = feats20(F19_HOLD, 'UFLD 사전학습', None, CUT_DIAG)
for r in RD:
    f = r['f']; lp, rp, _ = f['preds']['UFLD 사전학습']
    d19 = f['preds']['UFLD 19번'][2]
    r['agr19'] = agree_rng((lp, rp), (d19[0]['pts'], d19[1]['pts']), f, CUT_DIAG)
    r['agrfb'] = agree_rng((lp, rp), (f['U0b'][0]['pts'], f['U0b'][1]['pts']), f, CUT_DIAG)
bad = sorted([r for r in RD if r['conf'] >= 0.5 and r['err'] is not None and r['err'][1] > MAX_ERR], key=lambda r: -r['conf'])
good = [r for r in RD if r['conf'] >= 0.5 and r['err'] is not None and r['err'][1] <= MAX_ERR]
print(f'\n[진단 B] 범위 {CUT_DIAG} · 사전학습 · 존재 ≥ 0.5 — 검증 맞음 {len(good)}건 / 오답 {len(bad)}건의 검사값 (p10 / p50 / p90)')
print('  conf = 범위 안 존재 비율, strad = 걸침 여유, agr19 = 19번 모델과 차이, agrfb = full·bottom 차이 (9.99 = 비교 불가)')
for k in ('conf', 'strad', 'agr19', 'agrfb'):
    q = lambda rs: np.percentile([min(r[k], 9.99) for r in rs], [10, 50, 90]) if rs else [np.nan] * 3
    a, b = q(good), q(bad)
    print(f'  {k:<7} 맞음 {a[0]:>6.3f} / {a[1]:>6.3f} / {a[2]:>6.3f}    오답 {b[0]:>6.3f} / {b[1]:>6.3f} / {b[2]:>6.3f}')
print(f'[그림 B] 존재 비율이 높은 순 검증 오답 — 최대 {N_SHOW20}건 표시')
for r in bad[:N_SHOW20]:
    show19(r, CUT_DIAG, f'[B] 사전학습 검증 오답 (걸침 {r["strad"]:.2f}, 19번 차이 {min(r["agr19"], 9.99):.3f}, '
                        f'f/b 차이 {min(r["agrfb"], 9.99):.3f})')



# ============================== 21번 셀 ==============================
# 21. 합격 기준 재검토 자료 — '큰 오차 0건 + 작은 오차는 일정 비율까지 허용'으로 나누면 결과가 어떻게 되나
# 필요: 20번 셀 실행 (feats20·arr20·mask20·rows19·show19·M20·F19_HOLD·F19_MIS 사용). 학습·추론 없음, CPU로 충분
#
# 지금 기준: 프레임 하나라도, 판정 구간의 어느 한 행이라도 중심 오차가 차로 폭의 10%(MAX_ERR)를 넘으면 오답이고, 오답은 0이어야 한다.
# 그런데 실제 판단은 프레임 하나가 아니라 offset 시계열을 본다 (src/metrics/offset.py, configs/default.yaml).
#   · offset = (차량 x - 차로 중심) / 차로 폭 → 차로 중심 오차 e는 그 프레임 offset을 그대로 e만큼 옮긴다
#   · offset은 최근 5프레임 이동평균(metrics.smoothing_window: 5) → 한 프레임만 틀리면 영향은 약 e/5
#   · 차선 걸침 임계 0.25 · 표류 0.15 · 사행 최소 진폭 0.1 (risk 섹션, 전부 [TBD])
# 그래서 오차를 크기로 나눈다.
#   큰 오차   > 0.25 (GROSS21) : 한 프레임만으로도 걸침 판정을 뒤집을 수 있는 크기 → 0건이어야 함
#   중간 오차 0.15 ~ 0.25      : 여러 프레임 이어지면 표류 판정에 영향
#   작은 오차 0.10 ~ 0.25      : 전체 통과 프레임의 RATE까지 허용 (이어지지 않으면 이동평균으로 희석)
# ⚠️ 이 셀은 판단 자료만 만든다. 기준을 실제로 바꿀지는 팀·지도교수님 동의가 필요하다.
#    결과를 보고 잣대를 옮기는 것으로 보이지 않도록, 허용 비율 후보는 미리 0/1/2/5%로 고정했다.
#    허용 0%는 20번 셀과 같은 기준이다(걸침 규칙 끔) — 20번 결과와 같게 나오는지로 셀이 맞는지 확인할 수 있다.
# 오차는 판정 구간 12개 행 중 최대값이다. 차량이 실제로 있는 한 행만 보면 이보다 작거나 같다(보수적인 값).
GROSS21, MID21 = 0.25, 0.15
RATE21  = [0.00, 0.01, 0.02, 0.05]
RATE_RUN21 = 0.02                       # 연속 오답·그림을 볼 허용 비율
LANE_M21 = 3.5                          # 미터 환산용 차로 폭 (configs lane.lane_width_m과 같은 값)
M21 = {k: M20[k] for k in ('사전학습', '19번', '사전학습=19번')}
N_SHOW21 = 4
print(f'차로 폭 {LANE_M21}m 기준 — 0.10 = {0.10 * LANE_M21:.2f}m · 0.15 = {MID21 * LANE_M21:.2f}m · 0.25 = {GROSS21 * LANE_M21:.2f}m')


def errs21(R, m):
    """통과(m)했고 정답과 비교 가능한 프레임들의 최대 행 오차"""
    return np.array([r['err'][1] for r, ok in zip(R, m) if ok and r['err'] is not None], float)


def stat21(e, n):
    k = len(e)
    return dict(k=k, av=k / max(n, 1), mod=float((e > MAX_ERR).mean()) if k else 0.0,
                n10=int((e > MAX_ERR).sum()), mid=int((e > MID21).sum()), gross=int((e > GROSS21).sum()))


def fmt21(s):
    return f'{s["av"] * 100:>3.0f}% {s["mod"] * 100:>4.1f}% {s["mid"]:>3d} {s["gross"]:>3d}'


def n_gt(R):
    return sum(r['err'] is not None for r in R) or len(R)


def runs21(R, m):
    """같은 시퀀스 안에서 통과 프레임을 순서대로 놓고, 0.10 초과 오답이 연달아 나온 최대 길이·2연속 이상 횟수·프레임 번호 간격"""
    by = {}
    for r, ok in zip(R, m):
        if not ok or r['err'] is None:
            continue
        st = r['f']['ip'].stem
        sq, _, ix = st.rpartition('_')                 # 2_20200916_085235_I1000225 처럼 번호 앞에 글자가 붙은 이름도 있음
        dg = ''.join(ch for ch in ix if ch.isdigit())
        by.setdefault(sq or st, []).append((int(dg) if dg else 0, r['err'][1] > MAX_ERR))
    best, n2, gaps = 0, 0, []
    for v in by.values():
        v.sort(key=lambda t: t[0]); gaps += list(np.diff([i for i, _ in v])); cur = 0
        for _, w in v:
            cur = cur + 1 if w else 0
            best = max(best, cur); n2 += cur == 2
    return best, n2, (float(np.median(gaps)) if gaps else np.nan)


# ── 표 1 — 임계를 고정했을 때 통과 프레임 오차가 어떤 크기로 분포하나 ──
print('\n' + '═' * 100)
print('표 1. 오차 크기 분포 (임계 고정) — 칸 = 가용률 · 0.10 초과 비율 · 0.15 초과 건수 · 0.25 초과 건수')
print('═' * 100)
print(f'  {"범위":<6}{"방법":<22}{"검증 (가용 0.10↑ 중 큰)":>26}{"틀어짐 (가용 0.10↑ 중 큰)":>28}')
for cut in CUTS19:
    for name, th in (('P(0.50)', 0.50), ('V(0.47 현재)', 0.47)):
        out = []
        for frames in (F19_HOLD, F19_MIS):
            rs = rows19(frames, name, cut)
            out.append(stat21(errs21(rs, [r['conf'] >= th for r in rs]), len(rs)))
        print(f'  {cut:<6.2f}{name:<22}{fmt21(out[0]):>26}{fmt21(out[1]):>28}')
    out = []
    for frames in (F19_HOLD, F19_MIS):
        R = feats20(frames, 'UFLD 사전학습', None, cut)
        out.append(stat21(errs21(R, [r['conf'] >= 0.5 for r in R]), len(R)))
    print(f'  {cut:<6.2f}{"UFLD 사전학습 존재≥0.50":<22}{fmt21(out[0]):>26}{fmt21(out[1]):>28}')
    print()

# ── 표 2 — 2단계 기준으로 임계를 다시 고르면 (검증에서만 고름: 큰 오차 0 · 0.10 초과 비율 ≤ 허용 중 가용률 최대) ──
FEAT21 = {}
for cut in CUTS19:
    for name, (out_name, chk) in M21.items():
        RH, RM = feats20(F19_HOLD, out_name, chk, cut), feats20(F19_MIS, out_name, chk, cut)
        FEAT21[(name, cut)] = (RH, RM, arr20(RH), arr20(RM), chk)

SEL21 = {}
print('═' * 112)
print('표 2. 2단계 기준으로 임계를 다시 고른 결과 — 칸 = 가용률 · 0.10 초과 비율 · 0.15 초과 건수 · 0.25 초과 건수')
print('  임계는 검증에서만 고름: 0.25 초과 0건 · 0.10 초과 비율 ≤ 허용 중 가용률 최대. 틀어짐은 판정에만 사용')
print('  판정: ✅ = 검증 가용률 ≥ 40% · 틀어짐 0.25 초과 0건 · 틀어짐 0.10 초과 비율 ≤ 허용   △ = 틀어짐 비율만 넘음')
print('═' * 112)
for cut in CUTS19:
    print(f'\n▶ 판정 구간 화면 높이 {cut:.2f} ~ 아래끝')
    print(f'  {"방법":<16}{"허용":>6}{"일치≤":>7}{"존재≥":>7}{"검증":>22}{"틀어짐":>22}  판정')
    for name in M21:
        RH, RM, AH, AM, chk = FEAT21[(name, cut)]
        for rate in RATE21:
            best = None
            for ag in (AGR20 if chk else [None]):
                for th in TH20:
                    m = mask20(AH, th, None, ag)
                    s = stat21(errs21(RH, m), AH['n'])
                    if s['k'] and s['gross'] == 0 and s['mod'] <= rate + 1e-9:
                        key = (s['k'], ag is None, -th)
                        if best is None or key > best[0]:
                            best = (key, ag, th, s)
            if best is None:
                print(f'  {name:<16}{rate * 100:>5.0f}%   조건을 만족하는 임계 없음'); continue
            _, ag, th, sh = best
            sm = stat21(errs21(RM, mask20(AM, th, None, ag)), AM['n'])
            SEL21[(name, cut, rate)] = (ag, th, sh, sm)
            ok = sh['av'] >= 0.40 and sm['gross'] == 0
            mark = ('✅' if sm['mod'] <= rate + 1e-9 else '△') if ok else ''
            fa = '끔' if ag is None else f'{ag:.2f}'
            print(f'  {name:<16}{rate * 100:>5.0f}%{fa:>7}{th:>7.2f}{fmt21(sh):>22}{fmt21(sm):>22}  {mark}')

# ── 표 3 — 허용 비율 RATE_RUN21에서 고른 임계로, 작은 오차가 한 프레임에 그치나 연달아 나오나 ──
print('\n' + '═' * 100)
print(f'표 3. 연속 오답 (허용 {RATE_RUN21 * 100:.0f}% 임계) — 같은 시퀀스의 통과 프레임을 순서대로 놓고 0.10 초과가 연달아 나온 횟수')
print('  최대 연속 = 가장 길게 이어진 길이, 2연속↑ = 2번 이상 이어진 경우 수, 간격 = 통과 프레임 사이 프레임 번호 차이 중앙값')
print('  1프레임짜리는 이동평균(5프레임)으로 약 1/5로 희석된다. 다만 이 데이터는 영상에서 띄엄띄엄 뽑은 프레임이라 실제 연속성은 영상으로 다시 확인해야 함')
print('═' * 100)
print(f'  {"범위":<6}{"방법":<16}{"검증 최대연속/2연속↑/간격":>28}{"틀어짐 최대연속/2연속↑/간격":>30}')
for cut in CUTS19:
    for name in M21:
        if (name, cut, RATE_RUN21) not in SEL21:
            continue
        ag, th, _, _ = SEL21[(name, cut, RATE_RUN21)]
        RH, RM, AH, AM, _ = FEAT21[(name, cut)]
        c = [runs21(R, mask20(A, th, None, ag)) for R, A in ((RH, AH), (RM, AM))]
        print(f'  {cut:<6.2f}{name:<16}{f"{c[0][0]} / {c[0][1]} / {c[0][2]:.0f}":>28}{f"{c[1][0]} / {c[1][1]} / {c[1][2]:.0f}":>30}')

# ── 그림 C — 사전학습(존재 ≥ 0.50) 통과 프레임의 오차 누적 분포 ──
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, (tag, frames) in zip(axs, (('검증', F19_HOLD), ('틀어짐', F19_MIS))):
    for cut in CUTS19:
        R = feats20(frames, 'UFLD 사전학습', None, cut)
        e = np.sort(errs21(R, [r['conf'] >= 0.5 for r in R]))
        if len(e):
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), label=f'범위 {cut:.2f} ({len(e)}장)')
    for x, c, lb in ((MAX_ERR, 'g', '0.10 지금 기준'), (MID21, 'orange', '0.15 표류'), (GROSS21, 'r', '0.25 걸침')):
        ax.axvline(x, color=c, ls='--', lw=1, label=lb)
    ax.set_xlim(0, 0.5); ax.set_ylim(0.8, 1.001); ax.grid(alpha=0.3)
    ax.set_xlabel('최대 행 오차 (차로 폭 대비)'); ax.set_ylabel('누적 비율 (0.8 이상만 표시)')
    ax.set_title(f'[C] UFLD 사전학습 존재≥0.50 — {tag}'); ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.show()

# ── 그림 D — 허용 RATE_RUN21 기준에서 가장 넓은 범위의 사전학습: 0.15 초과 통과 프레임 (판단을 흔들 수 있는 것) ──
cands_d = [c for c in CUTS19 if ('사전학습', c, RATE_RUN21) in SEL21]
if cands_d:
    cut_d = min(cands_d)
    ag, th, _, _ = SEL21[('사전학습', cut_d, RATE_RUN21)]
    RH, RM, AH, AM, _ = FEAT21[('사전학습', cut_d)]
    big = []
    for tag, R, A in (('검증', RH, AH), ('틀어짐', RM, AM)):
        big += [(tag, r) for r, ok in zip(R, mask20(A, th, None, ag)) if ok and r['err'] is not None and r['err'][1] > MID21]
    big.sort(key=lambda t: -t[1]['err'][1])
    print(f'\n[그림 D] 범위 {cut_d} · 사전학습 · 존재≥{th:.2f} — 0.15 초과 통과 {len(big)}건 (최대 {N_SHOW21}건 표시)')
    for tag, r in big[:N_SHOW21]:
        show19(r, cut_d, f'[D] 사전학습 {tag} 0.15 초과')
else:
    print(f'\n[그림 D] 허용 {RATE_RUN21 * 100:.0f}%에서 사전학습 조건을 만족하는 범위가 없어 생략')



# ============================== 22번 셀 ==============================
# 22. 차로 폭 검사 — 모델이 가까운 선을 건너뛰고 '한 칸 바깥 선'을 잡은 큰 오답을 정답 없이 거를 수 있나
# 필요: 21번 셀 실행 (FEAT21·stat21·fmt21·M20·AGR20·TH20·x_bottom·show19 사용). 학습·추론 없음, CPU로 충분
#
# 21번 그림 D의 큰 오답(0.25 초과)은 전부 같은 유형이었다 — 노란 중앙선이나 가까운 점선을 놓치고 바깥 선을 잡아
#   예측 차로 폭이 평소의 1.5~2배(차로 두 개 폭, 또는 도로 전체 폭)가 된다.
# 그래서 예측 차로 폭이 '평소 폭'에서 크게 벗어나면 보류한다. 정답은 쓰지 않으므로 실제 시스템에 그대로 넣을 수 있다.
#   폭 = 자차 왼쪽·오른쪽 선 사이 거리 / 화면 폭, 판정 구간 아래끝 행과 위끝 행 두 곳에서 잼
#   평소 폭(기준 폭) 두 가지
#     카메라 : 그 카메라 전체 프레임에서 모델이 예측한 폭의 중앙값 — 설치할 때 한 번 재 두는 값에 해당
#              (실제 시스템이면 보정 행렬 + 차로 폭 3.5m로 계산 가능)
#     시퀀스 : 같은 주행(시퀀스)의 다른 프레임 예측 폭 중앙값 — 주행 중 최근 폭의 중앙값에 해당
#              (다른 프레임 4장 미만이면 카메라 기준 사용. 한 시퀀스 내내 틀리면 기준도 같이 틀리는 약점이 있음)
#   폭± T : 두 행 중 하나라도 폭이 기준의 (1+T)배 초과 또는 1/(1+T)배 미만이면 보류
#     T 후보는 결과를 보기 전에 정함 — 국내 차로 폭은 3.0~3.6m라 자연스러운 차이는 ±10~20%,
#     한 칸 건너뛴 오류는 폭이 약 2배(+100%)라 0.40으로도 걸러져야 한다. 0.15 / 0.25 / 0.40
# 임계(일치·존재)는 21번처럼 검증에서만 고르고, 틀어짐은 판정에만 쓴다. 기준은 두 가지를 나란히 보여 준다.
#   엄격  = 0.10 초과 0건 (지금 기준)
#   2단계 = 0.25 초과 0건 + 0.10 초과 비율 ≤ 2% (21번에서 본 안 — 채택 여부는 팀 결정)
T22    = [None, 0.15, 0.25, 0.40]
REF22  = ['카메라', '시퀀스']
M22    = {k: M20[k] for k in ('사전학습', '사전학습=19번')}
RATE22 = 0.02
N_SHOW22 = 4


def seq22(f):
    sq, _, _ = f['ip'].stem.rpartition('_')
    return sq or f['ip'].stem


def widths22(f, lp, rp, cut):
    """[아래끝 행 폭, 위끝 행 폭] / 화면 폭 — 못 재면 nan"""
    Wi, Hi, out = f['meta'][0] * f['s'], f['Himg'], []
    for y in (f['meta'][4] * Hi, cut * Hi):
        a, b = x_bottom(lp, y), x_bottom(rp, y)
        out.append((b - a) / Wi if a is not None and b is not None and b - a > 1 else np.nan)
    return out


def add_width22(R, out_name, cut):
    """각 프레임에 기준 폭 대비 차이(로그 비율의 두 행 최대값)를 붙인다. 못 재면 inf(검사 켜면 보류)"""
    Wd = np.array([widths22(r['f'], *r['f']['preds'][out_name][:2], cut) for r in R], float).reshape(-1, 2)
    okw = ~np.isnan(Wd).any(1)
    cam = np.median(Wd[okw], 0) if okw.any() else np.array([np.nan, np.nan])
    seqs = np.array([seq22(r['f']) for r in R])
    for i, r in enumerate(R):
        same = (seqs == seqs[i]) & okw; same[i] = False
        ref = {'카메라': cam, '시퀀스': np.median(Wd[same], 0) if same.sum() >= 4 else cam}
        r['w22'] = Wd[i]
        r['dev22'] = {k: float(np.max(np.abs(np.log(Wd[i] / v)))) if okw[i] and not np.isnan(v).any() else np.inf
                      for k, v in ref.items()}


def E22(R):
    return np.array([r['err'][1] if r['err'] is not None else np.nan for r in R], float)


def stat22(E, m, n):
    return stat21(E[m & ~np.isnan(E)], n)


def mask22(A, D, th, ag, T):
    m = mask20(A, th, None, ag)
    return m if T is None else m & (D <= np.log1p(T))


for cut in CUTS19:
    for name in M22:
        RH, RM, _, _, _ = FEAT21[(name, cut)]
        add_width22(RH, M22[name][0], cut); add_width22(RM, M22[name][0], cut)

# ── 표 1 — 폭 차이가 맞음/작은 오차/큰 오차를 갈라 주나 (사전학습 · 존재 ≥ 0.50 · 카메라 기준) ──
print('═' * 104)
print('표 1. 기준 폭 대비 폭 차이 (사전학습 · 존재≥0.50 · 카메라 기준) — p10 / p50 / p90, 차이 = 기준의 몇 % 벗어났나')
print('  맞음 = 0.10 이하, 작은 = 0.10~0.25, 큰 = 0.25 초과.  걸림 = 큰 오차 중 폭± 0.25 / 0.40에서 보류되는 수')
print('═' * 104)
for cut in CUTS19:
    for tag, R in (('검증', FEAT21[('사전학습', cut)][0]), ('틀어짐', FEAT21[('사전학습', cut)][1])):
        rs = [r for r in R if r['conf'] >= 0.5 and r['err'] is not None]
        line = f'  {cut:.2f} {tag:<5}'
        for lb, sel in (('맞음', lambda e: e <= MAX_ERR), ('작은', lambda e: MAX_ERR < e <= GROSS21), ('큰', lambda e: e > GROSS21)):
            d = [np.expm1(min(r['dev22']['카메라'], 3.0)) for r in rs if sel(r['err'][1])]
            line += (f'  {lb} {len(d):>3}장 ' + ('/'.join(f'{v * 100:>3.0f}%' for v in np.percentile(d, [10, 50, 90])) if d else '   -    ')).ljust(29)
        big = [r['dev22']['카메라'] for r in rs if r['err'][1] > GROSS21]
        line += f'  걸림 {sum(v > np.log1p(0.25) for v in big)}/{sum(v > np.log1p(0.40) for v in big)}'
        print(line)

# ── 표 2 — 폭 검사를 켜고 임계를 검증에서 다시 고르면 ──
print('\n' + '═' * 116)
print('표 2. 폭 검사 + 임계 재선택 — 칸 = 가용률 · 0.10 초과 비율 · 0.15 초과 건수 · 0.25 초과 건수')
print('  엄격 = 검증 0.10 초과 0건 중 가용률 최대 / 2단계 = 검증 0.25 초과 0건 · 0.10 초과 ≤ 2% 중 가용률 최대 (임계는 검증에서만)')
print('  판정: 엄격 ✅ = 검증 가용률 ≥ 40% · 틀어짐 0.10 초과 0건   2단계 ✅ = 가용률 ≥ 40% · 틀어짐 0.25 초과 0건 · 0.10 초과 ≤ 2%  △ = 비율만 넘음')
print('═' * 116)
RES22 = {}
for cut in CUTS19:
    print(f'\n▶ 판정 구간 화면 높이 {cut:.2f} ~ 아래끝')
    print(f'  {"방법":<14}{"기준폭":<6}{"폭±":>5}{"엄격 검증":>18}{"엄격 틀어짐":>18}{"2단계 검증":>18}{"2단계 틀어짐":>18}  판정')
    for name, (_, chk) in M22.items():
        RH, RM, AH, AM, _ = FEAT21[(name, cut)]
        EH, EM = E22(RH), E22(RM)
        for ref in REF22:
            for T in T22:
                if T is None and ref != REF22[0]:
                    continue                                     # 폭 검사 끔은 한 번만
                DH = np.array([r['dev22'][ref] for r in RH]); DM = np.array([r['dev22'][ref] for r in RM])
                cells, marks = [], []
                for crit in ('엄격', '2단계'):
                    best = None
                    for ag in (AGR20 if chk else [None]):
                        for th in TH20:
                            s = stat22(EH, mask22(AH, DH, th, ag, T), AH['n'])
                            ok = s['n10'] == 0 if crit == '엄격' else (s['gross'] == 0 and s['mod'] <= RATE22 + 1e-9)
                            if s['k'] and ok:
                                key = (s['k'], ag is None, -th)
                                if best is None or key > best[0]:
                                    best = (key, ag, th, s)
                    if best is None:
                        cells += ['없음', '']; marks.append(''); continue
                    _, ag, th, sh = best
                    mM = mask22(AM, DM, th, ag, T)
                    sm = stat22(EM, mM, AM['n'])
                    RES22[(name, cut, ref, T, crit)] = (ag, th, sh, sm, mask22(AH, DH, th, ag, T), mM)
                    cells += [fmt21(sh), fmt21(sm)]
                    if sh['av'] < 0.40:
                        marks.append('')
                    elif crit == '엄격':
                        marks.append('엄격✅' if sm['n10'] == 0 else '')
                    else:
                        marks.append(('2단계✅' if sm['mod'] <= RATE22 + 1e-9 else '2단계△') if sm['gross'] == 0 else '')
                lbT = '끔' if T is None else f'{T:.2f}'
                lbR = '-' if T is None else ref
                print(f'  {name:<14}{lbR:<6}{lbT:>5}' + ''.join(f'{c:>18}' for c in cells) + '  ' + ' '.join(m for m in marks if m))

# ── 그림 E — 폭 차이 vs 실제 오차 (사전학습 · 존재 ≥ 0.50 · 카메라 기준 · 범위 0.60) ──
CUT_E = 0.60
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (tag, R) in zip(axs, (('검증', FEAT21[('사전학습', CUT_E)][0]), ('틀어짐', FEAT21[('사전학습', CUT_E)][1]))):
    rs = [r for r in R if r['conf'] >= 0.5 and r['err'] is not None]
    x = np.array([np.expm1(min(r['dev22']['카메라'], np.log1p(2.0))) for r in rs])
    y = np.array([min(r['err'][1], 1.0) for r in rs])
    ax.scatter(x, y, s=10, c=np.where(y > GROSS21, 'r', np.where(y > MAX_ERR, 'orange', 'tab:blue')), alpha=0.6)
    for T in T22[1:]:
        ax.axvline(T, color='gray', ls=':', lw=1)
        ax.text(T, 0.97, f' 폭±{T:.2f}', fontsize=8, color='gray')
    ax.axhline(MAX_ERR, color='g', ls='--', lw=1); ax.axhline(GROSS21, color='r', ls='--', lw=1)
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)
    ax.set_xlabel('기준 폭 대비 차이 (1.0 = 100%, 2.0 이상은 끝에 모음)'); ax.set_ylabel('최대 행 오차 (1.0 이상은 끝에 모음)')
    ax.set_title(f'[E] 범위 {CUT_E} · 사전학습 존재≥0.50 · 카메라 기준 — {tag} ({len(rs)}장)')
plt.tight_layout(); plt.show()
print('그림 E: 파랑 = 맞음, 주황 = 0.10~0.25, 빨강 = 0.25 초과. 빨강이 오른쪽(폭 차이 큼)에 몰려 있으면 폭 검사로 걸러진다는 뜻')

# ── 그림 F — 2단계 · 카메라 기준 · 폭± 0.25로도 남는 틀어짐 큰 오차, 그리고 폭 검사 때문에 보류된 맞는 프레임 ──
key_f = [k for k in RES22 if k[0] == '사전학습' and k[2] == '카메라' and k[3] == 0.25 and k[4] == '2단계']
if key_f:
    k = min(key_f, key=lambda k: k[1]); cut_f = k[1]
    ag, th, sh, sm, mH, mM = RES22[k]
    RH, RM = FEAT21[('사전학습', cut_f)][:2]
    left = sorted([r for r, ok in zip(RM, mM) if ok and r['err'] is not None and r['err'][1] > GROSS21], key=lambda r: -r['err'][1])
    print(f'\n[그림 F-1] 범위 {cut_f} · 사전학습 · 카메라 기준 폭±0.25 · 존재≥{th:.2f} — 틀어짐에 남은 0.25 초과 {len(left)}건 (최대 {N_SHOW22}건)')
    for r in left[:N_SHOW22]:
        show19(r, cut_f, f'[F-1] 남은 큰 오차 (폭 차이 {np.expm1(min(r["dev22"]["카메라"], 5)) * 100:.0f}%)')
    m0 = mask20(FEAT21[('사전학습', cut_f)][2], th, None, ag)
    lost = [r for r, a, b in zip(RH, m0, mH) if a and not b and r['err'] is not None and r['err'][1] <= MAX_ERR]
    print(f'[그림 F-2] 같은 설정에서 폭 검사 때문에 보류된 검증의 맞는 프레임 {len(lost)}건 (최대 2건)')
    for r in lost[:2]:
        show19(r, cut_f, f'[F-2] 보류된 맞는 프레임 (폭 차이 {np.expm1(min(r["dev22"]["카메라"], 5)) * 100:.0f}%)')
else:
    print('\n[그림 F] 2단계 · 카메라 기준 · 폭±0.25 조건을 만족하는 범위가 없어 생략')



# ============================== 23번 셀 ==============================
# 23. 저장소 실행 코드 대조 — feature/lane-ufld 브랜치의 src/lane(ufld 경로)이 실제 가중치로 노트북과 같은 결과를 내나
# 필요: 22번 셀까지 실행 (F19_HOLD·F19_MIS·gt_rows·err_img·feats20·show19·load·UFLD_DIR 사용). CPU로 충분(ONNX Runtime CPU)
#
# 저장소 코드는 가짜 출력·무작위 가중치로만 테스트했다. 여기서 실제 CULane 가중치로 세 가지를 확인한다.
#   A. 출력 해석이 같은가 — 같은 사진에서 저장소 코드(ONNX)와 노트북(PyTorch)의 자차 차선 점을 비교
#   B. 판정이 같은가 — 측정 구간 0.68~0.98에서
#        노트북  : 존재 비율 ≥ 0.68만 (21번 셀 엄격 기준 결과와 같은 조건)
#        코드①  : 저장소 안전장치 중 존재 비율만 켬 → 노트북과 거의 같아야 함 (차이 = 구간 아래끝 0.98 고정 vs 라벨별 아래끝)
#        코드②  : 저장소 기본 안전장치 전부 (존재·형상·걸침·차로 폭)
#   C. 처리 시간 — 코랩 CPU 기준 (Pi5와 다름, 참고용)
# ⚠️ 차로 폭 검사는 '과거 프레임만' 쓴다. 이 데이터는 1초 간격 사진이라 설정(10초 창, 15장 이상)으로는 기록이 거의
#    안 쌓이므로, 여기서는 창 60초·최소 5장으로 바꿔 돌린다(WIDTH23). 실제 영상(15fps)에서는 기본값이 맞다.
import os, copy, subprocess, sys, time

REPO23   = Path('/content/SafeWatch_AI')
BRANCH23 = 'feature/lane-ufld'
ONNX23   = UFLD_DIR / 'ufld_culane_res18_rows.onnx'   # 드라이브에 저장 — 다음엔 변환 생략
REEXPORT23 = False
WIDTH23  = dict(window_sec=60.0, min_samples=5)        # 1초 간격 사진용 (위 ⚠️)
N_SHOW23 = 4

# ── 저장소 받기 (브랜치 최신으로 맞춤) ──
if not REPO23.exists():
    subprocess.run(['git', 'clone', '-q', '-b', BRANCH23,
                    'https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git', str(REPO23)], check=True)
else:
    for cmd in (['fetch', '-q', 'origin', BRANCH23], ['checkout', '-q', BRANCH23], ['reset', '-q', '--hard', f'origin/{BRANCH23}']):
        subprocess.run(['git', '-C', str(REPO23), *cmd], check=True)
print('저장소:', subprocess.run(['git', '-C', str(REPO23), 'log', '-1', '--oneline'], capture_output=True, text=True).stdout.strip())
try:
    import onnxruntime, onnx  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'onnxruntime', 'onnx'], check=True)
for m in [m for m in sys.modules if m in ('src', 'scripts') or m.startswith(('src.', 'scripts.'))]:
    del sys.modules[m]                                   # 다시 실행할 때 새 코드를 읽게
if str(REPO23) not in sys.path:
    sys.path.insert(0, str(REPO23))

from src.utils.config import Config, load_config
from src.lane.ufld import UFLDSpec, OnnxUFLDBackend, preprocess, decode_rows
from src.lane.ufld_guard import LaneGuard
from src.utils.profiler import profiler as profiler23

CFG23  = load_config(REPO23 / 'configs/default.yaml')
SPEC23 = UFLDSpec.from_config(CFG23.lane.ufld)
TOP23, BOT23 = CFG23.lane.ufld.measure_range_ratio
MIN_EX23 = CFG23.lane.ufld.guard.min_existence
print(f'설정: 측정 구간 {TOP23}~{BOT23}, 존재 비율 ≥ {MIN_EX23}, 폭 ±{CFG23.lane.ufld.guard.width.tolerance}')

# ── 가중치 → ONNX (저장소 scripts/export_ufld_onnx.py 그대로) ──
if REEXPORT23 or not ONNX23.exists():
    from scripts.export_ufld_onnx import UFLDv2RowNet, load_weights, export, verify
    net23 = UFLDv2RowNet(SPEC23, grid_col=100, num_col=81, fc_norm=True)
    load_weights(net23, UFLD_DIR / 'culane_res18.pth')
    net23.keep_rows_only()
    export(net23, ONNX23)
    print(f'ONNX 변환 {ONNX23.stat().st_size / 1e6:.0f}MB — PyTorch와 최대 차이 {verify(net23, ONNX23):.2e}')
    del net23
BACK23 = OnnxUFLDBackend(ONNX23, num_threads=os.cpu_count() or 0)


def guard23(**kw):
    """저장소 기본 안전장치 설정에서 일부만 바꾼 LaneGuard. kw: existence_only=True / width=dict(...)"""
    d = copy.deepcopy(CFG23.to_dict()['lane']['ufld']['guard'])
    d['width'].update(WIDTH23)
    if kw.get('existence_only'):
        d['min_straddle_margin'] = -1e9
        d['width']['enabled'] = False
        d['temporal']['enabled'] = False
    return LaneGuard(Config(d), SPEC23.row_anchor, (TOP23, BOT23))


def order23(f):
    st = f['ip'].stem
    sq, _, ix = st.rpartition('_')
    dg = ''.join(ch for ch in ix if ch.isdigit())
    return (sq or st, int(dg) if dg else 0)


# ── 프레임마다 한 번 추론 → 해석 결과 캐시 ──
profiler23.reset()
OUT23 = {}
for f in F19_HOLD + F19_MIS:
    key = str(f['ip'])
    if key in OUT23:
        continue
    bgr = load(f['ip'], None)
    H, W = bgr.shape[:2]
    with profiler23.stage('lane'):
        with profiler23.stage('preprocess'):
            x = preprocess(bgr, SPEC23)
        with profiler23.stage('lane_infer'):
            loc, ex = BACK23.infer(x)
        with profiler23.stage('decode'):
            lanes = decode_rows(loc, ex, SPEC23, W, H)
    OUT23[key] = (lanes[1], lanes[2], W, H)

# ── A. 출력 해석 대조 ──
print('\n' + '═' * 96)
print('A. 같은 사진, 저장소 코드(ONNX) vs 노트북(PyTorch) 자차 차선 점')
print('═' * 96)
for tag, frames in (('검증', F19_HOLD), ('틀어짐', F19_MIS)):
    dx, row_mismatch, n = [], 0, 0
    for f in frames:
        L, R, W, H = OUT23[str(f['ip'])]
        nb = f['preds']['UFLD 사전학습']
        for mine, theirs in ((L, nb[0]), (R, nb[1])):
            a = {round(y, 2): x for x, y in mine}
            b = {round(float(y), 2): float(x) for x, y in theirs}
            row_mismatch += len(set(a) ^ set(b))
            dx += [abs(a[y] - b[y]) for y in set(a) & set(b)]
        n += 1
    dx = np.array(dx) if dx else np.array([0.0])
    print(f'  {tag:<4} {n}장 — 같은 행 x 차이 중앙값 {np.median(dx):.3f}px · 최대 {dx.max():.2f}px · '
          f'존재 판정이 갈린 행 {row_mismatch}개 (차선 2개 × 72행 × {n}장 중)')
print('  기대: x 차이 1px 미만, 갈린 행은 존재 확률이 0.5에 아주 가까운 행 몇 개뿐 (GPU↔CPU 연산 차이)')

# ── B. 판정 대조 ──
RES23 = {}
for tag, frames in (('검증', F19_HOLD), ('틀어짐', F19_MIS)):
    rows = []
    for f in frames:
        gt = gt_rows(f['meta'], f['s'], TOP23)
        if gt is not None:
            rows.append(f)
    rows.sort(key=order23)
    nb_r = feats20(rows, 'UFLD 사전학습', None, TOP23)        # rows와 같은 순서·같은 장수 (정답 있는 프레임만 넣었으므로)
    assert len(nb_r) == len(rows)
    out = {'노트북 (존재만)': [], '코드① (존재만)': [], '코드② (기본 전부)': []}
    g1, g2, cur = None, None, None
    for f, r_nb in zip(rows, nb_r):
        sq, ix = order23(f)
        if sq != cur:                                    # 시퀀스가 바뀌면 기록 초기화 (실제로는 주행 시작)
            g1, g2, cur = guard23(existence_only=True), guard23(), sq
        L, R, W, H = OUT23[str(f['ip'])]
        ts = ix / 30.0
        gt = gt_rows(f['meta'], f['s'], TOP23)
        e = err_img(L, R, gt)
        out['노트북 (존재만)'].append((r_nb['conf'] >= MIN_EX23, None if r_nb['conf'] >= MIN_EX23 else 'existence', r_nb['err'], f, None))
        for name, g in (('코드① (존재만)', g1), ('코드② (기본 전부)', g2)):
            res = g.check(L, R, W, H, ts)
            out[name].append((res.valid, res.reason, e, f, res))
    RES23[tag] = out

print('\n' + '═' * 110)
print(f'B. 판정 대조 — 측정 구간 {TOP23}~{BOT23}. 칸 = 통과율 · 통과 중 0.10 초과 / 0.15 초과 / 0.25 초과 건수')
print('   노트북 21번 셀 같은 조건(사전학습·범위 0.68·존재 0.68): 검증 91%·0건 / 틀어짐 97%·0.10 초과 5.6%(0.15 초과 8건, 0.25 초과 6건)')
print('═' * 110)
for tag, out in RES23.items():
    print(f'\n▶ {tag}')
    for name, rs in out.items():
        ok = [r for r in rs if r[0] and r[2] is not None]
        e = np.array([r[2][1] for r in ok]) if ok else np.array([])
        reasons = {}
        for r in rs:
            if not r[0]:
                reasons[r[1]] = reasons.get(r[1], 0) + 1
        rs_txt = ', '.join(f'{k} {v}' for k, v in sorted(reasons.items(), key=lambda t: -t[1]))
        print(f'  {name:<16} {len(ok) / max(len(rs), 1) * 100:>4.0f}% ({len(ok)}/{len(rs)}) · '
              f'{(e > MAX_ERR).sum():>2} / {(e > 0.15).sum():>2} / {(e > 0.25).sum():>2}   보류: {rs_txt}')
        n_cal = reasons.get('width_calibrating', 0)
        if n_cal:
            print(f'  {"":<16} 보정 중 {n_cal}장을 빼면 통과율 {len(ok) / max(len(rs) - n_cal, 1) * 100:.0f}% '
                  f'(시퀀스 {len({order23(r[3])[0] for r in rs})}개)')
print('\n  · 코드①이 노트북과 몇 장 이내로 같으면 해석·존재 비율 계산이 맞다는 뜻')
print('  · 코드②의 width_calibrating = 시퀀스 앞부분 기록이 쌓이기 전 보류(1초 간격 사진이라 실제보다 많이 나옴)')

# ── 그림 — 코드② 통과인데 0.10 초과 (최대 N_SHOW23장) ──
bad23 = [(tag, r) for tag, out in RES23.items() for r in out['코드② (기본 전부)']
         if r[0] and r[2] is not None and r[2][1] > MAX_ERR]
bad23.sort(key=lambda t: -t[1][2][1])
print(f'\n[그림] 코드② 통과 중 0.10 초과 {len(bad23)}건 (최대 {N_SHOW23}건)')
for tag, (_, _, e, f, res) in bad23[:N_SHOW23]:
    L, R, _, _ = OUT23[str(f['ip'])]
    show19(dict(f=f, det=[dict(pts=list(map(tuple, L))), dict(pts=list(map(tuple, R)))], err=e, conf=res.confidence),
           TOP23, f'[23] 저장소 코드 {tag} 통과 오답')

# ── C. 처리 시간 (코랩 CPU) ──
print('\nC. 처리 시간 평균(ms, 코랩 CPU — Pi5와 다름, 참고용):',
      {k: round(v, 1) for k, v in profiler23.summary().items()})


# ============================== 24번 셀 ==============================
# 24. 차로 폭 기준 오염 진단 — 23번 그림 1(085235_I1000225, 오차 0.68)은 폭이 2배인데 왜 폭 검사를 통과했나
# 필요: 23번 셀까지 실행 (OUT23·order23·guard23·RES23·CFG23·TOP23·BOT23·gt_rows·err_img 사용). CPU로 충분, 추론 다시 안 함
#
# 가설: 저장소 폭 검사는 '지난 프레임'의 폭 중앙값을 기준으로 쓴다(22번 셀은 시퀀스 전체 중앙값).
#       주행 앞부분부터 모델이 한 칸 바깥 선을 잡으면 그 틀린 폭이 기준이 되어
#       ① 틀린 프레임은 통과시키고 ② 맞는 프레임은 'width'로 보류한다.
# 확인할 것
#   1. 시퀀스별: 건너뜀 프레임(예측 폭 / 정답 폭 > 1.4) 수, 그중 통과 수, 기준이 오염된 프레임 수(기준 폭 / 정답 폭이 허용 밖)
#   2. 23번 그림에 나온 시퀀스의 프레임별 시간표 + 그래프 — 기준 폭이 언제 어떻게 틀어지나
#   3. 대책 비교 — 설치 때 카메라별 폭을 한 번 재 둔다고 가정(여기서는 정답 폭 중앙값으로 흉내 → 실제보다 유리함)
#        코드②   : 지금 저장소 (지난 프레임 중앙값만)
#        고정     : 고정 폭 ± 허용치만
#        보정안   : 고정 폭 ± 허용치 안에 든 폭만 기록, 기준 = 그 기록의 중앙값(모자라면 고정 폭) — 오염이 원천 차단됨
#   4. 고정 폭이 현실적인가 — 날짜(카메라 추정)별 정답 폭의 흩어짐. 허용치 ±25% 안에 대부분 들어야 고정 기준을 쓸 수 있음
from src.lane.geometry import lane_at_row

SKIP24 = 1.4                                                 # 예측 폭 / 정답 폭이 이보다 크면 '한 칸 건너뜀'으로 본다
TOL24 = CFG23.lane.ufld.guard.width.tolerance
LTOL24 = np.log1p(TOL24)
SHOW_SEQ24 = 3                                               # 시간표·그래프를 볼 시퀀스 수 (23번 그림 오답이 많은 순)


def cam24(sq):
    return '_'.join(sq.split('_')[:2])                       # '2_20200916_085235' → '2_20200916' (날짜 = 카메라로 추정)


def widths24(L, R, W, H):
    """[측정 구간 아래끝 행 폭, 위끝 행 폭] / 화면 폭 — 저장소 LaneGuard와 같은 두 행. 못 재면 None"""
    out = []
    for yr in (BOT23, TOP23):
        lr = lane_at_row(L, R, yr * H)
        if lr is None:
            return None
        out.append((lr[1] - lr[0]) / W)
    return np.array(out)


def gt_pts24(f):
    gt = gt_rows(f['meta'], f['s'], TOP23)
    return (gt, None, None) if gt is None else (gt, gt[:, [1, 0]], gt[:, [2, 0]])


def ldev24(a, b):
    return float(np.max(np.abs(np.log(a / b))))


# ── 프레임마다: 23번 코드② 순서 그대로 다시 돌리면서, 판정 직전의 기준 폭을 기록 ──
D24 = {}
for tag, frames in (('검증', F19_HOLD), ('틀어짐', F19_MIS)):
    rows = sorted([f for f in frames if gt_rows(f['meta'], f['s'], TOP23) is not None], key=order23)
    recs, g, cur = [], None, None
    for f in rows:
        sq, ix = order23(f)
        if sq != cur:
            g, cur = guard23(), sq
        L, R, W, H = OUT23[str(f['ip'])]
        ts = ix / 30.0
        hist = [(b, t) for t0, b, t in g._widths if t0 >= ts - g._width_window_sec]   # check()가 볼 기록과 같음
        ref = np.median(np.array(hist), 0) if len(hist) >= g._width_min_samples else None
        res = g.check(L, R, W, H, ts)
        gt, gl, gr = gt_pts24(f)
        gw = widths24(gl, gr, W, H)
        pw = widths24(L, R, W, H)
        recs.append(dict(f=f, sq=sq, cam=cam24(sq), ix=ix, ts=ts, reason=res.reason, valid=res.valid,
                         err=err_img(L, R, gt), pw=pw, gw=gw, ref=ref, nhist=len(hist)))
    for r in recs:
        r['skip'] = r['pw'] is not None and r['gw'] is not None and bool((r['pw'] / r['gw'] > SKIP24).any())
        r['bad_ref'] = r['ref'] is not None and r['gw'] is not None and ldev24(r['ref'], r['gw']) > LTOL24
        r['past_straddle'] = r['reason'] in (None, 'width', 'width_calibrating')   # 폭 검사까지 왔나
    D24[tag] = recs

# ── 1. 시퀀스별 요약 ──
print('═' * 112)
print(f'1. 시퀀스별 — 건너뜀 = 예측 폭/정답 폭 > {SKIP24} · 기준 오염 = 판정 직전 기준 폭이 정답 폭에서 ±{TOL24:.0%} 밖')
print('═' * 112)
for tag, recs in D24.items():
    print(f'\n▶ {tag}')
    print(f'  {"시퀀스":<22}{"장수":>5}{"건너뜀":>7}{"└통과":>6}{"└보류(width)":>13}{"기준 오염":>9}{"맞는데 width 보류":>17}{"0.10 초과 통과":>15}')
    tot = Counter()
    for sq in sorted({r['sq'] for r in recs}):
        rs = [r for r in recs if r['sq'] == sq]
        c = Counter(n=len(rs),
                    skip=sum(r['skip'] for r in rs),
                    skip_pass=sum(r['skip'] and r['valid'] for r in rs),
                    skip_held=sum(r['skip'] and r['reason'] == 'width' for r in rs),
                    bad_ref=sum(r['bad_ref'] for r in rs),
                    good_held=sum((not r['skip']) and r['reason'] == 'width' for r in rs),
                    wrong_pass=sum(r['valid'] and r['err'] is not None and r['err'][1] > MAX_ERR for r in rs))
        tot.update(c)
        if c['skip'] or c['bad_ref'] or c['good_held'] or c['wrong_pass']:
            print(f'  {sq:<22}{c["n"]:>5}{c["skip"]:>7}{c["skip_pass"]:>6}{c["skip_held"]:>13}{c["bad_ref"]:>9}'
                  f'{c["good_held"]:>17}{c["wrong_pass"]:>15}')
    print(f'  {"합계 (" + str(len({r["sq"] for r in recs})) + "개 시퀀스)":<22}{tot["n"]:>5}{tot["skip"]:>7}{tot["skip_pass"]:>6}'
          f'{tot["skip_held"]:>13}{tot["bad_ref"]:>9}{tot["good_held"]:>17}{tot["wrong_pass"]:>15}')
    print('  (문제가 하나도 없는 시퀀스는 줄 생략)')

# ── 2. 문제 시퀀스 시간표 + 그래프 ──
wp = Counter(r['sq'] for recs in D24.values() for r in recs
             if r['valid'] and r['err'] is not None and r['err'][1] > MAX_ERR)
bs = Counter(r['sq'] for recs in D24.values() for r in recs if r['bad_ref'])
show24 = [sq for sq, _ in (wp + bs).most_common(SHOW_SEQ24)]
print('\n' + '═' * 112)
print('2. 프레임별 시간표 — 폭은 화면 폭 대비 [아래끝, 위끝]. 비율 = 예측/정답, 기준/정답 (1.00이 정상)')
print('═' * 112)
for sq in show24:
    rs = [r for recs in D24.values() for r in recs if r['sq'] == sq]
    print(f'\n▶ {sq}  ({len(rs)}장)')
    print(f'  {"프레임":>9} {"판정":<18}{"오차":>6}  {"예측/정답":>13}  {"기준/정답":>13}  {"기록":>4}  표시')
    for r in rs:
        pr = '' if r['pw'] is None or r['gw'] is None else '{:.2f} {:.2f}'.format(*(r['pw'] / r['gw']))
        rr = '' if r['ref'] is None or r['gw'] is None else '{:.2f} {:.2f}'.format(*(r['ref'] / r['gw']))
        er = '' if r['err'] is None else f'{r["err"][1]:.2f}'
        mark = ' '.join(m for m, ok in (('건너뜀', r['skip']), ('기준오염', r['bad_ref']),
                                        ('★오답통과', r['valid'] and r['err'] is not None and r['err'][1] > MAX_ERR)) if ok)
        print(f'  {r["ix"]:>9} {("통과" if r["valid"] else r["reason"]):<18}{er:>6}  {pr:>13}  {rr:>13}  {r["nhist"]:>4}  {mark}')

    t = np.array([r['ts'] - rs[0]['ts'] for r in rs])
    def col(key, k):
        return np.array([np.nan if r[key] is None or r['gw'] is None else (r[key] / r['gw'])[k] for r in rs])
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.6), sharey=True)
    for k, ax in enumerate(axs):
        ax.axhspan(1 / (1 + TOL24), 1 + TOL24, color='0.9', label=f'정답 ±{TOL24:.0%}')
        ax.plot(t, col('pw', k), 'o-', ms=4, label='예측 폭 / 정답 폭')
        ax.plot(t, col('ref', k), 's--', ms=4, label='판정 기준 폭 / 정답 폭')
        for r, x in zip(rs, t):
            if r['valid'] and r['err'] is not None and r['err'][1] > MAX_ERR:
                ax.axvline(x, color='r', alpha=.4)
            elif r['reason'] == 'width':
                ax.axvline(x, color='orange', alpha=.3)
        ax.set_title(f'{sq} — {["측정 구간 아래끝", "위끝"][k]} 행')
        ax.set_xlabel('시퀀스 시작부터 초'); ax.set_ylim(0.3, 3.0); ax.set_yscale('log')
        ax.set_yticks([0.5, 0.8, 1, 1.25, 1.4, 2, 3]); ax.set_yticklabels(['0.5', '0.8', '1', '1.25', '1.4', '2', '3'])
    axs[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('빨간 세로줄 = 오답(0.10 초과)인데 통과   주황 = width 보류', fontsize=9)
    plt.tight_layout(); plt.show()

# ── 3. 대책 비교 ──
FIX24 = {}
for recs in D24.values():
    for cam in {r['cam'] for r in recs}:
        g_ = np.array([r['gw'] for r in recs if r['cam'] == cam and r['gw'] is not None])
        FIX24[cam] = np.median(g_, 0) if len(g_) else None

min_n24 = WIDTH23['min_samples']
win24 = WIDTH23['window_sec']
for recs in D24.values():
    hist, cur = [], None
    for r in recs:
        if r['sq'] != cur:
            hist, cur = [], r['sq']
        fx = FIX24.get(r['cam'])
        ok_fix = r['past_straddle'] and r['pw'] is not None and fx is not None and ldev24(r['pw'], fx) <= LTOL24
        r['v_fixed'] = ok_fix
        hist = [(t0, w) for t0, w in hist if t0 >= r['ts'] - win24]
        ref = np.median([w for _, w in hist], 0) if len(hist) >= min_n24 else fx
        r['v_combo'] = ok_fix and ldev24(r['pw'], ref) <= LTOL24
        if ok_fix:
            hist.append((r['ts'], r['pw']))

print('\n' + '═' * 112)
print('3. 대책 비교 — 칸 = 통과율 · 통과 중 0.10 / 0.15 / 0.25 초과 건수 · 건너뜀 통과 · 맞는데 폭 때문에 보류')
print('   ⚠️ 고정 폭을 정답 폭 중앙값으로 흉내 냈으므로 실제(설치 때 한 번 측정)보다 유리한 숫자. 방향만 볼 것')
print('═' * 112)
for tag, recs in D24.items():
    print(f'\n▶ {tag}')
    for name, key in (('코드② (지난 프레임)', None), ('고정 폭만', 'v_fixed'), ('보정안 (고정 범위 안만 기록)', 'v_combo')):
        v = [r['valid'] if key is None else r[key] for r in recs]
        e = np.array([r['err'][1] for r, ok in zip(recs, v) if ok and r['err'] is not None])
        held_good = sum((not ok) and r['past_straddle'] and not r['skip'] for r, ok in zip(recs, v))
        held_good -= 0 if key else sum(r['reason'] == 'width_calibrating' and not r['skip'] for r in recs)
        print(f'  {name:<26}{sum(v) / len(v) * 100:>4.0f}% ({sum(v)}/{len(v)}) · {(e > MAX_ERR).sum():>2} / {(e > .15).sum():>2} / '
              f'{(e > .25).sum():>2} · 건너뜀 통과 {sum(ok and r["skip"] for r, ok in zip(recs, v)):>2} · 맞는데 폭 보류 {held_good:>3}')
print('  (코드②의 "맞는데 폭 보류"는 보정 중 보류를 뺀 width 보류만 셈)')

# ── 4. 고정 폭이 현실적인가 ──
print('\n' + '═' * 112)
print(f'4. 날짜(카메라 추정)별 정답 폭 — 중앙값 대비 흩어짐. 고정 기준 ±{TOL24:.0%} 밖에 드는 비율이 작아야 고정 기준이 쓸 만함')
print('═' * 112)
print(f'  {"날짜":<14}{"장수":>5}{"시퀀스":>6}  {"아래끝 폭 중앙값":>14}{"위끝":>7}  {"5~95% (중앙값 대비)":>22}  {"±" + f"{TOL24:.0%}" + " 밖":>8}')
for tag, recs in D24.items():
    for cam in sorted({r['cam'] for r in recs}):
        g_ = np.array([r['gw'] for r in recs if r['cam'] == cam and r['gw'] is not None])
        if len(g_) == 0:
            continue
        m = np.median(g_, 0)
        q = np.percentile(np.log(g_ / m).max(1) * np.sign(np.log(g_ / m).sum(1)), [5, 95])
        out = np.mean(np.abs(np.log(g_ / m)).max(1) > LTOL24) * 100
        print(f'  {cam:<14}{len(g_):>5}{len({r["sq"] for r in recs if r["cam"] == cam}):>6}  {m[0]:>14.3f}{m[1]:>7.3f}  '
              f'{np.expm1(q[0]):>+10.0%} ~ {np.expm1(q[1]):>+6.0%}  {out:>7.1f}%   ({tag})')

# ── 결론 자동 정리 ──
print('\n' + '═' * 112)
mis = D24['틀어짐']
target = [r for r in mis if r['valid'] and r['err'] is not None and r['err'][1] > 0.25 and r['skip']]
if target:
    n_bad = sum(r['bad_ref'] for r in target)
    print(f'가설 판정: 틀어짐에서 "건너뜀인데 통과한 0.25 초과" {len(target)}건 중 기준이 오염된 상태였던 것 {n_bad}건'
          f' → {"가설 맞음 (지난 프레임 기준이 틀린 폭을 배움)" if n_bad == len(target) else "일부만 설명됨 — 시간표 확인"}')
else:
    big = [r for r in mis if r['valid'] and r['err'] is not None and r['err'][1] > 0.25]
    print(f'가설 판정: 틀어짐에서 통과한 0.25 초과 {len(big)}건 중 폭 비율로 건너뜀(>{SKIP24})인 것이 없음'
          ' → 폭 기준 오염이 아니라 다른 원인(갈림길 등). 시간표의 예측/정답 비율 확인')
print('═' * 112)


# ============================== 25번 셀 ==============================
# 25. 평가 세트 확대 + 모델 비교 — Validation 60_전방 전체(정답 v3 약 2,300장)로 사전학습 vs 재학습(1~4차)
# 필요: 1번 셀만 (DRIVE_ROOT·VALID_DIR). 2~24번 셀 없이 혼자 돈다. GPU면 빠르고 CPU도 됨(수십 분)
#
# 왜: 지금까지 '검증 오답 0'은 257장 기준이었다(4번 셀이 OpenCV 보정용으로 시퀀스 절반을 떼어 놓고, 2번 셀이 2,000장만 뽑음).
#     오답 0이 나와도 257장이면 '오답률 1.2% 이하'까지만 말할 수 있다(95% 신뢰, 3/N 규칙).
#     이제 OpenCV 보정이 필요 없으므로 Validation 60_전방 차선 라벨 전부를 평가에 쓴다.
#   · 검증  = 20200916 이외 날짜 전부 / 틀어짐 = 20200916 (카메라 장착이 다른 차량)
#   · 재학습 모델(scripts/train_ufld.py)은 Validation 시퀀스를 전부 빼고 학습했으므로 이 평가에 쓸 수 있다.
#     (18·19번 셀 파인튜닝은 Validation 일부로 학습했으므로 여기서 비교하지 않는다)
#   · 정답·추론·판정은 저장소 코드 그대로: 정답 v3 = scripts/ufld_labels.py(노트북 19번과 일치 확인),
#     추론 = src/lane/ufld.py preprocess·decode_rows(23번에서 노트북과 일치 확인), 판정 = LaneGuard
#   · 오차 = 21번 셀과 같은 정의(측정 구간 행들에서 예측 차로 중심과 정답 중심의 차이 / 정답 차로 폭, 최대값)
# 모델 파일: 드라이브 SafeWatch/ufld/ 아래. 없는 모델은 건너뛴다.
import sys, subprocess, zipfile, tarfile, shutil, time, copy, re
import unicodedata as ud
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, cv2, torch
import matplotlib.pyplot as plt

REPO25   = Path('/content/SafeWatch_AI')
BRANCH25 = 'feature/lane-ufld'
UFLD25   = DRIVE_ROOT / 'ufld'
MODELS25 = {'사전학습': UFLD25 / 'culane_res18.pth',
            '재학습':   UFLD25 / 'ufld_aihub_ft.pth',       # 1차 (26번 셀 첫 실행, 뒤집기·밝기만)
            '재학습2':  UFLD25 / 'ufld_aihub_aug.pth',     # 2차 (기하 증강 12에폭)
            '재학습3':  UFLD25 / 'ufld_aihub_aug24.pth',   # 3차 (기하 증강 24에폭)
            '재학습4':  UFLD25 / 'ufld_aihub_clean.pth',   # 4차 (2차 + 가짜 옆 차선 라벨 정리)
            '재학습5':  UFLD25 / 'ufld_aihub_far.pth'}     # 5차 (2차 + 먼 행 가중, 30번 셀)
EVAL25   = Path('/content/lane_eval_v60')              # 세션 로컬
BACKUP25 = DRIVE_ROOT / 'lane_eval_v60.tar'            # 드라이브 백업 — 다음엔 압축만 푼다
MIS25    = '20200916'
CUTS25   = [0.56, 0.60, 0.64, 0.68]                    # 측정 구간 위끝 (팀 결정 대상 — 전부 본다)
BOT25    = 0.98
ERR25    = (0.10, 0.15, 0.25)                          # 오답 / 표류 기준 / 큰 오답(걸침 기준)
RATE25   = 0.02                                        # 2단계 기준: 큰 오답 0 + 0.10 초과 비율 ≤ 2%
WIDTH25  = dict(window_sec=60.0, min_samples=5)        # 1초 간격 사진용 (23번과 같음)
BS25     = 8
N_SHOW25 = 4
DEV25    = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 저장소 (23번과 같은 방식) ──
if not REPO25.exists():
    subprocess.run(['git', 'clone', '-q', '-b', BRANCH25,
                    'https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git', str(REPO25)], check=True)
else:
    for cmd in (['fetch', '-q', 'origin', BRANCH25], ['checkout', '-q', BRANCH25], ['reset', '-q', '--hard', f'origin/{BRANCH25}']):
        subprocess.run(['git', '-C', str(REPO25), *cmd], check=True)
print('저장소:', subprocess.run(['git', '-C', str(REPO25), 'log', '-1', '--oneline'], capture_output=True, text=True).stdout.strip())
for m in [m for m in sys.modules if m in ('src', 'scripts') or m.startswith(('src.', 'scripts.'))]:
    del sys.modules[m]
if str(REPO25) not in sys.path:
    sys.path.insert(0, str(REPO25))
from src.utils.config import Config, load_config
from src.lane.ufld import UFLDSpec, preprocess, decode_rows
from src.lane.ufld_guard import LaneGuard
from scripts.ufld_labels import parse_lanes, ego_lanes, frame_key, sequence_of, x_at
from scripts.export_ufld_onnx import UFLDv2RowNet, load_weights

CFG25  = load_config(REPO25 / 'configs/default.yaml')
SPEC25 = UFLDSpec.from_config(CFG25.lane.ufld)
nfc25  = lambda s: ud.normalize('NFC', s)
fov25  = lambda c: (m.group(0) if (m := re.search(r'\d+_(전방|측방)', c)) else None)
is_sl25 = lambda c: c.startswith('sl/') or '/sl/' in c


# ── 데이터: Validation zip에서 60_전방 차선(sl) 라벨·사진 전부 ──
def extract25():
    lab, img = {}, {}
    for z in sorted(VALID_DIR.rglob('*.zip')):
        with zipfile.ZipFile(z) as zf:
            for o in sorted(zf.namelist(), key=nfc25):
                c = nfc25(o)
                if not is_sl25(c) or fov25(c) != '60_전방':
                    continue
                if c.lower().endswith('.xml'):
                    lab[frame_key(c)] = (z, o)                     # 버전 중 이름순 마지막
                elif c.lower().endswith(('.jpg', '.png')):
                    img[frame_key(c)] = (z, o)
    keys = sorted(set(lab) & set(img))
    print(f'라벨 {len(lab)} · 사진 {len(img)} · 둘 다 있음 {len(keys)}')
    shutil.rmtree(EVAL25, ignore_errors=True)
    (EVAL25 / 'xml').mkdir(parents=True); (EVAL25 / 'img').mkdir(parents=True)
    by_zip = defaultdict(list)
    for k in keys:
        by_zip[lab[k][0]].append((lab[k][1], EVAL25 / 'xml' / f'{k}.xml'))
        by_zip[img[k][0]].append((img[k][1], EVAL25 / 'img' / f'{k}.jpg'))
    for z, items in by_zip.items():
        with zipfile.ZipFile(z) as zf:
            for o, dst in items:
                dst.write_bytes(zf.read(o))
        print(f'  {len(items)}개 <- {Path(z).name}')


if (EVAL25 / 'xml').exists() and any((EVAL25 / 'xml').iterdir()):
    print('기존 평가 데이터 사용')
elif BACKUP25.exists():
    print('드라이브 백업에서 복원 중...')
    EVAL25.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BACKUP25) as t:
        t.extractall(EVAL25)
else:
    extract25()
    print('드라이브에 백업 저장 중...')
    with tarfile.open(BACKUP25, 'w') as t:
        for sub in ('xml', 'img'):
            t.add(EVAL25 / sub, arcname=sub)


def idx25(key):
    d = ''.join(ch for ch in key.split('_')[-1] if ch.isdigit())
    return int(d) if d else 0


FR25, skip25 = [], Counter()
for xp in sorted((EVAL25 / 'xml').glob('*.xml')):
    k = xp.stem
    try:
        W, H, lanes = parse_lanes(xp.read_bytes())
    except Exception:
        skip25['XML 깨짐'] += 1; continue
    l4 = ego_lanes(lanes, W, H) if lanes else None
    if l4 is None:
        skip25['자차 쌍 없음/폭 이상'] += 1; continue
    FR25.append(dict(key=k, seq=sequence_of(k), idx=idx25(k), mis=sequence_of(k).startswith(MIS25),
                     W=W, H=H, l4=l4, ip=EVAL25 / 'img' / f'{k}.jpg'))
FR25.sort(key=lambda f: (f['seq'], f['idx']))
for tag, sel in (('검증', False), ('틀어짐', True)):
    fs = [f for f in FR25 if f['mis'] == sel]
    print(f'{tag}: {len(fs)}장 · 시퀀스 {len({f["seq"] for f in fs})}개')
print('정답 없음으로 제외:', dict(skip25))


def gt_rows25(f, s, cut, n=12):
    """19번 gt_rows와 같음 — 정답 자차 좌/우 (y, xl, xr), 원본 사진 좌표. 5행 미만이면 None"""
    rows = []
    for y in np.linspace(cut * f['H'], BOT25 * f['H'], n):
        a, b = x_at(f['l4'][1][3], y), x_at(f['l4'][2][3], y)
        if a is not None and b is not None:
            rows.append((y * s, a * s, b * s))
    return np.array(rows) if len(rows) >= 5 else None


def err25(L, R, gt):
    """15번 err_img와 같음 — (중앙값, 최대값) 또는 None"""
    e = []
    for y, gl, gr in gt:
        a = x_at(L, y) if len(L) >= 2 else None
        b = x_at(R, y) if len(R) >= 2 else None
        if a is None or b is None or gr - gl <= 1:
            continue
        e.append(abs((a + b) / 2 - (gl + gr) / 2) / (gr - gl))
    return (float(np.median(e)), float(np.max(e))) if len(e) >= 5 else None


# ── 추론 (모델마다 한 번) ──
@torch.no_grad()
def infer25(path):
    net = UFLDv2RowNet(SPEC25, grid_col=100, num_col=81, fc_norm=True)
    load_weights(net, path)
    net.keep_rows_only()
    net = net.eval().to(DEV25)
    out, t0 = {}, time.time()
    for i in range(0, len(FR25), BS25):
        batch = FR25[i:i + BS25]
        imgs = [cv2.imread(str(f['ip'])) for f in batch]
        x = torch.from_numpy(np.concatenate([preprocess(b, SPEC25) for b in imgs])).to(DEV25)
        loc, ex = net(x)
        loc, ex = loc.float().cpu().numpy(), ex.float().cpu().numpy()
        for j, (f, b) in enumerate(zip(batch, imgs)):
            h, w = b.shape[:2]
            lanes = decode_rows(loc[j], ex[j], SPEC25, w, h)
            out[f['key']] = (lanes[1], lanes[2], w, h)
    print(f'  {path.name}: {len(out)}장 {time.time() - t0:.0f}초 ({DEV25})')
    del net
    if DEV25 == 'cuda':
        torch.cuda.empty_cache()
    return out


PRED25 = {}
for name, path in MODELS25.items():
    if path.exists():
        PRED25[name] = infer25(path)
    else:
        print(f'  {name}: {path} 없음 — 건너뜀')


# ── 판정 ──
def guard25(cut, existence_only):
    d = copy.deepcopy(CFG25.to_dict()['lane']['ufld']['guard'])
    d['width'].update(WIDTH25)
    if existence_only:
        d['min_straddle_margin'] = -1e9
        d['width']['enabled'] = False
        d['temporal']['enabled'] = False
    return LaneGuard(Config(d), SPEC25.row_anchor, (cut, BOT25))


def run25(name, cut, existence_only, mis):
    rows, g, cur = [], None, None
    for f in FR25:
        if f['mis'] != mis:
            continue
        if f['seq'] != cur:
            g, cur = guard25(cut, existence_only), f['seq']
        L, R, w, h = PRED25[name][f['key']]
        res = g.check(L, R, w, h, f['idx'] / 30.0)
        gt = gt_rows25(f, w / f['W'], cut)
        rows.append(dict(f=f, ok=res.valid, reason=res.reason, err=None if gt is None else err25(L, R, gt)))
    return rows


def stat25(rows):
    n = sum(r['err'] is not None for r in rows)
    e = np.array([r['err'][1] for r in rows if r['ok'] and r['err'] is not None])
    k = len(e)
    c = [int((e > t).sum()) for t in ERR25]
    return dict(n=n, k=k, av=k / max(n, 1), rate=c[0] / max(k, 1), c=c,
                strict=k > 0 and c[0] == 0, two=k > 0 and c[2] == 0 and c[0] / k <= RATE25)


RES25 = {}
for name in PRED25:
    for cut in CUTS25:
        for eo in (True, False):
            for mis in (False, True):
                RES25[(name, cut, eo, mis)] = run25(name, cut, eo, mis)

print('\n' + '═' * 118)
print('칸 = 사용 가능 비율 · 통과 중 0.10 초과 비율 · 0.10/0.15/0.25 초과 건수 · 판정(엄격 = 0.10 초과 0건 / 2단계 = 0.25 초과 0건 + 0.10 초과 ≤ 2%)')
print('사용 가능 비율 = 통과 장수 / 정답 있는 장수. "0건"이어도 장수가 N이면 실제 오답률은 3/N 이하까지만 말할 수 있다(95%)')
print('═' * 118)
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    print(f'\n▶ {tag}')
    print(f'  {"모델":<8}{"안전장치":<10}' + ''.join(f'{"범위 " + str(c):>26}' for c in CUTS25))
    for name in PRED25:
        for eo, gname in ((True, '존재만'), (False, '기본 전부')):
            cells = []
            for cut in CUTS25:
                s = stat25(RES25[(name, cut, eo, mis)])
                v = '통과0' if s['k'] == 0 else ('엄격✅' if s['strict'] else ('2단계✅' if s['two'] else '❌'))
                cells.append(f'{s["av"] * 100:>3.0f}% {s["rate"] * 100:>4.1f}% {s["c"][0]:>2}/{s["c"][1]:>2}/{s["c"][2]:>2} {v:<5}')
            print(f'  {name:<8}{gname:<10}' + ''.join(f'{c:>26}' for c in cells))
    for name in PRED25:
        s = stat25(RES25[(name, CUTS25[-1], False, mis)])
        ub = f'{3 / s["k"] * 100:.2f}%' if s['k'] else '-'
        print(f'  {name} 범위 {CUTS25[-1]} 기본 전부: 정답 있는 {s["n"]}장 중 통과 {s["k"]}장 → 오답 0건이어도 오답률 상한 약 {ub}')

# ── 큰 오답이 몰린 시퀀스 ──
BEST25 = list(PRED25)[-1]                              # 있는 모델 중 가장 나중 것 (재학습4 > 재학습3 > …)
CUT_SHOW25 = 0.68
print(f'\n[{BEST25} · 범위 {CUT_SHOW25} · 기본 전부] 통과했는데 0.10 초과인 프레임이 있는 시퀀스')
bad25 = []
for mis in (False, True):
    rs = RES25[(BEST25, CUT_SHOW25, False, mis)]
    per = Counter(r['f']['seq'] for r in rs if r['ok'] and r['err'] is not None and r['err'][1] > ERR25[0])
    for sq, n in per.most_common(8):
        print(f'  {"틀어짐" if mis else "검증"} {sq}: {n}장')
    bad25 += [r for r in rs if r['ok'] and r['err'] is not None and r['err'][1] > ERR25[0]]
bad25.sort(key=lambda r: -r['err'][1])


def show25(r, cut, title):
    f = r['f']
    bgr = cv2.imread(str(f['ip'])); vis = bgr.copy()
    s = bgr.shape[1] / f['W']
    for q in f['l4']:
        if q is not None:
            p = np.int32([(x * s, y * s) for x, y in sorted(q[3], key=lambda t: t[1])])
            cv2.polylines(vis, [p], False, (200, 200, 200), 2)
    for y, a, b in gt_rows25(f, s, cut):
        cv2.circle(vis, (int(a), int(y)), 9, (255, 255, 0), 2); cv2.circle(vis, (int(b), int(y)), 9, (255, 255, 0), 2)
    L, R, _, _ = PRED25[BEST25][f['key']]
    for pts, col in ((L, (0, 0, 255)), (R, (0, 255, 0))):
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 4, col, -1)
    cv2.line(vis, (0, int(cut * bgr.shape[0])), (vis.shape[1], int(cut * bgr.shape[0])), (0, 255, 255), 2)
    plt.figure(figsize=(12, 6.5)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title(f'{title} — {f["key"]}  최대오차={r["err"][1]:.3f}\n'
              '회색 = 라벨(자차·옆 4선)  하늘색 원 = 정답 v3  노랑 = 측정 구간 위끝  빨강·초록 = 예측', fontsize=9)
    plt.show()


print(f'\n[그림] 통과 오답 {len(bad25)}건 중 최대 {N_SHOW25}건')
for r in bad25[:N_SHOW25]:
    show25(r, CUT_SHOW25, f'[25] {BEST25} 통과 오답')


# ============================== 26번 셀 ==============================
# 26. 재학습 (4차, 옆 차선 라벨 정리) — AI Hub Training 차선(sl) 전체로 UFLDv2 파인튜닝
# 4차 (2026-11-09): 27번 v2에서 재학습 모델이 라벨의 가짜 옆 차선(이중선·도류대·갓길)을 배운 것을 확인 →
#   데이터 만들 때 옆 차로 폭이 자차 차로 폭의 0.6~1.6배 밖인 옆 차선(0·3번) 라벨을 뺀다(약 11%, 자차 차선은 그대로).
#   그 외는 채택 모델인 2차와 같다(기하 증강·12에폭). 라벨이 바뀌어 데이터를 새로 만든다(ufld_aihub3, 약 5분).
#   결과는 ufld_aihub_clean.pth — 2차 ufld_aihub_aug.pth는 비교용으로 그대로 둔다. A100 기준 약 45분.
# 3차 (2026-11-09): 에폭만 24로 → 검증은 조금 낫고 틀어짐은 나빠져 채택 안 함.
# 필요: 1번 셀만 (DRIVE_ROOT·VALID_DIR). **GPU 런타임**. 2~25번 셀 없이 혼자 돈다.
#
# 2차에서 바뀐 것 (2026-11-08) — 1차(뒤집기·밝기만)는 검증은 좋아졌지만 장착이 다른 카메라(틀어짐)에서 나빠졌다
#   · 원 저장소처럼 이동(x ±12%, y ±17%)·회전(±6°)·확대(0.8~1.2배) 증강, 차선 정답도 같이 변환
#   · 그래서 데이터를 자르기 전 1600×533으로 저장(형식 2) → 1차 데이터(ufld_aihub.tar)는 못 쓰고 새로 만든다(약 15분)
#   · 에폭 선택 = 선택용 시퀀스에 같은 증강을 고정 난수로 건 손실, 에폭 20 → 12
#   · 결과는 ufld_aihub_aug.pth — 1차 ufld_aihub_ft.pth는 비교용으로 그대로 둔다
#
# 하는 일 (저장소 코드 그대로 실행 — docs/ufld-training-guide.md 3·4장의 코랩판)
#   ① 데이터 만들기 scripts/prepare_ufld_dataset.py
#      드라이브 Training/sl의 zip을 복사하지 않고 그 자리에서 읽어 60_전방 정답 v3 프레임만 1600×533으로 저장.
#      Validation 시퀀스 전부 + 20200916 날짜는 학습에서 뺀다(25번 평가와 겹치지 않게).
#      결과는 드라이브 ufld_aihub2.tar로 백업 → 다음부터는 압축만 푼다.
#   ② 학습 scripts/train_ufld.py — CULane 가중치에서 시작, 에폭마다 드라이브에 저장(끊기면 다시 실행 = 이어 하기)
#   ③ 끝나면 25번 셀을 돌리면 사전학습·1차·2차가 같은 표로 비교된다.
#
# MODE26 = 'test' 먼저 → 데이터 2,000장·학습 50스텝으로 돌려 전체 시간을 추정만 한다(10~20분).
#          'full'      → 전체. 코랩이 끊기면 이 셀을 그대로 다시 실행하면 이어서 한다.
import os, sys, subprocess, zipfile, tarfile, shutil, time, json, re
import unicodedata as ud
from pathlib import Path

MODE26    = 'full'                                   # 'test' → 'full'
REPO26    = Path('/content/SafeWatch_AI')
BRANCH26  = 'feature/lane-ufld'
SL26      = DRIVE_ROOT / '도로주행영상' / 'Training' / 'sl'
UFLD26    = DRIVE_ROOT / 'ufld'
PRETRAIN26 = UFLD26 / 'culane_res18.pth'
LOCAL26   = Path('/content/aihub26')                 # 라벨 zip 복사본 (작음)
DATA26    = Path('/content/ufld_aihub3' if MODE26 == 'full' else '/content/ufld_aihub3_test')
BACKUP26  = DRIVE_ROOT / 'ufld_aihub3.tar'           # 만든 학습 데이터 백업 (약 6GB, 형식 2 + 옆 차선 정리)
OUT26     = (UFLD26 / 'ufld_aihub_clean.pth') if MODE26 == 'full' else Path('/content/ufld_test/ufld_test.pth')
LIMIT26   = 2000                                     # test: 데이터 장수 (학습 50스텝 × 배치 32가 나오게)
EPOCHS26  = 12                                       # 설정 파일 값을 덮어쓴다(3차는 24) — 코사인 학습률도 여기에 맞춰진다
STEPS26   = 50                                       # test: 학습 스텝
CPU26     = os.cpu_count() or 2
nfc26     = lambda s: ud.normalize('NFC', s)

assert MODE26 in ('test', 'full'), MODE26
try:
    import torch
    assert torch.cuda.is_available()
    print('GPU:', torch.cuda.get_device_name(0), '· CPU', CPU26)
except AssertionError:
    raise SystemExit('GPU 런타임이 아닙니다 — 런타임 > 런타임 유형 변경 > GPU')

# ── 저장소 (25번과 같은 방식) ──
if not REPO26.exists():
    subprocess.run(['git', 'clone', '-q', '-b', BRANCH26,
                    'https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git', str(REPO26)], check=True)
else:
    for cmd in (['fetch', '-q', 'origin', BRANCH26], ['checkout', '-q', BRANCH26], ['reset', '-q', '--hard', f'origin/{BRANCH26}']):
        subprocess.run(['git', '-C', str(REPO26), *cmd], check=True)
print('저장소:', subprocess.run(['git', '-C', str(REPO26), 'log', '-1', '--oneline'], capture_output=True, text=True).stdout.strip())


def run26(args, marks=()):
    """저장소 폴더에서 python -u -m ... 실행, 출력을 그대로 보여 주고 marks 문자열이 처음 나온 시각을 돌려준다."""
    t0, seen = time.time(), {}
    p = subprocess.Popen([sys.executable, '-u', '-m', *args], cwd=REPO26, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    for line in p.stdout:
        print(line, end='')
        for m in marks:
            if m not in seen and m in line:
                seen[m] = (time.time() - t0, line)
    if p.wait() != 0:
        raise SystemExit(f'실패 (종료 코드 {p.returncode}) — 위 출력 확인')
    return time.time() - t0, seen


# ════════════════ ① 데이터 만들기 ════════════════
def find_zips26():
    if not SL26.exists():
        raise SystemExit(f'{SL26} 없음 — 드라이브 Training/sl 경로 확인')
    zs = sorted(SL26.glob('*.zip'), key=lambda p: nfc26(p.name))
    lab = [z for z in zs if '라벨' in nfc26(z.name)]
    img = [z for z in zs if '원천' in nfc26(z.name)]
    if len(lab) != 2 or len(img) != 2:
        raise SystemExit(f'sl zip이 예상과 다름 — 라벨 {[z.name for z in lab]} / 원천 {[z.name for z in img]}')
    for z in lab + img:
        print(f'  {z.name:<26} {z.stat().st_size / 1e9:6.1f}GB')
    return lab, img


def exclude_zip26():
    """Validation 라벨 zip(들) — 시퀀스 이름만 쓰므로 여러 개면 이름만 모은 작은 zip을 만든다."""
    labs = []
    for z in sorted(VALID_DIR.rglob('*.zip')):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith('.xml')]
        if names:
            labs.append((z, names))
    if not labs:
        raise SystemExit(f'{VALID_DIR}에서 라벨(xml) zip을 못 찾음')
    print('  제외용 Validation 라벨:', ', '.join(f'{z.name}({len(n)}장)' for z, n in labs))
    if len(labs) == 1:
        return labs[0][0]
    idx = LOCAL26 / 'validation_label_names.zip'
    with zipfile.ZipFile(idx, 'w') as zf:
        for _, names in labs:
            for n in names:
                zf.writestr(n, b'')
    return idx


NALL26 = None
LOCAL26.mkdir(parents=True, exist_ok=True)
if MODE26 == 'full' and (DATA26 / 'manifest.json').exists():
    print(f'\n① 학습 데이터 이미 있음 — {DATA26}')
elif MODE26 == 'full' and BACKUP26.exists():
    print(f'\n① 드라이브 백업에서 복원 — {BACKUP26.name} ({BACKUP26.stat().st_size / 1e9:.1f}GB)')
    t = time.time()
    with tarfile.open(BACKUP26) as tf:
        tf.extractall(DATA26.parent)
    print(f'  {time.time() - t:.0f}초')
else:
    print('\n① 학습 데이터 만들기')
    lab26, img26 = find_zips26()
    lab_local = []
    for z in lab26:                                    # 라벨은 작아서 복사(여러 번 읽음), 원천은 드라이브에서 바로 읽음
        dst = LOCAL26 / nfc26(z.name)
        if not dst.exists() or dst.stat().st_size != z.stat().st_size:
            shutil.copy(z, dst)
        lab_local.append(dst)
    args = ['scripts.prepare_ufld_dataset',
            '--label-zips', *map(str, lab_local), '--image-zips', *map(str, img26),
            '--exclude-label-zip', str(exclude_zip26()), '--out', str(DATA26), '--workers', str(CPU26)]
    if MODE26 == 'test':
        args += ['--limit', str(LIMIT26)]
    sec, seen = run26(args, marks=('정답 있는 프레임',))
    print(f'  걸린 시간 {sec / 60:.1f}분')
    if MODE26 == 'test' and '정답 있는 프레임' in seen:
        t_lab, line = seen['정답 있는 프레임']
        n_all = NALL26 = int(re.search(r'정답 있는 프레임 (\d+)장', line).group(1))
        est = t_lab + (sec - t_lab) * n_all / LIMIT26     # 이미지 목록 읽기 시간까지 비례로 늘려 잡아 약간 과대 추정
        print(f'  → 전체 {n_all}장 데이터 만들기 추정 약 {est / 60:.0f}분 (라벨 읽기 {t_lab / 60:.1f}분 + 이미지 {LIMIT26}장 {(sec - t_lab) / 60:.1f}분 기준)')
    if MODE26 == 'full':
        print(f'  드라이브 백업 → {BACKUP26.name}')
        t = time.time()
        part26 = BACKUP26.with_suffix('.tar.part')
        with tarfile.open(part26, 'w') as tf:
            tf.add(DATA26, arcname=DATA26.name)
        part26.replace(BACKUP26)
        print(f'  {time.time() - t:.0f}초 · {BACKUP26.stat().st_size / 1e9:.1f}GB')

man26 = json.loads((DATA26 / 'manifest.json').read_text())
if man26.get('format') != 2:
    raise SystemExit(f'{DATA26}는 옛 형식(잘린 사진) — 폴더를 지우고 다시 실행')
if 'adj_ratio_min' not in man26.get('label_rules', {}):
    raise SystemExit(f'{DATA26}는 옆 차선 정리 전 데이터 — 폴더(와 드라이브 {BACKUP26.name})를 지우고 다시 실행')
print('  라벨 규칙:', {k: man26['label_rules'][k] for k in ('adj_ratio_min', 'adj_ratio_max')})
print('  학습', man26['splits']['train'], '· 선택용', man26['splits']['val'])
print('  제외', man26['skipped'])

# ════════════════ ② 학습 ════════════════
if not PRETRAIN26.exists():
    raise SystemExit(f'{PRETRAIN26} 없음 — 15번 셀에서 받은 CULane 가중치')
import yaml
tc26 = yaml.safe_load((REPO26 / 'configs/ufld_train.yaml').read_text())
tc26['train']['num_workers'] = min(int(tc26['train']['num_workers']), CPU26)   # 코랩 CPU 수에 맞춤 (나머지 설정은 그대로)
tc26['train']['epochs'] = EPOCHS26
if MODE26 == 'test':
    tc26['train']['log_every'] = 10
cfg26 = Path('/content/ufld_train_colab.yaml')
cfg26.write_text(yaml.safe_dump(tc26, allow_unicode=True))
geo26 = tc26['train']['augment'].get('geometric')
print(f"\n② 학습 — 배치 {tc26['train']['batch_size']} · 에폭 {tc26['train']['epochs']} · workers {tc26['train']['num_workers']}"
      f" · 기하 증강 {geo26} · 에폭 선택 {tc26['train']['select_by']} · 저장 {OUT26}")
OUT26.parent.mkdir(parents=True, exist_ok=True)
args = ['scripts.train_ufld', '--data', str(DATA26), '--weights', str(PRETRAIN26), '--out', str(OUT26),
        '--train-config', str(cfg26)]
if MODE26 == 'test':
    for f in OUT26.parent.glob('ufld_test*'):
        f.unlink()
    args += ['--epochs', '1', '--max-steps', str(STEPS26)]
else:
    args += ['--resume']                               # 처음이면 _last.pth가 없어 새로 시작한다
sec, seen = run26(args, marks=('장치 ', '학습 전 선택용 손실', '스텝 10/', f'스텝 {STEPS26}/'))

if MODE26 == 'test':
    try:
        t_val = seen['학습 전 선택용 손실'][0] - seen['장치 '][0]          # 선택용 데이터 한 바퀴
        s_step = (seen[f'스텝 {STEPS26}/'][0] - seen['스텝 10/'][0]) / (STEPS26 - 10)
    except KeyError:
        t_val = s_step = None
    bs = int(tc26['train']['batch_size'])
    print('\n' + '═' * 100)
    if s_step:
        n_full = NALL26 or 22745                            # ①의 '정답 있는 프레임' (이미지 없는 프레임만큼 실제는 줄 수 있음)
        steps = int(n_full * 0.9) // bs
        val_full = t_val * (n_full * 0.1) / max(1, man26['splits']['val']['frames'])   # t_val = 선택용 두 번(증강 없음·증강) 합
        ep = s_step * steps + val_full
        print(f'스텝당 {s_step:.2f}초 · 에폭당 약 {steps}스텝 → 에폭 {ep / 60:.0f}분 · {tc26["train"]["epochs"]}에폭 약 {ep * tc26["train"]["epochs"] / 3600:.1f}시간'
              f'  (약 {n_full}장 기준)')
    else:
        print('학습 스텝이 50보다 적어 스텝 시간을 못 쟀다 — LIMIT26을 늘리거나 full의 "스텝 100/" 로그 시각으로 본다')
    print("이상 없으면 MODE26 = 'full'로 바꿔 다시 실행. 끊기면 그대로 다시 실행하면 이어서 한다.")
    print('═' * 100)
else:
    hist = json.loads(OUT26.with_name(OUT26.stem + '_history.json').read_text())
    print('\n' + '═' * 100)
    key26 = 'val_aug_loss' if tc26['train']['select_by'] == 'aug' else 'val_loss'
    print(f'{"에폭":>4} {"학습 손실":>10} {"선택용 손실":>11} {"증강 건 선택용":>14} {"분":>5}   (← 최저 = {key26} 기준)')
    best = min(hist, key=lambda h: h[key26])
    for h in hist:
        tl = '-' if h['train_loss'] is None else f"{h['train_loss']:.4f}"
        va = '-' if h.get('val_aug_loss') is None else f"{h['val_aug_loss']:.4f}"
        print(f"{h['epoch']:>4} {tl:>10} {h['val_loss']:>11.4f} {va:>14} {h.get('sec', 0) / 60:>5.0f}" + ('  ← 최저' if h is best else ''))
    if OUT26.exists():
        print(f'저장: {OUT26} — 이제 25번 → 27번 셀을 돌리면 이전 모델들과 같은 표로 비교된다')
    else:
        print('어느 에폭도 사전학습보다 선택용 손실이 낮지 않아 저장 안 됨 → 결과를 보고 학습 설정을 다시 정한다')
    import matplotlib.pyplot as plt
    ep_ = [h['epoch'] for h in hist]
    plt.figure(figsize=(7, 3.5))
    plt.plot(ep_, [h['val_loss'] for h in hist], 'o-', label='val')                 # 코랩 기본 폰트엔 한글이 없어 영어로
    if hist[0].get('val_aug_loss') is not None:
        plt.plot(ep_, [h['val_aug_loss'] for h in hist], '^-', label='val (geometric aug, selection)')
    plt.plot(ep_[1:], [h['train_loss'] for h in hist[1:]], 's-', label='train')
    plt.xlabel('epoch (0 = CULane pretrained)'); plt.ylabel('loss'); plt.legend(); plt.grid(alpha=.3)
    plt.title(f'26. fine-tuning loss (4th, cleaned adjacent labels, {EPOCHS26} epochs)'); plt.show()
    print('═' * 100)


# ============================== 27번 셀 ==============================
# 27. 옆 차로 평가 — 모델의 바깥 차선(0·3번)을 직접 쓰는 것 vs 지금 방식(자차 차로 폭 연장)
# 필요: **25번 셀을 같은 세션에서 먼저 실행** (FR25·RES25·MODELS25·SPEC25 등을 그대로 쓴다)
#
# 왜: 모델은 차선 4개 [왼쪽 옆, 자차 좌, 자차 우, 오른쪽 옆]을 내지만 실행 코드는 자차 좌·우(1·2번)만 쓴다.
#     옆 차로 차량은 src/metrics/offset.py가 자차 차로 폭을 바깥으로 한 칸 연장해 위치를 잰다(r ≈ ±1).
#     곡선·차로 폭이 다른 구간에서는 연장이 틀릴 수 있다 → 실제 0·3번 출력이 더 나은지 먼저 잰다.
#
# 비교하는 옆 차로 중심 (측정 구간의 각 행에서)
#   · 연장   = 자차 좌 L, 우 R로 추정 — 왼쪽 옆 = L − (R − L)/2,  오른쪽 옆 = R + (R − L)/2   (지금 코드)
#   · 직접   = 모델 출력 — 왼쪽 옆 = (0번 + 1번)/2,  오른쪽 옆 = (2번 + 3번)/2
#   · 직접+연장 = 직접이 측정 행 5개 미만이면 연장으로 대신 (실제로 넣는다면 이 형태)
#   오차 = |예측 옆 차로 중심 − 정답 옆 차로 중심| / 정답 옆 차로 폭, 측정 행들 중 최대값 (25번과 같은 정의)
#
# 대상 프레임 = 25번 '기본 전부' 안전장치를 자차 차로가 통과했고(실행 코드가 차로를 쓰는 프레임),
#              정답에 그쪽 옆 차선이 있는 프레임. 정답에 옆 차선이 없는데(라벨 없음 또는 실제로 없음) 모델이
#              옆 차선을 낸 경우는 따로 센다 — 정답이 없으니 오답인지는 판단 못 함, 참고용.
#
# v2 (2026-11-09): 정답 옆 차로 폭 검사 추가. 라벨의 '옆 차선'은 자차 차선 바로 바깥 선이라, 이중선·도류대·갓길이면
#     옆 차로가 아주 좁게 잡힌다(정답 폭이 작아 오차가 5~8배로 튐). 라벨 분포(옆 폭 ÷ 자차 폭, 같은 행)는
#     Training 84% · Validation 75%가 0.8~1.25에 몰리고 0.6 미만이 12% · 20% → 0.6~1.6인 행만 정답으로 쓴다.
#     (도로 규정 차로 폭 2.75~3.6m라 실제 차로끼리 비율은 0.76~1.31. 같은 행은 거리가 같아 화면 폭 비율 ≈ 실제 비율)
import time
from collections import Counter
import numpy as np, cv2, torch
import matplotlib.pyplot as plt

if 'RES25' not in globals():
    raise SystemExit('25번 셀을 이 세션에서 먼저 실행하세요 (FR25·RES25를 쓴다)')

MIN_ROWS27 = 5
ADJ_RATIO27 = (0.6, 1.6)                               # 정답 옆 차로 폭 ÷ 정답 자차 차로 폭 허용 범위
SIDES27 = (('왼쪽', 0, 1, 2, -1), ('오른쪽', 3, 2, 1, +1))  # (이름, 바깥 차선, 자차 쪽 차선, 자차 반대쪽 차선, 방향)
N_SHOW27 = 4


# ── 추론: 차선 4개 전부 (25번 PRED25는 1·2번만 저장) ──
@torch.no_grad()
def infer27(path):
    net = UFLDv2RowNet(SPEC25, grid_col=100, num_col=81, fc_norm=True)
    load_weights(net, path)
    net.keep_rows_only()
    net = net.eval().to(DEV25)
    out, t0 = {}, time.time()
    for i in range(0, len(FR25), BS25):
        batch = FR25[i:i + BS25]
        imgs = [cv2.imread(str(f['ip'])) for f in batch]
        x = torch.from_numpy(np.concatenate([preprocess(b, SPEC25) for b in imgs])).to(DEV25)
        loc, ex = net(x)
        loc, ex = loc.float().cpu().numpy(), ex.float().cpu().numpy()
        for j, (f, b) in enumerate(zip(batch, imgs)):
            h, w = b.shape[:2]
            out[f['key']] = (decode_rows(loc[j], ex[j], SPEC25, w, h), w, h)
    print(f'  {path.name}: {len(out)}장 {time.time() - t0:.0f}초')
    del net
    if DEV25 == 'cuda':
        torch.cuda.empty_cache()
    return out


if 'PRED27' not in globals():
    PRED27 = {}
for name in PRED25:
    if name not in PRED27:
        PRED27[name] = infer27(MODELS25[name])


# ── 오차 ──
def xa27(pts, y):
    return x_at(pts, y) if len(pts) >= 2 else None


def adj_err27(f, lanes, w, cut, outer, inner, other, sgn):
    """옆 차로 하나 → dict(ext=연장 오차, dir=직접 오차) — 각각 측정 행 5개 미만이면 None.
    정답 행이 5개 미만이면 'short', 폭 검사로 5개 미만이 되면 'narrow'(가짜 옆 차로)"""
    q_out, q_in, q_oth = f['l4'][outer], f['l4'][inner], f['l4'][other]
    s = w / f['W']
    gt, n_all = [], 0
    for y in np.linspace(cut * f['H'], BOT25 * f['H'], 12):
        a, b, c = x_at(q_out[3], y), x_at(q_in[3], y), x_at(q_oth[3], y)
        if a is None or b is None or c is None or abs(b - c) <= 1:
            continue
        n_all += 1
        if ADJ_RATIO27[0] <= abs(a - b) / abs(b - c) <= ADJ_RATIO27[1]:
            gt.append((y * s, a * s, b * s))
    if len(gt) < MIN_ROWS27:
        return 'narrow' if n_all >= MIN_ROWS27 else 'short'
    L, R = lanes[1], lanes[2]
    e_ext, e_dir = [], []
    for y, go, gi in gt:
        gc, gw = (go + gi) / 2, abs(go - gi)
        l, r = xa27(L, y), xa27(R, y)
        if l is not None and r is not None:
            pc = (l - (r - l) / 2) if sgn < 0 else (r + (r - l) / 2)
            e_ext.append(abs(pc - gc) / gw)
        o, i = xa27(lanes[outer], y), xa27(lanes[inner], y)
        if o is not None and i is not None:
            e_dir.append(abs((o + i) / 2 - gc) / gw)
    return dict(ext=max(e_ext) if len(e_ext) >= MIN_ROWS27 else None,
                dir=max(e_dir) if len(e_dir) >= MIN_ROWS27 else None)


def pred_rows27(lanes, outer, h, cut):
    p = lanes[outer]
    return int(((p[:, 1] >= cut * h) & (p[:, 1] <= BOT25 * h)).sum()) if len(p) else 0


RES27 = {}
for name in PRED25:
    for cut in CUTS25:
        for mis in (False, True):
            ok = {r['f']['key']: r['ok'] for r in RES25[(name, cut, False, mis)]}
            rows = []
            for f in FR25:
                if f['mis'] != mis or not ok.get(f['key']):
                    continue
                lanes, w, h = PRED27[name][f['key']]
                for side, outer, inner, other, sgn in SIDES27:
                    if f['l4'][outer] is None:
                        rows.append(dict(f=f, side=side, gt=False,
                                         extra=pred_rows27(lanes, outer, h, cut) >= MIN_ROWS27))
                        continue
                    e = adj_err27(f, lanes, w, cut, outer, inner, other, sgn)
                    if e == 'narrow':
                        rows.append(dict(f=f, side=side, gt=False, narrow=True,
                                         extra=pred_rows27(lanes, outer, h, cut) >= MIN_ROWS27))
                    elif e != 'short':
                        rows.append(dict(f=f, side=side, gt=True, **e))
            RES27[(name, cut, mis)] = rows


def stat27(rows, kind):
    rs = [r for r in rows if r['gt'] and r['ext'] is not None]
    if kind == 'ext':
        e = [r['ext'] for r in rs]
    elif kind == 'dir':
        e = [r['dir'] for r in rs if r['dir'] is not None]
    else:
        e = [r['dir'] if r['dir'] is not None else r['ext'] for r in rs]
    e = np.array(e)
    return dict(n=len(rs), k=len(e), c=[int((e > t).sum()) for t in ERR25],
                med=float(np.median(e)) if len(e) else float('nan'))


print('\n' + '═' * 118)
print(f'칸 = 측정 장수(옆 차로 수) · 오차 중앙값 · 0.10/0.15/0.25 초과 건수     (자차 안전장치 "기본 전부" 통과 + 정답 옆 차로 폭이 자차의 {ADJ_RATIO27[0]}~{ADJ_RATIO27[1]}배)')
print('직접의 장수가 적으면 = 모델이 그쪽 바깥 차선을 측정 구간에서 5행 이상 내지 않은 경우가 그만큼 있다는 뜻')
print('═' * 118)
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    print(f'\n▶ {tag}')
    print(f'  {"모델":<8}{"방식":<10}' + ''.join(f'{"범위 " + str(c):>26}' for c in CUTS25))
    for name in PRED25:
        for kind, kname in (('ext', '연장(지금)'), ('dir', '직접'), ('mix', '직접+연장')):
            cells = []
            for cut in CUTS25:
                s = stat27(RES27[(name, cut, mis)], kind)
                cells.append(f'{s["k"]:>4}장 {s["med"]:.3f} {s["c"][0]:>3}/{s["c"][1]:>3}/{s["c"][2]:>3}')
            print(f'  {name:<8}{kname:<10}' + ''.join(f'{c:>26}' for c in cells))
    for name in PRED25:
        rs = RES27[(name, CUTS25[-1], mis)]
        no_gt = [r for r in rs if not r['gt'] and not r.get('narrow')]
        nar = [r for r in rs if r.get('narrow')]
        by_side = Counter(r['side'] for r in rs if r['gt'] and r['ext'] is not None)
        print(f'  {name} 범위 {CUTS25[-1]}: 옆 차로 정답 있음 {dict(by_side)} · 폭 검사로 뺀 가짜 옆 차로 {len(nar)}건'
              f'(모델이 바깥 차선 냄 {sum(r["extra"] for r in nar)}) · 정답에 옆 차선 없음 {len(no_gt)}건 중 '
              f'모델이 바깥 차선을 냄 {sum(r["extra"] for r in no_gt)}건 (참고)')

# ── 오차 큰 장면 ──
BEST27 = list(PRED25)[-1]                              # 25번과 같은 모델 (가장 나중 것)
CUT_SHOW27 = 0.68
pool = [r for mis in (False, True) for r in RES27[(BEST27, CUT_SHOW27, mis)] if r['gt'] and r['ext'] is not None]
worst_dir = sorted([r for r in pool if r['dir'] is not None and r['dir'] > ERR25[0]], key=lambda r: -r['dir'])
worst_ext = sorted([r for r in pool if r['ext'] > ERR25[0]], key=lambda r: -r['ext'])


def show27(r, title):
    f = r['f']
    bgr = cv2.imread(str(f['ip'])); vis = bgr.copy()
    s = bgr.shape[1] / f['W']
    for q in f['l4']:
        if q is not None:
            p = np.int32([(x * s, y * s) for x, y in sorted(q[3], key=lambda t: t[1])])
            cv2.polylines(vis, [p], False, (200, 200, 200), 3)
    lanes, w, h = PRED27[BEST27][f['key']]
    for pts, col in zip(lanes, ((255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 128, 0))):
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 4, col, -1)
    L, R = lanes[1], lanes[2]
    sgn = -1 if r['side'] == '왼쪽' else 1
    for y in np.linspace(CUT_SHOW27 * h, BOT25 * h, 12):          # 연장으로 만든 바깥 선 (노란 x)
        l, rr = xa27(L, y), xa27(R, y)
        if l is not None and rr is not None:
            x = l - (rr - l) if sgn < 0 else rr + (rr - l)
            cv2.drawMarker(vis, (int(x), int(y)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.line(vis, (0, int(CUT_SHOW27 * h)), (int(w), int(CUT_SHOW27 * h)), (0, 255, 255), 2)
    plt.figure(figsize=(12, 6.5)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
    d = '-' if r['dir'] is None else f"{r['dir']:.3f}"
    plt.title(f'{title} — {f["key"]} {r["side"]} 옆 차로  연장 {r["ext"]:.3f} · 직접 {d}\n'
              'gray = label  magenta/blue = pred lane 0/3 (outer)  red·green = pred ego  yellow x = extension (current)', fontsize=9)
    plt.show()


print(f'\n[{BEST27} · 범위 {CUT_SHOW27}] 직접 오차 0.10 초과 {len(worst_dir)}건 · 연장 오차 0.10 초과 {len(worst_ext)}건')
for r in worst_dir[:N_SHOW27 // 2]:
    show27(r, '[27] 직접이 틀린 장면')
for r in worst_ext[:N_SHOW27 // 2]:
    show27(r, '[27] 연장이 틀린 장면')


# ============================== 28번 셀 ==============================
# 28. Pi용 ONNX 내보내기 — 2차 재학습 모델(ufld_aihub_aug.pth) → 드라이브 SafeWatch/ufld/ufld_aihub_aug.onnx
# 필요: 1번 셀만 (DRIVE_ROOT). GPU 필요 없음(CPU로 1~2분).
#
# 왜 2차인가: 25번 셀 비교에서 틀어짐(장착이 다른 카메라) 범위 0.68 '기본 전부'를 유일하게 2단계 기준으로 통과(1/0/0).
#   3차(24에폭)는 검증만 조금 낫고 틀어짐은 2/1/1로 나빠져 채택 안 함 (2026-11-09).
# 하는 일: 저장소 scripts/export_ufld_onnx.py 그대로 실행 — 열 방향 출력을 잘라 행 방향만 남기고(실행 코드가 쓰는 것만),
#   --verify로 PyTorch와 ONNX 출력 차이를 확인한다. 차이가 1e-3을 넘으면 실패로 멈춘다.
# 결과: 드라이브 ufld/ufld_aihub_aug.onnx (약 550MB — 파라미터 1.39억 개 × 4바이트) — Pi5에서는 data/models/ufld_aihub_aug.onnx로 두고
#   configs/pi5_ufld.yaml로 실행한다 (docs/hw-handoff-guide.md 2.5절). 아래 SHA-256을 Pi에서 받은 파일과 비교하면 된다.
import sys, subprocess, hashlib
from pathlib import Path

REPO28   = Path('/content/SafeWatch_AI')
BRANCH28 = 'feature/lane-ufld'
SRC28    = DRIVE_ROOT / 'ufld' / 'ufld_aihub_aug.pth'
OUT28    = DRIVE_ROOT / 'ufld' / 'ufld_aihub_aug.onnx'

if not SRC28.exists():
    raise SystemExit(f'{SRC28} 없음 — 26번 셀(2차) 결과 파일')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime'], check=True)

# ── 저장소 (25·26번과 같은 방식) ──
if not REPO28.exists():
    subprocess.run(['git', 'clone', '-q', '-b', BRANCH28,
                    'https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git', str(REPO28)], check=True)
else:
    for cmd in (['fetch', '-q', 'origin', BRANCH28], ['checkout', '-q', BRANCH28], ['reset', '-q', '--hard', f'origin/{BRANCH28}']):
        subprocess.run(['git', '-C', str(REPO28), *cmd], check=True)
print('저장소:', subprocess.run(['git', '-C', str(REPO28), 'log', '-1', '--oneline'], capture_output=True, text=True).stdout.strip())

tmp28 = Path('/content') / OUT28.name                  # 로컬에 먼저 만들고 다 되면 드라이브로 복사 (중간에 끊겨도 반쪽 파일이 안 남게)
p = subprocess.run([sys.executable, '-m', 'scripts.export_ufld_onnx', '--weights', str(SRC28), '--output', str(tmp28),
                    '--config', 'configs/default.yaml', '--verify'], cwd=REPO28, capture_output=True, text=True)
print(p.stdout + p.stderr)
if p.returncode != 0:
    raise SystemExit('변환 실패 — 위 출력 확인')

import shutil
shutil.copy(tmp28, OUT28)
h = hashlib.sha256(OUT28.read_bytes()).hexdigest()
print(f'드라이브 저장: {OUT28} ({OUT28.stat().st_size / 1e6:.0f}MB)')
print(f'SHA-256: {h}')
print('→ 드라이브에서 이 파일을 공유 링크로 넘기고, Pi5에서는 data/models/ufld_aihub_aug.onnx 로 둔다')


# ============================== 29번 셀 ==============================
# 29. 화면에 보이는 앞차를 다 재려면 — 차량마다 offset 오차를 높이별로 재서 방법 비교
# 필요: 25번 셀 실행 (FR25·PRED25·guard25·BOT25·ERR25·x_at·DEV25 사용). GPU 권장(차량 검출), 몇 분.
#
# 왜: 지금은 차량 바닥이 화면 높이 0.68~0.98(측정 구간) 안일 때만 offset을 잰다 → 19번 셀 기준 앞차의 약 14%만 잰다.
#     25번은 '구간 전체 행의 최대 오차'로 차선만 채점했다. 여기서는 실제로 필요한 것 —
#     **차량 한 대 한 대의 offset 오차**(그 차 바닥 높이에서 예측 차로 중심 대비 위치 − 정답 차로 기준 위치)를 높이별로 잰다.
#
# 비교하는 방법 (전부 같은 모델·같은 차량·같은 정답)
#   지금      : 판정 구간 = 측정 구간 = [위끝, 0.98]. 차 바닥이 구간 밖이면 안 잰다. 위끝 0.68 / 0.64 / 0.60
#   ① 직접    : 판정(안전장치)은 가까운 구간 [0.68, 0.98]에서만 하고, 통과한 프레임이면 차 바닥 높이의 예측 차선으로 바로 잰다
#              (모델은 화면 높이 0.42까지 차선을 낸다). 높이별 오차 표가 곧 '높이별로 믿는 정도'(가중치·허용치)의 근거
#   ① 연장    : 판정은 같고, 가까운 구간의 예측 차선 점들을 직선으로 맞춰 차 바닥 높이까지 늘려서 잰다
#              (원근에서 직선 도로의 차선은 화면에서도 직선. 곡선이면 틀어진다 — 표로 확인)
#   ② 시간 평균: 정답 사진이 1초 간격이라 연속 영상 평균은 여기서 직접 못 잰다. 대신 높이별 차선 오차가 1초 뒤에도
#              같은 방향으로 남는지(상관) 본다 — 상관이 크면 '고정 치우침'이라 평균 내도 안 줄고, 작으면 평균이 줄여 준다.
#   ③ 먼 곳 재학습: 학습이 필요해 이 셀에서는 안 한다. 이 표에서 어느 높이가 얼마나 틀리는지 보고 정한다.
#
# 오차 기준(ERR25): 0.10 = 사행 진폭 기준(risk.weaving.min_amplitude_ratio) · 0.15 = 표류 · 0.25 = 걸침
import numpy as np, cv2
from collections import defaultdict

MODEL29  = '재학습2'                                  # 채택 모델 (Pi용 ONNX와 같은 가중치)
NEAR29   = 0.68                                      # ①의 판정 구간 위끝 (지금 설정)
CUTS29   = [0.68, 0.64, 0.60]                        # '지금' 방식의 위끝 후보
BINS29   = [0.42, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.80, 0.98]   # 차 바닥 높이 구간 (화면 높이 비율)
CONF29   = 0.25                                      # 차량 검출 신뢰도 하한 (19번 셀과 같음)
LANES29  = 1.5                                       # 정답 offset |값| ≤ 이 값인 차만 (자차·옆 차로)
N_SHOW29 = 3

if MODEL29 not in PRED25:
    raise SystemExit(f'25번 셀에 {MODEL29} 결과가 없음 — MODELS25·모델 파일 확인')

# ── 차량 검출 (COCO 가중치, 차·버스·트럭) ──
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO
det29 = YOLO('yolov8n.pt')
if 'VEH29' not in globals():
    VEH29, t0 = {}, time.time()
    for i in range(0, len(FR25), 32):
        batch = FR25[i:i + 32]
        rs = det29.predict([str(f['ip']) for f in batch], imgsz=640, conf=CONF29, classes=[2, 5, 7],
                           verbose=False, device=0 if DEV25 == 'cuda' else 'cpu')
        for f, r in zip(batch, rs):
            VEH29[f['key']] = r.boxes.xyxy.cpu().numpy()
    print(f'차량 검출 {len(VEH29)}장 {time.time() - t0:.0f}초 · 차량 {sum(len(v) for v in VEH29.values())}대')


def fit_line29(pts, y0, y1):
    """y0~y1 안의 점으로 x = a·y + b 직선. 점 4개 미만이면 None"""
    q = [(x, y) for x, y in pts if y0 <= y <= y1]
    if len(q) < 4:
        return None
    ys, xs = np.array([p[1] for p in q]), np.array([p[0] for p in q])
    return np.polyfit(ys, xs, 1) if np.ptp(ys) > 1 else None


def offset29(xl, xr, cx):
    return None if xl is None or xr is None or xr - xl <= 1 else (cx - (xl + xr) / 2) / (xr - xl)


# ── 차량별 기록 ──
REC29 = []          # 차량 한 대 = 한 줄
LANE29 = []         # 프레임 · 높이별 차선 중심 오차(부호 있음) — ②용
for mis in (False, True):
    g_near, g_cut, cur = None, {}, None
    for f in FR25:
        if f['mis'] != mis:
            continue
        if f['seq'] != cur:
            g_near, cur = guard25(NEAR29, False), f['seq']
            g_cut = {c: guard25(c, False) for c in CUTS29}
        L, R, w, h = PRED25[MODEL29][f['key']]
        t = f['idx'] / 30.0
        ok_near = g_near.check(L, R, w, h, t).valid
        ok_cut = {c: g_cut[c].check(L, R, w, h, t).valid for c in CUTS29}
        s = w / f['W']
        gl_pts = [(x * s, y * s) for x, y in f['l4'][1][3]]
        gr_pts = [(x * s, y * s) for x, y in f['l4'][2][3]]
        fl = fit_line29(L, NEAR29 * h, BOT25 * h) if len(L) else None
        fr = fit_line29(R, NEAR29 * h, BOT25 * h) if len(R) else None
        # ②용 — 높이마다 예측 차로 중심 오차 (판정 통과 프레임만)
        if ok_near:
            for yr in np.arange(0.44, 0.99, 0.02):
                y = yr * h
                gl, gr = x_at(gl_pts, y), x_at(gr_pts, y)
                pl, pr = (x_at(L, y) if len(L) >= 2 else None), (x_at(R, y) if len(R) >= 2 else None)
                if None not in (gl, gr, pl, pr) and gr - gl > 1:
                    LANE29.append(dict(seq=f['seq'], idx=f['idx'], mis=mis, yr=round(float(yr), 2),
                                       e=((pl + pr) / 2 - (gl + gr) / 2) / (gr - gl)))
        for x1, y1, x2, y2 in VEH29.get(f['key'], []):
            cx, yb = (x1 + x2) / 2, float(y2)
            if yb > BOT25 * h:
                continue
            gt = offset29(x_at(gl_pts, yb), x_at(gr_pts, yb), cx)
            if gt is None or abs(gt) > LANES29:
                continue
            direct = offset29(x_at(L, yb) if len(L) >= 2 else None, x_at(R, yb) if len(R) >= 2 else None, cx)
            ext = offset29(np.polyval(fl, yb) if fl is not None else None,
                           np.polyval(fr, yb) if fr is not None else None, cx)
            row = dict(f=f, mis=mis, yr=yb / h, gt=gt, ego=abs(gt) <= 0.5, box=(x1, y1, x2, y2))
            for c in CUTS29:                                   # 지금 방식
                row[f'지금 {c}'] = direct if (ok_cut[c] and c * h <= yb) else None
            row['① 직접'] = direct if ok_near else None
            row['① 연장'] = ext if ok_near else None
            REC29.append(row)

METHODS29 = [f'지금 {c}' for c in CUTS29] + ['① 직접', '① 연장']
print(f'\n정답 차로 기준 위치를 알 수 있는 차량 {len(REC29)}대 (검증 {sum(not r["mis"] for r in REC29)} · '
      f'틀어짐 {sum(r["mis"] for r in REC29)}) — 자차·옆 차로(|offset| ≤ {LANES29}), 정답 차선이 그 높이까지 있는 차만')


def stat29(rows, m):
    e = np.array([abs(r[m] - r['gt']) for r in rows if r[m] is not None])
    n = len(rows)
    if len(e) == 0:
        return f'{0:>4}% {"-":>6} {"-":>11}'
    c = [int((e > t).sum()) for t in ERR25]
    return f'{len(e) / max(n, 1) * 100:>4.0f}% {np.median(e):>6.3f} {c[0]:>3}/{c[1]:>3}/{c[2]:>3}'


print('\n' + '═' * 120)
print('칸 = 잰 비율(그 높이 차량 중) · 오차 중앙값 · 0.10/0.15/0.25 초과 대수     오차 = |예측 offset − 정답 offset| (차로 폭 단위)')
print('═' * 120)
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    rows_all = [r for r in REC29 if r['mis'] == mis]
    print(f'\n▶ {tag} — 차 바닥 높이별 (위 = 멀리)')
    print(f'  {"높이":<11}{"대수":>5}' + ''.join(f'{m:>24}' for m in METHODS29))
    for a, b in zip(BINS29[:-1], BINS29[1:]):
        rows = [r for r in rows_all if a <= r['yr'] < b]
        print(f'  {a:.2f}~{b:.2f}{len(rows):>6}' + ''.join(f'{stat29(rows, m):>24}' for m in METHODS29))
    for sub, cond in (('전체', lambda r: True), ('자차 차로만', lambda r: r['ego'])):
        rows = [r for r in rows_all if cond(r)]
        print(f'  {sub:<10}{len(rows):>6}' + ''.join(f'{stat29(rows, m):>24}' for m in METHODS29))

# ── ② 시간 평균이 도울까 — 1초 뒤 같은 높이 차선 오차의 상관 ──
print('\n' + '═' * 120)
print('② 높이별 차선 중심 오차(부호 있음)의 1초 뒤 상관 · 오차 크기  — 상관이 1에 가까우면 고정 치우침(평균으로 안 줄어듦)')
print('═' * 120)
by29 = defaultdict(dict)
for d in LANE29:
    by29[(d['mis'], d['seq'], d['yr'])][d['idx']] = d['e']
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    print(f'\n▶ {tag}')
    print(f'  {"높이":<11}{"표본":>6}{"|오차| 중앙":>11}{"p90":>8}{"1초 뒤 상관":>12}{"쌍":>6}')
    for a, b in zip(BINS29[:-1], BINS29[1:]):
        es, pa, pb = [], [], []
        for (m_, sq, yr), d in by29.items():
            if m_ != mis or not (a <= yr < b):
                continue
            es += list(d.values())
            for i, e in d.items():
                if i + 30 in d:
                    pa.append(e); pb.append(d[i + 30])
        if not es:
            continue
        es = np.abs(np.array(es))
        corr = np.corrcoef(pa, pb)[0, 1] if len(pa) >= 10 else float('nan')
        print(f'  {a:.2f}~{b:.2f}{len(es):>7}{np.median(es):>11.3f}{np.percentile(es, 90):>8.3f}{corr:>12.2f}{len(pa):>6}')

# ── 그림 — ① 직접에서 오차가 큰 먼 차량 ──
bad29 = sorted([r for r in REC29 if r['① 직접'] is not None and r['yr'] < NEAR29
                and abs(r['① 직접'] - r['gt']) > ERR25[0]], key=lambda r: -abs(r['① 직접'] - r['gt']))
print(f'\n① 직접 · 판정 구간 위(먼 차)에서 0.10 초과 {len(bad29)}대 — 최대 {N_SHOW29}대 표시')
for r in bad29[:N_SHOW29]:
    f = r['f']; L, R, w, h = PRED25[MODEL29][f['key']]
    vis = cv2.imread(str(f['ip'])); s = w / f['W']
    for k in (1, 2):
        q = np.int32([(x * s, y * s) for x, y in sorted(f['l4'][k][3], key=lambda t: t[1])])
        cv2.polylines(vis, [q], False, (255, 255, 0), 3)
    for ln, col in ((L, (0, 0, 255)), (R, (0, 255, 0))):
        for x, y in ln:
            cv2.circle(vis, (int(x), int(y)), 4, col, -1)
    x1, y1, x2, y2 = map(int, r['box'])
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.line(vis, (0, int(NEAR29 * h)), (int(w), int(NEAR29 * h)), (0, 255, 255), 2)
    ext = '-' if r['① 연장'] is None else f"{r['① 연장']:+.2f}"
    plt.figure(figsize=(12, 6.5)); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.title(f'[29] {f["key"]}  car bottom {r["yr"]:.2f}  gt {r["gt"]:+.2f} · direct {r["① 직접"]:+.2f} · ext {ext}\n'
              'cyan = label ego lanes  red/green = pred  magenta = car  yellow line = guard band top', fontsize=9)
    plt.show()


# ============================== 30번 셀 ==============================
# 30. 먼 곳 재학습 (5차) — 2차와 같은 학습에 '먼 행 가중'만 더한다
# 필요: 1번 셀만 (DRIVE_ROOT). **GPU 런타임**. 드라이브에 2차 학습 데이터 ufld_aihub2.tar(26번 2차 실행 때 만든 백업)가 있어야 한다.
#
# 왜: 29번 셀 — 차 바닥이 화면 높이 0.58보다 먼 차는 offset 오차 0.10 초과가 7~20%라 지금은 안 잰다(앞차 약 59%를 못 잼).
# 하는 일: 저장소 scripts/train_ufld.py 그대로 실행. 바뀐 것은 configs/ufld_train.yaml train.far_rows 하나 —
#   행 앵커가 화면 높이 0.58보다 위(먼 쪽)인 행의 위치 손실에 FAR_W30배 가중. 에폭 선택 손실도 같은 가중으로 잰다.
#   그 외(데이터·기하 증강·12에폭)는 채택 모델 2차와 같다 → 차이는 가중 하나 때문.
# 결과: 드라이브 ufld/ufld_aihub_far.pth (2차 ufld_aihub_aug.pth는 그대로). A100 약 45분. 끊기면 그대로 다시 실행 = 이어 하기.
# 다음: 25번 셀 MODELS25에 '재학습5' 줄을 넣고 25 → 29 → 31번 셀.
import os, sys, subprocess, tarfile, time, json
from pathlib import Path

FAR_W30   = 3.0                                      # 먼 행 가중 [시험값 — 31번 결과로 정한다]
FAR_BELOW30 = 0.58                                   # 지금 측정 위끝 (configs/default.yaml measure_range_ratio)
REPO30    = Path('/content/SafeWatch_AI')
BRANCH30  = 'feature/lane-ufld'
UFLD30    = DRIVE_ROOT / 'ufld'
PRETRAIN30 = UFLD30 / 'culane_res18.pth'
DATA30    = Path('/content/ufld_aihub2')
BACKUP30  = DRIVE_ROOT / 'ufld_aihub2.tar'           # 2차 학습 데이터 (형식 2, 옆 차선 정리 전 = 2차와 같은 데이터)
OUT30     = UFLD30 / 'ufld_aihub_far.pth'
EPOCHS30  = 12
CPU30     = os.cpu_count() or 2

try:
    import torch
    assert torch.cuda.is_available()
    print('GPU:', torch.cuda.get_device_name(0), '· CPU', CPU30)
except AssertionError:
    raise SystemExit('GPU 런타임이 아닙니다 — 런타임 > 런타임 유형 변경 > GPU')

# ── 저장소 (25·26번과 같은 방식) ──
if not REPO30.exists():
    subprocess.run(['git', 'clone', '-q', '-b', BRANCH30,
                    'https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git', str(REPO30)], check=True)
else:
    for cmd in (['fetch', '-q', 'origin', BRANCH30], ['checkout', '-q', BRANCH30], ['reset', '-q', '--hard', f'origin/{BRANCH30}']):
        subprocess.run(['git', '-C', str(REPO30), *cmd], check=True)
print('저장소:', subprocess.run(['git', '-C', str(REPO30), 'log', '-1', '--oneline'], capture_output=True, text=True).stdout.strip())
if 'far_row_weights' not in (REPO30 / 'scripts/train_ufld.py').read_text():
    raise SystemExit('저장소에 먼 행 가중 코드가 아직 없음 — feature/lane-ufld 푸시 확인')

# ── 데이터 (2차 백업에서 복원) ──
if not (DATA30 / 'manifest.json').exists():
    if not BACKUP30.exists():
        raise SystemExit(f'{BACKUP30} 없음 — 2차 학습 데이터 백업. 드라이브에서 이름 확인')
    print(f'데이터 복원 — {BACKUP30.name} ({BACKUP30.stat().st_size / 1e9:.1f}GB)')
    t = time.time()
    with tarfile.open(BACKUP30) as tf:
        tf.extractall(DATA30.parent)
    print(f'  {time.time() - t:.0f}초')
man30 = json.loads((DATA30 / 'manifest.json').read_text())
if man30.get('format') != 2:
    raise SystemExit(f'{DATA30}는 옛 형식 — 2차 데이터가 아님')
print('학습', man30['splits']['train'], '· 선택용', man30['splits']['val'])

# ── 학습 ──
if not PRETRAIN30.exists():
    raise SystemExit(f'{PRETRAIN30} 없음 — 15번 셀에서 받은 CULane 가중치')
import yaml
tc30 = yaml.safe_load((REPO30 / 'configs/ufld_train.yaml').read_text())
tc30['train']['num_workers'] = min(int(tc30['train']['num_workers']), CPU30)
tc30['train']['epochs'] = EPOCHS30
tc30['train']['far_rows'] = {'below': FAR_BELOW30, 'weight': FAR_W30}
cfg30 = Path('/content/ufld_train_far.yaml')
cfg30.write_text(yaml.safe_dump(tc30, allow_unicode=True))
print(f"\n학습 — 배치 {tc30['train']['batch_size']} · 에폭 {EPOCHS30} · 먼 행 가중 {tc30['train']['far_rows']} · 저장 {OUT30}")
OUT30.parent.mkdir(parents=True, exist_ok=True)
p = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.train_ufld', '--data', str(DATA30), '--weights', str(PRETRAIN30),
                      '--out', str(OUT30), '--train-config', str(cfg30), '--resume'],
                     cwd=REPO30, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
for line in p.stdout:
    print(line, end='')
if p.wait() != 0:
    raise SystemExit(f'실패 (종료 코드 {p.returncode}) — 위 출력 확인')

hist = json.loads(OUT30.with_name(OUT30.stem + '_history.json').read_text())
key30 = 'val_aug_loss' if tc30['train']['select_by'] == 'aug' else 'val_loss'
best = min(hist, key=lambda h: h[key30])
print('\n' + '═' * 80)
print(f'{"에폭":>4} {"학습 손실":>10} {"선택용 손실":>11} {"증강 건 선택용":>14}   (먼 행 가중 손실이라 2차 숫자와 직접 비교 안 됨)')
for h in hist:
    tl = '-' if h['train_loss'] is None else f"{h['train_loss']:.4f}"
    va = '-' if h.get('val_aug_loss') is None else f"{h['val_aug_loss']:.4f}"
    print(f"{h['epoch']:>4} {tl:>10} {h['val_loss']:>11.4f} {va:>14}" + ('  ← 최저' if h is best else ''))
print(f'저장: {OUT30}' if OUT30.exists() else '어느 에폭도 사전학습보다 나아지지 않아 저장 안 됨')
print("다음: 25번 셀 MODELS25에 '재학습5': UFLD25 / 'ufld_aihub_far.pth' 넣고 25 → 29 → 31번")
print('═' * 80)


# ============================== 31번 셀 ==============================
# 31. 먼 곳 재학습(30번) 효과 — 차량마다 offset 오차를 높이별로, 재학습2 vs 재학습5
# 필요: 25번 셀(MODELS25에 '재학습5' 넣고) → 29번 셀을 이 세션에서 먼저 실행 (VEH29·offset29·guard25 등을 그대로 쓴다)
#
# 방식은 지금 실행 코드와 같다: 판정(안전장치)은 0.68~0.98, 통과한 프레임이면 차 바닥 높이의 예측 차선으로 바로 잰다.
# 칸 = 잰 비율 · 오차 중앙값 · 0.10/0.15/0.25 초과 대수.  보는 것:
#   · 0.58 위(먼 차) 행 — 재학습5의 0.10 초과가 줄었나  → 줄었으면 측정 위끝을 더 올릴 근거
#   · 0.58 아래(지금 재는 차) 행 — 재학습5가 나빠지지 않았나 → 나빠졌으면 채택 불가
import numpy as np

MODELS31 = ['재학습2', '재학습5']
NEAR31   = 0.68
SPLIT31  = 0.58                                       # 지금 측정 위끝

if 'VEH29' not in globals() or 'offset29' not in globals():
    raise SystemExit('29번 셀을 이 세션에서 먼저 실행하세요 (차량 검출 결과 VEH29를 쓴다)')
miss31 = [m for m in MODELS31 if m not in PRED25]
if miss31:
    raise SystemExit(f'25번 셀에 {miss31} 결과가 없음 — MODELS25에 넣고 25번부터 다시')


def rec31(model):
    """차량 한 대 = (틀어짐?, 차 바닥 높이 비율, 오차 또는 None(판정 보류))"""
    out = []
    for mis in (False, True):
        g, cur = None, None
        for f in FR25:
            if f['mis'] != mis:
                continue
            if f['seq'] != cur:
                g, cur = guard25(NEAR31, False), f['seq']
            L, R, w, h = PRED25[model][f['key']]
            ok = g.check(L, R, w, h, f['idx'] / 30.0).valid
            s = w / f['W']
            gl = [(x * s, y * s) for x, y in f['l4'][1][3]]
            gr = [(x * s, y * s) for x, y in f['l4'][2][3]]
            for x1, y1, x2, y2 in VEH29.get(f['key'], []):
                cx, yb = (x1 + x2) / 2, float(y2)
                if yb > BOT25 * h:
                    continue
                gt = offset29(x_at(gl, yb), x_at(gr, yb), cx)
                if gt is None or abs(gt) > LANES29:
                    continue
                d = offset29(x_at(L, yb) if len(L) >= 2 else None, x_at(R, yb) if len(R) >= 2 else None, cx) if ok else None
                out.append((mis, yb / h, None if d is None else abs(d - gt)))
    return out


def cell31(rows):
    if not rows:
        return f'{"-":>27}'
    e = np.array([r[2] for r in rows if r[2] is not None])
    if len(e) == 0:
        return f'{"0%":>5} {"-":>6} {"-":>13}'
    c = [int((e > t).sum()) for t in ERR25]
    return f'{len(e) / len(rows) * 100:>4.0f}% {np.median(e):>6.3f} {c[0]:>4}/{c[1]:>4}/{c[2]:>4}'


R31 = {m: rec31(m) for m in MODELS31}
print('═' * 100)
print(f'칸 = 잰 비율 · 오차 중앙값 · 0.10/0.15/0.25 초과 대수   (판정 {NEAR31}~{BOT25}, 차 바닥 높이에서 바로 잼)')
print('═' * 100)
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    print(f'\n▶ {tag}')
    print(f'  {"차 바닥 높이":<13}{"대수":>6}' + ''.join(f'{m:>28}' for m in MODELS31))
    groups = [(f'{a:.2f}~{b:.2f}', lambda y, a=a, b=b: a <= y < b) for a, b in zip(BINS29[:-1], BINS29[1:])]
    groups += [(f'{SPLIT31} 위 (먼 차)', lambda y: y < SPLIT31), (f'{SPLIT31} 이하 (지금 잼)', lambda y: y >= SPLIT31)]
    for name, cond in groups:
        rows = {m: [r for r in R31[m] if r[0] == mis and cond(r[1])] for m in MODELS31}
        n = len(rows[MODELS31[0]])
        print(f'  {name:<13}{n:>6}' + ''.join(f'{cell31(rows[m]):>28}' for m in MODELS31))
print('\n같은 차량·같은 정답 — 대수는 모델끼리 같고, 잰 비율 차이는 판정 통과율 차이다.')


# ============================== 32번 셀 ==============================
# 32. 먼 곳 확대 추론 — 소실점 기준으로 화면을 확대해 차선 모델에 한 번 더 넣으면 먼 차 offset이 정확해지나
# 필요: 25번 → 29번 셀을 이 세션에서 먼저 실행 (PRED25·VEH29·offset29·fit_line29·guard25 등). GPU, 모델 하나당 2~3분.
#
# 왜: 30·31번(먼 행 가중 재학습)은 먼 곳 오차를 조금만 줄였다. 남은 가설 = 해상도 — 모델 입력(1600×320)에서
#     먼 곳 차로는 몇십 픽셀뿐이다. 학습 없이 입력만 키워서 이 가설을 직접 확인한다.
# 방법: 1) 원본으로 한 번 추론(지금과 같음) → 가까운 구간(0.68~0.98) 좌우 차선을 직선으로 맞춰 교점 = 소실점
#       2) 소실점을 중심으로 Z배 확대한 화면을 같은 모델에 다시 넣는다. 소실점을 지나는 직선은 이 확대에서 그대로 직선이라
#          확대 화면도 '같은 도로를 더 멀리서 본 장면'처럼 보인다(학습 때 확대 0.8~1.2 증강과 같은 종류의 변환).
#          원본 사진(1920×1080)을 1600폭으로 줄여 넣는 지금과 달리, 확대하면 먼 곳의 원본 픽셀이 실제로 더 들어간다.
#       3) 확대 화면의 예측 점을 원래 좌표로 되돌려, 차 바닥 높이에서 offset을 잰다. 판정(안전장치)은 1)의 결과 그대로.
#       실행에 넣으면 프레임마다 차선 추론이 2번 — NPU 예산 문제는 결과를 보고 따진다.
# 비교: 원본(① 직접 = 지금 실행 코드) vs 확대 Z배. 칸 = 잰 비율 · 오차 중앙값 · 0.10/0.15/0.25 초과 대수 (31번과 같은 정의)
import numpy as np, cv2, torch, time

MODEL32 = '재학습2'                                   # 채택 모델
ZOOMS32 = [1.5, 2.0]
NEAR32  = 0.68
VP_X32  = (0.25, 0.75)                               # 소실점이 이 범위(화면 비율) 밖이면 그 프레임은 확대 안 함(차선 이상)
VP_Y32  = (0.25, 0.55)

for need in ('VEH29', 'offset29', 'fit_line29', 'PRED25'):
    if need not in globals():
        raise SystemExit('25번 → 29번 셀을 이 세션에서 먼저 실행하세요')
if MODEL32 not in PRED25:
    raise SystemExit(f'25번 셀에 {MODEL32} 결과가 없음')


def vp32(L, R, h):
    """가까운 구간 좌우 차선 직선의 교점 (x, y) 또는 None"""
    fl = fit_line29(L, NEAR32 * h, BOT25 * h) if len(L) else None
    fr = fit_line29(R, NEAR32 * h, BOT25 * h) if len(R) else None
    if fl is None or fr is None or abs(fl[0] - fr[0]) < 1e-6:
        return None
    y = (fr[1] - fl[1]) / (fl[0] - fr[0])            # x = a·y + b 두 직선의 교점
    return float(np.polyval(fl, y)), float(y)


VP32 = {}
for f in FR25:
    L, R, w, h = PRED25[MODEL32][f['key']]
    v = vp32(L, R, h)
    if v is not None and VP_X32[0] * w <= v[0] <= VP_X32[1] * w and VP_Y32[0] * h <= v[1] <= VP_Y32[1] * h:
        VP32[f['key']] = v
print(f'소실점 구함 {len(VP32)}/{len(FR25)}장 (나머지는 확대 안 함 → 그 프레임 먼 차는 "못 잼")')


@torch.no_grad()
def infer_zoom32(path, z):
    """소실점 중심 z배 확대 화면으로 추론 → 원래 좌표로 되돌린 (좌, 우) 차선 점"""
    net = UFLDv2RowNet(SPEC25, grid_col=100, num_col=81, fc_norm=True)
    load_weights(net, path)
    net.keep_rows_only()
    net = net.eval().to(DEV25)
    keys = [f for f in FR25 if f['key'] in VP32]
    out, t0 = {}, time.time()
    for i in range(0, len(keys), BS25):
        batch = keys[i:i + BS25]
        imgs = []
        for f in batch:
            b = cv2.imread(str(f['ip']))
            hh, ww = b.shape[:2]
            s = ww / PRED25[MODEL32][f['key']][2]      # 25번 예측 좌표 → 이 사진 좌표 (보통 1)
            vx, vy = VP32[f['key']][0] * s, VP32[f['key']][1] * s
            M = np.float32([[z, 0, (1 - z) * vx], [0, z, (1 - z) * vy]])
            imgs.append((cv2.warpAffine(b, M, (ww, hh), flags=cv2.INTER_LINEAR), vx, vy, s))
        x = torch.from_numpy(np.concatenate([preprocess(b, SPEC25) for b, *_ in imgs])).to(DEV25)
        loc, ex = net(x)
        loc, ex = loc.float().cpu().numpy(), ex.float().cpu().numpy()
        for j, (f, (b, vx, vy, s)) in enumerate(zip(batch, imgs)):
            hh, ww = b.shape[:2]
            lanes = decode_rows(loc[j], ex[j], SPEC25, ww, hh)
            back = []
            for k in (1, 2):
                p = lanes[k]
                q = np.c_[vx + (p[:, 0] - vx) / z, vy + (p[:, 1] - vy) / z] / s if len(p) else p
                back.append(q[np.argsort(q[:, 1])] if len(q) else q)
            out[f['key']] = tuple(back)
    print(f'  확대 {z}배: {len(out)}장 {time.time() - t0:.0f}초')
    del net
    if DEV25 == 'cuda':
        torch.cuda.empty_cache()
    return out


if 'ZPRED32' not in globals():
    ZPRED32 = {}
for z in ZOOMS32:
    if z not in ZPRED32:
        ZPRED32[z] = infer_zoom32(MODELS25[MODEL32], z)


def xa32(pts, y):
    """점 범위 안에서만 (확대 화면은 원래 화면의 위쪽 일부만 덮는다 — 밖으로 연장하지 않음)"""
    if len(pts) < 2 or not (pts[0, 1] <= y <= pts[-1, 1]):
        return None
    return x_at(pts, y)


METHODS32 = ['원본(지금)'] + [f'확대 {z}배' for z in ZOOMS32]
REC32 = []
for mis in (False, True):
    g, cur = None, None
    for f in FR25:
        if f['mis'] != mis:
            continue
        if f['seq'] != cur:
            g, cur = guard25(NEAR32, False), f['seq']
        L, R, w, h = PRED25[MODEL32][f['key']]
        ok = g.check(L, R, w, h, f['idx'] / 30.0).valid
        s = w / f['W']
        gl = [(x * s, y * s) for x, y in f['l4'][1][3]]
        gr = [(x * s, y * s) for x, y in f['l4'][2][3]]
        for x1, y1, x2, y2 in VEH29.get(f['key'], []):
            cx, yb = (x1 + x2) / 2, float(y2)
            if yb > BOT25 * h:
                continue
            gt = offset29(x_at(gl, yb), x_at(gr, yb), cx)
            if gt is None or abs(gt) > LANES29:
                continue
            row = dict(mis=mis, yr=yb / h)
            d = offset29(x_at(L, yb) if len(L) >= 2 else None, x_at(R, yb) if len(R) >= 2 else None, cx) if ok else None
            row['원본(지금)'] = None if d is None else abs(d - gt)
            for z in ZOOMS32:
                zl = ZPRED32[z].get(f['key'])
                d = offset29(xa32(zl[0], yb), xa32(zl[1], yb), cx) if (ok and zl is not None) else None
                row[f'확대 {z}배'] = None if d is None else abs(d - gt)
            REC32.append(row)


def cell32(rows, m):
    if not rows:
        return f'{"-":>27}'
    e = np.array([r[m] for r in rows if r[m] is not None])
    if len(e) == 0:
        return f'{"0%":>5} {"-":>6} {"-":>14}'
    c = [int((e > t).sum()) for t in ERR25]
    return f'{len(e) / len(rows) * 100:>4.0f}% {np.median(e):>6.3f} {c[0]:>4}/{c[1]:>4}/{c[2]:>4}'


print('\n' + '═' * 110)
print(f'[{MODEL32}] 칸 = 잰 비율 · 오차 중앙값 · 0.10/0.15/0.25 초과 대수   (판정은 전부 원본 추론 {NEAR32}~{BOT25})')
print('확대 화면은 원래 화면 위쪽 일부만 덮는다 → 가까운 차는 확대 방식으로 못 잴 수 있음(실제로는 원본으로 잼)')
print('═' * 110)
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    print(f'\n▶ {tag}')
    print(f'  {"차 바닥 높이":<13}{"대수":>6}' + ''.join(f'{m:>28}' for m in METHODS32))
    groups = [(f'{a:.2f}~{b:.2f}', lambda y, a=a, b=b: a <= y < b) for a, b in zip(BINS29[:-1], BINS29[1:])]
    groups += [('0.58 위 (먼 차)', lambda y: y < 0.58), ('0.54~0.58', lambda y: 0.54 <= y < 0.58)]
    for name, cond in groups:
        rows = [r for r in REC32 if r['mis'] == mis and cond(r['yr'])]
        print(f'  {name:<13}{len(rows):>6}' + ''.join(f'{cell32(rows, m):>28}' for m in METHODS32))
print('\n보는 법: 먼 차(0.58 위) 행에서 확대가 원본보다 0.10 초과가 크게 줄면 → 해상도가 원인. 비슷하면 → 해상도 아님')


# ============================== 33번 셀 ==============================
# 33. 확대 효과를 '같은 차끼리' 다시 비교 — 32번 결과(REC32)만 다시 센다. 추론 없음, 몇 초.
# 필요: 32번 셀을 이 세션에서 먼저 실행.
#
# 왜: 32번 표는 방법마다 잰 차가 다르다(먼 차 원본 69% vs 확대 77~79%). 확대가 원본이 못 잰 어려운 차까지 재서
#     초과 대수가 늘었을 수 있다. 그래서 ① 둘 다 잰 차만 놓고 오차를 비교하고,
#     ② '원본으로 재고, 원본이 못 잰 차만 확대로 보충'하는 방식이 원본보다 나쁜지 본다.
import numpy as np

if 'REC32' not in globals():
    raise SystemExit('32번 셀을 이 세션에서 먼저 실행하세요')

GROUPS33 = [(f'{a:.2f}~{b:.2f}', lambda y, a=a, b=b: a <= y < b) for a, b in zip(BINS29[:-1], BINS29[1:]) if a < 0.62]
GROUPS33 += [('0.58 위 (먼 차)', lambda y: y < 0.58)]


def cnt33(e):
    e = np.asarray(e)
    if len(e) == 0:
        return f'{"-":>6} {"-":>14}'
    c = [int((e > t).sum()) for t in ERR25]
    return f'{np.median(e):>6.3f} {c[0]:>4}/{c[1]:>4}/{c[2]:>4}'


print('═' * 112)
print('① 둘 다 잰 차만 — 칸 = 오차 중앙값 · 0.10/0.15/0.25 초과 대수 (원본 | 확대)')
print('═' * 112)
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    print(f'\n▶ {tag}')
    print(f'  {"차 바닥 높이":<13}' + ''.join(f'{f"확대 {z}배: 대수":>14}{"원본":>22}{"확대":>22}' for z in ZOOMS32))
    for name, cond in GROUPS33:
        line = f'  {name:<13}'
        for z in ZOOMS32:
            m = f'확대 {z}배'
            rs = [r for r in REC32 if r['mis'] == mis and cond(r['yr']) and r['원본(지금)'] is not None and r[m] is not None]
            line += f'{len(rs):>14}{cnt33([r["원본(지금)"] for r in rs]):>22}{cnt33([r[m] for r in rs]):>22}'
        print(line)

print('\n' + '═' * 112)
print('② 원본으로 재고, 원본이 못 잰 차만 확대로 보충 — 칸 = 잰 비율 · 오차 중앙값 · 0.10/0.15/0.25 초과 대수')
print('═' * 112)
for mis, tag in ((False, '검증'), (True, '틀어짐')):
    print(f'\n▶ {tag}')
    print(f'  {"차 바닥 높이":<13}{"대수":>6}{"원본(지금)":>28}' + ''.join(f'{f"원본+확대 {z}배 보충":>28}' for z in ZOOMS32))
    for name, cond in GROUPS33:
        rows = [r for r in REC32 if r['mis'] == mis and cond(r['yr'])]
        if not rows:
            print(f'  {name:<13}{0:>6}')
            continue
        line = f'  {name:<13}{len(rows):>6}'
        for m in [None] + [f'확대 {z}배' for z in ZOOMS32]:
            e = [r['원본(지금)'] if r['원본(지금)'] is not None or m is None else r[m] for r in rows]
            e = [v for v in e if v is not None]
            line += f'{f"{len(e) / len(rows) * 100:.0f}%":>6}{cnt33(e):>22}'
        print(line)
print('\n보는 법: ①에서 확대 쪽 초과가 원본보다 적으면 → 같은 차라도 확대가 더 정확(해상도가 원인). 많거나 같으면 → 해상도 아님.')
print('        ②에서 보충이 잰 비율을 올리면서 초과 대수가 "늘어난 대수만큼만" 늘면 → 못 재던 차를 원본 수준으로 더 재는 것')
