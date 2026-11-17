"""BDD100K 주행 영상 일부만 받기 — 단서 분포(scripts/cue_distribution.py)의 정상 주행 기준용.

공식 다운로드 페이지(http://bdd-data.berkeley.edu/download.html)의 영상 묶음(Video Parts) 링크는 2026-11-17 기준
404이고, 영상 10만 개를 한 파일로 묶은 zip(약 1.97TB)만 받을 수 있다. 서버가 부분 받기(HTTP Range)를 지원해서,
zip의 목차(central directory)만 읽고 **고른 영상만** 그 위치에서 잘라 받는다.

고르는 기준은 라벨 파일(bdd100k_labels.zip, 약 190MB 중 한 json)의 영상별 속성 —
시간대(timeofday), 날씨(weather), 장소(scene). 기본은 낮·맑음.

라이선스 — 교육·연구·비영리 목적 사용 허가(다운로드 페이지 License). 영상은 data/(커밋 금지)에만 둔다.

사용법 (src.* import 없음 — 그냥 실행해도 됨)
    python scripts/fetch_bdd100k.py --count 200 --out data/raw/bdd100k
    python scripts/fetch_bdd100k.py --count 50 --timeofday daytime --weather clear --scene highway --out data/raw/bdd100k_hw
    python scripts/fetch_bdd100k.py --list-only --count 20      # 받지 않고 고른 목록만

이미 받은 파일은 건너뛴다(중간에 끊겨도 다시 실행하면 이어서 받는다).
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import urllib.request
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "http://128.32.162.150/bdd100k"     # 다운로드 페이지 버튼이 가리키는 주소 (HTTPS 인증서 만료로 HTTP만 됨)
VIDEOS_ZIP = f"{BASE}/bdd100k_videos.zip"
LABELS_ZIP = f"{BASE}/bdd100k_labels.zip"
UA = {"User-Agent": "Mozilla/5.0 (research dataset fetch)"}


class HttpRangeFile(io.RawIOBase):
    """HTTP Range 요청으로 읽는 읽기 전용 파일 — zipfile이 원격 zip을 열 수 있게 한다."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block = url, block
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._buf_start, self._buf = 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:  # noqa: BLE001 — 네트워크 끊김은 잠깐 쉬고 다시
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        return b""

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        b0, b1 = self._buf_start, self._buf_start + len(self._buf)
        if not (b0 <= self.pos and end <= b1):
            if n >= self.block:                       # 큰 읽기(영상 본문)는 그대로
                data = self._fetch(self.pos, end - 1)
                self.pos += len(data)
                return data
            self._buf_start = self.pos                # 작은 읽기(목차·헤더)는 미리 읽어 둔다
            self._buf = self._fetch(self.pos, min(self.pos + self.block, self.size) - 1)
            b0, b1 = self._buf_start, self._buf_start + len(self._buf)
        data = self._buf[self.pos - b0:end - b0]
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def load_attributes(split: str, cache_dir: Path) -> dict[str, dict]:
    """라벨 zip(영상마다 json 하나, 약 190MB)을 한 번 받아 두고, split(train|val) 영상별 속성
    {이름: {timeofday, weather, scene}}을 읽는다. 작은 파일 수만 개를 원격으로 읽으면 느려서 통째로 받는다."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "bdd100k_labels.zip"
    if not path.exists():
        print(f"라벨 zip 받는 중 (약 190MB, 한 번만): {path}")
        tmp = path.with_suffix(".zip.part")
        req = urllib.request.Request(LABELS_ZIP, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(4 << 20):
                f.write(chunk)
        tmp.replace(path)
    out = {}
    with zipfile.ZipFile(path) as zf:
        for n in zf.namelist():
            if f"/{split}/" in n and n.endswith(".json"):
                d = json.loads(zf.read(n))
                out[d.get("name", Path(n).stem)] = d.get("attributes", {})
    if not out:
        raise SystemExit(f"라벨 zip에 {split} 라벨이 없습니다")
    return out


def load_members(split: str, cache_dir: Path) -> dict[str, zipfile.ZipInfo]:
    """영상 zip 목차에서 split 영상 위치 정보. 처음엔 원격으로 읽어(수 분) 캐시에 저장, 다음부턴 캐시."""
    cache = cache_dir / "videos_index.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        print("영상 zip 목차 읽는 중 (영상 10만 개 목록, 처음 한 번 몇 분 걸림)…")
        vz = zipfile.ZipFile(HttpRangeFile(VIDEOS_ZIP))
        rows = [[i.filename, i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type]
                for i in vz.infolist() if i.filename.lower().endswith((".mov", ".mp4"))]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")
    out = {}
    for name, off, csize, fsize, crc, ctype in rows:
        if f"/{split}/" not in name:
            continue
        info = zipfile.ZipInfo(name)
        info.header_offset, info.compress_size, info.file_size, info.CRC, info.compress_type = off, csize, fsize, crc, ctype
        out[Path(name).stem] = info
    return out


def _fetch_stored(info: zipfile.ZipInfo, dst: Path) -> float:
    """압축 없이 저장된(STORED) 멤버를 zip 안 위치에서 바로 잘라 받는다 — 연결마다 따로라 병렬로 돌릴 수 있다.
    CRC로 확인한다. 반환: 걸린 초."""
    t0 = time.time()
    req = urllib.request.Request(VIDEOS_ZIP, headers={**UA, "Range": f"bytes={info.header_offset}-{info.header_offset + 29}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        head = r.read()
    if head[:4] != b"PK\x03\x04":
        raise IOError(f"로컬 헤더가 아님: {info.filename}")
    n_name, n_extra = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + n_name + n_extra
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(5):
        try:
            crc, got = 0, 0
            req = urllib.request.Request(VIDEOS_ZIP, headers={**UA, "Range": f"bytes={start}-{start + info.compress_size - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    crc = zlib.crc32(chunk, crc)
                    got += len(chunk)
            if got != info.file_size or crc != info.CRC:
                raise IOError(f"크기·CRC 불일치 {got}/{info.file_size}")
            tmp.replace(dst)
            return time.time() - t0
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    return time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="BDD100K 영상 일부만 받기 (속성으로 골라서)")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--timeofday", default="daytime", help="daytime | night | dawn/dusk | any")
    p.add_argument("--weather", default="clear", help="clear | overcast | partly cloudy | rainy | ... | any")
    p.add_argument("--scene", default="any", help="highway | city street | residential | ... | any")
    p.add_argument("--out", default="data/raw/bdd100k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--workers", type=int, default=6, help="동시 다운로드 수 — 서버가 연결당 약 0.3~0.9MB/s라 여러 개가 빠르다")
    args = p.parse_args()

    attrs = load_attributes(args.split, Path(args.out).parent / "_bdd100k_cache")
    want = {k: v for k, v in (("timeofday", args.timeofday), ("weather", args.weather), ("scene", args.scene)) if v != "any"}
    ok = sorted(n for n, a in attrs.items() if all(a.get(k) == v for k, v in want.items()))
    print(f"{args.split} 라벨 {len(attrs)}개 중 조건 {want} 맞는 영상 {len(ok)}개")

    members = load_members(args.split, Path(args.out).parent / "_bdd100k_cache")
    cand = [n for n in ok if n in members]
    random.Random(args.seed).shuffle(cand)
    pick = cand[:args.count]
    total = sum(members[n].compress_size for n in pick)
    print(f"영상 zip의 {args.split} 영상 {len(members)}개 · 조건 맞는 것 {len(cand)}개 중 {len(pick)}개 고름 — 약 {total / 1e9:.1f}GB")
    if args.list_only:
        for n in pick:
            print(f"  {members[n].filename}  {members[n].compress_size / 1e6:.1f}MB  {attrs[n]}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {Path(members[n].filename).name: attrs[n] for n in pick}
    todo = [members[n] for n in pick
            if not ((out / Path(members[n].filename).name).exists()
                    and (out / Path(members[n].filename).name).stat().st_size == members[n].file_size)]
    print(f"받을 것 {len(todo)}개 (이미 받은 {len(pick) - len(todo)}개 건너뜀), 동시 {args.workers}개")
    if all(i.compress_type == zipfile.ZIP_STORED for i in todo):
        t0, done, fails = time.time(), 0, []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_fetch_stored, i, out / Path(i.filename).name): i for i in todo}
            for fut in as_completed(futs):
                info = futs[fut]
                try:
                    fut.result()
                    done += 1
                    el = time.time() - t0
                    print(f"[{done}/{len(todo)}] {Path(info.filename).name} {info.file_size / 1e6:.1f}MB "
                          f"(경과 {el / 60:.1f}분)", flush=True)
                except Exception as exc:  # noqa: BLE001 — 하나 실패해도 나머지는 받는다
                    fails.append(info.filename)
                    print(f"실패 {info.filename}: {exc}", flush=True)
        if fails:
            print(f"⚠️ 실패 {len(fails)}개 — 다시 실행하면 그것만 받는다")
    else:                                             # 압축된 멤버면 zipfile로 하나씩
        for k, info in enumerate(todo, 1):
            dst = out / Path(info.filename).name
            tmp = dst.with_suffix(dst.suffix + ".part")
            vz = zipfile.ZipFile(HttpRangeFile(VIDEOS_ZIP))
            with vz.open(vz.getinfo(info.filename)) as src, open(tmp, "wb") as f:
                while chunk := src.read(4 << 20):
                    f.write(chunk)
            tmp.replace(dst)
            print(f"[{k}/{len(todo)}] {dst.name}", flush=True)
    (out / "attributes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"완료: {out} ({len(pick)}개, 속성은 attributes.json)")


if __name__ == "__main__":
    sys.exit(main())
