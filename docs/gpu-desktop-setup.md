# GPU 데스크탑 환경 설정 가이드

팀원 데스크탑(RTX 4070, 2TB, Windows)에서 AI Hub 데이터 다운로드와 YOLO 파인튜닝을 진행하기 위한 설정 절차. 2026-09-28 결정 (`docs/sprint-plan.md` "다운로드·파인튜닝 환경 확정" 참고).

## 왜 WSL인가

`aihubshell`은 bash 스크립트라 Windows에서 직접 실행이 안 된다. AI Hub 공식 안내에도 "윈도우에서 파일 다운로드 시 WSL 설치가 필요"라고 명시돼 있다. PyTorch CUDA 학습도 WSL2 안에서 문제없이 GPU를 그대로 쓸 수 있다(Windows의 NVIDIA 드라이버를 WSL2가 그대로 통과시킴, WSL 안에 별도 드라이버 설치 불필요).

## 1. WSL2 + Ubuntu 설치 (Windows PowerShell, 관리자 권한)

```powershell
wsl --install -d Ubuntu
```

설치 후 재부팅, Ubuntu 최초 실행 시 사용자 계정 생성. 이후 모든 명령은 **WSL Ubuntu 터미널 안에서** 실행한다.

## 2. GPU 인식 확인

```bash
nvidia-smi
```

RTX 4070 정보가 뜨면 정상. 안 뜨면 Windows 쪽 NVIDIA 드라이버를 최신으로 업데이트한다(WSL 안에서 드라이버를 따로 설치하지 않는다 — 그러면 오히려 충돌한다).

## 3. 기본 패키지 + 저장소 클론

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git curl unzip
git clone https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git
cd SafeWatch_AI
git checkout dev
```

## 4. Python 환경 + CUDA PyTorch

`requirements.txt`만 바로 설치하면 CPU 전용 PyTorch가 깔릴 수 있으니, **CUDA 빌드를 먼저 설치**한다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

확인:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`True RTX 4070`처럼 나오면 정상.

## 5. AI Hub API Key 준비

키를 터미널에 직접 노출하지 않도록 파일로 저장한다 (Mac 세션에서 실수로 채팅에 키가 노출된 적이 있어 이 방식으로 정착함).

```bash
read -s -p "AI Hub API key: " k && echo "$k" > ~/.aihub_api_key && unset k && echo "저장 완료"
chmod 600 ~/.aihub_api_key
```

## 6. 데이터 다운로드 — 패치된 aihubshell 사용

**⚠️ AI Hub 공식 `aihubshell`을 그대로 받지 말고 이 저장소의 패치 버전을 쓴다.** 원본에는 버그가 2개 있다 (`scripts/aihubshell_fixed.sh` 상단 주석 및 `docs/sprint-plan.md` "aihubshell 자체 버그 2개" 참고):
1. 다운로드가 중간에 끊겨도 항상 "성공"으로 처리하는 버그
2. 조각 파일이 여러 개(대용량 파일)일 때 병합 순서가 뒤섞이는 버그 — 실제로 14GB짜리 파일에서 매번 재현됨

```bash
mkdir -p data/raw/aihub && cd data/raw/aihub
chmod +x ../../../scripts/aihubshell_fixed.sh
ALIAS=../../../scripts/aihubshell_fixed.sh

# 파일 목록 확인
$ALIAS -mode l -datasetkey 180   # 도로주행영상
$ALIAS -mode l -datasetkey 654   # 도로 로드마크 인식
$ALIAS -mode l -datasetkey 172   # 번호판 인식용
# 597(교통사고, 3.79TB)은 sprint-plan.md 기준 보류 대상 — 필요할 때만

# 다운로드 (filekey는 위 -mode l 결과에서 확인)
$ALIAS -aihubapikey "$(cat ~/.aihub_api_key)" -mode d -datasetkey 180 -filekey <filekey들 콤마구분>
```

2TB 여유가 있으니 Mac 세션 때처럼 서브셋만 골라 받을 필요 없이 필요한 데이터셋을 통째로 받아도 된다. 단, 어차피 처음엔 소량으로 파이프라인 검증부터 하는 걸 권장 (Validation 세트부터).

## 7. 압축 해제 — 한글 파일명 인코딩 주의

AI Hub zip 내부 한글 파일명이 CP949(EUC-KR)로 인코딩돼 있어 `unzip`이 깨질 수 있다. 먼저 시도:

```bash
unzip -O cp949 원천데이터.zip -d extracted
```

이게 안 되면(리눅스 unzip 버전에 따라 `-O` 옵션이 없을 수 있음) Python으로 직접 푼다:

```bash
python3 - <<'EOF'
import zipfile, pathlib

zpath = "원천데이터.zip"
outdir = pathlib.Path("extracted")
outdir.mkdir(exist_ok=True)

with zipfile.ZipFile(zpath) as z:
    for info in z.infolist():
        try:
            name = info.filename.encode('cp437').decode('cp949')
        except Exception:
            name = info.filename
        target = outdir / name
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(target, 'wb') as dst:
            dst.write(src.read())
EOF
```

## 8. 무결성 확인 (다운로드·병합 끝난 뒤 항상 확인)

```bash
python3 -c "import zipfile; zipfile.ZipFile('원천데이터.zip').testzip() and print('손상 있음') or print('정상')"
```
`손상 있음`이 나오면 재다운로드 — Mac 세션에서 5번 중 3번이 이 단계에서 깨진 걸 나중에야 발견했다. **압축 풀기 전에 반드시 이 확인부터 할 것.**

## 9. YOLO 라벨 변환

라벨(XML) 다운로드 후 `scripts/voc_to_yolo.py`로 YOLO 포맷 변환 (이미 실제 라벨 11,588개로 검증 완료된 스크립트):

```bash
python scripts/voc_to_yolo.py \
  --input-dir "data/raw/aihub/016.도로주행영상/2.Validation/라벨링데이터_0107/extracted/bb" \
  --output-dir data/processed/yolo_labels
```
