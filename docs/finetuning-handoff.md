# 차량 검출 모델 파인튜닝 인계 (강섬희)

**요청** 설만수 (AI 파트) · **작성** 2026-10-10
**맡아줄 일** AI Hub 데이터 받기 → 코랩에서 학습 → 결과 전달
**예상 소요** 다운로드 반나절, 학습 수 시간~하루(데이터 양에 따라)

필요한 스크립트는 전부 만들어서 저장소에 올려놨고 **합성 데이터로 한 번씩 돌려 동작까지 확인**했다. 명령어를 순서대로 따라가면 된다. 막히면 바로 연락 줘.

---

## 왜 이걸 하나

지금 차량 검출 모델이 **사전학습 가중치 그대로**다. COCO(미국·유럽 일반 사진)로 학습된 거라 국내 도로 대시캠 화면에서 정확도가 떨어진다. 국내 주행 데이터로 미세조정(파인튜닝)하면 이게 개선되고, 그래야 파이프라인 전체 성능 측정이 의미가 생긴다.

---

## 전체 그림

```
[섬희 PC]              [Google Drive]        [Colab]              [나]
AI Hub 다운로드   →    데이터 업로드    →   학습 실행      →   결과 전달
(한국 IP 필요)         (용량 주의)          (GPU)              (가중치+수치)
```

**왜 다운로드를 코랩에서 바로 안 하나** — AI Hub는 해외 IP를 제한하는 것으로 파악돼 있다. 코랩 서버는 해외(구글)라 거기서 바로 받으려 하면 막힐 가능성이 크다. 그래서 **받는 건 한국에 있는 섬희 PC에서** 하고, 그걸 Drive에 올려 코랩이 읽는 구조로 간다.

---

## 0. 시작 전 — 이것부터 정해야 함

### AI Hub 계정 문제

승인은 **신청한 사람 계정에만** 난다. 내 계정은 승인돼 있고 섬희 계정은 아니다.

| | 방법 | 내일 가능? |
|---|---|---|
| A | 섬희가 직접 신청 → 승인 대기 | ❌ 대기 시간 있음 |
| B | 내 API Key를 받아서 사용 | ⭕ |
| C | 내가 원격으로 접속해서 다운로드만 해줌 | ⭕ |

**B나 C로 가고, A도 병행 신청**해두는 걸 권한다. (B면 키는 메신저 본문 말고 따로 주자. 받은 키는 터미널에 직접 치지 말고 파일로 두는 게 낫다 — 히스토리에 남는다.)

### 코랩 등급

- **무료** — T4 GPU, 세션 12시간 제한에 자주 끊김. 디스크 약 78GB
- **Pro 이상** — 더 빠른 GPU, 덜 끊김. 권장

세션이 끊겨도 이어서 할 수 있게 **결과를 Drive에 저장**하도록 아래 명령에 반영해놨다.

### Drive 용량

데이터를 Drive에 올려야 하는데 무료는 15GB다. **아래 3절에서 받을 양을 그에 맞춰 조절하는 게 핵심**이다.

---

## 1. 지금 우리한테 뭐가 없나

맥에서 다운로드를 다섯 번쯤 시도했는데 **이미지는 한 장도 못 받았다.**

| | 상태 |
|---|---|
| 라벨 | ✅ 받아서 YOLO 변환까지 검증 완료 (11,588개) |
| **이미지(원천데이터)** | ❌ **0장** — 14GB짜리가 매번 깨짐. 원인은 2절 |

라벨만 있고 짝이 될 이미지가 없어서 학습 자체가 안 된다. **이미지 확보가 1순위다.**

---

## 2. 다운로드 — 공식 도구 쓰지 말 것

### ❗ AI Hub 공식 `aihubshell`에 버그가 2개 있다

내가 다섯 번 실패하고 나서 찾아낸 건데, OS 문제가 아니라 스크립트 결함이라 **어디서 받든 똑같이 재현된다.**

1. **다운로드가 끊겨도 "성공"이라고 보고한다** — 잘린 파일을 완성본 취급하고 넘어간다
2. **조각 파일 병합 순서가 뒤섞인다** — 대용량 파일은 1GB씩 조각나서 오는데, 정렬 버그로 순서가 섞여 합쳐진다. 14GB 파일(조각 14개)에서 **매번** 깨졌다

**고친 버전이 저장소에 있다 → `scripts/aihubshell_fixed.sh`**

### 받기

```bash
git clone https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git
cd SafeWatch_AI && git checkout dev
chmod +x scripts/aihubshell_fixed.sh

mkdir -p data/raw/aihub && cd data/raw/aihub
SH=../../../scripts/aihubshell_fixed.sh

# 파일 목록 확인 — filekey 번호를 여기서 본다
$SH -mode l -datasetkey 180
```

**목록 결과를 나한테 보내줘.** 어느 번호를 받아야 하는지 짚어줄게. 내가 아직 그 목록을 못 봐서 번호까지는 특정 못 한다.

```bash
# 받기 (filekey는 위 목록에서)
$SH -aihubapikey "$(cat ~/.aihub_api_key)" -mode d -datasetkey 180 -filekey <번호,번호>
```

> `-filekey`를 빼면 **278GB 전체**가 받아진다. 꼭 지정할 것.

### 받을 대상

**도로주행영상 (dataSetSn = 180)** 하나만. 그중에서도:

1. **Validation 원천데이터 + 라벨링데이터** ← 1순위
2. 여유 되면 **Training** 일부 (많을수록 성능이 좋아진다)

> 데이터셋이 `원천/라벨` × `부위(bb·fs·pg·sl)` × `도로유형`으로 쪼개져 있다. **우리가 쓰는 건 `bb`(bounding box)** 다. `pg`(폴리곤)는 변환이 따로 필요해서 안 쓴다.

다른 데이터셋(654 로드마크, 172 번호판)은 이번엔 **받지 말 것.** 나중에 쓴다. 597(교통사고, 3.79TB)은 아예 보류 대상이다.

### 받고 나서 반드시 — 압축 풀기 전에 무결성 확인

```bash
python3 -c "import zipfile; print('손상 있음' if zipfile.ZipFile('원천데이터.zip').testzip() else '정상')"
```

맥에서 **5번 중 3번이 깨져 있었는데 압축을 푼 뒤에야 알았다.** 깨졌으면 그 파일만 다시 받으면 된다.

### 압축 풀기 — 한글 파일명 주의

zip 안 한글 이름이 CP949라서 리눅스 `unzip`이 깨뜨린다.

```bash
unzip -O cp949 원천데이터.zip -d extracted
```

`-O` 옵션이 없다는 에러가 나면 [`docs/gpu-desktop-setup.md`](gpu-desktop-setup.md) 7절의 Python 스크립트를 쓰면 된다.

---

## 3. 데이터셋 만들기 (섬희 PC에서)

AI Hub 라벨은 XML이라 YOLO가 못 읽는다. 변환하고 학습용 구조로 바꿔야 한다. **스크립트 두 개로 끝난다.**

```bash
cd ~/SafeWatch_AI
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics pyyaml

# 1) XML 라벨 → YOLO 포맷
python scripts/voc_to_yolo.py \
  --input-dir data/raw/aihub/016.도로주행영상/2.Validation/라벨링데이터_.../extracted/bb \
  --output-dir data/processed/yolo_labels/val

# 2) 이미지와 라벨 짝 맞춰 학습용 구조로
python scripts/prepare_yolo_dataset.py \
  --images-dir data/raw/aihub/016.도로주행영상/2.Validation/원천데이터_.../extracted \
  --labels-dir data/processed/yolo_labels/val \
  --output-dir data/processed/yolo_dataset \
  --split val
```

Training 세트도 받았으면 같은 걸 `--split train`으로 한 번 더 돌린다.

> `prepare_yolo_dataset.py`가 **이미지와 라벨 개수가 안 맞으면 경고**를 띄운다. 짝이 안 맞는 게 많이 나오면 원천데이터와 라벨링데이터를 서로 다른 묶음으로 받은 것일 수 있으니 알려줘.

### ⚠️ Train 세트가 없으면

`data.yaml`에 "임시" 표시가 붙고 학습 데이터와 평가 데이터가 같아진다. 돌아가긴 하는데 **성능 수치가 의미 없어진다**(시험 문제를 미리 보고 푸는 것과 같다). 돌려보는 것 자체는 괜찮지만, 제대로 된 결과를 내려면 Training 세트가 있어야 한다.

### Drive에 올리기

`data/processed/yolo_dataset` 폴더를 통째로 Drive에 올린다. **용량이 크면 zip으로 묶어 올리고 코랩에서 푸는 쪽이 훨씬 빠르다** (작은 파일 수만 개를 Drive에 직접 올리면 매우 느리다).

```bash
cd data/processed && zip -r yolo_dataset.zip yolo_dataset
```

> `prepare_yolo_dataset.py`는 기본적으로 심볼릭 링크를 만든다. **zip으로 묶을 거면 `--copy` 옵션을 붙여 실제 파일로 만들어야 한다** — 링크만 묶이면 코랩에서 빈 파일이 된다.

---

## 4. 코랩에서 학습

새 노트북을 만들고 **런타임 → 런타임 유형 변경 → GPU**로 바꾼 뒤, 아래를 셀 단위로 실행한다.

```python
# [셀 1] GPU 확인
!nvidia-smi
```

```python
# [셀 2] Drive 연결
from google.colab import drive
drive.mount('/content/drive')
```

```python
# [셀 3] 저장소 + 패키지
!git clone -b dev https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git
%cd SafeWatch_AI
!pip install -q ultralytics
```

```python
# [셀 4] 데이터 풀기 (경로는 올린 위치에 맞게 수정)
!unzip -q /content/drive/MyDrive/yolo_dataset.zip -d /content/data
# data.yaml의 path가 섬희 PC 기준이라 코랩 경로로 고쳐준다
import pathlib, re
p = pathlib.Path('/content/data/yolo_dataset/data.yaml')
p.write_text(re.sub(r'(?m)^path:.*$', 'path: /content/data/yolo_dataset', p.read_text()))
print(p.read_text())   # path가 /content/... 로 바뀌었는지 눈으로 확인
```

```python
# [셀 5] 먼저 스모크 테스트 — 설정이 맞는지 2 epoch로 확인 (몇 분)
!python scripts/train.py \
  --data /content/data/yolo_dataset/data.yaml \
  --epochs 2 --name smoke \
  --project /content/drive/MyDrive/safewatch_train
```

```python
# [셀 6] 본 학습 — 결과가 Drive에 쌓이므로 세션이 끊겨도 남는다
!python scripts/train.py \
  --data /content/data/yolo_dataset/data.yaml \
  --name safewatch \
  --project /content/drive/MyDrive/safewatch_train
```

```python
# [셀 7] 세션이 끊겼을 때 — 이어서 하기
!python scripts/train.py \
  --data /content/data/yolo_dataset/data.yaml \
  --name safewatch \
  --project /content/drive/MyDrive/safewatch_train \
  --resume
```

### 성능 설정은 이미 잡아뒀다

정확도 위주로 기본값을 맞춰놨으니 **옵션을 따로 줄 필요 없다.** 참고로 뭐가 켜져 있냐면:

| 설정 | 왜 |
|---|---|
| epochs 200 + patience 50 | 길게 돌리되 더 안 좋아지면 알아서 멈춘다 |
| 해상도 640 | Pi5 추론은 320이지만 크게 학습해야 원거리 차량을 잘 배운다 |
| 코사인 학습률 | 후반 수렴이 안정적 |
| mosaic + mixup 증강 | 가려지거나 겹친 차량에 강해진다 |
| 마지막 15 epoch mosaic 해제 | 실제 화면 분포로 마무리 학습 — 보통 여기서 수치가 한 번 더 오른다 |
| 10 epoch마다 체크포인트 | 코랩이 끊겨도 되돌아갈 지점이 남는다 |

**메모리 부족(OOM) 에러가 나면** `--cache disk`를 붙여라. 그래도 나면 `--batch 16`처럼 숫자를 직접 지정하면 된다.

### 한 가지 바꾸면 안 되는 것

**모델은 반드시 `yolov8n`이다.** 더 큰 모델(s/m/l)을 쓰면 수치는 올라가지만 **최종 목표인 라즈베리파이5에서 3~5fps로 떨어져 실시간이 안 된다.** 코랩 GPU가 좋아도 이건 배포 장비 때문에 생긴 제약이라 안 바뀐다. 스크립트가 다른 모델을 넣으면 경고하고 물어본다.

---

## 5. 끝나면 보내줄 것

```python
# [셀 8] 결과 확인
!ls /content/drive/MyDrive/safewatch_train/safewatch/
```

이 4가지만 주면 된다.

- [ ] **`weights/best.pt`** ← 제일 중요. 이게 학습된 모델이다
- [ ] **`results.png`** — 학습 곡선 (제대로 수렴했는지 보는 용도)
- [ ] **`results.csv`** — epoch별 수치
- [ ] 마지막에 출력된 **요약 4줄** (mAP50-95 / mAP50 / Precision / Recall)

Drive 폴더를 공유해줘도 되고 파일만 따로 보내줘도 된다.

### 수치 보는 법

| | 의미 |
|---|---|
| **mAP50-95** | 종합 정확도. 가장 중요한 숫자 |
| mAP50 | 느슨한 기준 정확도 |
| Precision | 높을수록 오탐(차 아닌 걸 차라고 함)이 적음 |
| Recall | 높을수록 놓치는 차가 적음 |

사전학습 모델 대비 올랐는지가 핵심이라, 숫자가 낮아 보여도 그냥 보내주면 내가 비교한다. **이상하다 싶어도 그대로 보내줘** — 데이터 문제인지 설정 문제인지는 내가 보면 안다.

---

## 6. 자주 막히는 지점

| 증상 | 원인·해결 |
|---|---|
| 압축 풀었는데 파일 이름이 깨짐 | CP949 문제. 2절의 `-O cp949` 참고 |
| `data.yaml을 찾지 못했다` | 3절 2번 스크립트를 안 돌렸거나 경로 오타 |
| `이미지와 라벨의 파일 이름이 하나도 안 맞는다` | 원천데이터와 라벨링데이터를 다른 묶음으로 받음. 같은 날짜·`bb`로 다시 |
| 코랩에서 이미지가 0장 | 심볼릭 링크를 zip으로 묶음. `--copy`로 다시 (3절 참고) |
| CUDA out of memory | `--cache disk` 추가, 그래도 나면 `--batch 16` |
| 세션이 끊김 | 셀 7의 `--resume`. Drive에 저장돼 있어 처음부터 안 해도 된다 |

---

## 참고 — 데이터와 별개로 아직 회신 못 받은 것

[`docs/hw-handoff-guide.md`](hw-handoff-guide.md) 3장으로 2주 전에 보낸 건이다. 급하진 않은데 같이 봐주면 좋겠다.

1. **카메라 마운트 실측치** (높이·각도·화각) — 없어서 차선 좌표 변환 기준값이 아직 가짜다
2. **ESP32 UART 프로토콜** — GPS 속도를 못 받아 단서 5개 중 `차선 걸침` 하나가 계속 판정 보류 중
3. **카메라 해상도 확정** — 문서 640×384 vs 설정 320×320 불일치
4. **링버퍼 클립 요청 형태** (함수 호출 / IPC / 파일) — 정해져야 이벤트에 영상이 붙는다
