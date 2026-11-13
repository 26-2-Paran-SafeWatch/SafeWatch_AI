# 차량 검출 파인튜닝 — 데이터 목록 + 실행 코드

**전달** 설만수 → 강섬희 · 2026-10-10

지금 검출 모델이 사전학습(COCO) 그대로라 국내 도로에서 정확도가 낮다. AI Hub 국내 주행 데이터로 파인튜닝하는 작업이다.

스크립트는 전부 저장소에 있고 합성 데이터로 동작 확인까지 끝냈다. 아래 명령만 순서대로 돌리면 된다.

---

## 1. 받을 데이터

### AI Hub — 도로주행영상 (`dataSetSn = 180`, 전체 277.89GB)

| 받을 것 | 비고 |
|---|---|
| **Training** 원천데이터 + 라벨링데이터 | 많을수록 성능 ↑ |
| **Validation** 원천데이터 + 라벨링데이터 | 평가용, 필수 |

- 부위는 **`bb`(bounding box)** 만. `pg`(폴리곤)·`fs`·`sl`은 안 쓴다
- **원천데이터와 라벨링데이터는 같은 날짜 묶음으로** 받아야 짝이 맞는다
- 다른 데이터셋(654 로드마크, 172 번호판, 597 교통사고)은 이번엔 받지 말 것

API 키는 따로 전달.

### ⚠️ 공식 `aihubshell` 쓰지 말 것 — 버그 2개

1. 다운로드가 끊겨도 "성공"으로 보고 → 잘린 파일을 그대로 넘김
2. 조각 파일 병합 순서가 뒤섞임 → 14GB 파일(조각 14개)에서 **매번** 깨졌음

고친 버전이 저장소에 있다. 이것만 쓰면 된다.

```bash
git clone -b dev https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git
cd SafeWatch_AI && chmod +x scripts/aihubshell_fixed.sh

mkdir -p data/raw/aihub && cd data/raw/aihub
SH=../../../scripts/aihubshell_fixed.sh

# 파일 목록 — filekey 번호 확인
$SH -mode l -datasetkey 180

# 받기 (filekey 생략하면 278GB 전체가 받아지니 꼭 지정)
$SH -aihubapikey "<API키>" -mode d -datasetkey 180 -filekey <번호,번호>
```

> **`-mode l` 결과를 나한테 보내주면 어느 번호를 받아야 하는지 짚어줄게.** 내가 그 목록을 아직 못 봐서 번호까지는 특정 못 했다.

### 받은 뒤 — 압축 풀기 전에 무결성 확인

```bash
python3 -c "import zipfile; print('손상' if zipfile.ZipFile('원천데이터.zip').testzip() else '정상')"
```

맥에서 **5번 중 3번이 깨져 있었는데 압축 푼 뒤에야 알았다.** 깨졌으면 그 파일만 다시 받으면 된다.

### 압축 해제 — 한글 파일명 CP949

```bash
unzip -O cp949 원천데이터.zip -d extracted
```

`-O` 옵션 없다고 하면 [`docs/gpu-desktop-setup.md`](gpu-desktop-setup.md) 7절의 Python 코드 사용.

---

## 2. 코드

`pip install ultralytics` 외에 추가 의존성 없다.

### ① XML 라벨 → YOLO 포맷

```bash
# Training
python scripts/voc_to_yolo.py \
  --input-dir data/raw/aihub/016.도로주행영상/1.Training/라벨링데이터_<날짜>/extracted/bb \
  --output-dir data/processed/yolo_labels/train

# Validation
python scripts/voc_to_yolo.py \
  --input-dir data/raw/aihub/016.도로주행영상/2.Validation/라벨링데이터_<날짜>/extracted/bb \
  --output-dir data/processed/yolo_labels/val
```

### ② 이미지·라벨 짝 맞춰 학습용 구조 생성

```bash
python scripts/prepare_yolo_dataset.py \
  --images-dir data/raw/aihub/016.도로주행영상/1.Training/원천데이터_<날짜>/extracted \
  --labels-dir data/processed/yolo_labels/train \
  --output-dir data/processed/yolo_dataset --split train

python scripts/prepare_yolo_dataset.py \
  --images-dir data/raw/aihub/016.도로주행영상/2.Validation/원천데이터_<날짜>/extracted \
  --labels-dir data/processed/yolo_labels/val \
  --output-dir data/processed/yolo_dataset --split val
```

짝이 안 맞는 파일 수를 경고로 띄운다. **많이 나오면 알려줘** — 원천데이터와 라벨을 다른 묶음으로 받았다는 뜻이다.

### ③ 학습

```bash
# 설정 확인용 — 2 epoch 스모크 테스트 먼저
python scripts/train.py --data data/processed/yolo_dataset/data.yaml --epochs 2 --name smoke

# 본 학습
python scripts/train.py --data data/processed/yolo_dataset/data.yaml --name safewatch

# 중간에 멈췄으면 이어서
python scripts/train.py --data data/processed/yolo_dataset/data.yaml --name safewatch --resume
```

**옵션 따로 줄 필요 없다.** 정확도 위주로 기본값을 박아놨다 — epochs 200/patience 50, 코사인 LR, mosaic+mixup 증강, 마지막 15 epoch mosaic 해제, 10 epoch마다 체크포인트.

- RAM 여유가 데이터셋보다 크면 `--cache ram` 붙이면 빨라진다 (정확도는 동일)
- CUDA OOM 나면 `--batch 16`처럼 직접 지정

> **모델 크기는 `yolov8n` 고정.** 더 큰 걸 쓰면 수치는 오르지만 최종 배포 장비인 라즈베리파이5에서 3~5fps로 떨어져 실시간이 안 된다. 스크립트가 다른 모델 넣으면 경고한다.

---

## 3. 끝나면 보내줄 것

`results/train/safewatch/` 아래에 전부 있다.

- [ ] **`weights/best.pt`** ← 학습된 모델. 제일 중요
- [ ] **`results.png`** — 학습 곡선
- [ ] **`results.csv`** — epoch별 수치
- [ ] 마지막에 출력되는 **요약 4줄** (mAP50-95 / mAP50 / Precision / Recall)

수치가 낮아 보여도 그대로 보내줘. 사전학습 대비 올랐는지가 핵심이라 비교는 내가 한다.
