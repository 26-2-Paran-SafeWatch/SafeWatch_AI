# UFLDv2 차선 모델 학습 가이드 (GPU 데스크탑)

`docs/lane-detection-dl-plan.md` 5장 **2단계 파인튜닝**을 AI Hub Training 데이터 전체로 다시 하는 절차.
데스크탑 환경(WSL2·CUDA PyTorch·저장소 클론·API 키 파일)은 [`gpu-desktop-setup.md`](gpu-desktop-setup.md) 1~5장을 먼저 끝낸다.
모든 명령은 **WSL Ubuntu 터미널, 저장소 루트, `.venv` 활성화 상태**에서 실행한다.
학습 코드는 아직 `feature/lane-ufld` 브랜치에만 있다 — `git fetch && git checkout feature/lane-ufld`.

## 0. 왜 다시 하나 (2026-11-07)

- 노트북 18·19번 파인튜닝은 약 717장(Validation 데이터 일부)으로 해서 효과가 없었다 — 데이터가 너무 적었다.
- 사전학습 그대로는 노란 중앙선·점선을 건너뛰고 한 칸 바깥 선을 잡는 오류가 남는다. 규칙(폭 검사 등)으로 막는 건
  임시방편이고, 근본 대책은 한국 도로로 학습하는 것이다.
- Training 데이터 60_전방에서 정답 v3가 나오는 프레임이 **22,745장(466개 시퀀스)** — 이전의 32배.

### 1차 결과와 2차 변경 (2026-11-08)

- **1차** (코랩 T4, 20에폭, 뒤집기·밝기 증강만): 선택용 손실 최저는 6에폭(2.43 → 0.807), 이후 과적합.
  노트북 25번 평가(범위 0.68, 기본 안전장치) — 검증 사용 82 → 87%, 0.10/0.15/0.25 초과 6/1/0 → 3/0/0으로 **좋아졌지만**,
  장착이 다른 카메라(틀어짐)는 6/3/2 → 7/6/4로 **나빠졌다**. 오답 그림은 한 차선의 행별 점이 이웃한 두 선을 오가는 모양 —
  학습 카메라에서 차선이 화면 어디쯤 나오는지를 외운 것으로 본다. 우리 카메라도 '처음 보는 카메라'라 1차 가중치는 쓰지 않는다.
- **2차**: 원 저장소와 같은 **기하 증강**(이동 x ±12%·y ±17%, 회전 ±6°, 확대 0.8~1.2배 — `configs/ufld_train.yaml`
  train.augment.geometric)을 넣고 차선 정답도 같이 변환한다. 그러려면 사진을 자르기 전 크기(1600×533)로 저장해야 해서
  데이터 형식이 2로 바뀌었다(1차 데이터는 못 씀 — 다시 만든다). 에폭 선택은 선택용 시퀀스에 같은 증강을 고정 난수로 건 손실,
  에폭 12. 결과 파일 이름은 `ufld_aihub_aug.pth`(1차 `ufld_aihub_ft.pth`는 비교용으로 둔다).

## 1. 받을 파일 (데이터셋 180 도로주행영상)

| filekey | 내용 | 크기 |
|---|---|---|
| 38568 | Training 라벨 sl 도심로 | 102MB |
| 38569 | Training 라벨 sl 자동차전용도로 | 115MB |
| 38576 | Training 원천(이미지) sl 도심로 | 48GB |
| 38577 | Training 원천(이미지) sl 자동차전용도로 | 39GB |
| 38578 | Validation 라벨 (학습에서 뺄 시퀀스 목록용) | 61MB |

디스크 여유 **약 100GB** 필요(zip 87GB + 변환 결과 약 6GB). 변환이 끝나면 이미지 zip 두 개는 지워도 된다.

```bash
mkdir -p data/raw/aihub && cd data/raw/aihub
bash ../../../scripts/aihubshell_fixed.sh -aihubapikey "$(cat ~/.aihub_api_key)" \
    -mode d -datasetkey 180 -filekey 38568,38569,38576,38577,38578
cd ../../..
```

> 공식 `aihubshell` 말고 반드시 저장소의 `aihubshell_fixed.sh`를 쓴다 (조각 병합 순서 버그 — `gpu-desktop-setup.md` 6장).
> 키는 채팅·명령 기록에 직접 쓰지 말고 `~/.aihub_api_key` 파일로만 넘긴다.

## 2. 사전학습 가중치

노트북이 쓰던 CULane 가중치를 그대로 쓴다. 구글 드라이브 `SafeWatch/ufld/culane_res18.pth`를 받아
`data/models/culane_res18.pth`에 둔다 (원 저장소 README의 링크와 같은 파일).

## 3. 학습 데이터 만들기

```bash
D=data/raw/aihub/016.도로주행영상
python -m scripts.prepare_ufld_dataset \
    --label-zips "$D/1.Training/라벨링데이터_0107/sl/도심로.zip" "$D/1.Training/라벨링데이터_0107/sl/자동차전용도로.zip" \
    --image-zips "$D/1.Training/원천데이터_0107/sl/도심로.zip" "$D/1.Training/원천데이터_0107/sl/자동차전용도로.zip" \
    --exclude-label-zip "$D/2.Validation/라벨링데이터_0107/라벨링데이터.zip" \
    --out data/processed/ufld_aihub2 --workers 8
```

- **검증(Validation) 시퀀스는 전부 뺀다.** AI Hub 학습·검증 데이터는 같은 시퀀스·같은 프레임을 공유한다
  (60_전방 검증 시퀀스 98개 중 44개, 똑같은 프레임 1,968장). 노트북 평가가 그 시퀀스들이라 빼지 않으면 점수가 부풀려진다.
- **20200916 날짜도 뺀다** (`configs/ufld_train.yaml` data.exclude_dates). 노트북의 '틀어짐'(처음 보는 카메라) 시험용이다.
- 정답 규칙은 노트북 정답 v3와 같다(`scripts/ufld_labels.py`, 검증 라벨 2,766장으로 노트북 함수와 전부 일치 확인).
- 사진은 자르기 전 1600×533으로 저장한다(형식 2 — 기하 증강용). `train.npz`·`val.npz`에 차선 선(`polys`)도 들어간다.
  변환 없이 이 선으로 만든 정답은 기존 행 정답과 같다(실제 Training 6,000장: 346만 칸 중 10칸, 1칸 차).
- 끝나면 `data/processed/ufld_aihub2/manifest.json`에 장수·제외 이유가 남는다. 예상: 정답 있는 프레임 약 22,700장,
  그중 10% 시퀀스가 에폭 선택용.
- 중간에 끊기면 같은 명령을 다시 실행하면 된다 (만든 이미지는 건너뜀).

## 4. 학습

```bash
python -m scripts.train_ufld --data data/processed/ufld_aihub2 \
    --weights data/models/culane_res18.pth --out data/models/ufld_aihub_aug.pth
```

- 처음에 "학습 전 선택용 손실"(사전학습 그대로)을 찍고, 에폭마다 선택 기준 손실(기본: 증강 건 선택용 손실,
  `train.select_by`)이 그보다 낮아지면 `--out`에 저장한다. 증강 없는 선택용 손실도 같이 기록한다.
- 설정은 `configs/ufld_train.yaml` (12에폭, 배치 32, fp16, 기하 증강). GPU 메모리가 부족하면 batch_size를 16으로.
- 증강 비용: 사진당 약 11 → 15ms(개발 맥, 1600×533 디코딩 포함) — 코랩 T4(CPU 2개)에서도 GPU 시간(스텝당 약 1.3초)보다 작다.
- 끊기면 같은 명령에 `--resume`.
- 남는 파일: `ufld_aihub_aug.pth`(최저 손실 에폭), `ufld_aihub_aug_last.pth`(이어 하기용), `ufld_aihub_aug_history.json`(에폭별 손실).
- 바로 돌려 보기: `--epochs 1 --max-steps 50`으로 몇 분 안에 끝까지 도는지 먼저 확인한다.

## 5. 평가 — 노트북에서

학습 스크립트의 선택용 손실은 에폭을 고르는 데만 쓴다. **합격 판정은 노트북의 검증·틀어짐 평가**(학습에서 뺀 시퀀스, 정답 v3,
21~23번 셀과 같은 기준)로 한다.

1. `ufld_aihub_aug.pth`를 구글 드라이브 `SafeWatch/ufld/`에 올린다 (코랩 26번 셀은 바로 거기 저장한다).
2. 노트북 **25번 셀**(1번 셀만 실행하면 됨)을 돌린다. Validation 60_전방 전체(정답 v3 약 2,300장 — 이전 평가의 약 8배)에서
   사전학습·재학습을 측정 범위 0.56~0.68 × 안전장치(존재만/기본 전부)별로 같은 표에 낸다.
3. 좋아졌으면 ONNX로 바꿔 실행 경로에 넣는다:

```bash
python -m scripts.export_ufld_onnx --weights data/models/ufld_aihub_aug.pth \
    --output data/models/ufld_aihub_aug.onnx --verify
```

그다음 `configs/default.yaml`(또는 pi5.yaml)의 `lane.ufld.model_path`를 이 파일로 바꾼다.

## 6. 다음 후보 (결과를 보고)

- NPU용 작은 출력부(격자 200→100 등)로 같은 데이터에서 학습 — 마지막 완전연결층이 모델 대부분이라 NPU 메모리 부담이 크다.
- 정답이 안 나오는 프레임 약 18,000장(자차 쌍 없음/폭 이상)의 원인 분석 — 규칙을 넓히면 학습 데이터가 늘어난다.
- 우리 카메라 영상으로 추가 파인튜닝 (6단계).
