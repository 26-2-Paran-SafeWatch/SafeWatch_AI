# HW 파트(강섬희) 인수인계 가이드

**작성** 설만수 (AI 파트)
**작성일** 2026-10-01
**대상** 강섬희 (HW 파트) — Pi5에 AI 모듈을 올려서 실행할 사람

---

## 이 문서의 목적

AI 파트(이 저장소)가 만든 걸 Pi5에서 실제로 돌리려면 뭘 해야 하는지, 그리고
HW 쪽(카메라·ESP32·링 버퍼)에서 우리한테 맞춰줘야 하는 게 뭔지를 한 곳에
정리한다. 더 자세한 설계는 각 문서를 링크해 뒀으니 필요할 때 펼쳐보면 된다.

---

## 1. 우리가 만든 것 — 한 문장 요약

**카메라 영상 → 차량 검출 → 차선 인식 → 추적 → 위험 판단 → 이벤트 생성**까지
전부 `python -m src.main` 하나로 실행되는 **단일 Python 프로그램**이다. 모듈이
여러 개로 나뉘어 있다고 각각 따로 띄우는 게 아니라, 이 저장소를 통째로 Pi5에
올리고 명령 하나로 돌리면 된다.

| 구성 | 내용 | 상태 |
|---|---|---|
| 모델 | YOLOv8n(차량 검출) **1개뿐** | **사전학습(COCO) 가중치 그대로 — 아직 파인튜닝 전** |
| 차선 인식 | OpenCV (딥러닝 아님) | 완료, 실차 영상으로 미검증 |
| 추적 | ByteTrack (딥러닝 아님) | 완료 |
| 위험 판단 | rule-based (사행·차선걸침·스웨빙·표류·속도불규칙) | 완료, 임계값은 초안 |
| 이벤트 생성 | JSON 메타데이터 + 로컬 SQLite 큐 | 완료, 클립·전송은 미연결(3장 참고) |

**지금 상태로 돌려도 되나?** — 된다. 정확도는 낮지만(파인튜닝 전) 파이프라인
전체가 Pi5에서 fps 목표를 만족하는지, 끊김 없이 도는지는 지금 바로 확인할 수
있다. 파인튜닝은 나중에 모델 파일만 바꿔치기하면 된다.

---

## 2. Pi5에서 실행하는 법

### 2.1 환경 설치

```bash
git clone https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git
cd SafeWatch_AI

# picamera2는 pip이 아니라 시스템 패키지로 설치
sudo apt install -y python3-picamera2

# 가상환경에서 시스템 패키지(picamera2) 접근이 필요하다
python -m venv .venv --system-site-packages
source .venv/bin/activate

# Pi5 전용 최소 의존성 (개발 PC용 requirements.txt와 다름 — 모델 변환 도구 제외)
pip install -r requirements-pi5.txt
```

### 2.2 모델 파일 준비

`data/`는 git에 올리지 않으므로(용량·개인정보) 모델 파일은 직접 옮겨야 한다.

- 지금 당장 테스트만 하려면: 저장소에 없는 `data/models/yolov8n.pt`를
  [Ultralytics 공식 배포본](https://github.com/ultralytics/assets/releases)에서
  받거나, `pip install ultralytics` 후 `from ultralytics import YOLO; YOLO("yolov8n.pt")`를
  한 번 실행하면 자동으로 받아진다.
- **실제 배포용**은 개발 PC(또는 파인튜닝용 GPU 데스크탑)에서 아래처럼 변환한
  뒤 그 결과 파일/폴더를 Pi5로 `scp` 등으로 옮긴다.

```bash
# 개발 PC에서
python scripts/export_model.py --format onnx --imgsz 320   # 또는 --format ncnn
```

- ONNX는 `data/models/yolov8n.onnx` 파일 하나
- NCNN은 `data/models/yolov8n_ncnn_model/` 디렉터리 통째로 (안에 `.param`+`.bin`)

옮긴 뒤 `configs/pi5.yaml`의 `detection.weights`가 그 경로를 가리키는지 확인한다.
ONNX·NCNN 둘 다 당장 만들 수 있고, **어느 쪽을 최종으로 쓸지는 아직 안 정해졌다** —
Pi5에서 실측해서 고른다(4.1절 벤치마크가 그 실측 도구).

### 2.3 벤치마크로 먼저 확인

```bash
python -m scripts.benchmark --config configs/pi5.yaml --source camera --duration-sec 60
```

단계별 처리 시간(ms), fps, CPU 온도를 재서 콘솔에 보여주고 `results/benchmark/`에
JSON으로 남긴다. **목표는 10fps 이상**이다. 이게 먼저 통과해야 실제 실행이
의미가 있다.

### 2.4 실제 실행

```bash
python -m src.main --config configs/pi5.yaml --source camera
```

카메라에서 실시간으로 읽어 파이프라인을 돌린다. `Ctrl+C`로 중단하면 지금까지의
단계별 평균 처리 시간을 로그로 남기고 종료한다.

---

## 3. HW 쪽에서 우리한테 맞춰줘야 하는 것 (인터페이스 계약)

여기 3가지는 **우리 코드가 받을 준비는 해놨는데, 실제 HW 쪽 사양·프로토콜이
확정돼야 완전히 연결되는 부분**이다. 확정되면 알려주면 바로 연결한다 —
우리 쪽은 이 셋 다 "인터페이스만 정의해두고 자리 비워둔" 상태라, 값만 채우면
된다.

### 3.1 카메라 해상도 — 불일치 있음, 확인 필요

[`docs/system-architecture.md`](system-architecture.md)엔 카메라가
**lores(640×384, 추론용) / main(1280×720, 저장용) dual-stream**을 낸다고
적혀 있는데, 우리 `configs/pi5.yaml`의 `input.resolution`은 **`[320, 320]`**로
값이 달라. 아래 둘 중 하나로 정리돼야 함:
- lores 스트림이 정말 640×384로 고정이면 우리 config를 그 값에 맞춰 수정
- 아니면 실제 카메라가 어떤 해상도로 lores를 뽑는지 알려주면 그에 맞춤

또, `src/input/source.py`는 지금 **단일 해상도 스트림만** 받는 구조다(`picamera2`에서
lores만 받는지, main도 같이 받을지 미정) — 링 버퍼(main 스트림)를 우리 코드가
직접 안 보고 HW 쪽이 별도로 관리한다면 지금 구조로 충분하다.

### 3.2 GPS/IMU — ESP32 UART 프로토콜 미정

차선 걸침(`lane_departure`) 판정은 자차 속도가 60km/h 이상일 때만 평가한다
(저속 정체·골목에서 오탐 방지, LDWS 규정 근거). 이 속도를 받을 입력 경로는
[`src/input/sensors.py`](../src/input/sensors.py)에 만들어 놨지만, **ESP32가
UART로 보내는 실제 프레임 포맷(필드 구성, 보율, 타임스탬프 동기화 방식)을
몰라서 비워 둔 스텁 상태**다. 알려주면 `SerialGPSIMUSource` 하나만 채우면
끝난다 — 지금 당장은 Pi5에서 GPS 속도가 없는 걸로 치고(`unavailable`) 돌아가고,
그 결과 `lane_departure`만 항상 판정 보류되며 나머지 4개 단서(사행·스웨빙·표류·
속도불규칙)는 정상 동작한다.

IMU(yaw rate 등)는 지금 어떤 판정에도 안 쓴다 — 자차 흔들림 보정 공식 자체가
아직 연구/근거가 없어서다. GPS 속도만 먼저 필요하다.

### 3.3 이벤트 클립(링 버퍼) — 요청/반환 형태 미정

HW 아키텍처상 **링 버퍼는 HW 파트가 구현**하고(5초 세그먼트×12개 순환, 총
60초), 이벤트 발생 시 우리 쪽이 "전후 15초 클립 주세요"라고 요청하면 반환받는
구조로 설계돼 있다. 우리가 기대하는 요청 값은:

- 이벤트 발생 시각(Unix timestamp)
- 이전 10초, 이후 5초 (`configs/default.yaml`의 `event.clip_pre_sec`/`clip_post_sec`)

이 요청을 **어떤 형태(함수 호출 / IPC / 파일)로 주고받을지**가 아직 안
정해졌다. [`src/event/clip.py`](../src/event/clip.py)에 그 계약(`ClipProvider`)만
정의해 뒀고, 지금은 항상 "클립 없음"을 반환하는 자리채움(`NullClipProvider`)으로
동작 중이다. 형태가 정해지면 이 Protocol을 구현하는 클래스 하나만 추가하면
된다(우리 쪽 메인 로직은 안 바뀜).

---

## 4. 다운로드해야 할 데이터셋 (AI Hub)

**지금 당장 Pi5 실행엔 필요 없다** — 이건 모델을 국내 도로 환경에 맞게
**파인튜닝**할 때 쓰는 데이터다. 다운로드·학습은 팀원 데스크탑(RTX 4070,
2TB)에서 진행하기로 확정했다(2026-09-28). 전체 절차·주의사항은
[`docs/gpu-desktop-setup.md`](gpu-desktop-setup.md)에 다 정리해 뒀고, 여기는
"뭘 받아야 하는지"만 요약한다.

| 데이터셋 | 용량 | 용도 | 우선순위 |
|---|---|---|---|
| [도로주행영상](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=180) | 277.89GB | 차량 검출(YOLO) 파인튜닝/검증 | 1 |
| [도로 로드마크 인식을 위한 주행 영상 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=654) | 1005.77GB | 차선 인식 검증/튜닝 | 1 |
| [자동차 차종/연식/번호판 인식용 영상](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=172) | 231.15GB | PRIVACY 모듈(번호판 블러) 파인튜닝 | 1 |
| [교통사고 영상 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=597) | 3.79TB | 급접근 등 참고용, 여유 있을 때만 | 2(보류) |

**다운로드 시 주의 — 공식 `aihubshell`에 버그 2개 있음.** 그대로 쓰면 (1)
다운로드가 중간에 끊겨도 "성공"으로 처리되고 (2) 멀티파트 대용량 파일 병합
순서가 뒤섞여 파일이 깨진다. 반드시 이 저장소의 패치판을 쓸 것:

```bash
chmod +x scripts/aihubshell_fixed.sh
./scripts/aihubshell_fixed.sh -mode l -datasetkey 180            # 파일 목록 확인
./scripts/aihubshell_fixed.sh -aihubapikey "$(cat ~/.aihub_api_key)" \
    -mode d -datasetkey 180 -filekey <filekey1,filekey2,...>     # 선택 다운로드
```

- API 키 발급: AI Hub 로그인 → 마이페이지
- 압축 해제 시 한글 파일명 깨짐(CP949) 대응, 무결성 확인 절차 등 세부 사항은
  [`docs/gpu-desktop-setup.md`](gpu-desktop-setup.md) 6~8절 참고
- 라벨(XML) 다운로드 후 YOLO 포맷 변환은 `scripts/voc_to_yolo.py`로 이미 실제
  라벨 11,588개 검증까지 끝나 있음

**BDD100K**도 참고용으로 쓸 수 있다(AI Hub 승인 대기 중 선행용, 국내 데이터가
아니라 판단 기준 검증에는 안 씀) — `docs/sprint-plan.md` S1 표 참고.

---

## 5. 지금 상태 요약 (한눈에)

- ✅ 파이프라인 전체(입력→검출→차선→추적→판정→이벤트) Pi5에서 실행 가능
- ✅ ONNX·NCNN 변환 둘 다 가능, 벤치마크 도구 있음
- ⚠️ 모델은 아직 사전학습 그대로 — 정확도 낮음, 파인튜닝 전
- ⚠️ GPS 없어서 `lane_departure` 단서는 항상 판정 보류 (3.2절 — ESP32 프로토콜 확정 시 해소)
- ⚠️ 영상 클립 없음 — 이벤트는 메타데이터(JSON)만 생성됨 (3.3절 — 링 버퍼 연동 시 해소)
- ⚠️ 서버 실제 업로드 없음 — 로컬 SQLite 큐에 쌓이기만 함 (서버 API 스펙 확정 필요, 주민규 파트)

막히는 부분 있으면 언제든 물어봐.
