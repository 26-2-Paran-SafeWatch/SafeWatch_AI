# AGENTS.md

이 파일은 Claude Code 등 AI 코딩 에이전트가 이 저장소에서 작업할 때 따라야 할 규칙을 정의한다.

---

## 프로젝트 개요

**SafeWatch AI 모듈** — 차량 전방 카메라 영상에서 위험운전 의심 거동을 실시간 감지하는 엣지 AI 파이프라인.

라즈베리파이5에서 온디바이스로 동작하며, 위험 거동이 감지되면 영상 클립과 메타데이터를 이벤트로 생성해 서버로 전송한다. 최종 신고 판단은 사용자가 수행하며, **AI는 판단 보조 도구일 뿐 자동 신고를 실행하지 않는다.**

- 소속: 아주대학교 2026-2학기 파란학기제 SafeWatch 팀
- 이 저장소 담당: AI 파트 (설만수)
- 관련 파트: HW(라즈베리파이5·ESP32), 앱·서버 — **별도 저장소**

---

## 절대 원칙

### 1. 성능 예산을 항상 의식한다

타깃 하드웨어는 **NPU가 없는 라즈베리파이5**다. PC에서 잘 도는 코드가 Pi5에서는 4~5배 느려진다.

- 프레임당 전체 처리 시간 목표: **100ms 이하 (10fps)**
- 무거운 연산을 추가할 때는 반드시 처리 시간 측정 코드를 함께 넣는다
- 딥러닝 모델을 하나 더 추가하는 제안은 하지 않는다 (YOLO 외 추가 모델 = 실시간성 상실)

### 2. 설정값을 하드코딩하지 않는다

해상도, 검출 주기, 임계값, 버퍼 길이 등 튜닝 대상 값은 **반드시 `configs/` 아래 YAML로 분리**한다. Pi5에서 코드 수정 없이 값만 바꿔 실험할 수 있어야 한다.

```python
# 나쁨
if offset_ratio > 0.25:

# 좋음
if offset_ratio > cfg.risk.lane_departure.offset_ratio_threshold:
```

### 3. 판단 기준은 근거 없이 바꾸지 않는다

위험 판단의 임계값은 한국교통안전공단 위험운전행동 판별 기준과 NHTSA 음주운전 시각적 판별 지표에 근거해 설정되어 있다. `docs/risk-criteria.md`에 근거가 문서화되어 있다.

**임계값을 임의로 조정하는 코드 변경을 제안하지 않는다.** 조정이 필요하면 근거와 함께 문서 수정을 먼저 제안한다.

### 4. 이벤트 스키마는 계약이다

`docs/event-schema.md`의 JSON 구조는 앱·서버 파트와 합의된 인터페이스다. **필드 추가·삭제·이름 변경을 임의로 하지 않는다.** 변경이 필요하면 코드를 고치기 전에 사용자에게 알린다.

### 5. 개인정보 처리를 우회하지 않는다

- 이벤트 클립은 전송 전 번호판·보행자 얼굴 비식별화 처리를 거친다
- 원본 영상을 비식별화 없이 외부로 내보내는 코드를 작성하지 않는다
- 로그에 GPS 좌표 원본을 남기지 않는다

---

## 기술 스택

| 영역 | 사용 기술 | 비고 |
|---|---|---|
| 언어 | Python 3.11 | |
| 객체 검출 | YOLOv8n (Ultralytics) | **n 이상 크기 모델 사용 금지** |
| 차선 인식 | OpenCV | 딥러닝 모델 사용 안 함 (성능 제약) |
| 객체 추적 | ByteTrack | Ultralytics 내장 |
| 위험 판단 | Rule-based | ML 모델 아님. 설명 가능성이 요구사항 |
| 추론 엔진 | ONNX Runtime / NCNN | 실측 후 결정 |
| 카메라 입력 | picamera2 | Pi5 전용. PC에서는 cv2.VideoCapture로 대체 |
| 설정 관리 | YAML | |
| 테스트 | pytest | |

**사용하지 않는 것**
- 딥러닝 기반 차선 인식 모델 (UFLD, LaneNet 등) — 성능 제약. 단, OpenCV 방식이 최종 실패할 경우 재검토 대상
- YOLOv8s 이상 — Pi5에서 3~5fps로 실시간 불가
- 상시 구동 OCR — 이벤트 시점 단발 실행만 허용

---

## 디렉터리 구조

```
safewatch-ai/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml          # 기본 설정
│   ├── pi5.yaml              # Pi5 실기기용 (저해상도, 검출 주기 증가)
│   └── dev.yaml              # PC 개발용
├── src/
│   ├── input/                # 영상 입력 (picamera2 / VideoCapture 추상화)
│   ├── detection/            # YOLO 차량 검출
│   ├── lane/                 # OpenCV 차선 인식
│   ├── tracking/             # ByteTrack 래퍼
│   ├── metrics/              # lane offset, 시계열 버퍼
│   ├── risk/                 # 위험 점수 산출, 판정 로직
│   ├── event/                # 이벤트 생성, 클립 추출, 전송 큐
│   ├── privacy/              # 비식별화 처리
│   └── utils/                # 로깅, 프로파일링, config 로더
├── scripts/                  # 실행 스크립트 (벤치마크, 변환, 평가)
├── tests/
├── docs/
│   ├── pipeline-architecture.md
│   ├── event-schema.md
│   └── risk-criteria.md
└── data/                     # gitignore 대상
    ├── raw/
    ├── processed/
    └── models/
```

**`data/`는 절대 커밋하지 않는다.** 영상·모델 파일은 용량이 크고 개인정보를 포함할 수 있다.

---

## 코딩 규칙

### 모듈 경계

각 모듈은 명확한 입출력을 가지며 다른 모듈의 내부 구현에 의존하지 않는다.

```
프레임 → [detection] → 차량 박스 목록
프레임 → [lane] → 차선 곡선 + 신뢰도
차량 박스 → [tracking] → track_id 부여된 차량 목록
차량 + 차선 → [metrics] → lane offset 시계열
offset 시계열 + IMU → [risk] → 위험 점수 + 유형
위험 점수 → [event] → 이벤트 JSON + 클립
```

### 처리 시간 측정

주요 단계에는 처리 시간 측정을 넣는다. `src/utils/profiler.py`의 컨텍스트 매니저를 사용한다.

```python
with profiler.stage("detection"):
    boxes = detector.detect(frame)
```

### 타입 힌트

공개 함수·메서드에는 타입 힌트를 붙인다.

### 로깅

`print()` 대신 `logging`을 사용한다. Pi5에서 장시간 구동 시 로그가 성능에 영향을 주므로, 디버그 로그는 레벨로 제어한다.

### 예외 처리

차량 환경에서 장시간 무중단 구동이 요구된다. 프레임 단위 처리 실패가 전체 파이프라인을 중단시키지 않도록 한다. 실패한 프레임은 건너뛰고 로그를 남긴다.

---

## 주요 명령어

> 구현 진행에 따라 갱신한다.

```bash
# 환경 설정
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 영상 파일로 파이프라인 실행 (개발용)
python -m src.main --config configs/dev.yaml --source data/raw/sample.mp4

# Pi5 실기기 실행
python -m src.main --config configs/pi5.yaml --source camera

# 성능 벤치마크
python scripts/benchmark.py --config configs/pi5.yaml

# 성능 평가 (Precision/Recall/F1)
python scripts/evaluate.py --pred results/ --gt data/labels/

# 모델 변환
python scripts/export_model.py --format onnx
python scripts/export_model.py --format ncnn

# 테스트
pytest tests/
```

---

## 개발 환경 차이

PC와 Pi5는 환경이 다르다. 코드는 양쪽에서 모두 동작해야 한다.

| 항목 | PC (개발) | Pi5 (타깃) |
|---|---|---|
| 영상 입력 | `cv2.VideoCapture` (파일) | `picamera2` (카메라) |
| 추론 엔진 | PyTorch | ONNX / NCNN |
| 해상도 | 640 (검증용) | 320 |
| 검출 주기 | 매 프레임 | 3~5프레임마다 |
| IMU | 더미 데이터 | ESP32 UART 수신 |

`src/input/`과 `src/detection/`은 이 차이를 추상화한다. **플랫폼 분기는 이 두 모듈 안에서만 처리하고, 상위 로직에는 노출하지 않는다.**

---

## 작업 시 유의사항

### 하지 말 것

- 이벤트 JSON 스키마 임의 변경
- 판단 기준 임계값 근거 없이 조정
- 딥러닝 모델 추가 제안
- `data/` 커밋
- 비식별화 우회
- 성능 측정 없이 무거운 연산 추가

### 할 것

- 변경 시 처리 시간 영향 언급
- 설정값은 config로 분리
- 판단 로직 변경 시 `docs/risk-criteria.md` 동기화 제안
- Pi5에서의 동작 가능성 검토

### 불확실할 때

Pi5 실기기 성능에 관한 판단이 필요한데 실측 데이터가 없으면, 추정치를 단정하지 말고 **벤치마크 스크립트 실행을 제안**한다.

---

## 참고 문서

| 문서 | 내용 |
|---|---|
| `README.md` | 프로젝트 개요, 설치, 실행 방법 |
| `docs/pipeline-architecture.md` | 파이프라인 구조와 모듈 간 데이터 흐름 |
| `docs/event-schema.md` | 이벤트 JSON 스키마 명세 (앱·서버 인터페이스) |
| `docs/risk-criteria.md` | 위험 판단 기준과 근거 |
