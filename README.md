# SafeWatch AI

차량 전방 카메라 영상에서 위험운전 의심 거동을 실시간 감지하는 엣지 AI 파이프라인.

라즈베리파이5에서 온디바이스로 동작하며, 위험 거동이 감지되면 해당 구간의 영상 클립과 메타데이터를 이벤트로 생성한다. 운전자는 안전한 곳에 정차한 뒤 저장된 영상을 확인하고 신고 여부를 직접 판단한다.

> **AI는 신고를 자동으로 실행하지 않는다.** 위험 거동이 관측된 구간을 후보로 제시하고 증거를 자동으로 확보해 주는 신고 보조 도구이며, 최종 판단은 항상 사람이 수행한다.

---

## 프로젝트 정보

- 아주대학교 2026-2학기 파란학기제 · SafeWatch 팀
- 이 저장소: **AI 파트** (영상 분석 · 위험 판단 · 모델 경량화)
- 관련 파트는 별도 저장소로 관리
  - HW — 라즈베리파이5, ESP32, 카메라·센서 연동
  - 앱·서버 — 이벤트 수신, 알림, 신고 보조 인터페이스

---

## 동작 흐름

```
카메라 프레임
    │
    ├─→ 차량 검출 (YOLOv8n)
    ├─→ 차선 인식 (OpenCV)
    │
    ├─→ 차량 추적 (ByteTrack) — track_id 유지
    │
    ├─→ lane offset 산출 — 차량 중심과 차선 중심의 거리
    │
    ├─→ 위험 점수 산출 (rule-based)
    │      · 차선 걸침 지속 시간
    │      · 좌우 방향 전환 횟수 (사행)
    │      · 급가감속 (IMU)
    │      → 2개 이상 지표 동시 충족 시 이벤트 생성
    │
    └─→ 이벤트 출력 — JSON + 영상 클립 (비식별화 처리)
```

상세 구조는 [`docs/pipeline-architecture.md`](docs/pipeline-architecture.md) 참고.

---

## 기술 스택

| 영역 | 기술 |
|---|---|
| 언어 | Python 3.11 |
| 객체 검출 | YOLOv8n (Ultralytics) |
| 차선 인식 | OpenCV |
| 객체 추적 | ByteTrack |
| 위험 판단 | Rule-based (설명 가능성 확보 목적) |
| 추론 엔진 | ONNX Runtime / NCNN |
| 카메라 입력 | picamera2 (Pi5) / cv2.VideoCapture (PC) |

**타깃 하드웨어** — Raspberry Pi 5 (8GB). NPU가 없어 CPU 추론만 가능하므로, 모델 경량화와 파이프라인 최적화가 핵심 과제다.

---

## 설치

```bash
git clone https://github.com/26-2-Paran-SafeWatch/SafeWatch_AI.git
cd SafeWatch_AI

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Pi5 추가 설정

```bash
# picamera2는 시스템 패키지로 설치
sudo apt install -y python3-picamera2

# 가상환경에서 시스템 패키지 접근이 필요한 경우
python -m venv .venv --system-site-packages
```

---

## 실행

### 영상 파일로 실행 (PC 개발)

```bash
python -m src.main --config configs/dev.yaml --source data/raw/sample.mp4
```

### 카메라로 실행 (Pi5)

```bash
python -m src.main --config configs/pi5.yaml --source camera
```

### 성능 벤치마크

```bash
python scripts/benchmark.py --config configs/pi5.yaml
```

단계별 처리 시간, fps, CPU 온도를 측정하여 `results/benchmark/`에 기록한다.

### 성능 평가

```bash
python scripts/evaluate.py --pred results/ --gt data/labels/
```

Precision, Recall, F1을 산출한다.

---

## 설정

튜닝 대상 값은 모두 `configs/` 아래 YAML로 관리한다. 코드 수정 없이 설정만 바꿔 실험할 수 있다.

| 파일 | 용도 |
|---|---|
| `default.yaml` | 공통 기본값 |
| `dev.yaml` | PC 개발용 (고해상도, 매 프레임 검출) |
| `pi5.yaml` | Pi5 실기기용 (저해상도, 검출 주기 분리) |

주요 설정 항목

```yaml
input:
  resolution: [320, 320]
  fps: 15

detection:
  model: yolov8n
  interval: 3              # N프레임마다 검출, 사이는 추적으로 보간
  conf_threshold: 0.5

risk:
  observation_window_sec: 10
  min_indicators: 2        # 이벤트 생성에 필요한 최소 지표 수
  # 임계값 상세는 docs/risk-criteria.md 참고

event:
  clip_pre_sec: 10
  clip_post_sec: 5
  cooldown_sec: 60
```

---

## 디렉터리 구조

```
safewatch-ai/
├── configs/         설정 파일
├── src/
│   ├── input/       영상 입력 (플랫폼 추상화)
│   ├── detection/   차량 검출
│   ├── lane/        차선 인식
│   ├── tracking/    차량 추적
│   ├── metrics/     lane offset, 시계열 버퍼
│   ├── risk/        위험 점수 산출
│   ├── event/       이벤트 생성, 클립 추출, 전송 큐
│   ├── privacy/     비식별화 처리
│   └── utils/       로깅, 프로파일링, config
├── scripts/         벤치마크, 모델 변환, 평가
├── tests/
├── docs/
└── data/            (gitignore)
```

---

## 성능 목표

| 지표 | 목표 |
|---|---|
| Pi5 파이프라인 처리 속도 | 10 fps 이상 |
| 감지 후 이벤트 저장 완료까지 지연 | 3초 이내 |
| 위험운전 감지 정밀도 (Precision) | 90% 이상 |
| 위험운전 감지 재현율 (Recall) | 80% 이상 |
| 오탐률 | 10% 이하 |
| 연속 구동 무중단 시간 | 4시간 이상 |

오탐이 무고한 신고로 이어질 수 있으므로 **재현율보다 정밀도를 우선**한다.

---

## 개인정보 처리

- 이벤트 클립은 전송 전 번호판·보행자 얼굴을 자동 비식별화(블러) 처리한다
- 학습용 수집 영상도 라벨링 전 동일하게 비식별화한다
- 원본 영상은 암호화 보관하며 연구 목적 외 사용하지 않고, 프로젝트 종료 후 파기한다
- `data/` 디렉터리는 저장소에 커밋하지 않는다

---

## 문서

| 문서 | 내용 |
|---|---|
| [`AGENTS.md`](AGENTS.md) | AI 코딩 에이전트용 프로젝트 규칙 |
| [`docs/pipeline-architecture.md`](docs/pipeline-architecture.md) | 파이프라인 구조와 모듈 간 데이터 흐름 |
| [`docs/event-schema.md`](docs/event-schema.md) | 이벤트 JSON 스키마 (앱·서버 인터페이스) |
| [`docs/risk-criteria.md`](docs/risk-criteria.md) | 위험 판단 기준과 근거 |

---

## 라이선스 및 사용 데이터

- YOLOv8 — Ultralytics, AGPL-3.0
- 학습 데이터 — AI Hub 국내 주행 데이터셋, BDD100K, 자체 수집 실차 영상

각 데이터셋의 이용 약관을 준수한다.
