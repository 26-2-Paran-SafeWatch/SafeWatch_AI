# 이벤트 JSON 스키마 명세

**버전** 0.1 (협의 중)
**작성** 설만수 (AI 파트)
**협의 대상** 주민규 (앱·서버 파트)
**최종 수정** 2026-09-10

---

## 이 문서의 성격

이 스키마는 **AI 모듈과 앱·서버 간의 인터페이스 계약**이다. 확정 이후에는 양측 합의 없이 필드를 추가·삭제·변경하지 않는다. 한쪽이 임의로 변경하면 상대 파트의 작업이 무효화된다.

현재 버전은 협의를 위한 초안이다. 앱 측에서 불필요한 필드나 누락된 필드가 있으면 확정 전에 반영한다.

---

## 1. 전송 단위

이벤트 1건은 **JSON 메타데이터 1개 + 영상 클립 파일 1개**로 구성된다.

JSON에 영상을 직접 담지 않으며, 두 가지 전송 방식 중 하나를 서버 구조에 맞춰 선택한다.

| 방식 | 순서 | 비고 |
|---|---|---|
| A안 | JSON 전송 → 서버 응답 → 클립 multipart 업로드 | 메타데이터 우선 확보 |
| B안 | 클립 업로드 → URL 수신 → URL 포함 JSON 전송 | 클립 유실 방지 |

**결정 필요** — 서버 구조에 따라 앱·서버 파트에서 선택한다.

---

## 2. 스키마 전문

```json
{
  "event_id": "evt_20260904_143022_001",
  "device_id": "safewatch-001",
  "timestamp": "2026-09-04T14:30:22.451+09:00",

  "risk": {
    "score": 78,
    "level": "high",
    "types": ["lane_departure", "weaving"]
  },

  "indicators": {
    "lane_offset_max_ratio": 0.34,
    "lane_departure_duration_sec": 2.8,
    "direction_changes_count": 4,
    "observation_window_sec": 10.0,
    "longitudinal_accel_peak_g": -0.42,
    "heading_change_deg": 12.3
  },

  "target_vehicle": {
    "track_id": 17,
    "tracked_duration_sec": 12.4,
    "avg_detection_confidence": 0.87
  },

  "location": {
    "latitude": 37.3219,
    "longitude": 126.8309,
    "speed_kmh": 62.5,
    "heading": 271.3
  },

  "clip": {
    "filename": "evt_20260904_143022_001.mp4",
    "duration_sec": 15,
    "pre_event_sec": 10,
    "post_event_sec": 5,
    "resolution": "1280x720",
    "fps": 15,
    "size_bytes": 3821004,
    "blurred": true
  },

  "meta": {
    "model_version": "yolov8n-safewatch-v0.3",
    "rule_version": "risk-v1.2",
    "confidence_gate_passed": true
  }
}
```

---

## 3. 필드 명세

### 3.1 최상위

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `event_id` | string | ✓ | 이벤트 고유 식별자. 형식: `evt_YYYYMMDD_HHMMSS_NNN` |
| `device_id` | string | ✓ | 디바이스 식별자 |
| `timestamp` | string | ✓ | 이벤트 발생 시각. ISO 8601, 타임존 포함 |

**`event_id`의 역할**

통신 단절 시 로컬 큐에 저장했다가 재전송하는 구조이므로 **중복 전송이 발생할 수 있다.** 서버는 `event_id`로 중복을 판별하여 제거해야 한다. 이는 앱·서버 파트와 합의가 필요한 사항이다.

---

### 3.2 `risk` — 위험 판정 결과

앱 UI에 직접 표시되는 정보다.

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `score` | integer | ✓ | 0~100 위험 점수 |
| `level` | string | ✓ | `low` / `medium` / `high` |
| `types` | string[] | ✓ | 감지된 위험 유형 목록 |

**`level`을 별도로 두는 이유**

점수 구간과 등급의 매핑을 AI 모듈이 담당한다. 앱이 점수 구간을 알 필요가 없으므로, 향후 구간 기준이 조정되어도 앱 코드를 수정하지 않아도 된다.

**`types`가 배열인 이유**

차선 이탈과 사행이 동시에 발생할 수 있다. 실제로 본 시스템은 2개 이상의 지표가 동시 충족될 때만 이벤트를 생성하므로, 대부분의 경우 2개 이상의 값을 갖는다.

**`types` 허용값**

| 값 | 의미 |
|---|---|
| `lane_departure` | 차선 걸침·이탈 |
| `weaving` | 사행운전 (좌우 반복 이동) |
| `sudden_decel` | 급감속 |
| `sudden_accel` | 급가속 |
| `abrupt_lane_change` | 급진로변경 |

**이 목록은 고정한다.** 새 유형 추가 시 양측 합의 후 문서를 갱신한다.

---

### 3.3 `indicators` — 판단 근거 수치

앱 상세 화면에서 "왜 위험하다고 판단했는지"를 보여줄 때 사용한다.

| 필드 | 타입 | 단위 | 설명 |
|---|---|---|---|
| `lane_offset_max_ratio` | float | 비율 | 관측 구간 내 최대 차선 이탈 비율 (차로 폭 대비) |
| `lane_departure_duration_sec` | float | 초 | 차선 걸침 지속 시간 |
| `direction_changes_count` | integer | 회 | 관측 구간 내 좌우 방향 전환 횟수 |
| `observation_window_sec` | float | 초 | 판단에 사용된 관측 구간 길이 |
| `longitudinal_accel_peak_g` | float | g | 종방향 가속도 최댓값 (음수는 감속) |
| `heading_change_deg` | float | 도 | 방위각 변화량 |

**필수 여부**

해당 지표가 판단에 사용되지 않은 경우 필드를 생략한다. 예를 들어 차선 인식 신뢰도가 낮아 offset 기반 지표를 제외한 경우 `lane_offset_max_ratio`와 `lane_departure_duration_sec`은 포함되지 않는다.

**앱에서 사용하지 않더라도 유지해야 하는 이유**

심사의견에서 AI 판단 기준의 검증이 요구되었다. 판단 근거를 데이터로 남기는 것이 요구사항이며, 신고서 작성 시 "차선을 2.8초간 침범"과 같은 구체적 서술에도 활용할 수 있다.

---

### 3.4 `target_vehicle` — 대상 차량 정보

| 필드 | 타입 | 설명 |
|---|---|---|
| `track_id` | integer | 세션 내 차량 추적 ID |
| `tracked_duration_sec` | float | 연속 추적된 시간 |
| `avg_detection_confidence` | float | 평균 검출 신뢰도 (0.0~1.0) |

**신뢰도 참고용 필드**

`tracked_duration_sec`이 짧으면 잠깐 스쳐간 차량이므로 판단 신뢰도가 낮다. 앱에서 표시하지 않더라도 디버깅과 성능 분석에 활용된다.

`track_id`는 **세션 내에서만 유효**하며 재부팅 시 초기화된다. 차량을 전역적으로 식별하는 값이 아니다.

---

### 3.5 `location` — 위치 정보

| 필드 | 타입 | 단위 | 설명 |
|---|---|---|---|
| `latitude` | float | 도 | 위도 (WGS84) |
| `longitude` | float | 도 | 경도 (WGS84) |
| `speed_kmh` | float | km/h | 자차 속도 |
| `heading` | float | 도 | 자차 진행 방위각 (0~360) |

**주소 변환은 서버 또는 앱에서 수행한다.** Pi5에서 reverse geocoding을 수행하면 네트워크 왕복이 추가되어 이벤트 생성이 지연된다.

**GPS 미수신 시** — 터널·지하 구간에서는 위치 정보를 획득하지 못할 수 있다. 이 경우 `location` 객체 전체를 생략하거나 마지막 유효 좌표에 `stale: true` 플래그를 붙이는 방안을 검토 중이다. **협의 필요.**

---

### 3.6 `clip` — 영상 클립 정보

| 필드 | 타입 | 설명 |
|---|---|---|
| `filename` | string | 클립 파일명 |
| `duration_sec` | integer | 전체 길이 |
| `pre_event_sec` | integer | 이벤트 시점 이전 길이 |
| `post_event_sec` | integer | 이벤트 시점 이후 길이 |
| `resolution` | string | `WIDTHxHEIGHT` |
| `fps` | integer | 프레임 레이트 |
| `size_bytes` | integer | 파일 크기 |
| `blurred` | boolean | 비식별화 처리 여부 |

**`pre_event_sec`과 `post_event_sec`을 분리하는 이유**

이벤트 발생 시점 **이전** 영상이 증거로서 더 중요하다. 감지되기까지의 거동이 위험 판단의 근거이기 때문이다. 기본값은 전 10초 + 후 5초다.

**`blurred` 플래그**

비식별화를 Edge에서 수행할지 서버에서 수행할지가 미결 상태다. 이 플래그를 통해 서버가 추가 처리 필요 여부를 판단할 수 있다.

---

### 3.7 `meta` — 버전 정보

| 필드 | 타입 | 설명 |
|---|---|---|
| `model_version` | string | 사용된 모델 버전 |
| `rule_version` | string | 판단 규칙 버전 |
| `confidence_gate_passed` | boolean | 입력 신뢰도 검사 통과 여부 |

**버전 정보가 필요한 이유**

모델과 규칙은 프로젝트 기간 중 여러 차례 갱신된다. 버전을 남기지 않으면 "지난주에는 잘 잡혔는데 지금은 왜 안 잡히는지" 추적이 불가능하다. 성능 비교와 재학습 시점 판단에도 필수적이다.

---

## 4. 공통 규약

| 항목 | 규약 |
|---|---|
| 시간 | ISO 8601, 타임존 포함 (KST 고정) |
| 좌표 | WGS84 소수점 표기 |
| 속도 | km/h |
| 가속도 | g 단위 |
| 각도 | 도(degree), 0~360 |
| 인코딩 | UTF-8 |
| 필드명 | snake_case |

**단위 통일이 중요한 이유** — 단위 불일치는 통합 단계에서 발견하기 어려운 버그를 만든다. 확정 후 변경하지 않는다.

---

## 5. 협의 필요 사항

| 번호 | 항목 | 내용 | 담당 |
|---|---|---|---|
| 1 | 전송 방식 | A안(JSON 우선) vs B안(클립 우선) | 앱·서버 |
| 2 | 중복 제거 | 서버가 `event_id`로 중복 판별 | 앱·서버 |
| 3 | GPS 미수신 처리 | 필드 생략 vs `stale` 플래그 | 양측 |
| 4 | 비식별화 위치 | Edge vs 서버 | 양측 |
| 5 | 불필요 필드 | 앱에서 사용하지 않는 필드 확인 | 앱·서버 |
| 6 | 누락 필드 | 앱에 추가로 필요한 정보 확인 | 앱·서버 |
| 7 | 클립 사양 | 해상도·코덱·최대 용량 제한 | 양측 |

---

## 6. 개발 지원

앱·서버 파트의 선행 개발을 위해 **더미 이벤트 생성기**를 제공한다.

```bash
python scripts/generate_dummy_event.py --count 10 --output samples/
```

실제 AI 파이프라인 완성 전에도 앱 측에서 이벤트 수신·표시 로직을 구현할 수 있다. S4(7~8주차) 완료 시점에 제공 예정이다.

---

## 7. 변경 이력

| 버전 | 일자 | 내용 |
|---|---|---|
| 0.1 | 2026-09-10 | 초안 작성. 앱·서버 파트 협의 대기 |
