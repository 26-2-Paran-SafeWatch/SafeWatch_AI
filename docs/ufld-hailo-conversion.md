# UFLD 차선 모델 Hailo-8 변환·실행 절차

**작성** HW 파트(강섬희) 변환·Pi 검증, AI 파트(설만수) 저장소 반영
**최종 수정** 2026-11-15

차선 모델(UFLDv2, `ufld_aihub_aug.onnx`)을 Hailo-8 NPU(AI HAT+ 26 TOPS)에서 돌리는 방법이다.
**재학습으로 가중치가 바뀌면 3장 순서대로 다시 변환한다.** 학습은 따로 하지 않고, 같은 가중치를 형식만 바꾼다.
결정 배경은 [`lane-detection-dl-plan.md`](lane-detection-dl-plan.md) D7과 5장 4단계에 있다.

---

## 1. 구조 — NPU 백본 + CPU 뒷부분

```
입력 [1,3,320,1600] ─▶ ResNet18 … /pool/Conv [1,8,10,50] │ Reshape [1,4000] → LayerNorm → Gemm 4000→2048 → Relu
                       └──────── NPU (ufld_aihub_aug.hef) ──┘   → Gemm 2048→58176 → Slice/Reshape → loc_row, exist_row
                                                                └──────────── CPU (ufld_tail.onnx, ONNX Runtime) ─────┘
```

- Hailo 파서는 `/pool/Conv`까지만 변환할 수 있다며 그 위치를 끝 노드로 추천했다. LayerNorm 구간에서 막힌 것으로 보인다.
- 마지막 Gemm은 가중치가 약 1.2억 개(fp32 487MB)라 어차피 칩 안에 들어가지 않는다. NPU에 넣으면 매 프레임 PCIe x1로
  가중치를 실어 날라야 해서 오히려 느려질 수 있다.
- 그래서 계산량 대부분인 백본만 NPU에 두고, 뒷부분은 원본 ONNX에서 떼어낸 `ufld_tail.onnx`로 CPU에서 돌린다.
  뒷부분에 Slice/Reshape가 포함돼 있어 출력 이름과 형상이 원본과 같다 — 해석·안전장치 코드는 그대로 쓴다.

## 2. 환경

| | 내용 |
|---|---|
| 변환 PC | Windows WSL2 **Ubuntu 22.04**, Python 3.10, venv `~/hailo_dfc`, 작업 폴더 `~/hailo_work`, WSL 메모리 10GB(`.wslconfig`) |
| 컴파일러 | Hailo Dataflow Compiler(DFC) **3.34.0** — Developer Zone에서 받을 수 있는 버전이 이것뿐이었다. 짝이 맞는 버전은 3.33이지만 3.34로 만든 HEF가 HailoRT 4.23에서 정상으로 열렸다 |
| GPU | 없음 → 최적화 수준이 자동으로 0 (보정 이미지로 기본 INT8 양자화만, 미세 보정 생략) |
| Pi | Pi 5, Raspberry Pi OS Trixie, `sudo apt install dkms hailo-all`, HailoRT·펌웨어·드라이버 4.23.0, `/dev/hailo0`, PCIe Gen3 x1 |

- Ubuntu 24.04(Python 3.12)는 DFC와 맞지 않는다.
- 최신 opencv는 numpy 2를 끌어와 DFC를 깨뜨린다 — 아래처럼 버전을 고정한다.
- Pi 가상환경에서 `import hailo_platform`이 되어야 한다 (apt로 시스템에 깔리므로 가상환경은 `--system-site-packages`).

## 3. 변환 절차

```bash
source ~/hailo_dfc/bin/activate && cd ~/hailo_work
pip install "opencv-python-headless==4.10.0.84"     # numpy 2 방지
```

**(1) 파싱 — `/pool/Conv`에서 끊는다**
```bash
hailo parser onnx ufld_aihub_aug.onnx --hw-arch hailo8 --end-node-names /pool/Conv
```

**(2) 보정 이미지 256장** — 학습과 같은 전처리(RGB, 1600×533으로 줄이고 아래 320줄), 0~255 float32, NHWC.
우리 블랙박스 영상 프레임을 쓴다(첫 변환은 영상 3개). AI Hub 원본은 쓰지 않았다. 영상 폴더 경로만 바꾼다.
```bash
python - <<'EOF'
import cv2, glob, random, numpy as np
vids = sorted(glob.glob('<블랙박스 영상 폴더>/*.mp4'))
N = 256; per = N // len(vids) + 1; frames = []
for v in vids:
    cap = cv2.VideoCapture(v); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in np.linspace(0, n - 1, per + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if ok: frames.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1600, 533))[-320:])
random.seed(0); random.shuffle(frames)
np.save('calib.npy', np.stack(frames[:N]).astype(np.float32))
EOF
```
`(1600, 533)`·`[-320:]`은 `lane.ufld`의 `input_size`·`crop_ratio`(int(320/0.6)=533)와 같아야 한다. 설정이 바뀌면 함께 바꾼다.

**(3) 모델 스크립트** — ImageNet 정규화를 HEF 안에 넣는다. 그래서 Pi에서는 0~255 RGB uint8을 그대로 넣는다.
```bash
cat > ufld.alls <<'EOF'
normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
context_switch_param(mode=enabled)
allocator_param(timeout=3h)
EOF
```
- `context_switch_param`이 없으면 컴파일러가 모델을 칩에 한 번에 올리는 방식(single context)만 시도한다.
  그 상태로 약 40시간 돌다가 `Resolver didn't find possible solution`으로 실패했다.
- 두 줄을 넣은 뒤에는 2 context로 약 15분 만에 성공했다. 칩 사용률 약 60%.

**(4) 최적화(INT8) → 컴파일**
```bash
hailo optimize ufld_aihub_aug.har --hw-arch hailo8 --calib-set-path calib.npy --model-script ufld.alls
hailo compiler ufld_aihub_aug_optimized.har --hw-arch hailo8      # → ufld_aihub_aug.hef (약 17MB)
```

**(5) CPU 담당 뒷부분 떼어내기**
```bash
python -c "
import onnx
m = onnx.load('ufld_aihub_aug.onnx')
src = [n.output[0] for n in m.graph.node if n.name == '/pool/Conv'][0]
onnx.utils.extract_model('ufld_aihub_aug.onnx', 'ufld_tail.onnx', [src], ['loc_row', 'exist_row'])
"   # → ufld_tail.onnx (약 487MB), 입력 이름은 /pool/Conv의 출력 이름
```

**(6) Pi로 옮기기** — `ufld_aihub_aug.hef`, `ufld_tail.onnx`를 Pi의 `~/SafeWatch_AI/data/models/`에 둔다.
원본 `ufld_aihub_aug.onnx`도 같은 곳에 있어야 5장 비교를 할 수 있다. **모델 파일과 `calib.npy`는 git에 올리지 않는다**(`data/`는 제외 대상).

## 4. 저장소에서 쓰는 법

```bash
python -m src.main --config configs/pi5_ufld_hailo.yaml --source <영상>
```

- 설정: `lane.ufld.backend: hailo`, `hef_path`, `tail_path` ([`configs/pi5_ufld_hailo.yaml`](../configs/pi5_ufld_hailo.yaml)).
  기본값은 `backend: onnx`(CPU 전체)로 기존 동작과 같다. `num_threads`는 hailo일 때 뒷부분(CPU)에 적용된다.
- 구현: [`src/lane/ufld.py`](../src/lane/ufld.py) `HailoUFLDBackend`. 입력 형식이 백엔드마다 달라
  (`onnx` = 정규화 NCHW float, `hailo` = 자르기·RGB만 한 uint8 HWC) `UFLDLaneDetector`가 백엔드의 `input_format`으로 전처리를 고른다.
- `hailortcli parse-hef` 결과: 입력 `input_layer1` UINT8 NHWC(320×1600×3), 출력 `conv21` UINT8 FCR(10×50×8), Multi Context 2개.
  출력은 FLOAT32로 받으면 HailoRT가 양자화를 풀어 준다. (10,50,8) HWC → `transpose(2,0,1)[None]` → (1,8,10,50)로 뒷부분에 넣는다.
- **필수** — VDevice를 만들 때 `scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN`. 빠뜨리면 `HAILO_STREAM_NOT_ACTIVATED`로 출력이 빈다(hailo-apps 공식 코드와 같은 방식).
- 종료할 때 NPU 자원을 해제한다 — `src/main.py`·`scripts/benchmark.py`·`scripts/visualize_demo.py`가 `lane_detector.close()`를 부른다.

## 5. 변환 결과 확인

```bash
python -m scripts.check_hailo_ufld --config configs/pi5_ufld_hailo.yaml --source <영상> --start 0 --frames 100 --step 5
```
같은 프레임을 원본 ONNX(CPU 전체)와 NPU+뒷부분으로 돌려, 자차 좌우 차선의 '있음' 일치율과 위치 차이(격자 칸)와 시간을 낸다.

**첫 측정 (HW 파트, Pi 5, 영상 `2_20200915_074043` 60초부터 10프레임)**

| 항목 | CPU 전체 ONNX | NPU + CPU 뒷부분 |
|---|---|---|
| 전처리 | – | 3.8ms |
| 백본 | 477ms(전체) | 9.4ms (NPU) |
| 뒷부분 FC | (포함) | 42.8ms (CPU, 4스레드) |
| 합계 | 477ms (파이프라인 1.33 FPS, `pi5_ufld.yaml`) | **56ms, 차선만 17.9 FPS** |
| 원본 ONNX와 차선 '있음' 일치 | – | 100% (10프레임) |
| 원본 ONNX와 위치 차이 | – | 평균 0.02칸 / 200칸 |

- `hailortcli run` 단독 처리량 65.6 FPS.
- **아직 쉬운 구간 10프레임뿐이다.** 차선 변경·가림 구간(같은 영상 0~30초)을 포함해 넓게 확인해야 한다 —
  `lane-detection-dl-plan.md` 5장 4단계 통과 조건(양자화 뒤에도 오답 0, 가용률 하락 2%p 이내)은 아직 판정 전.

## 6. 남은 일

- **병목은 CPU 뒷부분 FC(42.8ms)** — 가중치 487MB를 매 프레임 읽는 시간이 대부분이다. ONNX Runtime 동적 양자화(INT8)로
  3~4배 빨라질 것으로 예상하지만 해 보지 않았다. 하면 5장처럼 원본과 비교해 정확도를 확인한다.
- 차량 검출(YOLOv8n) — Ultralytics `format=hailo`로 내보냄(2026-11-15, 폴더 `yolov8n-safewatch_hailo_model`).
  `detection.backend: hailo`로 실행한다 (`src/detection/hailo_yolo.py`).
  **두 모델은 VDevice 하나를 나눠 쓴다** (`src/utils/hailo_device.py`, ROUND_ROBIN). Ultralytics로 HEF를 바로 열면 장치를 따로
  열어 차선 모델과 `HAILO_OUT_OF_PHYSICAL_DEVICES`로 충돌한다. 내보낼 때 NMS 점수 기준은 `conf=0.1`로 맞춘다
  (기본 0.25면 0.1~0.25 박스가 칩에서 빠진다 — 추적의 2차 매칭·먼 영역 검출에 쓰는 박스).
