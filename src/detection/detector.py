"""YOLOv8n 기반 차량 검출. docs/pipeline-architecture.md 3.2 참고.

처리 순서 (문서와 동일):
1. ROI 크롭 — `detection.roi_top_ratio`만큼 위를 자른다 (기본 0 — 가까운 차량이 잘려 꺼 둠, 문서 3.2)
2. YOLOv8n 추론 (confidence·NMS는 ultralytics 내부 처리)
3. 차량 클래스 필터링 — 가중치에 실린 클래스 이름으로 고른다 (`detection.vehicle_class_names`)
4. 좌표를 원본 프레임 기준으로 역변환
5. (선택) 먼 영역 한 번 더 — `detection.far_region`. 화면 가운데 위쪽(먼 차가 있는 곳)만 잘라
   같은 모델에 한 번 더 넣고, 전체 화면 검출과 겹치지 않는 것만 더한다
6. 전방 차량 필터링 — 화면 중앙 영역 우선

AGENTS.md 원칙 — YOLOv8n 이상 크기 모델 사용 금지, 설정값은 config로 분리.
"""

from __future__ import annotations

import logging

import numpy as np

from src.detection.types import VehicleBox
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.detection")


class VehicleDetector:
    def __init__(self, cfg: Config):
        # ultralytics는 import 비용이 커서 detector 생성 시점에만 로드한다.
        from ultralytics import YOLO

        det_cfg = cfg.detection
        self._conf_threshold: float = det_cfg.conf_threshold
        self._iou_threshold: float = det_cfg.iou_threshold
        self._roi_top_ratio: float = det_cfg.roi_top_ratio
        self._center_region_ratio: float = det_cfg.center_region_ratio

        far = det_cfg.far_region
        self._far_enabled: bool = bool(far.enabled)
        self._far_x: tuple[float, float] = (float(far.x_ratio[0]), float(far.x_ratio[1]))
        self._far_y: tuple[float, float] = (float(far.y_ratio[0]), float(far.y_ratio[1]))
        self._far_edge_ratio: float = float(far.edge_ratio)
        self._far_merge_iou: float = float(far.merge_iou)
        for name, (a, b) in (("x_ratio", self._far_x), ("y_ratio", self._far_y)):
            if not 0.0 <= a < b <= 1.0:
                raise ValueError(f"detection.far_region.{name}는 0 ≤ 시작 < 끝 ≤ 1이어야 한다: {(a, b)}")
        # 먼 영역 박스가 전체 화면 박스를 대신할지 정하는 기준 — 새 트랙을 만들 수 있는 신뢰도
        self._track_high_thresh: float = float(cfg.tracking.track_high_thresh)

        logger.info("YOLO 가중치 로드: %s", det_cfg.weights)
        self._model = YOLO(det_cfg.weights, task="detect")
        self._vehicle_class_ids = self._resolve_vehicle_classes(det_cfg.vehicle_class_names)

    def _resolve_vehicle_classes(self, wanted: list[str]) -> frozenset[int]:
        """가중치에 실린 클래스 이름 중 차량으로 볼 것의 id.

        id가 아니라 이름으로 고르는 이유 — 사전학습(COCO) 가중치는 car=2·bus=5…이고
        파인튜닝 가중치는 vehicle=0 하나뿐이다. id를 config에 적어 두면 가중치만
        바꾸고 id를 안 바꿨을 때 차량이 하나도 안 잡히는데, 에러 없이 조용히
        판정 전체가 멈춘다. 이름은 .pt·ONNX·NCNN 모두 가중치 메타데이터에 실려 온다.
        """
        names: dict[int, str] = self._model.names
        ids = frozenset(i for i, name in names.items() if name in set(wanted))
        if not ids:
            # 프레임마다 실패하는 것보다 시작 시점에 멈추는 편이 원인을 찾기 쉽다
            raise ValueError(
                f"가중치의 클래스 {sorted(names.values())[:10]}… 중 "
                f"detection.vehicle_class_names {wanted}와 일치하는 것이 없다"
            )
        logger.info("차량 클래스: %s", {i: names[i] for i in sorted(ids)})
        return ids

    def detect(self, image: np.ndarray) -> list[VehicleBox]:
        """프레임에서 전방 차량을 검출한다. 좌표는 원본 프레임 기준."""
        with profiler.stage("detection"):
            height, width = image.shape[:2]
            roi_y0 = int(height * self._roi_top_ratio)
            boxes = self._predict(image[roi_y0:, :], 0, roi_y0)

            if self._far_enabled:
                x0, x1 = int(width * self._far_x[0]), int(width * self._far_x[1])
                y0, y1 = int(height * self._far_y[0]), int(height * self._far_y[1])
                far = self._predict(image[y0:y1, x0:x1], x0, y0)
                # 잘린 경계에 닿은 박스는 몸통이 잘린 차 — 전체 화면 검출이 맡는다
                mx, my = width * self._far_edge_ratio, height * self._far_edge_ratio
                far = [
                    b for b in far
                    if b.x1 > x0 + mx and b.x2 < x1 - mx and b.y1 > y0 + my and b.y2 < y1 - my
                ]
                boxes = merge_far_boxes(boxes, far, self._far_merge_iou, self._track_high_thresh)

            return self._filter_forward_vehicles(boxes, width)

    def _predict(self, image: np.ndarray, x_off: int, y_off: int) -> list[VehicleBox]:
        """한 장 추론 → 차량 박스. (x_off, y_off)만큼 더해 원본 프레임 좌표로 되돌린다."""
        results = self._model.predict(
            image,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )
        boxes: list[VehicleBox] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id not in self._vehicle_class_ids:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            boxes.append(
                VehicleBox(
                    x1=x1 + x_off,
                    y1=y1 + y_off,
                    x2=x2 + x_off,
                    y2=y2 + y_off,
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                )
            )
        return boxes

    def _filter_forward_vehicles(
        self, boxes: list[VehicleBox], frame_width: int
    ) -> list[VehicleBox]:
        """화면 중앙 영역(전방 차로로 추정)에 속한 차량만 남긴다.

        인접·반대 차로의 차량, 갓길에 정차한 차량 등을 배제해 위험 판단
        대상을 자차 전방 차량으로 좁히기 위함 (sprint-plan.md S2 참고).
        """
        margin = frame_width * (1 - self._center_region_ratio) / 2
        left_bound, right_bound = margin, frame_width - margin
        return [b for b in boxes if left_bound <= b.center_x <= right_bound]


def _iou(a: VehicleBox, b: VehicleBox) -> float:
    iw = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    ih = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = iw * ih
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def merge_far_boxes(
    full: list[VehicleBox], far: list[VehicleBox], merge_iou: float, high_thresh: float
) -> list[VehicleBox]:
    """전체 화면 검출에 먼 영역 검출을 더한다.

    - 전체 화면 박스와 IoU `merge_iou` 미만으로만 겹치는 먼 영역 박스는 그대로 더한다.
      전체 화면 박스끼리는 다시 거르지 않는다 — YOLO가 이미 NMS를 했고, 다시 거르면
      붙어 있는 차 둘 중 하나가 지워질 수 있다 (노트북 safewatch_vehicle_finetune 9번 셀).
    - 겹치면 같은 차다. 보통은 전체 화면 박스를 남기는데, 먼 영역 박스만 새 트랙을 만들 수 있는
      신뢰도(`high_thresh` 이상)이고 겹친 전체 화면 박스가 전부 그 아래면 먼 영역 박스로 바꾼다 —
      먼 차를 작게 봐서 확신이 낮았던 것을 확대해서 확신한 경우라, 바꾸지 않으면 트랙이 안 생긴다.
    """
    out = list(full)
    for b in sorted(far, key=lambda v: -v.confidence):
        over = [i for i, f in enumerate(out) if _iou(b, f) >= merge_iou]
        if not over:
            out.append(b)
        elif b.confidence >= high_thresh and all(out[i].confidence < high_thresh for i in over):
            out = [f for i, f in enumerate(out) if i not in set(over)] + [b]
    return out
