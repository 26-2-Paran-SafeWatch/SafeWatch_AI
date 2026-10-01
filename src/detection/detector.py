"""YOLOv8n 기반 차량 검출. docs/pipeline-architecture.md 3.2 참고.

처리 순서 (문서와 동일):
1. ROI 크롭 — 하늘 영역 제외
2. YOLOv8n 추론 (confidence·NMS는 ultralytics 내부 처리)
3. 차량 클래스 필터링 (COCO: car/motorcycle/bus/truck)
4. 좌표를 원본 프레임 기준으로 역변환
5. 전방 차량 필터링 — 화면 중앙 영역 우선

AGENTS.md 원칙 — YOLOv8n 이상 크기 모델 사용 금지, 설정값은 config로 분리.
"""

from __future__ import annotations

import logging

import numpy as np

from src.detection.types import VehicleBox
from src.utils.config import Config
from src.utils.profiler import profiler

logger = logging.getLogger("safewatch.detection")

# COCO 클래스 중 차량으로 취급할 클래스 id
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


class VehicleDetector:
    def __init__(self, cfg: Config):
        # ultralytics는 import 비용이 커서 detector 생성 시점에만 로드한다.
        from ultralytics import YOLO

        det_cfg = cfg.detection
        self._conf_threshold: float = det_cfg.conf_threshold
        self._iou_threshold: float = det_cfg.iou_threshold
        self._roi_top_ratio: float = det_cfg.roi_top_ratio
        self._center_region_ratio: float = det_cfg.center_region_ratio

        logger.info("YOLO 가중치 로드: %s", det_cfg.weights)
        self._model = YOLO(det_cfg.weights)

    def detect(self, image: np.ndarray) -> list[VehicleBox]:
        """프레임에서 전방 차량을 검출한다. 좌표는 원본 프레임 기준."""
        with profiler.stage("detection"):
            height, width = image.shape[:2]
            roi_y0 = int(height * self._roi_top_ratio)
            roi = image[roi_y0:, :]

            results = self._model.predict(
                roi,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                verbose=False,
            )

            boxes: list[VehicleBox] = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                boxes.append(
                    VehicleBox(
                        x1=x1,
                        y1=y1 + roi_y0,  # ROI 크롭 만큼 y 좌표 역변환
                        x2=x2,
                        y2=y2 + roi_y0,
                        confidence=float(box.conf[0]),
                        class_id=class_id,
                    )
                )

            return self._filter_forward_vehicles(boxes, width)

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
