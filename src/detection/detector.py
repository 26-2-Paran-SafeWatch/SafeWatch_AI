"""YOLOv8n 기반 차량 검출. docs/pipeline-architecture.md 3.2 참고.

처리 순서 (문서와 동일):
1. ROI 크롭 — `detection.roi_top_ratio`만큼 위를 자른다 (기본 0 — 가까운 차량이 잘려 꺼 둠, 문서 3.2)
2. YOLOv8n 추론 (confidence·NMS는 ultralytics 내부 처리)
3. 차량 클래스 필터링 — 가중치에 실린 클래스 이름으로 고른다 (`detection.vehicle_class_names`)
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
                if class_id not in self._vehicle_class_ids:
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
