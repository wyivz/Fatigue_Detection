# -*- coding: utf-8 -*-
"""YOLO wrapper for face / smoke / phone / water detection."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from django.conf import settings
from ultralytics import YOLO


def _cfg_float(configs: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(configs.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_int(configs: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(float(configs.get(key, default)))
    except (TypeError, ValueError):
        return default


def _bbox_center(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _bbox_diag(bbox: List[int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))


class YOLODetector:
    """Detect face / smoking / phone / drinking with optional spatial filters."""

    CLASS_NAMES = ["face", "smoke", "phone", "water"]

    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(settings.BASE_DIR, "weights", "best.pt")

        self.load_config()
        self.model = YOLO(weights_path)
        self._apply_runtime_params()
        self.class_names = list(self.CLASS_NAMES)

    def load_config(self):
        from detection.models import SystemConfig

        self.conf_thresh = 0.5
        self.iou_thresh = 0.5
        self.imgsz = 640
        self._imgsz_from_config = False
        self.device = "cpu"
        # Per-class floors (applied after global conf)
        self.conf_face = 0.35
        self.conf_smoke = 0.2
        self.conf_phone = 0.25
        self.conf_water = 0.25
        self.spatial_filter = True
        self.near_face_ratio = 1.2  # max center distance in face-diag units

        try:
            configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
            self.conf_thresh = _cfg_float(configs, "yolo_conf_thresh", 0.5)
            self.iou_thresh = _cfg_float(configs, "yolo_iou_thresh", 0.5)
            if "yolo_imgsz" in configs and str(configs.get("yolo_imgsz", "")).strip():
                self.imgsz = max(320, _cfg_int(configs, "yolo_imgsz", 640))
                self._imgsz_from_config = True
            else:
                self.imgsz = 640
                self._imgsz_from_config = False
            self.device = configs.get("device") or "cpu"
            self.conf_face = _cfg_float(configs, "yolo_conf_face", max(self.conf_thresh, 0.35))
            self.conf_smoke = _cfg_float(configs, "yolo_conf_smoke", min(self.conf_thresh, 0.2))
            self.conf_phone = _cfg_float(configs, "yolo_conf_phone", min(self.conf_thresh, 0.25))
            self.conf_water = _cfg_float(configs, "yolo_conf_water", min(self.conf_thresh, 0.25))
            self.spatial_filter = str(configs.get("yolo_spatial_filter", "true")).lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            self.near_face_ratio = _cfg_float(configs, "yolo_near_face_ratio", 1.2)
            print(
                f"已加载YOLO配置: conf={self.conf_thresh}, iou={self.iou_thresh}, "
                f"imgsz={self.imgsz}, device={self.device}, spatial={self.spatial_filter}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"加载YOLO配置失败: {e}")

        if hasattr(self, "model") and self.model is not None:
            self._apply_runtime_params()

    def _apply_runtime_params(self) -> None:
        try:
            self.model.conf = self.conf_thresh
            self.model.iou = self.iou_thresh
        except Exception:  # noqa: BLE001
            pass
        if self.device:
            try:
                self.model.to(self.device)
            except Exception as e:  # noqa: BLE001
                print(f"无法在设备 {self.device} 上运行模型: {e}")

    def detect(self, image):
        """Run inference.

        Keep call style close to the original (model(image, verbose=False)) so we do not
        force a heavier letterbox/imgsz path than before. conf/iou live on the model object.
        """
        kwargs = {"verbose": False}
        # Only override imgsz when user explicitly configured it in DB
        if getattr(self, "_imgsz_from_config", False):
            kwargs["imgsz"] = int(self.imgsz)
        results = self.model(image, **kwargs)
        return results[0]

    def _class_conf_floor(self, cls_id: int) -> float:
        if cls_id == 0:
            return float(self.conf_face)
        if cls_id == 1:
            return float(self.conf_smoke)
        if cls_id == 2:
            return float(self.conf_phone)
        if cls_id == 3:
            return float(self.conf_water)
        return float(self.conf_thresh)

    def process_results(self, results):
        processed_data = {
            "face_detected": False,
            "smoking_detected": False,
            "phone_detected": False,
            "drinking_detected": False,
            "detections": [],
            "raw_detections": [],
            "filtered_out": [],
            "behavior_debug": {},
        }

        boxes = results.boxes.cpu().numpy()
        raw: List[Dict[str, Any]] = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id < 0 or cls_id >= len(self.class_names):
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            det = {
                "class_id": cls_id,
                "class_name": self.class_names[cls_id],
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
            raw.append(det)
        processed_data["raw_detections"] = raw

        # Per-class confidence floors
        kept: List[Dict[str, Any]] = []
        for det in raw:
            floor = self._class_conf_floor(det["class_id"])
            if det["confidence"] < floor:
                processed_data["filtered_out"].append(
                    {**det, "reason": f"conf<{floor:.2f}"}
                )
                continue
            kept.append(det)

        face_boxes = [d for d in kept if d["class_id"] == 0]
        face_bbox = face_boxes[0]["bbox"] if face_boxes else None

        final: List[Dict[str, Any]] = []
        for det in kept:
            if det["class_id"] == 0:
                final.append(det)
                continue
            if self.spatial_filter and face_bbox is not None:
                ok, reason = self._near_face(det, face_bbox)
                if not ok:
                    processed_data["filtered_out"].append({**det, "reason": reason})
                    continue
            elif self.spatial_filter and face_bbox is None and det["class_id"] in (1, 2, 3):
                # No face: keep phone/smoke/water but mark weak
                det = {**det, "weak_no_face": True}
            final.append(det)

        for det in final:
            cls_id = det["class_id"]
            if cls_id == 0:
                processed_data["face_detected"] = True
            elif cls_id == 1:
                processed_data["smoking_detected"] = True
            elif cls_id == 2:
                processed_data["phone_detected"] = True
            elif cls_id == 3:
                processed_data["drinking_detected"] = True
            processed_data["detections"].append(det)

        processed_data["behavior_debug"] = {
            "raw_count": len(raw),
            "kept_count": len(final),
            "filtered_count": len(processed_data["filtered_out"]),
            "top_conf": {
                name: max(
                    (d["confidence"] for d in raw if d["class_name"] == name),
                    default=None,
                )
                for name in ("smoke", "phone", "water", "face")
            },
            "filtered_out": processed_data["filtered_out"][:8],
        }
        return processed_data

    def _near_face(self, det: Dict[str, Any], face_bbox: List[int]) -> Tuple[bool, str]:
        """Require behavior box center near the face (mouth / ear region)."""
        fc = _bbox_center(face_bbox)
        dc = _bbox_center(det["bbox"])
        diag = _bbox_diag(face_bbox)
        dist = ((dc[0] - fc[0]) ** 2 + (dc[1] - fc[1]) ** 2) ** 0.5
        max_dist = diag * float(self.near_face_ratio)

        cls_id = det["class_id"]
        fx1, fy1, fx2, fy2 = face_bbox
        face_h = max(1.0, float(fy2 - fy1))
        face_w = max(1.0, float(fx2 - fx1))

        if dist > max_dist:
            return False, f"far_face dist={dist:.0f}>{max_dist:.0f}"

        # Phone: prefer lateral (cheek / ear), allow either side
        if cls_id == 2:
            lateral = abs(dc[0] - fc[0]) >= 0.15 * face_w
            vertical_ok = abs(dc[1] - fc[1]) <= 0.75 * face_h
            if not (lateral or vertical_ok):
                return False, "phone_not_near_ear"
            return True, "ok"

        # Smoke / water: prefer lower face (mouth)
        if cls_id in (1, 3):
            mouth_y = fy1 + 0.65 * face_h
            if dc[1] < fy1 + 0.25 * face_h:
                return False, "above_face"
            # Soft preference — still accept if close enough overall
            if abs(dc[1] - mouth_y) > 0.9 * face_h and dist > 0.6 * diag:
                return False, "not_near_mouth"
            return True, "ok"

        return True, "ok"

    def draw_results(self, image, results):
        img = image.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

        for det in results.get("detections") or []:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = int(det["class_id"])
            conf = float(det["confidence"])
            label = f"{det['class_name']} {conf:.2f}"
            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                img,
                (x1, y1 - text_size[1] - 5),
                (x1 + text_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        return img
