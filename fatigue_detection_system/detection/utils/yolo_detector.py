# -*- coding: utf-8 -*-
"""YOLO wrapper for face / smoke / phone / water detection."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from django.conf import settings

# Lazy import: avoid loading ultralytics (and heavy train/FastSAM stack) at Django startup.
YOLO = None


def _yolo_cls():
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as _YOLO

        YOLO = _YOLO
    return YOLO


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


def _bbox_area(bbox: List[int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def pick_primary_face(
    face_boxes: List[Dict[str, Any]],
    sticky_bbox: Optional[List[int]] = None,
    iou_keep: float = 0.25,
    area_switch_ratio: float = 1.35,
) -> Optional[Dict[str, Any]]:
    """
    Choose the primary monitored face when multiple faces appear.

    Prefer the largest box, but keep the previous primary when IoU is decent
    unless another face is clearly larger (area_switch_ratio).
    """
    if not face_boxes:
        return None
    largest = max(
        face_boxes,
        key=lambda d: (_bbox_area(d["bbox"]), float(d.get("confidence") or 0.0)),
    )
    if sticky_bbox is None:
        return largest

    best_iou = -1.0
    sticky_match = None
    for d in face_boxes:
        iou = _bbox_iou(d["bbox"], sticky_bbox)
        if iou > best_iou:
            best_iou = iou
            sticky_match = d
    if sticky_match is None or best_iou < float(iou_keep):
        return largest

    # Stick unless another face is substantially larger
    if _bbox_area(largest["bbox"]) >= _bbox_area(sticky_match["bbox"]) * float(area_switch_ratio):
        return largest
    return sticky_match


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


class YOLODetector:
    """Detect face / smoking / phone / drinking with optional spatial filters."""

    CLASS_NAMES = ["face", "smoke", "phone", "water"]

    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(settings.BASE_DIR, "weights", "best.pt")

        self.load_config()
        self.model = _yolo_cls()(weights_path)
        self._apply_runtime_params()
        self.class_names = list(self.CLASS_NAMES)
        self._sticky_primary_bbox: Optional[List[int]] = None

    def load_config(self):
        from detection.utils.compute_scheduler import compute_scheduler
        from detection.utils.config_cache import get_configs

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
            configs = get_configs(force=True)
            compute_scheduler.configure(configs)
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
        from detection.utils.compute_scheduler import compute_scheduler

        plan = compute_scheduler.ensure_configured()
        # Prefer scheduler device (auto-falls back to cpu if CUDA unavailable)
        if plan.use_cuda:
            self.device = plan.device
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
                self.device = "cpu"

    def detect(self, image):
        """Run inference under YOLO thread quota; CUDA uses FP16 when enabled."""
        from detection.utils.compute_scheduler import compute_scheduler

        kwargs = {"verbose": False}
        if getattr(self, "_imgsz_from_config", False):
            kwargs["imgsz"] = int(self.imgsz)

        with compute_scheduler.yolo_context() as plan:
            if plan.use_cuda:
                kwargs["device"] = plan.device
                if plan.cuda_half:
                    kwargs["half"] = True
            results = self.model(image, **kwargs)
        return results[0]

    def warmup(self, runs: int = 2) -> float:
        """Run dummy inferences so first real frame is not a cold-start spike."""
        import time as _time

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        t0 = _time.perf_counter()
        for _ in range(max(1, int(runs))):
            try:
                self.detect(dummy)
            except Exception as e:  # noqa: BLE001
                print(f"YOLO warmup failed: {e}")
                break
        return (_time.perf_counter() - t0) * 1000.0

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
        primary = pick_primary_face(face_boxes, sticky_bbox=self._sticky_primary_bbox)
        face_bbox = primary["bbox"] if primary is not None else None
        self._sticky_primary_bbox = list(face_bbox) if face_bbox is not None else None

        final: List[Dict[str, Any]] = []
        for det in kept:
            if det["class_id"] == 0:
                # Mark primary so UI / EAR know which face is tracked
                if primary is not None and det is primary:
                    det = {**det, "is_primary": True}
                else:
                    det = {**det, "is_primary": False}
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

        processed_data["face_bbox"] = face_bbox
        processed_data["face_bboxes"] = [d["bbox"] for d in face_boxes]
        processed_data["face_count"] = len(face_boxes)
        processed_data["behavior_debug"] = {
            "raw_count": len(raw),
            "kept_count": len(final),
            "filtered_count": len(processed_data["filtered_out"]),
            "face_count": len(face_boxes),
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
            name = det["class_name"]
            if cls_id == 0 and det.get("is_primary"):
                label = f"face* {conf:.2f}"
                thickness = 3
            elif cls_id == 0:
                label = f"face {conf:.2f}"
                thickness = 1
            else:
                label = f"{name} {conf:.2f}"
                thickness = 2
            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
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
