# -*- coding: utf-8 -*-
"""YOLO wrapper for face / smoke / phone / water detection."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger("detection.detect")

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
    iou_keep: float = 0.12,
    area_switch_ratio: float = 2.2,
    frame_wh: Optional[Tuple[int, int]] = None,
    center_bias: float = 0.35,
) -> Optional[Dict[str, Any]]:
    """
    Choose the primary monitored face when multiple faces appear.

    Strong stickiness: keep the previous primary whenever IoU is still
    reasonable. Only switch when another face is clearly larger AND the
    sticky match is weak. With no sticky history, prefer large + centered.
    """
    if not face_boxes:
        return None

    def _score(d: Dict[str, Any]) -> Tuple[float, float]:
        area = _bbox_area(d["bbox"])
        conf = float(d.get("confidence") or 0.0)
        if frame_wh is None or frame_wh[0] <= 0 or frame_wh[1] <= 0:
            return (area, conf)
        cx, cy = _bbox_center(d["bbox"])
        fw, fh = float(frame_wh[0]), float(frame_wh[1])
        nx = abs(cx / fw - 0.5) * 2.0
        ny = abs(cy / fh - 0.5) * 2.0
        dist = min(1.0, (nx * nx + ny * ny) ** 0.5)
        centered = 1.0 - dist
        return (area * (1.0 + float(center_bias) * centered), conf)

    best = max(face_boxes, key=_score)
    if sticky_bbox is None:
        return best

    best_iou = -1.0
    sticky_match = None
    for d in face_boxes:
        iou = _bbox_iou(d["bbox"], sticky_bbox)
        if iou > best_iou:
            best_iou = iou
            sticky_match = d

    # Lost sticky track entirely → fall back to best score
    if sticky_match is None or best_iou < float(iou_keep):
        return best

    # Stick unless another face is substantially larger and overlap is weak
    sticky_area = _bbox_area(sticky_match["bbox"])
    best_area = _bbox_area(best["bbox"])
    if (
        sticky_match is not best
        and sticky_area > 0
        and best_area >= sticky_area * float(area_switch_ratio)
        and best_iou < 0.45
    ):
        return best
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
        self.model = None
        self.backend = "pt"
        self._load_model(weights_path)
        self._apply_runtime_params()
        self.class_names = list(self.CLASS_NAMES)
        self._sticky_primary_bbox: Optional[List[int]] = None
        # Per-session sticky primary-face bbox for the archive/realtime path
        # (browser + MVS may run distinct sessions against the same detector
        # singleton, so keep their "last primary face" memory separate).
        self._sticky_primary_by_session: Dict[Any, List[int]] = {}
        # ROI cascade: after sticky face is stable, infer on face neighborhood
        self._roi_stable_hits: int = 0
        self._detect_frame_i: int = 0
        self._roi_full_every: int = 4  # force full-frame every K detects
        self._last_roi_offset: Optional[Tuple[int, int]] = None
        self._last_used_roi: bool = False

    def _load_model(self, weights_path: str) -> None:
        """Prefer ONNX beside .pt; fall back to Ultralytics .pt."""
        pt_path = weights_path
        onnx_path = None
        if pt_path.lower().endswith(".pt"):
            onnx_path = pt_path[:-3] + ".onnx"
        elif pt_path.lower().endswith(".onnx"):
            onnx_path = pt_path
            pt_path = pt_path[:-5] + ".pt"

        loaded = False
        if onnx_path and os.path.isfile(onnx_path):
            try:
                self.model = _yolo_cls()(onnx_path)
                self.backend = "onnx"
                loaded = True
                logger.info("YOLO loaded ONNX: %s", onnx_path)
            except Exception:  # noqa: BLE001
                logger.exception("ONNX load failed, falling back to .pt: %s", onnx_path)
        if not loaded:
            if not os.path.isfile(pt_path):
                raise FileNotFoundError(f"YOLO weights not found: {pt_path}")
            self.model = _yolo_cls()(pt_path)
            self.backend = "pt"
            logger.info("YOLO loaded PyTorch: %s", pt_path)

    def _face_roi_crop(
        self, image: np.ndarray, face_bbox: List[int]
    ) -> Tuple[np.ndarray, int, int]:
        """Crop around primary face using near_face_ratio; return crop + origin."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        diag = _bbox_diag([x1, y1, x2, y2])
        pad = int(diag * float(self.near_face_ratio) * 0.55)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            return image, 0, 0
        return image[cy1:cy2, cx1:cx2].copy(), cx1, cy1

    def _should_use_roi(self) -> bool:
        self._detect_frame_i += 1
        if self._sticky_primary_bbox is None:
            self._roi_stable_hits = 0
            return False
        if self._roi_stable_hits < 2:
            return False
        # Every Kth frame: full image to re-lock face / catch far behaviors
        if (self._detect_frame_i % max(1, int(self._roi_full_every))) == 0:
            return False
        return True

    def _remap_result_boxes(self, result, ox: int, oy: int, full_shape) -> Any:
        """Shift YOLO boxes from crop coords back to full-frame coords."""
        try:
            if result is None:
                return result
            try:
                result.orig_shape = full_shape[:2]
            except Exception:  # noqa: BLE001
                pass
            if result.boxes is None or len(result.boxes) == 0:
                return result
            data = result.boxes.data
            if hasattr(data, "clone"):
                data = data.clone()
            else:
                data = data.copy()
            data[:, 0] += float(ox)
            data[:, 1] += float(oy)
            data[:, 2] += float(ox)
            data[:, 3] += float(oy)
            result.boxes.data = data
        except Exception:  # noqa: BLE001
            logger.exception("ROI remap failed")
        return result

    def load_config(self):
        from detection.utils.compute_scheduler import compute_scheduler
        from detection.utils.config_cache import get_configs

        # Archive-like defaults (low floors cause false positives / 乱识别)
        self.conf_thresh = 0.5
        self.iou_thresh = 0.5
        self.imgsz = 640
        self.device = "cpu"
        # Per-class floors (applied after model returns candidates)
        self.conf_face = 0.45
        self.conf_smoke = 0.45
        self.conf_phone = 0.45
        self.conf_water = 0.45
        self.spatial_filter = False
        self.near_face_ratio = 2.0  # max center distance in face-diag units

        try:
            configs = get_configs(force=True)
            compute_scheduler.configure(configs)
            self.conf_thresh = _cfg_float(configs, "yolo_conf_thresh", 0.28)
            self.iou_thresh = _cfg_float(configs, "yolo_iou_thresh", 0.5)
            if "yolo_imgsz" in configs and str(configs.get("yolo_imgsz", "")).strip():
                self.imgsz = max(320, _cfg_int(configs, "yolo_imgsz", 640))
            else:
                self.imgsz = 640
            self.device = configs.get("device") or "cpu"
            # Per-class: use saved value as-is (0.01–0.95). Do NOT clamp to global conf.
            self.conf_face = _cfg_float(configs, "yolo_conf_face", 0.35)
            self.conf_smoke = _cfg_float(configs, "yolo_conf_smoke", 0.22)
            self.conf_phone = _cfg_float(configs, "yolo_conf_phone", 0.22)
            self.conf_water = _cfg_float(configs, "yolo_conf_water", 0.22)
            self.spatial_filter = str(configs.get("yolo_spatial_filter", "false")).lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            self.near_face_ratio = _cfg_float(configs, "yolo_near_face_ratio", 2.0)
            logger.info(
                "已加载YOLO配置: conf=%s, iou=%s, imgsz=%s, device=%s, spatial=%s, "
                "floors=smoke:%s/phone:%s/water:%s/face:%s",
                self.conf_thresh, self.iou_thresh, self.imgsz, self.device,
                self.spatial_filter, self.conf_smoke, self.conf_phone,
                self.conf_water, self.conf_face,
            )
        except Exception:  # noqa: BLE001
            logger.exception("加载YOLO配置失败")

        if hasattr(self, "model") and self.model is not None:
            self._apply_runtime_params()

    def _infer_conf(self) -> float:
        """
        Ultralytics applies one conf before per-class post floors.

        Infer at min(global, per-class floors) so soft class floors (e.g. smoke)
        can still surface candidates; post-process raises per class. Absolute
        floor 0.15 caps NMS cost on CPU.
        """
        floors = (
            float(self.conf_thresh),
            float(self.conf_face),
            float(self.conf_smoke),
            float(self.conf_phone),
            float(self.conf_water),
        )
        return max(0.15, min(0.99, min(floors)))

    def _apply_runtime_params(self) -> None:
        from detection.utils.compute_scheduler import compute_scheduler

        plan = compute_scheduler.ensure_configured()
        # Prefer scheduler device (auto-falls back to cpu if CUDA unavailable)
        if plan.use_cuda:
            self.device = plan.device
        try:
            self.model.conf = self._infer_conf()
            self.model.iou = self.iou_thresh
        except Exception:  # noqa: BLE001
            pass
        # .to() is for PyTorch modules; ONNX Runtime selects EP via predict kwargs
        if self.device and getattr(self, "backend", "pt") == "pt":
            try:
                self.model.to(self.device)
            except Exception:  # noqa: BLE001
                logger.exception("无法在设备 %s 上运行模型，回退到 cpu", self.device)
                self.device = "cpu"

    def detect(self, image):
        """Run inference under YOLO thread quota; CUDA uses FP16 when enabled.

        Always full-frame (archive-like). ROI cascade disabled — it hurt
        smoke/phone/water recall when hands were away from the face crop.
        """
        from detection.utils.compute_scheduler import compute_scheduler

        kwargs = {
            "verbose": False,
            "conf": float(self._infer_conf()),
            "iou": float(self.iou_thresh),
            "imgsz": int(self.imgsz or 640),
            "max_det": 100,
            # Avoid Ultralytics dataloader worker processes on Windows (extra stalls)
            "workers": 0,
        }

        self._last_used_roi = False
        self._last_roi_offset = None
        # Keep counters for debug/meta only; do not crop
        self._detect_frame_i = int(getattr(self, "_detect_frame_i", 0) or 0) + 1

        with compute_scheduler.yolo_context() as plan:
            if plan.use_cuda:
                kwargs["device"] = plan.device
                if plan.cuda_half and getattr(self, "backend", "pt") == "pt":
                    kwargs["half"] = True
            elif self.device:
                kwargs["device"] = self.device
            results = self.model(image, **kwargs)
        return results[0]

    def warmup(self, runs: int = 2) -> float:
        """Run dummy inferences so first real frame is not a cold-start spike."""
        import time as _time

        side = int(self.imgsz or 640)
        dummy = np.zeros((side, side, 3), dtype=np.uint8)
        t0 = _time.perf_counter()
        for _ in range(max(1, int(runs))):
            try:
                self.detect(dummy)
            except Exception:  # noqa: BLE001
                logger.exception("YOLO warmup failed")
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

    def process_results_simple(self, results, session_id=None):
        """Archive-style detections (single conf, no spatial filter) with
        sticky + center-weighted primary-face selection (see pick_primary_face)
        instead of naive max-area, so the tracked face doesn't jump around
        when multiple people are visible (wide-FOV industrial cameras)."""
        processed_data = {
            "face_detected": False,
            "smoking_detected": False,
            "phone_detected": False,
            "drinking_detected": False,
            "detections": [],
            "raw_detections": [],
            "filtered_out": [],
            "behavior_debug": {},
            "face_bbox": None,
            "face_bboxes": [],
            "face_count": 0,
        }
        boxes = results.boxes.cpu().numpy()
        face_boxes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id < 0 or cls_id >= len(self.class_names):
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].astype(int)]
            det = {
                "class_id": cls_id,
                "class_name": self.class_names[cls_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
            processed_data["detections"].append(det)
            processed_data["raw_detections"].append(det)
            if cls_id == 0:
                processed_data["face_detected"] = True
                face_boxes.append(det)
            elif cls_id == 1:
                processed_data["smoking_detected"] = True
            elif cls_id == 2:
                processed_data["phone_detected"] = True
            elif cls_id == 3:
                processed_data["drinking_detected"] = True
        if face_boxes:
            frame_wh = None
            try:
                oh, ow = results.orig_shape
                frame_wh = (int(ow), int(oh))
            except Exception:  # noqa: BLE001
                frame_wh = None

            sticky_key = session_id if session_id is not None else "_default"
            sticky_bbox = self._sticky_primary_by_session.get(sticky_key)
            primary = pick_primary_face(
                face_boxes, sticky_bbox=sticky_bbox, frame_wh=frame_wh
            )
            if primary is not None:
                self._sticky_primary_by_session[sticky_key] = list(primary["bbox"])
                # Bound memory growth if callers pass many distinct session ids.
                if len(self._sticky_primary_by_session) > 64:
                    self._sticky_primary_by_session.pop(
                        next(iter(self._sticky_primary_by_session))
                    )
            processed_data["face_bbox"] = list(primary["bbox"]) if primary else None
            processed_data["face_bboxes"] = [list(d["bbox"]) for d in face_boxes]
            processed_data["face_count"] = len(face_boxes)
            for d in processed_data["detections"]:
                if d["class_id"] == 0:
                    d["is_primary"] = primary is not None and d["bbox"] == primary["bbox"]
        processed_data["behavior_debug"] = {
            "raw_count": len(processed_data["detections"]),
            "kept_count": len(processed_data["detections"]),
            "filtered_count": 0,
            "mode": "archive_simple",
        }
        return processed_data

    def detect_archive(self, image):
        """Archive infer: model(image) with global conf (Ultralytics default letterbox)."""
        from detection.utils.compute_scheduler import compute_scheduler

        conf = max(0.25, min(0.95, float(self.conf_thresh or 0.5)))
        # Match archive: set model.conf then call model(image); keep workers=0 for CPU stability
        try:
            self.model.conf = conf
        except Exception:  # noqa: BLE001
            pass
        kwargs = {
            "verbose": False,
            "conf": conf,
            "workers": 0,
        }
        with compute_scheduler.yolo_context() as plan:
            if plan.use_cuda:
                kwargs["device"] = plan.device
                if plan.cuda_half and getattr(self, "backend", "pt") == "pt":
                    kwargs["half"] = True
            elif self.device:
                kwargs["device"] = self.device
            results = self.model(image, **kwargs)
        return results[0]

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

        # Per-class floors raise the bar after a softer infer conf
        kept: List[Dict[str, Any]] = []
        for det in raw:
            floor = float(self._class_conf_floor(det["class_id"]))
            if det["confidence"] < floor:
                processed_data["filtered_out"].append(
                    {**det, "reason": f"conf<{floor:.2f}"}
                )
                continue
            kept.append(det)

        face_boxes = [d for d in kept if d["class_id"] == 0]
        frame_wh = None
        try:
            # ultralytics: orig_shape is (h, w)
            oh, ow = results.orig_shape
            frame_wh = (int(ow), int(oh))
        except Exception:  # noqa: BLE001
            frame_wh = None
        primary = pick_primary_face(
            face_boxes,
            sticky_bbox=self._sticky_primary_bbox,
            frame_wh=frame_wh,
        )
        face_bbox = primary["bbox"] if primary is not None else None
        # Keep sticky across brief misses so the next hit re-locks the same face
        if face_bbox is not None:
            if self._sticky_primary_bbox is not None:
                iou = _bbox_iou(face_bbox, self._sticky_primary_bbox)
                if iou >= 0.12:
                    self._roi_stable_hits = min(20, int(self._roi_stable_hits) + 1)
                else:
                    self._roi_stable_hits = 0
            else:
                self._roi_stable_hits = 1
            self._sticky_primary_bbox = list(face_bbox)
        else:
            self._roi_stable_hits = max(0, int(self._roi_stable_hits) - 1)

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
        # Primary first so downstream EAR / preview prefer the sticky face
        if face_bbox is not None:
            others = [d["bbox"] for d in face_boxes if d is not primary]
            processed_data["face_bboxes"] = [list(face_bbox)] + others
        else:
            processed_data["face_bboxes"] = [d["bbox"] for d in face_boxes]
        processed_data["face_count"] = len(face_boxes)
        processed_data["behavior_debug"] = {
            "raw_count": len(raw),
            "kept_count": len(final),
            "filtered_count": len(processed_data["filtered_out"]),
            "face_count": len(face_boxes),
            "used_roi": bool(self._last_used_roi),
            "roi_stable_hits": int(self._roi_stable_hits),
            "backend": getattr(self, "backend", "pt"),
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

        # Phone: prefer lateral (cheek / ear), allow either side; soft vertical band
        if cls_id == 2:
            lateral = abs(dc[0] - fc[0]) >= 0.12 * face_w
            vertical_ok = abs(dc[1] - fc[1]) <= 0.9 * face_h
            if not (lateral or vertical_ok):
                return False, "phone_not_near_ear"
            return True, "ok"

        # Smoke / water: prefer lower face (mouth); keep soft so side angles survive
        if cls_id in (1, 3):
            mouth_y = fy1 + 0.65 * face_h
            if dc[1] < fy1 + 0.15 * face_h:
                return False, "above_face"
            if abs(dc[1] - mouth_y) > 1.1 * face_h and dist > 0.75 * diag:
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
