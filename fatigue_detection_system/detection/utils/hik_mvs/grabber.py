# -*- coding: utf-8 -*-
"""Background GigE grab + detection (preview FPS decoupled from detect interval)."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from django.db import connection

from .camera import HikCamera

_EMA_ALPHA = 0.2


def _ema(prev: Optional[float], value: float, alpha: float = _EMA_ALPHA) -> float:
    if prev is None:
        return float(value)
    return float(alpha * value + (1.0 - alpha) * prev)


class MvsGrabber:
    """High-rate preview grabber + EAR sampler + lower-rate YOLO detection."""

    def __init__(self):
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._grab_thread: Optional[threading.Thread] = None
        self._ear_thread: Optional[threading.Thread] = None
        self._detect_thread: Optional[threading.Thread] = None
        self._camera: Optional[HikCamera] = None
        self._session_id: Optional[int] = None
        self._user_id: Optional[int] = None
        self._interval_ms = 500
        self._ear_interval_ms = 100
        self._preview_max_width = 1280
        self._detect_max_width = 960
        self._preview_jpeg_quality = 70
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_face_bbox: Optional[List[int]] = None
        self._latest_face_bboxes: List[List[int]] = []
        self._latest_faces_meta: List[Dict[str, Any]] = []
        # EAR-only landmark cache (never used for preview drawing)
        self._latest_landmarks = None
        self._latest_landmarks_size: Optional[Tuple[int, int]] = None
        self._latest_landmark_faces: List[Dict[str, Any]] = []
        # Face box for EAR (detect-frame coords); EMA is for EAR stability only.
        self._face_ema_bbox: Optional[List[float]] = None
        self._face_hold_until: float = 0.0
        self._face_miss_hold_sec: float = 0.8
        self._face_ema_alpha: float = 0.65
        # Rigid overlay from last detect (box+landmarks from SAME frame). Grab
        # composites this onto every new frame → smooth preview, no relative drift.
        self._overlay_bbox: Optional[List[int]] = None
        self._overlay_landmarks = None
        self._overlay_detections: List[Dict[str, Any]] = []
        self._overlay_wh: Optional[Tuple[int, int]] = None
        self._overlay_at: float = 0.0
        self._overlay_ttl_sec: float = 1.2
        self._frame_seq = 0
        self._detect_busy = False
        self._timing_avg: Dict[str, float] = {}
        self._last_fatigue_level = 0
        self._last_yawn = False
        self._event_save_cooldown_until = 0.0
        self._pending_fatigue_event = False
        self._last_persist_at = 0.0
        self._persist_interval_sec = 2.0
        self._warmup_until = 0.0
        self._warmup_sec = 5.0
        self._warmup_started_at = 0.0
        self._warmup_min_sec = 1.2
        self._warmup_ui_cleared = False
        self._exposure_ready = False
        self._models_warmed = False
        self._warmup_phase = "idle"
        self._model_warm_thread: Optional[threading.Thread] = None
        self.latest_jpeg: Optional[bytes] = None
        self.latest_meta: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self.running = False

    def start(
        self,
        session_id: int,
        user_id: int,
        interval_ms: int = 500,
        device_index: Optional[int] = None,
        camera_ip: Optional[str] = None,
    ) -> None:
        from detection.models import SystemConfig
        from detection.utils.fatigue_tracker import (
            behavior_tracker,
            fatigue_tracker,
            load_fatigue_config,
        )

        with self._lock:
            if self.running:
                raise RuntimeError("MVS grabber already running")
            self._stop.clear()
            self.error = None
            self.latest_jpeg = None
            self.latest_meta = {"timing": {}}
            self._latest_bgr = None
            self._latest_face_bbox = None
            self._latest_face_bboxes = []
            self._latest_faces_meta = []
            self._latest_landmarks = None
            self._latest_landmarks_size = None
            self._latest_landmark_faces = []
            self._face_ema_bbox = None
            self._face_hold_until = 0.0
            self._overlay_bbox = None
            self._overlay_landmarks = None
            self._overlay_detections = []
            self._overlay_wh = None
            self._overlay_at = 0.0
            self._frame_seq = 0
            self._timing_avg = {}
            self._session_id = session_id
            self._user_id = user_id
            self._interval_ms = max(50, int(interval_ms or 500))

            configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
            fcfg = load_fatigue_config(configs)
            self._ear_interval_ms = max(50, int(fcfg["ear_sample_interval_ms"]))
            try:
                from detection.utils.compute_scheduler import compute_scheduler

                plan = compute_scheduler.configure(configs)
                self.latest_meta = {
                    "timing": {},
                    "scheduler": compute_scheduler.snapshot(),
                }
                _ = plan
            except Exception:  # noqa: BLE001
                pass
            try:
                self._detect_max_width = max(640, int(float(configs.get("yolo_detect_max_width", 960))))
            except (TypeError, ValueError):
                self._detect_max_width = 960
            self._last_fatigue_level = 0
            self._last_yawn = False
            self._event_save_cooldown_until = 0.0
            self._pending_fatigue_event = False
            self._last_persist_at = 0.0
            try:
                self._persist_interval_sec = max(
                    0.5, float(configs.get("yolo_persist_interval_sec", 2.0))
                )
            except (TypeError, ValueError):
                self._persist_interval_sec = 2.0
            try:
                self._warmup_sec = max(0.0, float(configs.get("startup_warmup_sec", 5.0)))
            except (TypeError, ValueError):
                self._warmup_sec = 5.0
            self._warmup_ui_cleared = False
            self._exposure_ready = False
            self._models_warmed = False
            self._warmup_phase = "opening"
            fatigue_tracker.reset(session_id)
            behavior_tracker.reset(session_id)
            fatigue_tracker.configure(session_id, configs)
            behavior_tracker.configure(session_id, configs)
            try:
                from detection.views import yolo_detector

                if yolo_detector is not None:
                    yolo_detector._sticky_primary_bbox = None
            except Exception:  # noqa: BLE001
                pass

            cam = HikCamera()
            if camera_ip:
                cam.open_by_ip(camera_ip)
            else:
                cam.open_by_index(int(device_index or 0))
            cam.start_grab()
            self._camera = cam
            self.running = True
            # Warmup clock starts when stream is up; work runs in parallel inside it.
            now = time.time()
            self._warmup_started_at = now
            self._warmup_until = now + float(self._warmup_sec or 5.0)
            ae_ok = bool((getattr(cam, "last_diag", {}) or {}).get("ae"))
            self._warmup_phase = "ae_settle" if ae_ok else "software_calib"
            self.latest_meta = {
                "timing": {},
                "camera": dict(getattr(cam, "last_diag", {}) or {}),
                "warming_up": True,
                "warmup_phase": self._warmup_phase,
                "warmup_detail": self._warmup_detail_text(),
            }
            self._grab_thread = threading.Thread(
                target=self._grab_loop, name="mvs-grab", daemon=True
            )
            self._ear_thread = threading.Thread(
                target=self._ear_loop, name="mvs-ear", daemon=True
            )
            self._detect_thread = threading.Thread(
                target=self._detect_loop, name="mvs-detect", daemon=True
            )
            self._model_warm_thread = threading.Thread(
                target=self._warmup_models_worker, name="mvs-model-warm", daemon=True
            )
            self._grab_thread.start()
            self._ear_thread.start()
            self._detect_thread.start()
            self._model_warm_thread.start()

    def stop(self, complete_session: bool = True) -> None:
        self._stop.set()
        threads = []
        with self._lock:
            if self._grab_thread:
                threads.append(self._grab_thread)
            if self._ear_thread:
                threads.append(self._ear_thread)
            if self._detect_thread:
                threads.append(self._detect_thread)
            if self._model_warm_thread:
                threads.append(self._model_warm_thread)
        for thread in threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=5)
        with self._lock:
            if self._camera:
                try:
                    self._camera.stop_and_close()
                except Exception:  # noqa: BLE001
                    pass
                self._camera = None
            self.running = False
            self._grab_thread = None
            self._ear_thread = None
            self._detect_thread = None
            self._model_warm_thread = None
            self._detect_busy = False
            self._exposure_ready = False
            self._models_warmed = False
            self._warmup_phase = "idle"
            self._latest_face_bbox = None
            self._latest_face_bboxes = []
            self._latest_faces_meta = []
            self._latest_landmarks = None
            self._latest_landmarks_size = None
            self._latest_landmark_faces = []
            self._face_ema_bbox = None
            self._face_hold_until = 0.0
            self._overlay_bbox = None
            self._overlay_landmarks = None
            self._overlay_detections = []
            self._overlay_wh = None
            self._overlay_at = 0.0
            session_id = self._session_id
            self._session_id = None
        if session_id is not None:
            try:
                from detection.utils.fatigue_tracker import behavior_tracker, fatigue_tracker

                fatigue_tracker.reset(session_id)
                behavior_tracker.reset(session_id)
            except Exception:  # noqa: BLE001
                pass
        if complete_session and session_id:
            self._complete_session(session_id)

    def _warmup_detail_text(self) -> str:
        phase = self._warmup_phase or "idle"
        cam = dict(getattr(self._camera, "last_diag", {}) or {}) if self._camera else {}
        mode = cam.get("exposure_mode") or ""
        if phase == "ae_settle":
            return "硬件连续自动曝光收敛中"
        if phase == "software_calib":
            return "自动曝光不可用，正在软件校准固定曝光"
        if phase == "model_warmup":
            if mode == "software_fallback":
                return "曝光已锁定，模型预热中"
            if mode == "hardware_ae":
                return "硬件自动曝光就绪，模型预热中"
            return "模型与检测管线预热中"
        if phase == "ready":
            return "预热完成"
        if phase == "opening":
            return "正在打开相机"
        return "系统预热中"

    def _set_warmup_phase(self, phase: str) -> None:
        with self._lock:
            self._warmup_phase = phase
            meta = dict(self.latest_meta or {})
            meta["warmup_phase"] = phase
            meta["warmup_detail"] = self._warmup_detail_text()
            meta["warming_up"] = time.time() < float(self._warmup_until or 0.0)
            meta["exposure_ready"] = bool(self._exposure_ready)
            meta["models_warmed"] = bool(self._models_warmed)
            self.latest_meta = meta

    def _maybe_finish_warmup_early(self) -> None:
        """End warmup before hard deadline once exposure + models are ready."""
        with self._lock:
            if not self.running:
                return
            now = time.time()
            until = float(self._warmup_until or 0.0)
            if now >= until:
                return
            started = float(self._warmup_started_at or 0.0)
            min_sec = float(self._warmup_min_sec or 1.2)
            if (now - started) < min_sec:
                return
            if not (self._exposure_ready and self._models_warmed):
                return
            self._warmup_until = now
            self._warmup_phase = "ready"
            meta = dict(self.latest_meta or {})
            meta["warming_up"] = False
            meta["warmup_phase"] = "ready"
            meta["warmup_detail"] = "预热完成"
            meta["warmup_early_exit"] = True
            meta["exposure_ready"] = True
            meta["models_warmed"] = True
            self.latest_meta = meta

    def _warmup_models_worker(self) -> None:
        """Dummy YOLO/dlib passes in parallel with camera exposure settle/calib."""
        try:
            from detection.views import dlib_detector, yolo_detector

            warm_ms: Dict[str, float] = {}
            if yolo_detector is not None:
                warm_ms["yolo"] = round(float(yolo_detector.warmup(runs=2)), 1)
            if dlib_detector is not None:
                warm_ms["dlib"] = round(float(dlib_detector.warmup()), 1)
            with self._lock:
                meta = dict(self.latest_meta or {})
                timing = dict(meta.get("timing") or {})
                timing["model_warmup_ms"] = warm_ms
                meta["timing"] = timing
                self.latest_meta = meta
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                meta = dict(self.latest_meta or {})
                meta["model_warmup_error"] = str(exc)
                self.latest_meta = meta
        finally:
            with self._lock:
                self._models_warmed = True
            if self._exposure_ready:
                self._set_warmup_phase("ready")
            elif self._warmup_phase in ("ae_settle", "software_calib", "opening"):
                pass
            else:
                self._set_warmup_phase("model_warmup")
            self._maybe_finish_warmup_early()

    def status(self) -> Dict[str, Any]:
        """Read-only status snapshot (no tracker side effects)."""
        with self._lock:
            now = time.time()
            warming = bool(self.running and now < float(self._warmup_until or 0.0))
            remain = max(0.0, float(self._warmup_until or 0.0) - now) if warming else 0.0
            meta = dict(self.latest_meta)
            if warming:
                meta = {
                    "timing": meta.get("timing") or {},
                    "scheduler": meta.get("scheduler"),
                    "warming_up": True,
                    # Keep camera health during warmup so black-stream is visible early
                    "camera": meta.get("camera"),
                    "camera_warning": meta.get("camera_warning"),
                    "warmup_phase": meta.get("warmup_phase") or self._warmup_phase,
                    "warmup_detail": meta.get("warmup_detail") or self._warmup_detail_text(),
                    "exposure_ready": bool(self._exposure_ready),
                    "models_warmed": bool(self._models_warmed),
                    "exposure_calibrating": bool(meta.get("exposure_calibrating")),
                }
            return {
                "running": self.running,
                "session_id": self._session_id,
                "error": self.error,
                "meta": meta,
                "has_frame": self.latest_jpeg is not None,
                "warming_up": warming,
                "warmup_remaining_sec": round(remain, 1),
                "warmup_phase": meta.get("warmup_phase") if warming else "ready",
            }

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg

    def _merge_timing(self, patch: Dict[str, float]) -> Dict[str, Any]:
        """Update EMA averages and return a timing dict for meta."""
        out: Dict[str, Any] = {}
        for key, value in patch.items():
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            out[key] = round(v, 1)
            avg_key = key if key.endswith("_avg_ms") else key.replace("_ms", "_avg_ms")
            if key.endswith("_avg_ms"):
                continue
            prev = self._timing_avg.get(avg_key)
            self._timing_avg[avg_key] = _ema(prev, v)
            out[avg_key] = round(self._timing_avg[avg_key], 1)
        for avg_key, avg_val in self._timing_avg.items():
            out.setdefault(avg_key, round(avg_val, 1))
        return out

    def _complete_session(self, session_id: int) -> None:
        try:
            from django.utils import timezone
            from detection.models import DetectionSession

            session = DetectionSession.objects.filter(id=session_id).first()
            if session and session.status == "in_progress":
                session.status = "completed"
                session.end_time = timezone.now()
                session.save(update_fields=["status", "end_time"])
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                connection.close()
            except Exception:  # noqa: BLE001
                pass

    def _resize_for_preview(self, frame: np.ndarray) -> np.ndarray:
        return self._resize_max_width(frame, self._preview_max_width)

    def _resize_for_detect(self, frame: np.ndarray) -> np.ndarray:
        return self._resize_max_width(frame, self._detect_max_width)

    @staticmethod
    def _resize_max_width(frame: np.ndarray, max_w: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        nh = int(h * (max_w / float(w)))
        return cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _ema_bbox(prev: Optional[List[float]], new_box: List[int], alpha: float) -> List[float]:
        nb = [float(v) for v in new_box]
        if prev is None or len(prev) != 4:
            return nb
        a = max(0.05, min(1.0, float(alpha)))
        return [a * nb[i] + (1.0 - a) * float(prev[i]) for i in range(4)]

    @staticmethod
    def _bbox_i(box: Optional[List[float]]) -> Optional[List[int]]:
        if box is None or len(box) != 4:
            return None
        return [int(round(v)) for v in box]

    def _clip_bbox(self, bbox: List[int], fw: int, fh: int) -> List[int]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(fw - 1, x1))
        y1 = max(0, min(fh - 1, y1))
        x2 = max(0, min(fw - 1, x2))
        y2 = max(0, min(fh - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return [0, 0, max(1, fw - 1), max(1, fh - 1)]
        return [x1, y1, x2, y2]

    def _update_tracked_faces(
        self,
        primary_det: Optional[List[int]],
        all_det: List[List[int]],
        det_wh: Tuple[int, int],
    ) -> None:
        """
        EMA-smooth primary face in DETECT coordinates for EAR sampling.
        Preview overlay does NOT use these boxes.
        Must be called with self._lock held.
        """
        now = time.time()
        if primary_det is not None:
            self._face_ema_bbox = self._ema_bbox(
                self._face_ema_bbox, primary_det, self._face_ema_alpha
            )
            self._face_hold_until = now + float(self._face_miss_hold_sec)
            smoothed = self._bbox_i(self._face_ema_bbox)
            self._latest_face_bbox = smoothed
            self._latest_face_bboxes = [smoothed] if smoothed else []
            return

        if self._face_ema_bbox is not None and now <= float(self._face_hold_until or 0.0):
            held = self._bbox_i(self._face_ema_bbox)
            self._latest_face_bbox = held
            self._latest_face_bboxes = [held] if held else []
            return

        self._face_ema_bbox = None
        self._face_hold_until = 0.0
        self._latest_face_bbox = None
        self._latest_face_bboxes = []

    @staticmethod
    def _bbox_from_landmarks(landmarks, pad: float = 0.08) -> Optional[List[int]]:
        """Display box locked to landmark extents so box/points cannot diverge."""
        try:
            pts = np.asarray(landmarks, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 5 or pts.shape[1] < 2:
                return None
            x1 = float(np.min(pts[:, 0]))
            y1 = float(np.min(pts[:, 1]))
            x2 = float(np.max(pts[:, 0]))
            y2 = float(np.max(pts[:, 1]))
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            x1 -= bw * pad
            y1 -= bh * pad
            x2 += bw * pad
            y2 += bh * pad
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _bbox_iou(a: Optional[List[int]], b: Optional[List[int]]) -> float:
        if not a or not b or len(a) != 4 or len(b) != 4:
            return 0.0
        try:
            ax1, ay1, ax2, ay2 = [int(v) for v in a]
            bx1, by1, bx2, by2 = [int(v) for v in b]
        except (TypeError, ValueError):
            return 0.0
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = float(area_a + area_b - inter)
        if union <= 1e-6:
            return 0.0
        return inter / union

    def _refresh_face_overlay_from_ear(
        self,
        landmarks,
        sample_wh: Tuple[int, int],
        min_iou: float = 0.12,
    ) -> bool:
        """
        Update face box + landmarks from EAR (high rate). Keeps YOLO behavior
        detections untouched. Returns True if overlay was updated.
        """
        if landmarks is None or sample_wh is None:
            return False
        try:
            sw, sh = int(sample_wh[0]), int(sample_wh[1])
        except (TypeError, ValueError, IndexError):
            return False
        if sw <= 0 or sh <= 0:
            return False

        lm_box = self._bbox_from_landmarks(landmarks)
        if lm_box is None:
            return False
        lm_box = self._clip_bbox(lm_box, sw, sh)
        lm_copy = np.asarray(landmarks).copy()

        with self._lock:
            owh = self._overlay_wh
            # Size must match existing overlay canvas (avoid cross-resolution float)
            if owh is not None and (int(owh[0]) != sw or int(owh[1]) != sh):
                return False
            prev = self._overlay_bbox
            if prev is not None and float(min_iou) > 0:
                if self._bbox_iou(prev, lm_box) < float(min_iou):
                    # Likely jump-face / bad predict — wait for next YOLO lock
                    return False
            self._overlay_bbox = lm_box
            self._overlay_landmarks = lm_copy
            if owh is None:
                self._overlay_wh = (sw, sh)
            self._overlay_at = time.time()
            self._latest_landmarks = lm_copy
            self._latest_landmarks_size = (sw, sh)
            # Do NOT feed landmark hull back into YOLO/EAR face seed — that
            # expands the dlib rect over time ("lock then grow").
        return True

    def _compute_detect_overlay(
        self,
        det_frame: np.ndarray,
        face_bbox: Optional[List[int]],
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run dlib on the YOLO frame and return a rigid overlay packet.
        Box is derived from landmarks when available (prevents box/point split).
        """
        try:
            from detection.views import dlib_detector
        except Exception:  # noqa: BLE001
            dlib_detector = None

        dh, dw = det_frame.shape[:2]
        primary: Optional[List[int]] = None
        if face_bbox is not None:
            try:
                primary = self._clip_bbox([int(v) for v in face_bbox], dw, dh)
            except (TypeError, ValueError):
                primary = None

        lm = None
        if primary is not None and dlib_detector is not None:
            try:
                dlib_res = dlib_detector.detect_fatigue_multi(
                    det_frame,
                    face_bboxes=[primary],
                    primary_bbox=primary,
                    allow_hog=False,
                )
                lm = dlib_res.get("landmarks")
            except Exception:  # noqa: BLE001
                lm = None

        draw_box = primary
        if lm is not None:
            lm_box = self._bbox_from_landmarks(lm)
            if lm_box is not None:
                draw_box = self._clip_bbox(lm_box, dw, dh)

        return {
            "bbox": draw_box,
            "landmarks": None if lm is None else np.asarray(lm).copy(),
            "detections": list(detections or []),
            "wh": (dw, dh),
            "yolo_bbox": primary,
        }

    def _store_overlay(self, packet: Dict[str, Any]) -> None:
        with self._lock:
            self._overlay_bbox = packet.get("bbox")
            self._overlay_landmarks = packet.get("landmarks")
            self._overlay_detections = list(packet.get("detections") or [])
            self._overlay_wh = packet.get("wh")
            self._overlay_at = time.time()
            lm = packet.get("landmarks")
            if lm is not None and packet.get("wh"):
                self._latest_landmarks = lm
                self._latest_landmarks_size = packet.get("wh")

    def _draw_overlay_on_canvas(self, canvas: np.ndarray) -> np.ndarray:
        """Draw the last rigid detect overlay onto a detect-sized canvas."""
        try:
            from detection.views import dlib_detector
        except Exception:  # noqa: BLE001
            dlib_detector = None

        ch, cw = canvas.shape[:2]
        with self._lock:
            age = time.time() - float(self._overlay_at or 0.0)
            if float(self._overlay_at or 0.0) <= 0.0 or age > float(self._overlay_ttl_sec):
                return canvas
            owh = self._overlay_wh
            if owh is not None and (int(owh[0]) != cw or int(owh[1]) != ch):
                # Detect canvas size changed — do not rescale (avoids float)
                return canvas
            bbox = None if self._overlay_bbox is None else list(self._overlay_bbox)
            lm = None if self._overlay_landmarks is None else self._overlay_landmarks
            dets = list(self._overlay_detections or [])

        if bbox is not None:
            try:
                x1, y1, x2, y2 = self._clip_bbox([int(v) for v in bbox], cw, ch)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
            except Exception:  # noqa: BLE001
                pass

        if dlib_detector is not None and lm is not None:
            try:
                dlib_detector.draw_landmarks(
                    canvas, lm, landmarks_size=(cw, ch), color=(0, 255, 0)
                )
            except Exception:  # noqa: BLE001
                pass

        for det in dets:
            try:
                db = det.get("bbox")
                if not db:
                    continue
                cid = int(det.get("class_id", -1))
                if cid == 0:
                    continue
                x1, y1, x2, y2 = self._clip_bbox([int(v) for v in db], cw, ch)
                color = {
                    1: (0, 165, 255),
                    2: (0, 0, 255),
                    3: (255, 128, 0),
                }.get(cid, (200, 200, 0))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                label = str(det.get("class_name") or det.get("label") or cid)
                cv2.putText(
                    canvas,
                    label,
                    (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            except Exception:  # noqa: BLE001
                continue
        return canvas

    def _encode_preview_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """High-rate preview: detect-size canvas + rigid overlay → preview JPEG."""
        canvas = self._resize_for_detect(frame)
        try:
            canvas = self._draw_overlay_on_canvas(canvas)
        except Exception:  # noqa: BLE001
            pass
        preview = self._resize_for_preview(canvas)
        ok, buf = cv2.imencode(
            ".jpg",
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._preview_jpeg_quality)],
        )
        if not ok:
            return None
        return buf.tobytes()

    def _publish_preview_frame(self, frame: np.ndarray, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        # Always publish at grab rate for smooth MJPEG; overlay is rigid from detect.
        jpeg = self._encode_preview_jpeg(frame)
        with self._lock:
            self._latest_bgr = frame
            self._frame_seq += 1
            if jpeg:
                self.latest_jpeg = jpeg
            self.error = None
            try:
                cam_diag = dict(getattr(self._camera, "last_diag", {}) or {})
                meta = dict(self.latest_meta or {})
                meta["camera"] = cam_diag
                if extra_meta:
                    meta.update(extra_meta)
                mean = float(cam_diag.get("mean") or cam_diag.get("calib_mean") or 0)
                if mean < 3.0 and not cam_diag.get("fixed_exposure"):
                    meta["camera_warning"] = (
                        "画面过暗(mean=%.1f, %s)。请确认镜头盖已开、补光/曝光，"
                        "并关闭 MVS 客户端独占预览。"
                        % (
                            mean,
                            "黑白" if cam_diag.get("is_mono") else "彩色",
                        )
                    )
                else:
                    meta.pop("camera_warning", None)
                self.latest_meta = meta
            except Exception:  # noqa: BLE001
                pass

    def _prepare_exposure_during_warmup(self) -> None:
        """
        Prefer Continuous AE/AG settle; software fixed exposure is fallback only
        when those GenICam commands fail or hardware AE stays too dark.
        """
        with self._lock:
            if self._exposure_ready:
                return
            cam = self._camera
            warmup_until = float(self._warmup_until or 0.0)

        if cam is None:
            return

        remain = max(0.0, warmup_until - time.time())
        # Cap exposure work so model warmup (parallel) keeps wall-clock share.
        budget = min(max(0.6, remain * 0.55), 2.8)

        def _on_frame(frame: np.ndarray, calibrating: bool) -> None:
            self._publish_preview_frame(
                frame,
                {
                    "warming_up": True,
                    "exposure_calibrating": calibrating,
                    "warmup_phase": self._warmup_phase,
                    "warmup_detail": self._warmup_detail_text(),
                },
            )

        diag = dict(getattr(cam, "last_diag", {}) or {})
        ae_ok = bool(diag.get("ae"))
        try:
            if ae_ok:
                self._set_warmup_phase("ae_settle")
                diag = cam.settle_hardware_ae(
                    budget_sec=budget,
                    target_mean=90.0,
                    timeout_ms=300,
                    on_frame=lambda f: _on_frame(f, False),
                )
                if diag.get("needs_software_fallback"):
                    left = max(0.5, warmup_until - time.time())
                    self._set_warmup_phase("software_calib")
                    diag = cam.calibrate_fixed_exposure(
                        budget_sec=min(left, 2.2),
                        target_mean=105.0,
                        timeout_ms=300,
                        on_frame=lambda f: _on_frame(f, True),
                    )
                    diag["exposure_fallback_reason"] = "hardware_ae_too_dark"
            else:
                self._set_warmup_phase("software_calib")
                diag = cam.calibrate_fixed_exposure(
                    budget_sec=budget,
                    target_mean=105.0,
                    timeout_ms=300,
                    on_frame=lambda f: _on_frame(f, True),
                )
                diag["exposure_fallback_reason"] = "ae_ag_unsupported"
        except Exception as exc:  # noqa: BLE001
            diag = dict(getattr(cam, "last_diag", {}) or {})
            diag["calib_error"] = str(exc)

        with self._lock:
            self._exposure_ready = True
            meta = dict(self.latest_meta or {})
            meta["camera"] = diag
            meta["exposure_calibrating"] = False
            meta["exposure_ready"] = True
            self.latest_meta = meta

        if self._models_warmed:
            self._set_warmup_phase("ready")
        else:
            self._set_warmup_phase("model_warmup")
        self._maybe_finish_warmup_early()

    def _grab_loop(self) -> None:
        # Camera exposure phase (hardware AE settle or software fallback)
        try:
            self._prepare_exposure_during_warmup()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.error = str(exc)
                self._exposure_ready = True
            self._maybe_finish_warmup_early()

        nodata_streak = 0
        while not self._stop.is_set():
            try:
                frame = self._camera.get_bgr_frame(timeout_ms=500)
                nodata_streak = 0
                self._publish_preview_frame(frame)
                # Real frames also count toward model readiness via detect/ear loops
                if (not self._models_warmed) and self._exposure_ready:
                    # Keep phase visible until dummy model warm finishes
                    if self._warmup_phase != "model_warmup" and time.time() < float(
                        self._warmup_until or 0.0
                    ):
                        self._set_warmup_phase("model_warmup")
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                with self._lock:
                    self.error = msg
                # Sustained GigE NODATA → bump SCPD / shrink ROI / restart stream
                if "0x80000007" in msg or "NODATA" in msg.upper():
                    nodata_streak += 1
                    if nodata_streak in (3, 8, 15) and self._camera is not None:
                        try:
                            self._camera.recover_stream(tighten_roi=(nodata_streak >= 8))
                            with self._lock:
                                meta = dict(self.latest_meta or {})
                                meta["camera"] = dict(
                                    getattr(self._camera, "last_diag", {}) or {}
                                )
                                meta["camera_warning"] = (
                                    "工业相机取流中断，已自动恢复(SCPD/ROI)。"
                                    "若仍黑屏请关闭 MVS 客户端独占预览。"
                                )
                                self.latest_meta = meta
                        except Exception as rec_exc:  # noqa: BLE001
                            with self._lock:
                                self.error = "%s | recover: %s" % (msg, rec_exc)
                else:
                    nodata_streak = 0
                if self._stop.wait(0.05):
                    break

    def _ear_loop(self) -> None:
        """High-rate EAR sampling using shared YOLO face bbox when available."""
        from detection.views import dlib_detector
        from detection.utils.compute_scheduler import compute_scheduler
        from detection.utils.fatigue_tracker import fatigue_tracker

        last_seq = -1
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            with self._lock:
                frame = None if self._latest_bgr is None else self._latest_bgr.copy()
                seq = self._frame_seq
                session_id = self._session_id
                ear_ms = self._ear_interval_ms
                face_bbox = (
                    None if self._latest_face_bbox is None else list(self._latest_face_bbox)
                )
                # Stick to primary only — secondary YOLO faces cause landmark "jumps"
                face_bboxes = [list(face_bbox)] if face_bbox is not None else []
                detect_busy = bool(self._detect_busy)
                in_warmup = time.time() < float(self._warmup_until or 0.0)

            # Cooperative scheduling: EAR always runs; on CPU stretch/yield while YOLO busy.
            # On CUDA, YOLO is on GPU so EAR keeps full rate (no skip).
            ear_ms = compute_scheduler.ear_interval_ms(ear_ms, detect_busy)
            pause = compute_scheduler.ear_busy_pause_sec(detect_busy)
            if pause > 0 and self._stop.wait(pause):
                break

            if frame is None or session_id is None or seq == last_seq:
                if self._stop.wait(0.02):
                    break
                continue

            last_seq = seq
            try:
                # One-shot post-warmup clear (moved out of status() to avoid poll side effects)
                if (not in_warmup) and (not self._warmup_ui_cleared):
                    with self._lock:
                        if not self._warmup_ui_cleared:
                            self._warmup_ui_cleared = True
                            self._last_fatigue_level = 0
                            self._last_yawn = False
                            self._pending_fatigue_event = False
                    try:
                        from detection.utils.fatigue_tracker import (
                            behavior_tracker,
                            fatigue_tracker as _ft,
                        )

                        _ft.reset(session_id)
                        behavior_tracker.reset(session_id)
                    except Exception:  # noqa: BLE001
                        pass

                if dlib_detector is not None:
                    with compute_scheduler.ear_context():
                        # SAME canvas as YOLO / preview overlay — no full↔detect remapping
                        sample = self._resize_for_detect(frame)
                        try:
                            from detection.utils.mono_preprocess import (
                                enhance_for_mono,
                                load_mono_config,
                            )

                            if load_mono_config().get("enabled"):
                                sample = enhance_for_mono(sample)
                        except Exception:  # noqa: BLE001
                            pass

                        sh, sw = sample.shape[:2]
                        # face_bbox already stored in detect coordinates
                        scaled_boxes = [list(b) for b in face_bboxes if b is not None]
                        scaled_primary = list(face_bbox) if face_bbox is not None else None
                        if not scaled_boxes and scaled_primary is not None:
                            scaled_boxes = [scaled_primary]

                        t_ear = time.perf_counter()
                        if not scaled_boxes:
                            dlib_results = {
                                "faces_detected": 0,
                                "eye_aspect_ratio": None,
                                "mouth_aspect_ratio": None,
                                "yawn_detected": False,
                                "fatigue_level": 0,
                                "landmarks": None,
                                "faces": [],
                            }
                            ear_cost = (time.perf_counter() - t_ear) * 1000.0
                            with self._lock:
                                self._latest_landmarks = None
                                self._latest_landmarks_size = None
                                self._latest_landmark_faces = []
                        else:
                            dlib_results = dlib_detector.detect_fatigue_multi(
                                sample,
                                face_bboxes=scaled_boxes,
                                primary_bbox=scaled_primary,
                                allow_hog=False,
                            )
                            ear_cost = (time.perf_counter() - t_ear) * 1000.0

                    # Warmup: run models for cache, but do not feed trackers / UI state
                    if in_warmup:
                        with self._lock:
                            meta = dict(self.latest_meta)
                            timing = dict(meta.get("timing") or {})
                            timing.update(self._merge_timing({"ear_ms": ear_cost}))
                            meta["timing"] = timing
                            meta["scheduler"] = compute_scheduler.snapshot()
                            meta["warming_up"] = True
                            self.latest_meta = meta
                    else:
                        faces_n = int(dlib_results.get("faces_detected") or 0)
                        if faces_n <= 0 and scaled_boxes:
                            faces_n = 1 if dlib_results.get("landmarks") is not None else 0

                        # Keep landmarks in DETECT space — same as YOLO boxes / overlay canvas
                        lm = dlib_results.get("landmarks")
                        face_entries = list(dlib_results.get("faces") or [])
                        lm_size = (int(sw), int(sh)) if lm is not None else None
                        if lm is None and not in_warmup:
                            with self._lock:
                                self._latest_landmarks = None
                                self._latest_landmarks_size = None
                                self._latest_landmark_faces = []

                        # High-rate face overlay refresh (behavior boxes stay on YOLO cadence)
                        if lm is not None and lm_size is not None:
                            try:
                                self._refresh_face_overlay_from_ear(lm, lm_size)
                            except Exception:  # noqa: BLE001
                                pass

                        # Tracker / alerts: primary face only
                        snap = fatigue_tracker.update(
                            session_id,
                            ear=dlib_results.get("eye_aspect_ratio"),
                            yawn_detected=bool(dlib_results.get("yawn_detected")),
                            faces_detected=faces_n,
                            landmarks=lm,
                            landmarks_size=lm_size,
                            faces=face_entries,
                        )

                        faces_meta = []
                        for ent in face_entries:
                            faces_meta.append(
                                {
                                    "bbox": ent.get("bbox"),
                                    "ear": ent.get("eye_aspect_ratio"),
                                    "mar": ent.get("mouth_aspect_ratio"),
                                    "yawn": bool(ent.get("yawn_detected")),
                                    "is_primary": bool(ent.get("is_primary")),
                                }
                            )

                        level = int(snap.fatigue_level)
                        yawn = bool(snap.yawn_detected)
                        with self._lock:
                            prev_level = self._last_fatigue_level
                            prev_yawn = self._last_yawn
                            rising_fatigue = level > prev_level and level >= 2
                            rising_yawn = (not prev_yawn) and yawn
                            if rising_fatigue or rising_yawn:
                                self._pending_fatigue_event = True
                            self._last_fatigue_level = level
                            self._last_yawn = yawn
                            self._latest_faces_meta = faces_meta
                            self._latest_landmarks = lm
                            self._latest_landmarks_size = lm_size
                            self._latest_landmark_faces = face_entries

                            meta = dict(self.latest_meta)
                            timing = dict(meta.get("timing") or {})
                            timing.update(self._merge_timing({"ear_ms": ear_cost}))
                            meta["timing"] = timing
                            meta.update(
                                {
                                    "eye_aspect_ratio": snap.eye_aspect_ratio,
                                    "mouth_aspect_ratio": dlib_results.get("mouth_aspect_ratio"),
                                    "mouth_rel_open": dlib_results.get("mouth_rel_open"),
                                    "yawn_raw": bool(dlib_results.get("yawn_detected")),
                                    "perclos": snap.perclos,
                                    "eye_closed_ms": snap.eye_closed_ms,
                                    "is_microsleep": snap.is_microsleep,
                                    "fatigue_level": snap.fatigue_level,
                                    "yawn_detected": snap.yawn_detected,
                                    "has_landmarks": dlib_results.get("landmarks") is not None,
                                    "face_count": faces_n,
                                    "faces": faces_meta,
                                    "fatigue_event": bool(rising_fatigue or rising_yawn),
                                    "scheduler": compute_scheduler.snapshot(),
                                }
                            )
                            self.latest_meta = meta
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.error = str(exc)
            # Avoid closing DB every EAR tick (ear loop does not use ORM anymore)

            elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            sleep_ms = max(0.0, ear_ms - elapsed_ms)
            if sleep_ms > 0 and self._stop.wait(sleep_ms / 1000.0):
                break

    def _detect_loop(self) -> None:
        """Run YOLO at detection_interval; fatigue comes from EAR loop tracker snapshot."""
        from detection.models import DetectionSession
        from detection.views import persist_detection_snapshot, process_image

        last_seq = -1
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            if self._detect_busy:
                if self._stop.wait(0.02):
                    break
                continue

            with self._lock:
                frame = None if self._latest_bgr is None else self._latest_bgr.copy()
                seq = self._frame_seq
                session_id = self._session_id

            if frame is None or seq == last_seq or session_id is None:
                if self._stop.wait(0.02):
                    break
                continue

            self._detect_busy = True
            last_seq = seq
            try:
                session = DetectionSession.objects.filter(id=session_id).first()
                if session is None:
                    with self._lock:
                        self.error = "Session not found"
                    break
                det_frame = self._resize_for_detect(frame)
                now = time.time()
                with self._lock:
                    pending_event = bool(getattr(self, "_pending_fatigue_event", False))
                    due_persist = (now - float(self._last_persist_at or 0.0)) >= float(
                        self._persist_interval_sec or 2.0
                    )
                    in_warmup = now < float(self._warmup_until or 0.0)

                # Infer without JPEG/DB — main CPU delay source previously.
                result = process_image(
                    det_frame,
                    session,
                    detect_fatigue=False,
                    detect_behaviors=True,
                    include_image_data=False,
                    persist=False,
                )
                behavior_hit = bool(
                    result.get("smoking_detected")
                    or result.get("phone_detected")
                    or result.get("drinking_detected")
                )
                # Do not write history during startup warmup window
                face_bbox = result.get("face_bbox")
                face_bboxes = result.get("face_bboxes") or []
                try:
                    primary_draw = (
                        [int(v) for v in face_bbox] if face_bbox is not None else None
                    )
                except (TypeError, ValueError):
                    primary_draw = None

                # One dlib pass on this det_frame → rigid overlay for grab + persist
                overlay_packet: Dict[str, Any] = {
                    "bbox": primary_draw,
                    "landmarks": None,
                    "detections": result.get("detections") or [],
                    "wh": (int(det_frame.shape[1]), int(det_frame.shape[0])),
                }
                try:
                    overlay_packet = self._compute_detect_overlay(
                        det_frame,
                        primary_draw,
                        detections=result.get("detections") or [],
                    )
                    self._store_overlay(overlay_packet)
                except Exception:  # noqa: BLE001
                    pass

                if (not in_warmup) and (pending_event or behavior_hit or due_persist):
                    persist_result = dict(result)
                    persist_result["overlay_landmarks"] = overlay_packet.get("landmarks")
                    persist_result["overlay_landmarks_size"] = overlay_packet.get("wh")
                    # Prefer landmark-locked box for any face draw metadata
                    if overlay_packet.get("bbox") is not None:
                        persist_result["face_bbox"] = overlay_packet.get("bbox")
                    result = persist_detection_snapshot(
                        det_frame, session, persist_result
                    )
                    with self._lock:
                        self._last_persist_at = time.time()
                        self._pending_fatigue_event = False

                detect_loop_ms = (time.perf_counter() - loop_start) * 1000.0
                timing_src = dict(result.get("timing") or {})
                timing_src["detect_loop_ms"] = detect_loop_ms

                meta = {
                    "face_detected": result.get("face_detected"),
                    "smoking_detected": result.get("smoking_detected"),
                    "phone_detected": result.get("phone_detected"),
                    "drinking_detected": result.get("drinking_detected"),
                    "yawn_detected": result.get("yawn_detected"),
                    "fatigue_level": result.get("fatigue_level"),
                    "eye_aspect_ratio": result.get("eye_aspect_ratio"),
                    "mouth_aspect_ratio": result.get("mouth_aspect_ratio"),
                    "perclos": result.get("perclos"),
                    "eye_closed_ms": result.get("eye_closed_ms"),
                    "is_microsleep": result.get("is_microsleep"),
                    "has_landmarks": result.get("has_landmarks"),
                    "face_count": result.get("face_count"),
                    "detection_id": result.get("detection_id"),
                    "behavior_debug": result.get("behavior_debug") or {},
                    "confirm_progress": result.get("confirm_progress") or {},
                    "fatigue_event": False,
                }
                with self._lock:
                    dh, dw = det_frame.shape[:2]
                    primary_det = [int(v) for v in face_bbox] if face_bbox is not None else None
                    all_det = []
                    for b in face_bboxes:
                        try:
                            all_det.append([int(v) for v in b])
                        except (TypeError, ValueError):
                            continue
                    self._update_tracked_faces(primary_det, all_det, (dw, dh))

                    prev = dict(self.latest_meta or {})
                    prev_timing = dict(prev.get("timing") or {})
                    merged = self._merge_timing(timing_src)
                    for k, v in prev_timing.items():
                        if k.startswith("ear"):
                            merged.setdefault(k, v)
                    meta["timing"] = merged
                    # Keep EAR-loop faces list / fatigue metrics when fresher
                    if prev.get("faces"):
                        meta["faces"] = prev.get("faces")
                    if prev.get("face_count") is not None and meta.get("face_count") is None:
                        meta["face_count"] = prev.get("face_count")
                    try:
                        if int(prev.get("fatigue_level") or 0) >= int(meta.get("fatigue_level") or 0):
                            for k in (
                                "fatigue_level",
                                "eye_aspect_ratio",
                                "mouth_aspect_ratio",
                                "perclos",
                                "eye_closed_ms",
                                "is_microsleep",
                                "yawn_detected",
                                "has_landmarks",
                                "faces",
                                "face_count",
                            ):
                                if k in prev and prev[k] is not None:
                                    meta[k] = prev[k]
                    except (TypeError, ValueError):
                        pass
                    try:
                        from detection.utils.compute_scheduler import compute_scheduler

                        meta["scheduler"] = compute_scheduler.snapshot()
                    except Exception:  # noqa: BLE001
                        meta["scheduler"] = prev.get("scheduler")
                    # Preserve warmup / camera health fields written by grab loop
                    for k in (
                        "camera",
                        "camera_warning",
                        "warmup_phase",
                        "warmup_detail",
                        "exposure_ready",
                        "models_warmed",
                        "exposure_calibrating",
                        "warmup_early_exit",
                    ):
                        if k in prev and k not in meta:
                            meta[k] = prev[k]
                    still_warming = time.time() < float(self._warmup_until or 0.0)
                    meta["warming_up"] = still_warming
                    meta["exposure_ready"] = bool(self._exposure_ready)
                    meta["models_warmed"] = bool(self._models_warmed)
                    self.latest_meta = meta
                if in_warmup:
                    self._maybe_finish_warmup_early()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.error = str(exc)
            finally:
                self._detect_busy = False
                try:
                    connection.close()
                except Exception:  # noqa: BLE001
                    pass

            elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            sleep_ms = max(0.0, self._interval_ms - elapsed_ms)
            if sleep_ms > 0 and self._stop.wait(sleep_ms / 1000.0):
                break


mvs_grabber = MvsGrabber()
