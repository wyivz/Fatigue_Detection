# -*- coding: utf-8 -*-
"""Background GigE grab + detection (preview FPS decoupled from detect interval)."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

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
        self._preview_max_width = 960
        self._detect_max_width = 960
        self._preview_jpeg_quality = 70
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_face_bbox: Optional[List[int]] = None
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
        self._warmup_ui_cleared = False
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
            self._warmup_until = time.time() + self._warmup_sec
            self._warmup_ui_cleared = False
            fatigue_tracker.reset(session_id)
            behavior_tracker.reset(session_id)
            fatigue_tracker.configure(session_id, configs)
            behavior_tracker.configure(session_id, configs)

            cam = HikCamera()
            if camera_ip:
                cam.open_by_ip(camera_ip)
            else:
                cam.open_by_index(int(device_index or 0))
            cam.start_grab()
            # Discard a few frames so continuous AE/gain can settle (otherwise first
            # seconds look dark + detections look "stuck").
            for _ in range(10):
                try:
                    cam.get_bgr_frame(timeout_ms=400)
                except Exception:  # noqa: BLE001
                    break
            self._camera = cam
            self.running = True
            # Warmup clock starts when streams are actually up
            self._warmup_until = time.time() + float(self._warmup_sec or 5.0)
            # Clear any fatigue accumulation from dark AE frames after streams settle
            try:
                fatigue_tracker.reset(session_id)
                behavior_tracker.reset(session_id)
                fatigue_tracker.configure(session_id, configs)
                behavior_tracker.configure(session_id, configs)
            except Exception:  # noqa: BLE001
                pass
            self._grab_thread = threading.Thread(
                target=self._grab_loop, name="mvs-grab", daemon=True
            )
            self._ear_thread = threading.Thread(
                target=self._ear_loop, name="mvs-ear", daemon=True
            )
            self._detect_thread = threading.Thread(
                target=self._detect_loop, name="mvs-detect", daemon=True
            )
            self._grab_thread.start()
            self._ear_thread.start()
            self._detect_thread.start()

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
            self._detect_busy = False
            self._latest_face_bbox = None
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

    def status(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            warming = bool(self.running and now < float(self._warmup_until or 0.0))
            remain = max(0.0, float(self._warmup_until or 0.0) - now) if warming else 0.0
            if self.running and (not warming) and (not self._warmup_ui_cleared):
                self._warmup_ui_cleared = True
                sid = self._session_id
                if sid is not None:
                    try:
                        from detection.utils.fatigue_tracker import (
                            behavior_tracker,
                            fatigue_tracker,
                        )

                        fatigue_tracker.reset(sid)
                        behavior_tracker.reset(sid)
                        self._last_fatigue_level = 0
                        self._last_yawn = False
                        self._pending_fatigue_event = False
                    except Exception:  # noqa: BLE001
                        pass
            meta = dict(self.latest_meta)
            if warming:
                meta = {
                    "timing": meta.get("timing") or {},
                    "scheduler": meta.get("scheduler"),
                    "warming_up": True,
                }
            return {
                "running": self.running,
                "session_id": self._session_id,
                "error": self.error,
                "meta": meta,
                "has_frame": self.latest_jpeg is not None,
                "warming_up": warming,
                "warmup_remaining_sec": round(remain, 1),
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

    def _encode_preview_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        preview = self._resize_for_preview(frame)
        ok, buf = cv2.imencode(
            ".jpg",
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._preview_jpeg_quality)],
        )
        if not ok:
            return None
        return buf.tobytes()

    def _grab_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._camera.get_bgr_frame(timeout_ms=500)
                jpeg = self._encode_preview_jpeg(frame)
                with self._lock:
                    self._latest_bgr = frame
                    self._frame_seq += 1
                    if jpeg:
                        self.latest_jpeg = jpeg
                    self.error = None
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.error = str(exc)
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
                frame = None if self._latest_bgr is None else self._latest_bgr
                seq = self._frame_seq
                session_id = self._session_id
                ear_ms = self._ear_interval_ms
                face_bbox = (
                    None if self._latest_face_bbox is None else list(self._latest_face_bbox)
                )
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
                if dlib_detector is not None:
                    with compute_scheduler.ear_context():
                        sample = self._resize_for_preview(frame)
                        scaled_bbox = face_bbox
                        if face_bbox is not None:
                            fh, fw = frame.shape[:2]
                            sh, sw = sample.shape[:2]
                            if fw > 0 and fh > 0 and (sw != fw or sh != fh):
                                sx = sw / float(fw)
                                sy = sh / float(fh)
                                scaled_bbox = [
                                    int(face_bbox[0] * sx),
                                    int(face_bbox[1] * sy),
                                    int(face_bbox[2] * sx),
                                    int(face_bbox[3] * sy),
                                ]

                        t_ear = time.perf_counter()
                        dlib_results = dlib_detector.detect_fatigue(
                            sample, face_bbox=scaled_bbox
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
                        if faces_n <= 0 and scaled_bbox is not None:
                            faces_n = 1 if dlib_results.get("landmarks") is not None else 0

                        snap = fatigue_tracker.update(
                            session_id,
                            ear=dlib_results.get("eye_aspect_ratio"),
                            yawn_detected=bool(dlib_results.get("yawn_detected")),
                            faces_detected=faces_n,
                            landmarks=dlib_results.get("landmarks"),
                        )

                        level = int(snap.fatigue_level)
                        yawn = bool(snap.yawn_detected)
                        with self._lock:
                            prev_level = self._last_fatigue_level
                            prev_yawn = self._last_yawn
                            rising_fatigue = prev_level < 2 <= level
                            rising_yawn = (not prev_yawn) and yawn
                            if rising_fatigue or rising_yawn:
                                self._pending_fatigue_event = True
                            self._last_fatigue_level = level
                            self._last_yawn = yawn

                            meta = dict(self.latest_meta)
                            timing = dict(meta.get("timing") or {})
                            timing.update(self._merge_timing({"ear_ms": ear_cost}))
                            meta["timing"] = timing
                            meta.update(
                                {
                                    "eye_aspect_ratio": snap.eye_aspect_ratio,
                                    "mouth_aspect_ratio": dlib_results.get("mouth_aspect_ratio"),
                                    "perclos": snap.perclos,
                                    "eye_closed_ms": snap.eye_closed_ms,
                                    "is_microsleep": snap.is_microsleep,
                                    "fatigue_level": snap.fatigue_level,
                                    "yawn_detected": snap.yawn_detected,
                                    "has_landmarks": dlib_results.get("landmarks") is not None,
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
                if (not in_warmup) and (pending_event or behavior_hit or due_persist):
                    result = persist_detection_snapshot(det_frame, session, result)
                    with self._lock:
                        self._last_persist_at = time.time()
                        self._pending_fatigue_event = False

                detect_loop_ms = (time.perf_counter() - loop_start) * 1000.0

                face_bbox = result.get("face_bbox")
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
                    "detection_id": result.get("detection_id"),
                    "behavior_debug": result.get("behavior_debug") or {},
                    "confirm_progress": result.get("confirm_progress") or {},
                    "fatigue_event": False,
                }
                with self._lock:
                    if face_bbox is not None:
                        fh, fw = frame.shape[:2]
                        dh, dw = det_frame.shape[:2]
                        if dw > 0 and dh > 0 and (dw != fw or dh != fh):
                            sx = fw / float(dw)
                            sy = fh / float(dh)
                            self._latest_face_bbox = [
                                int(face_bbox[0] * sx),
                                int(face_bbox[1] * sy),
                                int(face_bbox[2] * sx),
                                int(face_bbox[3] * sy),
                            ]
                        else:
                            self._latest_face_bbox = [int(v) for v in face_bbox]
                    prev = dict(self.latest_meta or {})
                    prev_timing = dict(prev.get("timing") or {})
                    merged = self._merge_timing(timing_src)
                    for k, v in prev_timing.items():
                        if k.startswith("ear"):
                            merged.setdefault(k, v)
                    meta["timing"] = merged
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
                    self.latest_meta = meta
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
