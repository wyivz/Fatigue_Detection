# -*- coding: utf-8 -*-
"""Background GigE grab + detection (preview FPS decoupled from detect interval)."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from django.db import connection

from .camera import HikCamera


class MvsGrabber:
    """High-rate preview grabber + lower-rate detection worker."""

    def __init__(self):
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._grab_thread: Optional[threading.Thread] = None
        self._detect_thread: Optional[threading.Thread] = None
        self._camera: Optional[HikCamera] = None
        self._session_id: Optional[int] = None
        self._user_id: Optional[int] = None
        self._interval_ms = 500
        self._preview_max_width = 960
        self._preview_jpeg_quality = 70
        self._latest_bgr: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._detect_busy = False
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
        with self._lock:
            if self.running:
                raise RuntimeError("MVS grabber already running")
            self._stop.clear()
            self.error = None
            self.latest_jpeg = None
            self.latest_meta = {}
            self._latest_bgr = None
            self._frame_seq = 0
            self._session_id = session_id
            self._user_id = user_id
            self._interval_ms = max(50, int(interval_ms or 500))
            cam = HikCamera()
            if camera_ip:
                cam.open_by_ip(camera_ip)
            else:
                cam.open_by_index(int(device_index or 0))
            cam.start_grab()
            self._camera = cam
            self.running = True
            self._grab_thread = threading.Thread(
                target=self._grab_loop, name="mvs-grab", daemon=True
            )
            self._detect_thread = threading.Thread(
                target=self._detect_loop, name="mvs-detect", daemon=True
            )
            self._grab_thread.start()
            self._detect_thread.start()

    def stop(self, complete_session: bool = True) -> None:
        self._stop.set()
        threads = []
        with self._lock:
            if self._grab_thread:
                threads.append(self._grab_thread)
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
            self._detect_thread = None
            self._detect_busy = False
            session_id = self._session_id
            self._session_id = None
        if complete_session and session_id:
            self._complete_session(session_id)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "session_id": self._session_id,
                "error": self.error,
                "meta": dict(self.latest_meta),
                "has_frame": self.latest_jpeg is not None,
            }

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg

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
        h, w = frame.shape[:2]
        max_w = self._preview_max_width
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
        """Pull frames continuously for smooth MJPEG preview."""
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

    def _detect_loop(self) -> None:
        """Run YOLO/dlib at detection_interval only; does not drive preview FPS."""
        from detection.models import DetectionSession
        from detection.views import process_image

        last_seq = -1
        while not self._stop.is_set():
            loop_start = time.time()
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
                # Detection can use a moderately sized frame for speed
                det_frame = self._resize_for_preview(frame)
                result = process_image(det_frame, session, True, True)
                meta = {
                    "face_detected": result.get("face_detected"),
                    "smoking_detected": result.get("smoking_detected"),
                    "phone_detected": result.get("phone_detected"),
                    "drinking_detected": result.get("drinking_detected"),
                    "yawn_detected": result.get("yawn_detected"),
                    "fatigue_level": result.get("fatigue_level"),
                    "eye_aspect_ratio": result.get("eye_aspect_ratio"),
                    "detection_id": result.get("detection_id"),
                }
                with self._lock:
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

            elapsed_ms = (time.time() - loop_start) * 1000.0
            sleep_ms = max(0.0, self._interval_ms - elapsed_ms)
            if sleep_ms > 0 and self._stop.wait(sleep_ms / 1000.0):
                break


mvs_grabber = MvsGrabber()
