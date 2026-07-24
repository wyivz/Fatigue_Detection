# -*- coding: utf-8 -*-
"""Background GigE grab + detection thread (process singleton)."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import cv2
from django.db import connection

from .camera import HikCamera


class MvsGrabber:
    def __init__(self):
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._camera: Optional[HikCamera] = None
        self._session_id: Optional[int] = None
        self._user_id: Optional[int] = None
        self._interval_ms = 500
        self._busy = False
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
            self._thread = threading.Thread(target=self._loop, name="mvs-grabber", daemon=True)
            self._thread.start()

    def stop(self, complete_session: bool = True) -> None:
        self._stop.set()
        thread = None
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)
        with self._lock:
            if self._camera:
                try:
                    self._camera.stop_and_close()
                except Exception:  # noqa: BLE001
                    pass
                self._camera = None
            self.running = False
            self._thread = None
            self._busy = False
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

    def _loop(self) -> None:
        from detection.models import DetectionSession
        from detection.views import process_image

        while not self._stop.is_set():
            started = time.time()
            if self._busy:
                time.sleep(0.02)
                continue
            self._busy = True
            try:
                frame = self._camera.get_bgr_frame(timeout_ms=1000)
                session = DetectionSession.objects.filter(id=self._session_id).first()
                if session is None:
                    self.error = "Session not found"
                    break
                result = process_image(frame, session, True, True)
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
                # Prefer annotated image from process_image pipeline: re-encode latest saved path is heavy;
                # decode base64 image_data if present.
                jpeg = None
                image_data = result.get("image_data") or ""
                if image_data.startswith("data:image"):
                    import base64

                    b64 = image_data.split(",", 1)[1]
                    jpeg = base64.b64decode(b64)
                else:
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        jpeg = buf.tobytes()
                with self._lock:
                    self.latest_jpeg = jpeg
                    self.latest_meta = meta
                    self.error = None
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.error = str(exc)
            finally:
                self._busy = False
                try:
                    connection.close()
                except Exception:  # noqa: BLE001
                    pass
                elapsed = (time.time() - started) * 1000.0
                sleep_ms = max(0.0, self._interval_ms - elapsed)
                if sleep_ms > 0 and not self._stop.wait(sleep_ms / 1000.0):
                    pass

        with self._lock:
            if self._camera:
                try:
                    self._camera.stop_and_close()
                except Exception:  # noqa: BLE001
                    pass
                self._camera = None
            self.running = False


mvs_grabber = MvsGrabber()
