# -*- coding: utf-8 -*-
"""Background archive-JPEG-encode + DB-write worker.

`persist_detection_snapshot` used to run JPEG encoding (on the *original*,
possibly multi-megapixel frame) and the SQLite INSERT synchronously inside
the Django request thread (browser path) or the MVS detect thread. For a
high-resolution GigE frame this can take tens to a hundred+ ms, which
directly inflates `detection_interval` and starves the grab/detect loop.

This module moves that work onto a single background daemon thread with a
bounded queue: the hot path only copies the frame + result dict and hands
them off, returning immediately. If the queue is ever full (persistence
falling behind, e.g. slow disk), the snapshot is dropped and counted rather
than blocking detection.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("detection.persist")

_QUEUE_MAXSIZE = 16
_queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
_worker_thread: Optional[threading.Thread] = None
_start_lock = threading.Lock()
_dropped_count = 0
_persisted_count = 0


def _do_persist(image: np.ndarray, session_id: int, results: Dict[str, Any]) -> None:
    from django.core.files.base import ContentFile
    from django.utils import timezone

    from detection.models import DetectionResult, DetectionSession
    from detection.utils.archive_pipeline import encode_jpeg_bgr, render_annotated_bgr

    session = DetectionSession.objects.filter(id=session_id).first()
    if session is None:
        return

    persist_results = dict(results)
    if persist_results.get("overlay_landmarks") is not None:
        persist_results["landmarks"] = persist_results.get("overlay_landmarks")
        if persist_results.get("overlay_landmarks_size"):
            persist_results["landmarks_size"] = persist_results.get(
                "overlay_landmarks_size"
            )

    img = render_annotated_bgr(image, persist_results)

    row = DetectionResult(
        session=session,
        face_detected=bool(results.get("face_detected")),
        smoking_detected=bool(results.get("smoking_detected")),
        phone_detected=bool(results.get("phone_detected")),
        drinking_detected=bool(results.get("drinking_detected")),
        eye_aspect_ratio=results.get("eye_aspect_ratio"),
        yawn_detected=bool(results.get("yawn_detected")),
        fatigue_level=int(results.get("fatigue_level") or 0),
        perclos=results.get("perclos"),
        eye_closed_ms=results.get("eye_closed_ms"),
    )
    result_filename = (
        f"result_{session.id}_{timezone.now().strftime('%Y%m%d%H%M%S')}_"
        f"{os.urandom(3).hex()}.jpg"
    )
    try:
        raw = encode_jpeg_bgr(img, 75)
        if raw is not None:
            row.result_image.save(result_filename, ContentFile(raw), save=False)
    except Exception:  # noqa: BLE001
        logger.exception("archive image encode/save failed")
    row.save()
    global _persisted_count
    _persisted_count += 1


def _worker_loop() -> None:
    from django.db import close_old_connections

    while True:
        item = _queue.get()
        if item is None:
            continue
        image, session_id, results = item
        try:
            _do_persist(image, session_id, results)
        except Exception:  # noqa: BLE001
            logger.exception("persist worker failed for session=%s", session_id)
        finally:
            try:
                close_old_connections()
            except Exception:  # noqa: BLE001
                pass


def _ensure_worker() -> None:
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    with _start_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(
            target=_worker_loop, name="detection-persist", daemon=True
        )
        _worker_thread.start()


def enqueue_persist(image: np.ndarray, session: Any, results: Dict[str, Any]) -> bool:
    """Hand a snapshot off to the background persist worker.

    Copies `image` and `results` so the caller's buffers can be mutated /
    reused immediately after this call returns. Returns False (and drops
    the snapshot) if the queue is full instead of blocking the caller.
    """
    sid = getattr(session, "id", None)
    if sid is None:
        return False
    _ensure_worker()
    try:
        _queue.put_nowait((image.copy(), int(sid), dict(results)))
        return True
    except queue.Full:
        global _dropped_count
        _dropped_count += 1
        logger.warning(
            "persist queue full (maxsize=%d); dropped snapshot (total_dropped=%d)",
            _QUEUE_MAXSIZE,
            _dropped_count,
        )
        return False


def stats() -> Dict[str, int]:
    return {
        "queued": _queue.qsize(),
        "dropped": _dropped_count,
        "persisted": _persisted_count,
    }
