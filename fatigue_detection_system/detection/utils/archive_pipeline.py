# -*- coding: utf-8 -*-
"""
Archive-faithful detection (matches D:\\归档\\fatigue_detection_system\\detection\\views.py).

Exact order:
  YOLO → draw boxes on a copy → dlib HOG + 68pts on that canvas → instant EAR/MAR

Large frames are *uniformly scaled* (full FOV kept, no ROI crop) so HOG sees
webcam-like face sizes — full 5MP HOG is both slow and inaccurate.
Coordinates are remapped back to the original frame for live overlay / persist.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("detection.detect")


def _uniform_scale(
    image: np.ndarray, max_side: int
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Scale so max(h,w) <= max_side. Returns (scaled, scale, orig_wh)."""
    h, w = image.shape[:2]
    orig = (int(w), int(h))
    if max_side <= 0 or max(h, w) <= max_side:
        return image, 1.0, orig
    scale = float(max_side) / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    scaled = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return scaled, scale, orig


def _remap_xy(x: float, y: float, scale: float) -> List[int]:
    if scale == 1.0:
        return [int(x), int(y)]
    inv = 1.0 / scale
    return [int(round(float(x) * inv)), int(round(float(y) * inv))]


def _remap_bbox(bbox: List[int], scale: float) -> List[int]:
    x1, y1 = _remap_xy(bbox[0], bbox[1], scale)
    x2, y2 = _remap_xy(bbox[2], bbox[3], scale)
    return [x1, y1, x2, y2]


def _remap_landmarks(landmarks, scale: float):
    if landmarks is None or scale == 1.0:
        return landmarks
    pts = np.asarray(landmarks)
    out = np.zeros_like(pts)
    inv = 1.0 / scale
    out[:, 0] = np.rint(pts[:, 0].astype(np.float64) * inv).astype(pts.dtype)
    out[:, 1] = np.rint(pts[:, 1].astype(np.float64) * inv).astype(pts.dtype)
    return out


def run_archive_detect(
    image: np.ndarray,
    session_id: Optional[int] = None,
    *,
    detect_fatigue: bool = True,
    detect_behaviors: bool = True,
    max_side: int = 960,
    dlib_refine_mode: str = "light",
    dlib_landmark_mode: str = "yolo",
    yolo_detector=None,
    dlib_detector=None,
) -> Dict[str, Any]:
    """
    Byte-for-byte logic of archive process_image (plus optional uniform scale).

    max_side: 0 = no scale; default 960 matches typical webcam / good HOG size.
    """
    if yolo_detector is None or dlib_detector is None:
        try:
            from detection.views import dlib_detector as _dlib
            from detection.views import yolo_detector as _yolo

            if yolo_detector is None:
                yolo_detector = _yolo
            if dlib_detector is None:
                dlib_detector = _dlib
        except Exception:  # noqa: BLE001
            pass

    t0 = time.perf_counter()
    timing = {"yolo_ms": 0.0, "dlib_ms": 0.0, "total_ms": 0.0}

    work, scale, orig_wh = _uniform_scale(image, int(max_side or 0))
    ow, oh = orig_wh

    results: Dict[str, Any] = {
        "status": "success",
        "face_bbox": None,
        "detections": [],
        # Overlay / persist use ORIGINAL frame coordinates
        "image_size": [ow, oh],
        "landmarks_size": [ow, oh],
        "landmarks": None,
        "image_data": None,
        "detect_scale": float(scale),
        "detect_size": [int(work.shape[1]), int(work.shape[0])],
    }

    processed = None

    if detect_behaviors and yolo_detector is not None:
        t_yolo = time.perf_counter()
        # Archive: results = self.model(image) with global conf
        yolo_out = yolo_detector.detect_archive(work)
        processed = yolo_detector.process_results_simple(yolo_out, session_id=session_id)
        timing["yolo_ms"] = round((time.perf_counter() - t_yolo) * 1000.0, 1)
        # NOTE: dlib runs on the clean `work` buffer below (archive-faithful —
        # HOG never sees drawn boxes). We used to also render an annotated
        # copy here (`draw_results`) but nothing consumed it — it was a pure
        # copy+draw cost paid on every frame. Persist/live-overlay drawing is
        # done separately (render_annotated_bgr) only when actually needed.

        dets = []
        for det in processed.get("detections") or []:
            d = dict(det)
            d["bbox"] = _remap_bbox(list(d["bbox"]), scale)
            dets.append(d)
        face_bbox = processed.get("face_bbox")
        if face_bbox is not None:
            face_bbox = _remap_bbox(list(face_bbox), scale)
        face_bboxes = [
            _remap_bbox(list(b), scale) for b in (processed.get("face_bboxes") or [])
        ]
        results.update(
            {
                "face_detected": bool(processed["face_detected"]),
                "smoking_detected": bool(processed["smoking_detected"]),
                "phone_detected": bool(processed["phone_detected"]),
                "drinking_detected": bool(processed["drinking_detected"]),
                "face_bbox": face_bbox,
                "face_bboxes": face_bboxes,
                "face_count": int(processed.get("face_count") or 0),
                "detections": dets,
                "behavior_debug": processed.get("behavior_debug") or {},
                "confirm_progress": {},
            }
        )
    else:
        results.update(
            {
                "face_detected": False,
                "smoking_detected": False,
                "phone_detected": False,
                "drinking_detected": False,
                "face_bbox": None,
                "face_bboxes": [],
                "face_count": 0,
                "detections": [],
                "behavior_debug": {},
                "confirm_progress": {},
            }
        )

    if detect_fatigue and dlib_detector is not None:
        try:
            t_dlib = time.perf_counter()
            yolo_face_boxes = None
            yolo_primary = None
            if processed is not None:
                yolo_face_boxes = processed.get("face_bboxes") or None
                yolo_primary = processed.get("face_bbox")
            # Tie landmarks to the same face YOLO is tracking: IoU-match HOG/YOLO
            # candidates against the sticky primary bbox instead of blindly taking
            # HOG's first detection (which can belong to a different face when
            # multiple people are visible, causing landmarks to "jump" faces).
            # Falls back to full-frame HOG (allow_hog) when YOLO found no face,
            # or when dlib_landmark_mode is explicitly forced to "hog".
            if str(dlib_landmark_mode or "yolo").strip().lower() == "hog" or not (
                yolo_face_boxes or yolo_primary
            ):
                dlib_results = dlib_detector.detect_fatigue_multi(
                    work,
                    face_bboxes=None,
                    primary_bbox=None,
                    allow_hog=True,
                    refine_rect="off",
                )
            else:
                dlib_results = dlib_detector.detect_fatigue_multi(
                    work,
                    face_bboxes=yolo_face_boxes,
                    primary_bbox=yolo_primary,
                    allow_hog=True,
                    refine_rect=dlib_refine_mode,
                )
            timing["dlib_ms"] = round((time.perf_counter() - t_dlib) * 1000.0, 1)

            ear = dlib_results.get("eye_aspect_ratio")
            yawn = bool(dlib_results.get("yawn_detected"))
            level = int(dlib_results.get("fatigue_level") or 0)
            lm = dlib_results.get("landmarks")
            faces_n = int(dlib_results.get("faces_detected") or 0)
            lm_orig = _remap_landmarks(lm, scale)

            # PERCLOS / microsleep bookkeeping lives in fatigue_tracker; feed it
            # every frame and use its returned snapshot (not a hardcoded 0) so
            # temporal metrics actually reach the UI/alerts. In "instant" mode
            # snapshot.fatigue_level is mathematically identical to the dlib
            # per-frame level above; in "temporal" mode it is PERCLOS-driven.
            fatigue_snap = None
            if session_id is not None:
                try:
                    from detection.utils.fatigue_tracker import fatigue_tracker

                    fatigue_snap = fatigue_tracker.update(
                        int(session_id),
                        ear=ear,
                        yawn_detected=yawn,
                        faces_detected=faces_n,
                        landmarks=lm_orig,
                        landmarks_size=(ow, oh) if lm_orig is not None else None,
                        landmark_reliable=True,
                    )
                    level = int(fatigue_snap.fatigue_level)
                except Exception:  # noqa: BLE001
                    fatigue_snap = None

            perclos = float(fatigue_snap.perclos) if fatigue_snap is not None else 0.0
            eye_closed_ms = int(fatigue_snap.eye_closed_ms) if fatigue_snap is not None else 0
            is_microsleep = bool(fatigue_snap.is_microsleep) if fatigue_snap is not None else False

            face_entries = dlib_results.get("faces") or []
            primary_idx = dlib_results.get("primary_index")
            primary_entry = (
                face_entries[primary_idx]
                if isinstance(primary_idx, int) and 0 <= primary_idx < len(face_entries)
                else None
            )
            faces_ui = []
            for ent in face_entries:
                ent_lm = ent.get("landmarks")
                if ent_lm is not None:
                    ent_lm = _remap_landmarks(np.asarray(ent_lm), scale)
                faces_ui.append(
                    {
                        "ear": ent.get("eye_aspect_ratio"),
                        "is_primary": bool(ent.get("is_primary")),
                        "landmarks": (
                            [[int(p[0]), int(p[1])] for p in ent_lm]
                            if ent_lm is not None
                            else None
                        ),
                    }
                )

            results.update(
                {
                    "eye_aspect_ratio": float(ear) if ear is not None else None,
                    "mouth_aspect_ratio": (
                        float(dlib_results["mouth_aspect_ratio"])
                        if dlib_results.get("mouth_aspect_ratio") is not None
                        else None
                    ),
                    "yawn_detected": yawn,
                    "fatigue_level": level,
                    "perclos": perclos,
                    "eye_closed_ms": eye_closed_ms,
                    "is_microsleep": is_microsleep,
                    "has_landmarks": lm_orig is not None,
                    "face_count": max(int(results.get("face_count") or 0), faces_n),
                    "faces": faces_ui,
                    "dlib_debug": {
                        "hog_faces": faces_n,
                        "from_hog": bool(primary_entry.get("from_hog")) if primary_entry else None,
                        "landmark_quality": (
                            primary_entry.get("landmark_quality") if primary_entry else None
                        ),
                        "pose_ok": bool(primary_entry.get("pose_ok", True)) if primary_entry else None,
                    },
                }
            )
            if lm_orig is not None:
                results["landmarks"] = [
                    [int(p[0]), int(p[1])] for p in np.asarray(lm_orig)
                ]
            else:
                results["landmarks"] = None
        except Exception:  # noqa: BLE001
            logger.exception("archive dlib failed")
            results.update(
                {
                    "eye_aspect_ratio": None,
                    "yawn_detected": False,
                    "fatigue_level": 0,
                    "perclos": 0.0,
                    "eye_closed_ms": 0,
                    "is_microsleep": False,
                    "has_landmarks": False,
                    "landmarks": None,
                }
            )
    else:
        results.update(
            {
                "eye_aspect_ratio": None,
                "yawn_detected": False,
                "fatigue_level": 0,
                "perclos": 0.0,
                "eye_closed_ms": 0,
                "is_microsleep": False,
                "has_landmarks": False,
                "landmarks": None,
            }
        )

    timing["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    results["timing"] = timing
    results["result_image_url"] = None
    results["detection_id"] = None
    # Never put numpy frames on the result dict — breaks JsonResponse.
    return results


def render_annotated_bgr(
    image: np.ndarray,
    results: Dict[str, Any],
    *,
    yolo_detector=None,
    dlib_detector=None,
) -> np.ndarray:
    """Draw on ORIGINAL-resolution image for DB JPEG."""
    if yolo_detector is None or dlib_detector is None:
        try:
            from detection.views import dlib_detector as _dlib
            from detection.views import yolo_detector as _yolo

            if yolo_detector is None:
                yolo_detector = _yolo
            if dlib_detector is None:
                dlib_detector = _dlib
        except Exception:  # noqa: BLE001
            pass

    out = image.copy()
    dets = results.get("detections") or []
    if yolo_detector is not None and dets:
        out = yolo_detector.draw_results(
            out,
            {
                "detections": dets,
                "face_detected": results.get("face_detected"),
                "smoking_detected": results.get("smoking_detected"),
                "phone_detected": results.get("phone_detected"),
                "drinking_detected": results.get("drinking_detected"),
            },
        )
    if dlib_detector is not None and (
        results.get("landmarks") is not None
        or results.get("eye_aspect_ratio") is not None
        or int(results.get("fatigue_level") or 0) > 0
    ):
        lm = results.get("landmarks")
        lm_arr = np.asarray(lm, dtype=int) if lm is not None else None
        out = dlib_detector.draw_fatigue_results(
            out,
            {
                "landmarks": lm_arr,
                "landmarks_size": results.get("landmarks_size"),
                "eye_aspect_ratio": results.get("eye_aspect_ratio"),
                "mouth_aspect_ratio": results.get("mouth_aspect_ratio"),
                "yawn_detected": results.get("yawn_detected"),
                "fatigue_level": results.get("fatigue_level"),
                "perclos": results.get("perclos"),
                "eye_closed_ms": results.get("eye_closed_ms"),
            },
        )
    return out


def encode_jpeg_bgr(image: np.ndarray, quality: int = 75) -> Optional[bytes]:
    ok, buf = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        return None
    return buf.tobytes()
