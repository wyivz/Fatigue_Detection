# -*- coding: utf-8 -*-
"""dlib face landmarks + EAR/MAR fatigue helpers (thread-safe)."""
from __future__ import annotations

import os
import threading

import cv2
import dlib
import numpy as np
from django.conf import settings
from scipy.spatial import distance as dist


class DlibDetector:
    """
    使用 dlib 进行面部特征检测，用于疲劳驾驶检测。
    对外检测路径加锁，避免 MVS EAR 环与 detect 环并发竞态。
    """

    def __init__(self, predictor_path=None):
        if predictor_path is None:
            predictor_path = os.path.join(
                settings.BASE_DIR, "weights", "shape_predictor_68_face_landmarks.dat"
            )

        self._lock = threading.RLock()
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError:
            self.predictor = None
            print(f"无法加载面部特征点预测器，路径: {predictor_path}")

        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 41
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 47
        self.MOUTH_START = 48
        self.MOUTH_END = 67
        # Cap faces per frame so EAR loop stays real-time with multiple people.
        self.MAX_FACES = 4

        self.load_config()

    def warmup(self) -> float:
        """Touch HOG + predictor once so first live frame is not cold."""
        import time as _time

        t0 = _time.perf_counter()
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            # Synthetic mid-frame box so predictor path is exercised
            self.detect_fatigue(dummy, face_bbox=[160, 80, 480, 400])
        except Exception as e:  # noqa: BLE001
            print(f"dlib warmup failed: {e}")
        return (_time.perf_counter() - t0) * 1000.0

    def load_config(self):
        from detection.models import SystemConfig

        self.EYE_AR_THRESH = 0.25
        # Outer-lip MAR: closed ~0.25–0.45, yawn typically >= 0.55–0.70
        self.MOUTH_AR_THRESH = 0.55
        self.MOUTH_REL_OPEN_THRESH = 0.12  # mouth vertical / nose-chin
        self.MAX_FACES = 4

        try:
            eye_thresh_config = SystemConfig.objects.filter(config_key="eye_ar_thresh").first()
            if eye_thresh_config:
                self.EYE_AR_THRESH = float(eye_thresh_config.config_value)

            mouth_thresh_config = SystemConfig.objects.filter(
                config_key="mouth_ar_thresh"
            ).first()
            migrated = SystemConfig.objects.filter(
                config_key="mouth_ar_migrated_v2"
            ).first()
            if mouth_thresh_config:
                mar_th = float(mouth_thresh_config.config_value)
                # One-shot migration from old inner-lip / legacy defaults
                if migrated is None and (
                    abs(mar_th - 0.5) < 1e-6 or abs(mar_th - 0.6) < 1e-6
                ):
                    mar_th = 0.55
                    mouth_thresh_config.config_value = "0.55"
                    mouth_thresh_config.save(update_fields=["config_value"])
                    SystemConfig.objects.update_or_create(
                        config_key="mouth_ar_migrated_v2",
                        defaults={"config_value": "1"},
                    )
                elif migrated is None:
                    SystemConfig.objects.update_or_create(
                        config_key="mouth_ar_migrated_v2",
                        defaults={"config_value": "1"},
                    )
                self.MOUTH_AR_THRESH = mar_th
            elif migrated is None:
                SystemConfig.objects.update_or_create(
                    config_key="mouth_ar_migrated_v2",
                    defaults={"config_value": "1"},
                )

            rel_cfg = SystemConfig.objects.filter(config_key="mouth_rel_open_thresh").first()
            if rel_cfg:
                try:
                    rel_th = float(rel_cfg.config_value)
                    # Legacy/mis-set values (>=0.22) almost never fire for real yawns
                    # (typical yawn rel-open is ~0.10–0.20). Clamp + persist once.
                    migrated_rel = SystemConfig.objects.filter(
                        config_key="mouth_rel_open_migrated_v1"
                    ).first()
                    if migrated_rel is None and rel_th >= 0.22:
                        rel_th = 0.12
                        rel_cfg.config_value = "0.12"
                        rel_cfg.save(update_fields=["config_value"])
                        SystemConfig.objects.update_or_create(
                            config_key="mouth_rel_open_migrated_v1",
                            defaults={"config_value": "1"},
                        )
                    self.MOUTH_REL_OPEN_THRESH = max(0.05, min(0.22, float(rel_th)))
                except (TypeError, ValueError):
                    pass

            max_faces_cfg = SystemConfig.objects.filter(config_key="dlib_max_faces").first()
            if max_faces_cfg:
                self.MAX_FACES = max(1, min(8, int(float(max_faces_cfg.config_value))))

            print(
                f"已加载配置: EYE_AR_THRESH={self.EYE_AR_THRESH}, "
                f"MOUTH_AR_THRESH={self.MOUTH_AR_THRESH}, "
                f"MOUTH_REL={self.MOUTH_REL_OPEN_THRESH}, MAX_FACES={self.MAX_FACES}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"加载配置失败: {e}")

    @staticmethod
    def _expand_bbox(x1, y1, x2, y2, img_w, img_h, ratio=0.18):
        """Expand bbox for more stable landmark prediction (YOLO boxes are tight)."""
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * ratio)
        pad_y = int(bh * ratio)
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(img_w - 1, x2 + pad_x)
        ny2 = min(img_h - 1, y2 + pad_y)
        if nx2 <= nx1 or ny2 <= ny1:
            return x1, y1, x2, y2
        return nx1, ny1, nx2, ny2

    def _bbox_to_rect(self, face_bbox, image_shape):
        if face_bbox is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        h, w = image_shape[:2]
        x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, w, h, ratio=0.08)
        return dlib.rectangle(x1, y1, x2, y2)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with self._lock:
            faces = self.detector(gray, 0)
        return faces

    def get_landmarks(self, image, rect):
        if self.predictor is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with self._lock:
            shape = self.predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def eye_aspect_ratio(self, eye):
        a = dist.euclidean(eye[1], eye[5])
        b = dist.euclidean(eye[2], eye[4])
        c = dist.euclidean(eye[0], eye[3])
        return (a + b) / (2.0 * c)

    def mouth_aspect_ratio(self, landmarks):
        """
        Outer-lip MAR (stable on mono / noisy industrial video).

        Uses three vertical outer-lip spans over mouth width:
          (||p50-p58|| + ||p51-p57|| + ||p52-p56||) / (3 * ||p48-p54||)

        Typical ranges: closed/talking ~0.25–0.50; yawn usually >= 0.55–0.70.
        Inner-lip MAR was too noisy and caused false yawns.
        """
        v1 = dist.euclidean(landmarks[50], landmarks[58])
        v2 = dist.euclidean(landmarks[51], landmarks[57])
        v3 = dist.euclidean(landmarks[52], landmarks[56])
        horiz = dist.euclidean(landmarks[48], landmarks[54])
        if horiz < 1e-6:
            return 0.0
        return float((v1 + v2 + v3) / (3.0 * horiz))

    def mouth_relative_open(self, landmarks):
        """Vertical mouth opening normalized by nose-bridge → chin distance."""
        open_px = dist.euclidean(landmarks[51], landmarks[57])
        face_scale = dist.euclidean(landmarks[27], landmarks[8])
        if face_scale < 1e-6:
            # Fallback: jaw width
            face_scale = dist.euclidean(landmarks[3], landmarks[13])
        if face_scale < 1e-6:
            return 0.0
        return float(open_px / face_scale)

    def is_yawn(self, landmarks, mar=None):
        """
        Yawn when outer-lip MAR reaches the configured threshold.

        A tiny relative-open sanity floor rejects landmark glitches (MAR high
        but mouth not actually open). The configured mouth_rel_open_thresh is
        used as that soft floor (clamped), not a second hard AND gate.
        """
        if mar is None:
            mar = self.mouth_aspect_ratio(landmarks)
        mar_th = float(self.MOUTH_AR_THRESH)
        if mar < mar_th:
            return False
        rel = self.mouth_relative_open(landmarks)
        # Soft floor only — do not require the full configured rel as a second gate
        soft_floor = max(0.04, min(0.10, float(self.MOUTH_REL_OPEN_THRESH) * 0.5))
        return bool(rel >= soft_floor)

    def _metrics_from_landmarks(self, landmarks):
        left_eye = landmarks[self.LEFT_EYE_START : self.LEFT_EYE_END + 1]
        right_eye = landmarks[self.RIGHT_EYE_START : self.RIGHT_EYE_END + 1]
        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
        mar = self.mouth_aspect_ratio(landmarks)
        yawn = self.is_yawn(landmarks, mar=mar)
        if ear < self.EYE_AR_THRESH * 0.8:
            fatigue_level = 4
        elif ear < self.EYE_AR_THRESH:
            fatigue_level = 3 if yawn else 2
        else:
            fatigue_level = 1 if yawn else 0
        return {
            "eye_aspect_ratio": float(ear),
            "mouth_aspect_ratio": float(mar),
            "mouth_rel_open": float(self.mouth_relative_open(landmarks)),
            "yawn_detected": yawn,
            "fatigue_level": int(fatigue_level),
            "landmarks": landmarks,
        }

    def _analyze_landmarks(self, landmarks, results):
        results.update(self._metrics_from_landmarks(landmarks))

    @staticmethod
    def _bbox_area(bbox):
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            return 0.0
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _predict_one(self, gray, face_bbox, image_shape):
        """Run predictor for one bbox; returns metrics dict or None."""
        rect = self._bbox_to_rect(face_bbox, image_shape)
        if rect is None or self.predictor is None:
            return None
        shape = self.predictor(gray, rect)
        landmarks = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        metrics = self._metrics_from_landmarks(landmarks)
        metrics["bbox"] = [int(v) for v in face_bbox]
        return metrics

    def detect_fatigue_multi(
        self, image, face_bboxes=None, primary_bbox=None, allow_hog=True
    ):
        """
        Run 68-point + EAR/MAR for multiple YOLO face boxes.

        Returns aggregate fields for the primary face (largest / explicit) plus
        a per-face `faces` list for drawing / UI. Tracker callers should only
        feed the primary metrics into fatigue_tracker.

        allow_hog=False: never fall back to HOG when boxes are empty (MVS path).
        """
        results = {
            "faces_detected": 0,
            "eye_aspect_ratio": None,
            "mouth_aspect_ratio": None,
            "yawn_detected": False,
            "fatigue_level": 0,
            "landmarks": None,
            "faces": [],
            "primary_index": None,
        }

        boxes = []
        if face_bboxes is not None:
            for b in face_bboxes:
                if b is None:
                    continue
                try:
                    boxes.append([int(v) for v in b])
                except (TypeError, ValueError):
                    continue

        # Prefer largest faces when over the cap
        if len(boxes) > self.MAX_FACES:
            boxes = sorted(boxes, key=self._bbox_area, reverse=True)[: self.MAX_FACES]

        try:
            with self._lock:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_entries = []

                if boxes and self.predictor is not None:
                    for bbox in boxes:
                        one = self._predict_one(gray, bbox, image.shape)
                        if one is not None:
                            face_entries.append(one)
                elif allow_hog and self.predictor is not None:
                    hog_faces = list(self.detector(gray, 0))[: self.MAX_FACES]
                    for rect in hog_faces:
                        shape = self.predictor(gray, rect)
                        landmarks = np.zeros((68, 2), dtype=int)
                        for i in range(0, 68):
                            landmarks[i] = (shape.part(i).x, shape.part(i).y)
                        metrics = self._metrics_from_landmarks(landmarks)
                        metrics["bbox"] = [
                            int(rect.left()),
                            int(rect.top()),
                            int(rect.right()),
                            int(rect.bottom()),
                        ]
                        face_entries.append(metrics)

                results["faces_detected"] = len(face_entries)
                if not face_entries:
                    return results

                primary_idx = 0
                if primary_bbox is not None:
                    try:
                        pb = [int(v) for v in primary_bbox]
                    except (TypeError, ValueError):
                        pb = None
                    if pb is not None:
                        best_i, best_iou = 0, -1.0
                        for i, ent in enumerate(face_entries):
                            iou = self._bbox_iou(ent["bbox"], pb)
                            if iou > best_iou:
                                best_iou, best_i = iou, i
                        primary_idx = best_i
                else:
                    primary_idx = max(
                        range(len(face_entries)),
                        key=lambda i: self._bbox_area(face_entries[i]["bbox"]),
                    )

                for i, ent in enumerate(face_entries):
                    ent["is_primary"] = i == primary_idx
                    # JSON-friendly landmarks for meta/UI
                    ent["landmarks_list"] = [
                        [int(x), int(y)] for x, y in ent["landmarks"]
                    ]

                results["faces"] = face_entries
                results["primary_index"] = int(primary_idx)
                primary = face_entries[primary_idx]
                results["eye_aspect_ratio"] = primary["eye_aspect_ratio"]
                results["mouth_aspect_ratio"] = primary["mouth_aspect_ratio"]
                results["mouth_rel_open"] = primary.get("mouth_rel_open")
                results["yawn_detected"] = primary["yawn_detected"]
                results["fatigue_level"] = primary["fatigue_level"]
                results["landmarks"] = primary["landmarks"]
        except Exception as e:  # noqa: BLE001
            print(f"多人疲劳检测异常: {e}")
            results["eye_aspect_ratio"] = None
            results["yawn_detected"] = False
            results["fatigue_level"] = 0

        return results

    @staticmethod
    def _bbox_iou(a, b):
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

    def detect_fatigue(self, image, face_bbox=None):
        """
        Prefer external YOLO face_bbox for 68-point prediction when available.
        HOG is only used when no valid external box is provided.
        """
        if face_bbox is not None:
            multi = self.detect_fatigue_multi(
                image, face_bboxes=[face_bbox], primary_bbox=face_bbox
            )
            return {
                "faces_detected": multi.get("faces_detected") or 0,
                "eye_aspect_ratio": multi.get("eye_aspect_ratio"),
                "mouth_aspect_ratio": multi.get("mouth_aspect_ratio"),
                "yawn_detected": bool(multi.get("yawn_detected")),
                "fatigue_level": int(multi.get("fatigue_level") or 0),
                "landmarks": multi.get("landmarks"),
                "faces": multi.get("faces") or [],
            }

        results = {
            "faces_detected": 0,
            "eye_aspect_ratio": None,
            "mouth_aspect_ratio": None,
            "yawn_detected": False,
            "fatigue_level": 0,
            "landmarks": None,
            "faces": [],
        }

        try:
            with self._lock:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = list(self.detector(gray, 0))
                results["faces_detected"] = len(faces)

                if len(faces) > 0 and self.predictor is not None:
                    shape = self.predictor(gray, faces[0])
                    landmarks = np.zeros((68, 2), dtype=int)
                    for i in range(0, 68):
                        landmarks[i] = (shape.part(i).x, shape.part(i).y)
                    self._analyze_landmarks(landmarks, results)
                    results["faces"] = [
                        {
                            **self._metrics_from_landmarks(landmarks),
                            "bbox": [
                                int(faces[0].left()),
                                int(faces[0].top()),
                                int(faces[0].right()),
                                int(faces[0].bottom()),
                            ],
                            "is_primary": True,
                        }
                    ]
                elif hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
                    cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
                    eye_cascade = cv2.CascadeClassifier(cascade_path)
                    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(eyes) > 0:
                        results["eye_aspect_ratio"] = 0.3
                        results["fatigue_level"] = 0
        except Exception as e:  # noqa: BLE001
            print(f"疲劳检测异常: {e}")
            results["eye_aspect_ratio"] = None
            results["yawn_detected"] = False
            results["fatigue_level"] = 0

        return results

    @staticmethod
    def scale_landmarks(landmarks, src_wh, dst_wh):
        """Remap 68-point landmarks from src (w,h) to dst (w,h)."""
        if landmarks is None or src_wh is None or dst_wh is None:
            return landmarks
        try:
            sw, sh = int(src_wh[0]), int(src_wh[1])
            dw, dh = int(dst_wh[0]), int(dst_wh[1])
        except (TypeError, ValueError, IndexError):
            return landmarks
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            return landmarks
        if sw == dw and sh == dh:
            return landmarks
        pts = np.asarray(landmarks, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return landmarks
        out = pts.copy()
        out[:, 0] = out[:, 0] * (dw / float(sw))
        out[:, 1] = out[:, 1] * (dh / float(sh))
        return np.round(out).astype(int)

    def draw_landmarks(self, image, landmarks, landmarks_size=None, color=(0, 255, 0)):
        if landmarks is None:
            return image

        # Allow drawing onto a shared buffer without forced copy
        img = image
        h, w = img.shape[:2]
        pts = self.scale_landmarks(landmarks, landmarks_size, (w, h))
        pts = np.asarray(pts, dtype=int)
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 1, color, -1)

        left_eye = pts[self.LEFT_EYE_START : self.LEFT_EYE_END + 1]
        right_eye = pts[self.RIGHT_EYE_START : self.RIGHT_EYE_END + 1]
        cv2.drawContours(img, [cv2.convexHull(left_eye)], -1, color, 1)
        cv2.drawContours(img, [cv2.convexHull(right_eye)], -1, color, 1)
        mouth = pts[self.MOUTH_START : self.MOUTH_END + 1]
        cv2.drawContours(img, [cv2.convexHull(mouth)], -1, color, 1)
        return img

    def draw_fatigue_results(self, image, results):
        img = image.copy()
        lm_size = results.get("landmarks_size")

        faces = results.get("faces") or []
        if faces:
            for ent in faces:
                lm = ent.get("landmarks")
                if lm is None:
                    continue
                # Primary: bright green; others: cyan
                color = (0, 255, 0) if ent.get("is_primary") else (255, 255, 0)
                self.draw_landmarks(img, lm, landmarks_size=lm_size, color=color)
                ear = ent.get("eye_aspect_ratio")
                bbox = ent.get("bbox")
                if ear is not None and bbox is not None:
                    try:
                        x1, y1, _, _ = [int(v) for v in bbox]
                        if lm_size is not None:
                            scaled = self.scale_landmarks(
                                [[x1, y1]], lm_size, (img.shape[1], img.shape[0])
                            )
                            x1, y1 = int(scaled[0][0]), int(scaled[0][1])
                        tag = "P" if ent.get("is_primary") else ""
                        cv2.putText(
                            img,
                            f"{tag}EAR:{float(ear):.2f}",
                            (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                        )
                    except (TypeError, ValueError, IndexError):
                        pass
        elif results.get("landmarks") is not None:
            self.draw_landmarks(
                img,
                results["landmarks"],
                landmarks_size=lm_size,
            )

        if results.get("eye_aspect_ratio") is not None:
            ear = results["eye_aspect_ratio"]
            cv2.putText(
                img,
                f"EAR: {ear:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        perclos = results.get("perclos")
        if perclos is not None:
            cv2.putText(
                img,
                f"PERCLOS: {float(perclos):.1f}%",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        closed_ms = results.get("eye_closed_ms")
        if closed_ms is not None:
            cv2.putText(
                img,
                f"CLOSED: {int(closed_ms)} ms",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        yawn_text = "YAWN: Yes" if results.get("yawn_detected") else "YAWN: No"
        cv2.putText(img, yawn_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mar = results.get("mouth_aspect_ratio")
        if mar is not None:
            cv2.putText(
                img,
                f"MAR: {float(mar):.2f}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        fatigue_level = int(results.get("fatigue_level") or 0)
        if fatigue_level >= 3:
            color = (0, 0, 255)
        elif fatigue_level >= 1:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(
            img,
            f"FATIGUE LEVEL: {fatigue_level}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return img
