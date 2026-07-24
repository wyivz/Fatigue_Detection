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
        self.MOUTH_AR_THRESH = 0.5

        try:
            eye_thresh_config = SystemConfig.objects.filter(config_key="eye_ar_thresh").first()
            if eye_thresh_config:
                self.EYE_AR_THRESH = float(eye_thresh_config.config_value)

            mouth_thresh_config = SystemConfig.objects.filter(
                config_key="mouth_ar_thresh"
            ).first()
            if mouth_thresh_config:
                mar_th = float(mouth_thresh_config.config_value)
                # 旧版外唇 MAR 默认 0.6，换内唇算法后过高，自动落到 0.5
                if abs(mar_th - 0.6) < 1e-6:
                    mar_th = 0.5
                    mouth_thresh_config.config_value = "0.5"
                    mouth_thresh_config.save(update_fields=["config_value"])
                self.MOUTH_AR_THRESH = mar_th

            print(
                f"已加载配置: EYE_AR_THRESH={self.EYE_AR_THRESH}, "
                f"MOUTH_AR_THRESH={self.MOUTH_AR_THRESH}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"加载配置失败: {e}")

    @staticmethod
    def _expand_bbox(x1, y1, x2, y2, img_w, img_h, ratio=0.08):
        """Expand bbox slightly for more stable landmark prediction."""
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
        Inner-lip MAR (more sensitive to yawns than outer-lip ratio).
        Closed mouth ~0.05–0.30; yawn typically >= 0.45–0.50.
        """
        a = dist.euclidean(landmarks[61], landmarks[67])
        b = dist.euclidean(landmarks[62], landmarks[66])
        c = dist.euclidean(landmarks[63], landmarks[65])
        d = dist.euclidean(landmarks[60], landmarks[64])
        if d < 1e-6:
            return 0.0
        return (a + b + c) / (3.0 * d)

    def _analyze_landmarks(self, landmarks, results):
        left_eye = landmarks[self.LEFT_EYE_START : self.LEFT_EYE_END + 1]
        right_eye = landmarks[self.RIGHT_EYE_START : self.RIGHT_EYE_END + 1]
        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
        results["eye_aspect_ratio"] = float(ear)

        mar = self.mouth_aspect_ratio(landmarks)
        results["mouth_aspect_ratio"] = float(mar)
        results["yawn_detected"] = bool(mar > self.MOUTH_AR_THRESH)

        if ear < self.EYE_AR_THRESH * 0.8:
            fatigue_level = 4
        elif ear < self.EYE_AR_THRESH:
            fatigue_level = 3 if results["yawn_detected"] else 2
        else:
            fatigue_level = 1 if results["yawn_detected"] else 0
        results["fatigue_level"] = int(fatigue_level)
        results["landmarks"] = landmarks

    def detect_fatigue(self, image, face_bbox=None):
        """
        Prefer external YOLO face_bbox for 68-point prediction when available.
        HOG is only used when no valid external box is provided.
        """
        results = {
            "faces_detected": 0,
            "eye_aspect_ratio": None,
            "yawn_detected": False,
            "fatigue_level": 0,
            "landmarks": None,
        }

        try:
            with self._lock:
                faces = []
                ext_rect = self._bbox_to_rect(face_bbox, image.shape)
                if ext_rect is not None and self.predictor is not None:
                    faces = [ext_rect]
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = list(self.detector(gray, 0))

                results["faces_detected"] = len(faces)

                if len(faces) > 0 and self.predictor is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    shape = self.predictor(gray, faces[0])
                    landmarks = np.zeros((68, 2), dtype=int)
                    for i in range(0, 68):
                        landmarks[i] = (shape.part(i).x, shape.part(i).y)
                    self._analyze_landmarks(landmarks, results)
                elif hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    def draw_landmarks(self, image, landmarks):
        if landmarks is None:
            return image

        img = image.copy()
        for x, y in landmarks:
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

        left_eye = landmarks[self.LEFT_EYE_START : self.LEFT_EYE_END + 1]
        right_eye = landmarks[self.RIGHT_EYE_START : self.RIGHT_EYE_END + 1]
        cv2.drawContours(img, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
        mouth = landmarks[self.MOUTH_START : self.MOUTH_END + 1]
        cv2.drawContours(img, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)
        return img

    def draw_fatigue_results(self, image, results):
        img = image.copy()

        if results.get("landmarks") is not None:
            img = self.draw_landmarks(img, results["landmarks"])

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
