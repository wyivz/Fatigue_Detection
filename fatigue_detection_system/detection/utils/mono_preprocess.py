# -*- coding: utf-8 -*-
"""Mono / grayscale camera preprocessing for YOLO behavior detection."""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _cfg_bool(configs: Dict[str, Any], key: str, default: bool) -> bool:
    raw = configs.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


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


def load_mono_config(configs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load mono-camera enhancement settings from SystemConfig or a dict."""
    if configs is None:
        try:
            from detection.models import SystemConfig

            configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
        except Exception:  # noqa: BLE001
            configs = {}

    return {
        "enabled": _cfg_bool(configs, "mono_camera_mode", False),
        "clahe_clip": _cfg_float(configs, "mono_clahe_clip", 2.0),
        "clahe_tile": max(2, _cfg_int(configs, "mono_clahe_tile", 8)),
    }


def enhance_for_mono(image: np.ndarray, configs: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply CLAHE on luminance for industrial mono cameras.

    Returns a BGR image suitable for YOLO. When mono mode is off, returns the
    original image unchanged (no copy). Drawing should always use the original.
    """
    cfg = load_mono_config(configs)
    if not cfg["enabled"] or image is None:
        return image

    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tile = int(cfg["clahe_tile"])
    clahe = cv2.createCLAHE(clipLimit=float(cfg["clahe_clip"]), tileGridSize=(tile, tile))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
