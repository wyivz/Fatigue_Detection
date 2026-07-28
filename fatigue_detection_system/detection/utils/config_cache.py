# -*- coding: utf-8 -*-
"""Process-level SystemConfig cache to avoid DB hits every detect frame."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

_lock = threading.RLock()
_cache: Dict[str, Any] = {}
_loaded_at = 0.0
_TTL_SEC = 2.0
_migrated_v2 = False
_migrated_v3 = False


def invalidate() -> None:
    global _loaded_at
    with _lock:
        _loaded_at = 0.0


def _ensure_archive_scale_defaults() -> None:
    """One-shot: undo max_width=0 presets that made 5MP HOG slow & inaccurate."""
    global _migrated_v2
    if _migrated_v2:
        return
    try:
        from detection.models import SystemConfig

        if SystemConfig.objects.filter(config_key="archive_detect_scale_v2").exists():
            _migrated_v2 = True
            return
        SystemConfig.objects.update_or_create(
            config_key="yolo_detect_max_width",
            defaults={"config_value": "960"},
        )
        SystemConfig.objects.update_or_create(
            config_key="detection_interval",
            defaults={"config_value": "1000"},
        )
        SystemConfig.objects.update_or_create(
            config_key="dlib_landmark_mode",
            defaults={"config_value": "hog"},
        )
        SystemConfig.objects.update_or_create(
            config_key="dlib_refine_mode",
            defaults={"config_value": "off"},
        )
        SystemConfig.objects.update_or_create(
            config_key="fatigue_level_mode",
            defaults={"config_value": "instant"},
        )
        SystemConfig.objects.update_or_create(
            config_key="archive_detect_scale_v2",
            defaults={"config_value": "1"},
        )
        _migrated_v2 = True
    except Exception:  # noqa: BLE001
        pass


def _ensure_dlib_yolo_tie_defaults() -> None:
    """One-shot: v2 set dlib_landmark_mode=hog/dlib_refine_mode=off (pure archive
    HOG-first, ignoring YOLO's sticky primary face — the root cause of landmarks
    not locking onto the tracked face). Flip to YOLO-tied defaults so the fix in
    run_archive_detect actually takes effect for already-migrated databases."""
    global _migrated_v3
    if _migrated_v3:
        return
    try:
        from detection.models import SystemConfig

        if SystemConfig.objects.filter(config_key="dlib_yolo_tie_v3").exists():
            _migrated_v3 = True
            return
        SystemConfig.objects.update_or_create(
            config_key="dlib_landmark_mode",
            defaults={"config_value": "yolo"},
        )
        SystemConfig.objects.update_or_create(
            config_key="dlib_refine_mode",
            defaults={"config_value": "light"},
        )
        SystemConfig.objects.update_or_create(
            config_key="dlib_yolo_tie_v3",
            defaults={"config_value": "1"},
        )
        _migrated_v3 = True
    except Exception:  # noqa: BLE001
        pass


def get_configs(force: bool = False) -> Dict[str, Any]:
    """Return a shallow copy of SystemConfig key→value map."""
    global _cache, _loaded_at
    now = time.time()
    with _lock:
        if (not force) and _cache and (now - _loaded_at) < _TTL_SEC:
            return dict(_cache)
    try:
        from detection.models import SystemConfig

        _ensure_archive_scale_defaults()
        _ensure_dlib_yolo_tie_defaults()
        data = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
    except Exception:  # noqa: BLE001
        data = {}
    with _lock:
        _cache = data
        _loaded_at = time.time()
        return dict(_cache)


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    return get_configs().get(key, default)
