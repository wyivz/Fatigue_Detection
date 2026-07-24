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


def invalidate() -> None:
    global _loaded_at
    with _lock:
        _loaded_at = 0.0


def get_configs(force: bool = False) -> Dict[str, Any]:
    """Return a shallow copy of SystemConfig key→value map."""
    global _cache, _loaded_at
    now = time.time()
    with _lock:
        if (not force) and _cache and (now - _loaded_at) < _TTL_SEC:
            return dict(_cache)
    try:
        from detection.models import SystemConfig

        data = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
    except Exception:  # noqa: BLE001
        data = {}
    with _lock:
        _cache = data
        _loaded_at = time.time()
        return dict(_cache)


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    return get_configs().get(key, default)
