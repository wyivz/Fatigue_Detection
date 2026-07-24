# -*- coding: utf-8 -*-
"""Session-scoped temporal fatigue (EAR / PERCLOS) and behavior confirm trackers."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple


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


def load_fatigue_config(configs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if configs is None:
        try:
            from detection.models import SystemConfig

            configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
        except Exception:  # noqa: BLE001
            configs = {}

    return {
        "eye_ar_thresh": _cfg_float(configs, "eye_ar_thresh", 0.25),
        "mouth_ar_thresh": _cfg_float(configs, "mouth_ar_thresh", 0.6),
        "blink_max_ms": _cfg_int(configs, "blink_max_ms", 250),
        "microsleep_min_ms": _cfg_int(configs, "microsleep_min_ms", 500),
        "perclos_window_sec": max(5, _cfg_int(configs, "perclos_window_sec", 60)),
        "perclos_alert_pct": _cfg_float(configs, "perclos_alert_pct", 20.0),
        "ear_sample_interval_ms": max(50, _cfg_int(configs, "ear_sample_interval_ms", 100)),
        "behavior_confirm_frames": max(1, _cfg_int(configs, "behavior_confirm_frames", 2)),
    }


@dataclass
class FatigueSnapshot:
    eye_aspect_ratio: Optional[float] = None
    yawn_detected: bool = False
    fatigue_level: int = 0
    perclos: float = 0.0
    eye_closed_ms: int = 0
    is_microsleep: bool = False
    is_blink: bool = False
    faces_detected: int = 0
    landmarks: Any = None


@dataclass
class _SessionFatigueState:
    samples: Deque[Tuple[float, bool]] = field(default_factory=deque)  # (t, eyes_closed)
    closed_start: Optional[float] = None
    last_t: Optional[float] = None
    recent_microsleep_until: float = 0.0
    last_ear: Optional[float] = None
    last_yawn: bool = False
    last_landmarks: Any = None
    last_faces: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


class FatigueTemporalTracker:
    """Thread-safe per-session EAR / PERCLOS fatigue state machine."""

    _MICROSLEEP_HOLD_SEC = 2.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[int, _SessionFatigueState] = {}

    def reset(self, session_id: int) -> None:
        with self._lock:
            self._sessions.pop(int(session_id), None)

    def configure(self, session_id: int, configs: Optional[Dict[str, Any]] = None) -> None:
        cfg = load_fatigue_config(configs)
        with self._lock:
            state = self._sessions.get(int(session_id))
            if state is None:
                state = _SessionFatigueState(config=cfg)
                self._sessions[int(session_id)] = state
            else:
                state.config = cfg

    def _get_state(self, session_id: int) -> _SessionFatigueState:
        sid = int(session_id)
        state = self._sessions.get(sid)
        if state is None:
            state = _SessionFatigueState(config=load_fatigue_config())
            self._sessions[sid] = state
        return state

    def update(
        self,
        session_id: int,
        ear: Optional[float],
        yawn_detected: Optional[bool] = None,
        faces_detected: int = 0,
        landmarks: Any = None,
        timestamp: Optional[float] = None,
    ) -> FatigueSnapshot:
        """Feed one EAR sample. Skip PERCLOS denominator when no face / no EAR.

        yawn_detected=None keeps the previous yawn flag (used by high-rate EAR loop).
        """
        t = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            state = self._get_state(session_id)
            cfg = state.config or load_fatigue_config()
            if yawn_detected is not None:
                state.last_yawn = bool(yawn_detected)
            state.last_faces = int(faces_detected)
            if landmarks is not None:
                state.last_landmarks = landmarks

            if ear is None or faces_detected <= 0:
                # End any open closed-eye segment without counting gap as closed.
                if state.closed_start is not None and state.last_t is not None:
                    self._finalize_closed_segment(state, state.last_t, cfg)
                state.closed_start = None
                state.last_t = t
                state.last_ear = ear
                self._prune(state, t, cfg)
                return self._snapshot(state, cfg, t)

            ear_f = float(ear)
            state.last_ear = ear_f
            thresh = float(cfg["eye_ar_thresh"])
            closed = ear_f < thresh

            if state.last_t is not None and t > state.last_t:
                # Attribute the inter-sample interval to the *previous* eye state
                # already reflected by closed_start / open.
                pass

            if closed:
                if state.closed_start is None:
                    state.closed_start = t
                state.samples.append((t, True))
            else:
                if state.closed_start is not None:
                    self._finalize_closed_segment(state, t, cfg)
                state.closed_start = None
                state.samples.append((t, False))

            state.last_t = t
            self._prune(state, t, cfg)
            return self._snapshot(state, cfg, t)

    def get_snapshot(self, session_id: int) -> FatigueSnapshot:
        with self._lock:
            state = self._get_state(session_id)
            cfg = state.config or load_fatigue_config()
            return self._snapshot(state, cfg, time.time())

    def _finalize_closed_segment(
        self, state: _SessionFatigueState, end_t: float, cfg: Dict[str, Any]
    ) -> None:
        if state.closed_start is None:
            return
        duration_ms = max(0.0, (end_t - state.closed_start) * 1000.0)
        microsleep_min = float(cfg["microsleep_min_ms"])
        if duration_ms >= microsleep_min:
            state.recent_microsleep_until = end_t + self._MICROSLEEP_HOLD_SEC

    def _prune(self, state: _SessionFatigueState, now: float, cfg: Dict[str, Any]) -> None:
        window = float(cfg["perclos_window_sec"])
        cutoff = now - window
        while state.samples and state.samples[0][0] < cutoff:
            state.samples.popleft()

    def _current_closed_ms(self, state: _SessionFatigueState, now: float) -> int:
        if state.closed_start is None:
            return 0
        return int(max(0.0, (now - state.closed_start) * 1000.0))

    def _compute_perclos(self, state: _SessionFatigueState, now: float, cfg: Dict[str, Any]) -> float:
        """PERCLOS = closed time in window / full window duration (face-valid samples only for numerator)."""
        window = float(cfg["perclos_window_sec"])
        if window <= 1e-6:
            return 0.0

        samples = list(state.samples)
        closed_sec = 0.0

        if len(samples) >= 2:
            for i in range(1, len(samples)):
                t0, closed0 = samples[i - 1]
                t1, _ = samples[i]
                dt = max(0.0, t1 - t0)
                if closed0:
                    closed_sec += dt
            last_t, last_closed = samples[-1]
            if now > last_t and (last_closed or state.closed_start is not None):
                closed_sec += now - last_t
        elif state.closed_start is not None:
            closed_sec = max(0.0, now - state.closed_start)

        closed_sec = min(closed_sec, window)
        return max(0.0, min(100.0, (closed_sec / window) * 100.0))

    def _snapshot(
        self, state: _SessionFatigueState, cfg: Dict[str, Any], now: float
    ) -> FatigueSnapshot:
        closed_ms = self._current_closed_ms(state, now)
        blink_max = int(cfg["blink_max_ms"])
        microsleep_min = int(cfg["microsleep_min_ms"])
        perclos = self._compute_perclos(state, now, cfg)
        alert_pct = float(cfg["perclos_alert_pct"])
        warn_pct = alert_pct * 0.75

        is_microsleep = closed_ms >= microsleep_min or now < state.recent_microsleep_until
        is_blink = (
            state.closed_start is not None
            and 0 < closed_ms <= blink_max
            and not is_microsleep
        )

        yawn = bool(state.last_yawn)
        level = 0
        if yawn or (warn_pct <= perclos < alert_pct):
            level = 1
        if is_microsleep:
            level = max(level, 2)
        if perclos >= alert_pct:
            level = max(level, 3)
        if perclos >= alert_pct and (is_microsleep or perclos >= alert_pct * 1.5):
            level = 4

        return FatigueSnapshot(
            eye_aspect_ratio=state.last_ear,
            yawn_detected=yawn,
            fatigue_level=int(level),
            perclos=float(perclos),
            eye_closed_ms=int(closed_ms),
            is_microsleep=bool(is_microsleep),
            is_blink=bool(is_blink),
            faces_detected=int(state.last_faces),
            landmarks=state.last_landmarks,
        )


@dataclass
class _BehaviorState:
    smoking: int = 0
    phone: int = 0
    drinking: int = 0
    confirm_frames: int = 2


class BehaviorConfirmTracker:
    """Require N consecutive detections before confirming smoking/phone/drinking."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[int, _BehaviorState] = {}

    def reset(self, session_id: int) -> None:
        with self._lock:
            self._sessions.pop(int(session_id), None)

    def configure(self, session_id: int, configs: Optional[Dict[str, Any]] = None) -> None:
        cfg = load_fatigue_config(configs)
        with self._lock:
            state = self._sessions.get(int(session_id))
            if state is None:
                state = _BehaviorState(confirm_frames=cfg["behavior_confirm_frames"])
                self._sessions[int(session_id)] = state
            else:
                state.confirm_frames = cfg["behavior_confirm_frames"]

    def update(
        self,
        session_id: int,
        smoking: bool,
        phone: bool,
        drinking: bool,
        configs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        cfg = load_fatigue_config(configs)
        need = max(1, int(cfg["behavior_confirm_frames"]))
        with self._lock:
            sid = int(session_id)
            state = self._sessions.get(sid)
            if state is None:
                state = _BehaviorState(confirm_frames=need)
                self._sessions[sid] = state
            else:
                state.confirm_frames = need

            state.smoking = state.smoking + 1 if smoking else 0
            state.phone = state.phone + 1 if phone else 0
            state.drinking = state.drinking + 1 if drinking else 0

            return {
                "smoking_detected": state.smoking >= need,
                "phone_detected": state.phone >= need,
                "drinking_detected": state.drinking >= need,
            }


fatigue_tracker = FatigueTemporalTracker()
behavior_tracker = BehaviorConfirmTracker()
