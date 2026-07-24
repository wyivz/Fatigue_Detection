# -*- coding: utf-8 -*-
"""Session-scoped temporal fatigue (EAR / PERCLOS) and behavior confirm trackers."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


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
            from detection.utils.config_cache import get_configs

            configs = get_configs()
        except Exception:  # noqa: BLE001
            configs = {}

    return {
        "eye_ar_thresh": _cfg_float(configs, "eye_ar_thresh", 0.25),
        "mouth_ar_thresh": _cfg_float(configs, "mouth_ar_thresh", 0.55),
        "blink_max_ms": _cfg_int(configs, "blink_max_ms", 250),
        "microsleep_min_ms": _cfg_int(configs, "microsleep_min_ms", 500),
        "perclos_window_sec": max(5, _cfg_int(configs, "perclos_window_sec", 60)),
        "perclos_alert_pct": _cfg_float(configs, "perclos_alert_pct", 20.0),
        "ear_sample_interval_ms": max(50, _cfg_int(configs, "ear_sample_interval_ms", 100)),
        # Hits required inside sliding window (M of N)
        "behavior_confirm_frames": max(1, _cfg_int(configs, "behavior_confirm_frames", 2)),
        "behavior_window_frames": max(1, _cfg_int(configs, "behavior_window_frames", 5)),
        # Hysteresis band around EAR threshold to avoid flicker false closes
        "ear_hysteresis": _cfg_float(configs, "ear_hysteresis", 0.03),
        # Yawn must persist this many ms before latching (filters talking)
        "yawn_confirm_ms": _cfg_int(configs, "yawn_confirm_ms", 500),
        # How long a yawn latch stays after mouth closes
        "yawn_hold_ms": _cfg_int(configs, "yawn_hold_ms", 1200),
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
    # (w, h) of the image landmarks were computed on — needed to remap when drawing
    landmarks_size: Optional[Tuple[int, int]] = None
    # Per-face metrics/landmarks for multi-person drawing (alert still uses primary fields)
    faces: Optional[List[Dict[str, Any]]] = None


@dataclass
class _ClosedSegment:
    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.end - self.start) * 1000.0)


@dataclass
class _SessionFatigueState:
    # Closed segments longer than blink (for PERCLOS); short blinks excluded
    closed_segments: Deque[_ClosedSegment] = field(default_factory=deque)
    eyes_closed: bool = False  # hysteretic state
    closed_start: Optional[float] = None
    last_t: Optional[float] = None
    recent_microsleep_until: float = 0.0
    last_ear: Optional[float] = None
    last_yawn: bool = False
    yawn_candidate_start: Optional[float] = None
    last_raw_yawn_t: Optional[float] = None
    yawn_until: float = 0.0
    last_landmarks: Any = None
    last_landmarks_size: Optional[Tuple[int, int]] = None
    last_face_entries: Optional[List[Dict[str, Any]]] = None
    last_faces: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


class FatigueTemporalTracker:
    """Thread-safe per-session EAR / PERCLOS fatigue state machine."""

    _MICROSLEEP_HOLD_SEC = 1.5

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[int, _SessionFatigueState] = {}

    def reset(self, session_id: int) -> None:
        with self._lock:
            self._sessions.pop(int(session_id), None)

    def clear_history(self, session_id: int, configs: Optional[Dict[str, Any]] = None) -> FatigueSnapshot:
        """Clear PERCLOS / microsleep / yawn accumulators after user acknowledges alert."""
        cfg = load_fatigue_config(configs)
        with self._lock:
            sid = int(session_id)
            prev = self._sessions.get(sid)
            keep_cfg = (prev.config if prev and prev.config else None) or cfg
            self._sessions[sid] = _SessionFatigueState(config=keep_cfg)
            return self._snapshot(self._sessions[sid], keep_cfg, time.time())

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
        landmarks_size: Optional[Tuple[int, int]] = None,
        faces: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None,
    ) -> FatigueSnapshot:
        """Feed one EAR sample.

        - EAR uses hysteresis to avoid threshold flicker.
        - Normal blinks (<= blink_max_ms) do NOT count toward PERCLOS.
        - yawn_detected=None keeps prior yawn latch (EAR loop); True/False updates yawn FSM.
        - landmarks_size=(w,h) should match the image used for landmark prediction.
        - faces: optional multi-face list for drawing; alert metrics still use primary EAR.
        """
        t = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            state = self._get_state(session_id)
            cfg = state.config or load_fatigue_config()
            state.last_faces = int(faces_detected)
            if landmarks is not None:
                state.last_landmarks = landmarks
                if landmarks_size is not None:
                    try:
                        state.last_landmarks_size = (
                            int(landmarks_size[0]),
                            int(landmarks_size[1]),
                        )
                    except (TypeError, ValueError, IndexError):
                        state.last_landmarks_size = None
            if faces is not None:
                state.last_face_entries = list(faces)

            if yawn_detected is not None:
                self._update_yawn(state, bool(yawn_detected), t, cfg)

            if ear is None or faces_detected <= 0:
                # Lost face: end closed segment without treating gap as closed
                if state.eyes_closed and state.closed_start is not None:
                    self._close_segment(state, state.last_t or t, cfg)
                state.eyes_closed = False
                state.closed_start = None
                state.last_t = t
                state.last_ear = ear
                self._prune(state, t, cfg)
                return self._snapshot(state, cfg, t)

            ear_f = float(ear)
            state.last_ear = ear_f
            thresh = float(cfg["eye_ar_thresh"])
            hyst = max(0.0, float(cfg.get("ear_hysteresis", 0.03)))
            close_thresh = thresh
            open_thresh = thresh + hyst

            # Hysteresis: harder to enter closed, easier to stay open until clearly open
            if state.eyes_closed:
                want_closed = ear_f < open_thresh
            else:
                want_closed = ear_f < close_thresh

            if want_closed:
                if not state.eyes_closed:
                    state.eyes_closed = True
                    state.closed_start = t
            else:
                if state.eyes_closed and state.closed_start is not None:
                    self._close_segment(state, t, cfg)
                state.eyes_closed = False
                state.closed_start = None

            state.last_t = t
            self._prune(state, t, cfg)
            return self._snapshot(state, cfg, t)

    def get_snapshot(self, session_id: int) -> FatigueSnapshot:
        with self._lock:
            state = self._get_state(session_id)
            cfg = state.config or load_fatigue_config()
            return self._snapshot(state, cfg, time.time())

    def _update_yawn(
        self, state: _SessionFatigueState, raw_yawn: bool, t: float, cfg: Dict[str, Any]
    ) -> None:
        confirm_ms = float(cfg.get("yawn_confirm_ms", 500))
        hold_ms = float(cfg.get("yawn_hold_ms", 1200))
        gap_ms = 150.0
        if raw_yawn:
            state.last_raw_yawn_t = t
            if state.yawn_candidate_start is None:
                state.yawn_candidate_start = t
            elif (t - state.yawn_candidate_start) * 1000.0 >= confirm_ms:
                state.last_yawn = True
                state.yawn_until = t + hold_ms / 1000.0
        else:
            # Clear candidate only after a real gap since last positive raw sample
            if state.yawn_candidate_start is not None:
                last_pos = state.last_raw_yawn_t
                if last_pos is None or (t - last_pos) * 1000.0 > gap_ms:
                    state.yawn_candidate_start = None
            if t >= state.yawn_until:
                state.last_yawn = False

        if state.last_yawn and t >= state.yawn_until and not raw_yawn:
            state.last_yawn = False

    def _close_segment(
        self, state: _SessionFatigueState, end_t: float, cfg: Dict[str, Any]
    ) -> None:
        if state.closed_start is None:
            return
        start = state.closed_start
        duration_ms = max(0.0, (end_t - start) * 1000.0)
        blink_max = float(cfg["blink_max_ms"])
        microsleep_min = float(cfg["microsleep_min_ms"])

        # Only non-blink closures contribute to PERCLOS; store the portion
        # beyond blink_max so ongoing/closed math stay consistent.
        if duration_ms > blink_max:
            seg_start = start + blink_max / 1000.0
            if end_t > seg_start:
                state.closed_segments.append(_ClosedSegment(start=seg_start, end=end_t))

        if duration_ms >= microsleep_min:
            state.recent_microsleep_until = end_t + self._MICROSLEEP_HOLD_SEC

        state.closed_start = None

    def _prune(self, state: _SessionFatigueState, now: float, cfg: Dict[str, Any]) -> None:
        window = float(cfg["perclos_window_sec"])
        cutoff = now - window
        while state.closed_segments and state.closed_segments[0].end < cutoff:
            state.closed_segments.popleft()
        # Trim partial overlap at window start
        if state.closed_segments and state.closed_segments[0].start < cutoff:
            seg = state.closed_segments[0]
            state.closed_segments[0] = _ClosedSegment(start=cutoff, end=seg.end)

    def _current_closed_ms(self, state: _SessionFatigueState, now: float) -> int:
        if not state.eyes_closed or state.closed_start is None:
            return 0
        return int(max(0.0, (now - state.closed_start) * 1000.0))

    def _compute_perclos(self, state: _SessionFatigueState, now: float, cfg: Dict[str, Any]) -> float:
        """PERCLOS from non-blink closed segments / full window.

        Ongoing closure only counts after it exceeds blink_max (avoid blink inflation).
        """
        window = float(cfg["perclos_window_sec"])
        blink_max = float(cfg["blink_max_ms"])
        if window <= 1e-6:
            return 0.0

        cutoff = now - window
        closed_sec = 0.0
        for seg in state.closed_segments:
            a = max(seg.start, cutoff)
            b = min(seg.end, now)
            if b > a:
                closed_sec += b - a

        # Ongoing long closure (already past blink length)
        if state.eyes_closed and state.closed_start is not None:
            ongoing_ms = (now - state.closed_start) * 1000.0
            if ongoing_ms > blink_max:
                # Count only the portion beyond blink_max, from closed_start+blink
                count_from = state.closed_start + blink_max / 1000.0
                a = max(count_from, cutoff)
                if now > a:
                    closed_sec += now - a

        closed_sec = min(closed_sec, window)
        return max(0.0, min(100.0, (closed_sec / window) * 100.0))

    def _snapshot(
        self, state: _SessionFatigueState, cfg: Dict[str, Any], now: float
    ) -> FatigueSnapshot:
        # Refresh yawn latch expiry
        if state.last_yawn and now >= state.yawn_until:
            state.last_yawn = False

        closed_ms = self._current_closed_ms(state, now)
        blink_max = int(cfg["blink_max_ms"])
        microsleep_min = int(cfg["microsleep_min_ms"])
        perclos = self._compute_perclos(state, now, cfg)
        alert_pct = float(cfg["perclos_alert_pct"])
        warn_pct = alert_pct * 0.75

        is_microsleep = closed_ms >= microsleep_min or now < state.recent_microsleep_until
        is_blink = (
            state.eyes_closed
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
            landmarks_size=state.last_landmarks_size,
            faces=state.last_face_entries,
        )


@dataclass
class _BehaviorState:
    smoking_hits: Deque[bool] = field(default_factory=deque)
    phone_hits: Deque[bool] = field(default_factory=deque)
    drinking_hits: Deque[bool] = field(default_factory=deque)
    confirm_frames: int = 2
    window_frames: int = 5


class BehaviorConfirmTracker:
    """Confirm smoking/phone/drinking when hits >= M inside last N frames (tolerant to flicker)."""

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
                state = _BehaviorState(
                    confirm_frames=cfg["behavior_confirm_frames"],
                    window_frames=cfg["behavior_window_frames"],
                )
                self._sessions[int(session_id)] = state
            else:
                state.confirm_frames = cfg["behavior_confirm_frames"]
                state.window_frames = cfg["behavior_window_frames"]

    @staticmethod
    def _push(window: Deque[bool], hit: bool, size: int) -> int:
        window.append(bool(hit))
        while len(window) > size:
            window.popleft()
        return sum(1 for v in window if v)

    def update(
        self,
        session_id: int,
        smoking: bool,
        phone: bool,
        drinking: bool,
        configs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = load_fatigue_config(configs)
        need = max(1, int(cfg["behavior_confirm_frames"]))
        win = max(need, int(cfg["behavior_window_frames"]))
        with self._lock:
            sid = int(session_id)
            state = self._sessions.get(sid)
            if state is None:
                state = _BehaviorState(confirm_frames=need, window_frames=win)
                self._sessions[sid] = state
            else:
                state.confirm_frames = need
                state.window_frames = win

            s_hits = self._push(state.smoking_hits, smoking, win)
            p_hits = self._push(state.phone_hits, phone, win)
            d_hits = self._push(state.drinking_hits, drinking, win)

            return {
                "smoking_detected": s_hits >= need,
                "phone_detected": p_hits >= need,
                "drinking_detected": d_hits >= need,
                "confirm_progress": {
                    "smoking": {"hits": s_hits, "need": need, "window": win},
                    "phone": {"hits": p_hits, "need": need, "window": win},
                    "drinking": {"hits": d_hits, "need": need, "window": win},
                },
            }


fatigue_tracker = FatigueTemporalTracker()
behavior_tracker = BehaviorConfirmTracker()
