# -*- coding: utf-8 -*-
"""CPU / CUDA compute scheduling for concurrent EAR (dlib) + YOLO threads.

Goals:
- Both loops keep running (no hard skip of EAR).
- On CPU: partition OpenCV/Torch thread pools so the two workers do not oversubscribe.
- On CUDA: YOLO runs on GPU (optional FP16); EAR keeps full-rate on CPU cores.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


def _cfg_int(configs: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(float(configs.get(key, default)))
    except (TypeError, ValueError):
        return default


def _cfg_bool(configs: Dict[str, Any], key: str, default: bool) -> bool:
    raw = configs.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _detect_cuda(device_pref: str) -> bool:
    pref = (device_pref or "cpu").strip().lower()
    if pref == "cpu" or pref.startswith("cpu"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class SchedulerPlan:
    device: str = "cpu"
    use_cuda: bool = False
    cuda_half: bool = True
    cpu_count: int = 4
    yolo_threads: int = 2
    ear_threads: int = 2
    opencv_yolo_threads: int = 2
    opencv_ear_threads: int = 1
    # When CPU YOLO is busy, stretch EAR interval by this factor (still runs).
    ear_busy_stretch: float = 1.5
    # Soft yield sleep while YOLO holds CPU (ms); 0 = never sleep-yield
    ear_busy_yield_ms: int = 5


class ComputeScheduler:
    """Process-wide scheduler for MVS EAR / YOLO workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.plan = SchedulerPlan(cpu_count=max(1, os.cpu_count() or 4))
        self._configured = False

    def configure(self, configs: Optional[Dict[str, Any]] = None) -> SchedulerPlan:
        if configs is None:
            configs = {}
            try:
                from detection.models import SystemConfig

                configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
            except Exception:  # noqa: BLE001
                configs = {}

        cpu_n = max(1, os.cpu_count() or 4)
        device = str(configs.get("device") or "cpu").strip() or "cpu"
        use_cuda = _detect_cuda(device)
        if use_cuda and not device.lower().startswith("cuda"):
            device = "cuda:0"

        yolo_t = _cfg_int(configs, "compute_yolo_threads", 0)
        ear_t = _cfg_int(configs, "compute_ear_threads", 0)
        cuda_half = _cfg_bool(configs, "cuda_half", True)
        busy_stretch = float(_cfg_int(configs, "ear_busy_stretch_pct", 150)) / 100.0
        busy_stretch = max(1.0, min(3.0, busy_stretch))
        busy_yield = max(0, _cfg_int(configs, "ear_busy_yield_ms", 5 if not use_cuda else 0))

        if use_cuda:
            # GPU does YOLO math; leave most CPU to dlib/OpenCV preprocess.
            if yolo_t <= 0:
                yolo_t = max(1, min(4, cpu_n // 4 or 1))
            if ear_t <= 0:
                ear_t = max(1, cpu_n - yolo_t)
            opencv_yolo = max(1, min(2, yolo_t))
            opencv_ear = max(1, min(ear_t, 2))
            busy_yield = 0  # no need to yield GPU vs CPU
        else:
            # Split cores ~60/40 favoring YOLO (heavier), keep both alive.
            if yolo_t <= 0:
                yolo_t = max(1, (cpu_n * 3) // 5)
            if ear_t <= 0:
                ear_t = max(1, cpu_n - yolo_t)
            # Avoid oversubscribe: cap sum at cpu_n
            if yolo_t + ear_t > cpu_n:
                ear_t = max(1, cpu_n - yolo_t)
            opencv_yolo = max(1, min(yolo_t, 4))
            opencv_ear = max(1, min(ear_t, 2))

        plan = SchedulerPlan(
            device=device if use_cuda else "cpu",
            use_cuda=use_cuda,
            cuda_half=bool(cuda_half and use_cuda),
            cpu_count=cpu_n,
            yolo_threads=int(yolo_t),
            ear_threads=int(ear_t),
            opencv_yolo_threads=int(opencv_yolo),
            opencv_ear_threads=int(opencv_ear),
            ear_busy_stretch=busy_stretch,
            ear_busy_yield_ms=busy_yield,
        )
        with self._lock:
            self.plan = plan
            self._configured = True
        self._apply_global_defaults()
        return plan

    def _apply_global_defaults(self) -> None:
        """Set process thread pools once (avoid per-inference races)."""
        plan = self.plan
        total = max(1, min(plan.cpu_count, plan.yolo_threads + plan.ear_threads))
        try:
            import torch

            torch.set_num_threads(total)
            try:
                torch.set_num_interop_threads(max(1, min(2, plan.cpu_count // 2 or 1)))
            except RuntimeError:
                # Can only be set once per process in some builds
                pass
        except Exception:  # noqa: BLE001
            pass
        try:
            import cv2

            cv2.setNumThreads(max(1, min(4, plan.opencv_yolo_threads + plan.opencv_ear_threads)))
        except Exception:  # noqa: BLE001
            pass

    def ensure_configured(self) -> SchedulerPlan:
        with self._lock:
            if not self._configured:
                return self.configure()
            return self.plan

    def snapshot(self) -> Dict[str, Any]:
        p = self.ensure_configured()
        return {
            "device": p.device,
            "use_cuda": p.use_cuda,
            "cuda_half": p.cuda_half,
            "cpu_count": p.cpu_count,
            "yolo_threads": p.yolo_threads,
            "ear_threads": p.ear_threads,
            "ear_busy_stretch": p.ear_busy_stretch,
            "ear_busy_yield_ms": p.ear_busy_yield_ms,
        }

    @contextmanager
    def yolo_context(self) -> Iterator[SchedulerPlan]:
        # Thread pools are fixed at configure time to avoid process-wide races
        # between concurrent EAR / YOLO workers.
        yield self.ensure_configured()

    @contextmanager
    def ear_context(self) -> Iterator[SchedulerPlan]:
        yield self.ensure_configured()

    def ear_interval_ms(self, base_ms: int, detect_busy: bool) -> int:
        """Effective EAR period: stretch on CPU when YOLO is busy; full rate on CUDA."""
        plan = self.ensure_configured()
        base = max(50, int(base_ms))
        if not detect_busy or plan.use_cuda:
            return base
        return int(base * plan.ear_busy_stretch)

    def ear_busy_pause_sec(self, detect_busy: bool) -> float:
        """Short cooperative yield on CPU only (EAR still runs next iteration)."""
        plan = self.ensure_configured()
        if not detect_busy or plan.use_cuda or plan.ear_busy_yield_ms <= 0:
            return 0.0
        return float(plan.ear_busy_yield_ms) / 1000.0


compute_scheduler = ComputeScheduler()
