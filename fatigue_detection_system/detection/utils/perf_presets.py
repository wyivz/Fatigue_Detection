# -*- coding: utf-8 -*-
"""One-click performance presets for CPU industrial PCs vs GPU workstations."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def detect_cuda() -> Tuple[bool, str]:
    """Return (available, display_name)."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name or "CUDA GPU"
    except Exception:  # noqa: BLE001
        pass
    return False, ""


# Keys applied by presets (fatigue/alert defaults left alone unless listed)
PRESET_KEYS = (
    "performance_preset",
    "device",
    "cuda_half",
    "yolo_conf_thresh",
    "yolo_conf_smoke",
    "yolo_conf_phone",
    "yolo_conf_water",
    "yolo_conf_face",
    "yolo_iou_thresh",
    "yolo_imgsz",
    "yolo_detect_max_width",
    "yolo_spatial_filter",
    "yolo_near_face_ratio",
    "detection_interval",
    "ear_sample_interval_ms",
    "behavior_confirm_frames",
    "behavior_window_frames",
    "behavior_sensitivity",
    "compute_yolo_threads",
    "compute_ear_threads",
    "ear_busy_stretch_pct",
    "mono_camera_mode",
)

# Plain-language behavior modes (write existing confirm/conf keys)
BEHAVIOR_MODE_KEYS = (
    "behavior_sensitivity",
    "behavior_confirm_frames",
    "behavior_window_frames",
    "yolo_conf_thresh",
    "yolo_conf_smoke",
    "yolo_conf_phone",
    "yolo_conf_water",
    "yolo_spatial_filter",
    "yolo_near_face_ratio",
)


def _base_cpu_smooth() -> Dict[str, str]:
    return {
        "performance_preset": "cpu_smooth",
        "device": "cpu",
        "cuda_half": "false",
        "yolo_conf_thresh": "0.35",
        "yolo_conf_smoke": "0.35",
        "yolo_conf_phone": "0.35",
        "yolo_conf_water": "0.35",
        "yolo_conf_face": "0.40",
        "yolo_iou_thresh": "0.5",
        "yolo_imgsz": "512",
        "yolo_detect_max_width": "800",
        "yolo_spatial_filter": "true",
        "yolo_near_face_ratio": "1.5",
        "detection_interval": "600",
        "ear_sample_interval_ms": "120",
        "behavior_confirm_frames": "2",
        "behavior_window_frames": "5",
        "behavior_sensitivity": "normal",
        "compute_yolo_threads": "0",
        "compute_ear_threads": "0",
        "ear_busy_stretch_pct": "150",
        "mono_camera_mode": "false",
    }


def _base_balanced(use_cuda: bool) -> Dict[str, str]:
    d = {
        "performance_preset": "balanced",
        "device": "cuda:0" if use_cuda else "cpu",
        "cuda_half": "true" if use_cuda else "false",
        "yolo_conf_thresh": "0.30",
        "yolo_conf_smoke": "0.28",
        "yolo_conf_phone": "0.30",
        "yolo_conf_water": "0.30",
        "yolo_conf_face": "0.35",
        "yolo_iou_thresh": "0.5",
        "yolo_imgsz": "640",
        "yolo_detect_max_width": "960",
        "yolo_spatial_filter": "true",
        "yolo_near_face_ratio": "1.5",
        "detection_interval": "500",
        "ear_sample_interval_ms": "100",
        "behavior_confirm_frames": "2",
        "behavior_window_frames": "5",
        "behavior_sensitivity": "normal",
        "compute_yolo_threads": "0",
        "compute_ear_threads": "0",
        "ear_busy_stretch_pct": "150",
        "mono_camera_mode": "false",
    }
    return d


def _base_gpu_quality() -> Dict[str, str]:
    return {
        "performance_preset": "gpu_quality",
        "device": "cuda:0",
        "cuda_half": "true",
        "yolo_conf_thresh": "0.25",
        "yolo_conf_smoke": "0.22",
        "yolo_conf_phone": "0.25",
        "yolo_conf_water": "0.25",
        "yolo_conf_face": "0.35",
        "yolo_iou_thresh": "0.5",
        "yolo_imgsz": "960",
        "yolo_detect_max_width": "1280",
        "yolo_spatial_filter": "true",
        "yolo_near_face_ratio": "1.5",
        "detection_interval": "300",
        "ear_sample_interval_ms": "80",
        "behavior_confirm_frames": "2",
        "behavior_window_frames": "5",
        "behavior_sensitivity": "normal",
        "compute_yolo_threads": "0",
        "compute_ear_threads": "0",
        "ear_busy_stretch_pct": "120",
        "mono_camera_mode": "false",
    }


PRESET_META = {
    "cpu_smooth": {
        "title": "工控 CPU · 流畅",
        "subtitle": "推荐无独显的工业电脑",
        "hint": "优先不卡顿；降低送检分辨率。彩色相机请保持黑白增强关闭",
        "icon": "fa-microchip",
    },
    "balanced": {
        "title": "均衡（推荐）",
        "subtitle": "多数彩色现场默认选择",
        "hint": "速度与检出率折中；有 GPU 时自动用 GPU。彩色默认请选此项",
        "icon": "fa-balance-scale",
    },
    "gpu_quality": {
        "title": "GPU · 高精度",
        "subtitle": "需 NVIDIA 显卡 + CUDA 版 PyTorch",
        "hint": "更高分辨率与更短检测间隔，适合算力充足的工控/工控机",
        "icon": "fa-bolt",
    },
}


def build_behavior_mode(
    mode_id: str,
    base_conf: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Plain-language behavior sensitivity → confirm M/N + conf floors.

    Modes (locked values from optimization plan):
      normal  均衡：2/5
      strict  少误报：3/6，行为类 conf +0.05
      loose   少漏报：2/4，推理 conf 略降
    """
    mid = (mode_id or "normal").strip().lower()
    if mid in ("strict", "少误报"):
        mid = "strict"
    elif mid in ("loose", "少漏报"):
        mid = "loose"
    else:
        mid = "normal"

    base = dict(base_conf or {})
    # Fall back to balanced-like confs when caller has no current values
    g = float(base.get("yolo_conf_thresh") or 0.30)
    smoke = float(base.get("yolo_conf_smoke") or 0.28)
    phone = float(base.get("yolo_conf_phone") or 0.30)
    water = float(base.get("yolo_conf_water") or 0.30)

    def _c(v: float) -> str:
        return f"{max(0.15, min(0.95, v)):.2f}"

    if mid == "strict":
        return {
            "behavior_sensitivity": "strict",
            "behavior_confirm_frames": "3",
            "behavior_window_frames": "6",
            "yolo_conf_thresh": _c(g + 0.05),
            "yolo_conf_smoke": _c(smoke + 0.05),
            "yolo_conf_phone": _c(phone + 0.05),
            "yolo_conf_water": _c(water + 0.05),
            "yolo_spatial_filter": "true",
            "yolo_near_face_ratio": "1.3",
        }
    if mid == "loose":
        return {
            "behavior_sensitivity": "loose",
            "behavior_confirm_frames": "2",
            "behavior_window_frames": "4",
            "yolo_conf_thresh": _c(g - 0.05),
            "yolo_conf_smoke": _c(max(0.18, smoke - 0.05)),
            "yolo_conf_phone": _c(max(0.18, phone - 0.05)),
            "yolo_conf_water": _c(max(0.18, water - 0.05)),
            "yolo_spatial_filter": "true",
            "yolo_near_face_ratio": "1.8",
        }
    return {
        "behavior_sensitivity": "normal",
        "behavior_confirm_frames": "2",
        "behavior_window_frames": "5",
        "yolo_conf_thresh": _c(g),
        "yolo_conf_smoke": _c(smoke),
        "yolo_conf_phone": _c(phone),
        "yolo_conf_water": _c(water),
        "yolo_spatial_filter": "true",
        "yolo_near_face_ratio": "1.5",
    }


def build_preset(preset_id: str, cuda_available: Optional[bool] = None) -> Dict[str, str]:
    if cuda_available is None:
        cuda_available, _ = detect_cuda()
    pid = (preset_id or "balanced").strip().lower()
    if pid == "cpu_smooth":
        return _base_cpu_smooth()
    if pid == "gpu_quality":
        cfg = _base_gpu_quality()
        if not cuda_available:
            # Soft-fallback so apply never bricks a CPU-only box
            cfg = _base_balanced(False)
            cfg["performance_preset"] = "balanced"
        return cfg
    return _base_balanced(bool(cuda_available))


def preset_catalog(cuda_available: bool) -> Dict[str, Any]:
    return {
        "cuda_available": cuda_available,
        "presets": PRESET_META,
        "values": {
            "cpu_smooth": build_preset("cpu_smooth", cuda_available),
            "balanced": build_preset("balanced", cuda_available),
            "gpu_quality": build_preset("gpu_quality", cuda_available),
        },
    }
