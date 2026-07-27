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
    "compute_yolo_threads",
    "compute_ear_threads",
    "ear_busy_stretch_pct",
    "mono_camera_mode",
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
        "yolo_imgsz": "640",
        "yolo_detect_max_width": "960",
        "yolo_spatial_filter": "true",
        "yolo_near_face_ratio": "1.5",
        "detection_interval": "600",
        "ear_sample_interval_ms": "120",
        "behavior_confirm_frames": "2",
        "behavior_window_frames": "5",
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
        "compute_yolo_threads": "0",
        "compute_ear_threads": "0",
        "ear_busy_stretch_pct": "120",
        "mono_camera_mode": "false",
    }


PRESET_META = {
    "cpu_smooth": {
        "title": "工控 CPU · 流畅",
        "subtitle": "推荐无独显的工业电脑",
        "hint": "优先不卡顿，行为灵敏度适中",
        "icon": "fa-microchip",
    },
    "balanced": {
        "title": "均衡",
        "subtitle": "多数现场默认选择",
        "hint": "速度与检出率折中；有 GPU 时自动用 GPU",
        "icon": "fa-balance-scale",
    },
    "gpu_quality": {
        "title": "GPU · 高精度",
        "subtitle": "需 NVIDIA 显卡 + CUDA 版 PyTorch",
        "hint": "更高分辨率与更短检测间隔，适合算力充足的工控/工控机",
        "icon": "fa-bolt",
    },
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
