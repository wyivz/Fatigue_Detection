# -*- coding: utf-8 -*-
"""One-click performance presets — defaults biased toward archive-like quality."""
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
    "mvs_stream_max_width",
    "mvs_stream_max_height",
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
    "dlib_refine_mode",
    "dlib_landmark_mode",
    "fatigue_level_mode",
    "yawn_confirm_ms",
    "mouth_ar_thresh",
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


def _archive_common() -> Dict[str, str]:
    """Shared knobs matching archive: HOG landmarks + instant EAR/MAR."""
    return {
        "yolo_spatial_filter": "false",
        "yolo_near_face_ratio": "2.0",
        "behavior_confirm_frames": "1",
        "behavior_window_frames": "1",
        "behavior_sensitivity": "loose",
        "mono_camera_mode": "false",
        # Archive: frontal HOG on frame (after YOLO draw), not YOLO-box predictor
        "dlib_landmark_mode": "hog",
        "dlib_refine_mode": "off",
        "fatigue_level_mode": "instant",
        "yawn_confirm_ms": "0",
        "mouth_ar_thresh": "0.6",
        "ear_busy_stretch_pct": "100",
    }


def _base_cpu_smooth() -> Dict[str, str]:
    cfg = {
        "performance_preset": "cpu_smooth",
        "device": "cpu",
        "cuda_half": "false",
        "yolo_conf_thresh": "0.50",
        "yolo_conf_smoke": "0.45",
        "yolo_conf_phone": "0.45",
        "yolo_conf_water": "0.45",
        "yolo_conf_face": "0.45",
        "yolo_iou_thresh": "0.5",
        "yolo_imgsz": "640",
        # Uniform scale (not ROI crop): HOG needs ~webcam face size
        "yolo_detect_max_width": "960",
        "mvs_stream_max_width": "640",
        "mvs_stream_max_height": "0",
        # Async persist + no wasted intermediate draw (see archive_pipeline) let
        # this run noticeably tighter than the old 1000ms without starving I/O.
        "detection_interval": "800",
        "ear_sample_interval_ms": "100",
        "compute_yolo_threads": "4",
        "compute_ear_threads": "2",
    }
    cfg.update(_archive_common())
    return cfg


def _base_balanced(use_cuda: bool) -> Dict[str, str]:
    cfg = {
        "performance_preset": "balanced",
        "device": "cuda:0" if use_cuda else "cpu",
        "cuda_half": "true" if use_cuda else "false",
        "yolo_conf_thresh": "0.50",
        "yolo_conf_smoke": "0.45",
        "yolo_conf_phone": "0.45",
        "yolo_conf_water": "0.45",
        "yolo_conf_face": "0.45",
        "yolo_iou_thresh": "0.5",
        "yolo_imgsz": "640",
        "yolo_detect_max_width": "960",
        "mvs_stream_max_width": "640",
        "mvs_stream_max_height": "0",
        # Archive template default was 500ms; async persist + trimmed per-frame
        # copies close most of the previous overhead-driven gap.
        "detection_interval": "700" if not use_cuda else "500",
        "ear_sample_interval_ms": "100",
        "compute_yolo_threads": "4" if not use_cuda else "2",
        "compute_ear_threads": "2",
    }
    cfg.update(_archive_common())
    return cfg


def _base_gpu_quality() -> Dict[str, str]:
    cfg = {
        "performance_preset": "gpu_quality",
        "device": "cuda:0",
        "cuda_half": "true",
        "yolo_conf_thresh": "0.45",
        "yolo_conf_smoke": "0.40",
        "yolo_conf_phone": "0.40",
        "yolo_conf_water": "0.40",
        "yolo_conf_face": "0.40",
        "yolo_iou_thresh": "0.5",
        "yolo_imgsz": "640",
        "yolo_detect_max_width": "1280",
        "mvs_stream_max_width": "960",
        "mvs_stream_max_height": "0",
        "detection_interval": "400",
        "ear_sample_interval_ms": "80",
        "compute_yolo_threads": "2",
        "compute_ear_threads": "4",
    }
    cfg.update(_archive_common())
    return cfg


PRESET_META = {
    "cpu_smooth": {
        "title": "工控 CPU · 流畅",
        "subtitle": "算力紧张时使用",
        "hint": "归档同帧：YOLO→HOG；等比≤960 / 间隔 0.8s，异步归档减轻卡顿",
        "icon": "fa-microchip",
    },
    "balanced": {
        "title": "均衡（推荐）",
        "subtitle": "本机 CPU / 多数工作站",
        "hint": "归档逻辑：HOG 68 点 + 即时/PERCLOS 疲劳；等比≤960；间隔 0.5-0.7s",
        "icon": "fa-balance-scale",
    },
    "gpu_quality": {
        "title": "GPU · 高精度",
        "subtitle": "需 NVIDIA 显卡 + CUDA 版 PyTorch",
        "hint": "CUDA YOLO + 归档 HOG；等比≤1280；间隔 0.4s",
        "icon": "fa-bolt",
    },
}


def build_behavior_mode(
    mode_id: str,
    base_conf: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Plain-language behavior sensitivity → confirm M/N + conf floors.

    loose (default): archive instant 1/1, no spatial filter
    normal: 2/5
    strict: 3/6 + spatial
    """
    mid = (mode_id or "loose").strip().lower()
    if mid in ("strict", "少误报"):
        mid = "strict"
    elif mid in ("normal", "均衡"):
        mid = "normal"
    else:
        mid = "loose"

    base = dict(base_conf or {})
    g = float(base.get("yolo_conf_thresh") or 0.40)
    smoke = float(base.get("yolo_conf_smoke") or 0.35)
    phone = float(base.get("yolo_conf_phone") or 0.35)
    water = float(base.get("yolo_conf_water") or 0.35)

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
            "yolo_near_face_ratio": "1.5",
        }
    if mid == "normal":
        return {
            "behavior_sensitivity": "normal",
            "behavior_confirm_frames": "2",
            "behavior_window_frames": "5",
            "yolo_conf_thresh": _c(g),
            "yolo_conf_smoke": _c(smoke),
            "yolo_conf_phone": _c(phone),
            "yolo_conf_water": _c(water),
            "yolo_spatial_filter": "false",
            "yolo_near_face_ratio": "2.0",
        }
    return {
        "behavior_sensitivity": "loose",
        "behavior_confirm_frames": "1",
        "behavior_window_frames": "1",
        "yolo_conf_thresh": _c(max(0.25, g - 0.05)),
        "yolo_conf_smoke": _c(max(0.22, smoke - 0.05)),
        "yolo_conf_phone": _c(max(0.22, phone - 0.05)),
        "yolo_conf_water": _c(max(0.22, water - 0.05)),
        "yolo_spatial_filter": "false",
        "yolo_near_face_ratio": "2.0",
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
