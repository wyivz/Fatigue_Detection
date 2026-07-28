# -*- coding: utf-8 -*-
"""
Export SleepyDetect behavior YOLO weights to ONNX for faster deploy inference.

Usage (from repo root, ultralytics installed):
  python tools/export_yolo_onnx.py
  python tools/export_yolo_onnx.py --weights fatigue_detection_system/weights/best.pt --imgsz 640

Output (default):
  fatigue_detection_system/weights/best.onnx

The app prefers best.onnx when present (Ultralytics + ONNX Runtime), else falls
back to best.pt. Install onnxruntime (CPU) or onnxruntime-gpu as needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_weights = root / "fatigue_detection_system" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(description="Export YOLO best.pt → best.onnx")
    parser.add_argument("--weights", type=str, default=str(default_weights))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    simplify = bool(args.simplify) and not bool(args.no_simplify)
    out = model.export(
        format="onnx",
        imgsz=int(args.imgsz),
        opset=int(args.opset),
        simplify=simplify,
        dynamic=False,
    )
    out_path = Path(str(out))
    target = weights.with_suffix(".onnx")
    if out_path.resolve() != target.resolve() and out_path.is_file():
        target.write_bytes(out_path.read_bytes())
        print(f"Copied to deploy path: {target}")
    print(f"Done. ONNX: {target if target.is_file() else out_path}")
    print("Install runtime: pip install onnxruntime   # or onnxruntime-gpu")
    print("Restart the Django app; YOLODetector will prefer best.onnx.")


if __name__ == "__main__":
    main()
