# -*- coding: utf-8 -*-
"""
Fine-tune behavior YOLO on mono industrial camera data.

Prefer the generic color finetune for most sites:
  python tools/finetune_yolo.py --data datasets/color_behavior/data.yaml

This script remains for grayscale-only datasets (same as):
  python tools/finetune_yolo.py --data datasets/mono_behavior/data.yaml --mono

Dataset layout (YOLO format):
  datasets/mono_behavior/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

Usage (from repo root, with ultralytics installed):
  python tools/finetune_mono_yolo.py
  python tools/finetune_mono_yolo.py --epochs 80 --imgsz 960 --device 0

After training, copy runs/.../weights/best.pt over:
  fatigue_detection_system/weights/best.pt
Then optionally:
  python tools/export_yolo_onnx.py
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_data = root / "datasets" / "mono_behavior" / "data.yaml"
    default_weights = root / "fatigue_detection_system" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(description="Fine-tune mono behavior YOLO")
    parser.add_argument("--data", type=str, default=str(default_data))
    parser.add_argument("--weights", type=str, default=str(default_weights))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=str, default=str(root / "runs" / "mono_behavior"))
    parser.add_argument("--name", type=str, default="finetune")
    args = parser.parse_args()

    # Delegate to shared trainer with mono augment flags
    import runpy
    import sys

    sys.argv = [
        "finetune_yolo.py",
        "--data",
        str(args.data),
        "--weights",
        str(args.weights),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--project",
        str(args.project),
        "--name",
        str(args.name),
        "--mono",
    ]
    runpy.run_path(str(Path(__file__).resolve().parent / "finetune_yolo.py"), run_name="__main__")


if __name__ == "__main__":
    main()
