# -*- coding: utf-8 -*-
"""
Fine-tune behavior YOLO (face/smoke/phone/water) on mono industrial camera data.

Dataset layout (YOLO format):
  datasets/mono_behavior/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

data.yaml example:
  path: datasets/mono_behavior
  train: images/train
  val: images/val
  names:
    0: face
    1: smoke
    2: phone
    3: water

Usage (from repo root, with ultralytics installed):
  python tools/finetune_mono_yolo.py
  python tools/finetune_mono_yolo.py --epochs 80 --imgsz 960 --device 0

After training, copy runs/.../weights/best.pt over:
  fatigue_detection_system/weights/best.pt
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

    data_path = Path(args.data)
    weights_path = Path(args.weights)
    if not data_path.is_file():
        raise SystemExit(
            f"Missing {data_path}\n"
            "Create YOLO-format mono dataset first (see docstring)."
        )
    if not weights_path.is_file():
        raise SystemExit(f"Missing base weights: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_path),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device,
        project=str(args.project),
        name=str(args.name),
        exist_ok=True,
        pretrained=True,
        # Mono industrial: keep mosaic mild; grayscale-friendly HSV dampening
        hsv_h=0.0,
        hsv_s=0.1,
        hsv_v=0.3,
    )
    out = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Done. New weights: {out}")
    print(f"Replace deploy weights with:\n  copy {out} {weights_path}")


if __name__ == "__main__":
    main()
