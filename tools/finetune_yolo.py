# -*- coding: utf-8 -*-
"""
Fine-tune behavior YOLO (face/smoke/phone/water) on color or mono datasets.

Prefer color in-cabin / dashcam footage for best accuracy. Mono industrial
cameras only need light CLAHE at runtime; use --mono only when training on
grayscale data.

Dataset layout (YOLO format), e.g. datasets/color_behavior/:
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml

data.yaml example:
  path: datasets/color_behavior
  train: images/train
  val: images/val
  names:
    0: face
    1: smoke
    2: phone
    3: water

Usage (from repo root):
  python tools/finetune_yolo.py --data datasets/color_behavior/data.yaml
  python tools/finetune_yolo.py --data datasets/mono_behavior/data.yaml --mono
  python tools/finetune_yolo.py --epochs 80 --imgsz 640 --device 0

After training:
  1) copy runs/.../weights/best.pt → fatigue_detection_system/weights/best.pt
  2) python tools/export_yolo_onnx.py
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_layout() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    if (root / "manage.py").is_file():
        return root, root
    app_dir = root / "fatigue_detection_system"
    return root, app_dir


def main() -> None:
    root, app_dir = _resolve_layout()
    default_data = root / "datasets" / "color_behavior" / "data.yaml"
    default_weights = app_dir / "weights" / "best.pt"

    parser = argparse.ArgumentParser(
        description="Fine-tune behavior YOLO (color preferred; --mono for grayscale)"
    )
    parser.add_argument("--data", type=str, default=str(default_data))
    parser.add_argument("--weights", type=str, default=str(default_weights))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--project",
        type=str,
        default=str(root / "runs" / "behavior_finetune"),
    )
    parser.add_argument("--name", type=str, default="finetune")
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Damp HSV augments for grayscale industrial datasets",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    weights_path = Path(args.weights)
    if not data_path.is_file():
        raise SystemExit(
            f"Missing {data_path}\n"
            "Create a YOLO-format dataset first (see docstring). "
            "Color in-cabin data is preferred."
        )
    if not weights_path.is_file():
        raise SystemExit(f"Missing base weights: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    train_kw = dict(
        data=str(data_path),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device,
        project=str(args.project),
        name=str(args.name),
        exist_ok=True,
        pretrained=True,
    )
    if args.mono:
        # Grayscale-friendly HSV dampening (legacy mono path)
        train_kw.update(hsv_h=0.0, hsv_s=0.1, hsv_v=0.3)

    model.train(**train_kw)
    out = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Done. New weights: {out}")
    print(f"Replace deploy weights with:\n  copy {out} {weights_path}")
    print("Then export ONNX:\n  python tools/export_yolo_onnx.py")


if __name__ == "__main__":
    main()
