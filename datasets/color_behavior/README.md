# Color behavior fine-tune dataset (recommended)

Use in-cabin / dashcam **color** frames for best smoke / phone / water / face recall.

```
datasets/color_behavior/
  images/train/
  images/val/
  labels/train/   # YOLO txt: class x_center y_center w h (normalized)
  labels/val/
  data.yaml
```

Classes must match deploy weights: `face=0`, `smoke=1`, `phone=2`, `water=3`.

```yaml
# data.yaml
path: datasets/color_behavior
train: images/train
val: images/val
names:
  0: face
  1: smoke
  2: phone
  3: water
```

Train:

```
python tools/finetune_yolo.py --data datasets/color_behavior/data.yaml --device 0
```

Deploy:

1. Copy `runs/behavior_finetune/finetune/weights/best.pt` → `fatigue_detection_system/weights/best.pt`
2. `python tools/export_yolo_onnx.py`
3. Restart the app (loads `best.onnx` when present)
