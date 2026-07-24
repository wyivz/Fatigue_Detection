# Mono behavior fine-tune dataset

Place grayscale (or Mono8→BGR) frames from the industrial camera here.

```
datasets/mono_behavior/
  images/train/
  images/val/
  labels/train/   # YOLO txt: class x_center y_center w h (normalized)
  labels/val/
  data.yaml
```

Classes must match deploy weights: `face=0`, `smoke=1`, `phone=2`, `water=3`.

Then run:

```
python tools/finetune_mono_yolo.py --device cpu
```

Copy the produced `best.pt` to `fatigue_detection_system/weights/best.pt`.
