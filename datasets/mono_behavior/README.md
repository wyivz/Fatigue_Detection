# Mono behavior fine-tune dataset

Place grayscale (or Mono8→BGR) frames from the industrial camera here.

**Color sites:** prefer `datasets/color_behavior/` + `python tools/finetune_yolo.py`.
Runtime mono CLAHE is only a light adaptation — not a substitute for color training.

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
# or
python tools/finetune_yolo.py --data datasets/mono_behavior/data.yaml --mono
```

Copy the produced `best.pt` to `fatigue_detection_system/weights/best.pt`, then:

```
python tools/export_yolo_onnx.py
```
