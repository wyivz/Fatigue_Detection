# Fatigue Detection（基于 YOLO 与 dlib）

一套面向驾驶场景的实时检测系统：看人是否疲劳（眯眼、微睡、打哈欠），以及是否抽烟、打电话、喝水。

- YOLO：找 face / smoke / phone / water
- dlib：在人脸框上量眼睛开合（EAR）、嘴巴开合（MAR）

## 功能

- 实时检测（webcam / GigE）
- 视频与图片检测
- 疲劳评估：EAR/MAR、多帧 PERCLOS、哈欠确认
- 行为检测：YOLO（face / smoke / phone / water）
- 历史会话与结果统计
- Windows 便携包 / 一键安装脚本（见 `deploy/`）

- 画面
  → 按 yolo_detect_max_width 缩小（与工业相机一致）
  → [可选] 黑白 CLAHE 增强（彩色默认关）
  → YOLO 检测 + 分项门槛 + 贴脸过滤 + 主脸粘滞
  → BehaviorConfirmTracker（M 次命中 / 最近 N 次）
  → dlib light 精修 → EAR/MAR
  → FatigueTemporalTracker（含会话 EAR 校准、侧脸门控）
  → 按需画框 / 写库
## 环境

- Python 3.8（推荐 3.8.10）
- Django 4.2
- OpenCV、dlib、ultralytics、torch

Windows 上 dlib 建议用仓库根目录 wheel：

```bash
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
```

## 快速开始（开发）

```bash
git clone https://github.com/wyivz/Fatigue_Detection.git
cd Fatigue_Detection
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
```

权重文件放到 `fatigue_detection_system/weights/`：

- `best.pt`（YOLO）
- `shape_predictor_68_face_landmarks.dat`（[dlib 官方](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)）

若仓库已包含上述文件可跳过下载。

```bash
cd fatigue_detection_system
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 127.0.0.1:8000
```

或在仓库根目录双击 `start.bat`。

浏览器打开 http://127.0.0.1:8000/

### 环境变量（可选）

| 变量 | 说明 |
|------|------|
| `DJANGO_SECRET_KEY` | 生产环境必填 |
| `DJANGO_DEBUG` | `1`/`0`，默认 `1` |
| `DJANGO_ALLOWED_HOSTS` | 逗号分隔，默认 `127.0.0.1,localhost` |

## 工控机部署

- 便携包：`deploy/prepare_portable.ps1`
- 一键安装包：`deploy/prepare_install_bundle.ps1` → 目标机运行 `install.bat`
- GigE：安装海康 MVS Runtime，说明见 `deploy/MVS_SETUP.txt`
- 离线包安装完成后，可直接双击 `train_behavior.bat` 做现场行为识别微调

一键安装若自动创建管理员，默认口令为 `ChangeMeNow!`，登录后请立即修改。

## 黑白工业相机微调

见 `datasets/color_behavior/README.md`、`datasets/mono_behavior/README.md`、
`tools/finetune_yolo.py` 与 `tools/finetune_mono_yolo.py`。
**彩色现场请优先**用 `tools/finetune_yolo.py` 在彩色标注数据上微调，黑白增强（CLAHE）仅作轻度适配。
离线安装包场景下，安装完成后可直接双击根目录 `train_behavior.bat`，脚本会自动选择数据集、训练、回写 `best.pt` 并导出 `best.onnx`。

## ONNX 加速（可选）

```bash
pip install onnx onnxruntime   # GPU 可用时也可装 onnxruntime-gpu
python tools/export_yolo_onnx.py
```

将生成 `fatigue_detection_system/weights/best.onnx`。应用启动时优先加载 ONNX，失败则回退 `best.pt`。

## 技术栈

前端：HTML / Bootstrap 5 / JavaScript  
后端：Django  
视觉：OpenCV、dlib、YOLO（ultralytics）  
数据库：SQLite（开发）

## 目录摘要

```
Fatigue_Detection/
├── fatigue_detection_system/   # Django 应用
├── deploy/                     # 便携 / 一键安装
├── datasets/color_behavior/    # 彩色微调（推荐）
├── datasets/mono_behavior/     # 黑白微调（轻度）
├── tools/                      # 微调 / ONNX 导出
├── requirements.txt
├── start.bat / stop.bat
└── README.md
```

## 许可与声明

仅供学习与研究。实车/产线部署前请完成充分测试，并自行配置密钥与访问控制。


