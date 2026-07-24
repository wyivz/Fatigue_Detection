# 基于YOLO与dlib的疲劳驾驶检测系统

这是一个基于YOLO和dlib的疲劳驾驶检测系统，可以实时检测驾驶员的面部状态、疲劳程度和不良驾驶行为，如打哈欠、抽烟、打电话等。

## 功能特点

- 实时摄像头检测：使用摄像头实时监测驾驶员状态
- 视频文件检测：上传视频文件进行分析
- 图片检测：上传图片快速分析
- 多维度检测：
  - 面部检测和跟踪
  - 疲劳状态评估（基于眼睛闭合和打哈欠检测）
  - 不良驾驶行为检测（抽烟、打电话、喝水等）
- 检测结果可视化：实时显示检测结果和警告
- 检测数据统计分析：查看历史检测结果和统计图表

## 系统要求

- Python 3.8+
- Django 4.2
- OpenCV 4.5+
- dlib
- YOLO (YOLOv5/YOLOv8)
- 支持的操作系统：Windows、macOS、Linux

dlib 用whl安装

```bash
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
```

## 安装指南

1. 克隆仓库
```bash
git clone https://github.com/yourusername/fatigue-detection-system.git
cd fatigue-detection-system
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 下载必要的权重文件
   - 将YOLO权重文件 `best.pt` 放入 `weights` 目录
   - 将dlib面部特征点预测器 `shape_predictor_68_face_landmarks.dat` 放入 `weights` 目录

5. 运行数据库迁移
```bash
python manage.py makemigrations
python manage.py migrate
```

6. 创建超级用户
```bash
python manage.py createsuperuser
```

7. 启动开发服务器
```bash
python manage.py runserver
```

8. 访问系统
   - 在浏览器中打开 `http://127.0.0.1:8000/`
   - 使用创建的超级用户账号登录

## 权重文件下载

- YOLO权重文件：基于自定义数据集训练的YOLO权重文件，用于检测面部、抽烟、打电话和喝水行为
- dlib面部特征点预测器：可以从 [dlib官方模型](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) 下载

## 使用说明

### 实时检测
1. 登录系统后，点击"实时检测"
2. 选择摄像头，点击"开始检测"
3. 系统会实时分析摄像头画面，检测驾驶员状态
4. 检测到疲劳状态或不良行为时会发出警告

### 视频检测
1. 点击"视频检测"
2. 上传视频文件（支持mp4、avi、mov格式）
3. 选择需要检测的项目（疲劳状态、不良行为）
4. 点击"开始检测"，等待系统处理
5. 查看检测结果和统计信息

### 图片检测
1. 点击"图片检测"
2. 上传图片文件（支持jpg、png格式）
3. 选择需要检测的项目
4. 点击"开始检测"
5. 查看检测结果

## 系统架构

- 前端：HTML、CSS、JavaScript、Bootstrap 5
- 后端：Django
- 图像处理：OpenCV、dlib
- 目标检测：YOLO (YOLOv5/YOLOv8)
- 数据库：SQLite（开发环境）/ MySQL或PostgreSQL（生产环境）

## 文件结构

```
fatigue_detection_system/
├── accounts/            # 用户账户相关应用
├── detection/           # 检测功能相关应用
├── media/               # 媒体文件存储
│   ├── uploads/         # 上传的文件
│   └── results/         # 检测结果图像
├── static/              # 静态文件
├── templates/           # 全局模板
├── weights/             # 模型权重文件
├── manage.py            # Django管理脚本
└── fatigue_detection/   # 项目主配置
``` 