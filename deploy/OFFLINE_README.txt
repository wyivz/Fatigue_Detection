SleepyDetect 离线一键安装包
==========================

先看哪个文件
------------
第一次接手时，先打开：

- `00_START_HERE.txt`

它是唯一首看文档，告诉你：
- 先双击哪个脚本
- 浏览器打开哪个地址
- 训练图片放到哪里
- 标签怎么写
- 如何一键训练

本文件 `README.txt` 是完整说明，适合遇到问题时再查。

适用场景
--------
- 目标机无外网 / 无法访问 PyPI
- Windows 10/11 64 位
- 需要在目标机本地运行疲劳检测，并可后续在现场继续微调行为识别模型

目录说明
--------
- `app\`：实际 Django 应用、权重、训练脚本、训练数据模板
- `runtime\Python38\`：便携 Python 3.8（优先使用）
- `wheels\`：离线安装依赖包
- `install.bat`：一键安装
- `start.bat`：启动系统
- `stop.bat` / `stop.ps1`：停止系统
- `train_behavior.bat`：一键训练行为识别模型
- `MVS_SETUP.txt`：GigE / 海康 MVS 运行库说明
- `OFFLINE.flag`：离线包元信息（Torch 版本 / 构建时间）
- `UPDATE.stamp`：应用代码同步时间与版本标记

安装前检查
----------
- 路径尽量无中文，推荐 `D:\SleepyDetect_Install_Offline`
- 若缺 VC++ 运行库，请先离线安装 Microsoft Visual C++ 2015-2022 x64
- 工控 GigE 机器仍需单独安装海康 MVS Runtime（见 `MVS_SETUP.txt`）
- NVIDIA 机器请尽量使用打包时已带 CUDA torch 的离线包（见 `OFFLINE.flag`）
- 本包优先使用自带 `runtime\Python38\`，没有时才会尝试 `python-3.8.10-amd64.exe`

首次安装
--------
1. 把整个文件夹完整复制到目标机，例如 `D:\SleepyDetect_Install_Offline`
2. 双击 `install.bat`
3. 安装完成后双击 `start.bat`
4. 浏览器访问 `http://127.0.0.1:8000/`
5. 管理员账号默认 `admin`
6. 若安装脚本生成了 `ADMIN_CREDENTIALS.txt`，以该文件内密码登录；若没有则使用部署方约定密码并立即修改

日常运行
--------
- 启动：双击 `start.bat`
- 停止：双击 `stop.bat`
- 首次启动或升级后，建议先看浏览器页面能否正常打开，再接摄像头验证
- 若使用 GigE 工业相机，先确认 MVS Runtime 已装好，再在页面中选择 MVS 来源

升级 / 交接给下一个同事
----------------------
1. 先备份整个目录，至少备份：
   - `app\weights\best.pt`
   - `app\weights\best.onnx`
   - `app\db.sqlite3`（若需要保留历史）
   - `ADMIN_CREDENTIALS.txt`
2. 用开发机同步新的 `app\` 内容覆盖旧 `app\`
3. 保留目标机上的 `venv\`、`runtime\`、`wheels\`、`db.sqlite3`（除非明确要求重装）
4. 若权重也更新了，覆盖后重启 `start.bat`
5. 查看 `UPDATE.stamp` 核对同步时间和版本

一键训练行为识别模型
------------------
适用：现场新增了抽烟 / 打电话 / 喝水 / 人脸样本，需直接在目标机继续微调 YOLO。

训练前准备（YOLO 格式）
---------------------
把数据放到以下任一目录：

彩色场景（推荐）：
- `app\datasets\color_behavior\images\train\`
- `app\datasets\color_behavior\images\val\`
- `app\datasets\color_behavior\labels\train\`
- `app\datasets\color_behavior\labels\val\`
- `app\datasets\color_behavior\data.yaml`

黑白工业相机场景：
- `app\datasets\mono_behavior\images\train\`
- `app\datasets\mono_behavior\images\val\`
- `app\datasets\mono_behavior\labels\train\`
- `app\datasets\mono_behavior\labels\val\`
- `app\datasets\mono_behavior\data.yaml`

类别编号必须固定为：
- `0 = face`
- `1 = smoke`
- `2 = phone`
- `3 = water`

训练方法：
- 直接双击 `train_behavior.bat`
- 若彩色和黑白数据都在，脚本优先使用彩色数据
- 若只存在黑白数据，脚本自动走 mono 训练
- 训练完成后会自动：
  1. 备份旧 `best.pt`
  2. 用新 `best.pt` 覆盖 `app\weights\best.pt`
  3. 导出 `app\weights\best.onnx`

高级用法：
- `powershell -ExecutionPolicy Bypass -File .\train_behavior.ps1 -Dataset color`
- `powershell -ExecutionPolicy Bypass -File .\train_behavior.ps1 -Dataset mono -Epochs 80`
- `powershell -ExecutionPolicy Bypass -File .\train_behavior.ps1 -ForceCpu`

训练产物位置：
- 训练过程：`app\runs\behavior_finetune\...`
- 线上权重：`app\weights\best.pt`
- ONNX 权重：`app\weights\best.onnx`

常见问题
--------
- `ExitCode 1603`
  - 不要继续依赖静默安装
  - 手动双击 `python-3.8.10-amd64.exe` 安装 Python 3.8 后重跑 `install.bat`
- `wheels\ looks incomplete`
  - 说明离线依赖包不完整，需要在开发机重新打包
- `No trainable dataset found`
  - 说明训练目录下没有有效的 `images/train` 与 `images/val`
- 训练后效果没变化
  - 确认是否真的重启了 `start.bat`
  - 确认 `app\weights\best.pt` 修改时间已更新
  - 优先检查标注类别编号是否仍为 `0/1/2/3`

在有网开发机重新打包
--------------------
在仓库根目录执行：

`powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1 -Offline`

常用参数：
- CUDA 版（默认）：
  - `-Offline -TorchVariant Cuda -CudaIndex cu121`
- CPU 版：
  - `-Offline -TorchVariant Cpu`
- 包含海康 MVS 安装包：
  - `-Offline -IncludeMvs`

补充说明
--------
- 包体积通常数 GB（torch CUDA wheel 很大），属正常
- 目标机不要删 `wheels\`，安装完成后如需清空间可删，但以后重装前要重新补齐
- 安装脚本检测到 `OFFLINE.flag` 时不会访问外网
