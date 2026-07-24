SleepyDetect 一键安装包
======================

目标电脑要求
------------
- Windows 10/11 64 位
- 建议路径不要有中文（例：D:\SleepyDetect_Install）
- 有网：安装时 pip 会下载 torch / 依赖
  （若打包时用了 -DownloadCudaWheels，则 wheels\ 已含离线包，可少依赖外网）
- NVIDIA 显卡 + 已装驱动：安装脚本会自动装 CUDA 版 torch，并把系统配置 device 设为 cuda:0
- 工控 GigE 相机：另装海康 MVS 4.6.3 Runtime（见 MVS_SETUP.txt）
- 缺 VC++ 运行库时，安装 Microsoft Visual C++ 2015-2022 x64

使用步骤（目标机）
----------------
1. 把整个 SleepyDetect_Install 文件夹复制到目标机
2. 双击 install.bat（约几分钟到十几分钟，视网速）
3. 双击 start.bat
4. 浏览器打开 http://127.0.0.1:8000/
5. 默认管理员：admin / ChangeMeNow!（请登录后立刻改密）

常用参数（高级）
--------------
powershell -ExecutionPolicy Bypass -File .\install.ps1 -ForceCpu
powershell -ExecutionPolicy Bypass -File .\install.ps1 -CudaIndex cu118
powershell -ExecutionPolicy Bypass -File .\install.ps1 -AdminUser myadmin -AdminPass MyPass123

在开发机重新打包
--------------
powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1

可选：
  -IncludeMvs              把 MVS 安装包一并打进目录
  -DownloadCudaWheels      预下载 CUDA torch 到 wheels\（包更大，目标机更省网）
  -CudaIndex cu121         cu118 / cu121 / cu124
  -OutDir D:\SleepyDetect_Install

说明
----
- 本包不拷贝本机 venv（避免把 CPU 版 torch 带到 CUDA 机）
- 第一次 start 加载模型可能较慢，属正常
- 使用 GigE 前请先关掉 MVS 客户端预览（设备独占）
