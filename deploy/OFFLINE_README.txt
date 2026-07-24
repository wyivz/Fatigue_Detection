SleepyDetect 离线一键安装包
==========================

适用：目标机无外网 / 无法访问 PyPI。

目标机要求
----------
- Windows 10/11 64 位
- 路径尽量无中文（例：D:\SleepyDetect_Install_Offline）
- 本包优先使用自带的 runtime\Python38（免安装）。若没有该目录才会尝试运行 python-3.8.10-amd64.exe
- 若出现 ExitCode 1603：不要依赖静默安装；请手动双击 python-3.8.10-amd64.exe 安装后重跑 install.bat，或向开发机索取带 runtime\Python38 的新包
- 安装路径尽量用纯英文，例如 D:\SleepyDetect_Install_Offline
- NVIDIA 机：请使用打包装时的 CUDA 版 torch（见 OFFLINE.flag）
- 工控 GigE：仍需本机安装海康 MVS Runtime（见 MVS_SETUP.txt；可与外网无关的离线安装包）
- 若缺 VC++ 运行库，请提前用离线方式安装 Microsoft Visual C++ 2015-2022 x64

使用步骤
--------
1. 把整个文件夹 U 盘拷到目标机
2. 双击 install.bat（完全离线，只读本地 wheels\）
3. 双击 start.bat
4. 浏览器 http://127.0.0.1:8000/
5. 默认账号 admin / ChangeMeNow!（立刻改密）

在有网开发机重新打包
--------------------
powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1 -Offline

CUDA 版（默认）:
  -Offline -TorchVariant Cuda -CudaIndex cu121

CPU 版:
  -Offline -TorchVariant Cpu

可选 -IncludeMvs 把 MVS 安装包一并打进目录。

说明
----
- 包体积通常数 GB（torch CUDA wheel 很大），属正常
- 目标机不要删 wheels\，安装完成后再删亦可
- 安装脚本若检测到 OFFLINE.flag，不会访问外网
