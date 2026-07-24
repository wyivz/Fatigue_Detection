@echo off
chcp 65001 >nul
setlocal EnableExtensions

rem 始终以本 bat 所在目录为根，便于整包拷贝到任意盘符
set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist "%ROOT%runtime\Python38\python.exe" (
  echo [错误] 未找到 runtime\Python38\python.exe
  echo 请先在开发机运行 prepare_portable.ps1 生成便携包。
  pause
  exit /b 1
)

if not exist "%ROOT%venv\Scripts\python.exe" (
  echo [错误] 未找到 venv\Scripts\python.exe
  echo 请先在开发机运行 prepare_portable.ps1 生成便携包。
  pause
  exit /b 1
)

if not exist "%ROOT%app\manage.py" (
  echo [错误] 未找到 app\manage.py
  pause
  exit /b 1
)

rem 修复 venv 指向当前目录下的便携 Python（换盘符/换路径后必须）
> "%ROOT%venv\pyvenv.cfg" (
  echo home = %ROOT%runtime\Python38
  echo include-system-site-packages = false
  echo version = 3.8.10
)

if not exist "%ROOT%app\weights\best.pt" (
  echo [警告] 缺少 app\weights\best.pt ，YOLO 检测将不可用
)
if not exist "%ROOT%app\weights\shape_predictor_68_face_landmarks.dat" (
  echo [警告] 缺少 dlib 模型文件，疲劳检测将不可用
)

set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

echo ========================================
echo  疲劳驾驶检测系统 - 便携启动
echo  目录: %ROOT%
echo  地址: http://127.0.0.1:%PORT%/
echo  按 Ctrl+C 可停止
echo ========================================

cd /d "%ROOT%app"
"%ROOT%venv\Scripts\python.exe" manage.py migrate --noinput
"%ROOT%venv\Scripts\python.exe" manage.py runserver 127.0.0.1:%PORT%

endlocal
