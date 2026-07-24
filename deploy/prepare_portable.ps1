# 在开发机生成可拷贝到工控机的便携包
# 用法（PowerShell）:
#   cd C:\Users\REDACTED\Downloads\SleepyDetect
#   powershell -ExecutionPolicy Bypass -File .\deploy\prepare_portable.ps1
#
# 输出目录默认: 与项目同级的 SleepyDetect_Portable\

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutRoot = Join-Path (Split-Path -Parent $ProjectRoot) "SleepyDetect_Portable"
$PythonInstaller = Join-Path $ProjectRoot "python-3.8.10-amd64.exe"
$DlibWheel = Join-Path $ProjectRoot "dlib-19.19.0-cp38-cp38-win_amd64.whl"
$SrcApp = Join-Path $ProjectRoot "fatigue_detection_system"
$SrcVenv = Join-Path $ProjectRoot "venv"

Write-Host "项目目录: $ProjectRoot"
Write-Host "输出目录: $OutRoot"

if (-not (Test-Path $PythonInstaller)) { throw "缺少 python-3.8.10-amd64.exe" }
if (-not (Test-Path $SrcApp)) { throw "缺少 fatigue_detection_system" }
if (-not (Test-Path $SrcVenv)) { throw "缺少已安装好的 venv，请先在本机装好依赖" }

if (Test-Path $OutRoot) {
    Write-Host "清理旧输出..."
    Remove-Item -LiteralPath $OutRoot -Recurse -Force
}

$RuntimePy = Join-Path $OutRoot "runtime\Python38"
$OutVenv = Join-Path $OutRoot "venv"
$OutApp = Join-Path $OutRoot "app"

New-Item -ItemType Directory -Force -Path $RuntimePy | Out-Null

Write-Host "1/5 安装便携 Python 3.8.10 到 runtime\Python38 ..."
$p = Start-Process -FilePath $PythonInstaller -ArgumentList @(
    "/quiet",
    "InstallAllUsers=0",
    "PrependPath=0",
    "Include_launcher=0",
    "Include_test=0",
    "SimpleInstall=1",
    "TargetDir=$RuntimePy"
) -Wait -PassThru
if ($p.ExitCode -ne 0 -and -not (Test-Path (Join-Path $RuntimePy "python.exe"))) {
    throw "Python 静默安装失败, ExitCode=$($p.ExitCode)"
}
if (-not (Test-Path (Join-Path $RuntimePy "python.exe"))) {
    throw "未找到 $RuntimePy\python.exe"
}

Write-Host "2/5 复制已验证的 venv ..."
robocopy $SrcVenv $OutVenv /E /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "复制 venv 失败, robocopy=$LASTEXITCODE" }

Write-Host "3/5 复制应用与权重 ..."
robocopy $SrcApp $OutApp /E `
    /XD __pycache__ .git media\results `
    /XF *.pyc .DS_Store `
    /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
if ($LASTEXITCODE -ge 8) { throw "复制 app 失败, robocopy=$LASTEXITCODE" }

# 保证 weights 在
$Weights = Join-Path $OutApp "weights"
if (-not (Test-Path (Join-Path $Weights "best.pt"))) {
    Write-Host "[警告] app\weights\best.pt 不存在"
}
if (-not (Test-Path (Join-Path $Weights "shape_predictor_68_face_landmarks.dat"))) {
    Write-Host "[警告] dlib dat 不存在"
}

Write-Host "4/5 写入启动脚本与说明 ..."
Copy-Item (Join-Path $PSScriptRoot "start.bat") (Join-Path $OutRoot "start.bat") -Force

# 修正 pyvenv.cfg 指向便携 Python（start.bat 启动时还会再写一次）
@"
home = $RuntimePy
include-system-site-packages = false
version = 3.8.10
"@ | Set-Content -Path (Join-Path $OutVenv "pyvenv.cfg") -Encoding ASCII

@"
疲劳驾驶检测系统 - 便携包使用说明
================================

【工控机要求】
- Windows 10/11 或 Windows IoT / 工控版，64 位
- 建议安装 Microsoft Visual C++ 2015-2022 Redistributable (x64)
- 有摄像头时需系统能识别摄像头
- 路径不要包含中文或空格（推荐 D:\SleepyDetect_Portable）

【部署】
1. 将整个 SleepyDetect_Portable 文件夹拷贝到工控机
2. 双击 start.bat
3. 浏览器打开 http://127.0.0.1:8000/
4. 用已有账号登录，或访问 /accounts/register/ 注册

【指定端口】
  start.bat 8001

【不能单文件 exe 的原因】
本系统含 PyTorch / YOLO / dlib / OpenCV，体积约 2GB+，
用 PyInstaller 打单文件极不稳定。整包拷贝是工控场景更稳妥的方式。

【首次启动慢】
第一次加载 YOLO/dlib 模型需要几十秒，属正常。
"@ | Set-Content -Path (Join-Path $OutRoot "使用说明.txt") -Encoding UTF8

Write-Host "5/5 统计体积 ..."
$bytes = (Get-ChildItem -LiteralPath $OutRoot -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
Write-Host ("完成. 输出: {0}" -f $OutRoot)
Write-Host ("大约体积: {0:N2} GB" -f ($bytes / 1GB))
Write-Host "下一步: 把 SleepyDetect_Portable 整夹拷到工控机，双击 start.bat"
