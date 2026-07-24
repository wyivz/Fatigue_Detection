# SleepyDetect one-click installer (target PC).
# Usage:
#   Double-click install.bat
#   OR: powershell -ExecutionPolicy Bypass -File .\install.ps1
# Options:
#   -ForceCpu          Force CPU torch (skip CUDA)
#   -CudaIndex cu121   Torch CUDA index: cu118 / cu121 / cu124 (default cu121)
#   -SkipMigrate       Skip Django migrate
#   -AdminUser xxx     Default admin username (default: admin)
#   -AdminPass xxx     Default admin password (default: ChangeMeNow!)

param(
    [switch]$ForceCpu,
    [ValidateSet("cu118", "cu121", "cu124")]
    [string]$CudaIndex = "cu121",
    [switch]$SkipMigrate,
    [string]$AdminUser = "admin",
    [string]$AdminPass = "ChangeMeNow!"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "==== $msg ====" -ForegroundColor Cyan
}

function Assert-Ok([string]$what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit=$LASTEXITCODE)" }
}

function Test-HasNvidia {
    try {
        $p = Start-Process -FilePath "nvidia-smi.exe" -ArgumentList "-L" -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$env:TEMP\sleepy_nvsmi.txt" -RedirectStandardError "$env:TEMP\sleepy_nvsmi_err.txt"
        return ($p.ExitCode -eq 0)
    } catch {
        return $false
    }
}

function Find-Python38 {
    $candidates = @(
        (Join-Path $Root "runtime\Python38\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python38\python.exe"),
        "C:\Python38\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            $ver = & $cmd.Source -c "import sys; print('%d.%d'%sys.version_info[:2])"
            if ($ver -eq "3.8") { return $cmd.Source }
        } catch { }
    }
    return $null
}

function Ensure-Python38 {
    $py = Find-Python38
    if ($py) {
        Write-Host "Found Python: $py"
        & $py --version
        return $py
    }

    $installer = Join-Path $Root "python-3.8.10-amd64.exe"
    if (-not (Test-Path -LiteralPath $installer)) {
        throw "Python 3.8 not found and missing python-3.8.10-amd64.exe in package root."
    }

    $target = Join-Path $Root "runtime\Python38"
    Write-Host "Installing portable Python 3.8.10 to: $target"
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $p = Start-Process -FilePath $installer -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_test=0",
        "TargetDir=$target"
    ) -Wait -PassThru

    $pyExe = Join-Path $target "python.exe"
    if (-not (Test-Path -LiteralPath $pyExe)) {
        # Fallback: install to user default location
        Write-Host "TargetDir install missing python.exe, trying user install..."
        $p2 = Start-Process -FilePath $installer -ArgumentList @(
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_launcher=1",
            "Include_test=0"
        ) -Wait -PassThru
        Start-Sleep -Seconds 2
        $py = Find-Python38
        if (-not $py) {
            throw "Python 3.8 install failed. ExitCode=$($p.ExitCode)/$($p2.ExitCode)"
        }
        return $py
    }
    return $pyExe
}

Write-Host "SleepyDetect Installer"
Write-Host "Root = $Root"

$AppDir = Join-Path $Root "app"
if (-not (Test-Path -LiteralPath (Join-Path $AppDir "manage.py"))) {
    $alt = Join-Path $Root "fatigue_detection_system"
    if (Test-Path -LiteralPath (Join-Path $alt "manage.py")) {
        $AppDir = $alt
    } else {
        throw "Missing app\manage.py (or fatigue_detection_system\manage.py)"
    }
}

$DlibWheel = Get-ChildItem -LiteralPath $Root -Filter "dlib-*-cp38-*-win_amd64.whl" -File -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $DlibWheel) {
    throw "Missing dlib cp38 win_amd64 wheel in package root."
}

$ReqFile = Join-Path $Root "requirements-app.txt"
if (-not (Test-Path -LiteralPath $ReqFile)) {
    throw "Missing requirements-app.txt"
}

Write-Step "1/6 Python 3.8"
$PyBase = Ensure-Python38
Write-Host "Base Python: $PyBase"

Write-Step "2/6 Create venv"
$VenvDir = Join-Path $Root "venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPy)) {
    & $PyBase -m venv $VenvDir
    Assert-Ok "python -m venv"
} else {
    Write-Host "venv already exists: $VenvDir"
}
& $VenvPy --version

Write-Step "3/6 Upgrade pip / wheel"
& $VenvPy -m pip install --upgrade pip setuptools wheel
Assert-Ok "pip upgrade"

Write-Step "4/6 Install dlib + torch + app deps"
Write-Host "Installing dlib from: $($DlibWheel.FullName)"
& $VenvPy -m pip install --force-reinstall --no-deps $DlibWheel.FullName
Assert-Ok "pip install dlib"

$useCuda = $false
if (-not $ForceCpu) {
    $useCuda = Test-HasNvidia
}
if ($useCuda) {
    Write-Host "NVIDIA GPU detected. Installing torch+$CudaIndex ..."
    $index = "https://download.pytorch.org/whl/$CudaIndex"
    & $VenvPy -m pip install torch torchvision --index-url $index
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] $CudaIndex failed, trying cu118 ..." -ForegroundColor Yellow
        & $VenvPy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu118"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] CUDA torch failed, falling back to CPU torch." -ForegroundColor Yellow
        & $VenvPy -m pip install torch torchvision
        $useCuda = $false
    } else {
        $useCuda = $true
    }
} else {
    Write-Host "No NVIDIA GPU (or -ForceCpu). Installing CPU torch ..."
    & $VenvPy -m pip install torch torchvision
    Assert-Ok "pip install cpu torch"
}

# Prefer local wheels if present (offline / faster)
$WheelsDir = Join-Path $Root "wheels"
if (Test-Path -LiteralPath $WheelsDir) {
    Write-Host "Installing extra local wheels from: $WheelsDir"
    Get-ChildItem -LiteralPath $WheelsDir -Filter "*.whl" -File -ErrorAction SilentlyContinue | ForEach-Object {
        & $VenvPy -m pip install --no-deps $_.FullName
    }
}

& $VenvPy -m pip install -r $ReqFile
Assert-Ok "pip install requirements-app.txt"

Write-Step "5/6 Verify torch / CUDA"
$verify = @"
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"@
& $VenvPy -c $verify

Write-Step "6/6 Django migrate + admin"
Push-Location -LiteralPath $AppDir
try {
    if (-not $SkipMigrate) {
        & $VenvPy manage.py migrate --noinput
        Assert-Ok "migrate"
    }

    $createAdmin = @"
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fatigue_detection_system.settings')
import django
django.setup()
from django.contrib.auth import get_user_model
User = get_user_model()
u = '$AdminUser'
p = '$AdminPass'
if not User.objects.filter(username=u).exists():
    User.objects.create_superuser(u, '', p)
    print('CREATED_SUPERUSER', u)
else:
    print('SUPERUSER_EXISTS', u)
"@
    & $VenvPy -c $createAdmin

    if ($useCuda) {
        $setCuda = @"
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fatigue_detection_system.settings')
import django
django.setup()
from detection.models import SystemConfig
obj, _ = SystemConfig.objects.get_or_create(pk=1)
changed = False
if hasattr(obj, 'yolo_device'):
    if getattr(obj, 'yolo_device', None) != 'cuda:0':
        obj.yolo_device = 'cuda:0'
        changed = True
if hasattr(obj, 'device'):
    if getattr(obj, 'device', None) != 'cuda:0':
        obj.device = 'cuda:0'
        changed = True
# common field name in this project
for name, val in [('compute_device', 'cuda:0'), ('torch_device', 'cuda:0')]:
    if hasattr(obj, name) and getattr(obj, name) != val:
        setattr(obj, name, val)
        changed = True
if changed:
    obj.save()
    print('SYSTEM_CONFIG_CUDA_SET')
else:
    print('SYSTEM_CONFIG_CUDA_SKIP')
"@
        & $VenvPy -c $setCuda
    }
} finally {
    Pop-Location
}

# Fix pyvenv.cfg home if using portable runtime
$runtimePy = Join-Path $Root "runtime\Python38"
if (Test-Path -LiteralPath (Join-Path $runtimePy "python.exe")) {
    $cfg = Join-Path $VenvDir "pyvenv.cfg"
    @(
        "home = $runtimePy",
        "include-system-site-packages = false",
        "version = 3.8.10"
    ) | Set-Content -LiteralPath $cfg -Encoding ASCII
}

$marker = Join-Path $Root ".install_ok"
"installed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); cuda=$useCuda" | Set-Content -LiteralPath $marker -Encoding ASCII

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " INSTALL OK" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Next:"
Write-Host "  1) (GigE) Install Hikrobot MVS 4.6.3 Runtime if needed"
Write-Host "  2) Double-click start.bat"
Write-Host "  3) Open http://127.0.0.1:8000/"
Write-Host "Admin: $AdminUser / $AdminPass"
if ($useCuda) {
    Write-Host "Torch: CUDA enabled"
} else {
    Write-Host "Torch: CPU (set device to cuda:0 in System Config after GPU drivers work)"
}
Write-Host ""
