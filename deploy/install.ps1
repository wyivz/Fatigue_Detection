# SleepyDetect one-click installer (target PC).
# Usage:
#   Double-click install.bat
#   OR: powershell -ExecutionPolicy Bypass -File .\install.ps1
# Options:
#   -ForceCpu          Force CPU torch (skip CUDA device config)
#   -CudaIndex cu121   Online-only torch index (ignored when OFFLINE.flag present)
#   -SkipMigrate       Skip Django migrate
#   -AdminUser xxx     Default admin username (default: admin)
#   -AdminPass xxx     Default admin password (default: ChangeMeNow!)
#   -AllowOnline       Even if OFFLINE.flag exists, allow PyPI fallback (not recommended)

param(
    [switch]$ForceCpu,
    [ValidateSet("cu118", "cu121", "cu124")]
    [string]$CudaIndex = "cu121",
    [switch]$SkipMigrate,
    [string]$AdminUser = "admin",
    [string]$AdminPass = "ChangeMeNow!",
    [switch]$AllowOnline
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
        if (Test-Path -LiteralPath $c) { return [string]$c }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path -LiteralPath $cmd.Source)) {
        try {
            # Capture version to variable so it does NOT pollute function return value
            $ver = & $cmd.Source -c "import sys; print('%d.%d'%sys.version_info[:2])" 2>$null
            if ("$ver".Trim() -eq "3.8") { return [string]$cmd.Source }
        } catch { }
    }
    return $null
}

function Ensure-Python38 {
    $py = Find-Python38
    if ($py) {
        Write-Host "Found Python: $py"
        # IMPORTANT: do not let --version write to the success stream (pollutes return value in PS)
        Write-Host $((& $py --version) 2>&1 | Out-String).Trim()
        return [string]$py
    }

    $embedded = Join-Path $Root "runtime\Python38\python.exe"
    if (Test-Path -LiteralPath $embedded) {
        Write-Host "Using embedded portable Python: $embedded"
        Write-Host $((& $embedded --version) 2>&1 | Out-String).Trim()
        return [string]$embedded
    }

    $installer = Join-Path $Root "python-3.8.10-amd64.exe"
    if (-not (Test-Path -LiteralPath $installer)) {
        throw "Python 3.8 not found. Missing runtime\Python38\ and python-3.8.10-amd64.exe."
    }

    Write-Host ""
    Write-Host "No embedded Python found. Trying silent installer (may fail with 1603 on some PCs)..." -ForegroundColor Yellow
    Write-Host "If this fails: manually double-click python-3.8.10-amd64.exe, install for Current User,"
    Write-Host "check 'Add Python to PATH', then re-run install.bat"
    Write-Host ""

    $target = Join-Path $Root "runtime\Python38"
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $p = Start-Process -FilePath $installer -ArgumentList @(
        "/passive",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_test=0",
        "SimpleInstall=1",
        "TargetDir=$target"
    ) -Wait -PassThru

    $pyExe = Join-Path $target "python.exe"
    if (Test-Path -LiteralPath $pyExe) {
        return [string]$pyExe
    }

    Write-Host "TargetDir install missing python.exe (ExitCode=$($p.ExitCode)). Trying default user install..."
    $p2 = Start-Process -FilePath $installer -ArgumentList @(
        "/passive",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_launcher=1",
        "Include_test=0",
        "SimpleInstall=1"
    ) -Wait -PassThru
    Start-Sleep -Seconds 2
    $py = Find-Python38
    if ($py) { return [string]$py }

    throw @"
Python 3.8 install failed. ExitCode=$($p.ExitCode)/$($p2.ExitCode)

Common causes of 1603:
  - Package path contains Chinese/spaces (move to e.g. D:\SleepyDetect_Install_Offline)
  - Need to run install.bat as Administrator, or opposite: as normal user
  - Pending Windows reboot / another installer running
  - Antivirus blocking python-3.8.10-amd64.exe

Fix (offline):
  1) Double-click python-3.8.10-amd64.exe in this folder
  2) Install for Current User, enable Add to PATH
  3) Close window, run install.bat again

Or rebuild the offline pack on the dev PC so it includes runtime\Python38\ (no installer needed).
"@
}

Write-Host "SleepyDetect Installer"
Write-Host "Root = $Root"

$OfflineFlag = Join-Path $Root "OFFLINE.flag"
$WheelsDir = Join-Path $Root "wheels"
$strictOffline = (Test-Path -LiteralPath $OfflineFlag) -and (-not $AllowOnline)
if ($strictOffline) {
    Write-Host "OFFLINE mode: will NOT use network (wheels\ only)" -ForegroundColor Yellow
    if (-not (Test-Path -LiteralPath $WheelsDir)) {
        throw "OFFLINE.flag present but wheels\ folder missing."
    }
    $whlCount = @(Get-ChildItem -LiteralPath $WheelsDir -Filter "*.whl" -File -ErrorAction SilentlyContinue).Count
    if ($whlCount -lt 5) {
        throw "wheels\ looks incomplete ($whlCount whl files). Rebuild with prepare_install_bundle.ps1 -Offline"
    }
}

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
    $DlibWheel = Get-ChildItem -LiteralPath $WheelsDir -Filter "dlib-*-cp38-*-win_amd64.whl" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
}
if (-not $DlibWheel) {
    throw "Missing dlib cp38 win_amd64 wheel in package root or wheels\."
}

$ReqFile = Join-Path $Root "requirements-app.txt"
if (-not (Test-Path -LiteralPath $ReqFile)) {
    throw "Missing requirements-app.txt"
}

Write-Step "1/6 Python 3.8"
$PyBase = Ensure-Python38
# Defensive: PowerShell may return Object[] if any host output leaked into the pipeline
if ($PyBase -is [System.Array]) { $PyBase = [string]($PyBase | Select-Object -Last 1) }
$PyBase = [string]$PyBase
if (-not (Test-Path -LiteralPath $PyBase)) {
    throw "Resolved Python path is invalid: '$PyBase'"
}
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

# Always point venv home to THIS package's embedded Python (avoid sibling folder paths)
$runtimePy = Join-Path $Root "runtime\Python38"
if (Test-Path -LiteralPath (Join-Path $runtimePy "python.exe")) {
    $cfg = Join-Path $VenvDir "pyvenv.cfg"
    @(
        "home = $runtimePy",
        "include-system-site-packages = false",
        "version = 3.8.10"
    ) | Set-Content -LiteralPath $cfg -Encoding ASCII
    Write-Host "pyvenv.cfg -> $runtimePy"
}
& $VenvPy --version

Write-Step "3/6 Upgrade pip / wheel"
if ($strictOffline) {
    Write-Host "Offline: upgrading pip from wheels\"
    & $VenvPy -m pip install --no-index --find-links $WheelsDir --upgrade pip setuptools wheel
    Assert-Ok "offline pip upgrade"
} else {
    & $VenvPy -m pip install --upgrade pip setuptools wheel
    Assert-Ok "pip upgrade"
}

Write-Step "4/6 Install dlib + torch + app deps"
Write-Host "Installing dlib from: $($DlibWheel.FullName)"
& $VenvPy -m pip install --force-reinstall --no-deps $DlibWheel.FullName
Assert-Ok "pip install dlib"

$hasLocalTorch = $false
if (Test-Path -LiteralPath $WheelsDir) {
    $hasLocalTorch = [bool](Get-ChildItem -LiteralPath $WheelsDir -Filter "torch-*.whl" -File -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$useCuda = $false
if (-not $ForceCpu) {
    $useCuda = Test-HasNvidia
}

if ($hasLocalTorch) {
    Write-Host "Installing torch/torchvision from local wheels\ ..."
    & $VenvPy -m pip install --no-index --find-links $WheelsDir torch torchvision
    Assert-Ok "pip install local torch"
    Write-Host "Local torch installed."
} elseif ($strictOffline) {
    throw "OFFLINE mode but no torch-*.whl in wheels\. Rebuild package with -Offline."
} else {
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
}

if ($strictOffline -or $hasLocalTorch) {
    Write-Host "Installing app deps from wheels\ only ..."
    & $VenvPy -m pip install --no-index --find-links $WheelsDir -r $ReqFile
    Assert-Ok "pip install requirements (offline)"
} else {
    & $VenvPy -m pip install -r $ReqFile
    Assert-Ok "pip install requirements-app.txt"
}

# Force clean ultralytics install (USB/copy corruption often leaves null-bytes in .py files)
Write-Host "Reinstalling ultralytics cleanly from wheels ..."
& $VenvPy -m pip install --force-reinstall --no-deps --no-index --find-links $WheelsDir "ultralytics==8.3.40"
if ($LASTEXITCODE -ne 0) {
    & $VenvPy -m pip install --force-reinstall --no-deps --no-index --find-links $WheelsDir ultralytics
}
Assert-Ok "pip reinstall ultralytics"

$ultraCheck = & $VenvPy -c "from ultralytics import YOLO; import ultralytics; print('ultralytics', ultralytics.__version__, 'ok')"
if ($LASTEXITCODE -ne 0) {
    throw "ultralytics import failed after install (corrupted package?). Delete venv\ and re-run install.bat; or re-copy wheels\ from a clean USB."
}
Write-Host $ultraCheck

Write-Step "5/6 Verify torch / CUDA"
$verifyOut = & $VenvPy -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
Write-Host $verifyOut
$useCuda = $false
if ($verifyOut -match "True") {
    $useCuda = $true
}
if ($ForceCpu) { $useCuda = $false }

Write-Step "6/6 Django migrate + admin"
Push-Location -LiteralPath $AppDir
try {
    if (-not $SkipMigrate) {
        & $VenvPy manage.py migrate --noinput
        Assert-Ok "migrate"
    }

    $createAdmin = @"
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fatigue_detection.settings')
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
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fatigue_detection.settings')
import django
django.setup()
from detection.models import SystemConfig
obj, created = SystemConfig.objects.update_or_create(
    config_key='device',
    defaults={
        'config_value': 'cuda:0',
        'description': 'Inference device (cpu / cuda:0)',
    },
)
print('SYSTEM_CONFIG_DEVICE', 'created' if created else 'updated', 'cuda:0')
"@
        & $VenvPy -c $setCuda
    }
} finally {
    Pop-Location
}

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
"installed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); cuda=$useCuda; offline=$strictOffline" | Set-Content -LiteralPath $marker -Encoding ASCII

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " INSTALL OK" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Next:"
Write-Host "  1) (GigE) Install Hikrobot MVS 4.6.3 Runtime if needed"
Write-Host "  2) Double-click start.bat"
Write-Host "  3) Open http://127.0.0.1:8000/"
Write-Host "Admin: $AdminUser / $AdminPass"
if ($strictOffline) { Write-Host "Mode: OFFLINE" }
if ($useCuda) {
    Write-Host "Torch: CUDA enabled"
} else {
    Write-Host "Torch: CPU (set device to cuda:0 in System Config after GPU drivers work)"
}
Write-Host ""
