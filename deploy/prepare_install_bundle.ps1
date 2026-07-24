# Build copy-and-install package (no prebuilt venv; target runs install.bat).
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1
# Optional:
#   -IncludeMvs            Copy MVS installer into the package
#   -DownloadCudaWheels    Pre-download torch CUDA wheels into wheels\ (needs network now)
#   -CudaIndex cu121       cu118 / cu121 / cu124
#   -OutDir D:\SleepyDetect_Install

param(
    [string]$OutDir = "",
    [switch]$IncludeMvs,
    [switch]$DownloadCudaWheels,
    [ValidateSet("cu118", "cu121", "cu124")]
    [string]$CudaIndex = "cu121"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path (Split-Path -Parent $ProjectRoot) "SleepyDetect_Install"
}

$SrcApp = Join-Path $ProjectRoot "fatigue_detection_system"
$PythonInstaller = Join-Path $ProjectRoot "python-3.8.10-amd64.exe"
$DlibWheel = Get-ChildItem -LiteralPath $ProjectRoot -Filter "dlib-*-cp38-*-win_amd64.whl" -File |
    Select-Object -First 1
$MvsInstaller = Join-Path $ProjectRoot "MVS_STD_4.6.3_260205.exe"

Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "OutDir      = $OutDir"

if (-not (Test-Path -LiteralPath $SrcApp)) { throw "Missing fatigue_detection_system" }
if (-not (Test-Path -LiteralPath $PythonInstaller)) { throw "Missing python-3.8.10-amd64.exe" }
if (-not $DlibWheel) { throw "Missing dlib cp38 wheel" }

if (Test-Path -LiteralPath $OutDir) {
    Write-Host "Removing old output..."
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "1/5 Copy app + weights ..."
$OutApp = Join-Path $OutDir "app"
cmd /c "robocopy `"$SrcApp`" `"$OutApp`" /E /XD __pycache__ .git media\results /XF *.pyc .DS_Store /NFL /NDL /NJH /NJS /nc /ns /np" | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy app failed: $LASTEXITCODE" }

# Keep empty media\results
$ResultsDir = Join-Path $OutApp "media\results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
Get-ChildItem -LiteralPath $ResultsDir -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

$BestPt = Join-Path $OutApp "weights\best.pt"
$DlibDat = Join-Path $OutApp "weights\shape_predictor_68_face_landmarks.dat"
if (-not (Test-Path -LiteralPath $BestPt)) { Write-Host "[WARN] missing weights\best.pt" }
if (-not (Test-Path -LiteralPath $DlibDat)) { Write-Host "[WARN] missing landmarks dat" }

Write-Host "2/5 Copy installers / wheels / scripts ..."
Copy-Item -LiteralPath $PythonInstaller -Destination (Join-Path $OutDir "python-3.8.10-amd64.exe") -Force
Copy-Item -LiteralPath $DlibWheel.FullName -Destination (Join-Path $OutDir $DlibWheel.Name) -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "requirements-app.txt") -Destination (Join-Path $OutDir "requirements-app.txt") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "install.ps1") -Destination (Join-Path $OutDir "install.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "install.bat") -Destination (Join-Path $OutDir "install.bat") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "bundle_start.bat") -Destination (Join-Path $OutDir "start.bat") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "stop.bat") -Destination (Join-Path $OutDir "stop.bat") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "stop.ps1") -Destination (Join-Path $OutDir "stop.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "MVS_SETUP.txt") -Destination (Join-Path $OutDir "MVS_SETUP.txt") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "INSTALL_README.txt") -Destination (Join-Path $OutDir "README.txt") -Force

if ($IncludeMvs) {
    if (Test-Path -LiteralPath $MvsInstaller) {
        Write-Host "Copying MVS installer (large) ..."
        Copy-Item -LiteralPath $MvsInstaller -Destination (Join-Path $OutDir (Split-Path -Leaf $MvsInstaller)) -Force
    } else {
        Write-Host "[WARN] -IncludeMvs set but MVS installer not found"
    }
}

Write-Host "3/5 Optional CUDA wheel download ..."
if ($DownloadCudaWheels) {
    $wheels = Join-Path $OutDir "wheels"
    New-Item -ItemType Directory -Force -Path $wheels | Out-Null
    $tmpPy = Join-Path $env:TEMP "sleepy_wheel_py"
    if (-not (Test-Path -LiteralPath (Join-Path $tmpPy "Scripts\python.exe"))) {
        $sysPy = Join-Path $env:LOCALAPPDATA "Programs\Python\Python38\python.exe"
        if (-not (Test-Path -LiteralPath $sysPy)) {
            throw "Need local Python 3.8 to download wheels (or omit -DownloadCudaWheels)"
        }
        & $sysPy -m venv $tmpPy
    }
    $wp = Join-Path $tmpPy "Scripts\python.exe"
    & $wp -m pip install --upgrade pip
    Write-Host "Downloading torch/torchvision ($CudaIndex) into wheels\ ..."
    & $wp -m pip download torch torchvision -d $wheels --index-url "https://download.pytorch.org/whl/$CudaIndex"
    if ($LASTEXITCODE -ne 0) { throw "pip download torch failed" }
    Write-Host "Downloading app deps into wheels\ ..."
    & $wp -m pip download -r (Join-Path $OutDir "requirements-app.txt") -d $wheels
    # Also keep dlib wheel copy inside wheels for offline pip
    Copy-Item -LiteralPath $DlibWheel.FullName -Destination (Join-Path $wheels $DlibWheel.Name) -Force
}

Write-Host "4/5 Write tip file ..."
$tip = @"
SleepyDetect Install Package
============================
1. Copy this whole folder to the target PC (prefer ASCII path, e.g. D:\SleepyDetect_Install)
2. Double-click install.bat  (needs network unless wheels\ was pre-downloaded)
3. Double-click start.bat
4. Browser: http://127.0.0.1:8000/   admin / ChangeMeNow!
5. GigE camera: install Hikrobot MVS 4.6.3 Runtime (see MVS_SETUP.txt)
"@
$tip | Set-Content -LiteralPath (Join-Path $OutDir "请先读我.txt") -Encoding UTF8

Write-Host "5/5 Measuring size ..."
$bytes = (Get-ChildItem -LiteralPath $OutDir -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
Write-Host ("DONE. Output: {0}" -f $OutDir)
Write-Host ("Size: {0:N2} GB" -f ($bytes / 1GB))
Write-Host "Copy the folder to CUDA PC, then run install.bat"
