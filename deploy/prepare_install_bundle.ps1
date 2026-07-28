# Build OFFLINE copy-and-install package for air-gapped target PCs.
# Dev machine needs network ONCE to download wheels; target needs NO network.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1 -Offline
#   powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1 -Offline -TorchVariant Cpu
#   powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1 -Offline -IncludeMvs
#
# Output default: <parent>\SleepyDetect_Install_Offline

param(
    [string]$OutDir = "",
    [switch]$IncludeMvs,
    [switch]$Offline,
    [switch]$DownloadCudaWheels, # alias kept for compatibility; prefer -Offline
    [ValidateSet("Cuda", "Cpu")]
    [string]$TorchVariant = "Cuda",
    [ValidateSet("cu118", "cu121", "cu124")]
    [string]$CudaIndex = "cu121"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$wantOffline = $Offline -or $DownloadCudaWheels
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    if ($wantOffline) {
        $OutDir = Join-Path (Split-Path -Parent $ProjectRoot) "SleepyDetect_Install_Offline"
    } else {
        $OutDir = Join-Path (Split-Path -Parent $ProjectRoot) "SleepyDetect_Install"
    }
}

$SrcApp = Join-Path $ProjectRoot "fatigue_detection_system"
$PythonInstaller = Join-Path $ProjectRoot "python-3.8.10-amd64.exe"
$DlibWheel = Get-ChildItem -LiteralPath $ProjectRoot -Filter "dlib-*-cp38-*-win_amd64.whl" -File |
    Select-Object -First 1
$MvsInstaller = Join-Path $ProjectRoot "MVS_STD_4.6.3_260205.exe"

Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "OutDir      = $OutDir"
Write-Host "Offline     = $wantOffline"
Write-Host "Torch       = $TorchVariant $(if ($TorchVariant -eq 'Cuda') { $CudaIndex } else { 'cpu' })"

if (-not (Test-Path -LiteralPath $SrcApp)) { throw "Missing fatigue_detection_system" }
if (-not (Test-Path -LiteralPath $PythonInstaller)) { throw "Missing python-3.8.10-amd64.exe" }
if (-not $DlibWheel) { throw "Missing dlib cp38 wheel" }

if (Test-Path -LiteralPath $OutDir) {
    Write-Host "Removing old output..."
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "1/6 Copy app + weights (no sqlite / no media uploads) ..."
$OutApp = Join-Path $OutDir "app"
cmd /c "robocopy `"$SrcApp`" `"$OutApp`" /E /XD __pycache__ .git media\results media\uploads /XF *.pyc .DS_Store *.sqlite3 /NFL /NDL /NJH /NJS /nc /ns /np" | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy app failed: $LASTEXITCODE" }

Write-Host "1b/6 Copy training helpers (tools + dataset templates) ..."
$SrcTools = Join-Path $ProjectRoot "tools"
$SrcDatasets = Join-Path $ProjectRoot "datasets"
if (Test-Path -LiteralPath $SrcTools) {
    cmd /c "robocopy `"$SrcTools`" `"$OutApp\tools`" /E /XD __pycache__ /XF *.pyc .DS_Store /NFL /NDL /NJH /NJS /nc /ns /np" | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy tools failed: $LASTEXITCODE" }
}
if (Test-Path -LiteralPath $SrcDatasets) {
    cmd /c "robocopy `"$SrcDatasets`" `"$OutApp\datasets`" /E /XD __pycache__ /XF *.pyc .DS_Store /NFL /NDL /NJH /NJS /nc /ns /np" | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy datasets failed: $LASTEXITCODE" }
}

foreach ($sub in @("media\results", "media\uploads")) {
    $d = Join-Path $OutApp $sub
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    Set-Content -LiteralPath (Join-Path $d ".gitkeep") -Value "" -Encoding ASCII
}

$BestPt = Join-Path $OutApp "weights\best.pt"
$DlibDat = Join-Path $OutApp "weights\shape_predictor_68_face_landmarks.dat"
if (-not (Test-Path -LiteralPath $BestPt)) { Write-Host "[WARN] missing weights\best.pt" }
if (-not (Test-Path -LiteralPath $DlibDat)) { Write-Host "[WARN] missing landmarks dat" }

Write-Host "2/6 Copy installers / scripts / embedded Python ..."
Copy-Item -LiteralPath $PythonInstaller -Destination (Join-Path $OutDir "python-3.8.10-amd64.exe") -Force
Copy-Item -LiteralPath $DlibWheel.FullName -Destination (Join-Path $OutDir $DlibWheel.Name) -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "requirements-app.txt") -Destination (Join-Path $OutDir "requirements-app.txt") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "install.ps1") -Destination (Join-Path $OutDir "install.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "install.bat") -Destination (Join-Path $OutDir "install.bat") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "bundle_start.bat") -Destination (Join-Path $OutDir "start.bat") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "stop.bat") -Destination (Join-Path $OutDir "stop.bat") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "stop.ps1") -Destination (Join-Path $OutDir "stop.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "MVS_SETUP.txt") -Destination (Join-Path $OutDir "MVS_SETUP.txt") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "train_behavior.ps1") -Destination (Join-Path $OutDir "train_behavior.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "train_behavior.bat") -Destination (Join-Path $OutDir "train_behavior.bat") -Force

# Embed a ready-to-use Python 3.8 tree so target PCs do NOT need the MSI/exe installer
# (silent TargetDir installs often fail with exit 1603 on locked-down industrial Windows).
$SystemPy = Join-Path $env:LOCALAPPDATA "Programs\Python\Python38"
$OutRuntime = Join-Path $OutDir "runtime\Python38"
if (Test-Path -LiteralPath (Join-Path $SystemPy "python.exe")) {
    Write-Host "Embedding portable Python from $SystemPy ..."
    New-Item -ItemType Directory -Force -Path $OutRuntime | Out-Null
    cmd /c "robocopy `"$SystemPy`" `"$OutRuntime`" /E /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np" | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy embedded Python failed: $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath (Join-Path $OutRuntime "python.exe"))) {
        throw "Embedded Python missing python.exe"
    }
    Write-Host "Embedded Python OK"
} else {
    Write-Host "[WARN] No local Python38 to embed. Target may need to run python-3.8.10-amd64.exe manually if silent install fails (1603)."
}
if ($wantOffline) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "OFFLINE_README.txt") -Destination (Join-Path $OutDir "README.txt") -Force
} else {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "INSTALL_README.txt") -Destination (Join-Path $OutDir "README.txt") -Force
}

if ($IncludeMvs) {
    if (Test-Path -LiteralPath $MvsInstaller) {
        Write-Host "Copying MVS installer (large) ..."
        Copy-Item -LiteralPath $MvsInstaller -Destination (Join-Path $OutDir (Split-Path -Leaf $MvsInstaller)) -Force
    } else {
        Write-Host "[WARN] -IncludeMvs set but MVS installer not found"
    }
}

Write-Host "3/6 Download wheels for offline install ..."
$wheels = Join-Path $OutDir "wheels"
New-Item -ItemType Directory -Force -Path $wheels | Out-Null

if ($wantOffline) {
    # Prefer project venv (newer pip); fall back to system Python 3.8.
    $wp = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $wp)) {
        $sysPy = Join-Path $env:LOCALAPPDATA "Programs\Python\Python38\python.exe"
        if (-not (Test-Path -LiteralPath $sysPy)) {
            throw "Need project venv or Python 3.8 to download wheels (for -Offline)."
        }
        $tmpPy = Join-Path $env:TEMP "sleepy_wheel_py38"
        if (Test-Path -LiteralPath $tmpPy) {
            Remove-Item -LiteralPath $tmpPy -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Creating temp venv for pip download ..."
        & $sysPy -m venv $tmpPy
        if ($LASTEXITCODE -ne 0) { throw "venv create failed" }
        $wp = Join-Path $tmpPy "Scripts\python.exe"
        & $wp -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { throw "pip upgrade in temp venv failed" }
    } else {
        Write-Host "Using project venv for downloads: $wp"
    }

    # Bypass broken local Clash/IE proxy (e.g. 127.0.0.1:7897 when client is down)
    $env:HTTP_PROXY = ""
    $env:HTTPS_PROXY = ""
    $env:http_proxy = ""
    $env:https_proxy = ""
    $env:ALL_PROXY = ""
    $env:all_proxy = ""
    $env:NO_PROXY = "*"
    $env:no_proxy = "*"
    Remove-Item Env:PIP_INDEX_URL -ErrorAction SilentlyContinue
    Remove-Item Env:PIP_EXTRA_INDEX_URL -ErrorAction SilentlyContinue

    $pipTrust = @(
        "--proxy=",
        "--trusted-host", "pypi.org",
        "--trusted-host", "files.pythonhosted.org",
        "--trusted-host", "pypi.tuna.tsinghua.edu.cn",
        "--trusted-host", "download.pytorch.org"
    )

    Write-Host "Downloading pip/setuptools/wheel into wheels\ ..."
    & $wp -m pip download @pipTrust pip setuptools wheel -d $wheels -i "https://pypi.org/simple"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] pypi.org failed, trying tuna ..."
        & $wp -m pip download @pipTrust pip setuptools wheel -d $wheels -i "https://pypi.tuna.tsinghua.edu.cn/simple"
    }
    if ($LASTEXITCODE -ne 0) { throw "pip download bootstrap failed" }

    if ($TorchVariant -eq "Cuda") {
        Write-Host "Downloading torch/torchvision ($CudaIndex) — large, please wait ..."
        & $wp -m pip download @pipTrust torch torchvision -d $wheels --index-url "https://download.pytorch.org/whl/$CudaIndex"
        if ($LASTEXITCODE -ne 0) { throw "pip download CUDA torch failed" }
    } else {
        Write-Host "Downloading torch/torchvision (CPU) ..."
        & $wp -m pip download @pipTrust torch torchvision -d $wheels -i "https://pypi.org/simple"
        if ($LASTEXITCODE -ne 0) { throw "pip download CPU torch failed" }
    }

    Write-Host "Downloading app deps (Django/ultralytics/...) ..."
    & $wp -m pip download @pipTrust -r (Join-Path $OutDir "requirements-app.txt") -d $wheels -i "https://pypi.org/simple"
    if ($LASTEXITCODE -ne 0) {
        & $wp -m pip download @pipTrust -r (Join-Path $OutDir "requirements-app.txt") -d $wheels -i "https://pypi.tuna.tsinghua.edu.cn/simple"
    }
    if ($LASTEXITCODE -ne 0) { throw "pip download requirements-app failed" }

    # If CUDA package, drop PyPI CPU torch wheels that may have been pulled as deps.
    if ($TorchVariant -eq "Cuda") {
        Get-ChildItem -LiteralPath $wheels -Filter "torch-*.whl" -File | ForEach-Object {
            $n = $_.Name.ToLowerInvariant()
            if ($n -notmatch "cu1[12]") {
                Write-Host "Removing non-CUDA torch wheel: $($_.Name)"
                Remove-Item -LiteralPath $_.FullName -Force
            }
        }
        Get-ChildItem -LiteralPath $wheels -Filter "torchvision-*.whl" -File | ForEach-Object {
            $n = $_.Name.ToLowerInvariant()
            if ($n -notmatch "cu1[12]") {
                Write-Host "Removing non-CUDA torchvision wheel: $($_.Name)"
                Remove-Item -LiteralPath $_.FullName -Force
            }
        }
    }

    Copy-Item -LiteralPath $DlibWheel.FullName -Destination (Join-Path $wheels $DlibWheel.Name) -Force

    $meta = @(
        "offline=1"
        "torch_variant=$TorchVariant"
        "cuda_index=$CudaIndex"
        "built=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        "python=3.8"
    ) -join "`r`n"
    Set-Content -LiteralPath (Join-Path $OutDir "OFFLINE.flag") -Value $meta -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $wheels ".offline_ok") -Value "1" -Encoding ASCII

    $torchCount = @(Get-ChildItem -LiteralPath $wheels -Filter "torch-*.whl" -File).Count
    if ($torchCount -lt 1) { throw "No torch-*.whl in wheels\ after download" }
    Write-Host "wheels\ ready: $torchCount torch wheel(s)"
} else {
    Write-Host "Online package (no wheels pre-download). Target will need network for pip."
    Write-Host "For air-gapped targets, re-run with -Offline"
}

Write-Host "4/6 Write tip file ..."
if ($wantOffline) {
    $tip = @"
SleepyDetect OFFLINE Install Package
====================================
Target PC needs NO internet.

1. Copy this whole folder (e.g. D:\SleepyDetect_Install_Offline)
2. Double-click install.bat
3. Double-click start.bat
4. Browser http://127.0.0.1:8000/   admin / ChangeMeNow!
5. GigE: install MVS Runtime on target (see MVS_SETUP.txt) — MVS is separate
6. Fine-tune behavior model later with train_behavior.bat
Torch variant in this pack: $TorchVariant $(if ($TorchVariant -eq 'Cuda') { $CudaIndex })
"@
} else {
    $tip = @"
SleepyDetect Install Package (online pip)
=========================================
1. Copy folder to target
2. install.bat (needs network unless you rebuild with -Offline)
3. start.bat
"@
}
$tip | Set-Content -LiteralPath (Join-Path $OutDir "请先读我.txt") -Encoding UTF8

Write-Host "5/6 Measuring size ..."
$bytes = (Get-ChildItem -LiteralPath $OutDir -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
Write-Host ("DONE. Output: {0}" -f $OutDir)
Write-Host ("Size: {0:N2} GB" -f ($bytes / 1GB))
if ($wantOffline) {
    Write-Host "Copy SleepyDetect_Install_Offline to air-gapped PC, then run install.bat"
} else {
    Write-Host "Or rebuild offline: powershell -ExecutionPolicy Bypass -File .\deploy\prepare_install_bundle.ps1 -Offline"
}
Write-Host "6/6 OK"
