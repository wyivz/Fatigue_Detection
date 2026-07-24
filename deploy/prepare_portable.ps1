# Build portable package for industrial PC (no Python required on target).
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\deploy\prepare_portable.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutRoot = Join-Path (Split-Path -Parent $ProjectRoot) "SleepyDetect_Portable"
$PythonInstaller = Join-Path $ProjectRoot "python-3.8.10-amd64.exe"
$SrcApp = Join-Path $ProjectRoot "fatigue_detection_system"
$SrcVenv = Join-Path $ProjectRoot "venv"
$StartBat = Join-Path $PSScriptRoot "start.bat"
$ReadmeSrc = Join-Path $PSScriptRoot "PORTABLE_README.txt"

Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "OutRoot     = $OutRoot"

if (-not (Test-Path -LiteralPath $PythonInstaller)) { throw "Missing python-3.8.10-amd64.exe" }
if (-not (Test-Path -LiteralPath $SrcApp)) { throw "Missing fatigue_detection_system" }
if (-not (Test-Path -LiteralPath $SrcVenv)) { throw "Missing venv" }
if (-not (Test-Path -LiteralPath $StartBat)) { throw "Missing deploy\start.bat" }

if (Test-Path -LiteralPath $OutRoot) {
    Write-Host "Removing old output..."
    Remove-Item -LiteralPath $OutRoot -Recurse -Force
}

$RuntimePy = Join-Path $OutRoot "runtime\Python38"
$OutVenv = Join-Path $OutRoot "venv"
$OutApp = Join-Path $OutRoot "app"

New-Item -ItemType Directory -Force -Path $RuntimePy | Out-Null

Write-Host "1/5 Installing portable Python 3.8.10 ..."
$p = Start-Process -FilePath $PythonInstaller -ArgumentList @(
    "/quiet",
    "InstallAllUsers=0",
    "PrependPath=0",
    "Include_launcher=0",
    "Include_test=0",
    "SimpleInstall=1",
    "TargetDir=$RuntimePy"
) -Wait -PassThru

$PyExe = Join-Path $RuntimePy "python.exe"
if (-not (Test-Path -LiteralPath $PyExe)) {
    throw "Python install failed. ExitCode=$($p.ExitCode)"
}
Write-Host "Python OK: $PyExe"

Write-Host "2/5 Copying venv (large, please wait) ..."
cmd /c "robocopy `"$SrcVenv`" `"$OutVenv`" /E /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np"
if ($LASTEXITCODE -ge 8) { throw "robocopy venv failed: $LASTEXITCODE" }

Write-Host "3/5 Copying app + weights ..."
cmd /c "robocopy `"$SrcApp`" `"$OutApp`" /E /XD __pycache__ .git /XF *.pyc .DS_Store /NFL /NDL /NJH /NJS /nc /ns /np"
if ($LASTEXITCODE -ge 8) { throw "robocopy app failed: $LASTEXITCODE" }

# Drop bulky result images if present (keep folder)
$ResultsDir = Join-Path $OutApp "media\results"
if (Test-Path -LiteralPath $ResultsDir) {
    Get-ChildItem -LiteralPath $ResultsDir -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

$BestPt = Join-Path $OutApp "weights\best.pt"
$DlibDat = Join-Path $OutApp "weights\shape_predictor_68_face_landmarks.dat"
if (-not (Test-Path -LiteralPath $BestPt)) { Write-Host "[WARN] missing weights\best.pt" }
if (-not (Test-Path -LiteralPath $DlibDat)) { Write-Host "[WARN] missing dlib landmarks dat" }

Write-Host "4/5 Writing start.bat / pyvenv.cfg / readme ..."
Copy-Item -LiteralPath $StartBat -Destination (Join-Path $OutRoot "start.bat") -Force
if (Test-Path -LiteralPath $ReadmeSrc) {
    Copy-Item -LiteralPath $ReadmeSrc -Destination (Join-Path $OutRoot "README.txt") -Force
}

$cfgPath = Join-Path $OutVenv "pyvenv.cfg"
$cfgLines = @(
    "home = $RuntimePy",
    "include-system-site-packages = false",
    "version = 3.8.10"
)
$cfgLines | Set-Content -LiteralPath $cfgPath -Encoding ASCII

Write-Host "5/5 Measuring size ..."
$bytes = (Get-ChildItem -LiteralPath $OutRoot -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
Write-Host ("DONE. Output: {0}" -f $OutRoot)
Write-Host ("Size: {0:N2} GB" -f ($bytes / 1GB))
Write-Host "Copy SleepyDetect_Portable folder to industrial PC, then run start.bat"
