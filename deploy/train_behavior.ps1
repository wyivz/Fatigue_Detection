# One-click behavior training for offline/app bundles.
# Usage:
#   Double-click train_behavior.bat
#   powershell -ExecutionPolicy Bypass -File .\train_behavior.ps1
#   powershell -ExecutionPolicy Bypass -File .\train_behavior.ps1 -Dataset mono -Epochs 80

param(
    [ValidateSet("auto", "color", "mono")]
    [string]$Dataset = "auto",
    [int]$Epochs = 50,
    [int]$Imgsz = 640,
    [int]$Batch = 8,
    [string]$Name = "",
    [switch]$ForceCpu
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

function Resolve-AppRoot {
    $candidates = @(
        (Join-Path $Root "app"),
        (Join-Path $Root "fatigue_detection_system"),
        $Root
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $c "manage.py")) { return $c }
    }
    throw "manage.py not found under app\, fatigue_detection_system\, or package root."
}

function Ensure-Dir([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

function Count-Images([string]$dir) {
    if (-not (Test-Path -LiteralPath $dir)) { return 0 }
    $patterns = @("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    $count = 0
    foreach ($p in $patterns) {
        $count += @(Get-ChildItem -LiteralPath $dir -Filter $p -File -ErrorAction SilentlyContinue).Count
    }
    return $count
}

Write-Host "SleepyDetect Behavior Training"
Write-Host "Root = $Root"

$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
    throw "Missing venv\Scripts\python.exe. Run install.bat first."
}

$AppRoot = Resolve-AppRoot
$ToolsDir = Join-Path $AppRoot "tools"
$DatasetsDir = Join-Path $AppRoot "datasets"
$WeightsDir = Join-Path $AppRoot "weights"
$RunsDir = Join-Path $AppRoot "runs"

if (-not (Test-Path -LiteralPath (Join-Path $ToolsDir "finetune_yolo.py"))) {
    throw "Missing tools\finetune_yolo.py under $ToolsDir."
}

Ensure-Dir $RunsDir
Ensure-Dir (Join-Path $DatasetsDir "color_behavior\images\train")
Ensure-Dir (Join-Path $DatasetsDir "color_behavior\images\val")
Ensure-Dir (Join-Path $DatasetsDir "color_behavior\labels\train")
Ensure-Dir (Join-Path $DatasetsDir "color_behavior\labels\val")
Ensure-Dir (Join-Path $DatasetsDir "mono_behavior\images\train")
Ensure-Dir (Join-Path $DatasetsDir "mono_behavior\images\val")
Ensure-Dir (Join-Path $DatasetsDir "mono_behavior\labels\train")
Ensure-Dir (Join-Path $DatasetsDir "mono_behavior\labels\val")

$ColorYaml = Join-Path $DatasetsDir "color_behavior\data.yaml"
$MonoYaml = Join-Path $DatasetsDir "mono_behavior\data.yaml"
if (-not (Test-Path -LiteralPath $ColorYaml)) { throw "Missing $ColorYaml" }
if (-not (Test-Path -LiteralPath $MonoYaml)) { throw "Missing $MonoYaml" }

$colorTrain = Count-Images (Join-Path $DatasetsDir "color_behavior\images\train")
$colorVal = Count-Images (Join-Path $DatasetsDir "color_behavior\images\val")
$monoTrain = Count-Images (Join-Path $DatasetsDir "mono_behavior\images\train")
$monoVal = Count-Images (Join-Path $DatasetsDir "mono_behavior\images\val")

if ($Dataset -eq "auto") {
    if ($colorTrain -gt 0 -and $colorVal -gt 0) {
        $Dataset = "color"
    } elseif ($monoTrain -gt 0 -and $monoVal -gt 0) {
        $Dataset = "mono"
    } else {
        throw @"
No trainable dataset found.

Expected either:
  $DatasetsDir\color_behavior\images\train + images\val
or
  $DatasetsDir\mono_behavior\images\train + images\val

Put YOLO-format images/labels there, then re-run train_behavior.bat.
"@
    }
}

if ($Dataset -eq "color") {
    if ($colorTrain -lt 1 -or $colorVal -lt 1) {
        throw "Color dataset selected, but images/train or images/val is empty."
    }
    $DataYaml = $ColorYaml
    $MonoFlag = $false
} else {
    if ($monoTrain -lt 1 -or $monoVal -lt 1) {
        throw "Mono dataset selected, but images/train or images/val is empty."
    }
    $DataYaml = $MonoYaml
    $MonoFlag = $true
}

$BaseWeights = Join-Path $WeightsDir "best.pt"
if (-not (Test-Path -LiteralPath $BaseWeights)) {
    throw "Missing base weights: $BaseWeights"
}

Write-Step "1/4 Resolve training device"
if ($ForceCpu) {
    $Device = "cpu"
} else {
    $Device = & $Py -c "import torch; print('0' if torch.cuda.is_available() else 'cpu')"
    Assert-Ok "torch device probe"
    $Device = "$Device".Trim()
    if ([string]::IsNullOrWhiteSpace($Device)) { $Device = "cpu" }
}
Write-Host "Dataset : $Dataset"
Write-Host "Device  : $Device"
Write-Host "Epochs  : $Epochs"
Write-Host "Imgsz   : $Imgsz"
Write-Host "Batch   : $Batch"

Write-Step "2/4 Fine-tune YOLO"
if ([string]::IsNullOrWhiteSpace($Name)) {
    $Name = "bundle_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}
$ProjectDir = Join-Path $RunsDir "behavior_finetune"
$trainArgs = @(
    (Join-Path $ToolsDir "finetune_yolo.py"),
    "--data", $DataYaml,
    "--weights", $BaseWeights,
    "--epochs", "$Epochs",
    "--imgsz", "$Imgsz",
    "--batch", "$Batch",
    "--device", "$Device",
    "--project", $ProjectDir,
    "--name", $Name
)
if ($MonoFlag) { $trainArgs += "--mono" }
& $Py @trainArgs
Assert-Ok "YOLO finetune"

$NewBest = Join-Path $ProjectDir "$Name\weights\best.pt"
if (-not (Test-Path -LiteralPath $NewBest)) {
    throw "Training finished but best.pt not found: $NewBest"
}

Write-Step "3/4 Deploy new weights"
$BackupPt = Join-Path $WeightsDir ("best.before_train_{0}.pt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Copy-Item -LiteralPath $BaseWeights -Destination $BackupPt -Force
Copy-Item -LiteralPath $NewBest -Destination $BaseWeights -Force
Write-Host "Backed up old weights -> $BackupPt"
Write-Host "Deployed new weights   -> $BaseWeights"

Write-Step "4/4 Export ONNX"
& $Py (Join-Path $ToolsDir "export_yolo_onnx.py") --weights $BaseWeights --imgsz "$Imgsz"
Assert-Ok "export_yolo_onnx"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " TRAIN OK" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "New model:"
Write-Host "  PT  : $BaseWeights"
Write-Host "  ONNX: $(Join-Path $WeightsDir 'best.onnx')"
Write-Host "Training run:"
Write-Host "  $ProjectDir\$Name"
Write-Host "Next:"
Write-Host "  1) Double-click start.bat"
Write-Host "  2) Verify smoke / phone / water / face on the target scene"
Write-Host ""
