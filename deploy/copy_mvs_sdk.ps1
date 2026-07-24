# After installing Hikrobot MVS 4.4 on this PC, optionally refresh MvImport from Samples.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\deploy\copy_mvs_sdk.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $ProjectRoot "fatigue_detection_system\detection\utils\hik_mvs\MvImport"

$candidates = @(
    "${env:ProgramFiles(x86)}\MVS\Development\Samples\Python\MvImport",
    "$env:ProgramFiles\MVS\Development\Samples\Python\MvImport"
)

$src = $null
foreach ($c in $candidates) {
    if (Test-Path -LiteralPath $c) { $src = $c; break }
}
if (-not $src) {
    throw "MVS Samples MvImport not found. Install MVS 4.4 first."
}

Write-Host "Copy from: $src"
Write-Host "Copy to:   $Dest"
if (Test-Path -LiteralPath $Dest) {
    Remove-Item -LiteralPath $Dest -Recurse -Force
}
Copy-Item -LiteralPath $src -Destination $Dest -Recurse -Force
Write-Host "Done. Re-apply DLL path patch in MvCameraControl_class.py if needed."
