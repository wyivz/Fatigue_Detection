param(
    [switch]$Quiet
)

$ErrorActionPreference = 'SilentlyContinue'
$Root = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$Port = 8000

function Write-Info([string]$Message) {
    if (-not $Quiet) {
        Write-Host $Message
    }
}

Write-Info "Stopping SleepyDetect processes under:"
Write-Info "  $Root"

$killed = New-Object System.Collections.Generic.List[int]
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"

foreach ($p in $procs) {
    $cmd = $p.CommandLine
    if (-not $cmd) { continue }

    $underRoot = $cmd.IndexOf($Root, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    $isApp = ($cmd -match 'runserver') -or ($cmd -match 'manage\.py')
    if (-not ($underRoot -and $isApp)) { continue }

    try {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
        $killed.Add($p.ProcessId) | Out-Null
        Write-Info ("  Stopped PID {0}" -f $p.ProcessId)
    } catch {
        Write-Info ("  Failed PID {0}: {1}" -f $p.ProcessId, $_.Exception.Message)
    }
}

Start-Sleep -Seconds 1

$listen = netstat -ano | Select-String 'LISTENING' | Select-String (':{0}\s' -f $Port)
foreach ($line in $listen) {
    if ($line -notmatch '\s(\d+)\s*$') { continue }
    $procId = [int]$Matches[1]
    $proc = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $procId)
    if (-not $proc -or $proc.Name -ne 'python.exe') { continue }

    try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        if (-not $killed.Contains($procId)) {
            $killed.Add($procId) | Out-Null
            Write-Info ("  Freed port {0} by stopping PID {1}" -f $Port, $procId)
        }
    } catch {
    }
}

if ($killed.Count -eq 0) {
    Write-Info "No running SleepyDetect server process found."
} else {
    Write-Info ("Done. Stopped {0} process(es)." -f $killed.Count)
}

exit 0
