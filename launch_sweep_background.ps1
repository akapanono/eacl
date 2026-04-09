param(
    [string]$PythonExe = "python",
    [string]$ScriptPath = ".\sweep_random.py"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$logDir = Join-Path $projectRoot "sweep_logs"
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $logDir "launcher_$stamp.log"
$stderrLog = Join-Path $logDir "launcher_$stamp.err.log"
$metaFile = Join-Path $logDir "last_background_sweep.txt"

$proc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $ScriptPath `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru `
    -WindowStyle Hidden

$meta = @(
    "started_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "pid=$($proc.Id)"
    "python=$PythonExe"
    "script=$ScriptPath"
    "stdout_log=$stdoutLog"
    "stderr_log=$stderrLog"
    "summary_tsv=$(Join-Path $logDir 'summary.tsv')"
    "summary_csv=$(Join-Path $logDir 'summary.csv')"
)

$meta | Set-Content -Path $metaFile -Encoding UTF8

Write-Host "Background sweep started."
Write-Host "PID: $($proc.Id)"
Write-Host "Stdout log: $stdoutLog"
Write-Host "Stderr log: $stderrLog"
Write-Host "Meta file: $metaFile"
