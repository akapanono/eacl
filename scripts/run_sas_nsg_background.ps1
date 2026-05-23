param(
    [string]$Dataset = "IEMOCAP",
    [int]$GpuId = 0,
    [int]$Epochs = 30,
    [int]$MaxRuns = 0,
    [ValidateSet("all", "ablation", "targeted", "fusion")]
    [string]$ExperimentSet = "all",
    [string]$BertPath = "pretrained/sup-simcse-roberta-large",
    [string]$AnchorPath = "emo_anchors/sup-simcse-roberta-large",
    [string]$OutRoot = "run_logs/sas_nsg_queue",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$launcherLog = Join-Path $OutRoot "launcher_${Dataset}_${ExperimentSet}_${stamp}.log"
$launcherErr = Join-Path $OutRoot "launcher_${Dataset}_${ExperimentSet}_${stamp}.err.log"
$pidFile = Join-Path $OutRoot "last_${Dataset}_${ExperimentSet}.pid"

$arguments = @(
    "scripts\sas_nsg_train_queue.py",
    "--dataset", $Dataset,
    "--gpu-id", "$GpuId",
    "--epochs", "$Epochs",
    "--max-runs", "$MaxRuns",
    "--experiment-set", $ExperimentSet,
    "--bert-path", $BertPath,
    "--anchor-path", $AnchorPath,
    "--out-root", $OutRoot
)

$proc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $launcherLog `
    -RedirectStandardError $launcherErr `
    -PassThru `
    -WindowStyle Hidden

$proc.Id | Set-Content -Path $pidFile -Encoding UTF8

Write-Host "SAS-NSG-EACL background queue started."
Write-Host "PID: $($proc.Id)"
Write-Host "Experiment set: $ExperimentSet"
Write-Host "Launcher log: $launcherLog"
Write-Host "Launcher err: $launcherErr"
Write-Host "PID file: $pidFile"
