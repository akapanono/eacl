param(
    [string]$Dataset = "IEMOCAP",
    [string]$BertPath = ".\pretrained\sup-simcse-roberta-large",
    [string]$PythonExe = "",
    [int]$PhysicalGpu = 1,
    [int[]]$Seeds = @(1, 2, 3, 4, 5),
    [int[]]$NumSubanchors = @(2, 3, 4),
    [string[]]$Poolings = @("max", "logsumexp"),
    [string]$SweepRoot = ".\saved_models\sweeps",
    [double]$PrototypeMomentum = 0.9,
    [double]$CeLossWeight = 0.1,
    [double]$AngleLossWeight = 1.0,
    [double]$Temp = 0.5,
    [double]$StageTwoLr = 1e-4
)

$ErrorActionPreference = "Stop"
$pythonExe = $null

if ($PythonExe) {
    $resolvedPython = Resolve-Path $PythonExe -ErrorAction Stop
    $pythonExe = $resolvedPython.Path
}

if (-not $pythonExe -and $env:CONDA_PREFIX) {
    $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $candidate) {
        $pythonExe = $candidate
    }
}

if (-not $pythonExe) {
    $pythonExe = (Get-Command python).Source
}

function Get-LeafName([string]$PathValue) {
    return Split-Path -Path $PathValue -Leaf
}

function Read-Metric([string]$LogPath, [string]$Prefix) {
    if (-not (Test-Path $LogPath)) {
        return ""
    }

    $matches = Select-String -Path $LogPath -Pattern $Prefix | Select-Object -ExpandProperty Line
    if (-not $matches) {
        return ""
    }

    $lastLine = $matches[-1]
    if ($lastLine -match "([0-9]+\.[0-9]+|[0-9]+)$") {
        return $Matches[1]
    }
    return ""
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$modelLeaf = Get-LeafName $BertPath
$anchorRoot = Join-Path ".\emo_anchors" $modelLeaf
$datasetSweepRoot = Join-Path $SweepRoot $Dataset
New-Item -ItemType Directory -Force -Path $datasetSweepRoot | Out-Null

$summaryPath = Join-Path $datasetSweepRoot "summary.csv"
$records = @()

$env:CUDA_VISIBLE_DEVICES = "$PhysicalGpu"

Write-Host "Using physical GPU $PhysicalGpu"
Write-Host "Dataset: $Dataset"
Write-Host "Backbone: $BertPath"
Write-Host "Sweep root: $datasetSweepRoot"
Write-Host "Python: $pythonExe"

foreach ($numSubanchor in $NumSubanchors) {
    Write-Host ""
    Write-Host "Generating anchors for num_subanchors=$numSubanchor"
    & $pythonExe src\generate_anchors.py --bert_path $BertPath --num_subanchors $numSubanchor

    foreach ($seed in $Seeds) {
        foreach ($pooling in $Poolings) {
            $runName = "seed_${seed}__sub_${numSubanchor}__pool_${pooling}"
            $runRoot = Join-Path $datasetSweepRoot $runName
            $stdoutPath = Join-Path $runRoot "train.stdout.log"
            $savePath = "$runRoot\"

            New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

            Write-Host ""
            Write-Host "Running $runName"

            & $pythonExe src\run.py `
                --anchor_path $anchorRoot `
                --bert_path $BertPath `
                --dataset_name $Dataset `
                --ce_loss_weight $CeLossWeight `
                --temp $Temp `
                --seed $seed `
                --angle_loss_weight $AngleLossWeight `
                --stage_two_lr $StageTwoLr `
                --disable_training_progress_bar `
                --use_nearest_neighbour `
                --num_subanchors $numSubanchor `
                --prototype_momentum $PrototypeMomentum `
                --prototype_pooling $pooling `
                --gpu_id 0 `
                --save_path $savePath 2>&1 | Tee-Object -FilePath $stdoutPath

            $logPath = Join-Path $runRoot "$Dataset\logging.log"
            $bestVal = Read-Metric $logPath "Best F-Score based on validation:"
            $bestTest = Read-Metric $logPath "Best F-Score based on test:"

            $record = [PSCustomObject]@{
                dataset = $Dataset
                seed = $seed
                num_subanchors = $numSubanchor
                pooling = $pooling
                prototype_momentum = $PrototypeMomentum
                best_validation_f1 = $bestVal
                best_test_f1 = $bestTest
                run_dir = (Resolve-Path $runRoot).Path
                log_path = if (Test-Path $logPath) { (Resolve-Path $logPath).Path } else { "" }
            }
            $records += $record
            $records | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
        }
    }
}

Write-Host ""
Write-Host "Sweep complete."
Write-Host "Summary saved to: $(Resolve-Path $summaryPath)"
