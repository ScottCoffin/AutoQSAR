param(
    [string]$Python = "python",
    [string]$RunDir = "benchmark_results/qsarena_benchmark_oof_ensemble",
    [string]$SourceRun = "benchmark_results/autoqsar_benchmark_20260623_153839",
    [ValidateSet("cpu", "all")]
    [string]$Scope = "all",
    [string[]]$Datasets = @(),
    [string]$LogDir = "logs",
    [int]$MaxRetryPasses = 2
)

# Rebuild the OOF ensembles directly in the full run (AGENTS.md step 5/6) on the RTX laptop.
# Needs the gitignored predictions.csv files (docs/HANDOFF_SSD_TRANSFER.md). With -Scope all,
# only Chemprop fold models are trained; every other member reads saved OOF predictions.

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = Join-Path $LogDir "oof_ensemble_rtx_$stamp.log"

if ($Datasets.Count -eq 0) {
    $Datasets = Get-ChildItem -Path $RunDir -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "metrics.csv") } |
        Sort-Object Name | ForEach-Object { $_.Name }
}
$datasetArgs = @()
foreach ($name in $Datasets) { $datasetArgs += @("--dataset-name", $name) }

$runnerArgs = @(
    "portable_colab_qsar_bundle/run_qsarena_benchmarks.py",
    "--output-dir", $RunDir,
    "--benchmark-profile", "full"
) + $datasetArgs + @(
    "--only-model-names", "Ensemble",
    "--run-ensemble",
    "--rebuild-ensemble",
    "--ensemble-member-selection-split", "oof",
    "--ensemble-oof-folds", "5",
    "--ensemble-oof-scope", $Scope,
    "--ensemble-oof-source-run", $SourceRun,
    "--run-tabpfn", "--tabpfn-max-train-rows", "11000",
    "--unimol-batch-size", "32", "--unimol-max-atoms", "64",
    "--run-chemprop-mpnn", "--run-chemprop-dmpnn", "--run-chemprop-rdkit2d",
    "--run-chemprop-cmpnn", "--run-chemprop-attentivefp", "--run-chemprop-selected-features",
    "--chemprop-epochs", "40", "--chemprop-ensemble-size", "3", "--chemprop-random-seed", "42",
    # Windows: DataLoader workers each reload torch/CUDA and exhaust the paging file.
    "--chemprop-num-workers", "0",
    "--reuse-persistent-feature-store", "--reuse-shared-feature-matrix-cache",
    "--resume", "--no-run-tdc22-multiseed-best"
)

# cmd.exe redirection: PowerShell 5.1 `*>` with ErrorActionPreference=Stop makes the first
# stderr line fatal and writes UTF-16.
$env:PYTHONUNBUFFERED = "1"
$quotedArgs = ($runnerArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '

Write-Host "OOF ensemble rebuild ($Scope scope, $($Datasets.Count) datasets). Log: $runLog"
cmd /c "`"`"$Python`" $quotedArgs > `"$runLog`" 2>&1`""
if ($LASTEXITCODE -ne 0) { Write-Host "Run failed. Inspect $runLog"; exit $LASTEXITCODE }

# Intermittent native Chemprop crashes (0xC0000409): failed folds are not cached, so an
# identical pass retries only those folds.
for ($pass = 1; $pass -le $MaxRetryPasses; $pass++) {
    if (-not (Select-String -Path $runLog -Pattern "out-of-fold refit failed" -Quiet)) { break }
    $prevLog = $runLog
    $runLog = Join-Path $LogDir "oof_ensemble_rtx_${stamp}_retry$pass.log"
    Write-Host "Refit failures in $prevLog; retry pass $pass. Log: $runLog"
    cmd /c "`"`"$Python`" $quotedArgs > `"$runLog`" 2>&1`""
    if ($LASTEXITCODE -ne 0) { Write-Host "Retry pass failed. Inspect $runLog"; exit $LASTEXITCODE }
}
if (Select-String -Path $runLog -Pattern "out-of-fold refit failed" -Quiet) {
    Write-Host "WARNING: refit failures remain after $MaxRetryPasses retry passes; see $runLog"
}

Write-Host "Replanning after rebuild..."
& $Python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py $RunDir --scope $Scope --folds 5 --source-run $SourceRun
Write-Host "Done. Last log: $runLog"
