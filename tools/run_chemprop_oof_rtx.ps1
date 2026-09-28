param(
    [string]$Python = "python",
    [string]$SeedRun = "chemprop_oof_seed",
    [string]$PatchDir = "chemprop_oof_patch",
    [string]$LogDir = "logs",
    [switch]$CommitAndPush
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = Join-Path $LogDir "chemprop_oof_rtx_$stamp.log"

if (-not (Test-Path $SeedRun)) {
    throw "Seed run not found: $SeedRun. Pull the branch that contains chemprop_oof_seed first."
}

$datasetArgs = @()
Get-ChildItem -Path $SeedRun -Directory | Where-Object { Test-Path (Join-Path $_.FullName "metrics.csv") } | Sort-Object Name | ForEach-Object {
    $datasetArgs += @("--dataset-name", $_.Name)
}

if ($datasetArgs.Count -eq 0) {
    throw "No dataset metrics.csv files found under $SeedRun."
}

Write-Host "Planning Chemprop OOF repair from $SeedRun..."
& $Python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py `
    $SeedRun --scope all --folds 5
if ($LASTEXITCODE -ne 0) { throw "Chemprop OOF planner failed." }

$runnerArgs = @(
    "portable_colab_qsar_bundle/run_qsarena_benchmarks.py",
    "--output-dir", $SeedRun,
    "--benchmark-profile", "full"
) + $datasetArgs + @(
    "--only-model-names", "Ensemble",
    "--run-ensemble",
    "--rebuild-ensemble",
    "--ensemble-member-selection-split", "oof",
    "--ensemble-oof-folds", "5",
    "--ensemble-oof-scope", "all",
    "--run-chemprop-mpnn",
    "--run-chemprop-dmpnn",
    "--run-chemprop-rdkit2d",
    "--run-chemprop-cmpnn",
    "--run-chemprop-attentivefp",
    "--run-chemprop-selected-features",
    "--chemprop-epochs", "40",
    "--chemprop-ensemble-size", "3",
    "--chemprop-random-seed", "42",
    # Windows spawns each DataLoader worker as a fresh process that loads the full torch/CUDA DLL
    # set (~720 MB); the default 4 workers exhausted the paging file (WinError 1455) and failed
    # every fold. Loading in-process is fine for these small molecular datasets.
    "--chemprop-num-workers", "0",
    "--reuse-persistent-feature-store",
    "--reuse-shared-feature-matrix-cache",
    "--resume",
    "--no-run-tdc22-multiseed-best"
)

Write-Host "Running Chemprop OOF repair. Log: $runLog"
# Redirect through cmd.exe: under Windows PowerShell 5.1 with ErrorActionPreference=Stop,
# `*>` turns the runner's first stderr line into a terminating NativeCommandError, and it
# writes UTF-16. cmd redirection keeps stderr non-fatal and the log UTF-8.
$env:PYTHONUNBUFFERED = "1"
$quotedArgs = ($runnerArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
cmd /c "`"`"$Python`" $quotedArgs > `"$runLog`" 2>&1`""
if ($LASTEXITCODE -ne 0) {
    Write-Host "Chemprop OOF run failed. Inspect $runLog"
    exit $LASTEXITCODE
}

# Chemprop occasionally dies with a native crash (e.g. exit 0xC0000409) on a fold that trains
# fine when rerun. Failed folds are not cached, so an identical pass retries only those folds.
$maxRetryPasses = 2
for ($pass = 1; $pass -le $maxRetryPasses; $pass++) {
    $prevLog = $runLog
    if (-not (Select-String -Path $prevLog -Pattern "out-of-fold refit failed" -Quiet)) { break }
    $runLog = Join-Path $LogDir "chemprop_oof_rtx_${stamp}_retry$pass.log"
    Write-Host "Refit failures in $prevLog; retry pass $pass. Log: $runLog"
    cmd /c "`"`"$Python`" $quotedArgs > `"$runLog`" 2>&1`""
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Chemprop OOF retry pass failed. Inspect $runLog"
        exit $LASTEXITCODE
    }
}
if (Select-String -Path $runLog -Pattern "out-of-fold refit failed" -Quiet) {
    Write-Host "WARNING: refit failures remain after $maxRetryPasses retry passes; see $runLog"
}

Write-Host "Exporting Chemprop OOF patch to $PatchDir..."
& $Python tools/export_chemprop_oof_patch.py --run-dir $SeedRun --patch-dir $PatchDir --overwrite
if ($LASTEXITCODE -ne 0) { throw "Patch export failed." }

Write-Host "Replanning after repair..."
& $Python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py `
    $SeedRun --scope all --folds 5
if ($LASTEXITCODE -ne 0) { throw "Post-repair planner failed." }

if ($CommitAndPush) {
    git add $PatchDir
    git commit -m "Add Chemprop OOF prediction patch"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
    git push
    if ($LASTEXITCODE -ne 0) { throw "git push failed." }
} else {
    Write-Host "Patch exported. Review with: git status --short $PatchDir"
    Write-Host "Then commit/push when ready:"
    Write-Host "  git add $PatchDir"
    Write-Host "  git commit -m `"Add Chemprop OOF prediction patch`""
    Write-Host "  git push"
}
