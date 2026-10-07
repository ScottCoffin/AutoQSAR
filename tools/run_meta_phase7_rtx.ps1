param(
    [string]$Python = "python",
    [string]$PlanCsv = "docs/meta_analysis/RTX_PHASE7_LEARNING_CURVE_PLAN.csv",
    [string]$InputDir = "data/meta_analysis/phase7_inputs",
    [string]$RunRoot = "benchmark_results/qsarena_meta_phase7_gpu",
    [string]$PatchDir = "meta_phase7_gpu_patch",
    [string]$LogDir = "logs",
    [switch]$CommitAndPush
)

# Phase 7 learning curves (qsarena/meta_analysis/phase7.py). Each curve point is a user CSV holding a
# nested training subset plus the dataset's FULL benchmark test set, run with a predefined split, so
# the test set is identical at every training size. (The runner's --row-limit would subsample before
# the split and change the test set at every point.)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PlanCsv)) {
    throw "Phase 7 plan CSV not found: $PlanCsv"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "Building fixed-test-set learning-curve inputs in $InputDir..."
& $Python -m qsarena.meta_analysis.phase7 --out-dir $InputDir
if ($LASTEXITCODE -ne 0) { throw "Phase 7 input build failed." }
$manifest = @(Import-Csv (Join-Path $InputDir "manifest.csv"))

$env:PYTHONUNBUFFERED = "1"
foreach ($row in $manifest) {
    $dataset = [string]$row.dataset
    $label = [string]$row.label
    $runName = "${dataset}_${label}"
    $outputDir = Join-Path $RunRoot $runName
    $runLog = Join-Path $LogDir "meta_phase7_${runName}_$stamp.log"

    $runnerArgs = @(
        "portable_colab_qsar_bundle/run_qsarena_benchmarks.py",
        "--output-dir", $outputDir,
        "--benchmark-profile", "full",
        "--dataset", [string]$row.path,
        "--smiles-col", "SMILES",
        "--target-col", "TARGET",
        "--task", [string]$row.task,
        "--target-transform", "raw",
        "--split-strategy", "predefined",
        "--predefined-split-col", "split",
        "--only-model-names", "Random forest",
        "--only-model-names", "XGBoost",
        "--only-model-names", "ChemML MLP (PyTorch)",
        "--only-model-names", "Uni-Mol V1",
        "--run-unimol-v1",
        "--no-run-unimol-v2",
        "--unimol-batch-size", "32",
        "--unimol-max-atoms", "64",
        "--run-chemml-pytorch",
        "--no-run-chemml-tensorflow",
        "--no-run-cnn",
        "--no-run-tabpfn",
        "--no-run-cfa",
        "--no-run-ensemble",
        "--no-run-chemprop-mpnn",
        "--no-run-chemprop-dmpnn",
        "--no-run-chemprop-rdkit2d",
        "--no-run-chemprop-cmpnn",
        "--no-run-chemprop-attentivefp",
        "--no-run-chemprop-selected-features",
        "--no-run-maplight-gnn",
        "--reuse-persistent-feature-store",
        "--reuse-shared-feature-matrix-cache",
        "--resume",
        "--no-run-tdc22-multiseed-best"
    )

    # cmd.exe redirection: PowerShell 5.1 `*>` with ErrorActionPreference=Stop makes the first stderr
    # line fatal (see tools/run_oof_ensemble_rtx.ps1).
    $quotedArgs = ($runnerArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    Write-Host "Phase 7 point: $dataset / $label (n_train=$($row.n_train), n_test=$($row.n_test)). Log: $runLog"
    cmd /c "`"`"$Python`" $quotedArgs > `"$runLog`" 2>&1`""
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Phase 7 run failed. Inspect $runLog"
        exit $LASTEXITCODE
    }
}

Write-Host "Exporting Phase 7 GPU metrics patch to $PatchDir..."
& $Python tools/export_meta_phase7_patch.py `
    --run-root $RunRoot `
    --manifest $PlanCsv `
    --patch-dir $PatchDir `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "Phase 7 patch export failed." }

if ($CommitAndPush) {
    git add $PatchDir
    git commit -m "Add meta-analysis Phase 7 GPU metrics patch"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }
    git push
    if ($LASTEXITCODE -ne 0) { throw "git push failed." }
} else {
    Write-Host "Patch exported. Review with: git status --short $PatchDir"
}
