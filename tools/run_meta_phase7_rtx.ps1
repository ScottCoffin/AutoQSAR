param(
    [string]$Python = "python",
    [string]$PlanCsv = "docs/meta_analysis/RTX_PHASE7_LEARNING_CURVE_PLAN.csv",
    [string]$RunRoot = "benchmark_results/qsarena_meta_phase7_gpu",
    [string]$PatchDir = "meta_phase7_gpu_patch",
    [string]$LogDir = "logs",
    [switch]$CommitAndPush
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PlanCsv)) {
    throw "Phase 7 plan CSV not found: $PlanCsv"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$plan = @(Import-Csv $PlanCsv)
if ($plan.Count -eq 0) {
    throw "Phase 7 plan CSV contains no rows: $PlanCsv"
}

foreach ($row in $plan) {
    $dataset = [string]$row.dataset
    $label = [string]$row.label
    $rowLimit = [int]$row.row_limit
    $runName = "${dataset}_${label}"
    $outputDir = Join-Path $RunRoot $runName
    $runLog = Join-Path $LogDir "meta_phase7_${runName}_$stamp.log"

    $runnerArgs = @(
        "portable_colab_qsar_bundle/run_qsarena_benchmarks.py",
        "--output-dir", $outputDir,
        "--benchmark-profile", "full",
        "--dataset-name", $dataset,
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

    if ($rowLimit -gt 0) {
        $runnerArgs += @("--row-limit", [string]$rowLimit)
    }

    Write-Host "Running Phase 7 learning-curve point: $dataset / $label. Log: $runLog"
    & $Python @runnerArgs *> $runLog
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
    Write-Host "Then commit/push when ready:"
    Write-Host "  git add $PatchDir"
    Write-Host "  git commit -m `"Add meta-analysis Phase 7 GPU metrics patch`""
    Write-Host "  git push"
}
