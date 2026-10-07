param([string[]]$Datasets = @(), [string]$Tag = "full")
# -Datasets accepts comma-separated names (a WMI/-File command line passes "a,b,c" as one string).
$Datasets = @($Datasets | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
# Nested-selection CV run on the manuscript run (2026-10-05, approved by the author): refit the stage-3 selector
# inside every CV fold (method pinned per dataset), patch the selected-feature members' CV metrics and OOF rows,
# then rebuild every ensemble once with XGBoost (ADMETboost features) as a member. No base model is retrained
# (--only-model-names Ensemble). TabPFN's fold refits use the LOCAL backend (API key variables cleared, so no
# credits): that gives TabPFN nested CV metrics, but TabPFN stays out of the ensembles because most of its full-fit
# predictions came from the Prior Labs API (a different backend from the fold refits).
Set-Location C:\Users\scott\AutoQSAR
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
foreach ($k in @("PRIORLABS_API_KEY", "TABPFN_API_KEY", "PRIORLABS_API_KEYS", "TABPFN_API_KEYS")) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
if ($Datasets.Count -eq 0) {
  $Datasets = Get-ChildItem -Path benchmark_results\qsarena_benchmark_oof_ensemble -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName "metrics.csv") } | Sort-Object Name | ForEach-Object { $_.Name }
}
$args_ = @()
foreach ($d in $Datasets) { $args_ += @("--dataset-name", $d) }
$args_ += @("--output-dir","benchmark_results/qsarena_benchmark_oof_ensemble","--benchmark-profile","full",
  "--run-admetboost-xgboost","--cv-selection","nested",
  "--only-model-names","Ensemble","--run-ensemble","--rebuild-ensemble",
  "--ensemble-exclude-model","TabPFNRegressor","--ensemble-exclude-model","TabPFNClassifier",
  "--ensemble-member-selection-split","oof","--ensemble-oof-folds","5","--ensemble-oof-scope","all",
  "--ensemble-oof-source-run","benchmark_results/autoqsar_benchmark_20260623_153839",
  "--run-tabpfn","--tabpfn-max-train-rows","11000","--unimol-batch-size","32","--unimol-max-atoms","64",
  "--run-chemprop-mpnn","--run-chemprop-dmpnn","--run-chemprop-rdkit2d","--run-chemprop-cmpnn",
  "--run-chemprop-attentivefp","--run-chemprop-selected-features","--chemprop-epochs","40",
  "--chemprop-ensemble-size","3","--chemprop-random-seed","42","--chemprop-num-workers","0",
  "--reuse-persistent-feature-store","--reuse-shared-feature-matrix-cache","--resume","--no-run-tdc22-multiseed-best")
$log = "logs\nested_selection_${Tag}_20261005.log"
cmd /c "C:\Users\scott\.conda\envs\autoqsar-py311\python.exe -u -m portable_colab_qsar_bundle.run_qsarena_benchmarks $($args_ | ForEach-Object { '"' + $_ + '"' }) > $log 2>&1"
Add-Content $log "LAUNCHER_DONE exit=$LASTEXITCODE"
