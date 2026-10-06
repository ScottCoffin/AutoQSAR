# Retrain TabPFNClassifier on the four binary datasets catalogued as "rmse" (2026-10-05). Its rows there still hold
# hard 0/1 labels from before the predict_values_for_metric fix. Uses the LOCAL tabpfn backend (v2.6 weights): the
# API key variables are cleared so no Prior Labs credits are spent. Ensembles are rebuilt by the nested run.
# Backups (metrics, predictions, run_status) and removal of the old TabPFN rows were done beforehand into
# benchmark_results\_proba_fix_backup_tabpfn_20261005.
Set-Location C:\Users\scott\AutoQSAR
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
foreach ($k in @("PRIORLABS_API_KEY", "TABPFN_API_KEY", "PRIORLABS_API_KEYS", "TABPFN_API_KEYS")) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
$args_ = @("--dataset-name","tdc_cyp1a2_veith","--dataset-name","tdc_cyp2c19_veith","--dataset-name","tdc_herg_karim",
  "--output-dir","benchmark_results/qsarena_benchmark_oof_ensemble","--benchmark-profile","full","--run-admetboost-xgboost","--no-run-ensemble","--cv-selection","outer",
  "--only-model-names","TabPFNClassifier",
  "--ensemble-member-selection-split","oof","--ensemble-oof-folds","5","--ensemble-oof-scope","all",
  "--ensemble-oof-source-run","benchmark_results/autoqsar_benchmark_20260623_153839",
  "--run-tabpfn","--tabpfn-max-train-rows","11000","--unimol-batch-size","32","--unimol-max-atoms","64",
  "--run-chemprop-mpnn","--run-chemprop-dmpnn","--run-chemprop-rdkit2d","--run-chemprop-cmpnn",
  "--run-chemprop-attentivefp","--run-chemprop-selected-features","--chemprop-epochs","40",
  "--chemprop-ensemble-size","3","--chemprop-random-seed","42","--chemprop-num-workers","0",
  "--reuse-persistent-feature-store","--reuse-shared-feature-matrix-cache","--resume","--no-run-tdc22-multiseed-best")
cmd /c "C:\Users\scott\.conda\envs\autoqsar-py311\python.exe -u -m portable_colab_qsar_bundle.run_qsarena_benchmarks $($args_ | ForEach-Object { '"' + $_ + '"' }) > logs\tabpfn_proba_fix_20261005.log 2>&1"
Add-Content logs\tabpfn_proba_fix_20261005.log "LAUNCHER_DONE exit=$LASTEXITCODE"
