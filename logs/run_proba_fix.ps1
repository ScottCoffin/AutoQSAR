# Retrain the 11 models that saved hard 0/1 labels on the four binary datasets catalogued as "rmse"
# (2026-10-04; fix in predict_values_for_metric). Same flags as the OOF run; ensembles are rebuilt later.
Set-Location C:\Users\scott\AutoQSAR
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
$models = @("AdaBoost","CatBoost","Extra trees","HistGradientBoosting","LogisticRegression","Random forest","SVC","Tabular MLP","Voting Classifier (KNN, SVM)","XGBoost","XGBoost (ADMETboost features)")
$args_ = @("--dataset-name","tdc_pampa_ncats","--dataset-name","tdc_cyp1a2_veith","--dataset-name","tdc_cyp2c19_veith","--dataset-name","tdc_herg_karim",
  "--output-dir","benchmark_results/qsarena_benchmark_oof_ensemble","--benchmark-profile","full","--run-admetboost-xgboost","--no-run-ensemble","--cv-selection","outer",
  "--ensemble-member-selection-split","oof","--ensemble-oof-folds","5","--ensemble-oof-scope","all",
  "--ensemble-oof-source-run","benchmark_results/autoqsar_benchmark_20260623_153839",
  "--run-tabpfn","--tabpfn-max-train-rows","11000","--unimol-batch-size","32","--unimol-max-atoms","64",
  "--run-chemprop-mpnn","--run-chemprop-dmpnn","--run-chemprop-rdkit2d","--run-chemprop-cmpnn",
  "--run-chemprop-attentivefp","--run-chemprop-selected-features","--chemprop-epochs","40",
  "--chemprop-ensemble-size","3","--chemprop-random-seed","42","--chemprop-num-workers","0",
  "--reuse-persistent-feature-store","--reuse-shared-feature-matrix-cache","--resume","--no-run-tdc22-multiseed-best")
foreach ($m in $models) { $args_ += @("--only-model-names", $m) }
cmd /c "C:\Users\scott\.conda\envs\autoqsar-py311\python.exe -m portable_colab_qsar_bundle.run_qsarena_benchmarks $($args_ | ForEach-Object { '"' + $_ + '"' }) > logs\proba_fix_20261004.log 2>&1"
Add-Content logs\proba_fix_20261004.log "LAUNCHER_DONE exit=$LASTEXITCODE"
