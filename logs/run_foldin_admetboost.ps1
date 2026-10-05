# Fold XGBoost (ADMETboost features) into benchmark_results/qsarena_benchmark_oof_ensemble (2026-10-03, approved).
# Same flags as tools/run_oof_ensemble_rtx.ps1 so every stage signature matches; trains ONLY the new model.
# Ensembles are rebuilt later (after the nested-selection decision), hence --no-run-ensemble.
Set-Location C:\Users\scott\AutoQSAR
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
while (-not (Select-String -Path logs\selection_leak_20261003.log -Pattern "^EXIT" -Quiet)) { Start-Sleep -Seconds 60 }
$args_ = @("--dataset-name", "chemml_cep_homo", "--dataset-name", "chemml_organic_density", "--dataset-name", "esol_delaney", "--dataset-name", "freesolv_sampl", "--dataset-name", "lipophilicity", "--dataset-name", "poduam_pod_nc_std", "--dataset-name", "poduam_pod_rd_std", "--dataset-name", "polaris_adme_fang_hppb_1", "--dataset-name", "polaris_adme_fang_perm_1", "--dataset-name", "polaris_adme_fang_rclint_1", "--dataset-name", "polaris_adme_fang_rppb_1", "--dataset-name", "polaris_adme_fang_solu_1", "--dataset-name", "tdc_ames", "--dataset-name", "tdc_bbb_martins", "--dataset-name", "tdc_bioavailability_ma", "--dataset-name", "tdc_caco2_wang", "--dataset-name", "tdc_carcinogens_lagunin", "--dataset-name", "tdc_clearance_hepatocyte_az", "--dataset-name", "tdc_clearance_microsome_az", "--dataset-name", "tdc_clintox", "--dataset-name", "tdc_cyp1a2_veith", "--dataset-name", "tdc_cyp2c19_veith", "--dataset-name", "tdc_cyp2c9_substrate_carbonmangels", "--dataset-name", "tdc_cyp2c9_veith", "--dataset-name", "tdc_cyp2d6_substrate_carbonmangels", "--dataset-name", "tdc_cyp2d6_veith", "--dataset-name", "tdc_cyp3a4_substrate_carbonmangels", "--dataset-name", "tdc_cyp3a4_veith", "--dataset-name", "tdc_dili", "--dataset-name", "tdc_half_life_obach", "--dataset-name", "tdc_herg", "--dataset-name", "tdc_herg_karim", "--dataset-name", "tdc_hia_hou", "--dataset-name", "tdc_hydrationfreeenergy_freesolv", "--dataset-name", "tdc_ld50_zhu", "--dataset-name", "tdc_lipophilicity_astrazeneca", "--dataset-name", "tdc_pampa_ncats", "--dataset-name", "tdc_pgp_broccatelli", "--dataset-name", "tdc_ppbr_az", "--dataset-name", "tdc_skin_reaction", "--dataset-name", "tdc_solubility_aqsoldb", "--dataset-name", "tdc_tox21", "--dataset-name", "tdc_toxcast", "--dataset-name", "tdc_vdss_lombardo",  "--output-dir", "benchmark_results/qsarena_benchmark_oof_ensemble", "--benchmark-profile", "full",
  "--run-admetboost-xgboost", "--only-model-names", "XGBoost (ADMETboost features)", "--no-run-ensemble", "--cv-selection", "outer",
  "--ensemble-member-selection-split", "oof", "--ensemble-oof-folds", "5", "--ensemble-oof-scope", "all",
  "--ensemble-oof-source-run", "benchmark_results/autoqsar_benchmark_20260623_153839",
  "--run-tabpfn", "--tabpfn-max-train-rows", "11000", "--unimol-batch-size", "32", "--unimol-max-atoms", "64",
  "--run-chemprop-mpnn", "--run-chemprop-dmpnn", "--run-chemprop-rdkit2d", "--run-chemprop-cmpnn",
  "--run-chemprop-attentivefp", "--run-chemprop-selected-features", "--chemprop-epochs", "40",
  "--chemprop-ensemble-size", "3", "--chemprop-random-seed", "42", "--chemprop-num-workers", "0",
  "--reuse-persistent-feature-store", "--reuse-shared-feature-matrix-cache", "--resume", "--no-run-tdc22-multiseed-best")
cmd /c "C:\Users\scott\.conda\envs\autoqsar-py311\python.exe -m portable_colab_qsar_bundle.run_qsarena_benchmarks $($args_ | ForEach-Object { '"' + $_ + '"' }) > logs\foldin_admetboost_20261003.log 2>&1"
Add-Content logs\foldin_admetboost_20261003.log "LAUNCHER_DONE exit=$LASTEXITCODE"
