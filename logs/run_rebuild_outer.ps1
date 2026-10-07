# Rebuild CFA + OOF ensembles (outer selection) with XGBoost (ADMETboost features) as a member and the
# probability-fixed models (2026-10-05). No base model is retrained.
Set-Location C:\Users\scott\AutoQSAR
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
$args_ = @("--dataset-name", "chemml_cep_homo", "--dataset-name", "chemml_organic_density", "--dataset-name", "esol_delaney", "--dataset-name", "freesolv_sampl", "--dataset-name", "lipophilicity", "--dataset-name", "poduam_pod_nc_std", "--dataset-name", "poduam_pod_rd_std", "--dataset-name", "polaris_adme_fang_hppb_1", "--dataset-name", "polaris_adme_fang_perm_1", "--dataset-name", "polaris_adme_fang_rclint_1", "--dataset-name", "polaris_adme_fang_rppb_1", "--dataset-name", "polaris_adme_fang_solu_1", "--dataset-name", "tdc_ames", "--dataset-name", "tdc_bbb_martins", "--dataset-name", "tdc_bioavailability_ma", "--dataset-name", "tdc_caco2_wang", "--dataset-name", "tdc_carcinogens_lagunin", "--dataset-name", "tdc_clearance_hepatocyte_az", "--dataset-name", "tdc_clearance_microsome_az", "--dataset-name", "tdc_clintox", "--dataset-name", "tdc_cyp1a2_veith", "--dataset-name", "tdc_cyp2c19_veith", "--dataset-name", "tdc_cyp2c9_substrate_carbonmangels", "--dataset-name", "tdc_cyp2c9_veith", "--dataset-name", "tdc_cyp2d6_substrate_carbonmangels", "--dataset-name", "tdc_cyp2d6_veith", "--dataset-name", "tdc_cyp3a4_substrate_carbonmangels", "--dataset-name", "tdc_cyp3a4_veith", "--dataset-name", "tdc_dili", "--dataset-name", "tdc_half_life_obach", "--dataset-name", "tdc_herg", "--dataset-name", "tdc_herg_karim", "--dataset-name", "tdc_hia_hou", "--dataset-name", "tdc_hydrationfreeenergy_freesolv", "--dataset-name", "tdc_ld50_zhu", "--dataset-name", "tdc_lipophilicity_astrazeneca", "--dataset-name", "tdc_pampa_ncats", "--dataset-name", "tdc_pgp_broccatelli", "--dataset-name", "tdc_ppbr_az", "--dataset-name", "tdc_skin_reaction", "--dataset-name", "tdc_solubility_aqsoldb", "--dataset-name", "tdc_tox21", "--dataset-name", "tdc_toxcast", "--dataset-name", "tdc_vdss_lombardo",  "--output-dir", "benchmark_results/qsarena_benchmark_oof_ensemble", "--benchmark-profile", "full",
  "--run-admetboost-xgboost", "--cv-selection", "outer",
  "--only-model-names", "Ensemble", "--only-model-names", "CFA (Combinatorial Fusion)", "--run-ensemble", "--rebuild-ensemble",
  "--ensemble-member-selection-split", "oof", "--ensemble-oof-folds", "5", "--ensemble-oof-scope", "all",
  "--ensemble-oof-source-run", "benchmark_results/autoqsar_benchmark_20260623_153839",
  "--run-tabpfn", "--tabpfn-max-train-rows", "11000", "--unimol-batch-size", "32", "--unimol-max-atoms", "64",
  "--run-chemprop-mpnn", "--run-chemprop-dmpnn", "--run-chemprop-rdkit2d", "--run-chemprop-cmpnn",
  "--run-chemprop-attentivefp", "--run-chemprop-selected-features", "--chemprop-epochs", "40",
  "--chemprop-ensemble-size", "3", "--chemprop-random-seed", "42", "--chemprop-num-workers", "0",
  "--reuse-persistent-feature-store", "--reuse-shared-feature-matrix-cache", "--resume", "--no-run-tdc22-multiseed-best")
cmd /c "C:\Users\scott\.conda\envs\autoqsar-py311\python.exe -u -m portable_colab_qsar_bundle.run_qsarena_benchmarks $($args_ | ForEach-Object { '"' + $_ + '"' }) > logs\rebuild_outer_20261005.log 2>&1"
Add-Content logs\rebuild_outer_20261005.log "LAUNCHER_DONE exit=$LASTEXITCODE"
