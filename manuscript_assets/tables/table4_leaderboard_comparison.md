| Dataset | Metric | QSARena best model (test-selected) | QSARena value | Reference top-1 | Reference top-10 cutoff | References (n) | Est. rank | CV-selected model | CV-selected value | CV-selected est. rank |
|---|---|---|---|---|---|---|---|---|---|---|
| tdc_bioavailability_ma | ROC_AUC | CatBoost | 0.777 | 0.748 | 0.640 | 8 | 1 | LogisticRegression | 0.715 | 3 |
| tdc_carcinogens_lagunin | ROC_AUC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.929 | 0.848 | 0.795 | 20 | 1 | Voting Classifier (KNN, SVM) | 0.837 | 5 |
| tdc_clearance_microsome_az | SPEARMAN | Uni-Mol V2 (84m) | 0.678 | 0.652 | 0.599 | 17 | 1 | XGBoost | 0.305 | >10 |
| tdc_cyp2c9_substrate_carbonmangels | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.466 | 0.450 | 0.360 | 6 | 1 | SVC | 0.301 | 7 |
| tdc_toxcast | ROC_AUC | HistGradientBoosting | 0.716 | 0.714 | 0.714 | 2 | 1 | LogisticRegression | 0.595 | 3 |
| poduam_pod_nc_std | RMSE | Random forest | 0.720 | 0.550 | 0.730 | 2 | 2 | ElasticNetCV | 0.792 | 3 |
| poduam_pod_rd_std | RMSE | Random forest | 0.572 | 0.410 | 0.630 | 2 | 2 | ElasticNetCV | 0.703 | 3 |
| tdc_ames | ROC_AUC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.876 | 0.912 | 0.834 | 7 | 2 | Extra trees | 0.871 | 2 |
| tdc_bbb_martins | ROC_AUC | Random forest | 0.925 | 0.941 | 0.903 | 7 | 2 | Tabular MLP | 0.898 | 8 |
| tdc_cyp2d6_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.734 | 0.811 | 0.464 | 6 | 2 | AdaBoost | 0.615 | 6 |
| tdc_cyp3a4_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.884 | 0.923 | 0.750 | 6 | 2 | AdaBoost | 0.798 | 6 |
| tdc_half_life_obach | SPEARMAN | Uni-Mol V1 | 0.597 | 0.649 | 0.485 | 16 | 2 | Random forest | 0.108 | >10 |
| tdc_solubility_aqsoldb | MAE | Uni-Mol V1 | 0.710 | 0.557 | 0.776 | 17 | 2 | XGBoost | 0.744 | 6 |
| polaris_adme_fang_hppb_1 | MSE | MapLight + GNN (CatBoost, Strict Parity) | 0.201 | 0.143 | 0.383 | 10 | 3 | ElasticNetCV | 0.300 | 8 |
| polaris_adme_fang_perm_1 | MSE | Uni-Mol V1 | 0.159 | 0.113 | 0.257 | 10 | 3 | ChemML MLP (PyTorch) | 0.212 | 7 |
| polaris_adme_fang_rppb_1 | MSE | MapLight + GNN (CatBoost, Strict Parity) | 0.243 | 0.230 | 0.634 | 10 | 3 | ElasticNetCV | 0.361 | 5 |
| tdc_caco2_wang | MAE | MapLight CatBoost (Strict Parity) | 0.272 | 0.256 | 0.288 | 20 | 3 | ElasticNetCV | 0.345 | >10 |
| tdc_clearance_hepatocyte_az | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.516 | 0.633 | 0.440 | 16 | 3 | CatBoost | 0.224 | >10 |
| tdc_clintox | ROC_AUC | CFA (Combinatorial Fusion) | 0.974 | 0.996 | 0.889 | 12 | 3 | LogisticRegression | 0.896 | 5 |
| tdc_cyp2c9_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.814 | 0.877 | 0.770 | 6 | 3 | AdaBoost | 0.705 | 7 |
| tdc_ppbr_az | MAE | MapLight + GNN (CatBoost, Strict Parity) | 7.306 | 0.679 | 7.914 | 16 | 3 | ChemML MLP (PyTorch) | 9.741 | >10 |
| polaris_adme_fang_rclint_1 | MSE | Uni-Mol V1 | 0.273 | 0.216 | 0.403 | 10 | 4 | ElasticNetCV | 0.365 | 10 |
| tdc_dili | ROC_AUC | Uni-Mol V1 | 0.922 | 0.945 | 0.852 | 6 | 4 | LogisticRegression | 0.745 | 7 |
| tdc_herg | ROC_AUC | AdaBoost | 0.848 | 0.880 | 0.806 | 7 | 4 | LogisticRegression | 0.709 | 8 |
| tdc_hia_hou | ROC_AUC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.988 | 0.994 | 0.976 | 8 | 4 | LogisticRegression | 0.983 | 7 |
| tdc_pgp_broccatelli | ROC_AUC | Uni-Mol V2 (84m) | 0.933 | 0.994 | 0.911 | 8 | 4 | LogisticRegression | 0.868 | 9 |
| polaris_adme_fang_solu_1 | MSE | Uni-Mol V2 (84m) | 0.304 | 0.222 | 0.329 | 10 | 5 | ChemML MLP (PyTorch) | 0.410 | >10 |
| tdc_cyp2d6_substrate_carbonmangels | AUPRC | Uni-Mol V1 | 0.652 | 0.766 | 0.570 | 7 | 5 | AdaBoost | 0.620 | 7 |
| tdc_ld50_zhu | MAE | Ensemble (Weighted average (inverse train RMSE)) | 0.575 | 0.292 | 0.605 | 16 | 5 | XGBoost | 0.588 | 8 |
| tdc_vdss_lombardo | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.680 | 0.942 | 0.582 | 16 | 5 | Extra trees | 0.368 | >10 |
| esol_delaney | RMSE | Ensemble (Weighted average (inverse train RMSE)) | 0.617 | 0.558 | 0.743 | 30 | 6 | ElasticNetCV | 0.726 | 9 |
| lipophilicity | RMSE | Ensemble (Weighted average (inverse train RMSE)) | 0.589 | 0.549 | 0.610 | 27 | 7 | ElasticNetCV | 0.682 | >10 |
| tdc_cyp3a4_substrate_carbonmangels | ROC_AUC | Voting Classifier (KNN, SVM) | 0.655 | 0.692 | 0.651 | 7 | 7 | LogisticRegression | 0.624 | 8 |
| tdc_lipophilicity_astrazeneca | MAE | CFA (Combinatorial Fusion) | 0.470 | 0.406 | 0.515 | 17 | 7 | ChemML MLP (PyTorch) | 0.542 | >10 |
| tdc_hydrationfreeenergy_freesolv | RMSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 1.112 | 0.654 | 1.211 | 12 | 9 | ElasticNetCV | 1.321 | >10 |
| tdc_skin_reaction | ROC_AUC | Uni-Mol V2 (84m) | 0.658 | 0.741 | 0.677 | 21 | >10 | Voting Classifier (KNN, SVM) | 0.503 | >10 |
| tdc_tox21 | ROC_AUC | AdaBoost | 0.556 | 0.867 | 0.840 | 12 | >10 | SVC | 0.467 | >10 |
