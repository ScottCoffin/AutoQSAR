| Dataset | Metric | AutoQSAR best model (test-selected) | AutoQSAR value | Reference top-1 | Reference top-10 cutoff | References (n) | Est. rank | CV-selected model | CV-selected value | CV-selected est. rank |
|---|---|---|---|---|---|---|---|---|---|---|
| tdc_bioavailability_ma | ROC_AUC | CatBoost | 0.777 | 0.748 | 0.640 | 8 | 1 | LogisticRegression | 0.715 | 3 |
| tdc_carcinogens_lagunin | ROC_AUC | AdaBoost | 0.863 | 0.848 | 0.795 | 20 | 1 | LogisticRegression | 0.830 | 5 |
| tdc_cyp2c9_substrate_carbonmangels | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.482 | 0.450 | 0.360 | 6 | 1 | SVC | 0.301 | 7 |
| tdc_hydrationfreeenergy_freesolv | RMSE | CFA (Combinatorial Fusion) | 0.556 | 0.654 | 1.211 | 12 | 1 | ElasticNetCV | 1.352 | >10 |
| tdc_skin_reaction | ROC_AUC | CFA (Combinatorial Fusion) | 0.769 | 0.741 | 0.677 | 21 | 1 | LogisticRegression | 0.734 | 3 |
| tdc_toxcast | ROC_AUC | CatBoost | 0.791 | 0.714 | 0.714 | 2 | 1 | CatBoost | 0.791 | 1 |
| poduam_pod_nc_std | RMSE | Ensemble (Weighted average (inverse train RMSE)) | 0.699 | 0.550 | 0.730 | 2 | 2 | XGBoost | 0.735 | 3 |
| poduam_pod_rd_std | RMSE | XGBoost | 0.551 | 0.410 | 0.630 | 2 | 2 | ElasticNetCV | 0.713 | 3 |
| tdc_ames | ROC_AUC | XGBoost | 0.875 | 0.912 | 0.834 | 7 | 2 | Extra trees | 0.868 | 2 |
| tdc_bbb_martins | ROC_AUC | Random forest | 0.932 | 0.941 | 0.903 | 7 | 2 | Tabular MLP | 0.885 | 8 |
| tdc_cyp2d6_veith | AUPRC | XGBoost | 0.722 | 0.811 | 0.464 | 6 | 2 | AdaBoost | 0.615 | 6 |
| tdc_cyp3a4_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.890 | 0.923 | 0.750 | 6 | 2 | AdaBoost | 0.797 | 6 |
| tdc_solubility_aqsoldb | MAE | Uni-Mol V1 | 0.699 | 0.557 | 0.776 | 17 | 2 | Extra trees | 0.761 | 8 |
| polaris_adme_fang_hppb_1 | MSE | MapLight + GNN (CatBoost, Strict Parity) | 0.202 | 0.143 | 0.383 | 10 | 3 | ElasticNetCV | 0.272 | 7 |
| polaris_adme_fang_perm_1 | MSE | Uni-Mol V1 | 0.159 | 0.113 | 0.257 | 10 | 3 | ChemML MLP (PyTorch) | 0.224 | 7 |
| polaris_adme_fang_rppb_1 | MSE | MapLight + GNN (CatBoost, Strict Parity) | 0.244 | 0.230 | 0.634 | 10 | 3 | TabPFNRegressor | 0.284 | 4 |
| tdc_caco2_wang | MAE | Ensemble (Weighted average (inverse train RMSE)) | 0.269 | 0.256 | 0.288 | 20 | 3 | ElasticNetCV | 0.302 | >10 |
| tdc_clearance_hepatocyte_az | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.517 | 0.633 | 0.440 | 16 | 3 | TabPFNRegressor | 0.345 | >10 |
| tdc_clintox | ROC_AUC | CatBoost | 0.949 | 0.996 | 0.889 | 12 | 3 | LogisticRegression | 0.915 | 3 |
| tdc_cyp2c9_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.810 | 0.877 | 0.770 | 6 | 3 | AdaBoost | 0.705 | 7 |
| tdc_herg | ROC_AUC | AdaBoost | 0.861 | 0.880 | 0.806 | 7 | 3 | Tabular MLP | 0.724 | 8 |
| tdc_ppbr_az | MAE | Ensemble (Weighted average (inverse train RMSE)) | 7.315 | 0.679 | 7.914 | 16 | 3 | ChemML MLP (PyTorch) | 9.372 | >10 |
| esol_delaney | RMSE | Ensemble (Weighted average (inverse train RMSE)) | 0.592 | 0.558 | 0.743 | 30 | 4 | ElasticNetCV | 0.680 | 9 |
| polaris_adme_fang_rclint_1 | MSE | Uni-Mol V1 | 0.262 | 0.216 | 0.403 | 10 | 4 | ChemML MLP (PyTorch) | 0.333 | 7 |
| tdc_hia_hou | ROC_AUC | CFA (Combinatorial Fusion) | 0.990 | 0.994 | 0.976 | 8 | 4 | LogisticRegression | 0.987 | 4 |
| tdc_lipophilicity_astrazeneca | MAE | Uni-Mol V1 | 0.451 | 0.406 | 0.515 | 17 | 4 | ElasticNetCV | 0.559 | >10 |
| tdc_cyp2d6_substrate_carbonmangels | AUPRC | Ensemble (Weighted average (inverse train RMSE)) | 0.673 | 0.766 | 0.570 | 7 | 5 | AdaBoost | 0.652 | 5 |
| tdc_dili | ROC_AUC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.915 | 0.945 | 0.852 | 6 | 5 | LogisticRegression | 0.860 | 6 |
| tdc_ld50_zhu | MAE | CFA (Combinatorial Fusion) | 0.577 | 0.292 | 0.605 | 16 | 5 | XGBoost | 0.584 | 5 |
| tdc_pgp_broccatelli | ROC_AUC | Ensemble (Weighted average (inverse train RMSE)) | 0.929 | 0.994 | 0.911 | 8 | 5 | Tabular MLP | 0.906 | 9 |
| tdc_vdss_lombardo | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.678 | 0.942 | 0.582 | 16 | 5 | TabPFNRegressor | 0.527 | >10 |
| lipophilicity | RMSE | Ensemble (Weighted average (inverse train RMSE)) | 0.582 | 0.549 | 0.610 | 27 | 6 | ElasticNetCV | 0.658 | >10 |
| tdc_cyp3a4_substrate_carbonmangels | ROC_AUC | LogisticRegression | 0.659 | 0.692 | 0.651 | 7 | 7 | Tabular MLP | 0.634 | 8 |
| tdc_half_life_obach | SPEARMAN | Uni-Mol V1 | 0.533 | 0.649 | 0.485 | 16 | 9 | TabPFNRegressor | 0.396 | >10 |
| tdc_clearance_microsome_az | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.606 | 0.652 | 0.599 | 17 | 10 | Tabular MLP | 0.500 | >10 |
| polaris_adme_fang_solu_1 | MSE | Uni-Mol V1 | 0.331 | 0.222 | 0.329 | 10 | >10 | ChemML MLP (PyTorch) | 0.417 | >10 |
| tdc_tox21 | ROC_AUC | XGBoost | 0.812 | 0.867 | 0.840 | 12 | >10 | SVC | 0.789 | >10 |
