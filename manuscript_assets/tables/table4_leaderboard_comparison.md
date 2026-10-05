| Dataset | Metric | QSARena best model (test-selected) | QSARena value | Reference top-1 | Reference top-10 cutoff | References (n) | Est. rank | CV-selected model | CV-selected value | CV-selected est. rank |
|---|---|---|---|---|---|---|---|---|---|---|
| lipophilicity | RMSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.548 | 0.549 | 0.610 | 27 | 1 | ElasticNetCV | 0.682 | >10 |
| tdc_bioavailability_ma | ROC_AUC | CatBoost | 0.777 | 0.748 | 0.640 | 8 | 1 | TabPFNClassifier | 0.732 | 2 |
| tdc_carcinogens_lagunin | ROC_AUC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.929 | 0.848 | 0.795 | 20 | 1 | TabPFNClassifier | 0.870 | 1 |
| tdc_clearance_microsome_az | SPEARMAN | Uni-Mol V2 (84m) | 0.678 | 0.652 | 0.599 | 17 | 1 | TabPFNRegressor | 0.437 | >10 |
| tdc_toxcast | ROC_AUC | HistGradientBoosting | 0.716 | 0.714 | 0.714 | 2 | 1 | TabPFNClassifier | 0.656 | 3 |
| poduam_pod_nc_std | RMSE | Random forest | 0.720 | 0.550 | 0.730 | 2 | 2 | TabPFNRegressor | 0.745 | 3 |
| poduam_pod_rd_std | RMSE | CFA (Combinatorial Fusion) | 0.564 | 0.410 | 0.630 | 2 | 2 | TabPFNRegressor | 0.589 | 2 |
| tdc_ames | ROC_AUC | CFA (Combinatorial Fusion) | 0.877 | 0.912 | 0.834 | 7 | 2 | TabPFNClassifier | 0.876 | 2 |
| tdc_bbb_martins | ROC_AUC | Random forest | 0.925 | 0.941 | 0.903 | 7 | 2 | TabPFNClassifier | 0.909 | 4 |
| tdc_cyp2d6_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.759 | 0.811 | 0.464 | 6 | 2 | TabPFNClassifier | 0.457 | 7 |
| tdc_cyp3a4_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.902 | 0.923 | 0.750 | 6 | 2 | TabPFNClassifier | 0.740 | 7 |
| tdc_half_life_obach | SPEARMAN | Uni-Mol V1 | 0.597 | 0.649 | 0.485 | 16 | 2 | XGBoost (ADMETboost features) | 0.473 | >10 |
| tdc_hia_hou | ROC_AUC | XGBoost (ADMETboost features) | 0.994 | 0.994 | 0.976 | 8 | 2 | LogisticRegression | 0.983 | 7 |
| tdc_ld50_zhu | MAE | TabPFNRegressor | 0.552 | 0.292 | 0.605 | 16 | 2 | TabPFNRegressor | 0.552 | 2 |
| tdc_solubility_aqsoldb | MAE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.678 | 0.557 | 0.776 | 17 | 2 | TabPFNRegressor | 0.686 | 2 |
| polaris_adme_fang_rppb_1 | MSE | MapLight + GNN (CatBoost, Strict Parity) | 0.243 | 0.230 | 0.634 | 10 | 3 | TabPFNRegressor | 0.308 | 5 |
| tdc_caco2_wang | MAE | MapLight CatBoost (Strict Parity) | 0.272 | 0.256 | 0.288 | 20 | 3 | TabPFNRegressor | 0.302 | >10 |
| tdc_clearance_hepatocyte_az | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.516 | 0.633 | 0.440 | 16 | 3 | TabPFNRegressor | 0.186 | >10 |
| tdc_clintox | ROC_AUC | CFA (Combinatorial Fusion) | 0.974 | 0.996 | 0.889 | 12 | 3 | TabPFNClassifier | 0.923 | 3 |
| tdc_cyp2c9_substrate_carbonmangels | AUPRC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.438 | 0.450 | 0.360 | 6 | 3 | SVC | 0.301 | 7 |
| tdc_cyp2c9_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.827 | 0.877 | 0.770 | 6 | 3 | TabPFNClassifier | 0.530 | 7 |
| tdc_herg | ROC_AUC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.857 | 0.880 | 0.806 | 7 | 3 | TabPFNClassifier | 0.732 | 8 |
| tdc_lipophilicity_astrazeneca | MAE | Chemprop v2 (D-MPNN, ensemble=3) | 0.425 | 0.406 | 0.515 | 17 | 3 | TabPFNRegressor | 0.439 | 4 |
| tdc_ppbr_az | MAE | MapLight + GNN (CatBoost, Strict Parity) | 7.306 | 0.679 | 7.914 | 16 | 3 | ChemML MLP (PyTorch) | 9.741 | >10 |
| polaris_adme_fang_hppb_1 | MSE | Chemprop v2 (D-MPNN, ensemble=3) | 0.197 | 0.143 | 0.303 | 10 | 4 | ElasticNetCV | 0.300 | 10 |
| polaris_adme_fang_perm_1 | MSE | Uni-Mol V1 | 0.159 | 0.113 | 0.239 | 10 | 4 | ChemML MLP (PyTorch) | 0.212 | 9 |
| tdc_cyp2d6_substrate_carbonmangels | AUPRC | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | 0.693 | 0.766 | 0.570 | 7 | 4 | XGBoost (ADMETboost features) | 0.644 | 6 |
| tdc_dili | ROC_AUC | XGBoost (ADMETboost features) | 0.923 | 0.945 | 0.852 | 6 | 4 | LogisticRegression | 0.745 | 7 |
| tdc_pgp_broccatelli | ROC_AUC | Uni-Mol V2 (84m) | 0.933 | 0.994 | 0.911 | 8 | 4 | LogisticRegression | 0.868 | 9 |
| tdc_vdss_lombardo | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.680 | 0.942 | 0.582 | 16 | 5 | TabPFNRegressor | 0.489 | >10 |
| polaris_adme_fang_rclint_1 | MSE | Chemprop v2 (D-MPNN, ensemble=3) | 0.268 | 0.204 | 0.337 | 10 | 6 | TabPFNRegressor | 0.281 | 6 |
| esol_delaney | RMSE | Ensemble (Weighted average (inverse OOF error)) | 0.624 | 0.558 | 0.743 | 30 | 7 | TabPFNRegressor | 0.673 | 9 |
| polaris_adme_fang_solu_1 | MSE | Uni-Mol V2 (84m) | 0.304 | 0.222 | 0.323 | 10 | 7 | TabPFNRegressor | 0.373 | >10 |
| tdc_cyp3a4_substrate_carbonmangels | ROC_AUC | XGBoost (ADMETboost features) | 0.660 | 0.692 | 0.651 | 7 | 7 | TabPFNClassifier | 0.621 | 8 |
| tdc_hydrationfreeenergy_freesolv | RMSE | Ensemble (Weighted average (inverse OOF error)) | 1.085 | 0.654 | 1.211 | 12 | 7 | TabPFNRegressor | 1.116 | 9 |
| tdc_skin_reaction | ROC_AUC | XGBoost (ADMETboost features) | 0.667 | 0.741 | 0.677 | 21 | >10 | TabPFNClassifier | 0.525 | >10 |
| tdc_tox21 | ROC_AUC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.597 | 0.867 | 0.840 | 12 | >10 | TabPFNClassifier | 0.468 | >10 |
