| Dataset | Metric | QSARena best model (test-selected) | QSARena value | Reference top-1 | Reference top-10 cutoff | References (n) | Est. rank | CV-selected model | CV-selected value | CV-selected est. rank |
|---|---|---|---|---|---|---|---|---|---|---|
| lipophilicity | RMSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.507 | 0.549 | 0.610 | 27 | 1 | TabPFNRegressor | 0.595 | 8 |
| polaris_adme_fang_rppb_1 | MSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.228 | 0.230 | 0.634 | 10 | 1 | XGBoost (ADMETboost features) | 0.257 | 3 |
| tdc_bioavailability_ma | ROC_AUC | CatBoost | 0.777 | 0.748 | 0.640 | 8 | 1 | XGBoost (ADMETboost features) | 0.721 | 2 |
| tdc_carcinogens_lagunin | ROC_AUC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.929 | 0.848 | 0.795 | 20 | 1 | XGBoost (ADMETboost features) | 0.908 | 1 |
| tdc_clearance_microsome_az | SPEARMAN | Uni-Mol V2 (84m) | 0.678 | 0.652 | 0.599 | 17 | 1 | XGBoost (ADMETboost features) | 0.549 | >10 |
| tdc_toxcast | ROC_AUC | HistGradientBoosting | 0.716 | 0.714 | 0.714 | 2 | 1 | XGBoost (ADMETboost features) | 0.704 | 3 |
| poduam_pod_nc_std | RMSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.694 | 0.550 | 0.730 | 2 | 2 | XGBoost (ADMETboost features) | 0.729 | 2 |
| poduam_pod_rd_std | RMSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.556 | 0.410 | 0.630 | 2 | 2 | XGBoost (ADMETboost features) | 0.567 | 2 |
| tdc_ames | ROC_AUC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.883 | 0.912 | 0.834 | 7 | 2 | Extra trees | 0.871 | 2 |
| tdc_bbb_martins | ROC_AUC | Random forest | 0.925 | 0.941 | 0.903 | 7 | 2 | XGBoost (ADMETboost features) | 0.916 | 2 |
| tdc_cyp2d6_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.763 | 0.811 | 0.464 | 6 | 2 | AdaBoost | 0.615 | 6 |
| tdc_cyp3a4_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.905 | 0.923 | 0.750 | 6 | 2 | AdaBoost | 0.798 | 6 |
| tdc_half_life_obach | SPEARMAN | Uni-Mol V1 | 0.597 | 0.649 | 0.485 | 16 | 2 | XGBoost (ADMETboost features) | 0.473 | >10 |
| tdc_hia_hou | ROC_AUC | XGBoost (ADMETboost features) | 0.994 | 0.994 | 0.976 | 8 | 2 | XGBoost (ADMETboost features) | 0.994 | 2 |
| tdc_ld50_zhu | MAE | TabPFNRegressor | 0.552 | 0.292 | 0.605 | 16 | 2 | XGBoost | 0.588 | 8 |
| tdc_ppbr_az | MAE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 7.118 | 0.679 | 7.914 | 16 | 2 | TabPFNRegressor | 8.537 | >10 |
| tdc_solubility_aqsoldb | MAE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.676 | 0.557 | 0.776 | 17 | 2 | XGBoost (ADMETboost features) | 0.719 | 2 |
| tdc_caco2_wang | MAE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.269 | 0.256 | 0.288 | 20 | 3 | XGBoost (ADMETboost features) | 0.293 | >10 |
| tdc_clearance_hepatocyte_az | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.516 | 0.633 | 0.440 | 16 | 3 | XGBoost (ADMETboost features) | 0.369 | >10 |
| tdc_clintox | ROC_AUC | CFA (Combinatorial Fusion) | 0.974 | 0.996 | 0.889 | 12 | 3 | AdaBoost | 0.886 | >10 |
| tdc_cyp2c9_substrate_carbonmangels | AUPRC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.438 | 0.450 | 0.360 | 6 | 3 | SVC | 0.301 | 7 |
| tdc_cyp2c9_veith | AUPRC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.827 | 0.877 | 0.770 | 6 | 3 | AdaBoost | 0.705 | 7 |
| tdc_herg | ROC_AUC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.857 | 0.880 | 0.806 | 7 | 3 | XGBoost (ADMETboost features) | 0.792 | 8 |
| tdc_lipophilicity_astrazeneca | MAE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.412 | 0.406 | 0.515 | 17 | 3 | TabPFNRegressor | 0.439 | 4 |
| polaris_adme_fang_hppb_1 | MSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.178 | 0.143 | 0.303 | 10 | 4 | XGBoost (ADMETboost features) | 0.214 | 5 |
| polaris_adme_fang_perm_1 | MSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.144 | 0.113 | 0.239 | 10 | 4 | XGBoost (ADMETboost features) | 0.194 | 8 |
| tdc_cyp2d6_substrate_carbonmangels | AUPRC | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | 0.693 | 0.766 | 0.570 | 7 | 4 | ChemML MLP (PyTorch) | 0.512 | 8 |
| tdc_dili | ROC_AUC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.933 | 0.945 | 0.852 | 6 | 4 | XGBoost (ADMETboost features) | 0.923 | 4 |
| tdc_pgp_broccatelli | ROC_AUC | Uni-Mol V2 (84m) | 0.933 | 0.994 | 0.911 | 8 | 4 | Random forest | 0.904 | 9 |
| polaris_adme_fang_rclint_1 | MSE | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.234 | 0.204 | 0.337 | 10 | 5 | XGBoost | 0.293 | 6 |
| tdc_vdss_lombardo | SPEARMAN | MapLight + GNN (CatBoost, Strict Parity) | 0.680 | 0.942 | 0.582 | 16 | 5 | XGBoost (ADMETboost features) | 0.465 | >10 |
| tdc_cyp3a4_substrate_carbonmangels | ROC_AUC | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.665 | 0.692 | 0.651 | 7 | 6 | XGBoost (ADMETboost features) | 0.660 | 7 |
| esol_delaney | RMSE | Ensemble (Weighted average (inverse OOF error)) | 0.622 | 0.558 | 0.743 | 30 | 7 | TabPFNRegressor | 0.673 | 9 |
| polaris_adme_fang_solu_1 | MSE | Uni-Mol V2 (84m) | 0.304 | 0.222 | 0.323 | 10 | 7 | XGBoost (ADMETboost features) | 0.353 | >10 |
| tdc_hydrationfreeenergy_freesolv | RMSE | Ensemble (Weighted average (inverse OOF error)) | 1.075 | 0.654 | 1.211 | 12 | 7 | XGBoost (ADMETboost features) | 1.138 | 9 |
| tdc_skin_reaction | ROC_AUC | XGBoost (ADMETboost features) | 0.667 | 0.741 | 0.677 | 21 | >10 | XGBoost (ADMETboost features) | 0.667 | >10 |
| tdc_tox21 | ROC_AUC | Chemprop v2 (AttentiveFP, ensemble=3) | 0.597 | 0.867 | 0.840 | 12 | >10 | Extra trees | 0.549 | >10 |
