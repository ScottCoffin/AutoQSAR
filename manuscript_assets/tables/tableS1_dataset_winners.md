| dataset | suite | task_kind | family | model | analysis_metric | analysis_metric_value | cv_selected_model | cv_selected_family | cv_selected_gap_to_best |
|---|---|---|---|---|---|---|---|---|---|
| tdc_ames | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.883 | Extra trees | Conventional ML | 1.386 |
| tdc_bbb_martins | TDC | classification | Conventional ML | Random forest | test_roc_auc | 0.925 | XGBoost (ADMETboost features) | Conventional ML | 0.963 |
| tdc_bioavailability_ma | TDC | classification | Conventional ML | CatBoost | test_roc_auc | 0.777 | XGBoost (ADMETboost features) | Conventional ML | 7.129 |
| tdc_carcinogens_lagunin | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.929 | XGBoost (ADMETboost features) | Conventional ML | 2.237 |
| tdc_clintox | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.974 | AdaBoost | Conventional ML | 8.986 |
| tdc_cyp1a2_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_roc_auc | 0.974 | SVC | Conventional ML | 1.196 |
| tdc_cyp2c19_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.938 | XGBoost | Conventional ML | 1.338 |
| tdc_cyp2c9_substrate_carbonmangels | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_auprc | 0.438 | SVC | Conventional ML | 31.368 |
| tdc_cyp2c9_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.827 | AdaBoost | Conventional ML | 14.709 |
| tdc_cyp2d6_substrate_carbonmangels | TDC | classification | Chemprop v2 GNN | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | test_auprc | 0.693 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 26.129 |
| tdc_cyp2d6_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.763 | AdaBoost | Conventional ML | 19.417 |
| tdc_cyp3a4_substrate_carbonmangels | TDC | classification | Conventional ML | XGBoost (ADMETboost features) | test_auprc | 0.729 | XGBoost (ADMETboost features) | Conventional ML | 0.000 |
| tdc_cyp3a4_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.905 | AdaBoost | Conventional ML | 11.894 |
| tdc_dili | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.933 | XGBoost (ADMETboost features) | Conventional ML | 1.072 |
| tdc_herg | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.857 | XGBoost (ADMETboost features) | Conventional ML | 7.662 |
| tdc_herg_karim | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_roc_auc | 0.907 | XGBoost | Conventional ML | 0.494 |
| tdc_hia_hou | TDC | classification | Conventional ML | XGBoost (ADMETboost features) | test_roc_auc | 0.994 | XGBoost (ADMETboost features) | Conventional ML | 0.000 |
| tdc_pampa_ncats | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_roc_auc | 0.763 | Random forest | Conventional ML | 10.598 |
| tdc_pgp_broccatelli | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_roc_auc | 0.933 | Random forest | Conventional ML | 3.065 |
| tdc_skin_reaction | TDC | classification | Conventional ML | XGBoost (ADMETboost features) | test_roc_auc | 0.667 | XGBoost (ADMETboost features) | Conventional ML | 0.000 |
| tdc_tox21 | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.597 | Extra trees | Conventional ML | 8.142 |
| tdc_toxcast | TDC | classification | Conventional ML | HistGradientBoosting | test_roc_auc | 0.716 | XGBoost (ADMETboost features) | Conventional ML | 1.662 |
| chemml_cep_homo | ChemML | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.090 | TabPFNRegressor | TabPFN (tabular foundation) | 13.888 |
| chemml_organic_density | ChemML | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.005 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| esol_delaney | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_rmse | 0.622 | TabPFNRegressor | TabPFN (tabular foundation) | 8.211 |
| freesolv_sampl | MoleculeNet | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.933 | XGBoost (ADMETboost features) | Conventional ML | 121.064 |
| lipophilicity | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.507 | TabPFNRegressor | TabPFN (tabular foundation) | 17.351 |
| poduam_pod_nc_std | PODUAM | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.694 | XGBoost (ADMETboost features) | Conventional ML | 5.048 |
| poduam_pod_rd_std | PODUAM | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.556 | XGBoost (ADMETboost features) | Conventional ML | 1.824 |
| polaris_adme_fang_hppb_1 | Polaris | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.422 | XGBoost (ADMETboost features) | Conventional ML | 9.517 |
| polaris_adme_fang_perm_1 | Polaris | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.379 | XGBoost (ADMETboost features) | Conventional ML | 16.319 |
| polaris_adme_fang_rclint_1 | Polaris | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.484 | XGBoost | Conventional ML | 11.804 |
| polaris_adme_fang_rppb_1 | Polaris | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.478 | XGBoost (ADMETboost features) | Conventional ML | 6.138 |
| polaris_adme_fang_solu_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 0.551 | XGBoost (ADMETboost features) | Conventional ML | 7.782 |
| tdc_caco2_wang | TDC | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.338 | XGBoost (ADMETboost features) | Conventional ML | 11.813 |
| tdc_clearance_hepatocyte_az | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 44.113 | XGBoost (ADMETboost features) | Conventional ML | 6.388 |
| tdc_clearance_microsome_az | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 33.507 | XGBoost (ADMETboost features) | Conventional ML | 13.113 |
| tdc_half_life_obach | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 19.128 | XGBoost (ADMETboost features) | Conventional ML | 77.574 |
| tdc_hydrationfreeenergy_freesolv | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_rmse | 1.075 | XGBoost (ADMETboost features) | Conventional ML | 5.821 |
| tdc_ld50_zhu | TDC | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.806 | XGBoost | Conventional ML | 4.826 |
| tdc_lipophilicity_astrazeneca | TDC | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.551 | TabPFNRegressor | TabPFN (tabular foundation) | 8.398 |
| tdc_ppbr_az | TDC | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 10.768 | TabPFNRegressor | TabPFN (tabular foundation) | 21.078 |
| tdc_solubility_aqsoldb | TDC | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.955 | XGBoost (ADMETboost features) | Conventional ML | 4.572 |
| tdc_vdss_lombardo | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 4.670 | XGBoost (ADMETboost features) | Conventional ML | 69.878 |
