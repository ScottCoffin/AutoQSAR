| dataset | suite | task_kind | family | model | analysis_metric | analysis_metric_value | cv_selected_model | cv_selected_family | cv_selected_gap_to_best |
|---|---|---|---|---|---|---|---|---|---|
| tdc_ames | TDC | classification | Conventional ML | XGBoost | test_roc_auc | 0.875 | Extra trees | Conventional ML | 0.777 |
| tdc_bbb_martins | TDC | classification | Conventional ML | Random forest | test_roc_auc | 0.932 | Tabular MLP | Conventional ML | 5.108 |
| tdc_bioavailability_ma | TDC | classification | Conventional ML | CatBoost | test_roc_auc | 0.777 | LogisticRegression | Conventional ML | 7.943 |
| tdc_carcinogens_lagunin | TDC | classification | Conventional ML | AdaBoost | test_roc_auc | 0.863 | LogisticRegression | Conventional ML | 3.747 |
| tdc_clintox | TDC | classification | Conventional ML | CatBoost | test_roc_auc | 0.949 | LogisticRegression | Conventional ML | 3.552 |
| tdc_cyp1a2_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.970 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 2.383 |
| tdc_cyp2c19_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.921 | XGBoost | Conventional ML | 8.809 |
| tdc_cyp2c9_substrate_carbonmangels | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.482 | SVC | Conventional ML | 37.538 |
| tdc_cyp2c9_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.810 | AdaBoost | Conventional ML | 12.949 |
| tdc_cyp2d6_substrate_carbonmangels | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_auprc | 0.673 | AdaBoost | Conventional ML | 3.067 |
| tdc_cyp2d6_veith | TDC | classification | Conventional ML | XGBoost | test_auprc | 0.722 | AdaBoost | Conventional ML | 14.842 |
| tdc_cyp3a4_substrate_carbonmangels | TDC | classification | Conventional ML | LogisticRegression | test_auprc | 0.717 | Tabular MLP | Conventional ML | 5.325 |
| tdc_cyp3a4_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.890 | AdaBoost | Conventional ML | 10.365 |
| tdc_dili | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.915 | LogisticRegression | Conventional ML | 6.036 |
| tdc_herg | TDC | classification | Conventional ML | AdaBoost | test_roc_auc | 0.861 | Tabular MLP | Conventional ML | 15.894 |
| tdc_herg_karim | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.902 | SVC | Conventional ML | 9.739 |
| tdc_hia_hou | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.990 | LogisticRegression | Conventional ML | 0.270 |
| tdc_pampa_ncats | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.806 | Tabular MLP | Conventional ML | 5.601 |
| tdc_pgp_broccatelli | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.929 | Tabular MLP | Conventional ML | 2.457 |
| tdc_skin_reaction | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.769 | LogisticRegression | Conventional ML | 4.572 |
| tdc_tox21 | TDC | classification | Conventional ML | XGBoost | test_roc_auc | 0.812 | SVC | Conventional ML | 2.861 |
| tdc_toxcast | TDC | classification | Conventional ML | CatBoost | test_roc_auc | 0.791 | CatBoost | Conventional ML | 0.000 |
| chemml_cep_homo | ChemML | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.086 | ElasticNetCV | Conventional ML | 28.254 |
| chemml_organic_density | ChemML | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.005 | ElasticNetCV | Conventional ML | 26.007 |
| chemml_xyz_polarizability | ChemML | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.007 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| esol_delaney | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.592 | ElasticNetCV | Conventional ML | 14.805 |
| freesolv_sampl | MoleculeNet | regression | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=1) | test_rmse | 1.080 | ElasticNetCV | Conventional ML | 15.792 |
| lipophilicity | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.582 | ElasticNetCV | Conventional ML | 13.041 |
| poduam_pod_nc_std | PODUAM | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.699 | XGBoost | Conventional ML | 5.154 |
| poduam_pod_rd_std | PODUAM | regression | Conventional ML | XGBoost | test_rmse | 0.551 | ElasticNetCV | Conventional ML | 29.477 |
| polaris_adme_fang_hppb_1 | Polaris | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 0.449 | ElasticNetCV | Conventional ML | 16.167 |
| polaris_adme_fang_perm_1 | Polaris | regression | Uni-Mol V1 (3D pretrained) | Uni-Mol V1 | test_rmse | 0.399 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 18.507 |
| polaris_adme_fang_rclint_1 | Polaris | regression | Uni-Mol V1 (3D pretrained) | Uni-Mol V1 | test_rmse | 0.512 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 12.570 |
| polaris_adme_fang_rppb_1 | Polaris | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 0.494 | TabPFNRegressor | TabPFN (tabular foundation) | 8.043 |
| polaris_adme_fang_solu_1 | Polaris | regression | Uni-Mol V1 (3D pretrained) | Uni-Mol V1 | test_rmse | 0.575 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 12.303 |
| tdc_caco2_wang | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.336 | ElasticNetCV | Conventional ML | 13.842 |
| tdc_clearance_hepatocyte_az | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 43.976 | TabPFNRegressor | TabPFN (tabular foundation) | 17.126 |
| tdc_clearance_microsome_az | TDC | regression | Uni-Mol V1 (3D pretrained) | Uni-Mol V1 | test_rmse | 35.888 | Tabular MLP | Conventional ML | 13.242 |
| tdc_half_life_obach | TDC | regression | Uni-Mol V1 (3D pretrained) | Uni-Mol V1 | test_rmse | 17.222 | TabPFNRegressor | TabPFN (tabular foundation) | 11.498 |
| tdc_hydrationfreeenergy_freesolv | TDC | regression | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_rmse | 0.556 | ElasticNetCV | Conventional ML | 143.019 |
| tdc_ld50_zhu | TDC | regression | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_rmse | 0.834 | XGBoost | Conventional ML | 1.097 |
| tdc_lipophilicity_astrazeneca | TDC | regression | Uni-Mol V1 (3D pretrained) | Uni-Mol V1 | test_rmse | 0.587 | ElasticNetCV | Conventional ML | 27.840 |
| tdc_ppbr_az | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 11.014 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 28.555 |
| tdc_solubility_aqsoldb | TDC | regression | Uni-Mol V1 (3D pretrained) | Uni-Mol V1 | test_rmse | 0.989 | Extra trees | Conventional ML | 5.427 |
| tdc_vdss_lombardo | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 4.668 | TabPFNRegressor | TabPFN (tabular foundation) | 16.999 |
