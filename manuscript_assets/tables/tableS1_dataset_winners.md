| dataset | suite | task_kind | family | model | analysis_metric | analysis_metric_value | cv_selected_model | cv_selected_family | cv_selected_gap_to_best |
|---|---|---|---|---|---|---|---|---|---|
| tdc_ames | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.879 | TabPFNClassifier | TabPFN (tabular foundation) | 0.345 |
| tdc_bbb_martins | TDC | classification | Conventional ML | Random forest | test_roc_auc | 0.925 | TabPFNClassifier | TabPFN (tabular foundation) | 1.707 |
| tdc_bioavailability_ma | TDC | classification | Conventional ML | CatBoost | test_roc_auc | 0.777 | TabPFNClassifier | TabPFN (tabular foundation) | 5.802 |
| tdc_carcinogens_lagunin | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.929 | TabPFNClassifier | TabPFN (tabular foundation) | 6.339 |
| tdc_clintox | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.974 | TabPFNClassifier | TabPFN (tabular foundation) | 5.190 |
| tdc_cyp1a2_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.974 | ChemML MLP (TensorFlow) | Deep tabular NN (ChemML MLP) | 1.741 |
| tdc_cyp2c19_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.935 | XGBoost | Conventional ML | 10.141 |
| tdc_cyp2c9_substrate_carbonmangels | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_auprc | 0.438 | SVC | Conventional ML | 31.368 |
| tdc_cyp2c9_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.829 | TabPFNClassifier | TabPFN (tabular foundation) | 36.102 |
| tdc_cyp2d6_substrate_carbonmangels | TDC | classification | Chemprop v2 GNN | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | test_auprc | 0.693 | AdaBoost | Conventional ML | 10.520 |
| tdc_cyp2d6_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.762 | TabPFNClassifier | TabPFN (tabular foundation) | 39.976 |
| tdc_cyp3a4_substrate_carbonmangels | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.706 | TabPFNClassifier | TabPFN (tabular foundation) | 4.179 |
| tdc_cyp3a4_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.901 | TabPFNClassifier | TabPFN (tabular foundation) | 17.830 |
| tdc_dili | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_roc_auc | 0.922 | LogisticRegression | Conventional ML | 19.151 |
| tdc_herg | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.857 | TabPFNClassifier | TabPFN (tabular foundation) | 14.671 |
| tdc_herg_karim | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.904 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 3.496 |
| tdc_hia_hou | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.987 | LogisticRegression | Conventional ML | 0.438 |
| tdc_pampa_ncats | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_roc_auc | 0.763 | LogisticRegression | Conventional ML | 26.281 |
| tdc_pgp_broccatelli | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_roc_auc | 0.933 | LogisticRegression | Conventional ML | 6.966 |
| tdc_skin_reaction | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_roc_auc | 0.658 | TabPFNClassifier | TabPFN (tabular foundation) | 20.120 |
| tdc_tox21 | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.597 | TabPFNClassifier | TabPFN (tabular foundation) | 21.688 |
| tdc_toxcast | TDC | classification | Conventional ML | HistGradientBoosting | test_roc_auc | 0.716 | TabPFNClassifier | TabPFN (tabular foundation) | 8.338 |
| chemml_cep_homo | ChemML | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.099 | TabPFNRegressor | TabPFN (tabular foundation) | 3.353 |
| chemml_organic_density | ChemML | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.005 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| esol_delaney | MoleculeNet | regression | Conventional ML | MapLight CatBoost (Strict Parity) | test_rmse | 0.645 | TabPFNRegressor | TabPFN (tabular foundation) | 4.378 |
| freesolv_sampl | MoleculeNet | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.933 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| lipophilicity | MoleculeNet | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN, ensemble=3) | test_rmse | 0.551 | ElasticNetCV | Conventional ML | 23.912 |
| poduam_pod_nc_std | PODUAM | regression | Conventional ML | Random forest | test_rmse | 0.720 | TabPFNRegressor | TabPFN (tabular foundation) | 3.415 |
| poduam_pod_rd_std | PODUAM | regression | Conventional ML | Random forest | test_rmse | 0.572 | TabPFNRegressor | TabPFN (tabular foundation) | 2.852 |
| polaris_adme_fang_hppb_1 | Polaris | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN, ensemble=3) | test_rmse | 0.444 | ElasticNetCV | Conventional ML | 23.269 |
| polaris_adme_fang_perm_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 0.399 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 15.413 |
| polaris_adme_fang_rclint_1 | Polaris | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN, ensemble=3) | test_rmse | 0.517 | TabPFNRegressor | TabPFN (tabular foundation) | 2.507 |
| polaris_adme_fang_rppb_1 | Polaris | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 0.493 | TabPFNRegressor | TabPFN (tabular foundation) | 12.638 |
| polaris_adme_fang_solu_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 0.551 | TabPFNRegressor | TabPFN (tabular foundation) | 10.837 |
| tdc_caco2_wang | TDC | regression | Conventional ML | MapLight CatBoost (Strict Parity) | test_rmse | 0.350 | TabPFNRegressor | TabPFN (tabular foundation) | 9.799 |
| tdc_clearance_hepatocyte_az | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 44.113 | TabPFNRegressor | TabPFN (tabular foundation) | 14.669 |
| tdc_clearance_microsome_az | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 33.507 | TabPFNRegressor | TabPFN (tabular foundation) | 14.294 |
| tdc_half_life_obach | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 19.128 | TabPFNRegressor | TabPFN (tabular foundation) | 6.346 |
| tdc_hydrationfreeenergy_freesolv | TDC | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 1.116 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| tdc_ld50_zhu | TDC | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.806 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| tdc_lipophilicity_astrazeneca | TDC | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN, ensemble=3) | test_rmse | 0.577 | TabPFNRegressor | TabPFN (tabular foundation) | 3.556 |
| tdc_ppbr_az | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 11.356 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 27.931 |
| tdc_solubility_aqsoldb | TDC | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.961 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| tdc_vdss_lombardo | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 4.670 | TabPFNRegressor | TabPFN (tabular foundation) | 6.060 |
