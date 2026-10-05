| dataset | suite | task_kind | family | model | analysis_metric | analysis_metric_value | cv_selected_model | cv_selected_family | cv_selected_gap_to_best |
|---|---|---|---|---|---|---|---|---|---|
| tdc_ames | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.877 | TabPFNClassifier | TabPFN (tabular foundation) | 0.133 |
| tdc_bbb_martins | TDC | classification | Conventional ML | Random forest | test_roc_auc | 0.925 | TabPFNClassifier | TabPFN (tabular foundation) | 1.707 |
| tdc_bioavailability_ma | TDC | classification | Conventional ML | CatBoost | test_roc_auc | 0.777 | TabPFNClassifier | TabPFN (tabular foundation) | 5.802 |
| tdc_carcinogens_lagunin | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.929 | TabPFNClassifier | TabPFN (tabular foundation) | 6.339 |
| tdc_clintox | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.974 | TabPFNClassifier | TabPFN (tabular foundation) | 5.190 |
| tdc_cyp1a2_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.974 | ChemML MLP (TensorFlow) | Deep tabular NN (ChemML MLP) | 1.802 |
| tdc_cyp2c19_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.937 | XGBoost | Conventional ML | 1.217 |
| tdc_cyp2c9_substrate_carbonmangels | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_auprc | 0.438 | SVC | Conventional ML | 31.368 |
| tdc_cyp2c9_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.827 | TabPFNClassifier | TabPFN (tabular foundation) | 35.972 |
| tdc_cyp2d6_substrate_carbonmangels | TDC | classification | Chemprop v2 GNN | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | test_auprc | 0.693 | XGBoost (ADMETboost features) | Conventional ML | 7.044 |
| tdc_cyp2d6_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.759 | TabPFNClassifier | TabPFN (tabular foundation) | 39.761 |
| tdc_cyp3a4_substrate_carbonmangels | TDC | classification | Conventional ML | XGBoost (ADMETboost features) | test_auprc | 0.729 | TabPFNClassifier | TabPFN (tabular foundation) | 7.234 |
| tdc_cyp3a4_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.902 | TabPFNClassifier | TabPFN (tabular foundation) | 17.950 |
| tdc_dili | TDC | classification | Conventional ML | XGBoost (ADMETboost features) | test_roc_auc | 0.923 | LogisticRegression | Conventional ML | 19.227 |
| tdc_herg | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.857 | TabPFNClassifier | TabPFN (tabular foundation) | 14.671 |
| tdc_herg_karim | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_roc_auc | 0.907 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 3.832 |
| tdc_hia_hou | TDC | classification | Conventional ML | XGBoost (ADMETboost features) | test_roc_auc | 0.994 | LogisticRegression | Conventional ML | 1.077 |
| tdc_pampa_ncats | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_roc_auc | 0.763 | TabPFNClassifier | TabPFN (tabular foundation) | 24.296 |
| tdc_pgp_broccatelli | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_roc_auc | 0.933 | LogisticRegression | Conventional ML | 6.966 |
| tdc_skin_reaction | TDC | classification | Conventional ML | XGBoost (ADMETboost features) | test_roc_auc | 0.667 | TabPFNClassifier | TabPFN (tabular foundation) | 21.247 |
| tdc_tox21 | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.597 | TabPFNClassifier | TabPFN (tabular foundation) | 21.688 |
| tdc_toxcast | TDC | classification | Conventional ML | HistGradientBoosting | test_roc_auc | 0.716 | TabPFNClassifier | TabPFN (tabular foundation) | 8.338 |
| chemml_cep_homo | ChemML | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.094 | TabPFNRegressor | TabPFN (tabular foundation) | 8.816 |
| chemml_organic_density | ChemML | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.005 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| esol_delaney | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_rmse | 0.624 | TabPFNRegressor | TabPFN (tabular foundation) | 7.800 |
| freesolv_sampl | MoleculeNet | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.933 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| lipophilicity | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.548 | ElasticNetCV | Conventional ML | 24.598 |
| poduam_pod_nc_std | PODUAM | regression | Conventional ML | Random forest | test_rmse | 0.720 | TabPFNRegressor | TabPFN (tabular foundation) | 3.415 |
| poduam_pod_rd_std | PODUAM | regression | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_rmse | 0.564 | TabPFNRegressor | TabPFN (tabular foundation) | 4.435 |
| polaris_adme_fang_hppb_1 | Polaris | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN, ensemble=3) | test_rmse | 0.444 | ElasticNetCV | Conventional ML | 23.269 |
| polaris_adme_fang_perm_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 0.399 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 15.413 |
| polaris_adme_fang_rclint_1 | Polaris | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN, ensemble=3) | test_rmse | 0.517 | TabPFNRegressor | TabPFN (tabular foundation) | 2.507 |
| polaris_adme_fang_rppb_1 | Polaris | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 0.493 | TabPFNRegressor | TabPFN (tabular foundation) | 12.638 |
| polaris_adme_fang_solu_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 0.551 | TabPFNRegressor | TabPFN (tabular foundation) | 10.837 |
| tdc_caco2_wang | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_rmse | 0.344 | TabPFNRegressor | TabPFN (tabular foundation) | 11.718 |
| tdc_clearance_hepatocyte_az | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 44.113 | TabPFNRegressor | TabPFN (tabular foundation) | 14.669 |
| tdc_clearance_microsome_az | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 33.507 | TabPFNRegressor | TabPFN (tabular foundation) | 14.294 |
| tdc_half_life_obach | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 19.128 | XGBoost (ADMETboost features) | Conventional ML | 77.574 |
| tdc_hydrationfreeenergy_freesolv | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse OOF error)) | test_rmse | 1.085 | TabPFNRegressor | TabPFN (tabular foundation) | 2.803 |
| tdc_ld50_zhu | TDC | regression | TabPFN (tabular foundation) | TabPFNRegressor | test_rmse | 0.806 | TabPFNRegressor | TabPFN (tabular foundation) | 0.000 |
| tdc_lipophilicity_astrazeneca | TDC | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN, ensemble=3) | test_rmse | 0.577 | TabPFNRegressor | TabPFN (tabular foundation) | 3.556 |
| tdc_ppbr_az | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 11.356 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 27.931 |
| tdc_solubility_aqsoldb | TDC | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 0.956 | TabPFNRegressor | TabPFN (tabular foundation) | 0.520 |
| tdc_vdss_lombardo | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 4.670 | TabPFNRegressor | TabPFN (tabular foundation) | 6.060 |
