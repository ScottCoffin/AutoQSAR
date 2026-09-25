| dataset | suite | task_kind | family | model | analysis_metric | analysis_metric_value | cv_selected_model | cv_selected_family | cv_selected_gap_to_best |
|---|---|---|---|---|---|---|---|---|---|
| tdc_ames | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.876 | Extra trees | Conventional ML | 0.577 |
| tdc_bbb_martins | TDC | classification | Conventional ML | Random forest | test_roc_auc | 0.925 | Tabular MLP | Conventional ML | 2.943 |
| tdc_bioavailability_ma | TDC | classification | Conventional ML | CatBoost | test_roc_auc | 0.777 | LogisticRegression | Conventional ML | 7.943 |
| tdc_carcinogens_lagunin | TDC | classification | Chemprop v2 GNN | Chemprop v2 (AttentiveFP, ensemble=3) | test_roc_auc | 0.929 | Voting Classifier (KNN, SVM) | Conventional ML | 9.918 |
| tdc_clintox | TDC | classification | CFA combinatorial fusion | CFA (Combinatorial Fusion) | test_roc_auc | 0.974 | LogisticRegression | Conventional ML | 8.005 |
| tdc_cyp1a2_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.971 | ChemML MLP (TensorFlow) | Deep tabular NN (ChemML MLP) | 1.432 |
| tdc_cyp2c19_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.923 | XGBoost | Conventional ML | 8.976 |
| tdc_cyp2c9_substrate_carbonmangels | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.466 | SVC | Conventional ML | 35.513 |
| tdc_cyp2c9_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.814 | AdaBoost | Conventional ML | 13.326 |
| tdc_cyp2d6_substrate_carbonmangels | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_auprc | 0.652 | AdaBoost | Conventional ML | 4.966 |
| tdc_cyp2d6_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.734 | AdaBoost | Conventional ML | 16.154 |
| tdc_cyp3a4_substrate_carbonmangels | TDC | classification | Conventional ML | Random forest | test_auprc | 0.706 | LogisticRegression | Conventional ML | 2.791 |
| tdc_cyp3a4_veith | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_auprc | 0.884 | AdaBoost | Conventional ML | 9.750 |
| tdc_dili | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_roc_auc | 0.922 | LogisticRegression | Conventional ML | 19.151 |
| tdc_herg | TDC | classification | Conventional ML | AdaBoost | test_roc_auc | 0.848 | LogisticRegression | Conventional ML | 16.386 |
| tdc_herg_karim | TDC | classification | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_roc_auc | 0.905 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 3.672 |
| tdc_hia_hou | TDC | classification | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_roc_auc | 0.988 | LogisticRegression | Conventional ML | 0.500 |
| tdc_pampa_ncats | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_roc_auc | 0.763 | LogisticRegression | Conventional ML | 26.281 |
| tdc_pgp_broccatelli | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_roc_auc | 0.933 | LogisticRegression | Conventional ML | 6.966 |
| tdc_skin_reaction | TDC | classification | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_roc_auc | 0.658 | Voting Classifier (KNN, SVM) | Conventional ML | 23.497 |
| tdc_tox21 | TDC | classification | Conventional ML | AdaBoost | test_roc_auc | 0.556 | SVC | Conventional ML | 15.985 |
| tdc_toxcast | TDC | classification | Conventional ML | HistGradientBoosting | test_roc_auc | 0.716 | LogisticRegression | Conventional ML | 16.900 |
| chemml_cep_homo | ChemML | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.099 | ElasticNetCV | Conventional ML | 22.270 |
| chemml_organic_density | ChemML | regression | Chemprop v2 GNN | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | test_rmse | 0.005 | ElasticNetCV | Conventional ML | 34.525 |
| esol_delaney | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.617 | ElasticNetCV | Conventional ML | 17.596 |
| freesolv_sampl | MoleculeNet | regression | Deep tabular NN (ChemML MLP) | ChemML MLP (PyTorch) | test_rmse | 0.993 | CatBoost | Conventional ML | 115.432 |
| lipophilicity | MoleculeNet | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.589 | ElasticNetCV | Conventional ML | 15.769 |
| poduam_pod_nc_std | PODUAM | regression | Conventional ML | Random forest | test_rmse | 0.720 | ElasticNetCV | Conventional ML | 9.929 |
| poduam_pod_rd_std | PODUAM | regression | Conventional ML | Random forest | test_rmse | 0.572 | ElasticNetCV | Conventional ML | 22.906 |
| polaris_adme_fang_hppb_1 | Polaris | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 0.448 | ElasticNetCV | Conventional ML | 22.177 |
| polaris_adme_fang_perm_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 0.399 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 15.413 |
| polaris_adme_fang_rclint_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 0.522 | ElasticNetCV | Conventional ML | 15.664 |
| polaris_adme_fang_rppb_1 | Polaris | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 0.493 | ElasticNetCV | Conventional ML | 21.893 |
| polaris_adme_fang_solu_1 | Polaris | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 0.551 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 16.215 |
| tdc_caco2_wang | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 0.342 | ElasticNetCV | Conventional ML | 22.979 |
| tdc_clearance_hepatocyte_az | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 43.841 | CatBoost | Conventional ML | 12.399 |
| tdc_clearance_microsome_az | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V2 (84m) | test_rmse | 33.507 | XGBoost | Conventional ML | 16.477 |
| tdc_half_life_obach | TDC | regression | Ensemble (stacking / averaging) | Ensemble (Weighted average (inverse train RMSE)) | test_rmse | 18.541 | Random forest | Conventional ML | 21.086 |
| tdc_hydrationfreeenergy_freesolv | TDC | regression | Ensemble (stacking / averaging) | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | test_rmse | 1.112 | ElasticNetCV | Conventional ML | 18.818 |
| tdc_ld50_zhu | TDC | regression | Deep tabular NN (ChemML MLP) | ChemML MLP (TensorFlow) | test_rmse | 0.828 | XGBoost | Conventional ML | 2.079 |
| tdc_lipophilicity_astrazeneca | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 0.617 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 17.583 |
| tdc_ppbr_az | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 11.356 | ChemML MLP (PyTorch) | Deep tabular NN (ChemML MLP) | 27.931 |
| tdc_solubility_aqsoldb | TDC | regression | Uni-Mol (3D pretrained) | Uni-Mol V1 | test_rmse | 1.007 | XGBoost | Conventional ML | 1.814 |
| tdc_vdss_lombardo | TDC | regression | MapLight + GNN | MapLight + GNN (CatBoost, Strict Parity) | test_rmse | 4.670 | Extra trees | Conventional ML | 50.704 |
