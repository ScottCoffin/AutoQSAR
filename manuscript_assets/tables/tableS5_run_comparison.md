| Dataset | Task | Same split | A100 best model | A100 value | RTX 4060 best model | RTX 4060 value | Change (%) |
|---|---|---|---|---|---|---|---|
| tdc_ames | classification | yes | TabPFNClassifier | 0.876 | XGBoost | 0.875 | 0.166 |
| tdc_bbb_martins | classification | yes | Random forest | 0.925 | Random forest | 0.932 | -0.788 |
| tdc_bioavailability_ma | classification | yes | CatBoost | 0.777 | CatBoost | 0.777 | 0.000 |
| tdc_cyp1a2_veith | classification | yes | XGBoost | 0.969 | Uni-Mol V1 | 0.969 | -0.049 |
| tdc_cyp2c19_veith | classification | yes | XGBoost | 0.926 | Uni-Mol V1 | 0.921 | 0.551 |
| tdc_cyp2c9_substrate_carbonmangels | classification | yes | Chemprop v2 (AttentiveFP, ensemble=3) | 0.438 | Uni-Mol V1 | 0.454 | -3.439 |
| tdc_cyp2c9_veith | classification | yes | XGBoost | 0.796 | HistGradientBoosting | 0.792 | 0.541 |
| tdc_cyp2d6_substrate_carbonmangels | classification | yes | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | 0.693 | Random forest | 0.665 | 4.143 |
| tdc_cyp2d6_veith | classification | yes | XGBoost | 0.729 | XGBoost | 0.722 | 0.904 |
| tdc_cyp3a4_substrate_carbonmangels | classification | yes | XGBoost (ADMETboost features) | 0.729 | LogisticRegression | 0.717 | 1.759 |
| tdc_cyp3a4_veith | classification | yes | XGBoost (ADMETboost features) | 0.883 | XGBoost | 0.883 | 0.002 |
| tdc_dili | classification | yes | XGBoost (ADMETboost features) | 0.923 | Uni-Mol V1 | 0.914 | 0.903 |
| tdc_herg | classification | yes | Chemprop v2 (AttentiveFP, ensemble=3) | 0.857 | AdaBoost | 0.861 | -0.411 |
| tdc_herg_karim | classification | yes | XGBoost | 0.902 | Uni-Mol V1 | 0.893 | 1.055 |
| tdc_hia_hou | classification | yes | XGBoost (ADMETboost features) | 0.994 | CatBoost | 0.989 | 0.499 |
| tdc_pgp_broccatelli | classification | yes | Uni-Mol V2 (84m) | 0.933 | TabPFNClassifier | 0.927 | 0.625 |
| chemml_cep_homo | regression | yes | TabPFNRegressor | 0.102 | TabPFNRegressor | 0.086 | -19.231 |
| chemml_organic_density | regression | yes | TabPFNRegressor | 0.005 | TabPFNRegressor | 0.005 | 5.907 |
| esol_delaney | regression | yes | MapLight CatBoost (Strict Parity) | 0.645 | TabPFNRegressor | 0.612 | -5.269 |
| freesolv_sampl | regression | yes | TabPFNRegressor | 0.933 | Chemprop v2 (AttentiveFP, ensemble=1) | 1.080 | 13.562 |
| lipophilicity | regression | yes | Chemprop v2 (D-MPNN, ensemble=3) | 0.551 | Uni-Mol V1 | 0.592 | 6.932 |
| poduam_pod_nc_std | regression | yes | Random forest | 0.720 | Extra trees | 0.701 | -2.720 |
| poduam_pod_rd_std | regression | yes | XGBoost (ADMETboost features) | 0.567 | XGBoost | 0.551 | -2.872 |
| polaris_adme_fang_hppb_1 | regression | yes | Chemprop v2 (D-MPNN, ensemble=3) | 0.444 | MapLight + GNN (CatBoost, Strict Parity) | 0.449 | 1.072 |
| polaris_adme_fang_perm_1 | regression | yes | Uni-Mol V1 | 0.399 | Uni-Mol V1 | 0.399 | -0.040 |
| polaris_adme_fang_rclint_1 | regression | yes | Chemprop v2 (D-MPNN, ensemble=3) | 0.517 | Uni-Mol V1 | 0.512 | -0.968 |
| polaris_adme_fang_rppb_1 | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 0.493 | MapLight + GNN (CatBoost, Strict Parity) | 0.494 | 0.170 |
| polaris_adme_fang_solu_1 | regression | yes | Uni-Mol V2 (84m) | 0.551 | Uni-Mol V1 | 0.575 | 4.202 |
| tdc_caco2_wang | regression | yes | MapLight CatBoost (Strict Parity) | 0.350 | MapLight CatBoost (Strict Parity) | 0.350 | 0.000 |
| tdc_clearance_hepatocyte_az | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 44.113 | MapLight + GNN (CatBoost, Strict Parity) | 43.976 | -0.311 |
| tdc_clearance_microsome_az | regression | yes | Uni-Mol V2 (84m) | 33.507 | Uni-Mol V1 | 35.888 | 6.634 |
| tdc_half_life_obach | regression | yes | Uni-Mol V1 | 19.128 | Uni-Mol V1 | 17.222 | -11.067 |
| tdc_ld50_zhu | regression | yes | TabPFNRegressor | 0.806 | HistGradientBoosting | 0.839 | 3.931 |
| tdc_lipophilicity_astrazeneca | regression | yes | Chemprop v2 (D-MPNN, ensemble=3) | 0.577 | Uni-Mol V1 | 0.587 | 1.698 |
| tdc_ppbr_az | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 11.356 | MapLight + GNN (CatBoost, Strict Parity) | 11.444 | 0.762 |
| tdc_solubility_aqsoldb | regression | yes | TabPFNRegressor | 0.961 | Uni-Mol V1 | 0.989 | 2.807 |
| tdc_vdss_lombardo | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 4.670 | MapLight + GNN (CatBoost, Strict Parity) | 4.668 | -0.035 |
| tdc_carcinogens_lagunin | classification | no | Chemprop v2 (AttentiveFP, ensemble=3) | 0.929 | AdaBoost | 0.863 | 7.656 |
| tdc_clintox | classification | no | Random forest | 0.956 | CatBoost | 0.949 | 0.788 |
| tdc_pampa_ncats | classification | no | Uni-Mol V1 | 0.763 | Extra trees | 0.805 | -5.206 |
| tdc_skin_reaction | classification | no | XGBoost (ADMETboost features) | 0.667 | TabPFNClassifier | 0.747 | -10.737 |
| tdc_tox21 | classification | no | Chemprop v2 (AttentiveFP, ensemble=3) | 0.597 | XGBoost | 0.812 | -26.451 |
| tdc_toxcast | classification | no | HistGradientBoosting | 0.716 | CatBoost | 0.791 | -9.562 |
| tdc_hydrationfreeenergy_freesolv | regression | no | TabPFNRegressor | 1.116 | TabPFNRegressor | 0.592 | -88.600 |
