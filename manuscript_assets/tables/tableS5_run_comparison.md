| Dataset | Task | Same split | A100 best model | A100 value | RTX 4060 best model | RTX 4060 value | Change (%) |
|---|---|---|---|---|---|---|---|
| tdc_ames | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.876 | XGBoost | 0.875 | 0.142 |
| tdc_bbb_martins | classification | yes | Random forest | 0.925 | Random forest | 0.932 | -0.788 |
| tdc_bioavailability_ma | classification | yes | CatBoost | 0.777 | CatBoost | 0.777 | 0.000 |
| tdc_cyp1a2_veith | classification | yes | Ensemble (Weighted average (inverse train RMSE)) | 0.971 | Ensemble (Weighted average (inverse train RMSE)) | 0.970 | 0.091 |
| tdc_cyp2c19_veith | classification | yes | Ensemble (Weighted average (inverse train RMSE)) | 0.923 | Ensemble (Weighted average (inverse train RMSE)) | 0.921 | 0.194 |
| tdc_cyp2c9_substrate_carbonmangels | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.466 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.482 | -3.140 |
| tdc_cyp2c9_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.814 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.810 | 0.481 |
| tdc_cyp2d6_substrate_carbonmangels | classification | yes | Uni-Mol V1 | 0.652 | Ensemble (Weighted average (inverse train RMSE)) | 0.673 | -3.096 |
| tdc_cyp2d6_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.734 | XGBoost | 0.722 | 1.565 |
| tdc_cyp3a4_substrate_carbonmangels | classification | yes | Random forest | 0.706 | LogisticRegression | 0.717 | -1.494 |
| tdc_cyp3a4_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.884 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.890 | -0.637 |
| tdc_dili | classification | yes | Uni-Mol V1 | 0.922 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.915 | 0.760 |
| tdc_herg | classification | yes | AdaBoost | 0.848 | AdaBoost | 0.861 | -1.437 |
| tdc_herg_karim | classification | yes | Ensemble (Weighted average (inverse train RMSE)) | 0.905 | Ensemble (Weighted average (inverse train RMSE)) | 0.902 | 0.355 |
| tdc_hia_hou | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.988 | CFA (Combinatorial Fusion) | 0.990 | -0.187 |
| tdc_pgp_broccatelli | classification | yes | Uni-Mol V2 (84m) | 0.933 | Ensemble (Weighted average (inverse train RMSE)) | 0.929 | 0.423 |
| chemml_cep_homo | regression | yes | Ensemble (Weighted average (inverse train RMSE)) | 0.099 | TabPFNRegressor | 0.086 | -15.648 |
| chemml_organic_density | regression | yes | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | 0.005 | TabPFNRegressor | 0.005 | -1.200 |
| esol_delaney | regression | yes | Ensemble (Weighted average (inverse train RMSE)) | 0.617 | Ensemble (Weighted average (inverse train RMSE)) | 0.592 | -4.220 |
| freesolv_sampl | regression | yes | ChemML MLP (PyTorch) | 0.993 | Chemprop v2 (AttentiveFP, ensemble=1) | 1.080 | 8.015 |
| lipophilicity | regression | yes | Ensemble (Weighted average (inverse train RMSE)) | 0.589 | Ensemble (Weighted average (inverse train RMSE)) | 0.582 | -1.272 |
| poduam_pod_nc_std | regression | yes | Random forest | 0.720 | Ensemble (Weighted average (inverse train RMSE)) | 0.699 | -3.062 |
| poduam_pod_rd_std | regression | yes | Random forest | 0.572 | XGBoost | 0.551 | -3.896 |
| polaris_adme_fang_hppb_1 | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 0.448 | MapLight + GNN (CatBoost, Strict Parity) | 0.449 | 0.188 |
| polaris_adme_fang_perm_1 | regression | yes | Uni-Mol V1 | 0.399 | Uni-Mol V1 | 0.399 | -0.040 |
| polaris_adme_fang_rclint_1 | regression | yes | Uni-Mol V1 | 0.522 | Uni-Mol V1 | 0.512 | -1.976 |
| polaris_adme_fang_rppb_1 | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 0.493 | MapLight + GNN (CatBoost, Strict Parity) | 0.494 | 0.170 |
| polaris_adme_fang_solu_1 | regression | yes | Uni-Mol V2 (84m) | 0.551 | Uni-Mol V1 | 0.575 | 4.202 |
| tdc_caco2_wang | regression | yes | Ensemble (Weighted average (inverse train RMSE)) | 0.342 | Ensemble (Weighted average (inverse train RMSE)) | 0.336 | -1.531 |
| tdc_clearance_hepatocyte_az | regression | yes | Ensemble (Weighted average (inverse train RMSE)) | 43.841 | MapLight + GNN (CatBoost, Strict Parity) | 43.976 | 0.308 |
| tdc_clearance_microsome_az | regression | yes | Uni-Mol V2 (84m) | 33.507 | Uni-Mol V1 | 35.888 | 6.634 |
| tdc_half_life_obach | regression | yes | Ensemble (Weighted average (inverse train RMSE)) | 18.541 | Uni-Mol V1 | 17.222 | -7.656 |
| tdc_ld50_zhu | regression | yes | ChemML MLP (TensorFlow) | 0.828 | CFA (Combinatorial Fusion) | 0.834 | 0.687 |
| tdc_lipophilicity_astrazeneca | regression | yes | Uni-Mol V1 | 0.617 | Uni-Mol V1 | 0.587 | -5.135 |
| tdc_ppbr_az | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 11.356 | Ensemble (Weighted average (inverse train RMSE)) | 11.014 | -3.108 |
| tdc_solubility_aqsoldb | regression | yes | Uni-Mol V1 | 1.007 | Uni-Mol V1 | 0.989 | -1.796 |
| tdc_vdss_lombardo | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 4.670 | MapLight + GNN (CatBoost, Strict Parity) | 4.668 | -0.035 |
| tdc_carcinogens_lagunin | classification | no | Chemprop v2 (AttentiveFP, ensemble=3) | 0.929 | AdaBoost | 0.863 | 7.656 |
| tdc_clintox | classification | no | CFA (Combinatorial Fusion) | 0.974 | CatBoost | 0.949 | 2.644 |
| tdc_pampa_ncats | classification | no | Uni-Mol V1 | 0.763 | Ensemble (Weighted average (inverse train RMSE)) | 0.806 | -5.258 |
| tdc_skin_reaction | classification | no | Uni-Mol V2 (84m) | 0.658 | CFA (Combinatorial Fusion) | 0.769 | -14.470 |
| tdc_tox21 | classification | no | AdaBoost | 0.556 | XGBoost | 0.812 | -31.520 |
| tdc_toxcast | classification | no | HistGradientBoosting | 0.716 | CatBoost | 0.791 | -9.562 |
| tdc_hydrationfreeenergy_freesolv | regression | no | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 1.112 | CFA (Combinatorial Fusion) | 0.556 | -99.848 |
