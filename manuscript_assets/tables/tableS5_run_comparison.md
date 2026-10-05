| Dataset | Task | Same split | A100 best model | A100 value | RTX 4060 best model | RTX 4060 value | Change (%) |
|---|---|---|---|---|---|---|---|
| tdc_ames | classification | yes | CFA (Combinatorial Fusion) | 0.877 | XGBoost | 0.875 | 0.300 |
| tdc_bbb_martins | classification | yes | Random forest | 0.925 | Random forest | 0.932 | -0.788 |
| tdc_bioavailability_ma | classification | yes | CatBoost | 0.777 | CatBoost | 0.777 | 0.000 |
| tdc_cyp1a2_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.974 | Ensemble (Weighted average (inverse train RMSE)) | 0.970 | 0.467 |
| tdc_cyp2c19_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.937 | Ensemble (Weighted average (inverse train RMSE)) | 0.921 | 1.744 |
| tdc_cyp2c9_substrate_carbonmangels | classification | yes | Chemprop v2 (AttentiveFP, ensemble=3) | 0.438 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.482 | -8.990 |
| tdc_cyp2c9_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.827 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.810 | 2.157 |
| tdc_cyp2d6_substrate_carbonmangels | classification | yes | Chemprop v2 (D-MPNN + RDKit2D, ensemble=3) | 0.693 | Ensemble (Weighted average (inverse train RMSE)) | 0.673 | 2.919 |
| tdc_cyp2d6_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.759 | XGBoost | 0.722 | 5.106 |
| tdc_cyp3a4_substrate_carbonmangels | classification | yes | XGBoost (ADMETboost features) | 0.729 | LogisticRegression | 0.717 | 1.759 |
| tdc_cyp3a4_veith | classification | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.902 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.890 | 1.411 |
| tdc_dili | classification | yes | XGBoost (ADMETboost features) | 0.923 | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.915 | 0.856 |
| tdc_herg | classification | yes | Chemprop v2 (AttentiveFP, ensemble=3) | 0.857 | AdaBoost | 0.861 | -0.411 |
| tdc_herg_karim | classification | yes | Ensemble (Weighted average (inverse OOF error)) | 0.907 | Ensemble (Weighted average (inverse train RMSE)) | 0.902 | 0.522 |
| tdc_hia_hou | classification | yes | XGBoost (ADMETboost features) | 0.994 | CFA (Combinatorial Fusion) | 0.990 | 0.395 |
| tdc_pgp_broccatelli | classification | yes | Uni-Mol V2 (84m) | 0.933 | Ensemble (Weighted average (inverse train RMSE)) | 0.929 | 0.423 |
| chemml_cep_homo | regression | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.094 | TabPFNRegressor | 0.086 | -9.571 |
| chemml_organic_density | regression | yes | TabPFNRegressor | 0.005 | TabPFNRegressor | 0.005 | 5.907 |
| esol_delaney | regression | yes | Ensemble (Weighted average (inverse OOF error)) | 0.624 | Ensemble (Weighted average (inverse train RMSE)) | 0.592 | -5.423 |
| freesolv_sampl | regression | yes | TabPFNRegressor | 0.933 | Chemprop v2 (AttentiveFP, ensemble=1) | 1.080 | 13.562 |
| lipophilicity | regression | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.548 | Ensemble (Weighted average (inverse train RMSE)) | 0.582 | 5.905 |
| poduam_pod_nc_std | regression | yes | Random forest | 0.720 | Ensemble (Weighted average (inverse train RMSE)) | 0.699 | -3.062 |
| poduam_pod_rd_std | regression | yes | CFA (Combinatorial Fusion) | 0.564 | XGBoost | 0.551 | -2.320 |
| polaris_adme_fang_hppb_1 | regression | yes | Chemprop v2 (D-MPNN, ensemble=3) | 0.444 | MapLight + GNN (CatBoost, Strict Parity) | 0.449 | 1.072 |
| polaris_adme_fang_perm_1 | regression | yes | Uni-Mol V1 | 0.399 | Uni-Mol V1 | 0.399 | -0.040 |
| polaris_adme_fang_rclint_1 | regression | yes | Chemprop v2 (D-MPNN, ensemble=3) | 0.517 | Uni-Mol V1 | 0.512 | -0.968 |
| polaris_adme_fang_rppb_1 | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 0.493 | MapLight + GNN (CatBoost, Strict Parity) | 0.494 | 0.170 |
| polaris_adme_fang_solu_1 | regression | yes | Uni-Mol V2 (84m) | 0.551 | Uni-Mol V1 | 0.575 | 4.202 |
| tdc_caco2_wang | regression | yes | Ensemble (Weighted average (inverse OOF error)) | 0.344 | Ensemble (Weighted average (inverse train RMSE)) | 0.336 | -2.134 |
| tdc_clearance_hepatocyte_az | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 44.113 | MapLight + GNN (CatBoost, Strict Parity) | 43.976 | -0.311 |
| tdc_clearance_microsome_az | regression | yes | Uni-Mol V2 (84m) | 33.507 | Uni-Mol V1 | 35.888 | 6.634 |
| tdc_half_life_obach | regression | yes | Uni-Mol V1 | 19.128 | Uni-Mol V1 | 17.222 | -11.067 |
| tdc_ld50_zhu | regression | yes | TabPFNRegressor | 0.806 | CFA (Combinatorial Fusion) | 0.834 | 3.290 |
| tdc_lipophilicity_astrazeneca | regression | yes | Chemprop v2 (D-MPNN, ensemble=3) | 0.577 | Uni-Mol V1 | 0.587 | 1.698 |
| tdc_ppbr_az | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 11.356 | Ensemble (Weighted average (inverse train RMSE)) | 11.014 | -3.108 |
| tdc_solubility_aqsoldb | regression | yes | Ensemble (OOF Stacking (RidgeCV, 5-fold)) | 0.956 | Uni-Mol V1 | 0.989 | 3.310 |
| tdc_vdss_lombardo | regression | yes | MapLight + GNN (CatBoost, Strict Parity) | 4.670 | MapLight + GNN (CatBoost, Strict Parity) | 4.668 | -0.035 |
| tdc_carcinogens_lagunin | classification | no | Chemprop v2 (AttentiveFP, ensemble=3) | 0.929 | AdaBoost | 0.863 | 7.656 |
| tdc_clintox | classification | no | CFA (Combinatorial Fusion) | 0.974 | CatBoost | 0.949 | 2.644 |
| tdc_pampa_ncats | classification | no | Uni-Mol V1 | 0.763 | Ensemble (Weighted average (inverse train RMSE)) | 0.806 | -5.258 |
| tdc_skin_reaction | classification | no | XGBoost (ADMETboost features) | 0.667 | CFA (Combinatorial Fusion) | 0.769 | -13.246 |
| tdc_tox21 | classification | no | Chemprop v2 (AttentiveFP, ensemble=3) | 0.597 | XGBoost | 0.812 | -26.451 |
| tdc_toxcast | classification | no | HistGradientBoosting | 0.716 | CatBoost | 0.791 | -9.562 |
| tdc_hydrationfreeenergy_freesolv | regression | no | Ensemble (Weighted average (inverse OOF error)) | 1.085 | CFA (Combinatorial Fusion) | 0.556 | -95.052 |
