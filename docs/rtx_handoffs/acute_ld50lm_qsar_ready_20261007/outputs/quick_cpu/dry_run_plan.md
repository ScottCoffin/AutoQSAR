# QSARena dry-run plan

- Output directory: `benchmark_results\acute_ld50lm_qsar_ready_quick_20261007`
- Profile: quick; GPU: no
- Estimated wall-clock: ~2.1 h (order of magnitude; see preflight.json for the method)

## full_dataset_qsar_ready_ld50lm (6954 rows, regression)

Stages: 1 load + standardize, 2 features, 3 split + train-only feature selection, then one stage per model.
Estimated: ~2.1 h.

| Model | Family | Status | Reason |
|---|---|---|---|
| ElasticNetCV | conventional_ml | planned |  |
| SVR | conventional_ml | planned |  |
| Random forest | conventional_ml | planned |  |
| Extra trees | conventional_ml | planned |  |
| HistGradientBoosting | conventional_ml | planned |  |
| Voting Regressor (KNN, SVM) | conventional_ml | planned |  |
| AdaBoost | conventional_ml | planned |  |
| Tabular MLP | conventional_ml | planned |  |
| CFA (Combinatorial Fusion) | fusion | planned |  |
| Ensemble (OOF Stacking (RidgeCV)) | ensemble | planned |  |
| Ensemble (Weighted average (inverse train RMSE)) | ensemble | planned |  |
| Tabular CNN | deep_tabular | skipped | family deep_tabular switched off by the quick profile |
| XGBoost | gradient_boosting | skipped | family gradient_boosting switched off by the quick profile |
| LightGBM | gradient_boosting | skipped | family gradient_boosting switched off by the quick profile |
| CatBoost | gradient_boosting | skipped | family gradient_boosting switched off by the quick profile |
| MapLight CatBoost (Strict Parity) | gradient_boosting | skipped | family gradient_boosting switched off by the quick profile |
| TabPFNRegressor | deep_tabular | skipped | family deep_tabular switched off by the quick profile |
| ChemML MLP (PyTorch) | deep_tabular | skipped | family deep_tabular switched off by the quick profile |
| ChemML MLP (TensorFlow) | deep_tabular | skipped | family deep_tabular switched off by the quick profile |
| Chemprop v2 (all variants) | graph_nn | skipped | family graph_nn switched off by the quick profile |
| Uni-Mol V1 | pretrained_3d | skipped | family pretrained_3d switched off by the quick profile |
| Uni-Mol V2 (84m) | pretrained_3d | skipped | family pretrained_3d switched off by the quick profile |
| MapLight + GNN (CatBoost, Strict Parity) | maplight_gnn | skipped | family maplight_gnn switched off by the quick profile |
