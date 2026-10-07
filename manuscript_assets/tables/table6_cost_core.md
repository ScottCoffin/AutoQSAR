| Model family | Model-dataset fits timed | Median own wall-clock (s) | IQR own wall-clock (s) | Median cost incl. base pool (s) | Median trainable parameters |
|---|---|---|---|---|---|
| CFA combinatorial fusion | 44.0 | 0.3 | 0-0 | 1,378 |  |
| Ensemble (stacking / averaging) | 87.0 | 0.6 | 0-1 | 1,378 |  |
| Conventional ML | 440.0 | 6.8 | 3-20 |  | 157,953 |
| Deep tabular NN (ChemML MLP) | 88.0 | 39.8 | 27-94 |  | 90,369 |
| MapLight + GNN | 44.0 | 137.4 | 126-160 |  |  |
| Chemprop v2 GNN | 25.0 | 268.6 | 229-274 |  | 325,552 |
| Uni-Mol (3D pretrained) | 61.0 | 373.5 | 170-1285 |  | 47,331,652 |
