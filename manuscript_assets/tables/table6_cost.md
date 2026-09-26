| Model family | Model-dataset fits timed | Median own wall-clock (s) | IQR own wall-clock (s) | Median cost incl. base pool (s) | Median trainable parameters | Notes |
|---|---|---|---|---|---|---|
| CFA combinatorial fusion | 44.0 | 0.3 | 0–0 | 1,378 |  | fusion over fitted base-model predictions |
| Ensemble (stacking / averaging) | 87.0 | 0.6 | 0–1 | 1,378 |  | fusion over fitted base-model predictions |
| Conventional ML | 440.0 | 6.8 | 3–20 |  | 157,953 | measured in this run |
| Deep tabular NN (ChemML MLP) | 88.0 | 39.8 | 27–94 |  | 90,369 | measured in this run |
| MapLight + GNN | 44.0 | 137.4 | 126–160 |  |  | measured in this run |
| Chemprop v2 GNN | 25.0 | 268.6 | 229–274 |  | 325,552 | measured in this run |
| Uni-Mol (3D pretrained) | 61.0 | 373.5 | 170–1285 |  | 47,331,652 | measured in this run |
| MolGPS (published) |  |  |  |  |  | ~3B parameters; GPU pretraining/inference reported in literature |
| MolE (published) |  |  |  |  |  | ~100M parameters; GPU pretraining reported in literature |
| ADMET-AI (published) |  |  |  |  |  | Chemprop-RDKit; exact parameter count not recorded here; GPU-capable Chemprop-RDKit deployment |
