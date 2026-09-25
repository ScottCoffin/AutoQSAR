| Architecture family | Model-dataset fits timed | Median own wall-clock (s) | IQR own wall-clock (s) | Median cost incl. base pool (s) | Median trainable parameters | Notes |
|---|---|---|---|---|---|---|
| CFA combinatorial fusion | 44.0 | 0.2 | 0–0 | 3,147 |  | fusion over fitted base-model predictions |
| Ensemble (stacking / averaging) | 90.0 | 0.5 | 0–1 | 3,047 |  | fusion over fitted base-model predictions |
| Conventional ML | 472.0 | 17.1 | 5–88 |  | 520,961 | measured in this run |
| Deep tabular NN (ChemML MLP) | 45.0 | 18.1 | 8–67 |  | 324,097 | measured in this run |
| Chemprop v2 GNN | 44.0 | 97.9 | 83–132 |  | 395,618 | measured in this run |
| TabPFN (tabular foundation) | 18.0 | 125.2 | 40–382 |  |  | measured in this run |
| MapLight + GNN | 31.0 | 150.8 | 147–208 |  |  | measured in this run |
| Uni-Mol V1 (3D pretrained) | 44.0 | 1,964 | 383–5571 |  | 47,331,652 | measured in this run |
| MolGPS (published) |  |  |  |  |  | ~3B parameters; GPU pretraining/inference reported in literature |
| MolE (published) |  |  |  |  |  | ~100M parameters; GPU pretraining reported in literature |
| ADMET-AI (published) |  |  |  |  |  | Chemprop-RDKit; exact parameter count not recorded here; GPU-capable Chemprop-RDKit deployment |
