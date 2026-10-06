| Model family | Meta-feature | Datasets | Spearman rho | 95% CI | Permutation p | BH q |
|---|---|---|---|---|---|---|
| Conventional ML | log10_n_train | 44 | 0.034 | -0.29 to 0.35 | 0.818 | 0.873 |
| Conventional ML | internal_diversity | 44 | -0.329 | -0.59 to -0.01 | 0.028 | 0.238 |
| Conventional ML | mean_snn | 44 | 0.325 | 0.01 to 0.58 | 0.034 | 0.238 |
| Conventional ML | ood_fraction | 44 | -0.326 | -0.59 to -0.01 | 0.035 | 0.238 |
| Conventional ML | scaffolds_per_molecule | 44 | 0.139 | -0.20 to 0.43 | 0.367 | 0.710 |
| Conventional ML | label_asymmetry | 44 | 0.204 | -0.10 to 0.46 | 0.179 | 0.461 |
| Ensemble (stacking / averaging) | log10_n_train | 44 | -0.467 | -0.70 to -0.19 | 0.001 | 0.058 |
| Ensemble (stacking / averaging) | internal_diversity | 44 | 0.207 | -0.14 to 0.52 | 0.180 | 0.461 |
| Ensemble (stacking / averaging) | mean_snn | 44 | -0.393 | -0.66 to -0.09 | 0.008 | 0.202 |
| Ensemble (stacking / averaging) | ood_fraction | 44 | 0.288 | -0.03 to 0.59 | 0.062 | 0.297 |
| Ensemble (stacking / averaging) | scaffolds_per_molecule | 44 | 0.147 | -0.14 to 0.43 | 0.336 | 0.710 |
| Ensemble (stacking / averaging) | label_asymmetry | 44 | 0.275 | -0.03 to 0.54 | 0.070 | 0.308 |
| CFA combinatorial fusion | log10_n_train | 44 | -0.220 | -0.47 to 0.07 | 0.156 | 0.461 |
| CFA combinatorial fusion | internal_diversity | 44 | -0.325 | -0.59 to -0.01 | 0.033 | 0.238 |
| CFA combinatorial fusion | mean_snn | 44 | 0.125 | -0.21 to 0.43 | 0.421 | 0.710 |
| CFA combinatorial fusion | ood_fraction | 44 | -0.081 | -0.39 to 0.24 | 0.602 | 0.743 |
| CFA combinatorial fusion | scaffolds_per_molecule | 44 | 0.216 | -0.10 to 0.50 | 0.159 | 0.461 |
| CFA combinatorial fusion | label_asymmetry | 44 | 0.108 | -0.19 to 0.40 | 0.486 | 0.710 |
| Uni-Mol (3D pretrained) | log10_n_train | 44 | -0.108 | -0.41 to 0.20 | 0.488 | 0.710 |
| Uni-Mol (3D pretrained) | internal_diversity | 44 | 0.146 | -0.21 to 0.49 | 0.348 | 0.710 |
| Uni-Mol (3D pretrained) | mean_snn | 44 | 0.081 | -0.25 to 0.40 | 0.604 | 0.743 |
| Uni-Mol (3D pretrained) | ood_fraction | 44 | -0.118 | -0.46 to 0.22 | 0.442 | 0.710 |
| Uni-Mol (3D pretrained) | scaffolds_per_molecule | 44 | 0.102 | -0.23 to 0.41 | 0.516 | 0.729 |
| Uni-Mol (3D pretrained) | label_asymmetry | 44 | 0.075 | -0.26 to 0.41 | 0.628 | 0.753 |
| TabPFN (tabular foundation) | log10_n_train | 41 | 0.149 | -0.24 to 0.50 | 0.346 | 0.710 |
| TabPFN (tabular foundation) | internal_diversity | 41 | -0.242 | -0.54 to 0.10 | 0.130 | 0.461 |
| TabPFN (tabular foundation) | mean_snn | 41 | 0.098 | -0.21 to 0.40 | 0.540 | 0.730 |
| TabPFN (tabular foundation) | ood_fraction | 41 | -0.004 | -0.34 to 0.33 | 0.983 | 0.983 |
| TabPFN (tabular foundation) | scaffolds_per_molecule | 41 | 0.071 | -0.28 to 0.41 | 0.661 | 0.755 |
| TabPFN (tabular foundation) | label_asymmetry | 41 | -0.126 | -0.42 to 0.18 | 0.428 | 0.710 |
| Chemprop v2 GNN | log10_n_train | 42 | -0.008 | -0.31 to 0.30 | 0.965 | 0.983 |
| Chemprop v2 GNN | internal_diversity | 42 | -0.050 | -0.37 to 0.28 | 0.752 | 0.840 |
| Chemprop v2 GNN | mean_snn | 42 | 0.131 | -0.19 to 0.43 | 0.405 | 0.710 |
| Chemprop v2 GNN | ood_fraction | 42 | -0.119 | -0.44 to 0.21 | 0.454 | 0.710 |
| Chemprop v2 GNN | scaffolds_per_molecule | 42 | 0.141 | -0.16 to 0.42 | 0.370 | 0.710 |
| Chemprop v2 GNN | label_asymmetry | 42 | 0.325 | -0.03 to 0.62 | 0.031 | 0.238 |
| MapLight + GNN | log10_n_train | 44 | -0.205 | -0.47 to 0.10 | 0.182 | 0.461 |
| MapLight + GNN | internal_diversity | 44 | 0.091 | -0.23 to 0.40 | 0.559 | 0.730 |
| MapLight + GNN | mean_snn | 44 | -0.071 | -0.38 to 0.24 | 0.643 | 0.753 |
| MapLight + GNN | ood_fraction | 44 | -0.113 | -0.42 to 0.22 | 0.467 | 0.710 |
| MapLight + GNN | scaffolds_per_molecule | 44 | 0.291 | 0.01 to 0.54 | 0.052 | 0.278 |
| MapLight + GNN | label_asymmetry | 44 | -0.090 | -0.39 to 0.25 | 0.563 | 0.730 |
| Deep tabular NN (ChemML MLP) | log10_n_train | 44 | -0.315 | -0.57 to 0.00 | 0.042 | 0.250 |
| Deep tabular NN (ChemML MLP) | internal_diversity | 44 | -0.251 | -0.51 to 0.04 | 0.100 | 0.401 |
| Deep tabular NN (ChemML MLP) | mean_snn | 44 | -0.017 | -0.33 to 0.31 | 0.912 | 0.952 |
| Deep tabular NN (ChemML MLP) | ood_fraction | 44 | 0.039 | -0.31 to 0.36 | 0.807 | 0.873 |
| Deep tabular NN (ChemML MLP) | scaffolds_per_molecule | 44 | 0.224 | -0.10 to 0.51 | 0.141 | 0.461 |
| Deep tabular NN (ChemML MLP) | label_asymmetry | 44 | 0.181 | -0.14 to 0.47 | 0.242 | 0.581 |
