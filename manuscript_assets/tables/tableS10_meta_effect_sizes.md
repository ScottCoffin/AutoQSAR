| Model family | Meta-feature | Datasets | Spearman rho | 95% CI | Permutation p | BH q |
|---|---|---|---|---|---|---|
| Conventional ML | log10_n_train | 44 | -0.024 | -0.32 to 0.31 | 0.875 | 0.942 |
| Conventional ML | internal_diversity | 44 | -0.316 | -0.60 to 0.00 | 0.036 | 0.208 |
| Conventional ML | mean_snn | 44 | 0.226 | -0.10 to 0.53 | 0.144 | 0.434 |
| Conventional ML | ood_fraction | 44 | -0.332 | -0.61 to 0.00 | 0.028 | 0.208 |
| Conventional ML | scaffolds_per_molecule | 44 | 0.331 | 0.03 to 0.57 | 0.031 | 0.208 |
| Conventional ML | label_asymmetry | 44 | -0.191 | -0.45 to 0.11 | 0.215 | 0.465 |
| Ensemble (stacking / averaging) | log10_n_train | 44 | -0.401 | -0.66 to -0.09 | 0.005 | 0.208 |
| Ensemble (stacking / averaging) | internal_diversity | 44 | 0.047 | -0.28 to 0.35 | 0.756 | 0.879 |
| Ensemble (stacking / averaging) | mean_snn | 44 | -0.314 | -0.58 to -0.00 | 0.039 | 0.208 |
| Ensemble (stacking / averaging) | ood_fraction | 44 | 0.325 | -0.00 to 0.61 | 0.032 | 0.208 |
| Ensemble (stacking / averaging) | scaffolds_per_molecule | 44 | 0.106 | -0.17 to 0.37 | 0.491 | 0.732 |
| Ensemble (stacking / averaging) | label_asymmetry | 44 | 0.325 | 0.00 to 0.60 | 0.032 | 0.208 |
| CFA combinatorial fusion | log10_n_train | 38 | 0.072 | -0.26 to 0.40 | 0.653 | 0.804 |
| CFA combinatorial fusion | internal_diversity | 38 | -0.113 | -0.45 to 0.23 | 0.497 | 0.732 |
| CFA combinatorial fusion | mean_snn | 38 | 0.305 | -0.03 to 0.59 | 0.059 | 0.259 |
| CFA combinatorial fusion | ood_fraction | 38 | -0.320 | -0.63 to 0.04 | 0.049 | 0.236 |
| CFA combinatorial fusion | scaffolds_per_molecule | 38 | 0.140 | -0.18 to 0.45 | 0.398 | 0.682 |
| CFA combinatorial fusion | label_asymmetry | 38 | 0.162 | -0.17 to 0.46 | 0.330 | 0.633 |
| Uni-Mol (3D pretrained) | log10_n_train | 44 | -0.144 | -0.45 to 0.17 | 0.353 | 0.652 |
| Uni-Mol (3D pretrained) | internal_diversity | 44 | 0.243 | -0.11 to 0.57 | 0.114 | 0.400 |
| Uni-Mol (3D pretrained) | mean_snn | 44 | 0.006 | -0.31 to 0.31 | 0.971 | 0.971 |
| Uni-Mol (3D pretrained) | ood_fraction | 44 | -0.085 | -0.40 to 0.25 | 0.584 | 0.757 |
| Uni-Mol (3D pretrained) | scaffolds_per_molecule | 44 | 0.016 | -0.32 to 0.36 | 0.919 | 0.942 |
| Uni-Mol (3D pretrained) | label_asymmetry | 44 | 0.042 | -0.28 to 0.37 | 0.781 | 0.879 |
| TabPFN (tabular foundation) | log10_n_train | 44 | 0.243 | -0.13 to 0.55 | 0.117 | 0.400 |
| TabPFN (tabular foundation) | internal_diversity | 44 | -0.283 | -0.55 to 0.02 | 0.066 | 0.262 |
| TabPFN (tabular foundation) | mean_snn | 44 | 0.192 | -0.11 to 0.47 | 0.223 | 0.465 |
| TabPFN (tabular foundation) | ood_fraction | 44 | -0.139 | -0.45 to 0.20 | 0.371 | 0.660 |
| TabPFN (tabular foundation) | scaffolds_per_molecule | 44 | 0.106 | -0.23 to 0.43 | 0.487 | 0.732 |
| TabPFN (tabular foundation) | label_asymmetry | 44 | -0.211 | -0.49 to 0.09 | 0.163 | 0.435 |
| Chemprop v2 GNN | log10_n_train | 42 | -0.080 | -0.37 to 0.23 | 0.613 | 0.774 |
| Chemprop v2 GNN | internal_diversity | 42 | 0.041 | -0.28 to 0.37 | 0.787 | 0.879 |
| Chemprop v2 GNN | mean_snn | 42 | -0.016 | -0.33 to 0.30 | 0.922 | 0.942 |
| Chemprop v2 GNN | ood_fraction | 42 | -0.020 | -0.34 to 0.29 | 0.898 | 0.942 |
| Chemprop v2 GNN | scaffolds_per_molecule | 42 | 0.043 | -0.25 to 0.34 | 0.785 | 0.879 |
| Chemprop v2 GNN | label_asymmetry | 42 | 0.333 | -0.00 to 0.61 | 0.027 | 0.208 |
| MapLight + GNN | log10_n_train | 44 | -0.213 | -0.46 to 0.08 | 0.161 | 0.435 |
| MapLight + GNN | internal_diversity | 44 | 0.154 | -0.17 to 0.45 | 0.318 | 0.633 |
| MapLight + GNN | mean_snn | 44 | -0.102 | -0.40 to 0.21 | 0.506 | 0.732 |
| MapLight + GNN | ood_fraction | 44 | -0.096 | -0.39 to 0.22 | 0.535 | 0.732 |
| MapLight + GNN | scaffolds_per_molecule | 44 | 0.222 | -0.07 to 0.49 | 0.145 | 0.434 |
| MapLight + GNN | label_asymmetry | 44 | -0.092 | -0.39 to 0.24 | 0.549 | 0.732 |
| Deep tabular NN (ChemML MLP) | log10_n_train | 44 | -0.376 | -0.62 to -0.08 | 0.013 | 0.208 |
| Deep tabular NN (ChemML MLP) | internal_diversity | 44 | -0.194 | -0.46 to 0.10 | 0.209 | 0.465 |
| Deep tabular NN (ChemML MLP) | mean_snn | 44 | -0.098 | -0.40 to 0.22 | 0.530 | 0.732 |
| Deep tabular NN (ChemML MLP) | ood_fraction | 44 | 0.096 | -0.24 to 0.41 | 0.540 | 0.732 |
| Deep tabular NN (ChemML MLP) | scaffolds_per_molecule | 44 | 0.198 | -0.13 to 0.49 | 0.193 | 0.465 |
| Deep tabular NN (ChemML MLP) | label_asymmetry | 44 | 0.200 | -0.12 to 0.49 | 0.198 | 0.465 |
