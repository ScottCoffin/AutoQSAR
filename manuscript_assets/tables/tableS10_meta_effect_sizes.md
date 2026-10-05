| Model family | Meta-feature | Datasets | Spearman rho | 95% CI | Permutation p | BH q |
|---|---|---|---|---|---|---|
| Conventional ML | log10_n_train | 44 | 0.045 | -0.29 to 0.36 | 0.768 | 0.922 |
| Conventional ML | internal_diversity | 44 | -0.326 | -0.60 to -0.01 | 0.029 | 0.222 |
| Conventional ML | mean_snn | 44 | 0.226 | -0.10 to 0.51 | 0.143 | 0.430 |
| Conventional ML | ood_fraction | 44 | -0.277 | -0.55 to 0.04 | 0.071 | 0.342 |
| Conventional ML | scaffolds_per_molecule | 44 | 0.148 | -0.17 to 0.43 | 0.330 | 0.587 |
| Conventional ML | label_asymmetry | 44 | 0.178 | -0.13 to 0.45 | 0.248 | 0.509 |
| Ensemble (stacking / averaging) | log10_n_train | 44 | -0.445 | -0.70 to -0.13 | 0.001 | 0.062 |
| Ensemble (stacking / averaging) | internal_diversity | 44 | 0.037 | -0.29 to 0.35 | 0.808 | 0.923 |
| Ensemble (stacking / averaging) | mean_snn | 44 | -0.328 | -0.59 to -0.02 | 0.032 | 0.222 |
| Ensemble (stacking / averaging) | ood_fraction | 44 | 0.339 | 0.01 to 0.62 | 0.026 | 0.222 |
| Ensemble (stacking / averaging) | scaffolds_per_molecule | 44 | 0.151 | -0.14 to 0.42 | 0.325 | 0.587 |
| Ensemble (stacking / averaging) | label_asymmetry | 44 | 0.338 | 0.03 to 0.60 | 0.026 | 0.222 |
| CFA combinatorial fusion | log10_n_train | 44 | -0.240 | -0.47 to 0.02 | 0.119 | 0.430 |
| CFA combinatorial fusion | internal_diversity | 44 | -0.307 | -0.58 to 0.01 | 0.047 | 0.280 |
| CFA combinatorial fusion | mean_snn | 44 | 0.011 | -0.31 to 0.32 | 0.946 | 0.966 |
| CFA combinatorial fusion | ood_fraction | 44 | 0.016 | -0.30 to 0.32 | 0.922 | 0.962 |
| CFA combinatorial fusion | scaffolds_per_molecule | 44 | 0.200 | -0.10 to 0.47 | 0.186 | 0.496 |
| CFA combinatorial fusion | label_asymmetry | 44 | 0.111 | -0.21 to 0.41 | 0.481 | 0.739 |
| Uni-Mol (3D pretrained) | log10_n_train | 44 | -0.171 | -0.46 to 0.14 | 0.272 | 0.521 |
| Uni-Mol (3D pretrained) | internal_diversity | 44 | 0.278 | -0.07 to 0.60 | 0.070 | 0.342 |
| Uni-Mol (3D pretrained) | mean_snn | 44 | -0.020 | -0.33 to 0.28 | 0.894 | 0.962 |
| Uni-Mol (3D pretrained) | ood_fraction | 44 | -0.062 | -0.37 to 0.26 | 0.690 | 0.872 |
| Uni-Mol (3D pretrained) | scaffolds_per_molecule | 44 | 0.032 | -0.31 to 0.38 | 0.841 | 0.939 |
| Uni-Mol (3D pretrained) | label_asymmetry | 44 | 0.018 | -0.30 to 0.35 | 0.908 | 0.962 |
| TabPFN (tabular foundation) | log10_n_train | 44 | 0.241 | -0.13 to 0.55 | 0.118 | 0.430 |
| TabPFN (tabular foundation) | internal_diversity | 44 | -0.255 | -0.54 to 0.05 | 0.101 | 0.430 |
| TabPFN (tabular foundation) | mean_snn | 44 | 0.179 | -0.12 to 0.47 | 0.254 | 0.509 |
| TabPFN (tabular foundation) | ood_fraction | 44 | -0.131 | -0.44 to 0.20 | 0.397 | 0.681 |
| TabPFN (tabular foundation) | scaffolds_per_molecule | 44 | 0.101 | -0.24 to 0.42 | 0.508 | 0.739 |
| TabPFN (tabular foundation) | label_asymmetry | 44 | -0.203 | -0.48 to 0.10 | 0.181 | 0.496 |
| Chemprop v2 GNN | log10_n_train | 42 | -0.096 | -0.39 to 0.21 | 0.548 | 0.774 |
| Chemprop v2 GNN | internal_diversity | 42 | 0.074 | -0.25 to 0.40 | 0.635 | 0.824 |
| Chemprop v2 GNN | mean_snn | 42 | -0.039 | -0.35 to 0.28 | 0.805 | 0.923 |
| Chemprop v2 GNN | ood_fraction | 42 | -0.004 | -0.33 to 0.30 | 0.979 | 0.979 |
| Chemprop v2 GNN | scaffolds_per_molecule | 42 | 0.053 | -0.25 to 0.34 | 0.733 | 0.902 |
| Chemprop v2 GNN | label_asymmetry | 42 | 0.322 | -0.01 to 0.60 | 0.032 | 0.222 |
| MapLight + GNN | log10_n_train | 44 | -0.226 | -0.48 to 0.07 | 0.138 | 0.430 |
| MapLight + GNN | internal_diversity | 44 | 0.179 | -0.15 to 0.47 | 0.246 | 0.509 |
| MapLight + GNN | mean_snn | 44 | -0.112 | -0.41 to 0.20 | 0.468 | 0.739 |
| MapLight + GNN | ood_fraction | 44 | -0.087 | -0.39 to 0.23 | 0.574 | 0.787 |
| MapLight + GNN | scaffolds_per_molecule | 44 | 0.230 | -0.06 to 0.50 | 0.130 | 0.430 |
| MapLight + GNN | label_asymmetry | 44 | -0.080 | -0.38 to 0.25 | 0.602 | 0.803 |
| Deep tabular NN (ChemML MLP) | log10_n_train | 44 | -0.392 | -0.63 to -0.10 | 0.009 | 0.222 |
| Deep tabular NN (ChemML MLP) | internal_diversity | 44 | -0.178 | -0.45 to 0.12 | 0.251 | 0.509 |
| Deep tabular NN (ChemML MLP) | mean_snn | 44 | -0.110 | -0.40 to 0.20 | 0.478 | 0.739 |
| Deep tabular NN (ChemML MLP) | ood_fraction | 44 | 0.108 | -0.23 to 0.42 | 0.494 | 0.739 |
| Deep tabular NN (ChemML MLP) | scaffolds_per_molecule | 44 | 0.193 | -0.13 to 0.49 | 0.208 | 0.499 |
| Deep tabular NN (ChemML MLP) | label_asymmetry | 44 | 0.196 | -0.12 to 0.48 | 0.206 | 0.499 |
