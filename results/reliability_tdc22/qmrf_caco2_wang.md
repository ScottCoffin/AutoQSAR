# QMRF-style report: Random forest reference model (300 trees) on TDC ADMET Benchmark Group: caco2_wang

## 1. Defined endpoint

- **dataset**: TDC ADMET Benchmark Group: caco2_wang
- **endpoint**: Caco-2 cell effective permeability
- **task type**: regression
- **units**: log10(cm/s)
- **data source**: Therapeutics Data Commons, official train_val/test split (scaffold)

## 2. Unambiguous algorithm

- **model**: Random forest reference model (300 trees)
- **features**: Morgan fingerprint (radius 2, 2048 bits) + RDKit 2D descriptors
- **software**: qsarena 0.1.0
- **random seed**: 0
- **split**: strategy = TDC official scaffold split; train_hash = 729d7d7a0da3ebf611664245fc8509b21dec920925b88e5b8d800e156467839c; test_hash = 17c1867cfa548e0c99eac0a039bbc9b8510dbe3a449a724ee8513bb84bc9e300
- **reproduction**: python -m qsarena.reliability_study --out results/reliability_tdc22

## 3. Defined domain of applicability

- **method**: consensus of Roy standardization (RDKit descriptors, 3 SD) and 5-NN Tanimoto distance (95th percentile of training leave-one-out distances)
- **test coverage**: 0.7363
- **error in domain**: 0.3004
- **error out of domain**: 0.2327
- **reliability**: split-conformal 90% intervals, test coverage 0.951

## 4. Goodness-of-fit, robustness and predictivity

- **train**: n_fit = 582; out_of_bag_note = fit on 80% of train_val; 20% held for calibration
- **internal validation**: n_calibration = 146; conformal_quantile_source = calibration partition of train_val
- **test**: n_test = 182; mae = 0.2826
- **uncertainty**: alpha = 0.1; coverage = 0.9505

## 5. Mechanistic interpretation

- **status**: feature
- **note**: Impurity-based importances of the reference model; statistical association, not mechanism.

| rank | feature | importance |
|---:|---|---:|
| 1 | `rdkit_NHOHCount` | 0.1370 |
| 2 | `rdkit_NumHDonors` | 0.1368 |
| 3 | `rdkit_TPSA` | 0.0803 |
| 4 | `rdkit_VSA_EState3` | 0.0360 |
| 5 | `rdkit_qed` | 0.0251 |
| 6 | `rdkit_MolLogP` | 0.0249 |
| 7 | `rdkit_NOCount` | 0.0197 |
| 8 | `rdkit_PEOE_VSA1` | 0.0190 |
| 9 | `rdkit_VSA_EState8` | 0.0155 |
| 10 | `rdkit_BalabanJ` | 0.0140 |

## Caveats

- Reference model for applicability-domain and calibration analysis; not the benchmark's selected model.
- Single seed and single official split.
