# Family selector v2: pre-specified plan

Written 2026-09-29 before any v2 result was computed. It will not be edited after results exist;
deviations go in a dated addendum at the bottom.

## Why v1 looked discouraging, and what was wrong with judging it that way

v1 predicted the *winning family group* and scored balanced accuracy against chance (0.26 vs
0.25). That target is the noisiest quantity in the benchmark. Many datasets have two or three
families within 1-2% of each other, so "picking the wrong winner" usually costs almost nothing.
Algorithm selection (Meta-QSAR, Olier et al. 2018) is judged on **performance lost by the chosen
algorithm**, not on classification accuracy of the winner label.

## Primary evaluation (fixed now)

- **Regret** of a selector on a held-out dataset = the Fig 6 relative gap (%) of the family it
  recommends. Families that produced no valid result on that dataset are not recommendable there
  (in practice the user would see the failure and fall back).
- Leave-one-dataset-out (LODO). Everything a selector learns, including standardisation,
  imputation, hyperparameters and feature choice, is fitted on the 43 training datasets only.
- Summaries: mean regret (primary), median regret, share of datasets where the pick is within 5%
  of best, and gap closed = (SBS - selector) / (SBS - oracle) on mean regret.
- **Primary comparator:** the LODO single best solver (SBS), i.e. the family with the lowest mean
  gap on the 43 training datasets. Other references: oracle (0), random valid family (expected
  value), always-ensemble.
- **Headline:** a *nested* LODO. For each held-out dataset, an inner LODO over the other 43 picks
  one variant from the grid below by inner mean regret, and that variant is then fitted and applied
  to the held-out dataset. It is reported with a paired dataset-bootstrap CI of (SBS - nested)
  regret and a sign-flip permutation p-value. The per-variant table is shown in full and labelled
  as exploratory. No single grid cell is reported as the headline.

## Feature blocks

All are train-partition-only; nothing uses test labels.

- **F0 (v1):** log10 n_train, mean SNN, label asymmetry, is_classification.
- **F1 chemistry:** training-set means of MolWt, MolLogP, TPSA, FractionCSP3, aromatic ring count,
  and SD of MolWt.
- **F2 label landscape:** k = 5 Tanimoto nearest neighbours within the training set. Regression:
  leave-one-out kNN R^2 (similarity-weighted), a smoothness / modelability index. Classification:
  MODI (mean over classes of the share of molecules whose nearest neighbour has the same class).
  Both: the cliff fraction, i.e. nearest-neighbour pairs with Tanimoto >= 0.7 whose labels differ
  (different class, or |delta y| > 1 training SD).
- **F3 landmarkers** from the benchmark's own training-set CV (conventional ML and ChemML MLP):
  relative CV gap of the best linear model versus the best tree ensemble (non-linearity), of the
  KNN+SVM voting model versus the best tree (locality), of the ChemML MLP versus the best tree
  (deep tabular versus tree), and the coefficient of variation of CV scores across conventional
  models (how much model choice matters).

## Selectors in the grid

Each predicts a gap vector over families and recommends the argmin among valid families.

| id | model | feature blocks |
|---|---|---|
| sbs | single best solver | none |
| knn | k-nearest datasets (k = 5, distance-weighted, standardised features), mean of neighbours' gap vectors (log1p scale) | F0; F0+F1; F0+F2; F0+F3; F0-F3 |
| ridge | one ridge per family on log1p(gap), alpha = 3 | same five blocks |
| rf | one random forest per family (200 trees, depth 3, min leaf 3) | same five blocks |
| tree_cls | v1 depth-3 tree on grouped winner (for continuity) | F0 |

That makes 1 + 15 + 1 = 17 variants, and the nested headline chooses among the 16 feature-using
ones plus SBS.

## Secondary (reported, not headline)

- Per-family "within 5%" prediction accuracy.
- Whether F3 landmarkers dominate. They are cheap train-only CV evidence, so a gain from them means
  the recommendation should come from a quick CV screen more than from dataset descriptors.

## Guards

- Recommendation families: the 8 manuscript families. The GA-tuned family is absent in this run.
- Permutation p for the nested headline: 200 permutations of which dataset's gap vector goes with
  which meta-feature row, running the whole nested pipeline each time.
- All numbers go to `meta_numbers.json["selector_v2"]` and Table S11. The §3.14 recommender
  paragraph is re-rendered from them with wording decided by the CI, whatever the outcome.

## Addendum 1 (2026-09-29, before any v2 result)

- F3 gains a fifth landmarker, the relative CV gap of TabPFN versus the best tree ensemble. TabPFN
  turned out to carry training-set CV scores too. Landmark scores use each dataset's `cv_primary`
  (its primary metric, oriented so that a positive gap means worse than the best tree).

## Addendum 2 (2026-09-29, before any v2 result; compute-driven)

- The nested headline chooses among SBS, the five kNN variants, the five ridge variants and the v1
  tree. The five random-forest variants appear in the full per-variant table but are not nested
  candidates. Nesting refits every candidate about 1,900 times (44 outer x 43 inner), plus the same
  again for every permutation. With eight per-family forests per fit, that would take hours on
  this laptop while the GPU run is using the CPU.
- The permutation test for the nested headline uses 100 permutations instead of 200, for the same
  reason (minimum attainable p = 0.0099).
