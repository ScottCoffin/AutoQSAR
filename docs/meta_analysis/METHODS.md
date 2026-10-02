# Dataset-property meta-analysis: methods

Human-readable methods for `qsarena/meta_analysis/`, mirrored in manuscript §3.14 (whose numbers
are rendered from `manuscript_assets/meta_numbers.json`). The analysis is exploratory and
hypothesis-generating. It trains no model and uses only the deposited single-seed benchmark
artifacts (`docs/meta_analysis/INVENTORY.md`).

Regenerate everything with one command, which runs the notebook, then this analysis, then the
LaTeX tables:

```bash
python portable_colab_qsar_bundle/render_manuscript_assets.py --run-dir benchmark_results/qsarena_benchmark_oof_ensemble
python portable_colab_qsar_bundle/verify_manuscript_numbers.py
```

or only the meta-analysis: `python -m qsarena.meta_analysis --sync-manuscript` (about 4 minutes on CPU).

## Unit of analysis and outcome

- **Unit:** the dataset (n = 44). Every interval is a 95% percentile bootstrap over datasets
  (10,000 resamples, seed 0). None estimates seed variance: the benchmark is single-seed.
- **Primary outcome:** each family's relative gap to the per-dataset best,
  `|score - best| / |best|` on the ranking metric oriented so lower is better. This is exactly the
  Fig 6 quantity, read from the notebook's own export
  (`manuscript_assets/tables/figure6_family_best_models.csv`).
- **Secondary outcome:** within 5% of best (gap <= 5%); it reproduces Table 3.
- **The raw winner is never modelled as ground truth.** It is used once, grouped into four classes,
  as the recommender target, and the recommender is judged against permutation and majority
  baselines.

## Meta-features (Table S9)

All are computed on the benchmark's own train/test partition. The partition comes from each
dataset's `predictions.csv` and is checked against the recorded `split_train_hash` /
`split_test_hash` (all 44 match). Targets are on the scale the models saw.

| meta-feature | definition |
|---|---|
| `n_train`, `n_test`, `log10_n_train` | partition sizes |
| `pos_prevalence`, `imbalance_ratio` | classification: positives / n_train; max(p, 1-p) / min(p, 1-p) |
| `target_range`, `target_std`, `target_skew`, `target_kurtosis` | regression, training targets (SD with ddof 1; excess kurtosis) |
| `n_bemis_murcko_scaffolds`, `scaffolds_per_molecule`, `singleton_scaffold_frac` | core `murcko_scaffold_key` on training molecules; singleton fraction = share of distinct scaffolds with exactly one molecule. Acyclic molecules each get their own key, as in the benchmark's scaffold split. |
| `internal_diversity` | 1 - mean pairwise Tanimoto over distinct training pairs. Exact for n_train <= 2,000; otherwise 20,000 random distinct pairs with seed 0 (deterministic). |
| `mean_snn`, `median_snn` | SNN = max Tanimoto of a test molecule to any training molecule |
| `ood_fraction` | share of test molecules with SNN < 0.40 |
| `label_asymmetry` | log10(imbalance ratio) for classification, abs(skew) for regression. One predictor covers both tasks, so the recommender stays at four predictors. |

Fingerprints are Morgan radius 2 (ECFP4), 2,048 bits, bit-identical to the core's
`make_morgan_matrix` (asserted by a test). Tanimoto similarity on binary fingerprints is
1 - Jaccard distance: they are the same metric. `qsarena-applicability-domain` scores feature-space
AD and does not produce train-to-test Tanimoto distances, so SNN is computed locally, as the spec
allows.

## Statistics

- **Correlation grid (Table S10):** Spearman rho between each of six meta-features
  (log10 n_train, internal diversity, mean SNN, OOD fraction, scaffolds per molecule, label
  asymmetry) and each family's gap, with a bootstrap CI and a two-sided label-permutation p-value
  (10,000 permutations; p = (hits + 1) / (n + 1)). Benjamini-Hochberg q-values are computed over
  the whole grid; q < 0.10 is reported as surviving.
- **Families in trend fits:** families valid on at least 50% of datasets. The spec expected
  Chemprop to be valid on only 6/44 datasets and asked for it to be masked. After the Chemprop
  repair runs it is valid on 42/44, so the rule is applied generically and currently excludes no
  family. Its two missing cells stay NaN.
- **Size crossover:** each family's gap is fitted against log10 n_train with a Theil-Sen robust
  line. The crossover is where the two fitted lines intersect, reported only inside the observed
  size range. Bootstrap replicates give the fraction with a crossing in range and the percentile CI
  of the crossing size among them. The pre-registered contrast is conventional ML versus Uni-Mol
  (3D pretrained); ensembles versus Uni-Mol is secondary. Theil-Sen replaces LOESS: it is robust
  to the few heavy-tailed gaps, needs no new dependency, and gives a single crossing.
- **Difficulty:** Spearman of each meta-feature with the achievable best held-out metric (best
  test ROC-AUC for classification, best test R^2 for regression), within task and pooled on the
  within-task percentile rank.
- **Natural experiment (Fig 8b):** the seven datasets re-split from random or target-quartile to
  scaffold splits between the RTX 4060 run and the A100 lineage. Each family's gap is compared on
  identical chemistry. It is descriptive only: n = 7, and the runs also differ in hardware,
  benchmark profile and some backend settings (§3.10).
- **Recommender:** target = the winning family grouped into four classes (fusion = ensemble + CFA;
  descriptor-based ML = conventional, GA-tuned, MapLight + GNN, TabPFN, ChemML MLP; Uni-Mol;
  Chemprop), all with >= 4 datasets (the code refuses otherwise). The spec's grouping was
  (ensemble, conventional ML, Uni-Mol, other); with Chemprop now valid and winning 9 datasets,
  "other" would be mostly Chemprop, so Chemprop gets its own class. The four predictors are
  log10 n_train, mean SNN, label asymmetry and task. Models are a depth-3 decision tree
  (min 3 per leaf) and an L1 multinomial logistic regression (standardised, C = 1). Evaluation is
  leave-one-dataset-out balanced accuracy with a dataset-bootstrap CI, compared against the
  constant majority-class predictor (balanced accuracy = 1 / number of classes) and a 500-draw
  label-permutation null. A leave-one-out majority baseline is not used: with near-tied classes it
  flips whenever a leading-class dataset is held out, and scores about 0.
- **Fingerprint sensitivity:** the fingerprint-dependent features are recomputed with Morgan
  radius 3 (ECFP6) at 4,096 bits. The report gives the share of grid cells that keep their sign
  and the share that keep their BH significance classification.

## Honesty checklist

- The primary outcome is the continuous gap; winners are never ground truth.
- All CIs are dataset-bootstrap, and the text never implies seed-level variance.
- The meta-model has at most four predictors, depth <= 3 or L1, LODO evaluation and a permutation
  baseline, and is framed as exploratory.
- BH control is applied over the family x meta-feature grid, and q-values are reported.
- The diversity sensitivity check is reported.
- Every quantitative sentence in §3.14 is rendered from `meta_numbers.json` and carries its CI or
  q-value. Directional wording (narrowed / widened / no clear trend; consistent / outside the
  literature range) is chosen by the code from the CI, never written by hand.

## Phase 7 (optional learning curve)

Off by default and GPU-bound; `run_meta_analysis` never trains. Each learning-curve point keeps
the dataset's full benchmark test set fixed and uses a nested, seed-0 subset of its training
partition (`qsarena/meta_analysis/phase7.py`). The runner's `--row-limit` is not used because it
subsamples before splitting. The 15 points in `docs/meta_analysis/RTX_PHASE7_LEARNING_CURVE_PLAN.csv`
run through `tools/run_meta_phase7_rtx.ps1` into
`meta_phase7_gpu_patch/phase7_learning_curve_metrics.csv`. The full-size points are taken from the
benchmark run itself, which trained the same models on the identical split. `--include-phase7`
then adds Figure S2 and a per-dataset Uni-Mol-versus-best-tree crossover, interpolated in log n.
