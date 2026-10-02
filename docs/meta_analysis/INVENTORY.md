# Meta-analysis artifact inventory (Phase 0)

What the dataset-property meta-analysis (`qsarena/meta_analysis/`) reads, with exact paths. Every
backticked path below is checked for existence by `tests/meta/test_inventory.py`. That test fails,
never skips, so a wrong path stops the analysis instead of letting it invent data.

## Which run

The spec named the canonical A100 run, `benchmark_results/autoqsar_benchmark_20260623_153839`. The
manuscript has since moved to the out-of-fold ensemble rebuild of that run,
`benchmark_results/qsarena_benchmark_oof_ensemble`. It has the same base models plus working
Chemprop and TabPFN, and ensembles rebuilt from OOF predictions (AGENTS.md, "Run lineage"). Its
train/test partitions are byte-identical to the canonical run's (same split hashes). The
meta-analysis uses the OOF run so that its gap matrix is the one in the manuscript's Fig 6.

## Per-dataset metrics

`benchmark_results/qsarena_benchmark_oof_ensemble/<dataset>/metrics.csv`: 44 dataset folders, one
row per model attempt. Columns used here:

| column | meaning |
|---|---|
| `dataset`, `model`, `workflow` | identifiers |
| `primary_metric`, `primary_metric_value` | catalog metric; wrong for some binary tasks (see below) |
| `test_r2`, `test_roc_auc` (and `test_rmse`, `test_mae`, `test_spearman`, ...) | held-out metrics |
| `n_train`, `n_test`, `split_strategy`, `target_transform` | size and split protocol |
| `split_train_hash`, `split_test_hash` | split signatures (below) |
| `error` | non-empty = failed attempt; failed rows can coexist with a later valid retry |

There is no `family` or `is_valid` column. The model family is assigned by the notebook
(`classify_architecture_family`, then `pub_family`), and validity means an empty `error` with a
finite analysis metric. The catalog `primary_metric` says `rmse` for some binary tasks (e.g.
`tdc_herg_karim`). The analysis uses the notebook's `analysis_metric`, which infers
classification from strict 0/1 targets.

## Per-family-best gap (Fig 6)

Computed in the notebook export cell (`# MANUSCRIPT_FIGURE_EXPORT`, cell 23 of
`portable_colab_qsar_bundle/benchmark_results_summary.ipynb`):
`relative_gap_to_best = |comparison_score - best| / |best|`, where `comparison_score` is the analysis
metric oriented so lower is better. `family_best` keeps each family's best model per dataset, and
Fig 6 pivots it (min over the family, times 100). The export cell also writes both frames:

- `manuscript_assets/tables/figure6_family_best_models.csv`: `family_best` (dataset, family, model,
  workflow, task_kind, suite, analysis_metric, analysis_metric_direction, analysis_metric_value,
  comparison_score, relative_gap_to_best, rank_within_dataset).
- `manuscript_assets/tables/figure6_family_gap_matrix.csv`: the Fig 6 heatmap values (%).
- `manuscript_assets/tables/figure6_family_best_models_comparison_run.csv`: the same frame for the
  RTX 4060 comparison run (`benchmark_results/benchmark_name_date`), used for the re-split natural
  experiment.

`qsarena/meta_analysis/gap_matrix.py` reads these files and never re-derives the gap.

## Splits

Split signatures: `run_qsarena_benchmarks.build_split_signature` hashes the ordered, stripped
train and test SMILES (`smiles_hash`, sha256 of newline-joined SMILES) into `split_train_hash` /
`split_test_hash`. No split indices are persisted beyond that. The partition itself is in each
dataset's `predictions.csv` (columns `dataset, model, workflow, split, row_index, smiles, observed,
predicted, oof_signature`; `split` is `train`, `test` or `oof`). `observed` is on the transformed
scale the models saw (e.g. log10 for `chemml_organic_density`). Those files are gitignored, so
`python -m qsarena.meta_analysis --build-partitions` extracts one model's train/test rows per dataset into
`data/meta_analysis/dataset_partitions.csv.gz` (dataset, split, row_index, smiles, observed;
156,052 rows). It refuses to write unless every partition reproduces both recorded hashes, and
all 44 do.

## Dataset registry and split protocols

`data/benchmark_dataset_catalog.csv` (loaded by `portable_colab_qsar_bundle/benchmark_registry.py`)
holds one row per dataset. Relevant columns: `dataset, benchmark_suite, benchmark_id,
recommended_split, recommended_metric, smiles_column, target_column, predefined_split_column,
local_train_count`. Size, task and split used here come from the run itself (`metrics.csv`), not
the catalog. Split protocols in the run: 27 `predefined`, 12 `scaffold`, 4 `target_quartiles`,
1 `random`.

## Feature selection (Fig 4)

`benchmark_results/qsarena_benchmark_oof_ensemble/<dataset>/selected_features.csv` has one
`feature` column, and `selector_coefficients.csv` has `feature, coefficient, abs_coefficient`.
Families come from the feature-name prefix, as in the notebook's `feature_family_from_name`
(mirrored in `qsarena.meta_analysis.io.FEATURE_FAMILY_PREFIXES`). MapLight classic =
avalon + erg + maplight. The `selected_feature_families_json` column in `metrics.csv` lists the
families that were *built*, not selected counts.

## Entry points and reuse

- Scaffolds: `murcko_scaffold_key` in `portable_colab_qsar_bundle/qsar_workflow_core.py` (the
  benchmark's scaffold-split key; acyclic molecules get their own key).
- Fingerprints: `make_morgan_matrix` in the same module. The meta-analysis calls the same RDKit
  generator without the per-column DataFrame build; a test asserts the bits are identical.
- Tanimoto: `tanimoto_distance_matrix` in `qsarena/applicability_domain.py`.
- Applicability domain CLI: `qsarena-applicability-domain`
  (`portable_colab_qsar_bundle/simple_applicability_domain.py`). It scores feature-space AD
  (standardization and model confidence; per-dataset output `applicability_domain.csv`). It does
  not produce train-to-test Tanimoto distances, so SNN uses the local ECFP4 nearest-neighbour
  fallback the spec allows.
- Manuscript numbers: the notebook writes `manuscript_assets/manuscript_numbers.json`;
  `portable_colab_qsar_bundle/verify_manuscript_numbers.py` asserts values from it against the
  prose; `portable_colab_qsar_bundle/render_manuscript_assets.py` runs the notebook and then the
  meta-analysis, which writes `manuscript_assets/meta_numbers.json` and renders the §3.14 text
  into `manuscript.md` and `submission/body.tex` between `META` markers
  (`qsarena/meta_analysis/text.py`). The verifier re-renders and fails on drift.
