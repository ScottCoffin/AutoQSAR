# Reproducing the QSARena benchmark and paper

This file collects what a third party needs to reproduce the results of the accompanying paper, and what can
and cannot be expected to match exactly. The paper's results are computed from
`benchmark_results/qsarena_benchmark_oof_ensemble/` (see Additional file 1, Note S1 for its provenance).

## 1. Regenerate every number, figure and table (no training)

Everything the paper quotes is recomputed from committed artifacts:

```bash
# benchmark environment (Python 3.11; nbformat/nbclient are needed for the notebook)
python portable_colab_qsar_bundle/render_manuscript_assets.py   # figures, tables, manuscript_numbers.json,
                                                                 # meta-analysis, Tables S13-S14, LaTeX tables
python portable_colab_qsar_bundle/verify_manuscript_numbers.py  # fails if text and artifacts disagree
```

`reanalysis_coverage_posthoc.py` (run by the render script) produces the common-subset family comparison and the
post-hoc descriptor-model bound (Tables S13-S14) from the committed `metrics.csv` files only.

Per-molecule `predictions.csv` files are gitignored for size. They are archived with the Zenodo release
(`dist/zenodo/<run>_predictions.tar.gz`, built by `python tools/build_zenodo_bundle.py`, which checks every file
against the run's committed `artifact_manifest.csv`). See `ZENODO.md`.
<!-- TODO(author): add the minted Zenodo version DOI here once the release is deposited. -->

## 2. Feature selection across machines

ElasticNetCV feature selection falls back to random-forest importance when it exceeds a wall-clock limit
(`--selector-elasticnet-timeout-seconds`, default 7,200 s). Which datasets reach the limit depends on the speed of
the machine: re-running selection on a consumer RTX 4060 workstation changed the selected features on 6 of the 44
datasets (Tox21, Ames, LD50, ESOL, CYP2C9 and AqSolDB). Two runner options remove this dependence.

**Reuse the deposited selection** (recommended for reproducing the paper):

```bash
qsarena-benchmark ... --selected-features-from benchmark_results/qsarena_benchmark_oof_ensemble
```

Each dataset then uses `<run>/<dataset>/selected_features.csv` (the A100 selection that every reported model was
trained on) instead of refitting the selector, and keeps the recorded selector method so nested cross-validation
refits the same method inside each fold. A dataset without a deposited selection, or a deposited feature missing
from the new feature matrix, is an error rather than a silent refit.

**Deterministic selection for new runs:**

```bash
qsarena-benchmark ... --deterministic-selection
```

This removes the wall-clock limit (no timeout-triggered fallback; large datasets simply take longer) and runs the
selector in a single thread with BLAS/OpenMP pools pinned to one thread. The dataset-size pre-check
(`--selector-auto-rf-by-dataset-size`) depends only on the training-set size and still applies if enabled. Both
options enter the stage 2/3 cache signature only when used, so existing caches are unaffected.

Never let a repair or resume run recompute stage 2/3 on a different machine: transfer `stage23_resume_cache.pkl`
or use `--selected-features-from` (AGENTS.md, "Feature selection is NOT reproducible across machines").

## 3. Other known sources of run-to-run difference

- **Single split and seed.** Every result comes from one split and one seed per dataset. Multi-seed evaluation is
  implemented but was not run for the paper; close margins are provisional.
- **Scaffold cross-validation folds.** `GroupKFold` breaks group ties differently across scikit-learn versions,
  so cross-validated values can differ by about 0.01-0.03 between environments; within one environment they are
  deterministic. Test-set results are unaffected.
- **GPU backends.** Chemprop runs seeded with warn-only determinism (torch has no deterministic CUDA `cumsum`), and
  Uni-Mol uses mixed precision; repeated GPU runs can differ in the last digits.
- **TabPFN.** Most TabPFN results came from the Prior Labs API (metered by tokens); local refits used the TabPFN
  package (v2.6 weights). The two backends need not agree exactly, which is why TabPFN is not an ensemble member.
- **Hardware.** The consumer-GPU rerun of the benchmark changed the best single-model score by a median of +0.50%
  on the 37 identically split datasets (paper Section 3.10, Table S5).
