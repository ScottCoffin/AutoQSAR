# Nested-selection CV: removing the feature-selection leak from the benchmark

**Status (2026-10-06): DONE; the paper is regenerated from the nested run.** It ran with the author's go-ahead ("start the full leakage-fix run"). Reopened on
2026-10-03 after the controlled test below confirmed the leak (declined on 2026-10-02 on cost). Launcher:
`logs/run_nested_selection.ps1`, pilot on `tdc_caco2_wang`, then all 44 datasets. TabPFN's fold refits use the local
`tabpfn` package (no API credits); TabPFN gets nested CV metrics but is kept out of the ensembles with
`--ensemble-exclude-model`, because most of its full-fit predictions came from the Prior Labs API. Above
`--tabpfn-local-max-cells` (1.5M rows x selected features) local TabPFN cannot run on the 8 GB GPU, so its CV
metrics there are withdrawn (`cv_selection=outer_withdrawn`) instead of left leaky. Pilot (caco2, 18 min): test
metrics unchanged; outer CV overstatement +3% to +28% became -15% to +1%. Full run started 03:16 PDT.

## The problem, now measured causally

Stage 3 (`select_features`) fits a supervised selector on **all** training rows (ElasticNetCV on 34 datasets,
random-forest importance after an ElasticNetCV timeout on 10). Every model's CV then runs on that pre-selected
matrix, so each validation fold's labels helped choose its features.

`qsarena/feature_expansion/selection_leak.py` holds everything fixed (the full ADMETboost features, models, 5
folds, test split) and varies only where the selector is fitted. Output:
`benchmark_results/qsarena_feature_expansion/selection_leak.csv`. Final medians over all 9 small regression
datasets with predefined or scaffold test splits (completed 2026-10-03):

| model | CV overstatement, selection outside folds | selection inside folds | leak | datasets where leak > 0 | Wilcoxon p |
|---|---|---|---|---|---|
| ElasticNetCV | 17.7% | -4.8% | 22.7 points | 9/9 | 0.004 |
| Random forest | 2.8% | -3.4% | 6.1 points | 9/9 | 0.004 |
| SVR | -4.9% | -7.9% | 2.5 points | 8/9 | 0.055 |
| all pooled | | | 7.4 points | 26/27 | < 0.001 |

Nested selection makes CV roughly honest (slightly pessimistic). Test metrics are unaffected either way. The
deployed model legitimately uses the selection fitted on all training rows.

## What the leak touches

| Affected | Why | Fix |
|---|---|---|
| CV metrics of every model trained on selected features (conventional ML, gradient boosting except MapLight CatBoost, TabPFN, ChemML MLPs) | CV runs on the pre-selected matrix | refit selection per fold, using the dataset's recorded selector method |
| The CV-selected ("honest") pick, the CV-vs-test gap, the 35 -> 28 -> 27 decomposition (the CV leg) | built on those CV scores | follow from the fixed CV |
| OOF predictions of those members, and so ensemble selection, weights and stacking | the OOF predictions come from the same folds | the same per-fold selections |
| Chemprop "D-MPNN + Selected descriptors" OOF | its fold refits use the full-train descriptor selection | pass each fold's own selection as descriptors |
| CFA fusion | ranks candidates by in-sample **training** error (a related in-sample issue, not the selection leak) | rank on OOF predictions instead (cheap once OOF exist) |
| Colab notebook (`build_colab_qsar_tutorial.py`) | same select-then-CV pattern | the same nested option |

**Not affected:** test metrics and leaderboard placements, Uni-Mol, the other four Chemprop variants, MapLight +
GNN and MapLight CatBoost (none of them use the selected matrix), and `XGBoost (ADMETboost features)` (no
selection).

## Implementation status (2026-10-04)

**Code done; the benchmark run started 2026-10-05.**
- **Runner:** `--cv-selection {outer,nested}` (RunConfig `feature_selection.cv_selection`). The default is nested
  for the `full` and `cost_optimized` profiles and outer for `quick`. `nested_selection_columns()` refits the
  stage-3 selector per fold with the deployed method pinned; selections are cached in
  `<dataset>/nested_selection/fold_<k>.json`. `run_nested_selection()` runs inside the ensemble OOF stage, before
  the regular OOF pass. It refits every selected-feature member on its fold's own columns (conventional models and
  TabPFN when not API-billed, ChemML MLPs, and with `--ensemble-oof-scope all` the Chemprop selected-descriptor
  variant). Fold predictions are cached in `<dataset>/nested_selection/oof_folds/`. It overwrites those members'
  `split="oof"` rows and patches their metrics rows: `cv_*` from fold-averaged nested predictions, the old value
  kept as `cv_primary_outer`, and `cv_selection=nested` plus `cv_selection_signature` as the resume marker.
  `cv_selection` enters only the ensemble family's resume signature, and only when it is not outer, so outer runs
  resume unchanged. Tests: `tests/unit/test_nested_selection.py`. On the example data, ElasticNetCV's CV RMSE moved
  from 0.384 (outer) to 0.798 (nested) against a test RMSE of 0.676.
- **Notebooks:** block 4C option `nested_selection_cv` (default on; evidence entry in `notebook_default_evidence.py`).
  4B keeps `STATE["train_only_selector_refit"]` (all four selector methods). 4C cross-validates selected-matrix
  models with per-fold selection, and 7A refits conventional and tuned members on per-fold selections (OOF cache
  key `<fold signature>|nested`). Executed end to end on 2026-10-06 (`tools/notebook_nested_check.py`, FreeSolv and Caco-2): nested CV engages, test
  metrics are unchanged, and 7A uses the nested folds. Binary classification was added to the notebook the same day and passes the same check (HIA, BBB-Martins).
- **Applying it to the manuscript run:** re-run the OOF launcher (`tools/run_oof_ensemble_rtx.ps1` flags:
  `--only-model-names Ensemble --rebuild-ensemble`, scope `all`) with `--cv-selection nested` and
  `--run-admetboost-xgboost`. Base models are not retrained. The nested stage patches CV metrics and OOF, then the
  ensembles are rebuilt once with the new member.
- **Still open:** CFA ranks candidates by in-sample training error (related, not the selection leak). Moving it to
  OOF predictions needs the CFA stage after the OOF stage; not done.

## Design

- New option `--cv-selection {outer,nested}`; `nested` becomes the default for new runs. A dataset's stage 2/3
  signature is unchanged, so cached full-train selections and test results are reused.
- **Pin the selector method per dataset** to the one recorded in `metrics.csv` (`selector_method`). The nested fits
  then repeat exactly the deployed pipeline, and the 7,200 s ElasticNetCV timeout and its hardware-dependent RF
  fallback cannot change the method mid-run (the cross-machine reproducibility trap in AGENTS.md).
- One selection per fold serves both the CV metrics and the OOF predictions (the folds are identical). Cache it as
  `<dataset>/nested_selection/fold_<k>.json`, keyed by the stage 2/3 signature, so the stage resumes.
- `evaluate_model` gains per-fold feature matrices; the ensemble OOF refitters and the Chemprop selected-descriptor
  fold refits read the same per-fold selections. ChemML's internal CV gets the same treatment.

## Cost on the RTX 4060 workstation (16 cores)

| Step | Estimate | Basis |
|---|---|---|
| Code + tests (runner, notebook builder) | one working session | no compute |
| Pilot on one mid-size dataset, to calibrate | ~1 h | do this before the full run |
| Nested selector refits, 5 folds x 44 | **~34 h CPU** (+/-50%) | ~27 h for 34 ElasticNetCV datasets (~10 min per fit at 900 rows, scaling n^0.77, from the controlled test's timings) + ~7 h for 10 RF-importance datasets |
| Refit affected models on each fold | ~15 h CPU | same work as the 2026-10-01 CPU OOF rebuild |
| Chemprop selected-descriptor fold refits | ~8-10 h GPU | 220 fold trainings; runs in parallel with the CPU work |
| Ensemble rebuild + render + manuscript update | ~3 h | as in the 2026-10-02 refresh |
| **Total** | **~50-55 h CPU-bound, about 2-3 days of wall-clock** | GPU work overlaps |

A Jetstream2 A100 node (32 cores) would roughly halve the CPU part, but the allocation is nearly spent.

## What changes in the paper

- The CV metrics, the CV-selected picks, the CV-vs-test gap and the CV leg of the decomposition. The
  matched-candidate-set leg is test-selected, so it does not change.
- Ensembles whose members used selected features (most of them), and so the win counts.
- The `limitation_cvleak` caveat becomes a Methods sentence ("feature selection is refitted inside every CV fold").
  The controlled test can be reported as the evidence that this mattered.
- The verifier's expected values move through the render, as before; never loosen a check.

## Sequencing with the ADMETboost fold-in

`XGBoost (ADMETboost features)` was approved as a benchmark member on 2026-10-03. Its base model (no selection,
honest CV) is being trained into `qsarena_benchmark_oof_ensemble` now (`logs/run_foldin_admetboost.ps1`, after the
controlled test finishes). The ensembles are **not** rebuilt yet, so they are rebuilt only once, after the nested
selection run. If the nested run is not approved, rebuild them right after the fold-in instead.
