# Work order: one shared ensemble implementation for the runner and the Colab notebook

**For:** a coding agent working on `main` of this repository, on a workstation. Needs no GPU and no
Jetstream2 allocation.
**Read first:** `AGENTS.md`, especially "Data and analysis traps" (the ensemble bullets) and "Run
lineage".
**Status:** not started. Written 2026-09-26.

---

## 1. Why

The command-line runner and the Colab notebook each have their **own copy** of the ensemble code.
The copies drifted: the notebook kept leaking the test set (member choice, member filtering, CFA
inputs and the downstream strategy were all chosen on test metrics) after the runner's leak had
been fixed. Both copies are now leak-free and use out-of-fold (OOF) predictions, but they are
still separate implementations that can drift again, and they still differ in behaviour (§3).

The manuscript (§2.1, in both `manuscript.md` and `submission/body.tex`) says both entry points
call "the same feature-generation, splitting, fusion and evaluation library, so interactive and
batch results are produced by identical code paths". That is true for features, splits and
feature selection, which live in `qsar_workflow_core.py`. It is **not** true for ensembles.

**Goal:** move the model-agnostic ensemble logic into `portable_colab_qsar_bundle/qsar_workflow_core.py`
and have both entry points call it, so the ensemble code path really is shared. What each entry
point legitimately owns stays where it is: how members are collected, how fold models are
refitted, and persistence.

## 2. Where the code is today

**Runner:** `portable_colab_qsar_bundle/run_qsarena_benchmarks.py`. Search by name; line numbers
drift.

| Function | Role | Moves to core? |
|---|---|---|
| `is_fusion_payload` | CFA/ensemble outputs are never members | yes |
| `ensemble_oof_folds` | materialise K folds with the CV geometry (wraps `make_qsar_cv_splitter`, already in core) | yes |
| `ensemble_oof_fold_signature` | hash of the fold assignment | yes |
| `load_unimol_saved_oof` | read Uni-Mol's `cv.data`, inverse-scale with `target_scaler.ss`, alignment guards | yes |
| `ensure_ensemble_oof_predictions` | attach `payload["oof"]`: providers, then per-fold refits with on-disk fold cache | yes; the cache must become optional (§4) |
| `build_ensemble_result` | member filtering, correlated-pair tie-break, stacking, weighted and simple averages, three selection modes | the logic, yes; keep a thin wrapper with the same name and signature |
| `run_ensemble_oof_stage` (nested inside `run_dataset`) | builds runner-specific refitters (conventional, ChemML, Chemprop, Uni-Mol, MapLight+GNN) and providers; persists `split="oof"` rows | **no, stays in the runner** |
| ensemble stage in `run_dataset`, `ensemble_model_name_matches_method`, CLI flags, `_FAMILY_SIGNATURE_ARGS` | orchestration, resume, config | **no** |

Helpers `build_ensemble_result` depends on, all in the runner:
- `current_dataset_primary_metric` / `current_dataset_task_type` read the global
  `CURRENT_DATASET_SPEC`;
- `metric_lower_is_better` and `compute_primary_metric` depend on `normalize_benchmark_metric`;
- `regression_metrics`, `safe_correlation`, `is_ensemble_result_row`, `slugify`.

**Notebook:** block **7A** in `portable_colab_qsar_bundle/build_colab_qsar_tutorial.py` (search for
`# @title 7A.`). The builder generates `colab_qsar_tutorial.ipynb`; never hand-edit the notebook.
The cell holds inline copies of:
- member collection from `STATE` (all fitted conventional and tuned models; Uni-Mol only with
  `cv.data`);
- `_refit_oof` (K-fold `clone` refits cached in `STATE["ensemble_oof_cache"]`) and
  `_unimol_saved_oof`;
- `build_split_frame` and the OOF alignment;
- member filtering, the correlated-pair tie-break, stacking, the weighted average, CFA (via core
  `run_cfa_regression_fusion`) and the choice of downstream strategy.

**Users of these names that must keep working:** `plan_ensemble_oof_repair.py` imports
`runner.is_fusion_payload` and `runner.load_unimol_saved_oof`. `tests/unit/test_ensemble_oof.py`,
`tests/unit/test_work_order_guards.py` and `tests/unit/test_plan_ensemble_oof_repair.py` import
runner names. `qsarena/config.py` documents the CLI options.

## 3. Current behavioural differences (decide each one explicitly)

| Aspect | Runner | Notebook | Target |
|---|---|---|---|
| OOF folds | `make_qsar_cv_splitter` with the dataset's CV geometry (scaffold / target-quartile / random), `--ensemble-oof-folds`, seed `--random-seed` | plain `KFold(shuffle)` seeded by `stacking_random_seed` | **Shared:** core `make_oof_folds` with the CV geometry. The notebook passes its own CV strategy and seed, so notebook ensemble numbers change; that is acceptable, since no paper number comes from the notebook |
| Members | all families with OOF (conventional, ChemML, MapLight, Chemprop if scope `all`, Uni-Mol via `cv.data`; TabPFN only with API refits allowed) | conventional + tuned + Uni-Mol (`cv.data`) | **Keep different.** Member collection is entry-point specific; notebook sessions cannot cheaply refit deep models. Document it |
| Methods | OOF stacking (RidgeCV / logistic), weighted average, simple average; CFA is a separate stage and never a member | stacking, weighted average, **CFA as an ensemble method** | Core provides stacking, weighted and simple. CFA stays in both callers via core `run_cfa_regression_fusion`, fed OOF predictions |
| Weighted label | `Weighted average (inverse OOF error)` | `Weighted average (inverse OOF RMSE)` | **Keep the runner label exactly** (the A100 OOF run writes it; the resume matcher and the analysis notebook read it). The notebook may adopt it |
| Classification | supported; OOF weights = inverse RMS probability error | not supported (regression only) | Core supports both; the notebook stays regression-only (out of scope) |
| Metric names | `Train Primary`, `Train RMSE`, ... (runner `regression_metrics`: `train_r2`, ...) | notebook `summarize_regression`: `Train R2`, `OOF RMSE`, ... | Core returns plain arrays and a tidy weights frame; each caller formats its own result rows |
| Downstream strategy choice | not applicable (all methods reported) | lowest OOF RMSE | Core helper `choose_ensemble_by_oof(runs, metric)`; the notebook calls it |
| Legacy modes | `member_selection_split` `train` / `test` reproduce earlier runs | none | Core keeps all three modes; the notebook always uses `oof` |

## 4. Target API (in `qsar_workflow_core.py`)

Constraints on the core, which are hard requirements:
- **Single self-contained file.** Colab downloads *only* `qsar_workflow_core.py`
  (`ensure_qsarena_bundle_source` → `QSARENA_QSAR_CORE_URL`). New code may use only what the
  core already imports (numpy, pandas, RDKit, scikit-learn) plus `joblib`, which ships with
  scikit-learn. **It must never import the runner, `qsarena.*` or `benchmark_registry`.**
- **Additive and backward-compatible.** Colab fetches the core from `main` at run time, so
  notebooks already in users' hands will load the new core. Do not rename, remove or change the
  signature of any existing core function.
- **No global state.** Anything the runner reads from `CURRENT_DATASET_SPEC` is passed in
  explicitly.

Suggested surface. Names are illustrative, but keep the behaviour.

```python
ENSEMBLE_SELECTION_MODES = ("oof", "train", "test")

def is_fusion_member(model_name, workflow="") -> bool
def make_oof_folds(X, y, smiles, *, split_strategy, n_folds, random_seed) -> list[tuple[np.ndarray, np.ndarray]]
def oof_fold_signature(folds) -> str
def load_unimol_saved_oof(model_dir, *, n_train, reference_train_pred=None) -> tuple[np.ndarray | None, str]
def fill_oof_predictions(payloads, *, refitters, folds, fold_signature, n_train,
                         providers=None, cache_root=None, memory_cache=None,
                         on_model_done=None, log=print) -> list[str]
    # cache_root: on-disk per-fold .npy cache (runner). memory_cache: dict cache (notebook STATE).
def build_ensemble(payloads, *, method, task_type, primary_metric, lower_is_better, metric_fn,
                   selection_split="oof", stacking_cv_folds=5, random_seed=0,
                   drop_highly_correlated=True, max_correlation=0.995,
                   exclude_nonpositive_r2=True) -> EnsembleBuild
    # metric_fn(metric_name, y_true, y_pred) -> float. The runner passes compute_primary_metric;
    # the core ships default_metric_fn covering rmse/mae/r2/roc_auc/auprc.
@dataclass
class EnsembleBuild:
    label: str                 # e.g. "OOF Stacking (RidgeCV, 5-fold)", "Weighted average (inverse OOF error)"
    members: list[str]
    weights: pd.DataFrame      # Model, Weight, [Abs normalized contribution], Workflow
    train_pred: np.ndarray     # OOF-based in oof mode
    test_pred: np.ndarray
    aligned_fit: pd.DataFrame  # the frame used for fitting (OOF in oof mode), SMILES/Observed/members
    aligned_test: pd.DataFrame
    member_metrics: dict[str, dict[str, float]]
    notes: list[str]
    meta_model: Any
def choose_ensemble_by_oof(builds, *, metric_fn, primary_metric, lower_is_better) -> EnsembleBuild
```

**Payload schema** (unchanged; document it in the core docstring). Required keys: `workflow`,
`train_smiles`, `test_smiles`, `train_observed`, `test_observed`, `train`, `test`. Optional:
`train_row_id`, `test_row_id`, `oof`, `oof_signature`. In `oof` mode, `oof` is aligned
row-for-row with `train_smiles` / `train_row_id`.

**Runner after the refactor:**
- `build_ensemble_result(...)` keeps its exact signature and return tuple. It resolves
  `primary_metric`, `task_type` and `lower_is_better` from `CURRENT_DATASET_SPEC`, calls
  `core.build_ensemble` with `metric_fn=compute_primary_metric`, and formats `ensemble_results`
  with the runner's `regression_metrics`, exactly as today.
- `is_fusion_payload`, `ensemble_oof_folds`, `ensemble_oof_fold_signature`,
  `ensure_ensemble_oof_predictions` and `load_unimol_saved_oof` stay importable from the runner as
  thin aliases or wrappers.
- `run_ensemble_oof_stage`, persistence, CLI, config and resume are untouched.

**Notebook 7A after the refactor:**
- It keeps member collection and its refitters (`clone`-based) and its Uni-Mol provider. It
  replaces its inline helpers with `make_oof_folds` (the notebook's CV strategy and seed),
  `fill_oof_predictions(memory_cache=STATE.setdefault("ensemble_oof_cache", {}))`,
  `build_ensemble` for stacking and weighted, and `choose_ensemble_by_oof`.
- CFA stays in the cell, fed the OOF matrix from `EnsembleBuild.aligned_fit`.
- It keeps producing the same `STATE[...]` keys that blocks 7B+ read: `ensemble_results`,
  `ensemble_weight_table`, `ensemble_train_aligned`, `ensemble_oof_aligned`,
  `ensemble_test_aligned`, `ensemble_prediction_columns`, `ensemble_model_label`, `ensemble_method`,
  `ensemble_meta_model`, `ensemble_meta_intercept`, `ensemble_member_filter_notes`,
  `ensemble_weight_tables`, `ensemble_method_summaries`, `ensemble_cfa_*`. Grep the builder for
  every `STATE["ensemble_` read before changing any of them.
- The core is imported through the existing `from portable_colab_qsar_bundle.qsar_workflow_core
  import (...)` block in the setup cell. Add the new names there.

## 5. Phases and acceptance criteria

Do the phases in order and commit each one separately. Branch from `main`; do not push to `main`
until phase 6 passes.

**Phase 0: pre-flight and a decision for the user.**
- `QSARENA_QSAR_CORE_URL` in the builder points to `github.com/ScottCoffin/QSARena`, which **does
  not exist yet**: the repo is still `ScottCoffin/AutoQSAR`, and the rename is an open item in
  `TODO.md`. A fresh Colab session with no local clone therefore cannot download the core at all,
  before or after this refactor. **Ask the user** whether to point the URL at `AutoQSAR` now or
  wait for the rename. Do not rename repositories yourself.
- Ask whether the A100 OOF ensemble run is in progress. Landing this is safe only because phase 3
  requires bit-identical runner results; if phase 3 cannot achieve that, **wait for the run to
  finish before merging.**

**Phase 1: golden fixtures, before touching any logic.**
1. **Runner golden.** Add `tests/unit/test_ensemble_golden.py`. It runs the *current*
   `build_ensemble_result` on fixed synthetic payloads covering:
   - modes {oof, train, test} × methods {OOF Stacking, Weighted average, Simple average} ×
     tasks {regression, binary classification};
   - correlated members (which exercises the tie-break);
   - a member without OOF, and a CFA payload (the exclusion notes).

   Store `train_pred`, `test_pred`, the weights frame, the members and the notes under
   `tests/fixtures/ensemble_golden/` as `.npz` / `.json` (not pickle). Also cover
   `ensure_ensemble_oof_predictions` with a deterministic refitter and a provider.
2. **Notebook golden.** Add `tests/unit/test_colab_ensemble_cell.py`. It extracts the 7A cell
   from the generated `colab_qsar_tutorial.ipynb` and executes it against a synthetic `STATE`,
   stubbing `summarize_regression` (copy the builder's definition), `ensure_global_split_signature`,
   `display*`, and importing the CFA helpers from the core. It asserts:
   - the members, and that Uni-Mol joins via `cv.data`;
   - that a memorising extra-trees model is not dominant;
   - that the strategy chosen equals the lowest-OOF-RMSE one;
   - that every `STATE["ensemble_*"]` key listed in §4 exists.

   A working prototype of this harness was used on 2026-09-26 (it built a synthetic `STATE` with
   Ridge, random forest, extra trees and a fake Uni-Mol `cv.data` folder, then `exec`'d the cell).
   Rebuild it in `tests/`.
3. `python -m pytest -q tests/unit` passes before any refactor.

**Phase 2: add the core API.** Implement §4 in `qsar_workflow_core.py`, with new unit tests in
`tests/unit/test_core_ensemble.py` that run without the runner (import only the core). All
existing tests still pass.

**Phase 3: the runner delegates.** Replace the runner's implementations with calls into the core,
keeping the wrappers and aliases from §4.
- **Acceptance:** the phase 1 runner golden matches **bit-for-bit** (`np.testing.assert_array_equal`,
  not allclose) in every mode, method and task.
- **Acceptance:** `tests/unit`, `tests/test_model_filtering.py`,
  `tests/test_prepare_chemprop_repair_run.py` and `tests/integration` pass. The integration suite
  takes about 5 minutes of CLI runs.
- **Acceptance:** an end-to-end CLI run on the example data (command below) gives identical
  ensemble rows, weights and `split="oof"` prediction rows to the same command on the pre-refactor
  commit. Diff `metrics.csv` ensemble rows and the `ensemble_weights_*.csv` files.
- **Acceptance:** `plan_ensemble_oof_repair.py` still runs; its tests pass.

**Phase 4: the notebook delegates.** Edit block 7A in the builder, then regenerate:
`python portable_colab_qsar_bundle/build_colab_qsar_tutorial.py`.
- **Acceptance:** the phase 1 notebook harness passes. Members and the leak-free properties are
  unchanged. Weights and numbers may change *only* because of the fold-geometry alignment from §3;
  state that in the commit message.
- **Acceptance:** `git grep -n "Test R2\|Test RMSE\|Test MAE" -- portable_colab_qsar_bundle/build_colab_qsar_tutorial.py`
  shows no test metric feeding any choice, filter, weight or strategy inside 7A. Reporting and
  display sorts are fine.
- **Acceptance:** the setup-cell import list includes the new core names, and a local Jupyter run
  of the setup cell imports them. Colab cannot be tested from here, which is why the phase 0 URL
  decision matters.

**Phase 5: docs.**
- **Manuscript §2.1, in both `manuscript.md` and `submission/body.tex`:** state that ensembles use
  the same shared OOF ensemble code in both entry points. Note that the notebook draws members only
  from conventional, tuned and Uni-Mol models, because a notebook session has no OOF predictions for
  the other deep backends. Keep the two formats in sync (AGENTS.md "Manuscript workflow" step 3).
  Run `python portable_colab_qsar_bundle/verify_manuscript_numbers.py` afterwards; it must still
  pass.
- **`AGENTS.md`:**
  - add a "Where to edit" row: ensembles → `qsar_workflow_core.py` (shared), runner orchestration
    in `run_qsarena_benchmarks.py`, notebook member collection in builder block 7A;
  - update the ensemble trap bullets that say the notebook has "its own copy";
  - tick `TODO.md` (the §2.1 item).
- **README "Fusion And Ensemble Models":** one sentence noting the shared implementation.

**Phase 6: final verification, then merge.**
```bash
python -m pytest -q tests/unit tests/test_model_filtering.py tests/test_prepare_chemprop_repair_run.py
python -m pytest -q tests/integration
python -m qsarena.config --write-docs && git diff --exit-code docs/options_reference.md configs/run.example.yaml
# end-to-end golden comparison (run on the pre-refactor commit too, and diff):
QSARENA_HOME=$TMP/home python portable_colab_qsar_bundle/run_qsarena_benchmarks.py \
  --batch qsarena/examples/data/batch_dir --output-dir $TMP/e2e --benchmark-profile quick \
  --selector-method rf_importance --n-jobs 2 --use-gpu false \
  --only-model-names ElasticNetCV --only-model-names "Random forest" \
  --only-model-names "Extra trees" --only-model-names SVR --only-model-names Ensemble
python portable_colab_qsar_bundle/verify_manuscript_numbers.py
```
`tests/docs` fails on a workstation without an installed `qsarena-examples` console script. That
failure is environmental; install the wheel in a short-path venv to run it (AGENTS.md
"Packaging").

## 6. Out of scope

- Changing ensemble defaults, methods or labels in the runner. The A100 OOF run's outputs must
  stay reproducible.
- Adding Chemprop, ChemML, TabPFN or MapLight+GNN refits to the notebook.
- Notebook classification ensembles.
- Re-running any benchmark, and regenerating manuscript numbers or figures.
- Renaming the GitHub repository.

## 7. Pitfalls

- `run_qsarena_benchmarks.txt` and `qsar_workflow_core.txt` are stale mirrors. Do not edit or read
  them as source.
- The runner's per-dataset metric and task come from the global `CURRENT_DATASET_SPEC`. Tests set
  and restore it (see `tests/unit/test_ensemble_oof.py`). The core must receive them as arguments.
- `ensemble_model_name_matches_method` resumes runs by matching ensemble labels. Any label change
  breaks resuming the A100 run.
- `predictions.csv` holds `train`, `test` **and** `oof` rows. `rebuild_prediction_payloads`
  reads `oof` rows by `row_index` and `oof_signature`. Keep that contract.
- In the notebook, Uni-Mol payloads may cover a subset of the training molecules, and alignment is
  by SMILES. Keep the OOF alignment by SMILES, not position, for notebook members.
- Classification OOF weights in the runner use inverse RMS probability error, falling back to a
  floored rank-metric error for non-probability outputs. The legacy `train`/`test` modes use
  `1/(1-score)` and must keep doing so for reproduction.
- On Windows, keep `PYTHONIOENCODING=utf-8` for CLI runs (UTF-8 progress output). All
  `subprocess.run` calls use `_SUBPROCESS_TEXT_KWARGS`.
- `rdkit` must import for the core to import. The test environment already has it.
