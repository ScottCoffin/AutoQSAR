# RTX Chemprop OOF Handoff

This handoff lets an RTX workstation fill the remaining Chemprop out-of-fold
(OOF) rows without transferring the full local prediction corpus.

For the full RTX queue, including the optional meta-analysis Phase 7 GPU
experiment, start from `docs/HANDOFF_RTX_GPU_WORK.md`.

## Goal

Produce a small GitHub patch containing only:

```text
chemprop_oof_patch/<dataset>/chemprop_oof_predictions.csv
```

Those rows are later merged on the workstation that already has the full
`benchmark_results/qsarena_benchmark_oof_ensemble/*/predictions.csv` files.

## Current Open Work

As of this handoff, Chemprop OOF is still open in the main OOF run:

- Chemprop members needing GPU OOF refits: 202
- Classical/CPU OOF rows are already saved locally and do not need refitting
- Uni-Mol OOF rows are read from committed `cv.data`
- Chemprop `split=oof` rows are absent from the full local run

## Files Provided By This Branch

- `chemprop_oof_seed/`
  - Compact run-shaped seed directory.
  - Contains only Chemprop `train`/`test` prediction rows plus matching
    Chemprop metric rows.
  - This avoids pushing the full local `predictions.csv` corpus.
- `tools/run_chemprop_oof_rtx.ps1`
  - PowerShell wrapper for the RTX machine.
  - Plans, runs, exports the patch, and optionally commits/pushes it.
- `tools/export_chemprop_oof_patch.py`
  - Extracts Chemprop `split=oof` rows after the RTX run.
- `tools/apply_chemprop_oof_patch.py`
  - Used later on the workstation to merge the returned patch into the full run.

## RTX Instructions

From a fresh checkout on the RTX machine:

```powershell
git pull origin main
```

Activate the Python environment that has Chemprop v2, RDKit, PyTorch/CUDA, and
QSARena dependencies. Then run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_chemprop_oof_rtx.ps1 `
  -Python "C:\path\to\python.exe" `
  -SeedRun chemprop_oof_seed `
  -PatchDir chemprop_oof_patch `
  -CommitAndPush
```

If `python` on PATH already resolves to the right environment, this is enough:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_chemprop_oof_rtx.ps1 -CommitAndPush
```

The wrapper runs the equivalent of:

```powershell
python portable_colab_qsar_bundle/run_qsarena_benchmarks.py `
  --output-dir chemprop_oof_seed --benchmark-profile full `
  --dataset-name <each seeded dataset> `
  --only-model-names Ensemble --run-ensemble --rebuild-ensemble `
  --ensemble-member-selection-split oof --ensemble-oof-folds 5 `
  --ensemble-oof-scope all `
  --run-chemprop-mpnn --run-chemprop-dmpnn --run-chemprop-rdkit2d `
  --run-chemprop-cmpnn --run-chemprop-attentivefp `
  --run-chemprop-selected-features `
  --chemprop-epochs 40 --chemprop-ensemble-size 3 --chemprop-random-seed 42 `
  --reuse-persistent-feature-store --reuse-shared-feature-matrix-cache `
  --resume --no-run-tdc22-multiseed-best
```

Expected behavior:

- The seed run will rebuild features as needed.
- Only Chemprop fold models should be trained.
- Fold caches are written under `chemprop_oof_seed/<dataset>/ensemble_oof/`.
- The final Git payload should be `chemprop_oof_patch/`, not the fold caches.

If the run is interrupted, rerun the same PowerShell command. The runner resumes
from fold caches.

## RTX Validation Before Push

The wrapper automatically exports `chemprop_oof_patch/` and runs the planner
again. Before pushing manually, check:

```powershell
python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py `
  chemprop_oof_seed --scope all --folds 5

git status --short chemprop_oof_patch
```

Then commit and push if `-CommitAndPush` was not used:

```powershell
git add chemprop_oof_patch
git commit -m "Add Chemprop OOF prediction patch"
git push
```

Do not commit `chemprop_oof_seed/*/ensemble_oof/` or `chemprop_v2/` model
directories.

## Workstation Instructions After RTX Push

On the workstation with the full local OOF run:

```powershell
git pull origin main

python tools/apply_chemprop_oof_patch.py `
  --patch-dir chemprop_oof_patch `
  --target-run benchmark_results/qsarena_benchmark_oof_ensemble

python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py `
  benchmark_results/qsarena_benchmark_oof_ensemble --scope all --folds 5 `
  --source-run benchmark_results/autoqsar_benchmark_20260623_153839
```

After the planner confirms the Chemprop members are `saved_oof`, rerun the
normal ensemble-only refresh on the workstation and regenerate interpretation,
manuscript assets, and the PDF.
