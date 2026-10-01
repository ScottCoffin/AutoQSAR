# Handoff: move the gitignored predictions to the RTX by SSD

All remaining work (Chemprop OOF, the final OOF ensemble rebuild, the meta-analysis spec, the
manuscript refresh) moves to the RTX 4060 laptop. GitHub carries everything except the per-row
predictions, which are gitignored. This note says exactly which files to carry on the SSD.

**The source checkout is ~120 GB, but only ~1.1 GB of it is needed.**

## What to copy

| Files | Size | Needed? |
|---|---|---|
| `benchmark_results/qsarena_benchmark_oof_ensemble/**/predictions.csv` (45 files: 44 datasets + 1 run-level) | ~1.1 GB | **Required.** Every base model's train/test predictions plus the CPU out-of-fold rows from commit `631f668`. The Chemprop OOF merge, the final ensemble rebuild and the planner check all read them. Without them the ensembles cannot be rebuilt without retraining. |
| `benchmark_results/qsarena_benchmark_oof_ensemble/**/ensemble_oof/**/fold_*.npy` | ~14 MB | Optional, free backup. The CPU fold vectors; their contents are already saved inside `predictions.csv`. |
| `benchmark_results/qsarena_benchmark_chemprop_fixed/**/predictions.csv` | ~0.6 GB | Optional. Only needed to reseed the OOF run from scratch with `prepare_chemprop_repair_run.py` if something goes wrong. |
| `benchmark_results/qsarena_benchmark_oof_ensemble/*/stage23_resume_cache.pkl` | 6.1 GB | Skip. It only saves rebuilding features on CPU (seconds to minutes per dataset). |

**Do not copy the rest (~110 GB+).** It is Uni-Mol `.pth` weights (their OOF predictions are the
committed `cv.data` files), conventional-ML model files, Chemprop model directories (OOF refits
train new fold models, never reuse the full-fit ones), and other runs' model folders. The
meta-analysis spec phases 1-6 read only committed artifacts; optional Phase 7 retrains from the
datasets, which the RTX already has.

## On the source laptop

1. Pull first so the checkout matches `main` (it must include `3310456` or later):
   ```powershell
   git pull origin main
   ```
2. Copy only the needed files, keeping the folder layout. From the repo root in PowerShell,
   with the SSD at `E:` (adjust the drive letter):
   ```powershell
   robocopy benchmark_results\qsarena_benchmark_oof_ensemble `
     D:\qsarena_transfer\benchmark_results\qsarena_benchmark_oof_ensemble `
     predictions.csv fold_*.npy /S

   # optional backup for reseeding
   robocopy benchmark_results\qsarena_benchmark_chemprop_fixed `
     E:\qsarena_transfer\benchmark_results\qsarena_benchmark_chemprop_fixed `
     predictions.csv /S
   ```
   `robocopy` exit codes 0-7 mean success; 8 or higher is a failure.
3. Optional check before unplugging. Every file should match the committed manifest:
   ```powershell
   python tools/verify_transferred_predictions.py benchmark_results/qsarena_benchmark_oof_ensemble
   ```
   If this fails on the source laptop itself, its predictions are not the ones `631f668`
   committed metrics for. Stop and sort that out before copying.

## On the RTX

1. Pull, then copy the SSD tree into the repo. It only adds gitignored files, so `git status`
   stays clean:
   ```powershell
   git pull origin main
   robocopy E:\qsarena_transfer\benchmark_results benchmark_results /S
   ```
2. Verify the transfer. **Do not start any ensemble work until this prints `45/45` and exits 0:**
   ```powershell
   python tools/verify_transferred_predictions.py benchmark_results/qsarena_benchmark_oof_ensemble
   ```
   The script compares each `predictions.csv` with the size and SHA-256 recorded in the run's
   committed `artifact_manifest.csv`, so it catches a stale, partial or wrong-run copy.
3. Continue with the RTX queue in `docs/HANDOFF_RTX_GPU_WORK.md` (status section at the top).
   With the full predictions on the RTX, Chemprop OOF could run directly in
   `qsarena_benchmark_oof_ensemble` instead of the `chemprop_oof_seed` + patch route, which would
   skip the export/apply step and could carry over the 113 folds already done. Decide that at
   resume time after checking it against the real data.

The Python environment is not part of the transfer: on the RTX, use the `qsarena-py311` conda
env (`C:\Users\scott\.conda\envs\qsarena-py311\python.exe`); system `python` has no torch.
