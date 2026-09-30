# RTX GPU Work Handoff

This handoff separates the GPU work that should happen on the RTX machine from
the CPU/manuscript work that should stay on the workstation.

## Scope From The Meta-Analysis Spec

`QSARena_Meta_Analysis_IDE_Agent_Implementation_Spec.docx` is primarily a CPU
analysis spec:

- Phases 1-6 build dataset descriptors, family-level summaries, regressions,
  bootstrap uncertainty, figures, tables and manuscript text from existing
  single-seed benchmark artifacts.
- They must not retrain models.
- They belong on the workstation after the OOF ensemble run is finalized.

The only GPU item in the spec is optional Phase 7, the learning-curve
experiment. It is off by default and should run only after explicit approval.

## Status as of 2026-09-28 (RTX 4060 Laptop, paused by the user)

**Before resuming, bring the gitignored predictions over by SSD and verify them:**
[HANDOFF_SSD_TRANSFER.md](HANDOFF_SSD_TRANSFER.md) (only ~1.1 GB of the ~120 GB checkout is needed).

Chemprop OOF was started on the RTX and **paused deliberately** after 4 of 42 datasets
(`tdc_carcinogens_lagunin`, `tdc_skin_reaction`, `tdc_dili`, `chemml_cep_homo`); 113 fold vectors
are committed as `chemprop_oof_seed/*/ensemble_oof/*/*/fold_*.npy` and are reused on resume. No
`chemprop_oof_patch/` has been exported yet. Two members failed with an intermittent native
crash and have no complete OOF yet: `tdc_skin_reaction` CMPNN (fold 4) and `chemml_cep_homo`
D-MPNN + Selected descriptors; the wrapper's retry passes refit only those folds.

Resume with the same command (Python env on this machine: `qsarena-py311` conda env):

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_chemprop_oof_rtx.ps1 `
  -Python "C:\Users\scott\.conda\envs\qsarena-py311\python.exe" -CommitAndPush
```

Measured pace: ~55 s per fold on 280-row datasets (40 epochs, ensemble=3). Larger datasets are
slower; take a real estimate from the first mid-sized dataset before promising a finish time.

Windows fixes that made it run (all committed; see AGENTS.md traps):

- Wrapper launches the runner through `cmd /c` redirection. Under Windows PowerShell 5.1 with
  `$ErrorActionPreference = "Stop"`, `*>` turned the first stderr line into a fatal error.
- `--chemprop-num-workers 0`: each DataLoader worker is a new process that loads torch/CUDA
  (~720 MB); four of them exhausted the paging file (`WinError 1455`) and every fold failed.
- `portable_colab_qsar_bundle/chemprop_cli_launcher.py`: Chemprop enables
  `torch.use_deterministic_algorithms(True)` whenever `--pytorch-seed` is set, and torch 2.5.1 has
  no deterministic CUDA `cumsum`, so every classification fold aborted in the AUROC metric. The
  launcher makes that warn-only; seeds and deterministic kernel selection are unchanged.
- Retry passes (up to 2) in the wrapper for intermittent `0xC0000409` Chemprop crashes.

## Required RTX Work: Chemprop OOF

Chemprop OOF is the required GPU task before the final OOF ensembles can include
Chemprop. Use the dedicated handoff:

```powershell
git pull origin main
powershell -ExecutionPolicy Bypass -File tools/run_chemprop_oof_rtx.ps1 -CommitAndPush
```

If the RTX Python environment is not on `PATH`, pass it explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_chemprop_oof_rtx.ps1 `
  -Python "C:\path\to\python.exe" `
  -CommitAndPush
```

Expected pushed payload:

```text
chemprop_oof_patch/
```

Do not commit `chemprop_oof_seed/*/ensemble_oof/`, Chemprop model directories,
or any full `predictions.csv` corpus.

## Optional RTX Work: Meta-Analysis Phase 7

Run this only if the user explicitly approves the optional learning-curve
experiment from the meta-analysis spec.

Plan file:

```text
docs/meta_analysis/RTX_PHASE7_LEARNING_CURVE_PLAN.csv
```

It runs these 18 points:

- datasets: `tdc_ld50_zhu`, `tdc_tox21`, `tdc_solubility_aqsoldb`
- train row caps: 250, 500, 1000, 2000, 4000, full
- models: `Random forest`, `XGBoost`, `ChemML MLP (PyTorch)`, `Uni-Mol V1`

RTX command:

```powershell
git pull origin main
powershell -ExecutionPolicy Bypass -File tools/run_meta_phase7_rtx.ps1 -CommitAndPush
```

With an explicit Python:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_meta_phase7_rtx.ps1 `
  -Python "C:\path\to\python.exe" `
  -CommitAndPush
```

The wrapper writes run artifacts under:

```text
benchmark_results/qsarena_meta_phase7_gpu/
```

and exports only the compact return payload:

```text
meta_phase7_gpu_patch/
```

Do not commit `benchmark_results/qsarena_meta_phase7_gpu/`. It may contain
heavy model and prediction artifacts; the workstation only needs
`meta_phase7_gpu_patch/phase7_learning_curve_metrics.csv`.

## Workstation After RTX Push

Pull the returned RTX patch or patches:

```powershell
git pull origin main
```

For Chemprop OOF:

```powershell
python tools/apply_chemprop_oof_patch.py `
  --patch-dir chemprop_oof_patch `
  --target-run benchmark_results/qsarena_benchmark_oof_ensemble

python portable_colab_qsar_bundle/plan_ensemble_oof_repair.py `
  benchmark_results/qsarena_benchmark_oof_ensemble --scope all --folds 5 `
  --source-run benchmark_results/autoqsar_benchmark_20260623_153839
```

Proceed with the final ensemble refresh only after the planner reports Chemprop
members as `saved_oof`.

For optional Phase 7, the future meta-analysis implementation should read:

```text
meta_phase7_gpu_patch/phase7_learning_curve_metrics.csv
```

No importer is needed for Phase 7 because the patch is already a compact
analysis table.

## GitHub Payload Rules

Keep the cross-machine exchange limited to:

```text
chemprop_oof_patch/
meta_phase7_gpu_patch/
```

These are small enough for normal GitHub push/pull handoff. Avoid external drive
transfers unless a later task explicitly requires full model artifacts.
