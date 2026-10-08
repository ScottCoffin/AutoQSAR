# Acute LD50 QSAR_READY_SMILES RTX handoff

This folder packages the acute oral toxicity dataset and the completed CPU quick-run outputs so the same work can be rerun on an RTX GPU with QSARena's GPU-capable families.

## Dataset

- Input file: `input/full_dataset_qsar_ready_ld50lm.csv`
- Structure column: `QSAR_READY_SMILES`
- Regression target: `LD50_LM`
- Target transform: `raw`

The notebook export produced 7,011 rows in the full merged table. The packaged benchmark input keeps the two columns used by the runner and drops 56 rows with missing `QSAR_READY_SMILES`, leaving 6,955 rows. QSARena/RDKit drops one additional unparseable SMILES during canonicalization, so the completed quick run used 6,954 molecules.

`LD50_LM` is already on a log scale, so do not use the default log transform. All commands below pass `--target-transform raw`.

## Baseline From The Dataset Paper

Reference baseline: Helman, Shah, and Patlewicz, "Transitioning the Generalised Read-Across approach (GenRA) to quantitative predictions: A case study using acute oral toxicity data", DOI `10.1016/j.comtox.2019.100097`.

Reported global GenRA performance:

- R2: `0.61`
- RMSE: `0.58`
- Monte Carlo CV R2 range: `0.47` to `0.62`
- Local-domain R2 up to `0.91`

The QSARena run uses the package's benchmark split/protocol, not the paper's exact GenRA setup, so compare the values as a benchmark-style reference rather than as a paired reproduction.

## Completed CPU Quick Run

Curated outputs from `benchmark_results/acute_ld50lm_qsar_ready_quick_20261007` are under `outputs/quick_cpu/`.

Best quick-run row:

| model | test_r2 | test_rmse | test_mae | delta_r2_vs_genra | delta_rmse_vs_genra |
|---|---:|---:|---:|---:|---:|
| Ensemble (OOF Stacking, RidgeCV 5-fold) | 0.580427 | 0.579295 | 0.416678 | -0.029573 | -0.000705 |

Key files:

- `outputs/quick_cpu/genra_comparison.md`
- `outputs/quick_cpu/genra_comparison.csv`
- `outputs/quick_cpu/summary_metrics.csv`
- `outputs/quick_cpu/report.html`
- `outputs/quick_cpu/dataset/metrics.csv`

Heavy/regenerable quick-run files were not packaged: prediction tables, model/cache directories, and the stage 2/3 resume pickle.

## RTX Run

From the repository root on the RTX workstation:

```powershell
powershell -ExecutionPolicy Bypass -File docs\rtx_handoffs\acute_ld50lm_qsar_ready_20261007\scripts\run_acute_ld50_rtx.ps1
```

Default output:

```text
benchmark_results\acute_ld50lm_qsar_ready_rtx_full_20261007
```

The script runs the `full` profile with GPU use requested, Chemprop variants enabled, Uni-Mol enabled, TabPFN enabled, and the same input columns/target transform as the CPU quick run. It also disables the TDC-22 multi-seed side task.

After the RTX benchmark finishes, write the GenRA comparison:

```powershell
& "C:\Users\Scott.Coffin\AppData\Local\miniconda3\envs\autoqsar-py311\python.exe" docs\rtx_handoffs\acute_ld50lm_qsar_ready_20261007\scripts\compare_genra.py `
  --run-dir benchmark_results\acute_ld50lm_qsar_ready_rtx_full_20261007 `
  --dataset-name full_dataset_qsar_ready_ld50lm
```

The comparison script writes `genra_comparison.csv` and `genra_comparison.md` into the run directory.

## GitHub Payload Policy

This handoff is intended to be pushed to GitHub. The handoff `.gitignore` excludes local RTX outputs and common large model/cache/prediction artifacts. The repository root `.gitignore` also excludes benchmark prediction CSVs, model directories, `*.pkl`, and `.model_cache/`.

Before committing any RTX output, inspect file sizes and stage only compact summaries such as metrics, reports, run config, and GenRA comparison files.
