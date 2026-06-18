# HPC Deployment — AutoQSAR TDC-22 Multi-Seed Evaluation

This directory contains Slurm job-array submission scripts and the task manifest
for the TDC-22 multi-seed re-evaluation on NSF ACCESS HPC clusters.

---

## Overview

The multi-seed evaluation runs each of the 22 TDC ADMET benchmark datasets across
5 seeds (1–5), producing 110 independent `run_one.py` invocations.
The tasks are embarrassingly parallel and dispatched as Slurm array jobs.

Two scripts are provided:

| Script | Partition | Target models | Tasks |
|---|---|---|---|
| `submit_multiseed.sbatch` | GPU-shared | MapLight+GNN, Chemprop v2, TabPFN | ~35 tasks |
| `submit_multiseed_cpu.sbatch` | CPU-only | Conventional ML, CFA fusion, ensembles | ~75 tasks |

---

## Operator Placeholders

Before submitting, fill in the `<PLACEHOLDER>` values in both sbatch scripts:

| Placeholder | Description | Examples |
|---|---|---|
| `<ACCOUNT>` | Slurm allocation account | `aab123` (Delta), `pXXX-gpu` (Bridges-2) |
| `<GPU_SHARED_PARTITION>` | GPU-shared partition name | `gpuA100x4-shared` (Delta), `GPU-shared` (Bridges-2), `gpu-shared` (Expanse) |
| `<CPU_PARTITION>` | CPU-only partition name | `cpu` (Delta), `RM` (Bridges-2), `compute` (Expanse) |
| `<CUDA_OR_APPTAINER_MODULE>` | Module to load | `module load apptainer` or `module load cuda/12.1 apptainer` |
| `<MAX_CONCURRENT>` | Max simultaneous array tasks | GPU: 10–20; CPU: 20–40 (check fairshare policy) |

---

## Model–Device Routing

The winning model per TDC-22 dataset (from the canonical benchmark run) determines
whether the task goes to the GPU or CPU array:

**GPU array (submit_multiseed.sbatch):**
- `tdc_ppbr_az` — MapLight+GNN
- `tdc_vdss_lombardo` — MapLight+GNN
- `tdc_clearance_microsome_az` — MapLight+GNN
- `tdc_clearance_hepatocyte_az` — MapLight+GNN
- `tdc_half_life_obach` — TabPFN
- Any dataset where Chemprop v2 was the winner

**CPU array (submit_multiseed_cpu.sbatch):**
All remaining 17 datasets (conventional ML, CFA fusion, OOF stacking, inverse-RMSE
weighted average winners). The CPU script explicitly disables GPU model families
(`--no-run-maplight-gnn`, `--no-run-chemprop-*`, `--no-run-tabpfn`) to avoid
unnecessary dependency on GPU availability.

---

## Generating Sub-Manifests

The full manifest `task_manifest.tsv` (110 rows) has a `device` column.
Generate the GPU and CPU sub-manifests before submitting:

```bash
# GPU sub-manifest (skip header, filter device==gpu, drop notes column)
head -1 hpc/task_manifest.tsv | cut -f1-2 > hpc/task_manifest_gpu.tsv
awk -F'\t' 'NR>1 && $3=="gpu" {print $1"\t"$2}' hpc/task_manifest.tsv >> hpc/task_manifest_gpu.tsv

# CPU sub-manifest
head -1 hpc/task_manifest.tsv | cut -f1-2 > hpc/task_manifest_cpu.tsv
awk -F'\t' 'NR>1 && $3=="cpu" {print $1"\t"$2}' hpc/task_manifest.tsv >> hpc/task_manifest_cpu.tsv

# Count tasks (update --array upper bound in sbatch scripts to match)
wc -l hpc/task_manifest_gpu.tsv hpc/task_manifest_cpu.tsv
```

Update the `--array=0-N` directive in each sbatch script to `N = (line_count - 2)`.

---

## Staging Inputs

The container expects:
```
/in/
  data/               # benchmark CSV/tab files and dataset catalog
  model_cache/        # MapLight pretrained GIN weights, feature store
  .cache/             # pytdc-1.1.15.tar.gz and TDC split caches
```

Stage from your project directory to `$SCRATCH` before submitting:

```bash
rsync -av /path/to/AutoQSAR/data/        $SCRATCH/autoqsar_in/data/
rsync -av /path/to/AutoQSAR/model_cache/ $SCRATCH/autoqsar_in/model_cache/
rsync -av /path/to/AutoQSAR/.cache/      $SCRATCH/autoqsar_in/.cache/
```

Set `AUTOQSAR_INPUT` and `AUTOQSAR_OUTPUT` before submitting (or edit the defaults
in the sbatch scripts):

```bash
export AUTOQSAR_INPUT=$SCRATCH/autoqsar_in
export AUTOQSAR_OUTPUT=$SCRATCH/autoqsar_out
export SIF=/path/to/autoqsar.sif
```

---

## Submitting

```bash
# Generate sub-manifests first (see above)

# Submit GPU array
sbatch hpc/submit_multiseed.sbatch

# Submit CPU array
sbatch hpc/submit_multiseed_cpu.sbatch

# Monitor
squeue -u $USER
```

---

## Interpreting Outputs

Each task writes to `$AUTOQSAR_OUTPUT/<dataset>/seed_<seed>/`:

```
metrics.csv                  # per-model test metrics
predictions.csv              # test-set predictions
selected_features.csv        # retained feature names
run_one_provenance.json      # image hash, git commit, device, seed
run_status.json              # pass/fail and elapsed time
ensemble_results.csv         # ensemble member details
```

To aggregate across seeds after all tasks complete:

```bash
python portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py \
    --multi-seed-summary-only \
    --output-dir $SCRATCH/autoqsar_out
```

---

## Time Estimates

Based on the canonical benchmark run (cost_optimized profile):

| Dataset family | Approx. wall time per seed |
|---|---|
| MapLight+GNN wins | 2–4 h (13 h total in single-process run) |
| Chemprop wins | 0.5–1 h |
| TabPFN wins | 0.1–0.5 h |
| Conventional ML / ensemble wins | 0.25–1 h |

The `--time=04:00:00` limit in the GPU script covers all GPU-family datasets.
CPU jobs with large datasets (lipophilicity, solubility) may approach 8 h.
