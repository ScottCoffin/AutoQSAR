# Running AutoQSAR on HPC via Apptainer

This document explains how to build the AutoQSAR container, transfer it to an
NSF ACCESS HPC cluster, and dispatch the TDC-22 multi-seed evaluation as a
Slurm job array.

---

## Prerequisites

### Local machine
- Docker (for building the image)
- `apptainer` or `singularity` (for converting to `.sif` and local testing)
- `make`

### HPC cluster
- Apptainer or Singularity module (ask your cluster admin)
- NVIDIA GPU nodes with driver supporting CUDA ≥ 12.1 (A100/V100/H100 on ACCESS clusters)
- Slurm scheduler

---

## 1 — Build the Docker image

```bash
# The working tree MUST be clean (enforced by Makefile)
git status    # confirm no uncommitted changes

make image    # builds autoqsar:<commit> and autoqsar:latest, populates BUILD_PROVENANCE.md
```

The Makefile tags the current commit (`git tag autoqsar-vYYYYMMDD`), embeds the
hash into the image labels, and records `pip freeze` output in `BUILD_PROVENANCE.md`.

---

## 2 — Convert to Apptainer .sif

```bash
make sif      # exports Docker image → autoqsar.sif, records SHA-256 in BUILD_PROVENANCE.md
```

Or on HPC where Docker is unavailable, build directly from the Apptainer def file:

```bash
# On the HPC login node (requires internet access or pre-pulled base image)
apptainer build autoqsar.sif containers/autoqsar.def
```

Verify the image:

```bash
apptainer run autoqsar.sif --help
```

---

## 3 — Local smoke tests

```bash
# CPU test (no GPU needed)
bash tests/smoke_cpu.sh

# GPU test (skips automatically if no GPU is detected)
bash tests/smoke_gpu.sh
```

Both tests run `tdc_herg` at `--row-limit 200` and assert that `metrics.csv`
and `run_one_provenance.json` exist with finite metric values.

---

## 4 — Transfer to HPC cluster

```bash
# Replace <USER> and <CLUSTER> with your ACCESS credentials
scp autoqsar.sif <USER>@<CLUSTER>:~/autoqsar/
scp -r hpc/       <USER>@<CLUSTER>:~/autoqsar/

# Stage repo data to $SCRATCH (do this once; subsequent jobs reuse it)
rsync -av data/        <USER>@<CLUSTER>:$SCRATCH/autoqsar_in/data/
rsync -av model_cache/ <USER>@<CLUSTER>:$SCRATCH/autoqsar_in/model_cache/
rsync -av .cache/      <USER>@<CLUSTER>:$SCRATCH/autoqsar_in/.cache/
```

---

## 5 — Generate sub-manifests (local or on cluster)

```bash
# Generates hpc/task_manifest_gpu.tsv and hpc/task_manifest_cpu.tsv
make manifests    # or run the awk commands from hpc/README.md §Manifest
```

Then update the `--array=0-N` directive in each sbatch script to match the
task count reported by `make manifests`.

---

## 6 — Fill operator placeholders

Edit `hpc/submit_multiseed.sbatch` and `hpc/submit_multiseed_cpu.sbatch` and
replace every `<PLACEHOLDER>`:

| Placeholder | What to fill |
|---|---|
| `<ACCOUNT>` | Your Slurm allocation account (e.g. `aab123`) |
| `<GPU_SHARED_PARTITION>` | GPU-shared partition (see cluster docs) |
| `<CPU_PARTITION>` | CPU-only partition |
| `<CUDA_OR_APPTAINER_MODULE>` | `module load apptainer` (or `cuda/12.1 apptainer`) |
| `<MAX_CONCURRENT>` | Max simultaneous jobs (respect fairshare; e.g. 10 GPU, 20 CPU) |

Examples by cluster:

| Cluster | GPU_SHARED_PARTITION | CPU_PARTITION | Module |
|---|---|---|---|
| Delta (NCSA) | `gpuA100x4-shared` | `cpu` | `module load apptainer` |
| Bridges-2 (PSC) | `GPU-shared` | `RM` | `module load apptainer` |
| Expanse (SDSC) | `gpu-shared` | `compute` | `module load singularitypro` |
| Stampede3 (TACC) | check cluster docs | `normal` | check cluster docs |

---

## 7 — Submit jobs

```bash
# On the HPC login node
export AUTOQSAR_INPUT=$SCRATCH/autoqsar_in
export AUTOQSAR_OUTPUT=$SCRATCH/autoqsar_out
export SIF=~/autoqsar/autoqsar.sif

# Submit GPU array (MapLight+GNN, TabPFN, Chemprop winners)
sbatch hpc/submit_multiseed.sbatch

# Submit CPU array (conventional ML, CFA fusion, ensemble winners)
sbatch hpc/submit_multiseed_cpu.sbatch

# Monitor
squeue -u $USER
```

---

## 8 — Retrieving results

Each task writes to `$AUTOQSAR_OUTPUT/<dataset>/seed_<seed>/`:

```
metrics.csv               per-model test metrics
predictions.csv           test-set SMILES + predicted + true values
selected_features.csv     retained feature names after ElasticNetCV selection
run_one_provenance.json   image SHA-256, git commit, device, resolved seed
run_status.json           pass/fail, elapsed wall time
```

Sync results back:

```bash
rsync -av <USER>@<CLUSTER>:$SCRATCH/autoqsar_out/ ./benchmark_results/multiseed_hpc/
```

---

## 9 — Reproducing a single result locally

```bash
apptainer run --nv \
    --bind ./:/in \
    --bind /tmp/autoqsar_out:/out \
    autoqsar.sif \
        --dataset tdc_clearance_microsome_az \
        --seed 3 \
        --input-dir /in \
        --output-dir /out \
        --device auto
```

Every artifact includes `run_one_provenance.json` recording the image SHA-256,
git commit, CUDA version, GPU model, and resolved seed, making all results
independently traceable.

---

---

## 10 — Jetstream2 cloud execution path

Jetstream2 (NSF ACCESS cloud, OpenStack) is an alternative to Slurm HPC. The
orchestrator at `js2/run_queue.py` provides:

- Resumable queue via JSON state file (survives shelve/unshelve cycles)
- Automatic SU burn-rate tracking (g3.large = 32 SU/hr)
- `--su-budget` with 80% warning and auto-shelve at 100%

See `js2/README.md` for full documentation including instance flavor table,
persistent volume layout, conformer pre-generation, and quick-start examples.

```bash
# Dry-run to validate manifest before burning SU
python js2/run_queue.py \
    --manifest js2/manifests/gpu_tasks.tsv \
    --sif ~/autoqsar/autoqsar.sif \
    --input-dir /vol/autoqsar/autoqsar_in \
    --output-dir /vol/autoqsar/autoqsar_out \
    --su-budget 200 \
    --dry-run
```

The v1 Slurm scripts (`hpc/`) are unchanged and remain the primary path for
NSF ACCESS HPC clusters.

---

## 11 — Precision and Uni-Mol2 options (Update 2)

### Precision flag

| Flag | Effect | Use when |
|---|---|---|
| `--precision tf32_bf16` | TF32 matmul + BF16 autocast for supported workflows | Production throughput on A100/g3.large |
| `--precision fp32` | Strict FP32, TF32 disabled | Reproducibility reference run |

Default is `tf32_bf16`. BF16 autocast is **not** applied to TabPFN (blocklisted
due to float64 internal paths). All precision decisions are logged and recorded
in `run_one_provenance.json`.

### Uni-Mol2 V2

```bash
apptainer run --nv autoqsar.sif \
    --dataset tdc_ppbr_az --seed 1 \
    --input-dir /in --output-dir /out \
    --run-unimol-v2 \
    --unimol-size 84m \
    --conformer-cache /in/conformer_cache \
    --device auto
```

The Uni-Mol2 workflow uses its own 3D geometry representation path (ETKDGv3+MMFF
conformers cached in SQLite) and does NOT consume MapLight/ElasticNet features.
Results land in `unimolv2_predictions.csv` and `unimolv2_metadata.json` alongside
the standard benchmark outputs.

Operator must:
1. Download the 84m checkpoint to `model_cache/unimolv2_checkpoints/84m/checkpoint.pt`
2. Record its SHA-256 in `autoqsar/unimolv2.py:UNIMOLV2_CHECKPOINT_SHA256["84m"]`

### Conformer pre-generation (CPU mode)

```bash
apptainer run autoqsar.sif \
    --generate-conformers-only \
    --dataset tdc_ppbr_az \
    --input-dir /in \
    --conformer-cache /in/conformer_cache \
    --conformer-seed 42
```

This mode generates and caches 3D conformers on CPU with no GPU required.
Run this before GPU tasks to eliminate conformer-gen overhead on the g3.large.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA error: no kernel image` | Container CUDA 12.1 wheel vs. host driver mismatch | Check driver with `nvidia-smi`; driver must support CUDA ≥ 12.1 |
| `ModuleNotFoundError: lightgbm` | Old cache reused from pre-fix build | Rebuild `.sif` from this branch |
| Feature store reads fail | `model_cache/` not in bind-mount | Ensure `/in` bind includes `model_cache/` sub-tree |
| `ERROR: manifest not found` | Sub-manifests not generated | Run `make manifests` locally, then copy to cluster |
| Out-of-memory on GPU | Request more VRAM or use `--device cpu` | Edit sbatch `--mem` or switch to CPU script |
| Uni-Mol2 CUDA OOM | batch size too large | The OOM-retry loop halves batch size up to 3 times automatically |
| Uni-Mol2 checkpoint SHA-256 mismatch | wrong file downloaded | Re-download and update `UNIMOLV2_CHECKPOINT_SHA256` in `autoqsar/unimolv2.py` |
| `openstack server shelve` fails | credentials not configured | Run `openstack` CLI setup or shelve manually via Horizon web UI |
| Conformer cache empty on GPU VM | pre-gen step was skipped | Run `--generate-conformers-only` on a CPU instance first, then sync cache |
