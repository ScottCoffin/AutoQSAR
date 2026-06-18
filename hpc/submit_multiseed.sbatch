#!/bin/bash
# submit_multiseed.sbatch — GPU job array for TDC-22 multi-seed evaluation
#
# Dispatches one Apptainer container invocation per (dataset, seed) task.
# GPU tasks: MapLight+GNN, Chemprop v2, TabPFN winners (see hpc/README.md §Routing).
#
# Usage:
#   sbatch hpc/submit_multiseed.sbatch
#
# Operator TODOs (cluster-specific — fill before submitting):
#   <ACCOUNT>              : your Slurm allocation account
#   <GPU_SHARED_PARTITION> : GPU-shared partition (e.g. gpuA100x4-shared on Delta,
#                            gpu on Bridges-2, gpu-shared on Expanse)
#   <CUDA_OR_APPTAINER_MODULE> : e.g. "module load cuda/12.1 apptainer"
#   <MAX_CONCURRENT>       : max simultaneous tasks (e.g. 10–20 to stay within fairshare)
#
# Array size: 110 tasks total (22 TDC-22 datasets × 5 seeds).
# GPU subset (this script): tasks where device==gpu in hpc/task_manifest_gpu.tsv
# See generate_manifests.py or hpc/README.md §Manifest to generate the GPU-only sub-manifest.

#SBATCH --job-name=autoqsar_multiseed_gpu
#SBATCH --account=<ACCOUNT>                  # TODO: replace with your allocation
#SBATCH --partition=<GPU_SHARED_PARTITION>   # TODO: replace with cluster GPU-shared partition
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-34%<MAX_CONCURRENT>        # TODO: set <MAX_CONCURRENT> (e.g. 10)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
module load <CUDA_OR_APPTAINER_MODULE>       # TODO: replace with cluster module name(s)

# Fail fast on any error
set -euo pipefail

# Create log directory if missing
mkdir -p logs

# ── Resolve (dataset, seed) from manifest ───────────────────────────────────
# hpc/task_manifest_gpu.tsv contains only GPU-routed tasks (device==gpu rows
# from hpc/task_manifest.tsv, without the header or "notes" column).
MANIFEST="${SLURM_SUBMIT_DIR}/hpc/task_manifest_gpu.tsv"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST" >&2
    exit 1
fi

# +1 to skip header; task IDs are 0-indexed
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
DATASET=$(echo "$LINE" | cut -f1)
SEED=$(echo "$LINE"    | cut -f2)

if [[ -z "$DATASET" || -z "$SEED" ]]; then
    echo "ERROR: empty dataset/seed at array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET seed=$SEED"

# ── Paths ────────────────────────────────────────────────────────────────────
# Operator: set AUTOQSAR_INPUT to the directory containing data/, model_cache/,
# .cache/ relative to the repo root structure. Typically stage to $SCRATCH before
# the job runs (see hpc/README.md §Staging).
AUTOQSAR_INPUT="${AUTOQSAR_INPUT:-$SCRATCH/autoqsar_in}"
AUTOQSAR_OUTPUT="${AUTOQSAR_OUTPUT:-$SCRATCH/autoqsar_out}"
SIF="${SIF:-$SLURM_SUBMIT_DIR/autoqsar.sif}"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: .sif not found at $SIF" >&2
    exit 1
fi

# ── Stage inputs to node-local NVMe if available ─────────────────────────────
# NODE_LOCAL_STORAGE is set to $TMPDIR on most ACCESS clusters with NVMe scratch.
# If not available, we use the shared AUTOQSAR_INPUT directly.
if [[ -n "${TMPDIR:-}" && -d "$TMPDIR" && "$AUTOQSAR_INPUT" != "$TMPDIR/autoqsar_in" ]]; then
    echo "Staging input to node-local NVMe: $TMPDIR/autoqsar_in"
    mkdir -p "$TMPDIR/autoqsar_in"
    # Rsync only the needed sub-trees to keep staging fast
    rsync -a --quiet \
        "$AUTOQSAR_INPUT/data/"        "$TMPDIR/autoqsar_in/data/"       2>/dev/null || true
    rsync -a --quiet \
        "$AUTOQSAR_INPUT/model_cache/" "$TMPDIR/autoqsar_in/model_cache/" 2>/dev/null || true
    rsync -a --quiet \
        "$AUTOQSAR_INPUT/.cache/"      "$TMPDIR/autoqsar_in/.cache/"     2>/dev/null || true
    LOCAL_INPUT="$TMPDIR/autoqsar_in"
else
    LOCAL_INPUT="$AUTOQSAR_INPUT"
fi

# ── Run container ─────────────────────────────────────────────────────────────
apptainer run --nv \
    --bind "${LOCAL_INPUT}:/in" \
    --bind "${AUTOQSAR_OUTPUT}:/out" \
    "$SIF" \
        --dataset "$DATASET" \
        --seed    "$SEED" \
        --input-dir  /in \
        --output-dir /out \
        --device auto

echo "Task $SLURM_ARRAY_TASK_ID completed: $DATASET seed=$SEED"
