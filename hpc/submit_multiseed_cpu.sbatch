#!/bin/bash
# submit_multiseed_cpu.sbatch — CPU-only job array for TDC-22 multi-seed evaluation
#
# Routes datasets whose winning model is conventional ML, CFA fusion, or ensemble
# meta-model — all of which run on CPU (no GPU required).
#
# Operator TODOs:
#   <ACCOUNT>              : your Slurm allocation account
#   <CPU_PARTITION>        : CPU-only partition (e.g. cpu on Bridges-2,
#                            compute on Expanse, cpu on Delta)
#   <CUDA_OR_APPTAINER_MODULE> : e.g. "module load apptainer"
#   <MAX_CONCURRENT>       : max simultaneous CPU tasks (e.g. 20–40)

#SBATCH --job-name=autoqsar_multiseed_cpu
#SBATCH --account=<ACCOUNT>                  # TODO: replace with your allocation
#SBATCH --partition=<CPU_PARTITION>          # TODO: replace with cluster CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-74%<MAX_CONCURRENT>        # TODO: set <MAX_CONCURRENT> (e.g. 20)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module load <CUDA_OR_APPTAINER_MODULE>       # TODO: replace with cluster module name(s)

set -euo pipefail
mkdir -p logs

# ── Resolve (dataset, seed) from CPU manifest ────────────────────────────────
MANIFEST="${SLURM_SUBMIT_DIR}/hpc/task_manifest_cpu.tsv"
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST" >&2
    exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
DATASET=$(echo "$LINE" | cut -f1)
SEED=$(echo "$LINE"    | cut -f2)

if [[ -z "$DATASET" || -z "$SEED" ]]; then
    echo "ERROR: empty dataset/seed at array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID (CPU): dataset=$DATASET seed=$SEED"

AUTOQSAR_INPUT="${AUTOQSAR_INPUT:-$SCRATCH/autoqsar_in}"
AUTOQSAR_OUTPUT="${AUTOQSAR_OUTPUT:-$SCRATCH/autoqsar_out}"
SIF="${SIF:-$SLURM_SUBMIT_DIR/autoqsar.sif}"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: .sif not found at $SIF" >&2
    exit 1
fi

# Stage to node-local NVMe if available
if [[ -n "${TMPDIR:-}" && -d "$TMPDIR" && "$AUTOQSAR_INPUT" != "$TMPDIR/autoqsar_in" ]]; then
    mkdir -p "$TMPDIR/autoqsar_in"
    rsync -a --quiet "$AUTOQSAR_INPUT/data/"        "$TMPDIR/autoqsar_in/data/"       2>/dev/null || true
    rsync -a --quiet "$AUTOQSAR_INPUT/model_cache/" "$TMPDIR/autoqsar_in/model_cache/" 2>/dev/null || true
    rsync -a --quiet "$AUTOQSAR_INPUT/.cache/"      "$TMPDIR/autoqsar_in/.cache/"     2>/dev/null || true
    LOCAL_INPUT="$TMPDIR/autoqsar_in"
else
    LOCAL_INPUT="$AUTOQSAR_INPUT"
fi

# Run WITHOUT --nv (no GPU)
apptainer run \
    --bind "${LOCAL_INPUT}:/in" \
    --bind "${AUTOQSAR_OUTPUT}:/out" \
    "$SIF" \
        --dataset "$DATASET" \
        --seed    "$SEED" \
        --input-dir  /in \
        --output-dir /out \
        --device cpu \
        --no-run-maplight-gnn \
        --no-run-chemprop-mpnn \
        --no-run-chemprop-attentivefp \
        --no-run-unimol-v1 \
        --no-run-tabpfn

echo "Task $SLURM_ARRAY_TASK_ID (CPU) completed: $DATASET seed=$SEED"
