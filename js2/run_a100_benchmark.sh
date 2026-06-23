#!/usr/bin/env bash
# js2/run_a100_benchmark.sh — Canonical A100 launcher for the AutoQSAR full benchmark.
#
# Resource profile: NVIDIA A100 (40 GB or 80 GB), 32 CPU cores, ~117 GB RAM.
# All Uni-Mol and Chemprop parameters are set explicitly so this script is
# fully self-documenting and reproducible on any compatible A100 system.
# GPU-aware auto-detection in run_autoqsar_ga_benchmarks.py will also apply
# these same values automatically when no CLI override is provided.
#
# Usage (fresh run):
#   bash js2/run_a100_benchmark.sh
#
# Usage (resume an interrupted run on the same or a different A100):
#   bash js2/run_a100_benchmark.sh --output-dir /path/to/existing/benchmark_results/autoqsar_benchmark_<timestamp>
#
# Transfer an in-progress run to a new machine:
#   rsync -av --exclude='*.pth' --exclude='*.pt' --exclude='*.joblib' \
#       benchmark_results/autoqsar_benchmark_<timestamp>/ \
#       user@new-a100:/path/to/AutoQSAR/benchmark_results/autoqsar_benchmark_<timestamp>/
#   Then run: bash js2/run_a100_benchmark.sh --output-dir benchmark_results/autoqsar_benchmark_<timestamp>
#
# Parallelism controls (environment variables — override before invoking the script):
#   AUTOQSAR_PARALLEL_DATASETS   number of datasets run concurrently (default: 2 on A100, set to 1 for serial)
#                                Example: AUTOQSAR_PARALLEL_DATASETS=1 bash js2/run_a100_benchmark.sh
#   AUTOQSAR_PARALLEL_TDC22_SEEDS  "true" or "false" (default: true) — whether to run TDC-22 multi-seeds in parallel
#
# Settings rationale (A100-40 GB, 32 cores):
#   --n-jobs 0                    use all 32 detected CPU cores for conventional ML
#   --unimol-epochs 20            meaningful fine-tuning in ~5-20 min per dataset
#   --unimol-batch-size 32        84m/batch-32: attention+pair matrices ~16 GB; safe with zombie VRAM ctx
#   --unimol-early-stopping 5     5 non-improving epochs before stopping
#   --unimol-num-workers 8        auto_data_loader_workers() = min(8, 32//4) = 8
#   --chemprop-epochs 40          standard for benchmark parity
#   --chemprop-batch-size 32      Chemprop default; A100 handles this trivially
#   --chemprop-num-workers 8      same as unimol-num-workers
#   --benchmark-profile full      enable all model variants (Chemprop DMPNN/CMPNN, etc.)
#   --run-unimol-v1               GPU auto-enables V1; explicit here for clarity
#   --run-unimol-v2               GPU auto-enables V2; explicit here for clarity
#   --unimol-model-size 84m       84m: 12 transformer layers, ~9 GB training peak; fits safely on A100-40 GB
#                                  alongside V1 (total ~18 GB). 164m (24 layers) needs ~28 GB which
#                                  exceeds available headroom on a 40 GB A100 after zombie CUDA contexts.
#   --unimol-max-atoms 64         capped from 96: pair matrices scale as n_atoms^2, so 64 vs 96
#                                  reduces peak VRAM by (64/96)^2 ≈ 44%; covers >95% of drug molecules
#   --unimol-use-amp              enable AMP (automatic mixed precision) for V2 on A100
#   --resume                      pick up any compatible incomplete run automatically
#   --unimol-reuse-model-cache    skip re-training if weights already exist
#   --chemprop-reuse-model-cache  same for Chemprop
#   --parallel-datasets N         run N datasets concurrently (each gets n_jobs/N cores)
#   --parallel-tdc22-seeds        run TDC-22 best-model re-seeds (5 seeds) in parallel at end

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${AUTOQSAR_CONDA_ENV:-autoqsar-py311}"
OUTPUT_DIR_ARG=""

# Parallelism: off by default (serial). Opt in via env var when confident in stability.
#   AUTOQSAR_PARALLEL_DATASETS=2 bash js2/run_a100_benchmark.sh   # 2 concurrent datasets
#   AUTOQSAR_PARALLEL_TDC22_SEEDS=true bash js2/run_a100_benchmark.sh   # parallel seeds
PARALLEL_DATASETS="${AUTOQSAR_PARALLEL_DATASETS:-1}"
PARALLEL_TDC22_SEEDS="${AUTOQSAR_PARALLEL_TDC22_SEEDS:-false}"

# Parse --output-dir; everything else passes through to the Python script
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir=*)
            OUTPUT_DIR_ARG="--output-dir ${1#*=}"
            shift ;;
        --output-dir)
            OUTPUT_DIR_ARG="--output-dir $2"
            shift 2 ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift ;;
    esac
done

# Build --parallel-tdc22-seeds flag
if [ "${PARALLEL_TDC22_SEEDS}" = "true" ]; then
    SEEDS_FLAG="--parallel-tdc22-seeds"
else
    SEEDS_FLAG="--no-parallel-tdc22-seeds"
fi

LOG_FILE="$REPO_ROOT/benchmark_run.log"
MAX_RETRIES="${AUTOQSAR_MAX_RETRIES:-20}"
RETRY_DELAY="${AUTOQSAR_RETRY_DELAY:-15}"

echo "=== AutoQSAR A100 Full Benchmark ==="
echo "Repo:               $REPO_ROOT"
echo "Conda env:          $CONDA_ENV"
echo "Output:             ${OUTPUT_DIR_ARG:-auto-timestamped under benchmark_results/}"
echo "Parallel datasets:  $PARALLEL_DATASETS  (set AUTOQSAR_PARALLEL_DATASETS=1 for serial/low-resource)"
echo "Parallel TDC-22:    $PARALLEL_TDC22_SEEDS  (set AUTOQSAR_PARALLEL_TDC22_SEEDS=false to disable)"
echo "Auto-resume:        up to $MAX_RETRIES retries on crash (AUTOQSAR_MAX_RETRIES to change)"
echo "Started:            $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

cd "$REPO_ROOT"

# Lockfile: prevent two wrapper instances from running simultaneously.
# The auto-resume loop means kills of the Python child won't start a second wrapper,
# but manually launching a second script invocation would cause dual GPU usage.
LOCKFILE="/tmp/autoqsar_benchmark_${USER}.lock"
if [ -f "$LOCKFILE" ]; then
    LOCK_PID=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "ERROR: Another benchmark is already running (PID $LOCK_PID, lock: $LOCKFILE)" >&2
        echo "       Kill it first: kill -9 $LOCK_PID" >&2
        exit 1
    else
        echo "Stale lock found (PID $LOCK_PID no longer running). Removing." | tee -a "$LOG_FILE"
        rm -f "$LOCKFILE"
    fi
fi
echo "$$" > "$LOCKFILE"
trap "rm -f '$LOCKFILE'" EXIT INT TERM

# Disable exit-on-error so we can handle crash exit codes ourselves in the retry loop.
set +e

attempt=1
EXIT_CODE=0
while [ "$attempt" -le "$MAX_RETRIES" ]; do
    if [ "$attempt" -gt 1 ]; then
        {
            echo ""
            echo "=== Auto-resume attempt $attempt / $MAX_RETRIES: $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
        } | tee -a "$LOG_FILE"
    fi

    conda run -n "$CONDA_ENV" \
        python portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py \
            --benchmark-profile full \
            --run-unimol-v1 \
            --run-unimol-v2 \
            --unimol-model-size 84m \
            --unimol-max-atoms 64 \
            --unimol-use-amp \
            --n-jobs 0 \
            --unimol-epochs 20 \
            --unimol-batch-size 32 \
            --unimol-early-stopping 5 \
            --unimol-num-workers 8 \
            --unimol-reuse-model-cache \
            --chemprop-epochs 40 \
            --chemprop-batch-size 32 \
            --chemprop-num-workers 8 \
            --chemprop-reuse-model-cache \
            --parallel-datasets "$PARALLEL_DATASETS" \
            $SEEDS_FLAG \
            --resume \
            $OUTPUT_DIR_ARG \
            "${PASSTHROUGH_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE="${PIPESTATUS[0]}"

    if [ "$EXIT_CODE" -eq 0 ]; then
        break
    fi

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        break
    fi

    {
        echo ""
        echo "=== Benchmark exited with code $EXIT_CODE at $(date -u '+%Y-%m-%dT%H:%M:%SZ'). Retrying in ${RETRY_DELAY}s (attempt $attempt / $MAX_RETRIES)... ==="
    } | tee -a "$LOG_FILE"
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
done

echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "=== Benchmark complete: $(date -u '+%Y-%m-%dT%H:%M:%SZ') ===" | tee -a "$LOG_FILE"
else
    echo "=== Benchmark failed (exit $EXIT_CODE) after $attempt attempt(s): $(date -u '+%Y-%m-%dT%H:%M:%SZ') ===" | tee -a "$LOG_FILE"
    exit "$EXIT_CODE"
fi
