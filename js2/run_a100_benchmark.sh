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
#   --unimol-batch-size 128       A100-40GB has 39+ GB VRAM; 128 fits without OOM
#   --unimol-early-stopping 5     5 non-improving epochs before stopping
#   --unimol-num-workers 8        auto_data_loader_workers() = min(8, 32//4) = 8
#   --chemprop-epochs 40          standard for benchmark parity
#   --chemprop-batch-size 32      Chemprop default; A100 handles this trivially
#   --chemprop-num-workers 8      same as unimol-num-workers
#   --benchmark-profile full      enable all model variants (Chemprop DMPNN/CMPNN, etc.)
#   --run-unimol-v1               GPU auto-enables V1; explicit here for clarity
#   --run-unimol-v2               GPU auto-enables V2; explicit here for clarity
#   --unimol-model-size 84m       V2 model variant: 84m (fast), 164m, or 310m (most accurate)
#   --unimol-max-atoms 96         max atoms per molecule for Uni-Mol V2
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

echo "=== AutoQSAR A100 Full Benchmark ==="
echo "Repo:               $REPO_ROOT"
echo "Conda env:          $CONDA_ENV"
echo "Output:             ${OUTPUT_DIR_ARG:-auto-timestamped under benchmark_results/}"
echo "Parallel datasets:  $PARALLEL_DATASETS  (set AUTOQSAR_PARALLEL_DATASETS=1 for serial/low-resource)"
echo "Parallel TDC-22:    $PARALLEL_TDC22_SEEDS  (set AUTOQSAR_PARALLEL_TDC22_SEEDS=false to disable)"
echo "Started:            $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

cd "$REPO_ROOT"

conda run -n "$CONDA_ENV" \
    python portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py \
        --benchmark-profile full \
        --run-unimol-v1 \
        --run-unimol-v2 \
        --unimol-model-size 310m \
        --unimol-max-atoms 96 \
        --unimol-use-amp \
        --n-jobs 0 \
        --unimol-epochs 20 \
        --unimol-batch-size 128 \
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
    2>&1 | tee "$REPO_ROOT/benchmark_run.log"

echo ""
echo "=== Benchmark complete: $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
