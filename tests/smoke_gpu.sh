#!/usr/bin/env bash
# smoke_gpu.sh — GPU smoke test for the AutoQSAR container
#
# Runs one (dataset, seed) unit on GPU inside the Apptainer container and
# asserts that the expected artifacts exist and the primary metric is finite.
# Skips gracefully if no NVIDIA GPU is visible.
#
# Usage:
#   bash tests/smoke_gpu.sh [SIF_PATH] [INPUT_DIR] [OUTPUT_DIR]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF="${1:-${REPO_ROOT}/autoqsar.sif}"
INPUT_DIR="${2:-${REPO_ROOT}}"
OUTPUT_DIR="${3:-/tmp/autoqsar_smoke_gpu_$$}"
DATASET="tdc_herg"
SEED=1

echo "=== AutoQSAR GPU smoke test ==="
echo "  SIF        : $SIF"
echo "  INPUT_DIR  : $INPUT_DIR"
echo "  OUTPUT_DIR : $OUTPUT_DIR"
echo "  dataset    : $DATASET  seed=$SEED"
echo ""

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -f "$SIF" ]]; then
    echo "SKIP: $SIF not found. Run 'make sif' to build the container first."
    exit 0
fi

if ! command -v apptainer &>/dev/null && ! command -v singularity &>/dev/null; then
    echo "SKIP: neither 'apptainer' nor 'singularity' found in PATH."
    exit 0
fi

# Check for GPU
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
    echo "SKIP: No NVIDIA GPU detected (nvidia-smi not found or failed)."
    echo "      GPU smoke test requires an NVIDIA GPU with a compatible driver."
    exit 0
fi

echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

RUNNER="apptainer"
command -v apptainer &>/dev/null || RUNNER="singularity"

mkdir -p "$OUTPUT_DIR"

# ── Run container ─────────────────────────────────────────────────────────────
"$RUNNER" run --nv \
    --bind "${INPUT_DIR}:/in" \
    --bind "${OUTPUT_DIR}:/out" \
    "$SIF" \
        --dataset "$DATASET" \
        --seed    "$SEED" \
        --input-dir /in \
        --output-dir /out \
        --device auto \
        --row-limit 200

# ── Assertions ────────────────────────────────────────────────────────────────
ARTIFACT_DIR="${OUTPUT_DIR}/${DATASET}/seed_${SEED}"

echo ""
echo "Checking artifacts in $ARTIFACT_DIR ..."

MISSING=0
for f in metrics.csv run_one_provenance.json; do
    if [[ ! -f "${ARTIFACT_DIR}/$f" ]]; then
        echo "FAIL: missing ${ARTIFACT_DIR}/$f"
        MISSING=$((MISSING + 1))
    else
        echo "OK  : ${ARTIFACT_DIR}/$f"
    fi
done

# Check provenance records GPU info
if [[ -f "${ARTIFACT_DIR}/run_one_provenance.json" ]]; then
    GPU_FIELD=$(python3 -c "
import json
with open('${ARTIFACT_DIR}/run_one_provenance.json') as f:
    d = json.load(f)
print(d.get('gpu_model', 'unknown'))
" 2>/dev/null || echo "unknown")
    if [[ "$GPU_FIELD" == "none" || "$GPU_FIELD" == "unknown" ]]; then
        echo "WARN: provenance records gpu_model=$GPU_FIELD (expected a real GPU name)"
    else
        echo "OK  : GPU model in provenance: $GPU_FIELD"
    fi
fi

# Check that metrics.csv contains at least one finite metric value
if [[ -f "${ARTIFACT_DIR}/metrics.csv" ]]; then
    METRIC_VALUE=$(python3 -c "
import csv, math, sys
with open('${ARTIFACT_DIR}/metrics.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key in ('test_auroc', 'test_roc_auc', 'test_rmse', 'test_r2'):
            val = row.get(key, '')
            if val not in ('', 'None', 'nan', 'NaN'):
                try:
                    v = float(val)
                    if math.isfinite(v):
                        print(f'{key}={v:.4f}')
                        sys.exit(0)
                except ValueError:
                    pass
sys.exit(1)
" 2>/dev/null || echo "")

    if [[ -z "$METRIC_VALUE" ]]; then
        echo "FAIL: no finite metric found in metrics.csv"
        MISSING=$((MISSING + 1))
    else
        echo "OK  : finite metric found: $METRIC_VALUE"
    fi
fi

echo ""
if [[ $MISSING -eq 0 ]]; then
    echo "PASS: GPU smoke test completed successfully."
    exit 0
else
    echo "FAIL: $MISSING assertion(s) failed."
    exit 1
fi
