#!/usr/bin/env bash
# js2/preflight.sh — Jetstream2 VM first-boot preflight checks for AutoQSAR.
#
# Run once after provisioning a new g3.large or m3.medium instance before
# submitting any tasks via js2/run_queue.py.
#
# Usage:
#   bash js2/preflight.sh [--gpu-required]
#
# Exit codes:
#   0  all checks passed
#   1  a required check failed (error printed to stderr)

set -euo pipefail

GPU_REQUIRED=false
for arg in "$@"; do
    [[ "$arg" == "--gpu-required" ]] && GPU_REQUIRED=true
done

pass() { echo "[PASS] $*"; }
warn() { echo "[WARN] $*" >&2; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

echo "=== AutoQSAR Jetstream2 Preflight ($(date -u '+%Y-%m-%dT%H:%M:%SZ')) ==="

# ── 1. apptainer ─────────────────────────────────────────────────────────────
if command -v apptainer &>/dev/null; then
    APPTAINER_VER=$(apptainer version 2>/dev/null || true)
    pass "apptainer found: $APPTAINER_VER"
else
    fail "apptainer not found. Install with: sudo snap install apptainer --classic"
fi

# ── 2. GPU / nvidia-smi ──────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)

    echo "  GPU: $GPU_NAME"
    echo "  Driver: $DRIVER_VER"
    echo "  VRAM: ${VRAM_MIB} MiB"

    # CUDA 12.1 requires driver >= 525.60
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
    if [[ -n "$DRIVER_MAJOR" && "$DRIVER_MAJOR" -ge 525 ]]; then
        pass "Host driver $DRIVER_VER supports CUDA >= 12.1"
    else
        # TODO (operator): confirm driver version after first boot — g3.large
        # Jetstream2 nodes are provisioned with a specific NVIDIA vGPU driver.
        # The check below is advisory until you can verify the installed version.
        warn "Driver version $DRIVER_VER may be below 525.60 (needed for CUDA 12.1). " \
             "Verify with: nvidia-smi and consult Jetstream2 support if needed."
    fi

    # Verify VRAM >= 18 GB (g3.large 50% A100 slice has ~20 GB)
    if [[ -n "$VRAM_MIB" && "$VRAM_MIB" -ge 18000 ]]; then
        pass "VRAM ${VRAM_MIB} MiB >= 18000 MiB (sufficient for 84m Uni-Mol2 + GNN)"
    else
        warn "VRAM ${VRAM_MIB:-unknown} MiB is below recommended 18 GB. " \
             "Uni-Mol2 84m needs ~6 GB; GNN heads need additional VRAM."
    fi
else
    if $GPU_REQUIRED; then
        fail "nvidia-smi not found. This check requires an NVIDIA GPU. " \
             "Are you on a g3.large instance? If on m3.medium (CPU-only) remove --gpu-required."
    else
        warn "nvidia-smi not found. Assuming CPU-only instance (m3.medium). " \
             "GPU-dependent tasks (MapLight+GNN, Uni-Mol2) will fall back to CPU."
    fi
fi

# ── 3. openstack CLI (for auto-shelve) ───────────────────────────────────────
if command -v openstack &>/dev/null; then
    pass "openstack CLI found: $(openstack --version 2>&1 | head -1)"
else
    warn "openstack CLI not found. Auto-shelve (--su-budget) will not work. " \
         "Install with: pip install python-openstackclient"
fi

# ── 4. Persistent volume ─────────────────────────────────────────────────────
VOL_DIR="${AUTOQSAR_VOL:-/vol/autoqsar}"
if [[ -d "$VOL_DIR" ]]; then
    AVAIL=$(df -BG "$VOL_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo "unknown")
    pass "Persistent volume $VOL_DIR present (avail: $AVAIL)"
else
    warn "Persistent volume not found at $VOL_DIR. " \
         "Expected layout: $VOL_DIR/{autoqsar_in,autoqsar_out,conformer_cache}. " \
         "Create with: sudo mkdir -p $VOL_DIR && sudo chown \$USER:$USER $VOL_DIR"
fi

# ── 5. SIF present ────────────────────────────────────────────────────────────
SIF_PATH="${AUTOQSAR_SIF:-$HOME/autoqsar/autoqsar.sif}"
if [[ -f "$SIF_PATH" ]]; then
    SIF_SHA=$(sha256sum "$SIF_PATH" | cut -d' ' -f1)
    pass "autoqsar.sif found: $SIF_PATH (sha256: ${SIF_SHA:0:16}...)"
else
    warn "autoqsar.sif not found at $SIF_PATH. " \
         "Transfer with: scp autoqsar.sif user@js2.jetstream-cloud.org:~/autoqsar/"
fi

# ── 6. Record environment to disk ────────────────────────────────────────────
REPORT_PATH="${AUTOQSAR_VOL:-$HOME}/preflight_$(date -u '+%Y%m%d_%H%M%S').txt"
{
    echo "=== AutoQSAR Preflight Report ==="
    echo "date_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "hostname: $(hostname)"
    echo "kernel:   $(uname -r)"
    echo ""
    command -v nvidia-smi &>/dev/null && nvidia-smi || echo "nvidia-smi: not found"
    echo ""
    command -v apptainer &>/dev/null && apptainer version || echo "apptainer: not found"
    echo ""
    echo "disk:"
    df -h 2>/dev/null || true
} > "$REPORT_PATH" 2>&1
echo "Preflight report written to $REPORT_PATH"

echo "=== Preflight complete ==="
