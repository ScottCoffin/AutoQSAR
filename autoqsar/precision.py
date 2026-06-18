"""
autoqsar.precision — TF32/BF16 precision startup hook.

Activated by AUTOQSAR_PRECISION env var (set by run_one.py before
launching any subprocess) and by a 3-line call in the benchmark runner's
main() — see portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.py.

Two modes:
  tf32_bf16  Enables TF32 matmul/cuDNN + BF16 autocast where amp_ok.
             Introduces minor numerical non-determinism vs strict FP32;
             this is logged prominently so results remain interpretable.
             Use for production throughput runs on A100/g3.large.

  fp32       Disables TF32; no autocast. Strictly reproducible.
             Use for the determinism/reproducibility reference run.

Workflow AMP eligibility (amp_ok):
  - Uni-Mol2 (§B) : True  — designed for bf16 inference
  - Chemprop v2   : True  — PyTorch Lightning handles AMP scaler correctly
  - ChemML nets   : True  — simple MLP, stable in bf16
  - MapLight+GNN  : True  — GIN encoder + CatBoost head; AMP on GIN only
  - TabPFN        : False — uses float64 attention internally in some paths
  - Conventional ML / CatBoost / XGBoost : not applicable (no torch)

vGPU unified-memory guard (A5):
  Jetstream2 g3.large is a 50% A100 vGPU slice; cudaMallocManaged is
  unsupported. We check for vGPU partition and log a warning. There is
  no portable way to intercept allocator calls across all libraries,
  so the check is advisory.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Iterator

logger = logging.getLogger(__name__)

# Workflows known NOT to be stable under bf16 autocast
_AMP_BLOCKLIST: frozenset[str] = frozenset({"tabpfn", "tabpfn_regressor", "tabpfn_classifier"})

# Workflows confirmed AMP-safe
_AMP_ALLOWLIST: frozenset[str] = frozenset({
    "unimolv2",
    "chemprop",
    "chemml_pytorch",
    "maplight_gnn",
})


def apply_global_precision(mode: str) -> None:
    """Apply global PyTorch precision settings.

    Call this once at process startup, before any CUDA operations.
    The mode is read from AUTOQSAR_PRECISION env var by the benchmark runner.
    """
    mode = (mode or "fp32").strip().lower()
    if mode not in ("tf32_bf16", "fp32"):
        logger.warning("Unknown precision mode %r; defaulting to fp32", mode)
        mode = "fp32"

    try:
        import torch
    except ImportError:
        logger.debug("torch not available; precision hook is a no-op")
        return

    if not torch.cuda.is_available():
        logger.debug("No CUDA device; precision hook is a no-op (CPU run)")
        return

    if mode == "tf32_bf16":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info(
            "Precision mode: tf32_bf16 — TF32 matmul/cuDNN enabled, BF16 autocast "
            "available for supported workflows. NOTE: TF32 introduces minor numerical "
            "non-determinism relative to strict FP32. Use --precision fp32 for the "
            "reproducibility reference run."
        )
        _check_vgpu_unified_memory()
    else:
        # Explicit fp32: ensure TF32 is off (it may be on by default in older torch)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        logger.info("Precision mode: fp32 — TF32 disabled, strict FP32 matmul.")


def _check_vgpu_unified_memory() -> None:
    """Warn if running on a vGPU slice that does not support unified memory."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,vbios_version,mig.mode.current",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.strip()
        if "mig" in output.lower() or "virtual" in output.lower() or "vgpu" in output.lower():
            logger.warning(
                "vGPU or MIG slice detected (%s). cudaMallocManaged (CUDA unified "
                "memory) is NOT supported on this device configuration. If any "
                "dependency triggers a unified-memory allocation, the process will "
                "crash. Avoid torch.cuda.memory.set_per_process_memory_fraction() "
                "and any code path that calls cudaMallocManaged.",
                output,
            )
    except Exception:
        pass  # nvidia-smi not available or timed out; skip check


def amp_context(workflow: str, mode: str, device: str) -> "_AmpContext":
    """Return a context manager that enables BF16 autocast if appropriate.

    Usage:
        with amp_context("chemprop", args.precision, "cuda") as enabled:
            loss = model(batch)
            loss.backward()
        # enabled is True when autocast was active, False otherwise

    Only activates when mode == "tf32_bf16", device == "cuda", and the
    workflow is in the AMP allowlist (not in the blocklist).
    """
    active = (
        mode == "tf32_bf16"
        and device == "cuda"
        and workflow.lower() in _AMP_ALLOWLIST
        and workflow.lower() not in _AMP_BLOCKLIST
    )
    return _AmpContext(active, workflow)


class _AmpContext:
    def __init__(self, active: bool, workflow: str) -> None:
        self._active = active
        self._workflow = workflow
        self._ctx = None

    def __enter__(self) -> bool:
        if self._active:
            try:
                import torch
                self._ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                self._ctx.__enter__()
                logger.debug("BF16 autocast enabled for workflow: %s", self._workflow)
            except Exception as exc:
                logger.warning("Failed to activate BF16 autocast for %s: %s", self._workflow, exc)
                self._active = False
        return self._active

    def __exit__(self, *args: object) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*args)
            self._ctx = None


def precision_metadata(mode: str, device: str) -> dict:
    """Return a dict of precision-related provenance fields for run records."""
    meta: dict = {
        "precision_mode": mode or "fp32",
        "tf32_enabled": mode == "tf32_bf16" and device == "cuda",
        "bf16_amp_available": mode == "tf32_bf16" and device == "cuda",
    }
    try:
        import torch
        if torch.cuda.is_available():
            meta["torch_float32_matmul_precision"] = torch.get_float32_matmul_precision()
            meta["torch_backends_cuda_matmul_allow_tf32"] = bool(
                torch.backends.cuda.matmul.allow_tf32
            )
    except Exception:
        pass
    return meta
