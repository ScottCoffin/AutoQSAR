"""
run_one.py — thin, seed-injectable CLI wrapper for one (dataset, seed) unit.

Designed for Apptainer/Slurm job-array dispatch:

    python run_one.py \
        --dataset tdc_caco2_wang \
        --seed 3 \
        --config /opt/autoqsar/run_config.json \
        --input-dir /in \
        --output-dir /out \
        --device auto \
        --precision tf32_bf16

All file I/O is routed through --input-dir / --output-dir.
Artifacts land under  <output-dir>/<dataset>/seed_<seed>/.

Randomness is fully deterministic from --seed:
  - Python random, NumPy, PyTorch (CPU + CUDA), chemprop_random_seed
  - PYTHONHASHSEED exported before any imports
  - Conformer generation uses --conformer-seed (separate from model seed)

Precision (--precision):
  tf32_bf16  Enable TF32 matmul/cuDNN + BF16 autocast where amp_ok.
             Faster on A100/g3.large; introduces minor numerical nondeterminism.
  fp32       Strict FP32; fully deterministic reference runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _set_env_before_imports(seed: int) -> None:
    """Set PYTHONHASHSEED before any imports that might use it."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _seed_everything(seed: int, device: str) -> None:
    """Seed Python random, NumPy, and torch (CPU + CUDA)."""
    import random as _random
    _random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    if device != "cpu":
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # Enable reproducible CUDA ops where supported.
                # CuBLAS non-determinism is unavoidable for some ops;
                # this catches the rest at modest runtime cost.
                torch.use_deterministic_algorithms(True, warn_only=True)
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except ImportError:
            pass


def _detect_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    # auto: prefer CUDA when available
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _gpu_model(device: str) -> str:
    if device != "cuda":
        return "none"
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or "none"
    except Exception:
        return "none"


def _host_driver_version() -> str:
    """Return the NVIDIA host driver version string via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "unknown"


def _image_hash(image_hash_file: Path | None) -> str:
    """Read pre-recorded .sif SHA-256 from a sidecar file, if present."""
    candidates = [
        image_hash_file,
        Path(__file__).resolve().parent / "BUILD_PROVENANCE.md",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if "sif_sha256" in line.lower() or "image_sha256" in line.lower():
                    parts = line.split(":")
                    if len(parts) >= 2:
                        return parts[-1].strip()
    return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (e.g. tdc_caco2_wang). Passed as --dataset-name to the benchmark runner.",
    )
    parser.add_argument(
        "--seed", type=int, default=13,
        help="Master random seed (default: 13). Sets Python/NumPy/PyTorch/chemprop RNGs.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "Optional path to run_config.json. When supplied, non-seed/non-path args "
            "are loaded from it and may be overridden by additional CLI flags."
        ),
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help=(
            "Directory containing the repo's data/, model_cache/, and .cache/ sub-trees. "
            "Defaults to the repo root (parent of this file's directory)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root output directory. Artifacts land under <output-dir>/<dataset>/seed_<seed>/.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Device selection. auto detects CUDA; cpu forces CPU-only model families.",
    )
    # ── Precision (Update 2 §A) ──────────────────────────────────────────────
    parser.add_argument(
        "--precision", choices=["tf32_bf16", "fp32"], default="tf32_bf16",
        help=(
            "Floating-point precision mode for all torch-based workflows. "
            "tf32_bf16: enable TF32 matmul/cuDNN + BF16 autocast where safe (faster, "
            "minor nondeterminism). fp32: strict FP32 for reproducibility reference runs. "
            "Default: tf32_bf16 (production / A100 g3.large)."
        ),
    )
    # ── Uni-Mol2 V2 (Update 2 §B) ────────────────────────────────────────────
    parser.add_argument(
        "--run-unimol-v2", action=argparse.BooleanOptionalAction, default=False,
        help="Run Uni-Mol2 V2 workflow after the main benchmark runner stage.",
    )
    parser.add_argument(
        "--unimol-size", choices=["84m", "164m", "310m", "1.1b"], default="84m",
        help=(
            "Uni-Mol2 model size. 84m (default) fits both RTX 4070 (12 GB) and A100 "
            "g3.large (20 GB). Larger sizes warn about VRAM requirements."
        ),
    )
    parser.add_argument(
        "--unimol-v2-batch-size", type=int, default=32,
        help="Initial Uni-Mol2 batch size. Halved automatically on CUDA OOM.",
    )
    parser.add_argument(
        "--conformer-cache", type=Path, default=None,
        help=(
            "Persistent directory for pre-generated 3D conformers (ETKDGv3+MMFF). "
            "Conformers are keyed by canonical SMILES + seed; generated once and reused "
            "across all seeds. Defaults to <input-dir>/model_cache/conformer_cache."
        ),
    )
    parser.add_argument(
        "--conformer-seed", type=int, default=42,
        help="RDKit conformer generation seed (separate from the model seed, default: 42).",
    )
    parser.add_argument(
        "--generate-conformers-only", action="store_true",
        help=(
            "Run only the conformer-generation stage (§B4 standalone CPU step) "
            "then exit. Use on a cheap CPU VM before GPU runs."
        ),
    )
    # ── Provenance ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--image-hash-file", type=Path, default=None,
        help="Path to a file containing the .sif SHA-256, embedded in provenance records.",
    )
    # Passthrough: any extra --key value pairs are forwarded verbatim to the benchmark runner.
    parser.add_argument(
        "extra_args", nargs=argparse.REMAINDER,
        help="Extra flags forwarded verbatim to run_autoqsar_ga_benchmarks.py.",
    )
    return parser.parse_args(argv)


def build_runner_argv(
    args: argparse.Namespace,
    resolved_device: str,
    output_dir: Path,
) -> list[str]:
    """Assemble the argv list for the underlying benchmark runner."""
    runner = Path(__file__).resolve().parent / "portable_colab_qsar_bundle" / "run_autoqsar_ga_benchmarks.py"

    argv = [
        sys.executable, str(runner),
        "--dataset-name", args.dataset,
        "--random-seed", str(args.seed),
        "--chemprop-random-seed", str(args.seed),
        "--output-dir", str(output_dir),
    ]

    if resolved_device == "cpu":
        argv += ["--no-run-unimol-v1"]
    else:
        argv += ["--run-unimol-v1"]

    if args.input_dir is not None:
        input_root = Path(args.input_dir)
        argv += [
            "--persistent-feature-store-path",
            str(input_root / "model_cache" / "feature_store_parquet"),
            "--shared-feature-matrix-cache-path",
            str(input_root / "model_cache" / "benchmark_feature_matrix_cache"),
        ]

    if args.extra_args:
        extra = args.extra_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        argv += extra

    return argv


def _resolve_conformer_cache(args: argparse.Namespace) -> Path:
    if args.conformer_cache is not None:
        return Path(args.conformer_cache)
    if args.input_dir is not None:
        return Path(args.input_dir) / "model_cache" / "conformer_cache"
    return Path(__file__).resolve().parent / "model_cache" / "conformer_cache"


def write_provenance(
    output_dir: Path,
    dataset: str,
    seed: int,
    device: str,
    args: argparse.Namespace,
    conformer_cache_hash: str = "unknown",
) -> None:
    precision_mode = getattr(args, "precision", "fp32")
    unimol_size = getattr(args, "unimol_size", "none")
    conformer_seed = getattr(args, "conformer_seed", 42)

    try:
        from autoqsar.precision import precision_metadata
        prec_meta = precision_metadata(precision_mode, device)
    except ImportError:
        prec_meta = {"precision_mode": precision_mode}

    # Resolve GPU identity for g3.large vGPU provenance
    gpu_identity = _gpu_model(device)
    host_driver = _host_driver_version()

    record = {
        # v1 fields
        "dataset": dataset,
        "seed": seed,
        "device": device,
        "git_commit": _git_commit(),
        "cuda_version": _cuda_version(),
        "gpu_model": gpu_identity,
        "image_sha256": _image_hash(args.image_hash_file),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "run_one_version": "2.0.0",
        # Update 2 §D fields
        "precision_mode": precision_mode,
        **prec_meta,
        "host_driver_version": host_driver,
        "gpu_identity": gpu_identity,
        "unimolv2_size": unimol_size if getattr(args, "run_unimol_v2", False) else "not_run",
        "unimolv2_checkpoint_sha256": "TODO_operator_fill",
        "unimolv2_3d_input_flag": getattr(args, "run_unimol_v2", False),
        "conformer_method": "ETKDGv3+MMFF",
        "conformer_seed": conformer_seed,
        "conformer_cache_hash": conformer_cache_hash,
        "precision_nondeterminism_note": (
            "TF32/bf16 introduces minor numerical nondeterminism relative to fp32. "
            "Use --precision fp32 for byte-for-byte reproducibility."
            if precision_mode == "tf32_bf16" else "fp32 mode: strict reproducibility"
        ),
    }
    provenance_path = output_dir / "run_one_provenance.json"
    provenance_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    _set_env_before_imports(args.seed)

    # Propagate precision mode to subprocess before any torch is imported there.
    os.environ["AUTOQSAR_PRECISION"] = args.precision

    resolved_device = _detect_device(args.device)
    _seed_everything(args.seed, resolved_device)

    seed_output_dir = args.output_dir / args.dataset / f"seed_{args.seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    precision = args.precision
    conformer_cache_dir = _resolve_conformer_cache(args)

    print("=" * 72, flush=True)
    print(
        f"run_one.py  dataset={args.dataset}  seed={args.seed}  "
        f"device={resolved_device}  precision={precision}",
        flush=True,
    )
    print(f"  git commit    : {_git_commit()}", flush=True)
    print(f"  CUDA          : {_cuda_version()}", flush=True)
    print(f"  GPU           : {_gpu_model(resolved_device)}", flush=True)
    print(f"  host driver   : {_host_driver_version()}", flush=True)
    print(f"  output_dir    : {seed_output_dir}", flush=True)
    print(f"  conformer_cache: {conformer_cache_dir}", flush=True)
    if precision == "tf32_bf16":
        print(
            "  NOTE: tf32_bf16 mode — TF32 matmul + BF16 autocast enabled. "
            "Minor numerical nondeterminism vs fp32.",
            flush=True,
        )
    print("=" * 72, flush=True)

    # ── Conformer-generation-only mode (standalone CPU step §B4) ─────────────
    if getattr(args, "generate_conformers_only", False):
        return _run_conformer_generation(args, conformer_cache_dir)

    # ── Write provenance stub before run (exists even if run fails) ──────────
    write_provenance(seed_output_dir, args.dataset, args.seed, resolved_device, args)

    # ── Stage 1: main benchmark runner ───────────────────────────────────────
    runner_argv = build_runner_argv(args, resolved_device, seed_output_dir)
    print(f"Invoking benchmark runner: {' '.join(runner_argv[:8])} ...", flush=True)
    result = subprocess.run(runner_argv)
    if result.returncode != 0:
        print(f"Benchmark runner exited with code {result.returncode}", flush=True)
        return result.returncode

    # ── Stage 2: Uni-Mol2 V2 workflow (optional §B) ──────────────────────────
    conformer_cache_hash = "not_generated"
    if getattr(args, "run_unimol_v2", False):
        conformer_cache_hash = _run_unimolv2_workflow(
            args, resolved_device, seed_output_dir, conformer_cache_dir
        )

    # ── Final provenance with updated cache hash ──────────────────────────────
    write_provenance(
        seed_output_dir, args.dataset, args.seed, resolved_device, args,
        conformer_cache_hash=conformer_cache_hash,
    )

    return 0


def _run_conformer_generation(args: argparse.Namespace, cache_dir: Path) -> int:
    """Standalone conformer-generation step — runs on CPU, no GPU needed."""
    try:
        from autoqsar.conformers import ConformerCache, generate_dataset_conformers
    except ImportError as exc:
        print(f"ERROR: autoqsar.conformers not importable: {exc}", flush=True)
        return 1

    cache_dir.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.input_dir) if args.input_dir else Path(__file__).resolve().parent
    data_dir = input_root / "data"

    print(f"Conformer generation: dataset={args.dataset}  conformer_seed={args.conformer_seed}", flush=True)
    print(f"  cache_dir: {cache_dir}", flush=True)
    cache = ConformerCache(cache_dir)
    n_generated, n_cached = generate_dataset_conformers(
        dataset_name=args.dataset,
        data_dir=data_dir,
        cache=cache,
        conformer_seed=args.conformer_seed,
    )
    print(f"Conformers: {n_generated} generated, {n_cached} cache hits. Hash: {cache.cache_hash()}", flush=True)
    return 0


def _run_unimolv2_workflow(
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
    conformer_cache_dir: Path,
) -> str:
    """Run the Uni-Mol2 V2 workflow and return the conformer cache hash."""
    try:
        from autoqsar.unimolv2 import UnimolV2Workflow
        from autoqsar.conformers import ConformerCache
    except ImportError as exc:
        print(f"WARNING: Uni-Mol2 workflow skipped — import failed: {exc}", flush=True)
        return "import_failed"

    unimol_size = getattr(args, "unimol_size", "84m")
    batch_size = getattr(args, "unimol_v2_batch_size", 32)
    conformer_seed = getattr(args, "conformer_seed", 42)

    # Warn about VRAM for large sizes
    vram_warnings = {"164m": "~8 GB", "310m": "~14 GB", "1.1b": "~32 GB"}
    if unimol_size in vram_warnings:
        print(
            f"WARNING: Uni-Mol2 {unimol_size} requires approximately "
            f"{vram_warnings[unimol_size]} VRAM. Ensure your GPU can accommodate this.",
            flush=True,
        )

    conformer_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ConformerCache(conformer_cache_dir)

    workflow = UnimolV2Workflow(
        model_size=unimol_size,
        device=device,
        precision=args.precision,
        batch_size=batch_size,
        seed=args.seed,
        conformer_seed=conformer_seed,
        conformer_cache=cache,
        output_dir=output_dir,
    )

    input_root = Path(args.input_dir) if args.input_dir else Path(__file__).resolve().parent
    data_dir = input_root / "data"

    try:
        workflow.run(dataset_name=args.dataset, data_dir=data_dir)
    except Exception as exc:
        print(f"WARNING: Uni-Mol2 workflow failed: {exc}", flush=True)

    return cache.cache_hash()


if __name__ == "__main__":
    sys.exit(main())
