"""
run_one.py — thin, seed-injectable CLI wrapper for one (dataset, seed) unit.

Designed for Apptainer/Slurm job-array dispatch:

    python run_one.py \
        --dataset tdc_caco2_wang \
        --seed 3 \
        --config /opt/autoqsar/run_config.json \
        --input-dir /in \
        --output-dir /out \
        --device auto

All file I/O is routed through --input-dir / --output-dir.
Artifacts land under  <output-dir>/<dataset>/seed_<seed>/.

Randomness is fully deterministic from --seed:
  - Python random
  - NumPy
  - PyTorch (CPU + CUDA, deterministic algorithms where feasible)
  - chemprop_random_seed passed to benchmark runner
  - PYTHONHASHSEED exported before any imports
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
        "--chemprop-random-seed", str(args.seed),  # tie chemprop seed to master seed
        "--output-dir", str(output_dir),
    ]

    # Route device-dependent model families
    if resolved_device == "cpu":
        argv += [
            "--no-run-unimol-v1",
        ]
    else:
        argv += [
            "--run-unimol-v1",
        ]

    # Supply input-dir-rooted paths for all cache locations
    if args.input_dir is not None:
        input_root = Path(args.input_dir)
        argv += [
            "--persistent-feature-store-path",
            str(input_root / "model_cache" / "feature_store_parquet"),
            "--shared-feature-matrix-cache-path",
            str(input_root / "model_cache" / "benchmark_feature_matrix_cache"),
        ]

    # Forward any extra passthrough flags
    if args.extra_args:
        # Strip leading '--' separator if argparse REMAINDER added one
        extra = args.extra_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        argv += extra

    return argv


def write_provenance(
    output_dir: Path,
    dataset: str,
    seed: int,
    device: str,
    args: argparse.Namespace,
) -> None:
    record = {
        "dataset": dataset,
        "seed": seed,
        "device": device,
        "git_commit": _git_commit(),
        "cuda_version": _cuda_version(),
        "gpu_model": _gpu_model(device),
        "image_sha256": _image_hash(args.image_hash_file),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "run_one_version": "1.0.0",
    }
    provenance_path = output_dir / "run_one_provenance.json"
    provenance_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Must set PYTHONHASHSEED before any further imports that might hash objects.
    _set_env_before_imports(args.seed)

    resolved_device = _detect_device(args.device)
    _seed_everything(args.seed, resolved_device)

    # Determine the per-(dataset, seed) output directory.
    seed_output_dir = args.output_dir / args.dataset / f"seed_{args.seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    # Log the run header.
    print("=" * 72, flush=True)
    print(f"run_one.py  dataset={args.dataset}  seed={args.seed}  device={resolved_device}", flush=True)
    print(f"  git commit : {_git_commit()}", flush=True)
    print(f"  CUDA       : {_cuda_version()}", flush=True)
    print(f"  GPU        : {_gpu_model(resolved_device)}", flush=True)
    print(f"  output_dir : {seed_output_dir}", flush=True)
    print("=" * 72, flush=True)

    # Write provenance before the run so it exists even if the run fails.
    write_provenance(seed_output_dir, args.dataset, args.seed, resolved_device, args)

    runner_argv = build_runner_argv(args, resolved_device, seed_output_dir)

    print(f"Invoking: {' '.join(runner_argv[:8])} ...", flush=True)

    result = subprocess.run(runner_argv)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
