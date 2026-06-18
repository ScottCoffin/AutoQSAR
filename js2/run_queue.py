#!/usr/bin/env python3
"""
js2/run_queue.py — AutoQSAR Jetstream2 task orchestrator.

Reads a task manifest TSV (dataset, seed, device, notes), runs each task via
    apptainer run --nv autoqsar.sif ...
and checkpoints progress to a JSON state file so the queue can be resumed after
a cloud VM shelve/unshelve cycle.

Usage:
    python js2/run_queue.py \\
        --manifest js2/manifests/gpu_tasks.tsv \\
        --sif ~/autoqsar/autoqsar.sif \\
        --input-dir /vol/autoqsar_in \\
        --output-dir /vol/autoqsar_out \\
        --state js2/queue_state.json \\
        --su-budget 200 \\
        [--dry-run]

SU billing:
    g3.large  (GPU, 10 vCPU, 60 GB RAM, 50% A100) = 32 SU/hr
    m3.medium (CPU, 8 vCPU,  30 GB RAM)            =  8 SU/hr
    The --instance-flavor flag selects the rate; default: g3.large.

Shelving:
    When SU burn exceeds 80% of --su-budget a warning is printed.
    When 100% is reached the script calls
        openstack server shelve <INSTANCE_ID>
    and exits.  The operator must unshelve the VM and re-run this script
    to resume; the state file tracks what was already completed.

Resume logic:
    Tasks with state "done" or "error" are skipped on resume.
    Tasks with state "running" are re-queued (the previous container
    was interrupted).

Requirements on the Jetstream2 VM:
    - apptainer (sudo snap install apptainer --classic)
    - openstack CLI (pip install python-openstackclient)
    - INSTANCE_ID in environment or passed via --instance-id
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# SU/hour rates by Jetstream2 flavor
_SU_RATES: dict[str, float] = {
    "g3.large":  32.0,   # 50% A100 vGPU slice
    "m3.medium":  8.0,   # CPU-only
    "m3.large":  16.0,
    "m3.xlarge": 32.0,
}
_WARN_THRESHOLD = 0.80   # 80% of su_budget triggers a warning


# ── State management ──────────────────────────────────────────────────────────

class QueueState:
    """JSON checkpoint tracking task outcomes and elapsed wall-time SUs."""

    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._data: dict = {
            "schema_version": 1,
            "created_utc": _now_iso(),
            "last_updated_utc": _now_iso(),
            "elapsed_seconds": 0.0,
            "tasks": {},
        }
        if state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                self._data.update(loaded)
                logger.info("Resumed from state file: %s", state_path)
            except Exception as exc:
                logger.warning("Could not load state file (%s); starting fresh", exc)

    def task_key(self, row: dict) -> str:
        return f"{row['dataset']}__seed{row['seed']}"

    def is_done(self, row: dict) -> bool:
        return self._data["tasks"].get(self.task_key(row), {}).get("status") == "done"

    def mark_running(self, row: dict) -> None:
        self._data["tasks"][self.task_key(row)] = {
            "status": "running",
            "started_utc": _now_iso(),
            "dataset": row["dataset"],
            "seed": row["seed"],
            "device": row.get("device", ""),
        }
        self._save()

    def mark_done(self, row: dict, elapsed: float, returncode: int) -> None:
        key = self.task_key(row)
        self._data["tasks"][key]["status"] = "done" if returncode == 0 else "error"
        self._data["tasks"][key]["finished_utc"] = _now_iso()
        self._data["tasks"][key]["elapsed_seconds"] = round(elapsed, 1)
        self._data["tasks"][key]["returncode"] = returncode
        self._save()

    def add_elapsed(self, seconds: float) -> None:
        self._data["elapsed_seconds"] += seconds
        self._data["last_updated_utc"] = _now_iso()
        self._save()

    def elapsed_hours(self) -> float:
        return self._data["elapsed_seconds"] / 3600.0

    def task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"done": 0, "error": 0, "pending": 0, "running": 0}
        for v in self._data["tasks"].values():
            s = v.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1
        return counts

    def _save(self) -> None:
        self._data["last_updated_utc"] = _now_iso()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self._path)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_queue(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    state = QueueState(Path(args.state))
    su_rate = _SU_RATES.get(args.instance_flavor, 32.0)

    pending = [row for row in manifest if not state.is_done(row)]
    logger.info(
        "Manifest: %d total tasks, %d already done, %d to run.",
        len(manifest),
        len(manifest) - len(pending),
        len(pending),
    )

    if not pending:
        logger.info("All tasks complete. Nothing to do.")
        return 0

    queue_start = time.monotonic()

    for i, row in enumerate(pending):
        # SU guard
        su_used = state.elapsed_hours() * su_rate
        if args.su_budget and su_used >= args.su_budget:
            logger.error(
                "SU budget exhausted (used %.1f / %.1f SU). Shelving instance.",
                su_used, args.su_budget,
            )
            _shelve_instance(args)
            return 1

        if args.su_budget and su_used >= args.su_budget * _WARN_THRESHOLD:
            logger.warning(
                "SU usage at %.0f%% of budget (%.1f / %.1f SU).",
                100.0 * su_used / args.su_budget,
                su_used, args.su_budget,
            )

        # Build apptainer command
        cmd = _build_apptainer_cmd(row, args)
        logger.info(
            "[%d/%d] Running %s seed=%s (device=%s)%s",
            i + 1, len(pending),
            row["dataset"], row["seed"], row.get("device", "?"),
            "  [DRY RUN]" if args.dry_run else "",
        )
        if args.dry_run:
            logger.info("  CMD: %s", " ".join(cmd))
            continue

        state.mark_running(row)
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                cmd,
                check=False,
                text=True,
            )
            elapsed = time.monotonic() - t0
            state.mark_done(row, elapsed, proc.returncode)
            state.add_elapsed(elapsed)

            if proc.returncode != 0:
                logger.error(
                    "Task %s seed=%s FAILED (returncode=%d, elapsed=%.0fs)",
                    row["dataset"], row["seed"], proc.returncode, elapsed,
                )
            else:
                su_this_task = (elapsed / 3600.0) * su_rate
                logger.info(
                    "Task %s seed=%s done in %.0fs (%.2f SU)",
                    row["dataset"], row["seed"], elapsed, su_this_task,
                )
        except KeyboardInterrupt:
            elapsed = time.monotonic() - t0
            state.mark_done(row, elapsed, returncode=-1)
            state.add_elapsed(elapsed)
            logger.warning("Interrupted. State saved. Re-run to resume.")
            return 1

    total_wall = time.monotonic() - queue_start
    total_su = state.elapsed_hours() * su_rate
    counts = state.task_counts()
    logger.info(
        "Queue complete. Wall time: %.1f min. Total SU used (this session): %.1f. "
        "Tasks: done=%d, error=%d.",
        total_wall / 60.0, total_su,
        counts.get("done", 0), counts.get("error", 0),
    )
    return 0 if counts.get("error", 0) == 0 else 1


def _build_apptainer_cmd(row: dict, args: argparse.Namespace) -> list[str]:
    device = row.get("device", "gpu").lower()
    use_gpu = device == "gpu"

    cmd = ["apptainer", "run"]
    if use_gpu:
        cmd.append("--nv")
    cmd += [
        "--bind", f"{args.input_dir}:/in",
        "--bind", f"{args.output_dir}:/out",
        args.sif,
        "--dataset", row["dataset"],
        "--seed", str(row["seed"]),
        "--input-dir", "/in",
        "--output-dir", "/out",
        "--device", "auto" if use_gpu else "cpu",
    ]

    if args.precision:
        cmd += ["--precision", args.precision]
    if args.run_unimol_v2:
        cmd.append("--run-unimol-v2")
        cmd += ["--unimol-size", args.unimol_size]
    if args.conformer_cache:
        cmd += ["--conformer-cache", "/in/conformer_cache"]

    return cmd


def _load_manifest(path: str) -> list[dict]:
    tasks = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("dataset", "").startswith("#"):
                continue
            tasks.append(row)
    return tasks


def _shelve_instance(args: argparse.Namespace) -> None:
    instance_id = args.instance_id or os.environ.get("INSTANCE_ID")
    if not instance_id:
        logger.warning(
            "No instance ID available for auto-shelve. "
            "Set --instance-id or INSTANCE_ID env var. Exiting without shelving."
        )
        return
    logger.info("Shelving Jetstream2 instance %s ...", instance_id)
    try:
        subprocess.run(
            ["openstack", "server", "shelve", instance_id],
            check=True, timeout=60,
        )
        logger.info("Instance shelved. Unshelve and re-run to resume.")
    except Exception as exc:
        logger.error("Shelve failed: %s. Please shelve manually.", exc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoQSAR Jetstream2 resumable task orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True,
                   help="Path to task manifest TSV (dataset, seed, device, notes).")
    p.add_argument("--sif", required=True,
                   help="Path to autoqsar.sif Apptainer image.")
    p.add_argument("--input-dir", required=True,
                   help="Host path bind-mounted as /in inside the container.")
    p.add_argument("--output-dir", required=True,
                   help="Host path bind-mounted as /out inside the container.")
    p.add_argument("--state", default="js2/queue_state.json",
                   help="JSON checkpoint file for resume support.")
    p.add_argument("--instance-id", default=None,
                   help="Jetstream2 OpenStack instance ID for auto-shelve.")
    p.add_argument("--instance-flavor", default="g3.large",
                   choices=list(_SU_RATES),
                   help="Instance flavor (determines SU/hr rate).")
    p.add_argument("--su-budget", type=float, default=None,
                   help="Maximum SU budget; triggers auto-shelve at 100%%, warning at 80%%.")
    p.add_argument("--precision", choices=["tf32_bf16", "fp32"], default=None,
                   help="Precision mode forwarded to run_one.py --precision.")
    p.add_argument("--run-unimol-v2", action="store_true", default=False,
                   help="Enable Uni-Mol2 V2 workflow.")
    p.add_argument("--unimol-size", default="84m",
                   choices=["84m", "164m", "310m", "1.1b"],
                   help="Uni-Mol2 model size.")
    p.add_argument("--conformer-cache", action="store_true", default=False,
                   help="Pass --conformer-cache /in/conformer_cache to each task.")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print commands without executing. Safe for CI/testing.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_queue(args)


if __name__ == "__main__":
    sys.exit(main())
