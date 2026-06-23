#!/usr/bin/env bash
# js2/watch_benchmark.sh — live benchmark status watcher
#
# Usage:
#   bash js2/watch_benchmark.sh [output_dir]
#
# If output_dir is omitted, uses the most recent autoqsar_benchmark_* directory.
#
# Run as a one-shot check, or wrap in watch:
#   watch -n 10 bash js2/watch_benchmark.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="$REPO_ROOT/benchmark_results"

if [[ -n "$1" ]]; then
    RUNDIR="$1"
else
    RUNDIR="$(ls -dt "$RESULTS_ROOT"/autoqsar_benchmark_* 2>/dev/null | head -1)"
fi

if [[ -z "$RUNDIR" || ! -d "$RUNDIR" ]]; then
    echo "No benchmark run directory found under $RESULTS_ROOT"
    exit 1
fi

python3 - "$RUNDIR" <<'EOF'
import json, sys, pathlib, datetime

b = pathlib.Path(sys.argv[1])
timing_file = b / "run_timing.json"
if not timing_file.exists():
    print(f"No run_timing.json in {b}")
    sys.exit(1)

t = json.loads(timing_file.read_text())
run_start_epoch = t["started_epoch"]
total = t["dataset_count"]
elapsed_h = t["elapsed_seconds"] / 3600
eta_h = (t.get("eta_seconds") or 0) / 3600
avg_min = (t.get("average_finished_dataset_seconds") or 0) / 60

# Count completed directly from run_status.json files — run_timing.json's
# completed_dataset_count resets each time the benchmark process restarts,
# so it under-counts when there have been multiple resume sessions.
running_current, running_stale, completed_current, completed_prior = [], [], [], []
for ds_dir in sorted(b.iterdir()):
    rs = ds_dir / "run_status.json"
    if not rs.exists():
        continue
    s = json.loads(rs.read_text())
    status = s.get("status", "")
    try:
        mtime = rs.stat().st_mtime
    except OSError:
        mtime = run_start_epoch
    if status == "completed":
        elap = s.get("elapsed_seconds", 0)
        entry = f"  ✓  {ds_dir.name:<45}  {elap:.0f}s"
        if mtime >= run_start_epoch:
            completed_current.append(entry)
        else:
            completed_prior.append(entry)
    elif status == "running":
        stage = s.get("checkpoint_stage", "(starting)")
        entry = f"  ⟳  {ds_dir.name:<45}  @ {stage}"
        if mtime < run_start_epoch:
            running_stale.append(entry + "  [stale from prior run]")
        else:
            running_current.append(entry)

done = len(completed_current) + len(completed_prior)

print(f"=== {datetime.datetime.now().strftime('%H:%M:%S')}  |  {b.name} ===")
print(f"Done: {done} / {total}   elapsed: {elapsed_h:.1f}h   avg/dataset: {avg_min:.0f}m   ETA: {eta_h:.1f}h (updates as data accumulates)")
print()

if running_current:
    print("Currently running:")
    print("\n".join(running_current))
    print()
if running_stale:
    print("Stale (prior run, will be picked up):")
    print("\n".join(running_stale))
    print()
if completed_current:
    print("Completed this session:")
    print("\n".join(completed_current))
    print()
if completed_prior:
    print("Completed in prior sessions:")
    print("\n".join(completed_prior))
EOF
