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
done = t["completed_dataset_count"]
total = t["dataset_count"]
elapsed_h = t["elapsed_seconds"] / 3600
eta_h = (t.get("eta_seconds") or 0) / 3600
avg_min = (t.get("average_finished_dataset_seconds") or 0) / 60

print(f"=== {datetime.datetime.now().strftime('%H:%M:%S')}  |  {b.name} ===")
print(f"Done: {done} / {total}   elapsed: {elapsed_h:.1f}h   avg/dataset: {avg_min:.0f}m   ETA: {eta_h:.1f}h (updates as data accumulates)")
print()

running_current, running_stale, completed = [], [], []
for ds_dir in sorted(b.iterdir()):
    rs = ds_dir / "run_status.json"
    if not rs.exists():
        continue
    s = json.loads(rs.read_text())
    status = s.get("status", "")
    if status == "completed":
        elap = s.get("elapsed_seconds", 0)
        completed.append(f"  ✓  {ds_dir.name:<45}  {elap:.0f}s")
    elif status == "running":
        stage = s.get("checkpoint_stage", "(starting)")
        # Distinguish stale entries from the previous killed run vs. the current run
        # run_status.json "started_at" is local-time string; compare epoch via mtime
        try:
            mtime = rs.stat().st_mtime
            is_stale = mtime < run_start_epoch
        except OSError:
            is_stale = False
        entry = f"  ⟳  {ds_dir.name:<45}  @ {stage}"
        if is_stale:
            running_stale.append(entry + "  [stale from prior run]")
        else:
            running_current.append(entry)

if running_current:
    print("Currently running:")
    print("\n".join(running_current))
    print()
if running_stale:
    print("Stale (prior run, will be picked up):")
    print("\n".join(running_stale))
    print()
if completed:
    print("Completed this session:")
    print("\n".join(completed))
EOF
