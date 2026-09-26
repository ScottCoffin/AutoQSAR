"""F4: resume at dataset and stage granularity, config-signature aware and interrupt safe."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from tests.conftest import FAST_MODELS, cli_env, command_prefix

#: Includes the (comparatively slow) tabular MLP so a run lasts long enough to be interrupted.
KILLABLE_MODELS = ["--benchmark-profile", "quick", "--selector-method", "rf_importance", "--n-jobs", "1", "--use-gpu", "false",
                   "--only-model-names", "ElasticNetCV", "--only-model-names", "Random forest",
                   "--only-model-names", "Tabular MLP", "--only-model-names", "Extra trees",
                   "--only-model-names", "Ensemble"]


def _stage_labels(dataset_dir: Path) -> list[str]:
    return pd.read_csv(dataset_dir / "step_runtime.csv")["stage_label"].astype(str).tolist()


def _metrics(dataset_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(dataset_dir / "metrics.csv")
    return frame.set_index("model")[["test_rmse", "test_mae", "test_r2"]].sort_index()


def _sol(examples: Path, out: Path) -> list:
    return ["qsarena-benchmark", "--dataset", examples / "solubility.csv", "--target-col", "logS", "--output-dir", out]


def test_resume_skips_completed_dataset(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "r"
    run_cli([*_sol(examples, out), *FAST_MODELS], cwd=tmp_path)
    second = run_cli([*_sol(examples, out), *FAST_MODELS], cwd=tmp_path)
    assert "solubility: already completed in" in second.stdout
    assert "reusing saved outputs" in second.stdout
    assert "status=resumed" in second.stdout


def test_config_change_invalidates_only_affected_stages(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "r"
    run_cli([*_sol(examples, out), *FAST_MODELS], cwd=tmp_path)
    before = _metrics(out / "solubility")
    changed = run_cli([*_sol(examples, out), *FAST_MODELS, "--ensemble-max-train-correlation", "0.99"], cwd=tmp_path)
    assert "configuration or input changed since this dataset completed" in changed.stdout
    match = re.search(r"config signature changed for (\d+) cached model stage\(s\) \((.*)\); recomputing", changed.stdout)
    assert match, changed.stdout[-3000:]
    # exactly the two ensemble methods are recomputed; no base model is
    assert int(match.group(1)) == 2
    assert match.group(2).startswith("Ensemble (OOF Stacking") and ", Ensemble (Weighted average" in match.group(2)
    assert not any(model in match.group(2) for model in ("ElasticNetCV", "SVR", "Random forest"))
    assert "stage 2/3 cache hit (signature match)" in changed.stdout
    labels = _stage_labels(out / "solubility")
    for model in ("ElasticNetCV", "SVR", "Random forest"):
        assert f"conventional model {model} (cached)" in labels
    assert "ensemble (2 method(s))" in labels  # recomputed
    after = _metrics(out / "solubility")
    pd.testing.assert_frame_equal(before.drop(index=[i for i in before.index if i.startswith("Ensemble")]),
                                  after.drop(index=[i for i in after.index if i.startswith("Ensemble")]))
    # a split change invalidates everything, including the stage 2/3 cache
    split = run_cli([*_sol(examples, out), *FAST_MODELS, "--ensemble-max-train-correlation", "0.99", "--test-fraction", "0.3"], cwd=tmp_path)
    assert "stage 2/3 cache hit" not in split.stdout
    assert not any(label.endswith("(cached)") for label in _stage_labels(out / "solubility"))


def test_fresh_ignores_existing_artifacts(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "r"
    run_cli([*_sol(examples, out), *FAST_MODELS], cwd=tmp_path)
    fresh = run_cli([*_sol(examples, out), *FAST_MODELS, "--fresh"], cwd=tmp_path)
    assert "--fresh: moved the previous contents" in fresh.stdout
    assert "already completed" not in fresh.stdout
    superseded = [p for p in out.parent.iterdir() if p.name.startswith("r_superseded_")]
    assert len(superseded) == 1 and (superseded[0] / "solubility" / "metrics.csv").exists()
    assert not any(label.endswith("(cached)") for label in _stage_labels(out / "solubility"))


def _start(argv: list, cwd: Path, home: Path) -> subprocess.Popen:
    command = command_prefix(argv[0]) + [str(a) for a in argv[1:]]
    return subprocess.Popen(command, cwd=str(cwd), env=cli_env(home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for(predicate, process: subprocess.Popen, timeout: float = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        if process.poll() is not None:
            return predicate()
        time.sleep(0.1)
    return False


def _running_stage(dataset_dir: Path) -> str:
    try:
        rows = json.loads((dataset_dir / "step_runtime.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    running = [r for r in rows if r.get("status") == "running"]
    return str(running[-1].get("stage_label", "")) if running else ""


def test_kill_mid_stage_and_resume_gives_identical_results_without_rework(run_cli, examples, tmp_path, qsarena_home):
    reference = tmp_path / "runs" / "reference"
    run_cli([*_sol(examples, reference), *KILLABLE_MODELS], cwd=tmp_path)
    out = tmp_path / "runs" / "killed"
    process = _start([*_sol(examples, out), *KILLABLE_MODELS], tmp_path, qsarena_home)
    try:
        reached = _wait_for(lambda: "Tabular MLP" in _running_stage(out / "solubility"), process)
    finally:
        process.kill()
        process.wait(timeout=60)
    assert reached, "the run finished before it could be interrupted"
    status = json.loads((out / "solubility" / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"  # interrupted mid-dataset, files intact (atomic writes)
    finished_before_kill = set(pd.read_csv(out / "solubility" / "metrics.csv")["model"])
    assert {"ElasticNetCV", "Random forest"} <= finished_before_kill
    resumed = run_cli([*_sol(examples, out), *KILLABLE_MODELS], cwd=tmp_path)
    assert "restored" in resumed.stdout and "stage 2/3 cache hit" in resumed.stdout
    labels = _stage_labels(out / "solubility")
    for model in finished_before_kill:
        assert f"conventional model {model} (cached)" in labels  # no rework
    assert "conventional model Tabular MLP" in labels  # the interrupted stage is redone
    pd.testing.assert_frame_equal(_metrics(out / "solubility"), _metrics(reference / "solubility"), check_exact=False, rtol=1e-9)


def test_kill_mid_batch_and_resume(run_cli, examples, tmp_path, qsarena_home):
    out = tmp_path / "runs" / "batch"
    argv = ["qsarena-benchmark", "--batch", examples / "batch_dir", "--output-dir", out, *KILLABLE_MODELS]
    process = _start(argv, tmp_path, qsarena_home)

    def first_done() -> bool:
        path = out / "lipophilicity" / "run_status.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
        except (OSError, ValueError):
            return False

    try:
        reached = _wait_for(first_done, process)
    finally:
        process.kill()
        process.wait(timeout=60)
    assert reached
    resumed = run_cli(argv, cwd=tmp_path)
    assert "lipophilicity: already completed in" in resumed.stdout
    summary = pd.read_csv(out / "dataset_summary.csv")
    assert set(summary["status"]) == {"completed"}
