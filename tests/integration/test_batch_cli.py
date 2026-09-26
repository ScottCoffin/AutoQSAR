"""F3: batch mode over a manifest or a directory; failures are isolated and summarized."""

from __future__ import annotations

import json

import pandas as pd

from tests.conftest import FAST_MODELS


def test_batch_manifest_three_datasets_one_malformed_records_failure(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "batch"
    result = run_cli(["qsarena-benchmark", "--batch", examples / "batch_manifest.csv", "--output-dir", out, *FAST_MODELS], cwd=tmp_path)
    assert result.returncode == 0  # the batch finishes even though one dataset failed
    summary = pd.read_csv(out / "dataset_summary.csv").set_index("dataset")
    assert summary.loc["solubility", "status"] == "completed"
    assert summary.loc["bbb", "status"] == "completed"
    assert summary.loc["broken", "status"] == "failed"
    assert "could not infer the SMILES column" in summary.loc["broken", "error"]
    assert "--smiles-col" in summary.loc["broken", "remedy"]
    # each dataset isolated in its own directory
    for name in ("solubility", "bbb", "broken"):
        assert (out / name / "run_status.json").exists()
    assert "2 completed, 1 failed" in result.stdout
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "| broken | failed |" in report


def test_batch_per_dataset_overrides_columns_and_split(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "batch"
    run_cli(["qsarena-benchmark", "--batch", examples / "batch_manifest.csv", "--output-dir", out, *FAST_MODELS], cwd=tmp_path)
    sol = pd.read_csv(out / "solubility" / "metrics.csv")
    assert set(sol["target_column"]) == {"logS"}
    assert set(sol["split_strategy"]) == {"random"}
    assert int(sol["n_test"].iloc[0]) == 12  # test_fraction 0.25 of 46 rows (manifest override)
    bbb = pd.read_csv(out / "bbb" / "metrics.csv")
    assert int(bbb["n_test"].iloc[0]) == 9  # run-level default 0.2 of 44 rows
    status = json.loads((out / "bbb" / "run_status.json").read_text(encoding="utf-8"))
    assert status["task_type"] == "classification"
    predictions = pd.read_csv(out / "bbb" / "predictions.csv")
    assert predictions["id"].str.startswith("BBB-").all()
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    overrides = {d["name"]: d["overrides"] for d in config["datasets"]}
    assert overrides["solubility"]["test_fraction"] == 0.25


def test_batch_directory_runs_all_csvs(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "dir"
    result = run_cli(["qsarena-benchmark", "--batch", examples / "batch_dir", "--output-dir", out, *FAST_MODELS], cwd=tmp_path)
    summary = pd.read_csv(out / "dataset_summary.csv")
    assert sorted(summary["dataset"]) == ["lipophilicity", "permeability"]
    assert set(summary["status"]) == {"completed"}
    assert "2 completed" in result.stdout


def test_no_continue_on_error_stops_the_batch(run_cli, examples, tmp_path):
    bad = examples / "all_invalid.csv"
    bad.write_text("smiles,target\n" + "\n".join(f"nonsense{i},{i}" for i in range(30)) + "\n", encoding="utf-8")
    listing = examples / "list.txt"
    listing.write_text("all_invalid.csv\nbatch_dir/permeability.csv\n", encoding="utf-8")
    out = tmp_path / "runs" / "strict"
    result = run_cli(["qsarena-benchmark", "--batch", listing, "--output-dir", out, "--no-drop-unparseable",
                      "--no-continue-on-error", *FAST_MODELS], cwd=tmp_path, check=False)
    assert result.returncode != 0
    status = json.loads((out / "all_invalid" / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and "could not be parsed by RDKit" in status["error"]
    assert not (out / "permeability" / "metrics.csv").exists()
