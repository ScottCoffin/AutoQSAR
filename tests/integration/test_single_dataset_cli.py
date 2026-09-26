"""F2: one command on one CSV yields models, metrics, predictions and both reports."""

from __future__ import annotations

import json

import pandas as pd

from tests.conftest import command_prefix


def test_single_dataset_command_uses_public_console_script():
    prefix = command_prefix("qsarena-benchmark")
    assert prefix  # installed console script in CI, python -m fallback in a bare checkout


def test_qsarena_benchmark_single_csv_writes_model_metrics_predictions_and_reports(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "solubility"
    result = run_cli(
        ["qsarena-benchmark", "--dataset", examples / "solubility.csv", "--target-col", "logS", "--id-col", "compound_id",
         "--benchmark-profile", "quick", "--output-dir", out, "--n-jobs", "2"],
        cwd=tmp_path,
    )
    assert "Reports written: report.html and report.md" in result.stdout
    dataset_dir = out / "solubility"
    metrics = pd.read_csv(dataset_dir / "metrics.csv")
    ok = metrics[metrics["error"].fillna("").astype(str).str.strip() == ""] if "error" in metrics else metrics
    assert len(ok) >= 8  # the quick profile's scikit-learn models plus the ensembles
    assert ok["test_rmse"].notna().any()
    predictions = pd.read_csv(dataset_dir / "predictions.csv")
    assert {"model", "split", "smiles", "id", "observed", "predicted"}.issubset(predictions.columns)
    assert predictions["id"].str.startswith("SOL-").all()
    for name in ("run_config.json", "run_config.yaml", "environment_manifest.json", "report.html", "report.md",
                 "report_data.json", "run.log", "events.jsonl", "preflight.json", "dataset_summary.csv"):
        assert (out / name).exists(), name
    status = json.loads((dataset_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["cleanup_counts"]["unparseable_smiles"] == 1
    assert status["task_type"] == "regression"
    assert status["applicability_domain"]["method"] == "both"
    assert (dataset_dir / "applicability_domain.csv").exists()
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert config["run_config"]["input"]["target_col"] == "logS"
    assert config["run_config"]["models"]["profile"] == "quick"
    assert len(config["run_config_signature"]) == 64


def test_classification_with_threshold_and_primary_metric(run_cli, examples, tmp_path):
    from tests.conftest import FAST_MODELS

    out = tmp_path / "runs" / "logd_class"
    run_cli(
        ["qsarena-benchmark", "--dataset", examples / "solubility.csv", "--target-col", "logD",
         "--classification-threshold", "2.0", "--primary-metric", "auprc", "--output-dir", out, *FAST_MODELS],
        cwd=tmp_path,
    )
    status = json.loads((out / "solubility" / "run_status.json").read_text(encoding="utf-8"))
    assert status["task_type"] == "classification"
    assert status["primary_metric"] == "auprc"
    metrics = pd.read_csv(out / "solubility" / "metrics.csv")
    assert set(metrics["primary_metric"].dropna()) == {"auprc"}
