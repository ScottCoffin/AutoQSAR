"""qsarena.reporting: best-model selection, remedies, and the HTML + Markdown reports."""

from __future__ import annotations

import json
import re

import pandas as pd

from qsarena import reporting


def _metrics(rows):
    return pd.DataFrame(rows)


def test_best_model_by_test_and_by_cv_regression():
    metrics = _metrics([
        {"model": "A", "primary_metric": "rmse", "test_rmse": 0.50, "cv_rmse": 0.70, "error": ""},
        {"model": "B", "primary_metric": "rmse", "test_rmse": 0.60, "cv_rmse": 0.40, "error": ""},
        {"model": "C", "primary_metric": "rmse", "test_rmse": 0.10, "cv_rmse": None, "error": "boom"},
        {"model": "Ensemble (x)", "primary_metric": "rmse", "test_rmse": 0.45, "cv_rmse": None, "error": ""},
    ])
    best = reporting.best_model_rows(metrics)
    assert best["metric"] == "rmse"
    assert best["test"]["model"] == "Ensemble (x)"  # failed rows never win
    assert best["cv"]["model"] == "B" and best["cv"]["test_value"] == 0.60


def test_best_model_higher_is_better_for_classification():
    metrics = _metrics([
        {"model": "A", "primary_metric": "roc_auc", "test_roc_auc": 0.7, "cv_roc_auc": 0.9, "error": ""},
        {"model": "B", "primary_metric": "roc_auc", "test_roc_auc": 0.8, "cv_roc_auc": 0.6, "error": ""},
    ])
    best = reporting.best_model_rows(metrics)
    assert best["test"]["model"] == "B" and best["cv"]["model"] == "A"


def test_remedies_map_backend_errors():
    assert "qsarena[graph]" in reporting.remedy_for_error("No module named 'chemprop'")
    assert "DGL wheel index" in reporting.remedy_for_error("OSError: graphbolt DLL")
    assert "run.log" in reporting.remedy_for_error("something unexpected")


def _run_dir(tmp_path):
    output = tmp_path / "run"
    ds = output / "demo"
    ds.mkdir(parents=True)
    pd.DataFrame([
        {"model": "ElasticNetCV", "workflow": "conventional", "primary_metric": "rmse", "test_rmse": 0.5, "test_mae": 0.4,
         "test_r2": 0.8, "test_spearman": 0.9, "cv_rmse": 0.6, "stage_duration_seconds": 1.0, "error": ""},
        {"model": "Chemprop v2 (D-MPNN, ensemble=1)", "workflow": "Chemprop v2", "primary_metric": "rmse",
         "error": "No module named 'chemprop'", "stage_duration_seconds": 0.1},
        {"model": "Ensemble (OOF Stacking (RidgeCV, 5-fold))", "workflow": "ensemble", "primary_metric": "rmse",
         "test_rmse": 0.55, "stage_duration_seconds": 0.2, "error": ""},
    ]).to_csv(ds / "metrics.csv", index=False)
    (ds / "run_status.json").write_text(json.dumps({
        "status": "completed", "task_type": "regression", "n_rows": 40, "primary_metric": "rmse",
        "applicability_domain": {"method": "both", "n_test": 8, "in_domain_fraction": 0.75},
    }), encoding="utf-8")
    bad = output / "broken"
    bad.mkdir()
    (bad / "run_status.json").write_text(json.dumps({"status": "failed", "error": "no SMILES column", "remedy": "use --smiles-col"}), encoding="utf-8")
    return output


def test_report_writes_html_md_and_contains_config_summary(tmp_path):
    output = _run_dir(tmp_path)
    data = reporting.collect_run_report_data(
        output, dataset_ids=["demo", "broken"], run_info={"mode": "batch", "profile": "quick", "config_signature": "abc123"},
        config={"selection": {"protocol": "both"}}, config_yaml="split:\n  seed: 13\n",
        warnings=[{"code": "small_dataset", "message": "tiny", "remedy": "more data", "dataset": "demo"}],
    )
    html_path, md_path = reporting.write_run_reports(output, data)
    html = html_path.read_text(encoding="utf-8")
    md = md_path.read_text(encoding="utf-8")
    for text in (html, md):
        assert "Best model per dataset" in text
        assert "ElasticNetCV" in text
        assert "What to do next" in text
        assert "Resolved configuration" in text and "seed: 13" in text
        assert "abc123" in text
        assert "qsarena[graph]" in text  # failure remedy
        assert "no SMILES column" in text  # failed dataset listed
    # self-contained HTML: inline SVG plots, no external resources
    assert html.count("<svg") == 3
    assert not re.search(r'(src|href)="https?://', html)
    assert "report_assets/won_by_family.svg" in md
    assert (output / "report_assets" / "cost_vs_gap.svg").exists()
    manifest = json.loads((output / "report_data.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"completed": 1, "failed": 1}
    assert manifest["families_won"]["cv"] == {"conventional_ml": 1}
    assert any("75%" not in item["text"] for item in manifest["what_next"])


def test_report_without_plots_or_ad(tmp_path):
    output = _run_dir(tmp_path)
    (output / "demo" / "run_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    data = reporting.collect_run_report_data(output, dataset_ids=["demo"])
    html_path, md_path = reporting.write_run_reports(output, data, include_plots=False, what_next=False)
    md = md_path.read_text(encoding="utf-8")
    assert "Applicability domain: not run" in md
    assert "What to do next" not in md
