"""F5: preflight, --dry-run, run.log / events.jsonl and the HTML + Markdown reports."""

from __future__ import annotations

import json

from tests.conftest import FAST_MODELS


def test_dry_run_writes_plan_and_runs_no_models(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "dry"
    result = run_cli(["qsarena-benchmark", "--batch", examples / "batch_manifest.csv", "--output-dir", out,
                      "--benchmark-profile", "quick", "--dry-run"], cwd=tmp_path)
    text = result.stdout
    assert "Preflight checks:" in text
    assert "parse rate 97.9% (1 unparseable, 0 missing)" in text
    assert "task=classification, classes={'0': 16, '1': 28}" in text
    assert "Planned model stages (datasets x models):" in text
    assert "XGBoost (family gradient_boosting switched off by the quick profile)" in text
    assert "Estimated total wall-clock: ~" in text
    assert "Dry run complete: no model was fitted." in text
    for name in ("dry_run_plan.json", "dry_run_plan.md", "preflight.json", "run.log", "events.jsonl"):
        assert (out / name).exists(), name
    assert not list(out.glob("*/metrics.csv"))
    assert not (out / "broken").exists()  # a dry run records nothing per dataset
    plan = json.loads((out / "dry_run_plan.json").read_text(encoding="utf-8"))
    assert plan["discovery_failures"][0]["dataset"] == "broken"
    datasets = {p["dataset"]: p for p in plan["preflight"]["plans"]}
    assert {m["model"] for m in datasets["solubility"]["models"] if m["status"] == "planned"} >= {"ElasticNetCV", "SVR"}


def test_run_writes_report_html_md_runlog_events_and_manifest(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "rep"
    result = run_cli(["qsarena-benchmark", "--dataset", examples / "bbb.csv", "--target-col", "bbb_penetrant",
                      "--output-dir", out, *FAST_MODELS], cwd=tmp_path)
    assert "[preflight] warning bbb: 44 usable rows: the test split will hold about 9 molecules" in result.stdout
    html = (out / "report.html").read_text(encoding="utf-8")
    md = (out / "report.md").read_text(encoding="utf-8")
    for text in (html, md):
        assert "Best model per dataset" in text and "Leaderboard / rank table" in text
        assert "What to do next" in text and "Resolved configuration" in text
    assert html.count("<svg") == 3
    run_log = (out / "run.log").read_text(encoding="utf-8")
    assert "stage 3/" in run_log and "Reports written" in run_log
    events = [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {e["event"] for e in events}
    assert {"run_started", "preflight", "dataset_started", "model_finished", "dataset_finished", "run_completed"} <= kinds
    assert "stage_started" not in kinds  # stage events only at --verbosity verbose
    status = json.loads((out / "bbb" / "run_status.json").read_text(encoding="utf-8"))
    assert status["applicability_domain"]["methods"].keys() >= {"standardization", "confidence"}
    manifest = json.loads((out / "report_data.json").read_text(encoding="utf-8"))
    assert manifest["best_models"][0]["task"] == "classification"


def test_quiet_verbosity_and_ad_off(run_cli, examples, tmp_path):
    out = tmp_path / "runs" / "quiet"
    result = run_cli(["qsarena-benchmark", "--dataset", examples / "batch_dir" / "permeability.csv", "--output-dir", out,
                      "--verbosity", "quiet", "--ad-method", "off", *FAST_MODELS], cwd=tmp_path)
    assert "stage 4/" not in result.stdout  # quiet console
    assert "Reports written" in result.stdout
    assert "stage 4/" in (out / "run.log").read_text(encoding="utf-8")  # the log keeps everything
    assert not (out / "permeability" / "applicability_domain.csv").exists()
    assert "Applicability domain: not run" in (out / "report.md").read_text(encoding="utf-8")
