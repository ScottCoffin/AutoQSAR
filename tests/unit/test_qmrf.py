"""Unit tests for qsarena.qmrf: all five OECD sections filled; loud failure on missing input."""

from __future__ import annotations

import copy
import json

import pytest

from qsarena.qmrf import OECD_SECTIONS, QMRFInputError, build_qmrf_report, render_markdown, write_qmrf_report


@pytest.fixture
def mock_run():
    return {
        "dataset": "tdc_caco2_wang",
        "endpoint": "Caco-2 permeability",
        "task_type": "regression",
        "units": "log10(cm/s)",
        "data_source": "TDC ADMET Benchmark Group",
        "model_name": "CatBoost",
        "features": "MapLight descriptor panel",
        "software_version": "qsarena 0.1.0",
        "random_seed": 0,
        "split": {"strategy": "scaffold", "train_hash": "abc", "test_hash": "def"},
        "applicability_domain": {"method": "kNN Tanimoto", "test_coverage": 0.8,
                                 "error_in_domain": 0.3, "error_out_of_domain": 0.4},
        "metrics": {"train": {"mae": 0.1}, "internal_validation": {"cv_mae": 0.3}, "test": {"mae": 0.31}},
        "interpretation": {"status": "feature",
                           "top_features": [{"rank": 1, "feature": "MolLogP", "importance": 0.2}]},
    }


def test_all_five_sections_are_populated(mock_run):
    report = build_qmrf_report(mock_run)
    for section in OECD_SECTIONS:
        assert section in report and report[section], section
    md = render_markdown(report)
    for section in OECD_SECTIONS:
        assert f"## {section}" in md
    assert "`MolLogP`" in md


def test_missing_inputs_fail_loudly_and_name_every_gap(mock_run):
    bad = copy.deepcopy(mock_run)
    del bad["endpoint"]
    bad["split"]["test_hash"] = ""
    del bad["applicability_domain"]
    with pytest.raises(QMRFInputError) as exc:
        build_qmrf_report(bad)
    missing = exc.value.missing
    assert missing[OECD_SECTIONS[0]] == ["endpoint"]
    assert "split.test_hash" in missing[OECD_SECTIONS[1]]
    assert set(missing[OECD_SECTIONS[2]]) == {"applicability_domain.method", "applicability_domain.test_coverage"}


def test_feature_model_without_top_features_fails(mock_run):
    bad = copy.deepcopy(mock_run)
    bad["interpretation"] = {"status": "feature"}
    with pytest.raises(QMRFInputError, match="top_features"):
        build_qmrf_report(bad)


def test_graph_model_may_omit_features(mock_run):
    run = copy.deepcopy(mock_run)
    run["interpretation"] = {"status": "no per-feature attribution"}
    assert build_qmrf_report(run)[OECD_SECTIONS[4]]["status"] == "no per-feature attribution"


def test_write_emits_markdown_and_json(mock_run, tmp_path):
    md, js = write_qmrf_report(mock_run, tmp_path)
    assert md.exists() and js.exists()
    assert set(OECD_SECTIONS) <= set(json.loads(js.read_text(encoding="utf-8")))
