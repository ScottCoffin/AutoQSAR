"""End-to-end tests on the tiny TDC fixture: AD CLI, reliability study and parity table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def test_ad_cli_runs_end_to_end_and_writes_csv(tdc_tiny, tmp_path):
    from portable_colab_qsar_bundle.simple_applicability_domain import main

    out_csv, meta = tmp_path / "ad.csv", tmp_path / "ad.json"
    main([
        "--train_csv", str(tdc_tiny / "caco2_wang" / "train_val.csv"),
        "--smiles_col", "Drug",
        "--query_smiles", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
        "--output_csv", str(out_csv),
        "--metadata_json", str(meta),
    ])
    row = pd.read_csv(out_csv).iloc[0]
    for col in ("ad_standardization_in_domain", "ad_knn_in_domain", "ad_mahalanobis_in_domain",
                "ad_consensus_concern"):
        assert col in row.index, col
    assert row["ad_method_available_count"] == 3
    assert json.loads(meta.read_text(encoding="utf-8"))["n_training_chemicals"] == 36


def test_reliability_study_writes_all_artifacts(tdc_tiny, tmp_path):
    from qsarena.reliability_study import run_study

    summary = run_study(tdc_tiny, tmp_path, datasets=["caco2_wang", "herg"], n_estimators=20,
                        canonical_run=None, cache_dir=None, qmrf_dataset="caco2_wang", n_jobs=1)
    for name in ("applicability_domain.csv", "calibration.csv", "per_molecule_predictions.csv",
                 "feature_importance_top10.csv", "split_hashes.csv", "summary.json",
                 "environment_manifest.json", "qmrf_caco2_wang.md", "qmrf_caco2_wang.json"):
        assert (tmp_path / name).exists(), name

    ad = pd.read_csv(tmp_path / "applicability_domain.csv")
    assert set(ad["dataset"]) == {"caco2_wang", "herg"}
    assert {"standardization", "knn_tanimoto", "consensus"} <= set(ad["ad_method"])
    assert ad["coverage"].between(0, 1).all()
    # Coverage is consistent with the in/out counts.
    np.testing.assert_allclose(ad["coverage"], ad["n_in"] / (ad["n_in"] + ad["n_out"]))

    per_mol = pd.read_csv(tmp_path / "per_molecule_predictions.csv")
    assert len(per_mol) == 24  # 12 test molecules per dataset
    assert {"ad_consensus_in_domain", "conformal_covered"} <= set(per_mol.columns)
    assert summary["n_datasets"] == 2 and summary["n_regression"] == 1

    from qsarena.reliability_tables import build_table, to_latex, to_markdown

    table = build_table(tmp_path)
    assert list(table["Dataset"]) == ["caco2_wang", "herg"]  # regression first
    # 12 test molecules cannot give >= 5 on both sides everywhere; unreported ratios must be NaN.
    tex, md = to_latex(table), to_markdown(table)
    assert "\\label{tab:s8}" in tex and tex.count("\\\\") >= 2
    assert md.count("\n") == 4


def test_reliability_study_is_deterministic(tdc_tiny, tmp_path):
    from qsarena.reliability_study import run_study

    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        run_study(tdc_tiny, out, datasets=["herg"], n_estimators=15, canonical_run=None,
                  cache_dir=None, qmrf_dataset=None, n_jobs=1)
    pd.testing.assert_frame_equal(pd.read_csv(a / "per_molecule_predictions.csv"),
                                  pd.read_csv(b / "per_molecule_predictions.csv"))


def _write_metrics(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)


def test_parity_table_keeps_every_dataset_and_marks_na(tmp_path):
    from qsarena.admet_ai_parity import build_parity_table

    run = tmp_path / "run"
    variant = "Chemprop v2 (D-MPNN + RDKit2D, ensemble=3)"
    # caco2: valid on retry row 2 (first row is a harness error) -> must pick the valid row.
    _write_metrics(run / "tdc_caco2_wang", [
        {"model": variant, "test_mae": np.nan, "error": "Chemprop harness error"},
        {"model": variant, "test_mae": 0.30, "error": np.nan},
    ])
    # lipophilicity: variant failed everywhere.
    _write_metrics(run / "tdc_lipophilicity_astrazeneca", [
        {"model": variant, "test_mae": np.nan, "error": "Chemprop training failed"},
    ])
    # herg: variant valid, but no reference in the leaderboard.
    _write_metrics(run / "tdc_herg", [{"model": variant, "test_roc_auc": 0.8, "error": np.nan}])
    lb = pd.DataFrame([
        {"dataset": "tdc_caco2_wang", "leaderboard_metric_name": "MAE", "model": "Chemprop-RDKit",
         "metric_value_numeric": 0.32, "rank_numeric": 5},
        {"dataset": "tdc_lipophilicity_astrazeneca", "leaderboard_metric_name": "MAE",
         "model": "Chemprop-RDKit", "metric_value_numeric": 0.467, "rank_numeric": 2},
    ])
    lb_path = tmp_path / "lb.csv"
    lb.to_csv(lb_path, index=False)

    datasets = ("caco2_wang", "lipophilicity_astrazeneca", "herg", "dili")
    table = build_parity_table(run, lb_path, datasets=datasets).set_index("dataset")
    assert list(table.index) == list(datasets)  # nothing dropped
    assert table.loc["caco2_wang", "ours_value"] == pytest.approx(0.30)
    assert bool(table.loc["caco2_wang", "parity_holds_within_5pct"]) is True
    assert table.loc["lipophilicity_astrazeneca", "ours_status"] == "variant failed"
    assert pd.isna(table.loc["lipophilicity_astrazeneca", "parity_holds_within_5pct"])
    assert table.loc["herg", "leaderboard_metric"] == "AUROC"  # official-metric fallback
    assert table.loc["herg", "reference_status"].startswith("Chemprop-RDKit not in")
    assert table.loc["dili", "ours_status"] == "dataset not in run"

    out = tmp_path / "parity.csv"
    table.reset_index().to_csv(out, index=False, na_rep="NA")
    assert len(pd.read_csv(out, keep_default_na=False)) == len(datasets)
