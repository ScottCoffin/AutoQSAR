"""Phase 7 (optional): deterministic, nested, fixed-test-set learning-curve inputs; off by default."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from qsarena.meta_analysis import io, phase7, pipeline


@pytest.fixture(scope="module")
def partitions():
    return io.load_partitions()


def test_plan_has_one_row_per_dataset_and_size():
    plan = phase7.load_plan()
    assert not plan.duplicated(["dataset", "row_limit"]).any()
    assert (plan["row_limit"] > 0).all(), "full-size points come from the benchmark run, not Phase 7"
    assert set(plan["dataset"]) == {"tdc_ld50_zhu", "tdc_tox21", "tdc_solubility_aqsoldb"}


def test_subsampling_is_deterministic_nested_and_keeps_the_test_set(partitions):
    small = phase7.curve_frame(partitions, "tdc_tox21", 250)
    again = phase7.curve_frame(partitions, "tdc_tox21", 250)
    large = phase7.curve_frame(partitions, "tdc_tox21", 1000)
    pd.testing.assert_frame_equal(small, again)
    train_small = set(small.loc[small["split"] == "train", "SMILES"])
    train_large = set(large.loc[large["split"] == "train", "SMILES"])
    assert len(train_small) <= 250 and train_small <= train_large
    test_small = small.loc[small["split"] == "test"].reset_index(drop=True)
    test_large = large.loc[large["split"] == "test"].reset_index(drop=True)
    pd.testing.assert_frame_equal(test_small, test_large)
    full_test = partitions[(partitions["dataset"] == "tdc_tox21") & (partitions["split"] == "test")]
    assert len(test_small) == len(full_test)
    assert small.loc[small["split"] == "train", "TARGET"].nunique() == 2


def test_build_inputs_writes_manifest(tmp_path, partitions):
    plan = pd.DataFrame(
        {"dataset": ["tdc_ld50_zhu"], "row_limit": [500], "label": ["n0500"], "models": ["Random forest"]}
    )
    manifest = phase7.build_inputs(tmp_path, plan=plan, partitions=partitions)
    assert manifest.loc[0, "n_train"] == 500 and manifest.loc[0, "task"] == "regression"
    assert (tmp_path / "n0500" / "tdc_ld50_zhu.csv").exists()


def test_phase7_is_off_by_default():
    signature = inspect.signature(pipeline.run_meta_analysis)
    assert signature.parameters["include_phase7"].default is False


def test_summarize_finds_crossover():
    rows = []
    for n, rf, unimol in [(250, 0.70, 0.60), (1000, 0.75, 0.74), (4000, 0.78, 0.82)]:
        rows += [
            {"dataset": "d", "model": "Random forest", "n_train": n, "metric": "test_r2", "value": rf},
            {"dataset": "d", "model": "Uni-Mol V1", "n_train": n, "metric": "test_r2", "value": unimol},
        ]
    summary = phase7.summarize(pd.DataFrame(rows))
    assert 1000 < summary["d"]["crossover_n"] < 4000
    assert np.isclose(summary["d"]["unimol_minus_best_tree"][250], -0.10)


def test_figure_s2_renders_from_curves(tmp_path):
    from qsarena.meta_analysis import figures

    rows = [
        {"dataset": d, "model": m, "n_train": n, "metric": "test_r2", "value": 0.5 + 0.01 * k}
        for d in ("tdc_ld50_zhu", "tdc_solubility_aqsoldb")
        for k, m in enumerate(phase7.CURVE_MODELS)
        for n in (250, 1000, 4000)
    ]
    notes = figures.figure_s2(pd.DataFrame(rows), tmp_path)
    for name in notes["files"]:
        assert (tmp_path / name).stat().st_size > 1000
