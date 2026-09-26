from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from portable_colab_qsar_bundle import run_qsarena_benchmarks as runner


def test_full_chemprop_configuration_enumerates_five_variants() -> None:
    args = argparse.Namespace(
        run_chemprop_dmpnn=True,
        run_chemprop_cmpnn=True,
        run_chemprop_attentivefp=True,
        run_chemprop_selected_features=True,
        run_chemprop_rdkit2d=True,
        chemprop_ensemble_size=3,
    )
    specs = runner.chemprop_variant_specs(args)
    labels = {str(spec["label"]) for spec in specs}
    tags = {str(spec["variant_tag"]) for spec in specs}
    assert len(specs) == 5
    assert {"dmpnn", "cmpnn", "attentivefp", "dmpnn_rdkit2d", "dmpnn_selected_features"} == tags
    assert any("D-MPNN + RDKit2D" in label for label in labels)
    assert any("Selected descriptors" in label for label in labels)


def _payload(train_pred, test_pred):
    return {
        "workflow": "Conventional ML",
        "train_smiles": pd.Series(["C", "CC", "CCC", "CCCC"]),
        "test_smiles": pd.Series(["N", "NN", "NNN", "NNNN"]),
        "train_observed": np.array([0.0, 1.0, 2.0, 3.0]),
        "test_observed": np.array([0.0, 1.0, 2.0, 3.0]),
        "train": np.asarray(train_pred, dtype=float),
        "test": np.asarray(test_pred, dtype=float),
    }


def test_ensemble_train_member_selection_never_reads_test_split() -> None:
    previous = runner.CURRENT_DATASET_SPEC
    runner.CURRENT_DATASET_SPEC = runner.DatasetSpec(
        name="toy",
        source="unit",
        frame=pd.DataFrame({"SMILES": [], "y": []}),
        smiles_column="SMILES",
        target_column="y",
        recommended_metric="rmse",
        task_type="regression",
    )
    try:
        payloads = {
            "good_a": _payload([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]),
            "good_b": _payload([0.1, 1.0, 1.9, 3.0], [3.0, 2.0, 1.1, 0.0]),
            # Excellent on held-out rows, but bad on train. Leakage-free filtering must drop it.
            "test_star_train_bad": _payload([3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]),
        }
        _results, _weights, _train, _test, active, notes, _meta = runner.build_ensemble_result(
            payloads=payloads,
            method="Simple average",
            stacking_cv_folds=2,
            random_seed=0,
            drop_highly_correlated_members=False,
            max_train_prediction_correlation=0.995,
            exclude_negative_test_r2_members=True,
            member_selection_split="train",
        )
    finally:
        runner.CURRENT_DATASET_SPEC = previous
    assert active == ["good_a", "good_b"]
    assert any("Train R2" in note for note in notes)
    assert not any("Test R2" in note for note in notes)
