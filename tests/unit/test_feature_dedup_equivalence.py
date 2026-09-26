"""The faster exact-duplicate scan in drop_exact_and_near_duplicate_features must drop exactly the
columns the original hash_pandas_object + Series.equals scan dropped (benchmark reproducibility)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portable_colab_qsar_bundle.qsar_workflow_core import (
    build_feature_matrix_from_smiles,
    drop_exact_and_near_duplicate_features,
)


def _original_exact_duplicates(frame: pd.DataFrame) -> tuple[list, list]:
    """Verbatim copy of the pre-optimization loop."""
    dropped, pairs, buckets = [], [], {}
    for column in list(frame.columns):
        series = frame[column]
        key = int(pd.util.hash_pandas_object(series, index=False).sum())
        bucket = buckets.setdefault(key, [])
        duplicate_of = None
        for prior in bucket:
            if series.equals(frame[prior]):
                duplicate_of = prior
                break
        if duplicate_of is not None:
            dropped.append(column)
            pairs.append((str(duplicate_of), str(column)))
        else:
            bucket.append(column)
    return dropped, pairs


def _adversarial_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 30
    base = rng.integers(0, 2, size=(n, 12)).astype(float)
    cols = {f"b{i}": base[:, i] for i in range(12)}
    cols["dup_float"] = base[:, 3].copy()
    cols["int_twin"] = base[:, 3].astype(np.int64)  # same values, different dtype: NOT a duplicate
    cols["int_twin2"] = base[:, 3].astype(np.int64)  # duplicate of int_twin
    nan_col = rng.normal(size=n)
    nan_col[[2, 7]] = np.nan
    cols["nan_a"] = nan_col
    cols["nan_b"] = nan_col.copy()  # NaN in the same rows: duplicate
    shifted = nan_col.copy()
    shifted[2], shifted[3] = shifted[3], np.nan
    cols["nan_c"] = shifted  # NaN elsewhere: not a duplicate
    zero = np.zeros(n)
    cols["zero_pos"] = zero + 1.0
    cols["zero_neg"] = -zero + 1.0
    signed = np.zeros(n)
    signed[0] = -0.0
    cols["signed_a"] = np.where(np.arange(n) % 2 == 0, 1.5, 0.0)
    cols["signed_b"] = np.where(np.arange(n) % 2 == 0, 1.5, -0.0)  # hashes differ: never a duplicate
    cols["bool_a"] = base[:, 5].astype(bool)
    cols["bool_b"] = base[:, 5].astype(bool)
    cols["obj_a"] = pd.Series(["x", "y"] * (n // 2), dtype=object)
    cols["obj_b"] = pd.Series(["x", "y"] * (n // 2), dtype=object)
    return pd.DataFrame(cols)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_duplicate_scan_matches_original(seed):
    frame = _adversarial_frame(seed)
    expected_dropped, expected_pairs = _original_exact_duplicates(frame)
    _train, _other, meta = drop_exact_and_near_duplicate_features(
        frame, None, variance_threshold=-1.0, binary_prevalence_min=float("nan"), binary_prevalence_max=float("nan"),
        report_limit=10_000,
    )
    assert meta["dropped_exact_columns"] == expected_dropped
    assert [(p["kept"], p["dropped"]) for p in meta["exact_duplicate_examples"]] == expected_pairs
    assert not {"int_twin", "nan_c", "signed_b"} & set(expected_dropped)
    assert {"dup_float", "int_twin2", "nan_b", "bool_b", "obj_b"} <= set(expected_dropped)


def test_exact_duplicate_scan_matches_original_on_real_fingerprints():
    smiles = ["CCO", "CCN", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCC", "C1CCNCC1", "OC(=O)CCC(=O)O",
              "Clc1ccccc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "CN1CCC[C@H]1c1cccnc1"]
    frame, _meta = build_feature_matrix_from_smiles(
        smiles, selected_feature_families=["morgan", "maccs", "rdkit"], enable_persistent_feature_store=False
    )
    expected_dropped, _ = _original_exact_duplicates(frame)
    _train, _other, meta = drop_exact_and_near_duplicate_features(
        frame, None, variance_threshold=-1.0, binary_prevalence_min=float("nan"), binary_prevalence_max=float("nan")
    )
    assert meta["dropped_exact_columns"] == expected_dropped
    assert len(expected_dropped) > 100
