"""Feature-expansion featurizers: row order, cache validation, Mol2Vec correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsarena.feature_expansion import featurize as fz

pytest.importorskip("skfp")


def _partitions():
    rows = []
    for split, smiles in (("test", ["CCN", "CCCl"]), ("train", ["c1ccccc1O", "CCO", "CC(=O)O"])):
        for i, s in enumerate(smiles):
            rows.append({"dataset": "toy", "split": split, "row_index": 10 - i, "smiles": s, "observed": 1.0})
    return pd.DataFrame(rows)


def test_row_order_is_train_then_test_by_row_index():
    smiles, n_train = fz.dataset_smiles(_partitions(), "toy")
    assert n_train == 3
    assert smiles == ["CC(=O)O", "CCO", "c1ccccc1O", "CCCl", "CCN"]


def test_cache_roundtrip_and_staleness(tmp_path):
    smiles, _ = fz.dataset_smiles(_partitions(), "toy")
    fz.featurize_dataset("maccs", "toy", smiles, tmp_path)
    assert fz.is_cached("maccs", "toy", smiles, tmp_path)
    assert "cached" in fz.featurize_dataset("maccs", "toy", smiles, tmp_path)
    assert not fz.is_cached("maccs", "toy", smiles[::-1], tmp_path)  # different rows -> stale
    X, n_train = fz.load_features(["maccs"], "toy", _partitions(), tmp_path)
    assert X.shape == (5, 166) and n_train == 3 and X.dtype == np.float32


def test_nonfinite_descriptors_become_zero():
    X, extra = fz.compute_family("rdkit2d", ["CCO", "c1ccccc1"])
    assert np.isfinite(X).all() and "nonfinite_fraction" in extra


@pytest.mark.skipif(not fz.MOL2VEC_MODEL.exists(), reason="Mol2Vec model not downloaded")
def test_mol2vec_uses_legacy_morgan_identifiers():
    X, hit_rate = fz.mol2vec_matrix(["CC(=O)Nc1ccc(O)cc1", "c1ccccc1O", "CCN(CC)CC"])
    assert hit_rate > 0.95  # identifiers must match the vocabulary the model was trained on
    assert X.shape == (3, 300) and np.all(np.abs(X).sum(axis=1) > 0)
    again, _ = fz.mol2vec_matrix(["CC(=O)Nc1ccc(O)cc1", "c1ccccc1O", "CCN(CC)CC"])
    assert np.array_equal(X, again)


def test_mol2vec_sentence_interleaves_radii():
    from rdkit import Chem

    words = fz.mol2vec_sentence(Chem.MolFromSmiles("CCO"))
    assert len(words) == 6  # 3 atoms x radii 0 and 1
