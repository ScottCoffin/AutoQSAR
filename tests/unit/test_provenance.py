"""Unit tests for qsarena.provenance: manifests carry versions + commit; split hashes are stable."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from qsarena.provenance import KEY_PACKAGES, split_hash, write_environment_manifest


def test_manifest_has_versions_commit_and_hardware(tmp_path, repo_root):
    path = write_environment_manifest(tmp_path, repo_dir=repo_root, extra={"seed": 0})
    m = json.loads(path.read_text(encoding="utf-8"))
    assert set(KEY_PACKAGES) <= set(m["key_packages"])
    for pkg in ("numpy", "pandas", "scikit-learn", "rdkit"):
        assert m["key_packages"][pkg], pkg
    assert any(d.lower().startswith("numpy==") for d in m["installed_distributions"])
    assert "commit" in m["git"]
    if (repo_root / ".git").exists():
        assert m["git"]["commit"] and len(m["git"]["commit"]) == 40
    assert m["hardware"]["cpu_count"] >= 1
    assert m["extra"] == {"seed": 0}


def test_split_hash_matches_runner_definition():
    from portable_colab_qsar_bundle.run_qsarena_benchmarks import smiles_hash

    smiles = [" CCO", "c1ccccc1 ", "CC(=O)O"]
    assert split_hash(smiles) == smiles_hash(smiles)


def test_split_hash_stable_across_same_seed_resplits():
    from sklearn.model_selection import train_test_split

    smiles = [f"C{'C' * i}O" for i in range(60)]
    a_train, a_test = train_test_split(smiles, test_size=0.2, random_state=3)
    b_train, b_test = train_test_split(smiles, test_size=0.2, random_state=3)
    c_train, _ = train_test_split(smiles, test_size=0.2, random_state=4)
    assert split_hash(a_train) == split_hash(b_train)
    assert split_hash(a_test) == split_hash(b_test)
    assert split_hash(a_train) != split_hash(c_train)
    assert split_hash(pd.Series(a_train)) == split_hash(np.array(a_train))
