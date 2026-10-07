"""Class-stratified CV folds for classification (notebook path); the default keeps the runner's folds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from portable_colab_qsar_bundle import qsar_workflow_core as core


def _data(n=60, positives=12):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 3)))
    y = np.array([1.0] * positives + [0.0] * (n - positives))
    smiles = [f"C{'C' * i}O" for i in range(n)]
    return X, y, smiles


def test_classification_folds_are_class_stratified():
    X, y, smiles = _data()
    folds, n_folds, strategy = core.make_qsar_cv_splitter(
        X, y, smiles, split_strategy="target_quartiles", cv_folds=5, random_seed=1, task_type="classification"
    )
    assert strategy == "class_stratified" and n_folds == 5
    for _fit, val in folds:
        assert int(y[val].sum()) in (2, 3)  # 12 positives spread over 5 folds


def test_default_task_type_reproduces_the_runner_folds():
    X, y, smiles = _data()
    default = core.make_oof_folds(X, y, smiles, split_strategy="random", n_folds=5, random_seed=1)
    explicit = core.make_oof_folds(X, y, smiles, split_strategy="random", n_folds=5, random_seed=1, task_type="regression")
    assert core.oof_fold_signature(default) == core.oof_fold_signature(explicit)
    stratified = core.make_oof_folds(X, y, smiles, split_strategy="random", n_folds=5, random_seed=1, task_type="classification")
    assert core.oof_fold_signature(stratified) != core.oof_fold_signature(default)
