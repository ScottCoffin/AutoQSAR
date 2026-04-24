from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdReducedGraphs, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split

try:
    import pyarrow  # noqa: F401

    FEATURE_STORE_SHARD_FORMAT = "parquet"
except Exception:
    try:
        import fastparquet  # noqa: F401

        FEATURE_STORE_SHARD_FORMAT = "parquet"
    except Exception:
        FEATURE_STORE_SHARD_FORMAT = "csv"


FEATURE_FAMILY_LABELS = {
    "morgan": "Morgan fingerprints",
    "ecfp6": "ECFP6 fingerprints",
    "fcfp6": "FCFP6 fingerprints",
    "layered": "RDKit layered fingerprints",
    "atom_pair": "Atom-pair fingerprints",
    "topological_torsion": "Topological torsion fingerprints",
    "rdk_path": "RDKit path fingerprints",
    "maccs": "MACCS keys",
    "rdkit": "RDKit descriptors",
    "maplight": "MapLight classic (Morgan + Avalon + ErG + chosen descriptors)",
}


CHEMPROP_ARCHITECTURE_REGISTRY = {
    "dmpnn": {
        "display_name": "D-MPNN",
        "variant_tag": "dmpnn",
        "workflow": "Chemprop v2",
        "train_args": [],
        "molecule_featurizers": [],
        "notes": "Chemprop v2 default directed message-passing network.",
    },
    "cmpnn": {
        "display_name": "CMPNN",
        "variant_tag": "cmpnn",
        "workflow": "Chemprop v2",
        "train_args": ["--atom-messages", "--undirected"],
        "molecule_featurizers": [],
        "notes": "CMPNN-style Chemprop configuration via atom-level message passing.",
    },
    "attentivefp": {
        "display_name": "AttentiveFP",
        "variant_tag": "attentivefp",
        "workflow": "Chemprop v2",
        "train_args": [
            "--atom-messages",
            "--aggregation", "norm",
            "--aggregation-norm", "100",
            "--dropout", "0.1",
        ],
        "molecule_featurizers": ["rdkit_2d"],
        "notes": "AttentiveFP-style Chemprop proxy with atom messages plus descriptor fusion.",
    },
}


class TabularCNNRegressor(BaseEstimator, RegressorMixin):
    """Lightweight 1D-CNN regressor for tabular molecular descriptors.

    The model treats each feature vector as a 1D signal and applies small
    convolutional blocks before dense regression heads. It is intentionally
    compact so it can run on CPU.
    """

    def __init__(
        self,
        conv_filters: int = 64,
        kernel_size: int = 5,
        dense_units: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 64,
        validation_split: float = 0.1,
        early_stopping_patience: int = 5,
        random_seed: int = 42,
        verbose: int = 0,
    ):
        self.conv_filters = int(conv_filters)
        self.kernel_size = int(kernel_size)
        self.dense_units = int(dense_units)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.validation_split = float(validation_split)
        self.early_stopping_patience = int(early_stopping_patience)
        self.random_seed = int(random_seed)
        self.verbose = int(verbose)
        self.model_ = None
        self.history_ = None
        self.n_features_in_ = None

    def _build_model(self, n_features: int):
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(int(n_features), 1)),
                tf.keras.layers.Conv1D(
                    filters=int(max(8, self.conv_filters)),
                    kernel_size=int(max(1, self.kernel_size)),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv1D(
                    filters=int(max(8, self.conv_filters)),
                    kernel_size=int(max(1, self.kernel_size)),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(int(max(8, self.dense_units)), activation="relu"),
                tf.keras.layers.Dropout(float(min(max(self.dropout_rate, 0.0), 0.8))),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(max(self.learning_rate, 1e-6))),
            loss="mse",
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf

        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("TabularCNNRegressor expects a 2D feature matrix.")
        if X_arr.shape[0] < 2:
            raise ValueError("TabularCNNRegressor requires at least two training rows.")

        tf.keras.utils.set_random_seed(int(self.random_seed))

        self.n_features_in_ = int(X_arr.shape[1])
        self.model_ = self._build_model(self.n_features_in_)
        X_seq = X_arr[:, :, None]

        validation_split = float(min(max(self.validation_split, 0.0), 0.3))
        if X_arr.shape[0] < max(20, int(self.batch_size) * 2):
            validation_split = 0.0

        callbacks = []
        if validation_split > 0.0 and int(self.early_stopping_patience) > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(self.early_stopping_patience),
                    restore_best_weights=True,
                )
            )

        history = self.model_.fit(
            X_seq,
            y_arr,
            epochs=int(max(1, self.epochs)),
            batch_size=int(max(8, self.batch_size)),
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=int(self.verbose),
        )
        self.history_ = dict(history.history)
        return self

    def predict(self, X):
        import tensorflow as tf

        if self.model_ is None:
            raise RuntimeError(
                "TabularCNNRegressor model object is unavailable in this session. "
                "Retrain the model to restore prediction support."
            )
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X_arr.ndim != 2:
            raise ValueError("TabularCNNRegressor expects a 2D feature matrix.")
        if self.n_features_in_ is not None and int(X_arr.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"TabularCNNRegressor expected {int(self.n_features_in_)} features but received {int(X_arr.shape[1])}."
            )
        X_seq = X_arr[:, :, None]
        try:
            # Prefer eager forward-pass to avoid repeated tf.function retracing
            # warnings during many small predict calls inside CV loops.
            preds = self.model_(X_seq, training=False)
            if tf.is_tensor(preds):
                preds = preds.numpy()
        except Exception:
            preds = self.model_.predict(
                X_seq,
                batch_size=int(max(8, self.batch_size)),
                verbose=0,
            )
        return np.asarray(preds, dtype=float).reshape(-1)


def _cfa_normalize_vector(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    min_val = float(np.nanmin(arr[finite]))
    max_val = float(np.nanmax(arr[finite]))
    span = max_val - min_val
    if not np.isfinite(span) or abs(span) <= float(epsilon):
        return np.zeros_like(arr, dtype=float)
    out = (arr - min_val) / span
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _cfa_metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "mae") -> float:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("CFA metric inputs must have the same number of rows.")
    metric_key = str(metric or "mae").strip().lower()
    metric_key = metric_key.replace(" ", "_").replace("-", "_")

    def _binary_labels(values: np.ndarray) -> np.ndarray:
        series = pd.Series(values, dtype=float)
        uniques = sorted(series.dropna().unique().tolist())
        if len(uniques) != 2:
            raise ValueError("Binary classification metrics require exactly two classes.")
        mapping = {float(uniques[0]): 0, float(uniques[1]): 1}
        mapped = series.map(mapping)
        if mapped.isna().any():
            raise ValueError("Binary classification labels could not be encoded.")
        return mapped.astype(int).to_numpy()

    def _binary_scores(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = np.isfinite(arr)
        if not bool(finite.any()):
            return np.zeros_like(arr, dtype=float)
        if float(np.nanmin(arr)) >= 0.0 and float(np.nanmax(arr)) <= 1.0:
            return arr
        clipped = np.clip(arr, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _binary_predictions(values: np.ndarray) -> np.ndarray:
        return (_binary_scores(values) >= 0.5).astype(int)

    if metric_key == "rmse":
        return float(np.sqrt(np.mean(np.square(truth - pred))))
    if metric_key in {"roc_auc", "auroc", "auc"}:
        try:
            y_bin = _binary_labels(truth)
            score = float(roc_auc_score(y_bin, _binary_scores(pred)))
            return float(1.0 - score)
        except Exception:
            return 1.0
    if metric_key in {"auprc", "average_precision", "pr_auc"}:
        try:
            y_bin = _binary_labels(truth)
            score = float(average_precision_score(y_bin, _binary_scores(pred)))
            return float(1.0 - score)
        except Exception:
            return 1.0
    if metric_key in {"accuracy"}:
        try:
            y_bin = _binary_labels(truth)
            score = float(np.mean(_binary_predictions(pred) == y_bin))
            return float(1.0 - score)
        except Exception:
            return 1.0
    if metric_key in {"balanced_accuracy", "balanced_acc", "bal_acc", "bac"}:
        try:
            y_bin = _binary_labels(truth)
            score = float(balanced_accuracy_score(y_bin, _binary_predictions(pred)))
            return float(1.0 - score)
        except Exception:
            return 1.0
    if metric_key in {"mcc", "matthews_corrcoef", "matthews_correlation_coefficient"}:
        try:
            y_bin = _binary_labels(truth)
            score = float(matthews_corrcoef(y_bin, _binary_predictions(pred)))
            score01 = 0.5 * (score + 1.0)
            return float(1.0 - score01)
        except Exception:
            return 1.0
    return float(np.mean(np.abs(truth - pred)))


def _cfa_score_to_rank(values: np.ndarray, *, descending: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    series = pd.Series(arr)
    ranked = series.rank(method="average", ascending=not bool(descending))
    return np.asarray(ranked, dtype=float).reshape(-1)


def _cfa_fit_rank_calibration(
    rank_signal_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[float, float, float]:
    signal_train = np.asarray(rank_signal_train, dtype=float).reshape(-1)
    target = np.asarray(y_train, dtype=float).reshape(-1)
    if signal_train.shape[0] != target.shape[0]:
        raise ValueError("Rank calibration inputs must align with y_train.")
    finite_mask = np.isfinite(signal_train) & np.isfinite(target)
    if int(finite_mask.sum()) < 2:
        fallback = float(np.nanmean(target)) if np.isfinite(np.nanmean(target)) else 0.0
        return 0.0, float(fallback), float(fallback)
    x = signal_train[finite_mask]
    y = target[finite_mask]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if abs(denom) <= float(epsilon):
        return 0.0, float(y_mean), float(y_mean)
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return float(slope), float(intercept), float(y_mean)


def _cfa_calibrate_rank_signal(
    rank_signal_train: np.ndarray,
    rank_signal_predict: np.ndarray,
    y_train: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Map rank-space fusion signals to score-space via 1D least-squares calibration."""
    signal_train = np.asarray(rank_signal_train, dtype=float).reshape(-1)
    signal_predict = np.asarray(rank_signal_predict, dtype=float).reshape(-1)
    slope, intercept, y_fill = _cfa_fit_rank_calibration(
        signal_train,
        np.asarray(y_train, dtype=float).reshape(-1),
        epsilon=float(epsilon),
    )
    pred_train = slope * signal_train + intercept
    pred_predict = slope * signal_predict + intercept
    pred_train = np.where(np.isfinite(pred_train), pred_train, y_fill)
    pred_predict = np.where(np.isfinite(pred_predict), pred_predict, y_fill)
    return pred_train, pred_predict


def _cfa_coerce_prediction_map(prediction_map: dict[str, Any] | pd.DataFrame) -> dict[str, np.ndarray]:
    if isinstance(prediction_map, pd.DataFrame):
        if prediction_map.empty:
            return {}
        return {
            str(col): np.asarray(prediction_map[col], dtype=float).reshape(-1)
            for col in prediction_map.columns
        }
    out: dict[str, np.ndarray] = {}
    for key, values in dict(prediction_map or {}).items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = np.asarray(values, dtype=float).reshape(-1)
    return out


def run_cfa_regression_fusion(
    train_prediction_map: dict[str, Any] | pd.DataFrame,
    test_prediction_map: dict[str, Any] | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    *,
    min_models: int = 2,
    max_models: int | None = None,
    optimize_metric: str = "mae",
    epsilon: float = 1e-12,
    include_rank_combinations: bool = True,
    rank_prefer_when_diverse: bool = True,
    rank_diversity_threshold: float = 0.15,
    rank_metric_discount: float = 0.98,
) -> dict[str, Any]:
    """Run a lightweight CFA-style combinatorial fusion search for regression.

    This follows the core CFA concept (combinatorial score fusion with
    performance-strength and diversity-strength weighting), adapted to operate
    on already-produced train/test prediction vectors from existing models.
    """
    train_map = _cfa_coerce_prediction_map(train_prediction_map)
    test_map = _cfa_coerce_prediction_map(test_prediction_map)
    common_models = [name for name in sorted(train_map.keys()) if name in test_map]
    if not common_models:
        raise ValueError("CFA fusion requires at least one model with both train and test predictions.")

    y_train_arr = np.asarray(y_train, dtype=float).reshape(-1)
    train_lengths = {name: int(np.asarray(train_map[name]).shape[0]) for name in common_models}
    test_lengths = {name: int(np.asarray(test_map[name]).shape[0]) for name in common_models}
    if any(length != int(y_train_arr.shape[0]) for length in train_lengths.values()):
        raise ValueError("CFA train predictions must align exactly with y_train length.")
    unique_test_lengths = sorted(set(test_lengths.values()))
    if len(unique_test_lengths) != 1:
        raise ValueError("CFA test predictions must all have the same number of rows.")

    n_models = len(common_models)
    min_models = int(max(1, min_models))
    max_models = int(max_models) if max_models is not None else n_models
    max_models = max(min_models, min(max_models, n_models))
    if n_models < min_models:
        raise ValueError(
            f"CFA fusion requires at least {int(min_models)} base models; found {int(n_models)}."
        )

    train_matrix = np.column_stack([np.asarray(train_map[name], dtype=float).reshape(-1) for name in common_models])
    test_matrix = np.column_stack([np.asarray(test_map[name], dtype=float).reshape(-1) for name in common_models])

    # Performance strength: lower train error means higher contribution weight.
    base_train_metric = np.asarray(
        [_cfa_metric_value(y_train_arr, train_matrix[:, idx], metric=optimize_metric) for idx in range(n_models)],
        dtype=float,
    )
    performance_strength = 1.0 / np.maximum(base_train_metric, float(epsilon))

    # Diversity strength: average pairwise distance to other model score profiles.
    diversity_strength = np.zeros((n_models,), dtype=float)
    if n_models > 1:
        normalized_sorted = np.column_stack(
            [
                _cfa_normalize_vector(np.sort(train_matrix[:, idx]), epsilon=float(epsilon))
                for idx in range(n_models)
            ]
        )
        for idx in range(n_models):
            distances = []
            for jdx in range(n_models):
                if jdx == idx:
                    continue
                distances.append(float(np.mean(np.square(normalized_sorted[:, idx] - normalized_sorted[:, jdx]))))
            diversity_strength[idx] = float(np.mean(distances)) if distances else 0.0
    if not np.any(np.isfinite(diversity_strength)) or float(np.nanmax(np.abs(diversity_strength))) <= float(epsilon):
        diversity_strength = np.ones((n_models,), dtype=float)

    pairwise_cd = np.zeros((n_models, n_models), dtype=float)
    if n_models > 1:
        normalized_sorted = np.column_stack(
            [
                _cfa_normalize_vector(np.sort(train_matrix[:, idx]), epsilon=float(epsilon))
                for idx in range(n_models)
            ]
        )
        for idx in range(n_models):
            for jdx in range(idx + 1, n_models):
                cd_value = float(np.mean(np.square(normalized_sorted[:, idx] - normalized_sorted[:, jdx])))
                pairwise_cd[idx, jdx] = cd_value
                pairwise_cd[jdx, idx] = cd_value

    candidates: list[dict[str, Any]] = []

    def _safe_weights(raw_weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(raw_weights, dtype=float).reshape(-1)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        total = float(np.sum(weights))
        if not np.isfinite(total) or abs(total) <= float(epsilon):
            return np.ones_like(weights, dtype=float) / max(1, len(weights))
        return weights / total

    for subset_size in range(min_models, max_models + 1):
        for subset in itertools.combinations(range(n_models), subset_size):
            subset_idx = np.asarray(subset, dtype=int)
            subset_models = [common_models[idx] for idx in subset_idx]
            subset_train = train_matrix[:, subset_idx]
            subset_diversity_values = pairwise_cd[np.ix_(subset_idx, subset_idx)]
            if subset_size > 1:
                subset_diversity_mean = float(np.mean(subset_diversity_values[np.triu_indices(subset_size, k=1)]))
            else:
                subset_diversity_mean = 0.0
            score_variants = {
                "score_ac": np.ones((subset_size,), dtype=float),
                "score_wcp": performance_strength[subset_idx],
                "score_wcds": diversity_strength[subset_idx],
            }
            rank_variants = {
                "rank_ac": np.ones((subset_size,), dtype=float),
                "rank_wcp": performance_strength[subset_idx],
                "rank_wcds": diversity_strength[subset_idx],
            }

            for variant_name, raw_weights in score_variants.items():
                weights = _safe_weights(raw_weights)
                fused_train = np.dot(subset_train, weights)
                fused_test = np.dot(test_matrix[:, subset_idx], weights)
                metric_value = _cfa_metric_value(y_train_arr, fused_train, metric=optimize_metric)
                adjusted_metric = float(metric_value)
                candidates.append(
                    {
                        "variant": str(variant_name),
                        "combination_space": "score",
                        "n_models": int(subset_size),
                        "models": "|".join(subset_models),
                        "train_metric": float(metric_value),
                        "adjusted_metric": float(adjusted_metric),
                        "subset_diversity_mean": float(subset_diversity_mean),
                        "rank_calibration_slope": np.nan,
                        "rank_calibration_intercept": np.nan,
                        "rank_descending": np.nan,
                        "weights": {name: float(weight) for name, weight in zip(subset_models, weights)},
                        "pred_train": np.asarray(fused_train, dtype=float).reshape(-1),
                        "pred_test": np.asarray(fused_test, dtype=float).reshape(-1),
                    }
                )

            if bool(include_rank_combinations):
                subset_rank_train = np.column_stack(
                    [_cfa_score_to_rank(train_matrix[:, idx], descending=True) for idx in subset_idx]
                )
                subset_rank_test = np.column_stack(
                    [_cfa_score_to_rank(test_matrix[:, idx], descending=True) for idx in subset_idx]
                )
                for variant_name, raw_weights in rank_variants.items():
                    weights = _safe_weights(raw_weights)
                    fused_rank_train = np.dot(subset_rank_train, weights)
                    fused_rank_test = np.dot(subset_rank_test, weights)
                    # Lower rank value means stronger score; invert before calibration.
                    rank_signal_train = -np.asarray(fused_rank_train, dtype=float).reshape(-1)
                    rank_signal_test = -np.asarray(fused_rank_test, dtype=float).reshape(-1)
                    slope, intercept, _y_fill = _cfa_fit_rank_calibration(
                        rank_signal_train,
                        y_train_arr,
                        epsilon=float(epsilon),
                    )
                    fused_train = slope * rank_signal_train + intercept
                    fused_test = slope * rank_signal_test + intercept
                    metric_value = _cfa_metric_value(y_train_arr, fused_train, metric=optimize_metric)
                    adjusted_metric = float(metric_value)
                    if bool(rank_prefer_when_diverse) and float(subset_diversity_mean) >= float(rank_diversity_threshold):
                        adjusted_metric = float(metric_value) * float(rank_metric_discount)
                    candidates.append(
                        {
                            "variant": str(variant_name),
                            "combination_space": "rank",
                            "n_models": int(subset_size),
                            "models": "|".join(subset_models),
                            "train_metric": float(metric_value),
                            "adjusted_metric": float(adjusted_metric),
                            "subset_diversity_mean": float(subset_diversity_mean),
                            "rank_calibration_slope": float(slope),
                            "rank_calibration_intercept": float(intercept),
                            "rank_descending": True,
                            "weights": {name: float(weight) for name, weight in zip(subset_models, weights)},
                            "pred_train": np.asarray(fused_train, dtype=float).reshape(-1),
                            "pred_test": np.asarray(fused_test, dtype=float).reshape(-1),
                        }
                    )

    if not candidates:
        raise RuntimeError("CFA fusion candidate generation returned no combinations.")

    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values(
        by=["adjusted_metric", "train_metric", "n_models", "combination_space", "variant", "models"],
        ascending=[True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    best = dict(candidates_df.iloc[0].to_dict())
    best_models = str(best["models"]).split("|")
    weights_map = dict(best["weights"])
    best_train = np.asarray(best["pred_train"], dtype=float).reshape(-1)
    best_test = np.asarray(best["pred_test"], dtype=float).reshape(-1)
    # Drop large arrays from candidate table to keep artifacts compact.
    candidates_df = candidates_df.drop(columns=["pred_train", "pred_test"], errors="ignore")

    return {
        "pred_train": np.asarray(best_train, dtype=float).reshape(-1),
        "pred_test": np.asarray(best_test, dtype=float).reshape(-1),
        "selected_models": list(best_models),
        "variant": str(best["variant"]),
        "combination_space": str(best.get("combination_space", "score")),
        "optimize_metric": str(optimize_metric),
        "train_metric": float(best["train_metric"]),
        "adjusted_metric": float(best.get("adjusted_metric", best["train_metric"])),
        "subset_diversity_mean": float(best.get("subset_diversity_mean", np.nan)),
        "rank_calibration_slope": float(best.get("rank_calibration_slope", np.nan)),
        "rank_calibration_intercept": float(best.get("rank_calibration_intercept", np.nan)),
        "rank_descending": bool(best.get("rank_descending", True)) if str(best.get("combination_space", "score")) == "rank" else np.nan,
        "weights": {name: float(weights_map[name]) for name in best_models},
        "candidate_table": candidates_df.copy(),
        "base_strengths": pd.DataFrame(
            {
                "model": common_models,
                "train_metric": base_train_metric,
                "performance_strength": performance_strength,
                "diversity_strength": diversity_strength,
            }
        ),
    }


def cfa_candidate_subset_count(
    n_models: int,
    min_models: int = 2,
    max_models: int | None = None,
) -> int:
    n_models = int(max(0, n_models))
    if n_models <= 0:
        return 0
    min_models = int(max(1, min_models))
    if max_models is None:
        max_models = n_models
    max_models = int(max(min_models, min(max_models, n_models)))
    total = 0
    for subset_size in range(min_models, max_models + 1):
        total += int(math.comb(n_models, subset_size))
    return int(total)


def resolve_cfa_max_models_for_budget(
    n_models: int,
    *,
    min_models: int = 2,
    requested_max_models: int | None = None,
    max_candidate_subsets: int | None = None,
) -> tuple[int, int]:
    """Resolve the largest max_models that satisfies subset budget constraints.

    Returns:
        (effective_max_models, resulting_subset_count)
    """
    n_models = int(max(0, n_models))
    if n_models <= 0:
        return 0, 0
    min_models = int(max(1, min_models))
    requested = int(requested_max_models) if requested_max_models is not None else n_models
    requested = int(max(min_models, min(requested, n_models)))
    if max_candidate_subsets is None or int(max_candidate_subsets) <= 0:
        count = cfa_candidate_subset_count(n_models, min_models=min_models, max_models=requested)
        return int(requested), int(count)
    budget = int(max_candidate_subsets)
    effective = int(requested)
    while effective >= min_models:
        count = cfa_candidate_subset_count(n_models, min_models=min_models, max_models=effective)
        if count <= budget or effective <= min_models:
            return int(effective), int(count)
        effective -= 1
    count = cfa_candidate_subset_count(n_models, min_models=min_models, max_models=min_models)
    return int(min_models), int(count)


def list_supported_chemprop_architectures() -> list[str]:
    return list(CHEMPROP_ARCHITECTURE_REGISTRY.keys())


def resolve_chemprop_architecture_specs(
    architecture_keys: list[str] | None = None,
    *,
    ensemble_size: int = 1,
    include_rdkit2d_extra: bool = False,
    include_selected_feature_variant: bool = False,
) -> list[dict[str, Any]]:
    requested = [str(key).strip().lower() for key in (architecture_keys or list_supported_chemprop_architectures())]
    requested = [key for key in requested if key]
    if not requested:
        requested = list_supported_chemprop_architectures()

    specs: list[dict[str, Any]] = []
    seen_variant_tags: set[str] = set()

    def _append_spec(architecture_key: str, *, force_rdkit2d: bool = False):
        if architecture_key not in CHEMPROP_ARCHITECTURE_REGISTRY:
            supported = ", ".join(list_supported_chemprop_architectures())
            raise ValueError(
                f"Unsupported Chemprop architecture '{architecture_key}'. Supported: {supported}."
            )
        base = CHEMPROP_ARCHITECTURE_REGISTRY[architecture_key]
        featurizers = [str(item).strip() for item in list(base.get("molecule_featurizers", [])) if str(item).strip()]
        variant_tag = str(base.get("variant_tag", architecture_key))
        display_name = str(base.get("display_name", architecture_key.upper()))
        if force_rdkit2d and "rdkit_2d" not in featurizers:
            featurizers.append("rdkit_2d")
            variant_tag = f"{variant_tag}_rdkit2d"
            display_name = f"{display_name} + RDKit2D"
        if variant_tag in seen_variant_tags:
            return
        seen_variant_tags.add(variant_tag)
        specs.append(
            {
                "architecture_key": architecture_key,
                "variant_tag": variant_tag,
                "label": f"Chemprop v2 ({display_name}, ensemble={int(ensemble_size)})",
                "workflow": str(base.get("workflow", "Chemprop v2")),
                "train_args": [str(item) for item in list(base.get("train_args", []))],
                "featurizers": featurizers,
                "notes": str(base.get("notes", "")),
            }
        )

    for key in requested:
        _append_spec(key, force_rdkit2d=False)
        if bool(include_rdkit2d_extra) and key == "dmpnn":
            _append_spec(key, force_rdkit2d=True)

    if bool(include_selected_feature_variant):
        selected_architecture_key = "dmpnn" if "dmpnn" in requested else requested[0]
        base = CHEMPROP_ARCHITECTURE_REGISTRY[selected_architecture_key]
        base_variant_tag = str(base.get("variant_tag", selected_architecture_key))
        base_display_name = str(base.get("display_name", selected_architecture_key.upper()))
        selected_variant_tag = f"{base_variant_tag}_selected_features"
        if selected_variant_tag not in seen_variant_tags:
            seen_variant_tags.add(selected_variant_tag)
            specs.append(
                {
                    "architecture_key": selected_architecture_key,
                    "variant_tag": selected_variant_tag,
                    "label": (
                        f"Chemprop v2 ({base_display_name} + Selected descriptors, "
                        f"ensemble={int(ensemble_size)})"
                    ),
                    "workflow": str(base.get("workflow", "Chemprop v2")),
                    "train_args": [str(item) for item in list(base.get("train_args", []))],
                    # This variant intentionally uses the train-only selected descriptor matrix
                    # instead of predefined molecule featurizer sets.
                    "featurizers": [],
                    "use_selected_descriptors": True,
                    "notes": (
                        "Chemprop graph encoder with train-only selected tabular descriptors "
                        "provided through descriptors-path."
                    ),
                }
            )

    return specs

MAPLIGHT_DESCRIPTOR_NAMES = [
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
    "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2",
    "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9",
    "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "FractionCSP3", "HallKierAlpha",
    "HeavyAtomCount", "HeavyAtomMolWt", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "MaxAbsEStateIndex",
    "MaxAbsPartialCharge", "MaxEStateIndex", "MaxPartialCharge", "MinAbsEStateIndex", "MinAbsPartialCharge",
    "MinEStateIndex", "MinPartialCharge", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRadicalElectrons", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
    "NumSaturatedRings", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12",
    "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "RingCount", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3",
    "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10",
    "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
    "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "TPSA", "VSA_EState1", "VSA_EState10", "VSA_EState2",
    "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8",
    "VSA_EState9", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N",
    "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN",
    "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2",
    "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid",
    "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur",
    "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide",
    "fr_ester", "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone",
    "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss",
    "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom",
    "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol",
    "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide",
    "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene",
    "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea", "qed",
]


def murcko_scaffold_key(smiles_text):
    smiles_text = str(smiles_text)
    mol = Chem.MolFromSmiles(smiles_text)
    if mol is None:
        return f"INVALID::{smiles_text}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or f"NO_SCAFFOLD::{smiles_text}"


def scaffold_train_test_split(X, y, smiles, test_size=0.2, random_state=42):
    smiles_series = pd.Series(smiles, dtype=str).reset_index(drop=True)
    y_series = pd.Series(y, dtype=float).reset_index(drop=True)
    X_frame = pd.DataFrame(X).reset_index(drop=True).copy()

    n_total = len(smiles_series)
    if n_total < 3:
        raise ValueError("Scaffold splitting requires at least 3 molecules.")

    desired_test = int(round(float(test_size) * n_total))
    desired_test = min(max(desired_test, 1), n_total - 1)

    scaffold_to_indices = {}
    for idx, smiles_value in enumerate(smiles_series):
        scaffold_to_indices.setdefault(murcko_scaffold_key(smiles_value), []).append(idx)

    rng = np.random.RandomState(int(random_state))
    grouped = list(scaffold_to_indices.items())
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: (-len(item[1]), item[0]))

    train_indices = []
    test_indices = []
    for _scaffold_key, indices in grouped:
        current_test = len(test_indices)
        proposed_test = current_test + len(indices)
        assign_to_test = current_test < desired_test and (
            abs(proposed_test - desired_test) <= abs(current_test - desired_test) or current_test == 0
        )
        if assign_to_test:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if len(test_indices) == 0 and len(train_indices) > 1:
        test_indices.append(train_indices.pop())
    if len(train_indices) == 0 and len(test_indices) > 1:
        train_indices.append(test_indices.pop())

    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)

    return (
        X_frame.iloc[train_indices].reset_index(drop=True),
        X_frame.iloc[test_indices].reset_index(drop=True),
        y_series.iloc[train_indices].reset_index(drop=True),
        y_series.iloc[test_indices].reset_index(drop=True),
        smiles_series.iloc[train_indices].reset_index(drop=True),
        smiles_series.iloc[test_indices].reset_index(drop=True),
    )


def target_quartile_labels(y, q=4):
    y_series = pd.Series(y, dtype=float).reset_index(drop=True)
    if len(y_series) < int(q) * 2:
        raise ValueError(f"Target-quartile splitting needs at least {int(q) * 2} rows; found {len(y_series)}.")
    labels = pd.qcut(y_series, q=int(q), labels=False, duplicates="drop")
    labels = labels.astype("Int64")
    if labels.isna().any() or labels.nunique(dropna=True) < 2:
        raise ValueError(
            "Target-quartile splitting could not create at least two non-empty target bins. "
            "Use the random split for very small or low-variance target data."
        )
    label_counts = labels.value_counts(dropna=True)
    if int(label_counts.min()) < 2:
        raise ValueError(
            "Target-quartile splitting requires at least two samples per target bin. "
            "Use a smaller test fraction or the random split for this dataset."
        )
    return labels.astype(int).astype(str)


class TargetQuartileStratifiedKFold:
    def __init__(self, n_splits=5, random_state=42, q=4):
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.q = int(q)

    def get_n_splits(self, X=None, y=None, groups=None):
        return int(self.n_splits)

    def split(self, X, y, groups=None):
        X_frame = pd.DataFrame(X).reset_index(drop=True)
        y_series = pd.Series(y, dtype=float).reset_index(drop=True)
        try:
            labels = target_quartile_labels(y_series, q=self.q)
            max_supported_folds = int(labels.value_counts(dropna=True).min())
            effective_folds = min(int(self.n_splits), max_supported_folds)
            if effective_folds < 2:
                raise ValueError("Target-quartile CV requires at least two samples per quartile bin in the training data.")
            splitter = StratifiedKFold(n_splits=int(effective_folds), shuffle=True, random_state=int(self.random_state))
            yield from splitter.split(X_frame, labels)
            return
        except Exception:
            # Some ADME/Tox labels are effectively binary/low-variance in a
            # given train split; fall back to random CV instead of failing.
            fallback_folds = min(int(self.n_splits), len(X_frame))
            if fallback_folds < 2:
                raise ValueError("Random CV fallback requires at least 2 samples.")
            fallback = KFold(n_splits=int(fallback_folds), shuffle=True, random_state=int(self.random_state))
            yield from fallback.split(X_frame, y_series)


def make_qsar_cv_splitter(X, y, smiles, split_strategy="random", cv_folds=5, random_seed=42):
    X_frame = pd.DataFrame(X).reset_index(drop=True)
    y_series = pd.Series(y, dtype=float).reset_index(drop=True)
    smiles_series = pd.Series(smiles, dtype=str).reset_index(drop=True)
    requested_folds = min(int(cv_folds), len(X_frame))
    if requested_folds < 2:
        raise ValueError("At least 2 CV folds are required.")

    split_strategy = str(split_strategy).strip().lower()
    if split_strategy == "random":
        splitter = KFold(n_splits=requested_folds, shuffle=True, random_state=int(random_seed))
        return splitter, int(requested_folds), "random"

    if split_strategy == "target_quartiles":
        try:
            labels = target_quartile_labels(y_series, q=4)
            max_supported_folds = int(labels.value_counts(dropna=True).min())
            effective_folds = min(requested_folds, max_supported_folds)
            if effective_folds < 2:
                raise ValueError("Target-quartile CV requires at least two samples per quartile bin in the training data.")
            splitter = StratifiedKFold(n_splits=int(effective_folds), shuffle=True, random_state=int(random_seed))
            return list(splitter.split(X_frame, labels)), int(effective_folds), "target_quartiles"
        except Exception:
            splitter = KFold(n_splits=requested_folds, shuffle=True, random_state=int(random_seed))
            return splitter, int(requested_folds), "random_fallback_from_target_quartiles"

    if split_strategy == "scaffold":
        groups = smiles_series.map(murcko_scaffold_key)
        unique_groups = int(pd.Series(groups).nunique(dropna=False))
        effective_folds = min(requested_folds, unique_groups)
        if effective_folds < 2:
            raise ValueError("Scaffold CV requires at least two distinct Murcko scaffold groups in the training data.")
        splitter = GroupKFold(n_splits=int(effective_folds))
        return list(splitter.split(X_frame, y_series, groups)), int(effective_folds), "scaffold"

    raise ValueError("CV split strategy must be 'random', 'target_quartiles', or 'scaffold'.")


def make_reusable_inner_cv_splitter(split_strategy="random", cv_folds=5, random_seed=42):
    requested_folds = int(cv_folds)
    if requested_folds < 2:
        raise ValueError("At least 2 CV folds are required.")

    split_strategy = str(split_strategy).strip().lower()
    if split_strategy == "random":
        return KFold(n_splits=requested_folds, shuffle=True, random_state=int(random_seed)), requested_folds, "random"
    if split_strategy == "target_quartiles":
        return TargetQuartileStratifiedKFold(n_splits=requested_folds, random_state=int(random_seed), q=4), requested_folds, "target_quartiles"
    if split_strategy == "scaffold":
        fallback = KFold(n_splits=requested_folds, shuffle=True, random_state=int(random_seed))
        return fallback, requested_folds, "random_inner_fallback_for_scaffold"
    raise ValueError("CV split strategy must be 'random', 'target_quartiles', or 'scaffold'.")


def make_morgan_matrix(smiles_list, radius=2, n_bits=1024):
    generator = AllChem.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"morgan_bit_{i:04d}" for i in range(n_bits)])


def make_circular_fingerprint_matrix(smiles_list, radius=3, n_bits=1024, use_features=False, prefix="ecfp6"):
    generator = AllChem.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
        atomInvariantsGenerator=(AllChem.GetMorganFeatureAtomInvGen() if bool(use_features) else None),
    )
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"{prefix}_bit_{i:04d}" for i in range(n_bits)])


def make_ecfp6_matrix(smiles_list, n_bits=1024):
    return make_circular_fingerprint_matrix(smiles_list, radius=3, n_bits=int(n_bits), use_features=False, prefix="ecfp6")


def make_fcfp6_matrix(smiles_list, n_bits=1024):
    return make_circular_fingerprint_matrix(smiles_list, radius=3, n_bits=int(n_bits), use_features=True, prefix="fcfp6")


def make_layered_matrix(smiles_list, n_bits=1024):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = Chem.LayeredFingerprint(mol, fpSize=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"layered_bit_{i:04d}" for i in range(int(n_bits))])


def make_atom_pair_matrix(smiles_list, n_bits=1024):
    atom_pair_generator = None
    try:
        atom_pair_generator = AllChem.GetAtomPairGenerator(fpSize=int(n_bits))
    except Exception:
        atom_pair_generator = None
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if atom_pair_generator is not None:
            fp = atom_pair_generator.GetFingerprint(mol)
        else:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"atom_pair_bit_{i:04d}" for i in range(int(n_bits))])


def make_topological_torsion_matrix(smiles_list, n_bits=1024):
    torsion_generator = None
    try:
        torsion_generator = AllChem.GetTopologicalTorsionGenerator(fpSize=int(n_bits))
    except Exception:
        torsion_generator = None
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if torsion_generator is not None:
            fp = torsion_generator.GetFingerprint(mol)
        else:
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"topological_torsion_bit_{i:04d}" for i in range(int(n_bits))])


def make_rdk_path_matrix(smiles_list, n_bits=1024, min_path=1, max_path=7):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = Chem.RDKFingerprint(
            mol,
            fpSize=int(n_bits),
            minPath=int(min_path),
            maxPath=int(max_path),
        )
        arr = np.zeros((int(n_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"rdk_path_bit_{i:04d}" for i in range(int(n_bits))])


def make_maccs_matrix(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return pd.DataFrame(rows, columns=[f"maccs_bit_{i:03d}" for i in range(fp.GetNumBits())])


def make_rdkit_descriptor_matrix(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rows.append({f"rdkit_{name}": func(mol) for name, func in Descriptors._descList})
    return pd.DataFrame(rows)


def make_maplight_morgan_count_matrix(smiles_list, radius=2, n_bits=1024):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros((int(n_bits),), dtype=np.float32))
            continue
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=int(radius), nBits=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.int32)
        for bit_id, count in fp.GetNonzeroElements().items():
            bit_id = int(bit_id)
            if 0 <= bit_id < int(n_bits):
                arr[bit_id] = int(count)
        rows.append(arr.astype(np.float32))
    return pd.DataFrame(rows, columns=[f"maplight_morgan_{i:04d}" for i in range(int(n_bits))])


def make_avalon_count_matrix(smiles_list, n_bits=1024):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros((int(n_bits),), dtype=np.float32))
            continue
        fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=int(n_bits))
        arr = np.zeros((int(n_bits),), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr.astype(np.float32))
    return pd.DataFrame(rows, columns=[f"avalon_count_{i:04d}" for i in range(int(n_bits))])


def make_erg_matrix(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        erg = rdReducedGraphs.GetErGFingerprint(mol)
        rows.append(np.asarray(erg, dtype=np.float32))
    arr = np.vstack(rows)
    return pd.DataFrame(arr, columns=[f"erg_{i:03d}" for i in range(arr.shape[1])])


def make_maplight_descriptor_matrix(smiles_list):
    calculator = MolecularDescriptorCalculator(MAPLIGHT_DESCRIPTOR_NAMES)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rows.append(np.asarray(calculator.CalcDescriptors(mol), dtype=np.float32))
    arr = np.vstack(rows)
    return pd.DataFrame(arr, columns=[f"maplight_desc_{name}" for name in MAPLIGHT_DESCRIPTOR_NAMES])


def make_maplight_classic_matrix(smiles_list, radius=2, n_bits=1024):
    parts = [
        make_maplight_morgan_count_matrix(smiles_list, radius=radius, n_bits=n_bits),
        make_avalon_count_matrix(smiles_list, n_bits=n_bits),
        make_erg_matrix(smiles_list),
        make_maplight_descriptor_matrix(smiles_list),
    ]
    return pd.concat(parts, axis=1)


def finalize_feature_matrix(feature_df):
    feature_df = feature_df.copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    float32_limit = np.finfo(np.float32).max
    feature_df = feature_df.mask(feature_df.abs() > float32_limit, np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.fillna(0.0)
    return feature_df.astype(np.float32)


def align_feature_matrix_to_training_columns(feature_df, expected_columns):
    aligned = feature_df.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned = aligned.loc[:, list(expected_columns)].copy()
    return finalize_feature_matrix(aligned)


def drop_exact_and_near_duplicate_features(
    X_train,
    X_other=None,
    *,
    correlation_threshold=0.999,
    moderate_correlation_threshold=0.995,
    variance_threshold=1e-8,
    binary_prevalence_min=0.005,
    binary_prevalence_max=0.995,
    binary_value_tolerance=1e-6,
    report_limit=20,
):
    """Apply cheap pre-filters and exact-duplicate pruning.

    Filtering order:
    1. Near-zero variance columns
    2. Binary prevalence outliers (rare / near-constant bits)
    3. Exact duplicates (identical values across all training rows)

    Parameters
    ----------
    X_train : DataFrame-like
        Training feature matrix used to detect redundant columns.
    X_other : DataFrame-like or None
        Optional second matrix (for example test features) that will be aligned to
        the retained training columns.
    correlation_threshold : float
        Retained for backward compatibility. Correlation pruning is disabled.
    moderate_correlation_threshold : float
        Retained for backward compatibility. Correlation pruning is disabled.
    variance_threshold : float
        Columns with training variance <= this threshold are removed.
    binary_prevalence_min : float
        For binary columns, minimum prevalence of the positive class to keep.
    binary_prevalence_max : float
        For binary columns, maximum prevalence of the positive class to keep.
    binary_value_tolerance : float
        Tolerance used to detect binary columns near {0, 1}.
    report_limit : int
        Maximum number of representative duplicate pairs to keep in metadata.
    """

    train_df = pd.DataFrame(X_train).copy()
    other_df = pd.DataFrame(X_other).copy() if X_other is not None else None

    original_columns = list(train_df.columns)
    dropped_low_variance_columns = []
    dropped_binary_prevalence_columns = []
    dropped_binary_prevalence_examples = []

    def _drop_from_both(columns_to_drop):
        nonlocal train_df, other_df
        if not columns_to_drop:
            return
        train_df = train_df.drop(columns=columns_to_drop, errors="ignore")
        if other_df is not None:
            drop_in_other = [column for column in columns_to_drop if column in other_df.columns]
            if drop_in_other:
                other_df = other_df.drop(columns=drop_in_other, errors="ignore")

    variance_cutoff = float(variance_threshold)
    if train_df.shape[1] > 0 and np.isfinite(variance_cutoff) and variance_cutoff >= 0.0:
        try:
            variance_series = (
                train_df.var(axis=0, ddof=0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            dropped_low_variance_columns = (
                variance_series[variance_series <= variance_cutoff]
                .index.astype(str)
                .tolist()
            )
            _drop_from_both(dropped_low_variance_columns)
        except Exception:
            dropped_low_variance_columns = []

    prevalence_min = float(binary_prevalence_min)
    prevalence_max = float(binary_prevalence_max)
    binary_tol = float(binary_value_tolerance)
    binary_prevalence_filter_enabled = (
        train_df.shape[1] > 0
        and np.isfinite(prevalence_min)
        and np.isfinite(prevalence_max)
        and 0.0 <= prevalence_min < prevalence_max <= 1.0
    )
    if binary_prevalence_filter_enabled:
        try:
            train_values = train_df.to_numpy(dtype=np.float64, copy=True)
            if train_values.size > 0:
                train_values[~np.isfinite(train_values)] = np.nan
                is_zero = np.isclose(train_values, 0.0, atol=binary_tol, rtol=0.0)
                is_one = np.isclose(train_values, 1.0, atol=binary_tol, rtol=0.0)
                finite_mask = np.isfinite(train_values)
                is_binary_entry = finite_mask & (is_zero | is_one)
                is_binary_column = np.all(is_binary_entry, axis=0)
                if np.any(is_binary_column):
                    binary_indices = np.flatnonzero(is_binary_column)
                    binary_prevalence = np.mean(is_one[:, binary_indices], axis=0)
                    prevalence_drop_mask = (
                        (binary_prevalence < prevalence_min)
                        | (binary_prevalence > prevalence_max)
                    )
                    if np.any(prevalence_drop_mask):
                        drop_binary_indices = binary_indices[np.flatnonzero(prevalence_drop_mask)]
                        drop_binary_columns = train_df.columns[drop_binary_indices].astype(str).tolist()
                        dropped_binary_prevalence_columns = list(drop_binary_columns)
                        for idx, column_name in enumerate(drop_binary_columns):
                            if idx >= int(report_limit):
                                break
                            prevalence_value = float(binary_prevalence[np.flatnonzero(prevalence_drop_mask)[idx]])
                            dropped_binary_prevalence_examples.append(
                                {
                                    "feature": str(column_name),
                                    "prevalence": prevalence_value,
                                }
                            )
                        _drop_from_both(dropped_binary_prevalence_columns)
        except Exception:
            dropped_binary_prevalence_columns = []
            dropped_binary_prevalence_examples = []

    dropped_exact_columns = []
    dropped_exact_pairs = []
    hash_buckets = {}
    current_columns = list(train_df.columns)
    for column in current_columns:
        series = train_df[column]
        series_hash = int(pd.util.hash_pandas_object(series, index=False).sum())
        bucket = hash_buckets.setdefault(series_hash, [])
        duplicate_of = None
        for prior_column in bucket:
            if series.equals(train_df[prior_column]):
                duplicate_of = prior_column
                break
        if duplicate_of is not None:
            dropped_exact_columns.append(column)
            if len(dropped_exact_pairs) < int(report_limit):
                dropped_exact_pairs.append((str(duplicate_of), str(column)))
        else:
            bucket.append(column)

    if dropped_exact_columns:
        _drop_from_both(dropped_exact_columns)

    dropped_near_columns = []
    dropped_near_pairs = []
    dropped_moderate_columns = []
    dropped_moderate_pairs = []
    near_duplicate_scan_error = ""
    threshold = float(correlation_threshold)
    moderate_threshold = float(moderate_correlation_threshold)

    total_dropped = (
        len(dropped_low_variance_columns)
        + len(dropped_binary_prevalence_columns)
        + len(dropped_exact_columns)
        + len(dropped_near_columns)
        + len(dropped_moderate_columns)
    )
    dedup_metadata = {
        "correlation_threshold": float(threshold),
        "moderate_correlation_threshold": np.nan,
        "variance_threshold": float(variance_cutoff),
        "binary_prevalence_min": float(prevalence_min) if np.isfinite(prevalence_min) else np.nan,
        "binary_prevalence_max": float(prevalence_max) if np.isfinite(prevalence_max) else np.nan,
        "binary_value_tolerance": float(binary_tol),
        "original_feature_count": int(len(original_columns)),
        "post_dedup_feature_count": int(train_df.shape[1]),
        "dropped_feature_count": int(total_dropped),
        "dropped_low_variance_count": int(len(dropped_low_variance_columns)),
        "dropped_binary_prevalence_count": int(len(dropped_binary_prevalence_columns)),
        "dropped_exact_count": int(len(dropped_exact_columns)),
        "dropped_near_count": int(len(dropped_near_columns)),
        "dropped_moderate_count": int(len(dropped_moderate_columns)),
        "dropped_low_variance_columns": [str(column) for column in dropped_low_variance_columns],
        "dropped_binary_prevalence_columns": [str(column) for column in dropped_binary_prevalence_columns],
        "dropped_exact_columns": [str(column) for column in dropped_exact_columns],
        "dropped_near_columns": [str(column) for column in dropped_near_columns],
        "dropped_moderate_columns": [str(column) for column in dropped_moderate_columns],
        "binary_prevalence_examples": list(dropped_binary_prevalence_examples),
        "exact_duplicate_examples": [
            {"kept": str(kept), "dropped": str(dropped)} for kept, dropped in dropped_exact_pairs
        ],
        "near_duplicate_examples": [
            {"kept": str(kept), "dropped": str(dropped)} for kept, dropped in dropped_near_pairs
        ],
        "moderate_duplicate_examples": [
            {"kept": str(kept), "dropped": str(dropped)} for kept, dropped in dropped_moderate_pairs
        ],
        "near_duplicate_scan_error": near_duplicate_scan_error or "",
    }
    return train_df.reset_index(drop=True), (other_df.reset_index(drop=True) if other_df is not None else None), dedup_metadata


def normalize_selected_feature_families(selected_feature_families=None, **feature_flags):
    selected = []
    family_aliases = {
        "layeredfp": "layered",
        "rdk": "rdk_path",
        "rdkfp": "rdk_path",
        "atompair": "atom_pair",
        "atom_pair": "atom_pair",
        "topologicaltorsion": "topological_torsion",
        "topo_torsion": "topological_torsion",
        "tt": "topological_torsion",
    }
    if selected_feature_families is not None:
        for family in selected_feature_families:
            family_key = str(family).strip().lower()
            family_key = family_aliases.get(family_key, family_key)
            if family_key in FEATURE_FAMILY_LABELS and family_key not in selected:
                selected.append(family_key)
    flag_mapping = {
        "use_morgan_features": "morgan",
        "use_ecfp6_features": "ecfp6",
        "use_fcfp6_features": "fcfp6",
        "use_layered_features": "layered",
        "use_atom_pair_features": "atom_pair",
        "use_topological_torsion_features": "topological_torsion",
        "use_rdk_path_features": "rdk_path",
        "use_maccs_keys": "maccs",
        "use_rdkit_descriptors": "rdkit",
        "use_maplight_classic": "maplight",
    }
    for flag_name, family_key in flag_mapping.items():
        if bool(feature_flags.get(flag_name)) and family_key not in selected:
            selected.append(family_key)
    return selected


def feature_family_label(family_key, radius=2, n_bits=1024):
    family_key = str(family_key).strip().lower()
    if family_key == "morgan":
        return f"Morgan(radius={int(radius)}, bits={int(n_bits)})"
    if family_key == "ecfp6":
        return f"ECFP6(bits={int(n_bits)})"
    if family_key == "fcfp6":
        return f"FCFP6(bits={int(n_bits)})"
    if family_key == "layered":
        return f"RDKit layered(bits={int(n_bits)})"
    if family_key == "atom_pair":
        return f"Atom-pair(bits={int(n_bits)})"
    if family_key == "topological_torsion":
        return f"Topological torsion(bits={int(n_bits)})"
    if family_key == "rdk_path":
        return f"RDKit path(bits={int(n_bits)})"
    if family_key == "maplight":
        return f"MapLight classic (Morgan+Avalon+ErG, bits={int(n_bits)})"
    return FEATURE_FAMILY_LABELS[family_key]


def default_feature_store_path():
    return Path("./model_cache/feature_store_parquet")


def resolve_feature_store_path(feature_store_path="AUTO"):
    path_text = str(feature_store_path).strip()
    if not path_text or path_text.upper() == "AUTO":
        return default_feature_store_path()
    return Path(path_text)


def feature_store_representation_payload(selected_families, radius=2, n_bits=1024):
    selected_families = [str(family).strip().lower() for family in selected_families]
    return {
        "families": selected_families,
        "morgan_radius": int(radius) if any(family in {"morgan", "maplight"} for family in selected_families) else None,
        "fingerprint_bits": int(n_bits)
        if any(
            family in {"morgan", "ecfp6", "fcfp6", "layered", "atom_pair", "topological_torsion", "rdk_path", "maplight"}
            for family in selected_families
        )
        else None,
    }


def feature_store_representation_key(selected_families, radius=2, n_bits=1024):
    payload = feature_store_representation_payload(selected_families, radius=radius, n_bits=n_bits)
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]
    family_prefix = "_".join(payload["families"]) or "none"
    return f"{family_prefix}__{payload_hash}"


def ensure_feature_store(feature_store_path):
    feature_store_path = Path(feature_store_path)
    feature_store_path.mkdir(parents=True, exist_ok=True)
    return feature_store_path


def feature_store_representation_dir(feature_store_path, representation_key):
    return Path(feature_store_path) / representation_key


def feature_store_representation_schema_path(feature_store_path, representation_key):
    return feature_store_representation_dir(feature_store_path, representation_key) / "_schema.json"


def feature_store_representation_shard_dir(feature_store_path, representation_key):
    return feature_store_representation_dir(feature_store_path, representation_key) / "shards"


def write_feature_store_schema(feature_store_path, representation_key, representation_payload, columns):
    representation_dir = feature_store_representation_dir(feature_store_path, representation_key)
    representation_dir.mkdir(parents=True, exist_ok=True)
    feature_store_representation_shard_dir(feature_store_path, representation_key).mkdir(parents=True, exist_ok=True)
    schema_path = feature_store_representation_schema_path(feature_store_path, representation_key)
    schema_payload = {
        "representation_key": representation_key,
        "representation_payload": representation_payload,
        "columns": list(columns),
    }
    if schema_path.exists():
        existing = json.loads(schema_path.read_text(encoding="utf-8"))
        if list(existing.get("columns", [])) != list(columns) or dict(existing.get("representation_payload", {})) != dict(representation_payload):
            raise RuntimeError(
                "The molecular feature store already has a different schema for this representation. "
                "Use a new feature-store path if you need to rebuild that representation definition."
            )
        return
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")


def feature_store_shard_paths(feature_store_path, representation_key):
    shard_dir = feature_store_representation_shard_dir(feature_store_path, representation_key)
    if not shard_dir.exists():
        return []
    pattern = "*.parquet" if FEATURE_STORE_SHARD_FORMAT == "parquet" else "*.csv.gz"
    return sorted(shard_dir.glob(pattern))


def read_feature_store_shard(shard_path):
    if str(shard_path).lower().endswith(".parquet"):
        return pd.read_parquet(shard_path)
    return pd.read_csv(shard_path)


def write_feature_store_shard(shard_df, shard_path):
    if str(shard_path).lower().endswith(".parquet"):
        shard_df.to_parquet(shard_path, index=False)
    else:
        shard_df.to_csv(shard_path, index=False, compression="gzip")


def load_feature_store_rows(feature_store_path, representation_key, smiles_list, expected_columns):
    smiles_list = [str(smiles) for smiles in smiles_list]
    if not smiles_list:
        return pd.DataFrame(columns=list(expected_columns))
    shard_paths = feature_store_shard_paths(feature_store_path, representation_key)
    if not shard_paths:
        return pd.DataFrame(columns=list(expected_columns))
    requested_smiles = set(smiles_list)
    loaded_frames = []
    for shard_path in shard_paths:
        shard_df = read_feature_store_shard(shard_path)
        if "canonical_smiles" not in shard_df.columns:
            continue
        shard_df["canonical_smiles"] = shard_df["canonical_smiles"].astype(str)
        shard_df = shard_df.loc[shard_df["canonical_smiles"].isin(requested_smiles)].copy()
        if shard_df.empty:
            continue
        for column in expected_columns:
            if column not in shard_df.columns:
                shard_df[column] = np.nan
        loaded_frames.append(shard_df.loc[:, ["canonical_smiles", *list(expected_columns)]])
    if not loaded_frames:
        return pd.DataFrame(columns=list(expected_columns))
    combined = pd.concat(loaded_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["canonical_smiles"], keep="first")
    combined = combined.set_index("canonical_smiles")
    combined = combined.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    ordered_rows = [combined.loc[str(smiles)].to_numpy(dtype=np.float32) for smiles in smiles_list if str(smiles) in combined.index]
    ordered_index = [str(smiles) for smiles in smiles_list if str(smiles) in combined.index]
    return pd.DataFrame(ordered_rows, columns=list(expected_columns), index=ordered_index, dtype=np.float32)


def append_feature_store_rows(feature_store_path, representation_key, feature_df, smiles_list):
    feature_df = feature_df.copy()
    feature_df = feature_df.loc[:, list(feature_df.columns)].astype(np.float32)
    smiles_list = [str(smiles) for smiles in smiles_list]
    if feature_df.empty or not smiles_list:
        return 0
    shard_dir = feature_store_representation_shard_dir(feature_store_path, representation_key)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_df = feature_df.reset_index(drop=True).copy()
    shard_df.insert(0, "canonical_smiles", smiles_list)
    suffix = ".parquet" if FEATURE_STORE_SHARD_FORMAT == "parquet" else ".csv.gz"
    shard_path = shard_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    write_feature_store_shard(shard_df, shard_path)
    return int(len(shard_df))


def build_feature_family_frames(smiles_list, selected_families, radius=2, n_bits=1024):
    frames = []
    labels = []
    built_families = []
    skipped_families = []
    warnings_list = []
    for family_key in selected_families:
        if family_key == "morgan":
            frames.append(make_morgan_matrix(smiles_list, radius=int(radius), n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "maplight":
            frames.append(make_maplight_classic_matrix(smiles_list, radius=int(radius), n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "ecfp6":
            frames.append(make_ecfp6_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "fcfp6":
            frames.append(make_fcfp6_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "layered":
            frames.append(make_layered_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "atom_pair":
            frames.append(make_atom_pair_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "topological_torsion":
            frames.append(make_topological_torsion_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "rdk_path":
            frames.append(make_rdk_path_matrix(smiles_list, n_bits=int(n_bits)))
            labels.append(feature_family_label(family_key, radius=radius, n_bits=n_bits))
            built_families.append(family_key)
        elif family_key == "maccs":
            frames.append(make_maccs_matrix(smiles_list))
            labels.append(feature_family_label(family_key))
            built_families.append(family_key)
        elif family_key == "rdkit":
            frames.append(make_rdkit_descriptor_matrix(smiles_list))
            labels.append(feature_family_label(family_key))
            built_families.append(family_key)
        else:
            raise ValueError(f"Unsupported feature family: {family_key}")
    return frames, labels, built_families, skipped_families, warnings_list


def build_feature_matrix_from_smiles(
    smiles_list,
    selected_feature_families=None,
    radius=2,
    n_bits=1024,
    enable_persistent_feature_store=True,
    reuse_persistent_feature_store=True,
    persistent_feature_store_path="AUTO",
    **feature_flags,
):
    selected_families = normalize_selected_feature_families(
        selected_feature_families=selected_feature_families,
        **feature_flags,
    )
    if not selected_families and bool(feature_flags.get("use_maplight_classic")):
        selected_families = ["maplight"]
    if not selected_families:
        raise ValueError("Please select at least one molecular feature family.")
    smiles_list = [str(smiles) for smiles in smiles_list]
    representation_payload = feature_store_representation_payload(selected_families, radius=radius, n_bits=n_bits)
    representation_key = feature_store_representation_key(selected_families, radius=radius, n_bits=n_bits)

    frames, labels, built_families, skipped_families, warnings_list = build_feature_family_frames(
        smiles_list[:1],
        selected_families,
        radius=radius,
        n_bits=n_bits,
    )
    schema_feature_df = finalize_feature_matrix(pd.concat(frames, axis=1))
    expected_columns = list(schema_feature_df.columns)

    cached_feature_df = pd.DataFrame(columns=expected_columns)
    missing_smiles = list(smiles_list)
    feature_store_path_resolved = None
    if bool(enable_persistent_feature_store):
        feature_store_path_resolved = ensure_feature_store(resolve_feature_store_path(persistent_feature_store_path))
        write_feature_store_schema(
            feature_store_path_resolved,
            representation_key,
            representation_payload,
            expected_columns,
        )
        if bool(reuse_persistent_feature_store):
            cached_feature_df = load_feature_store_rows(
                feature_store_path_resolved,
                representation_key,
                smiles_list,
                expected_columns,
            )
            missing_smiles = [smiles for smiles in smiles_list if smiles not in set(cached_feature_df.index.astype(str))]

    generated_feature_df = pd.DataFrame(columns=expected_columns)
    if missing_smiles:
        frames, labels, built_families, skipped_families, warnings_list = build_feature_family_frames(
            missing_smiles,
            selected_families,
            radius=radius,
            n_bits=n_bits,
        )
        if not frames:
            raise RuntimeError("None of the requested molecular feature families could be built.")
        generated_feature_df = finalize_feature_matrix(pd.concat(frames, axis=1))
        generated_feature_df = generated_feature_df.loc[:, expected_columns].copy()
        generated_feature_df.index = pd.Index(missing_smiles, dtype=str)
        if feature_store_path_resolved is not None:
            append_feature_store_rows(
                feature_store_path_resolved,
                representation_key,
                generated_feature_df,
                missing_smiles,
            )

    cached_row_map = (
        {
            str(index): np.asarray(row.to_numpy(dtype=np.float32), dtype=np.float32)
            for index, row in cached_feature_df.groupby(level=0).first().iterrows()
        }
        if not cached_feature_df.empty
        else {}
    )
    generated_row_map = (
        {
            str(index): np.asarray(row.to_numpy(dtype=np.float32), dtype=np.float32)
            for index, row in generated_feature_df.groupby(level=0).first().iterrows()
        }
        if not generated_feature_df.empty
        else {}
    )
    combined_rows = []
    for smiles_text in smiles_list:
        if smiles_text in cached_row_map:
            combined_rows.append(cached_row_map[smiles_text])
        elif smiles_text in generated_row_map:
            combined_rows.append(generated_row_map[smiles_text])
        else:
            raise RuntimeError(f"Feature generation did not return a row for canonical SMILES: {smiles_text}")
    combined = pd.DataFrame(combined_rows, columns=expected_columns, dtype=np.float32)
    build_info = {
        "selected_feature_families": list(selected_families),
        "built_feature_families": list(built_families),
        "skipped_feature_families": list(skipped_families),
        "warnings": list(warnings_list),
        "representation_label": " + ".join(labels),
        "representation_key": representation_key,
        "feature_store_path": str(feature_store_path_resolved) if feature_store_path_resolved is not None else "",
        "feature_store_shard_format": FEATURE_STORE_SHARD_FORMAT,
        "cached_rows_loaded": int(len(smiles_list) - len(missing_smiles)),
        "generated_rows_added": int(len(missing_smiles)),
    }
    return combined, build_info
