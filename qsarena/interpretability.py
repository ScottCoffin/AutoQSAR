"""
qsarena.interpretability — per-model feature attribution (OECD (Q)SAR principle 5).

The manuscript's feature-family enrichment (Fig. 4) is the aggregate story across datasets. This
module adds per-model detail for models that act on an explicit feature vector:

  feature_importance   native importances (tree ``feature_importances_`` or linear ``coef_``),
                       or held-out permutation importance; always normalised to sum to 1.
  attribution_status   whether a model family HAS per-feature attributions. Graph and 3D
                       models (Chemprop, Uni-Mol, MapLight + GNN embeddings) learn their own
                       representation, so they are reported as "no per-feature attribution"
                       rather than given a misleading importance vector.

SHAP is deliberately not a hard dependency; tree SHAP values and native gain importances rank the
top features of a random forest near-identically, and ``permutation`` is model-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "NO_ATTRIBUTION",
    "attribution_status",
    "feature_importance",
    "top_features",
]

NO_ATTRIBUTION = "no per-feature attribution"

#: Substrings (lower-cased) of model names whose inputs are not an explicit feature vector.
_NO_ATTRIBUTION_MARKERS = ("chemprop", "uni-mol", "unimol", "gnn", "d-mpnn", "attentivefp", "cmpnn")
#: Ensembles and fusion combine other models' predictions, not features.
_META_MARKERS = ("ensemble", "cfa", "combinatorial fusion", "stacking")


def attribution_status(model_name: str) -> str:
    """Return ``"feature"``, ``"meta-model"`` or :data:`NO_ATTRIBUTION` for a model name."""
    name = str(model_name).lower()
    # Graph/3D markers first: Chemprop variant names contain "ensemble=3" (a seed ensemble of one
    # architecture), which is not a meta-model over other families.
    if any(m in name for m in _NO_ATTRIBUTION_MARKERS):
        return NO_ATTRIBUTION
    if any(m in name for m in _META_MARKERS):
        return "meta-model"
    return "feature"


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.nan_to_num(np.asarray(values, dtype=float)), 0.0, None)
    total = values.sum()
    return values / total if total > 0 else values


def feature_importance(
    model,
    feature_names: list[str],
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    method: str = "auto",
    n_repeats: int = 5,
    random_state: int = 0,
    scoring: str | None = None,
) -> pd.DataFrame:
    """Normalised feature importances for a fitted estimator.

    ``method`` is ``"native"``, ``"permutation"`` or ``"auto"`` (native if the estimator exposes
    it, otherwise permutation). Permutation importance needs held-out ``X``/``y``; negative mean
    importances (permuting helped) are clipped to zero before normalising. Returns a frame with
    columns ``feature``, ``importance`` (sums to 1 unless every value is zero), ``method``,
    sorted descending.
    """
    names = list(feature_names)
    if method not in {"auto", "native", "permutation"}:
        raise ValueError(f"unknown method {method!r}")

    raw = None
    used = method
    if method in {"auto", "native"}:
        if hasattr(model, "feature_importances_"):
            raw = np.asarray(model.feature_importances_, dtype=float)
            used = "native"
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            raw = np.abs(coef).reshape(-1, len(names)).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            used = "native"
        elif method == "native":
            raise ValueError(f"{type(model).__name__} has no native feature importances")
    if raw is None:
        if X is None or y is None:
            raise ValueError("permutation importance needs held-out X and y")
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=1
        )
        raw = result.importances_mean
        used = "permutation"

    if raw.shape[0] != len(names):
        raise ValueError(f"model has {raw.shape[0]} importances but {len(names)} feature names were given")
    frame = pd.DataFrame({"feature": names, "importance": _normalise(raw), "method": used})
    return frame.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)


def top_features(importances: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """First ``n`` rows with a 1-based ``rank`` column."""
    out = importances.head(n).copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
