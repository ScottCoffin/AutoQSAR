#!/usr/bin/env python
"""Benchmark the conventional AutoQSAR workflow.

This script benchmarks the notebook's example SMILES-based datasets: runnable
ChemML bundled examples plus the PyTDC QSAR benchmark datasets. It builds
molecular features, runs the train/test split plus train-only ElasticNetCV
feature selection, evaluates conventional models, optionally runs a small GA
tuning pass for selected models, and writes cross-dataset performance tables.
It intentionally does not run deep-learning models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
import tempfile
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

try:
    from portable_colab_qsar_bundle.qsar_workflow_core import (
        build_feature_matrix_from_smiles,
        make_qsar_cv_splitter,
        make_reusable_inner_cv_splitter,
        scaffold_train_test_split,
        target_quartile_labels,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portable_colab_qsar_bundle.qsar_workflow_core import (
        build_feature_matrix_from_smiles,
        make_qsar_cv_splitter,
        make_reusable_inner_cv_splitter,
        scaffold_train_test_split,
        target_quartile_labels,
    )

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


SMILES_CANDIDATES = ["QSAR_READY_SMILES", "canonical_smiles", "SMILES", "smiles", "Smiles"]
TARGET_CANDIDATES = ["TARGET", "target", "Target", "Repro/Dev", "Non-Repro/Dev", "density_Kg/m3"]
CHEMML_EXAMPLE_OPTIONS = ["organic_density", "cep_homo", "xyz_polarizability"]
PYTDC_SOURCE_URL = "https://files.pythonhosted.org/packages/db/bf/db7525f0e9c48d340a66ae11ed46bbb1966234660a6882ce47d1e1d52824/pytdc-1.1.15.tar.gz"
TDC_QSAR_OPTIONS = {
    "caco2_wang": {"task": "ADME", "label": "Caco-2 permeability"},
    "lipophilicity_astrazeneca": {"task": "ADME", "label": "Lipophilicity"},
    "solubility_aqsoldb": {"task": "ADME", "label": "AqSolDB solubility"},
    "ppbr_az": {"task": "ADME", "label": "Plasma protein binding"},
    "vdss_lombardo": {"task": "ADME", "label": "Volume of distribution"},
    "half_life_obach": {"task": "ADME", "label": "Half-life"},
    "clearance_hepatocyte_az": {"task": "ADME", "label": "Hepatocyte clearance"},
    "clearance_microsome_az": {"task": "ADME", "label": "Microsome clearance"},
    "ld50_zhu": {"task": "Tox", "label": "Acute toxicity LD50"},
}

CURRENT_DATASET_SPEC: "DatasetSpec | None" = None


def ensure_chemml_from_source() -> bool:
    if importlib.util.find_spec("chemml") is not None:
        return True

    search_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[1].parent,
        Path("/content"),
        Path("/content/chemml"),
        Path("/content/drive/MyDrive"),
    ]
    repo_candidates: list[Path] = []
    for root in search_roots:
        repo_candidates.extend([root, root / "chemml"])

    for candidate in repo_candidates:
        if (candidate / "chemml" / "__init__.py").exists() and (candidate / "setup.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            importlib.invalidate_caches()
            if importlib.util.find_spec("chemml") is not None:
                return True
    return False


def ensure_tdc_from_source() -> bool:
    if importlib.util.find_spec("tdc") is not None:
        return True

    def try_load_tdc_from_path(candidate_path: Path) -> bool:
        possible_roots: list[Path] = []
        if (candidate_path / "tdc" / "__init__.py").exists():
            possible_roots.append(candidate_path)
        possible_roots.extend(
            sorted(
                {
                    p.parent.parent
                    for p in candidate_path.rglob("__init__.py")
                    if p.parent.name == "tdc"
                },
                key=lambda path: len(str(path)),
            )
        )
        for root_path in possible_roots:
            if str(root_path) not in sys.path:
                sys.path.insert(0, str(root_path))
            importlib.invalidate_caches()
            if importlib.util.find_spec("tdc") is not None:
                return True
        return False

    search_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[1].parent,
        Path("/content"),
        Path("/content/chemml"),
        Path("/content/drive/MyDrive"),
    ]
    archive_candidates: list[Path] = []
    source_dir_candidates: list[Path] = []
    for root in search_roots:
        archive_candidates.extend([root / "TDC" / "pytdc-1.1.15.tar.gz", root / "pytdc-1.1.15.tar.gz"])
        source_dir_candidates.append(root / "TDC")

    for candidate in source_dir_candidates:
        if candidate.exists() and try_load_tdc_from_path(candidate):
            return True

    for candidate in archive_candidates:
        if candidate.exists():
            extract_root = Path(tempfile.mkdtemp(prefix="pytdc_src_"))
            with tarfile.open(candidate, "r:gz") as tf:
                tf.extractall(extract_root)
            if try_load_tdc_from_path(extract_root):
                return True

    download_dir = Path("/content") if str(Path.cwd()).startswith("/content") else (Path.cwd() / ".cache")
    download_dir.mkdir(parents=True, exist_ok=True)
    download_target = download_dir / "pytdc-1.1.15.tar.gz"
    if not download_target.exists():
        try:
            urllib.request.urlretrieve(PYTDC_SOURCE_URL, download_target)
        except Exception:
            return False
    extract_root = Path(tempfile.mkdtemp(prefix="pytdc_src_"))
    with tarfile.open(download_target, "r:gz") as tf:
        tf.extractall(extract_root)
    return try_load_tdc_from_path(extract_root)


@dataclass
class DatasetSpec:
    name: str
    source: str
    frame: pd.DataFrame
    smiles_column: str
    target_column: str
    recommended_split: str | None = None
    recommended_metric: str | None = None


@dataclass
class DatasetRunResult:
    metrics_rows: list[dict[str, Any]]
    prediction_tables: list[pd.DataFrame]
    ga_history_tables: list[pd.DataFrame]
    status: str
    elapsed_seconds: float


def slugify(text: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).split("_")) or "dataset"


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def benchmark_config_signature(args: argparse.Namespace) -> str:
    config = vars(args).copy()
    config.pop("output_dir", None)
    config.pop("dry_run", None)
    return json.dumps(config, sort_keys=True, default=str)


def infer_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_map: dict[str, list[str]] = {}
    for column in columns:
        lower_map.setdefault(str(column).strip().lower(), []).append(column)
    for candidate in candidates:
        matches = lower_map.get(candidate.lower(), [])
        if len(matches) == 1:
            return matches[0]
    return None


def discover_local_datasets(root: Path, explicit_paths: list[str] | None = None) -> list[DatasetSpec]:
    paths = [Path(path) for path in explicit_paths or []]
    datasets: list[DatasetSpec] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        columns = list(frame.columns)
        smiles_column = infer_column(columns, SMILES_CANDIDATES)
        target_column = infer_column(columns, TARGET_CANDIDATES)
        if smiles_column is None or target_column is None:
            print(f"[skip] {path}: could not infer SMILES/target columns")
            continue
        datasets.append(DatasetSpec(path.stem, str(path), frame, smiles_column, target_column))
    return datasets


def load_chemml_datasets() -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    ensure_chemml_from_source()
    for name in CHEMML_EXAMPLE_OPTIONS:
        try:
            if name == "organic_density":
                frame = None
                try:
                    from chemml.datasets.base import load_organic_density

                    smiles_df, target_df, _ = load_organic_density()
                    frame = pd.concat([smiles_df.reset_index(drop=True), target_df.reset_index(drop=True)], axis=1)
                except Exception:
                    import importlib.util

                    chemml_spec = importlib.util.find_spec("chemml")
                    if chemml_spec is None or not chemml_spec.submodule_search_locations:
                        raise
                    data_path = Path(list(chemml_spec.submodule_search_locations)[0]) / "datasets" / "data" / "moldescriptor_density_smiles.csv"
                    frame = pd.read_csv(data_path)
                target = "density_Kg/m3"
                datasets.append(DatasetSpec("chemml_organic_density", "ChemML bundled dataset: organic_density", frame, "smiles", target))
            elif name == "cep_homo":
                frame = None
                try:
                    from chemml.datasets.base import load_cep_homo

                    smiles_df, target_df = load_cep_homo()
                    frame = pd.concat([smiles_df.reset_index(drop=True), target_df.reset_index(drop=True)], axis=1)
                except Exception:
                    import importlib.util

                    chemml_spec = importlib.util.find_spec("chemml")
                    if chemml_spec is None or not chemml_spec.submodule_search_locations:
                        raise
                    data_path = Path(list(chemml_spec.submodule_search_locations)[0]) / "datasets" / "data" / "cep_homo.csv"
                    frame = pd.read_csv(data_path)
                target = infer_column(list(frame.columns), ["homo", "HOMO", "target", "TARGET"]) or frame.columns[-1]
                datasets.append(DatasetSpec("chemml_cep_homo", "ChemML bundled dataset: cep_homo", frame, "smiles", target))
            elif name == "xyz_polarizability":
                from chemml.datasets.base import load_xyz_polarizability

                molecules, target_df = load_xyz_polarizability()
                smiles_values = []
                for idx, molecule in enumerate(molecules):
                    smiles_value = getattr(molecule, "smiles", None)
                    if not smiles_value:
                        raise ValueError(f"missing SMILES for row {idx}")
                    smiles_values.append(smiles_value)
                target = target_df.columns[0]
                frame = pd.DataFrame({"smiles": smiles_values, target: target_df.iloc[:, 0].to_numpy()})
                datasets.append(DatasetSpec("chemml_xyz_polarizability", "ChemML bundled dataset: xyz_polarizability", frame, "smiles", target))
            else:
                raise ValueError(f"Unknown ChemML example: {name}")
        except Exception as exc:
            print(f"[skip] ChemML {name}: {exc}")
    return datasets


def load_tdc_datasets(path: str = "./data") -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    ensure_tdc_from_source()
    try:
        from tdc.single_pred import ADME, Tox
        from tdc.metadata import admet_metrics, admet_splits
    except Exception as exc:
        print(f"[skip] PyTDC benchmark datasets: {exc}")
        return datasets
    loader_by_task = {"ADME": ADME, "Tox": Tox}
    for dataset_name, config in TDC_QSAR_OPTIONS.items():
        try:
            loader = loader_by_task[config["task"]](name=dataset_name, path=path, print_stats=False)
            frame = loader.get_data().copy().rename(columns={"Drug": "smiles", "Y": "target"})
            dataset_key = dataset_name.lower().strip()
            recommended_metric = admet_metrics.get(dataset_key) if isinstance(admet_metrics, dict) else None
            recommended_split = admet_splits.get(dataset_key) if isinstance(admet_splits, dict) else None
            datasets.append(
                DatasetSpec(
                    f"tdc_{dataset_name}",
                    f"PyTDC {config['task']} benchmark: {dataset_name}",
                    frame,
                    "smiles",
                    "target",
                    recommended_split=recommended_split,
                    recommended_metric=recommended_metric,
                )
            )
        except Exception as exc:
            print(f"[skip] PyTDC {dataset_name}: {exc}")
    return datasets


def discover_default_example_datasets(root: Path) -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    datasets.extend(load_chemml_datasets())
    datasets.extend(load_tdc_datasets(path=str(root / "data")))
    return datasets


def normalize_tdc_split(recommended_split: str | None, fallback: str) -> str:
    if not recommended_split:
        return fallback
    text = str(recommended_split).strip().lower()
    if "scaffold" in text:
        return "scaffold"
    if "random" in text:
        return "random"
    if "strat" in text or "stratif" in text:
        return "target_quartiles"
    return fallback


def normalize_tdc_metric(recommended_metric: str | None, fallback: str = "rmse") -> str:
    if not recommended_metric:
        return fallback
    text = str(recommended_metric).strip().lower()
    if "rmse" in text:
        return "rmse"
    if "mae" in text:
        return "mae"
    if "spearman" in text:
        return "spearman"
    if "pearson" in text:
        return "pearson"
    if text in {"r2", "r^2"}:
        return "r2"
    return fallback


def compute_primary_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric = str(metric_name).strip().lower()
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric in {"spearman", "pearson"}:
        try:
            from scipy.stats import spearmanr, pearsonr

            if metric == "spearman":
                return float(spearmanr(y_true, y_pred).correlation)
            return float(pearsonr(y_true, y_pred).statistic)
        except Exception:
            series_true = pd.Series(y_true, dtype=float)
            series_pred = pd.Series(y_pred, dtype=float)
            return float(series_true.corr(series_pred, method=metric))
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def primary_metric_scorer(metric_name: str):
    metric = str(metric_name).strip().lower()
    if metric == "rmse":
        return make_scorer(lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    if metric == "mae":
        return make_scorer(mean_absolute_error, greater_is_better=False)
    if metric == "r2":
        return make_scorer(r2_score)
    if metric in {"spearman", "pearson"}:
        def corr_score(y_true, y_pred):
            try:
                from scipy.stats import spearmanr, pearsonr

                if metric == "spearman":
                    return float(spearmanr(y_true, y_pred).correlation)
                return float(pearsonr(y_true, y_pred).statistic)
            except Exception:
                series_true = pd.Series(y_true, dtype=float)
                series_pred = pd.Series(y_pred, dtype=float)
                return float(series_true.corr(series_pred, method=metric))
        return make_scorer(corr_score, greater_is_better=True)
    return make_scorer(lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)


def canonicalize_frame(spec: DatasetSpec, log10_target: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    global CURRENT_DATASET_SPEC
    CURRENT_DATASET_SPEC = spec
    df = spec.frame[[spec.smiles_column, spec.target_column]].copy()
    df.columns = ["smiles", "target"]
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["smiles", "target"])
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[~df["smiles"].str.lower().isin(["", "nan", "none", "na"])].reset_index(drop=True)
    canonical = []
    keep = []
    for idx, smiles in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        keep.append(idx)
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    df = df.iloc[keep].reset_index(drop=True)
    df["canonical_smiles"] = canonical
    transform = "raw"
    if log10_target:
        if (df["target"] <= 0).any():
            print(f"[info] {spec.name}: non-positive target values; using raw target.")
        else:
            df["target"] = np.log10(df["target"].astype(float))
            transform = "log10"
    return df, {"target_transform": transform, "smiles_column": spec.smiles_column, "target_column": spec.target_column}


def parse_l1_grid(text: str) -> list[float]:
    return sorted({float(token.strip()) for token in str(text).split(",") if token.strip()})


def regularization_grid(min_log10: float, max_log10: float, size: int) -> np.ndarray:
    return np.logspace(float(min_log10), float(max_log10), int(size))


def target_quartile_labels(y: pd.Series) -> pd.Series | None:
    try:
        labels = pd.qcut(y, q=4, labels=False, duplicates="drop")
    except Exception:
        return None
    if labels.isna().any() or labels.nunique(dropna=True) < 2 or (labels.value_counts() < 2).any():
        return None
    return labels.astype(int)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    smiles: pd.Series,
    args: argparse.Namespace,
    split_strategy_override: str | None = None,
) -> dict[str, Any]:
    split_strategy = str(split_strategy_override or args.split_strategy).strip().lower()
    if CURRENT_DATASET_SPEC is not None and CURRENT_DATASET_SPEC.recommended_split:
        split_strategy = normalize_tdc_split(CURRENT_DATASET_SPEC.recommended_split, split_strategy)
    if split_strategy not in {"random", "target_quartiles", "scaffold"}:
        raise ValueError("split_strategy must be one of: random, target_quartiles, scaffold")
    if split_strategy in {"random", "target_quartiles"}:
        stratify = None
        if split_strategy == "target_quartiles":
            try:
                stratify = target_quartile_labels(y)
            except Exception:
                print("[info] quartile split unavailable; falling back to random split.")
                split_strategy = "random"
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
            X, y, smiles, test_size=args.test_fraction, random_state=args.random_seed, stratify=stratify
        )
    else:
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = scaffold_train_test_split(
            X, y, smiles, test_size=args.test_fraction, random_state=args.random_seed
        )
    split_strategy_used = split_strategy
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "smiles_train": smiles_train.reset_index(drop=True),
        "smiles_test": smiles_test.reset_index(drop=True),
        "split_strategy_used": split_strategy_used,
    }


def select_features(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, smiles_train: pd.Series, args: argparse.Namespace):
    if args.selector_method == "none":
        return X_train.copy(), X_test.copy(), {
            "selector_method": "none",
            "selected_feature_count": int(X_train.shape[1]),
            "original_feature_count": int(X_train.shape[1]),
            "selected_features": list(X_train.columns),
        }
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    selector_cv, selector_cv_folds, selector_cv_strategy = make_qsar_cv_splitter(
        X_train,
        y_train,
        smiles_train,
        split_strategy=args.split_strategy,
        cv_folds=args.selector_cv_folds,
        random_seed=args.random_seed,
    )
    selector = ElasticNetCV(
        l1_ratio=parse_l1_grid(args.selector_l1_ratio_grid),
        alphas=regularization_grid(args.selector_alpha_min_log10, args.selector_alpha_max_log10, args.selector_alpha_grid_size),
        cv=selector_cv,
        max_iter=args.selector_max_iter,
        n_jobs=-1,
        random_state=args.random_seed,
    )
    selector.fit(X_scaled, y_train)
    coef = np.asarray(selector.coef_, dtype=float)
    abs_coef = np.abs(coef)
    mask = abs_coef > args.selector_coefficient_threshold
    if not mask.any():
        mask[int(np.argmax(abs_coef))] = True
    max_features = args.max_selected_features if args.max_selected_features > 0 else max(1, math.ceil(0.10 * len(y_train)))
    if mask.sum() > max_features:
        selected = np.flatnonzero(mask)
        keep = selected[np.argsort(abs_coef[selected])[::-1]][:max_features]
        mask = np.zeros_like(mask, dtype=bool)
        mask[keep] = True
    columns = X_train.columns[mask].tolist()
    coef_df = pd.DataFrame({"feature": X_train.columns, "coefficient": coef, "abs_coefficient": abs_coef})
    return X_train[columns].copy(), X_test[columns].copy(), {
        "selector_method": "elasticnet_cv",
        "selected_feature_count": int(len(columns)),
        "original_feature_count": int(X_train.shape[1]),
        "selector_alpha": float(selector.alpha_),
        "selector_l1_ratio": float(selector.l1_ratio_),
        "selector_n_iter": int(np.max(np.atleast_1d(selector.n_iter_))),
        "selector_cv_folds": int(selector_cv_folds),
        "selector_cv_split_strategy": selector_cv_strategy,
        "selected_features": columns,
        "selector_coefficients": coef_df.sort_values("abs_coefficient", ascending=False),
        "max_selected_features": int(max_features),
    }


def regression_metrics(y_train, pred_train, y_test, pred_test) -> dict[str, float]:
    train_series = pd.Series(y_train, dtype=float)
    train_pred_series = pd.Series(pred_train, dtype=float)
    test_series = pd.Series(y_test, dtype=float)
    test_pred_series = pd.Series(pred_test, dtype=float)
    train_spearman = train_series.corr(train_pred_series, method="spearman")
    test_spearman = test_series.corr(test_pred_series, method="spearman")
    return {
        "train_r2": float(r2_score(y_train, pred_train)),
        "train_rmse": float(math.sqrt(mean_squared_error(y_train, pred_train))),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "train_spearman": float(train_spearman) if pd.notna(train_spearman) else np.nan,
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(math.sqrt(mean_squared_error(y_test, pred_test))),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_spearman": float(test_spearman) if pd.notna(test_spearman) else np.nan,
    }


def conventional_models(args: argparse.Namespace, X_train: pd.DataFrame, y_train: pd.Series, smiles_train: pd.Series) -> dict[str, Any]:
    elasticnet_cv, elasticnet_cv_folds, elasticnet_cv_strategy = make_reusable_inner_cv_splitter(
        split_strategy=args.split_strategy,
        cv_folds=args.elasticnet_cv_folds,
        random_seed=args.random_seed,
    )
    models: dict[str, Any] = {
        "ElasticNetCV": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=parse_l1_grid(args.elasticnet_l1_ratio_grid),
                        alphas=regularization_grid(args.elasticnet_alpha_min_log10, args.elasticnet_alpha_max_log10, args.elasticnet_alpha_grid_size),
                        cv=elasticnet_cv,
                        max_iter=args.elasticnet_max_iter,
                        n_jobs=-1,
                        random_state=args.random_seed,
                    ),
                ),
            ]
        ),
        "SVR": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(C=10.0, epsilon=0.1, gamma="scale")),
            ]
        ),
        "Random forest": RandomForestRegressor(n_estimators=400, random_state=args.random_seed, n_jobs=-1),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=args.random_seed,
            n_jobs=2,
        )
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=args.random_seed,
            verbose=False,
        )
        models["MapLight CatBoost"] = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=args.random_seed,
            verbose=False,
        )
    models["_elasticnet_cv_meta"] = {
        "elasticnet_cv_folds": int(elasticnet_cv_folds),
        "elasticnet_cv_split_strategy": elasticnet_cv_strategy,
    }
    return models


def evaluate_model(
    name: str,
    estimator: Any,
    X_train,
    X_test,
    y_train,
    y_test,
    smiles_train,
    args: argparse.Namespace,
    primary_metric: str | None = None,
):
    effective_split_strategy = str(args.split_strategy).strip().lower()
    if CURRENT_DATASET_SPEC is not None and CURRENT_DATASET_SPEC.recommended_split:
        effective_split_strategy = normalize_tdc_split(CURRENT_DATASET_SPEC.recommended_split, effective_split_strategy)
    if primary_metric is None:
        if CURRENT_DATASET_SPEC is not None and CURRENT_DATASET_SPEC.recommended_metric:
            primary_metric = normalize_tdc_metric(CURRENT_DATASET_SPEC.recommended_metric, fallback="rmse")
        else:
            primary_metric = "rmse"
    cv, cv_folds, cv_split_strategy = make_qsar_cv_splitter(
        X_train,
        y_train,
        smiles_train,
        split_strategy=effective_split_strategy,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
    )
    primary_scorer = primary_metric_scorer(primary_metric)
    scores = cross_validate(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "primary": primary_scorer,
        },
        n_jobs=1,
    )
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    pred_train = np.asarray(fitted.predict(X_train)).reshape(-1)
    pred_test = np.asarray(fitted.predict(X_test)).reshape(-1)
    primary_test_value = compute_primary_metric(primary_metric, y_test, pred_test)
    primary_cv_values = scores.get("test_primary")
    if primary_cv_values is not None and len(primary_cv_values):
        primary_cv = float(np.mean(primary_cv_values))
        if primary_metric in {"rmse", "mae"}:
            primary_cv = float(abs(primary_cv))
    else:
        primary_cv = np.nan
    row = {
        "model": name,
        "workflow": "conventional",
        "cv_folds": int(cv_folds),
        "cv_split_strategy": cv_split_strategy,
        "cv_r2": float(np.mean(scores["test_r2"])),
        "cv_rmse": float(np.mean(np.sqrt(-scores["test_mse"]))),
        "cv_mae": float(np.mean(-scores["test_mae"])),
        "primary_metric": primary_metric,
        "cv_primary": primary_cv,
    }
    row.update(regression_metrics(y_train, pred_train, y_test, pred_test))
    row["primary_metric_value"] = primary_test_value
    step = fitted.named_steps.get("model") if hasattr(fitted, "named_steps") else fitted
    if hasattr(step, "alpha_") or hasattr(step, "alpha"):
        row["model_alpha"] = float(getattr(step, "alpha_", getattr(step, "alpha", np.nan)))
    if hasattr(step, "l1_ratio_") or hasattr(step, "l1_ratio"):
        row["model_l1_ratio"] = float(getattr(step, "l1_ratio_", getattr(step, "l1_ratio", np.nan)))
    return row, pred_train, pred_test


def sample_value(spec: dict[str, Any], rng: random.Random) -> Any:
    if "choice" in spec:
        return rng.choice(list(spec["choice"]))
    if "int" in spec:
        return rng.randint(int(spec["int"][0]), int(spec["int"][1]))
    lo, hi = map(float, spec["uniform"])
    return rng.uniform(lo, hi)


def mutate_value(value: Any, spec: dict[str, Any], rng: random.Random, probability: float) -> Any:
    if rng.random() >= probability:
        return value
    if "choice" in spec or "int" in spec:
        return sample_value(spec, rng)
    lo, hi = map(float, spec["uniform"])
    return min(hi, max(lo, float(value) + rng.gauss(0, 0.20 * (hi - lo))))


def ga_model_specs(args: argparse.Namespace):
    specs = {
        "ElasticNet": (
            lambda p: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=float(np.exp(p["log_alpha"])),
                            l1_ratio=float(p["l1_ratio"]),
                            max_iter=args.elasticnet_max_iter,
                            random_state=args.random_seed,
                        ),
                    ),
                ]
            ),
            {
                "log_alpha": {"uniform": [float(np.log(1e-4)), float(np.log(1e-1))]},
                "l1_ratio": {"choice": parse_l1_grid(args.elasticnet_l1_ratio_grid)},
            },
            lambda p: {"alpha": float(np.exp(p["log_alpha"])), "l1_ratio": float(p["l1_ratio"])},
        )
    }
    if CatBoostRegressor is not None:
        specs["CatBoost"] = (
            lambda p: CatBoostRegressor(
                iterations=int(p["iterations"]),
                depth=int(p["depth"]),
                learning_rate=float(np.exp(p["log_learning_rate"])),
                l2_leaf_reg=float(np.exp(p["log_l2_leaf_reg"])),
                loss_function="RMSE",
                random_seed=args.random_seed,
                verbose=False,
            ),
            {
                "iterations": {"int": [200, 800]},
                "depth": {"int": [4, 8]},
                "log_learning_rate": {"uniform": [float(np.log(0.01)), float(np.log(0.2))]},
                "log_l2_leaf_reg": {"uniform": [float(np.log(1.0)), float(np.log(10.0))]},
            },
            lambda p: {
                "iterations": int(p["iterations"]),
                "depth": int(p["depth"]),
                "learning_rate": float(np.exp(p["log_learning_rate"])),
                "l2_leaf_reg": float(np.exp(p["log_l2_leaf_reg"])),
            },
        )
    return specs


def run_simple_ga(name: str, build_estimator: Callable, space: dict[str, dict[str, Any]], decode: Callable, X_train, X_test, y_train, y_test, smiles_train, args):
    rng = random.Random(args.random_seed)
    keys = list(space)
    cv, cv_folds, cv_split_strategy = make_qsar_cv_splitter(
        X_train,
        y_train,
        smiles_train,
        split_strategy=args.split_strategy,
        cv_folds=args.ga_cv_folds,
        random_seed=args.random_seed,
    )

    def random_individual():
        return {key: sample_value(space[key], rng) for key in keys}

    def score(individual):
        estimator = build_estimator(individual)
        rmses = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            fitted = clone(estimator)
            fitted.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = np.asarray(fitted.predict(X_train.iloc[valid_idx])).reshape(-1)
            rmses.append(math.sqrt(mean_squared_error(y_train.iloc[valid_idx], pred)))
        return float(np.mean(rmses))

    population = [random_individual() for _ in range(args.ga_population_size)]
    evaluated: dict[str, tuple[dict[str, Any], float]] = {}
    history = []
    for generation in range(args.ga_generations):
        scored = []
        for individual in population:
            key = json.dumps(individual, sort_keys=True)
            if key not in evaluated:
                evaluated[key] = (individual, score(individual))
            scored.append(evaluated[key])
        scored = sorted(scored, key=lambda item: item[1])
        best_individual, best_score = scored[0]
        history.append({"model": name, "generation": generation, "best_cv_rmse": best_score, "best_params": json.dumps(decode(best_individual), sort_keys=True)})
        elites = [item[0] for item in scored[: max(2, len(scored) // 2)]]
        next_population = elites[: max(1, args.ga_elites)]
        while len(next_population) < args.ga_population_size:
            parent_a, parent_b = rng.sample(elites, 2)
            next_population.append({key: mutate_value(rng.choice([parent_a[key], parent_b[key]]), space[key], rng, args.ga_mutation_probability) for key in keys})
        population = next_population
    best_individual, best_score = min(evaluated.values(), key=lambda item: item[1])
    fitted = build_estimator(best_individual)
    fitted.fit(X_train, y_train)
    pred_train = np.asarray(fitted.predict(X_train)).reshape(-1)
    pred_test = np.asarray(fitted.predict(X_test)).reshape(-1)
    row = {
        "model": f"{name} GA",
        "workflow": "ga_tuned",
        "cv_rmse": best_score,
        "cv_folds": int(cv_folds),
        "cv_split_strategy": cv_split_strategy,
        "best_params": json.dumps(decode(best_individual), sort_keys=True),
    }
    row.update(regression_metrics(y_train, pred_train, y_test, pred_test))
    return row, pd.DataFrame(history), pred_train, pred_test


def prediction_frame(dataset: str, model: str, workflow: str, split: str, smiles: pd.Series, observed: pd.Series, predicted: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "workflow": workflow,
            "split": split,
            "smiles": smiles.to_numpy(),
            "observed": observed.to_numpy(dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    )


def write_selector_outputs(dataset_dir: Path, selector_meta: dict[str, Any]) -> None:
    selected_features = selector_meta.get("selected_features", [])
    pd.DataFrame({"feature": selected_features}).to_csv(dataset_dir / "selected_features.csv", index=False)
    coefficients = selector_meta.get("selector_coefficients")
    if isinstance(coefficients, pd.DataFrame):
        coefficients.to_csv(dataset_dir / "selector_coefficients.csv", index=False)


def dataset_status_paths(dataset_dir: Path) -> tuple[Path, Path]:
    return dataset_dir / "run_status.json", dataset_dir / "metrics.csv"


def load_completed_dataset_result(dataset_dir: Path) -> DatasetRunResult | None:
    status_path, metrics_path = dataset_status_paths(dataset_dir)
    predictions_path = dataset_dir / "predictions.csv"
    ga_history_path = dataset_dir / "ga_history.csv"
    if not status_path.exists() or not metrics_path.exists():
        return None
    try:
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if status_payload.get("status") != "completed":
        return None
    metrics_rows = pd.read_csv(metrics_path).to_dict(orient="records")
    prediction_tables = [pd.read_csv(predictions_path)] if predictions_path.exists() else []
    ga_history_tables = [pd.read_csv(ga_history_path)] if ga_history_path.exists() else []
    return DatasetRunResult(
        metrics_rows=metrics_rows,
        prediction_tables=prediction_tables,
        ga_history_tables=ga_history_tables,
        status="resumed",
        elapsed_seconds=float(status_payload.get("elapsed_seconds", 0.0)),
    )


def write_dataset_status(dataset_dir: Path, payload: dict[str, Any]) -> None:
    status_path, _metrics_path = dataset_status_paths(dataset_dir)
    status_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def selected_conventional_model_names(args: argparse.Namespace) -> list[str]:
    names = ["ElasticNetCV", "SVR", "Random forest"]
    if XGBRegressor is not None:
        names.append("XGBoost")
    if CatBoostRegressor is not None:
        names.append("CatBoost")
    return names


def run_dataset(spec: DatasetSpec, output_dir: Path, args: argparse.Namespace, dataset_position: int | None = None, dataset_total: int | None = None) -> DatasetRunResult:
    start = time.time()
    dataset_id = slugify(spec.name)
    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    completed_result = load_completed_dataset_result(dataset_dir)
    if completed_result is not None:
        prefix = f"[{dataset_position}/{dataset_total}] " if dataset_position is not None and dataset_total is not None else ""
        print(f"\n{prefix}{dataset_id}: already completed in {format_seconds(completed_result.elapsed_seconds)}; reusing saved outputs")
        return completed_result

    requested_ga_models = [model.strip() for model in args.ga_models.split(",") if model.strip()]
    total_stages = 3 + len(selected_conventional_model_names(args)) + len(requested_ga_models)

    def stage_message(stage_index: int, label: str) -> None:
        elapsed = time.time() - start
        avg_stage = elapsed / max(1, stage_index - 1) if stage_index > 1 else 0.0
        remaining = max(0, total_stages - stage_index + 1)
        eta = avg_stage * remaining if avg_stage else 0.0
        prefix = f"[{dataset_position}/{dataset_total}] " if dataset_position is not None and dataset_total is not None else ""
        print(f"\n{prefix}{dataset_id} | stage {stage_index}/{total_stages}: {label} | elapsed {format_seconds(elapsed)} | dataset ETA {format_seconds(eta)}")

    write_dataset_status(
        dataset_dir,
        {
            "status": "running",
            "dataset": dataset_id,
            "source": spec.source,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    stage_message(1, f"loading {spec.source}")
    df, input_meta = canonicalize_frame(spec, args.log10_target)
    if len(df) < args.minimum_rows:
        print(f"[skip] {dataset_id}: only {len(df)} valid rows after cleanup")
        write_dataset_status(dataset_dir, {"status": "skipped", "reason": "too_few_rows_after_cleanup", "n_rows": int(len(df))})
        return DatasetRunResult([], [], [], "skipped", time.time() - start)
    if args.row_limit and len(df) > args.row_limit:
        df = df.sample(n=args.row_limit, random_state=args.random_seed).reset_index(drop=True)

    stage_message(2, "building molecular features")
    X, feature_meta = build_feature_matrix_from_smiles(
        df["canonical_smiles"].tolist(),
        selected_feature_families=["morgan", "ecfp6", "fcfp6", "maccs", "rdkit", "maplight"],
        radius=2,
        n_bits=args.fingerprint_bits,
        enable_persistent_feature_store=args.enable_persistent_feature_store,
        reuse_persistent_feature_store=args.reuse_persistent_feature_store,
        persistent_feature_store_path=args.persistent_feature_store_path,
    )
    y = df["target"].astype(float).reset_index(drop=True)
    smiles = df["canonical_smiles"].reset_index(drop=True)

    if len(df) < args.minimum_rows:
        print(f"[skip] {dataset_id}: only {len(df)} valid rows after feature generation")
        write_dataset_status(dataset_dir, {"status": "skipped", "reason": "too_few_rows_after_features", "n_rows": int(len(df))})
        return DatasetRunResult([], [], [], "skipped", time.time() - start)

    selector_args = argparse.Namespace(**vars(args))
    if selector_args.max_selected_features <= 0:
        selector_args.max_selected_features = max(1, math.ceil(0.10 * len(y)))
    stage_message(3, "splitting data and selecting features")
    split = split_data(X, y, smiles, args)
    X_train, X_test, selector_meta = select_features(split["X_train"], split["X_test"], split["y_train"], split["smiles_train"], selector_args)
    write_selector_outputs(dataset_dir, selector_meta)
    maplight_prefixes = ("maplight_morgan_", "avalon_count_", "erg_", "maplight_desc_")
    maplight_feature_cols = [
        col for col in split["X_train"].columns if str(col).startswith(maplight_prefixes)
    ]

    base_meta = {
        "dataset": dataset_id,
        "dataset_source": spec.source,
        "n_molecules": int(len(df)),
        "n_train": int(len(split["y_train"])),
        "n_test": int(len(split["y_test"])),
        "target_transform": input_meta["target_transform"],
        "smiles_column": input_meta["smiles_column"],
        "target_column": input_meta["target_column"],
        "split_strategy": split["split_strategy_used"],
        "original_feature_count": int(selector_meta["original_feature_count"]),
        "selected_feature_count": int(selector_meta["selected_feature_count"]),
        "selector_method": selector_meta["selector_method"],
        "selector_alpha": selector_meta.get("selector_alpha", np.nan),
        "selector_l1_ratio": selector_meta.get("selector_l1_ratio", np.nan),
        "selector_n_iter": selector_meta.get("selector_n_iter", np.nan),
        "selector_cv_folds": selector_meta.get("selector_cv_folds", np.nan),
        "selector_cv_split_strategy": selector_meta.get("selector_cv_split_strategy", ""),
        "selector_max_selected_features": selector_meta.get("max_selected_features", np.nan),
        "feature_store_path": feature_meta.get("feature_store_path", ""),
        "feature_store_shard_format": feature_meta.get("feature_store_shard_format", ""),
        "feature_store_cached_rows": feature_meta.get("cached_rows_loaded", np.nan),
        "feature_store_generated_rows": feature_meta.get("generated_rows_added", np.nan),
        "representation_key": feature_meta.get("representation_key", ""),
    }

    metrics_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    ga_history_tables: list[pd.DataFrame] = []

    stage_index = 4
    model_bundle = conventional_models(args, X_train, split["y_train"], split["smiles_train"])
    elasticnet_cv_meta = model_bundle.pop("_elasticnet_cv_meta", {})
    model_items = list(model_bundle.items())
    for model_name, estimator in model_items:
        stage_message(stage_index, f"conventional model {model_name}")
        if model_name == "MapLight CatBoost":
            if not maplight_feature_cols:
                print(f"[skip] {dataset_id} {model_name}: MapLight classic features were not found in the feature matrix")
                stage_index += 1
                continue
            model_X_train = split["X_train"].loc[:, maplight_feature_cols].reset_index(drop=True)
            model_X_test = split["X_test"].loc[:, maplight_feature_cols].reset_index(drop=True)
        else:
            model_X_train = X_train
            model_X_test = X_test
        try:
            row, pred_train, pred_test = evaluate_model(model_name, estimator, model_X_train, model_X_test, split["y_train"], split["y_test"], split["smiles_train"], args)
        except Exception as exc:
            row = {"model": model_name, "workflow": "conventional", "error": str(exc)}
            pred_train, pred_test = np.array([]), np.array([])
        if model_name == "ElasticNetCV":
            row.update(elasticnet_cv_meta)
        row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
        metrics_rows.append(row)
        if len(pred_train):
            prediction_tables.extend(
                [
                    prediction_frame(dataset_id, model_name, "conventional", "train", split["smiles_train"], split["y_train"], pred_train),
                    prediction_frame(dataset_id, model_name, "conventional", "test", split["smiles_test"], split["y_test"], pred_test),
                ]
            )
        stage_index += 1

    ga_specs = ga_model_specs(args)
    for model_name in requested_ga_models:
        if model_name not in ga_specs:
            print(f"[skip] {dataset_id} GA {model_name}: model unavailable")
            stage_index += 1
            continue
        stage_message(stage_index, f"GA tuning {model_name}")
        try:
            row, history, pred_train, pred_test = run_simple_ga(model_name, *ga_specs[model_name], X_train, X_test, split["y_train"], split["y_test"], split["smiles_train"], args)
        except Exception as exc:
            row = {"model": f"{model_name} GA", "workflow": "ga_tuned", "error": str(exc)}
            history, pred_train, pred_test = pd.DataFrame(), np.array([]), np.array([])
        row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
        metrics_rows.append(row)
        if not history.empty:
            history.insert(0, "dataset", dataset_id)
            ga_history_tables.append(history)
        if len(pred_train):
            prediction_tables.extend(
                [
                    prediction_frame(dataset_id, f"{model_name} GA", "ga_tuned", "train", split["smiles_train"], split["y_train"], pred_train),
                    prediction_frame(dataset_id, f"{model_name} GA", "ga_tuned", "test", split["smiles_test"], split["y_test"], pred_test),
                ]
            )
        stage_index += 1

    pd.DataFrame(metrics_rows).to_csv(dataset_dir / "metrics.csv", index=False)
    if prediction_tables:
        pd.concat(prediction_tables, ignore_index=True).to_csv(dataset_dir / "predictions.csv", index=False)
    if ga_history_tables:
        pd.concat(ga_history_tables, ignore_index=True).to_csv(dataset_dir / "ga_history.csv", index=False)
    elapsed_seconds = time.time() - start
    write_dataset_status(
        dataset_dir,
        {
            "status": "completed",
            "dataset": dataset_id,
            "source": spec.source,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "n_metrics_rows": len(metrics_rows),
        },
    )
    return DatasetRunResult(metrics_rows, prediction_tables, ga_history_tables, "completed", elapsed_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="CSV dataset path. Repeat to override the default notebook example set with local CSVs only.")
    parser.add_argument("--include-local-csv", action="append", help="Optional extra local CSV dataset path to add on top of the default ChemML + PyTDC example set.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to benchmark_results/autoqsar_benchmark_<timestamp>.")
    parser.add_argument("--dry-run", action="store_true", help="List discovered datasets and planned configuration without fitting models.")

    parser.add_argument("--minimum-rows", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=0, help="Optional deterministic row cap for smoke tests. 0 uses all rows.")
    parser.add_argument("--fingerprint-bits", type=int, default=2048, help="Morgan/ECFP/FCFP fingerprint bit length.")
    parser.add_argument("--log10-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-strategy", choices=["target_quartiles", "random", "scaffold"], default="target_quartiles")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--enable-persistent-feature-store", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-persistent-feature-store", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-feature-store-path", default="AUTO")

    parser.add_argument("--selector-method", choices=["elasticnet_cv", "none"], default="elasticnet_cv")
    parser.add_argument("--selector-l1-ratio-grid", default="0.1, 0.3, 0.5, 0.7, 0.9")
    parser.add_argument("--selector-alpha-min-log10", type=float, default=-5)
    parser.add_argument("--selector-alpha-max-log10", type=float, default=-1)
    parser.add_argument("--selector-alpha-grid-size", type=int, default=40)
    parser.add_argument("--selector-cv-folds", type=int, default=5)
    parser.add_argument("--selector-max-iter", type=int, default=50000)
    parser.add_argument("--selector-coefficient-threshold", type=float, default=1e-10)
    parser.add_argument("--max-selected-features", type=int, default=0, help="0 means auto cap at 10%% of valid molecules; positive values override.")

    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--elasticnet-l1-ratio-grid", default="0.4, 0.5, 0.6, 0.7, 0.9")
    parser.add_argument("--elasticnet-alpha-min-log10", type=float, default=-4)
    parser.add_argument("--elasticnet-alpha-max-log10", type=float, default=0)
    parser.add_argument("--elasticnet-alpha-grid-size", type=int, default=40)
    parser.add_argument("--elasticnet-cv-folds", type=int, default=5)
    parser.add_argument("--elasticnet-max-iter", type=int, default=50000)

    parser.add_argument("--ga-models", default="", help="Comma-separated GA models to tune. Empty default skips GA. Example: ElasticNet,CatBoost.")
    parser.add_argument("--ga-generations", type=int, default=12)
    parser.add_argument("--ga-population-size", type=int, default=16)
    parser.add_argument("--ga-elites", type=int, default=2)
    parser.add_argument("--ga-cv-folds", type=int, default=5)
    parser.add_argument("--ga-mutation-probability", type=float, default=0.35)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume a compatible incomplete run when possible.")
    return parser.parse_args()


def select_output_dir(root: Path, args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    benchmark_root = root / "benchmark_results"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    config_signature = benchmark_config_signature(args)
    if args.resume:
        for candidate in sorted(benchmark_root.glob("autoqsar_benchmark_*"), key=lambda path: path.name, reverse=True):
            run_config_path = candidate / "run_config.json"
            run_complete_path = candidate / "run_complete.json"
            if not run_config_path.exists() or run_complete_path.exists():
                continue
            try:
                payload = json.loads(run_config_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("config_signature") == config_signature:
                print(f"Resuming compatible incomplete run: {candidate}")
                return candidate
    return benchmark_root / f"autoqsar_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = select_output_dir(root, args)

    if args.dataset:
        datasets = discover_local_datasets(root, args.dataset)
    else:
        datasets = discover_default_example_datasets(root)
        if args.include_local_csv:
            datasets.extend(discover_local_datasets(root, args.include_local_csv))
    if not datasets:
        print("No datasets found. Provide --dataset PATH or verify ChemML / PyTDC dependencies are available.")
        return 1

    if args.dry_run:
        print(f"Planned output directory: {output_dir}")
        print("Datasets:")
        for dataset in datasets:
            print(f"  - {dataset.name}: smiles={dataset.smiles_column}, target={dataset.target_column}, source={dataset.source}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["config_signature"] = benchmark_config_signature(args)
    config["datasets"] = [{"name": item.name, "source": item.source, "smiles_column": item.smiles_column, "target_column": item.target_column} for item in datasets]
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print("Datasets:")
    for dataset in datasets:
        print(f"  - {dataset.name}: smiles={dataset.smiles_column}, target={dataset.target_column}, source={dataset.source}")

    overall_start = time.time()
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_histories: list[pd.DataFrame] = []
    completed_dataset_times: list[float] = []
    for index, spec in enumerate(datasets, start=1):
        result = run_dataset(spec, output_dir, args, dataset_position=index, dataset_total=len(datasets))
        all_metrics.extend(result.metrics_rows)
        all_predictions.extend(result.prediction_tables)
        all_histories.extend(result.ga_history_tables)
        if result.status in {"completed", "resumed"}:
            completed_dataset_times.append(float(result.elapsed_seconds))
        elapsed = time.time() - overall_start
        avg_dataset_time = sum(completed_dataset_times) / max(1, len(completed_dataset_times)) if completed_dataset_times else 0.0
        remaining = len(datasets) - index
        eta = avg_dataset_time * remaining if avg_dataset_time else 0.0
        print(
            f"[overall {index}/{len(datasets)}] "
            f"status={result.status} | elapsed {format_seconds(elapsed)} | "
            f"average per finished dataset {format_seconds(avg_dataset_time) if avg_dataset_time else 'n/a'} | "
            f"ETA {format_seconds(eta)}"
        )

    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary.to_csv(output_dir / "summary_metrics.csv", index=False)
        if {"dataset", "model", "test_rmse"}.issubset(summary.columns):
            pivot = summary.pivot_table(index="dataset", columns="model", values="test_rmse", aggfunc="min")
            pivot.to_csv(output_dir / "test_rmse_pivot.csv")
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    if all_histories:
        pd.concat(all_histories, ignore_index=True).to_csv(output_dir / "ga_history.csv", index=False)
    (output_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - overall_start, 3),
                "dataset_count": len(datasets),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    primary_files = ["summary_metrics.csv", "test_rmse_pivot.csv", "predictions.csv", "run_config.json"]
    if all_histories:
        primary_files.append("ga_history.csv")
    print(f"\nWrote benchmark outputs to {output_dir}")
    print("Primary files: " + ", ".join(primary_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
