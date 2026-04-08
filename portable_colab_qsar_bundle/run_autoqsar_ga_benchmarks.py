#!/usr/bin/env python
"""Benchmark the conventional AutoQSAR workflow.

This script discovers selected example CSV datasets, builds molecular features,
runs the train/test split plus train-only ElasticNetCV feature selection,
evaluates conventional models, optionally runs a small GA tuning pass for
selected models, and writes cross-dataset performance tables. It intentionally
does not run deep-learning models.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

RDLogger.DisableLog("rdApp.warning")

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
DEFAULT_DATASET_FILES = ["fu.csv", "HLe_invivo.csv", "VDss.csv"]


@dataclass
class DatasetSpec:
    name: str
    source: str
    frame: pd.DataFrame
    smiles_column: str
    target_column: str


def slugify(text: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).split("_")) or "dataset"


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


def default_dataset_paths(root: Path) -> list[Path]:
    test_data = root / "test_data"
    available = {path.name.lower(): path for path in test_data.glob("*.csv")}
    paths = []
    for name in DEFAULT_DATASET_FILES:
        match = available.get(name.lower())
        if match is None:
            print(f"[skip] default dataset {name}: file not found under {test_data}")
            continue
        paths.append(match)
    return paths


def discover_local_datasets(root: Path, explicit_paths: list[str] | None = None) -> list[DatasetSpec]:
    paths = [Path(path) for path in explicit_paths or []] or default_dataset_paths(root)
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
    try:
        from chemml.datasets.base import load_cep_homo, load_organic_density
    except Exception as exc:
        print(f"[skip] ChemML bundled datasets: {exc}")
        return datasets
    for name, loader, target_column in [
        ("chemml_organic_density", load_organic_density, "density_Kg/m3"),
        ("chemml_cep_homo", load_cep_homo, None),
    ]:
        try:
            payload = loader()
            smiles_df, target_df = payload[0], payload[1]
            target = target_column or target_df.columns[0]
            frame = pd.concat([smiles_df.reset_index(drop=True), target_df.reset_index(drop=True)], axis=1)
            datasets.append(DatasetSpec(name, f"ChemML bundled dataset: {name}", frame, "smiles", target))
        except Exception as exc:
            print(f"[skip] {name}: {exc}")
    return datasets


def canonicalize_frame(spec: DatasetSpec, log10_target: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
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


def bitvect_array(bitvect: Any) -> np.ndarray:
    arr = np.zeros((bitvect.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def build_feature_matrix(smiles_values: pd.Series, fingerprint_bits: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, valid = [], []
    descriptor_names = [name for name, _ in Descriptors._descList]
    for idx, smiles in enumerate(smiles_values.astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        chunks = [
            bitvect_array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fingerprint_bits)),
            bitvect_array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fingerprint_bits)),
            bitvect_array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fingerprint_bits, useFeatures=True)),
            bitvect_array(MACCSkeys.GenMACCSKeys(mol)),
        ]
        descriptor_values = []
        for _name, fn in Descriptors._descList:
            try:
                descriptor_values.append(float(fn(mol)))
            except Exception:
                descriptor_values.append(np.nan)
        chunks.append(np.asarray(descriptor_values, dtype=np.float32))
        rows.append(np.concatenate(chunks).astype(np.float32))
        valid.append(idx)
    columns = (
        [f"morgan_r2_{i}" for i in range(fingerprint_bits)]
        + [f"ecfp6_{i}" for i in range(fingerprint_bits)]
        + [f"fcfp6_{i}" for i in range(fingerprint_bits)]
        + [f"maccs_{i}" for i in range(167)]
        + [f"rdkit_{name}" for name in descriptor_names]
    )
    X = pd.DataFrame(rows, columns=columns).replace([np.inf, -np.inf], np.nan)
    X = X.drop(columns=X.columns[X.isna().all()].tolist())
    return X.reset_index(drop=True), {"valid_indices": valid, "feature_count": int(X.shape[1])}


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


def split_data(X: pd.DataFrame, y: pd.Series, smiles: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    stratify = target_quartile_labels(y) if args.split_strategy == "target_quartiles" else None
    if args.split_strategy == "target_quartiles" and stratify is None:
        print("[info] quartile split unavailable; falling back to random split.")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, smiles, test_size=args.test_fraction, random_state=args.random_seed, stratify=stratify
    )
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "smiles_train": smiles_train.reset_index(drop=True),
        "smiles_test": smiles_test.reset_index(drop=True),
        "split_strategy_used": "target_quartiles" if stratify is not None else "random",
    }


def select_features(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, args: argparse.Namespace):
    if args.selector_method == "none":
        return X_train.copy(), X_test.copy(), {
            "selector_method": "none",
            "selected_feature_count": int(X_train.shape[1]),
            "original_feature_count": int(X_train.shape[1]),
            "selected_features": list(X_train.columns),
        }
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    selector = ElasticNetCV(
        l1_ratio=parse_l1_grid(args.selector_l1_ratio_grid),
        alphas=regularization_grid(args.selector_alpha_min_log10, args.selector_alpha_max_log10, args.selector_alpha_grid_size),
        cv=min(args.selector_cv_folds, len(X_train)),
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
        "selected_features": columns,
        "selector_coefficients": coef_df.sort_values("abs_coefficient", ascending=False),
        "max_selected_features": int(max_features),
    }


def regression_metrics(y_train, pred_train, y_test, pred_test) -> dict[str, float]:
    return {
        "train_r2": float(r2_score(y_train, pred_train)),
        "train_rmse": float(math.sqrt(mean_squared_error(y_train, pred_train))),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(math.sqrt(mean_squared_error(y_test, pred_test))),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
    }


def conventional_models(args: argparse.Namespace) -> dict[str, Any]:
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
                        cv=args.elasticnet_cv_folds,
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
    return models


def evaluate_model(name: str, estimator: Any, X_train, X_test, y_train, y_test, args: argparse.Namespace):
    cv = KFold(n_splits=min(args.cv_folds, len(X_train)), shuffle=True, random_state=args.random_seed)
    scores = cross_validate(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"},
        n_jobs=1,
    )
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    pred_train = np.asarray(fitted.predict(X_train)).reshape(-1)
    pred_test = np.asarray(fitted.predict(X_test)).reshape(-1)
    row = {
        "model": name,
        "workflow": "conventional",
        "cv_r2": float(np.mean(scores["test_r2"])),
        "cv_rmse": float(np.mean(np.sqrt(-scores["test_mse"]))),
        "cv_mae": float(np.mean(-scores["test_mae"])),
    }
    row.update(regression_metrics(y_train, pred_train, y_test, pred_test))
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


def run_simple_ga(name: str, build_estimator: Callable, space: dict[str, dict[str, Any]], decode: Callable, X_train, X_test, y_train, y_test, args):
    rng = random.Random(args.random_seed)
    keys = list(space)
    cv = KFold(n_splits=min(args.ga_cv_folds, len(X_train)), shuffle=True, random_state=args.random_seed)

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
    row = {"model": f"{name} GA", "workflow": "ga_tuned", "cv_rmse": best_score, "best_params": json.dumps(decode(best_individual), sort_keys=True)}
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


def run_dataset(spec: DatasetSpec, output_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    start = time.time()
    dataset_id = slugify(spec.name)
    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{dataset_id}] loading {spec.source}")

    df, input_meta = canonicalize_frame(spec, args.log10_target)
    if len(df) < args.minimum_rows:
        print(f"[skip] {dataset_id}: only {len(df)} valid rows after cleanup")
        return [], [], []
    if args.row_limit and len(df) > args.row_limit:
        df = df.sample(n=args.row_limit, random_state=args.random_seed).reset_index(drop=True)

    X, feature_meta = build_feature_matrix(df["canonical_smiles"], args.fingerprint_bits)
    valid_indices = feature_meta["valid_indices"]
    df = df.iloc[valid_indices].reset_index(drop=True)
    y = df["target"].astype(float).reset_index(drop=True)
    smiles = df["canonical_smiles"].reset_index(drop=True)

    if len(df) < args.minimum_rows:
        print(f"[skip] {dataset_id}: only {len(df)} valid rows after feature generation")
        return [], [], []

    selector_args = argparse.Namespace(**vars(args))
    if selector_args.max_selected_features <= 0:
        selector_args.max_selected_features = max(1, math.ceil(0.10 * len(y)))
    split = split_data(X, y, smiles, args)
    X_train, X_test, selector_meta = select_features(split["X_train"], split["X_test"], split["y_train"], selector_args)
    write_selector_outputs(dataset_dir, selector_meta)

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
        "selector_max_selected_features": selector_meta.get("max_selected_features", np.nan),
    }

    metrics_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    ga_history_tables: list[pd.DataFrame] = []

    for model_name, estimator in conventional_models(args).items():
        print(f"[{dataset_id}] conventional: {model_name}")
        try:
            row, pred_train, pred_test = evaluate_model(model_name, estimator, X_train, X_test, split["y_train"], split["y_test"], args)
        except Exception as exc:
            row = {"model": model_name, "workflow": "conventional", "error": str(exc)}
            pred_train, pred_test = np.array([]), np.array([])
        row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
        metrics_rows.append(row)
        if len(pred_train):
            prediction_tables.extend(
                [
                    prediction_frame(dataset_id, model_name, "conventional", "train", split["smiles_train"], split["y_train"], pred_train),
                    prediction_frame(dataset_id, model_name, "conventional", "test", split["smiles_test"], split["y_test"], pred_test),
                ]
            )

    ga_specs = ga_model_specs(args)
    requested_ga_models = [model.strip() for model in args.ga_models.split(",") if model.strip()]
    for model_name in requested_ga_models:
        if model_name not in ga_specs:
            print(f"[skip] {dataset_id} GA {model_name}: model unavailable")
            continue
        print(f"[{dataset_id}] GA tuning: {model_name}")
        try:
            row, history, pred_train, pred_test = run_simple_ga(model_name, *ga_specs[model_name], X_train, X_test, split["y_train"], split["y_test"], args)
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

    pd.DataFrame(metrics_rows).to_csv(dataset_dir / "metrics.csv", index=False)
    if prediction_tables:
        pd.concat(prediction_tables, ignore_index=True).to_csv(dataset_dir / "predictions.csv", index=False)
    if ga_history_tables:
        pd.concat(ga_history_tables, ignore_index=True).to_csv(dataset_dir / "ga_history.csv", index=False)
    return metrics_rows, prediction_tables, ga_history_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="CSV dataset path. Repeat to run a subset. Defaults to test_data/fu.csv, test_data/HLe_invivo.csv, and test_data/VDss.csv.")
    parser.add_argument("--include-chemml", action="store_true", help="Also run ChemML bundled example datasets if ChemML is installed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to benchmark_results/autoqsar_benchmark_<timestamp>.")
    parser.add_argument("--dry-run", action="store_true", help="List discovered datasets and planned configuration without fitting models.")

    parser.add_argument("--minimum-rows", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=0, help="Optional deterministic row cap for smoke tests. 0 uses all rows.")
    parser.add_argument("--fingerprint-bits", type=int, default=2048, help="Morgan/ECFP/FCFP fingerprint bit length.")
    parser.add_argument("--log10-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-strategy", choices=["target_quartiles", "random"], default="target_quartiles")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=13)

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or root / "benchmark_results" / f"autoqsar_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"

    datasets = discover_local_datasets(root, args.dataset)
    if args.include_chemml:
        datasets.extend(load_chemml_datasets())
    if not datasets:
        print("No datasets found. Provide --dataset PATH or add CSV files under test_data/.")
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
    config["datasets"] = [{"name": item.name, "source": item.source, "smiles_column": item.smiles_column, "target_column": item.target_column} for item in datasets]
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print("Datasets:")
    for dataset in datasets:
        print(f"  - {dataset.name}: smiles={dataset.smiles_column}, target={dataset.target_column}, source={dataset.source}")

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_histories: list[pd.DataFrame] = []
    for spec in datasets:
        metrics, predictions, histories = run_dataset(spec, output_dir, args)
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_histories.extend(histories)

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

    primary_files = ["summary_metrics.csv", "test_rmse_pivot.csv", "predictions.csv", "run_config.json"]
    if all_histories:
        primary_files.append("ga_history.csv")
    print(f"\nWrote benchmark outputs to {output_dir}")
    print("Primary files: " + ", ".join(primary_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
