#!/usr/bin/env python
"""Benchmark the AutoQSAR notebook workflow (CPU-focused).

This script benchmarks the notebook's example SMILES-based datasets: runnable
ChemML bundled examples plus QSAR benchmark suites (PyTDC, MoleculeNet
physchem, and Polaris ADME Fang splits). It builds
molecular features, runs the train/test split plus train-only ElasticNetCV
feature selection, evaluates conventional models, optionally runs a small GA
tuning pass, runs deep workflows (ChemML backends, Chemprop v2 graph variants,
and MapLight + GNN), optionally runs CFA combinatorial fusion over conventional
predictions, builds an optional ensemble over available members, and
writes cross-dataset performance tables.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import io
import importlib.util
import json
import getpass
import logging
import math
import os
import pickle
import random
import re
import shutil
import time
import tempfile
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, RidgeCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

# Reduce TensorFlow/absl startup noise when optional backends are imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

try:
    from portable_colab_qsar_bundle.qsar_workflow_core import (
        build_feature_matrix_from_smiles,
        drop_exact_and_near_duplicate_features,
        make_maplight_classic_matrix,
        make_qsar_cv_splitter,
        make_reusable_inner_cv_splitter,
        run_cfa_regression_fusion,
        resolve_chemprop_architecture_specs,
        scaffold_train_test_split,
        target_quartile_labels,
    )
    from portable_colab_qsar_bundle.benchmark_registry import (
        CHEMML_EXAMPLE_OPTIONS,
        FREESOLV_EXPANDED_SCALED_OPTION,
        MOLECULENET_LEADERBOARD_README_URL,
        MOLECULENET_PHYSCHEM_OPTIONS,
        POLARIS_ADME_OPTIONS,
        PYTDC_SOURCE_URL,
        TDC_LEADERBOARD_URLS,
        TDC_QSAR_OPTIONS,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portable_colab_qsar_bundle.qsar_workflow_core import (
        build_feature_matrix_from_smiles,
        drop_exact_and_near_duplicate_features,
        make_maplight_classic_matrix,
        make_qsar_cv_splitter,
        make_reusable_inner_cv_splitter,
        run_cfa_regression_fusion,
        resolve_chemprop_architecture_specs,
        scaffold_train_test_split,
        target_quartile_labels,
    )
    from portable_colab_qsar_bundle.benchmark_registry import (
        CHEMML_EXAMPLE_OPTIONS,
        FREESOLV_EXPANDED_SCALED_OPTION,
        MOLECULENET_LEADERBOARD_README_URL,
        MOLECULENET_PHYSCHEM_OPTIONS,
        POLARIS_ADME_OPTIONS,
        PYTDC_SOURCE_URL,
        TDC_LEADERBOARD_URLS,
        TDC_QSAR_OPTIONS,
    )

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from tabpfn import TabPFNRegressor
    TABPFN_REGRESSOR_SOURCE = "tabpfn"
except Exception:
    try:
        from tabpfn_client import TabPFNRegressor
        TABPFN_REGRESSOR_SOURCE = "tabpfn_client"
    except Exception:
        TabPFNRegressor = None
        TABPFN_REGRESSOR_SOURCE = "unavailable"


def detect_gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


SMILES_CANDIDATES = ["QSAR_READY_SMILES", "canonical_smiles", "SMILES", "smiles", "Smiles"]
TARGET_CANDIDATES = ["TARGET", "target", "Target", "Repro/Dev", "Non-Repro/Dev", "density_Kg/m3"]
DEFAULT_BENCHMARK_FEATURE_FAMILIES = [
    "morgan",
    "ecfp6",
    "fcfp6",
    "layered",
    "atom_pair",
    "topological_torsion",
    "rdk_path",
    "maccs",
    "rdkit",
    "maplight",
]

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
            try:
                with tarfile.open(candidate, "r:gz") as tf:
                    tf.extractall(extract_root)
            except Exception:
                continue
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
    try:
        with tarfile.open(download_target, "r:gz") as tf:
            tf.extractall(extract_root)
    except Exception:
        return False
    return try_load_tdc_from_path(extract_root)


def ensure_tabpfn_installed(prefer_local_backend: bool = False) -> bool:
    global TabPFNRegressor
    global TABPFN_REGRESSOR_SOURCE
    current_source = str(TABPFN_REGRESSOR_SOURCE).strip().lower()
    if TabPFNRegressor is not None and not (prefer_local_backend and current_source != "tabpfn"):
        return True

    backend_order = ["tabpfn", "tabpfn_client"] if prefer_local_backend else ["tabpfn", "tabpfn_client"]
    if current_source == "tabpfn_client" and not prefer_local_backend:
        backend_order = ["tabpfn_client", "tabpfn"]

    install_messages = {
        "tabpfn": "local tabpfn backend",
        "tabpfn_client": "tabpfn-client API backend",
    }
    import_targets = {
        "tabpfn": ("tabpfn", "TabPFNRegressor"),
        "tabpfn_client": ("tabpfn_client", "TabPFNRegressor"),
    }
    install_targets = {
        "tabpfn": "tabpfn",
        "tabpfn_client": "tabpfn-client",
    }

    for backend in backend_order:
        module_name, class_name = import_targets[backend]
        try:
            module = importlib.import_module(module_name)
            TabPFNRegressor = getattr(module, class_name)
            TABPFN_REGRESSOR_SOURCE = backend
            return True
        except Exception:
            pass

        if importlib.util.find_spec(module_name) is None:
            print(f"[installing] {install_targets[backend]}", flush=True)
            install_cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-input", install_targets[backend]]
            install_result = subprocess.run(install_cmd, capture_output=True, text=True)
            if install_result.returncode != 0:
                install_logs = ((install_result.stdout or "") + "\n" + (install_result.stderr or "")).strip()
                print(
                    f"{install_messages[backend]} auto-install failed.\n"
                    f"Command: {' '.join(install_cmd)}\n"
                    f"Installer output (tail):\n{install_logs[-800:]}",
                    flush=True,
                )
                continue
            importlib.invalidate_caches()
        try:
            module = importlib.import_module(module_name)
            TabPFNRegressor = getattr(module, class_name)
            TABPFN_REGRESSOR_SOURCE = backend
            return True
        except Exception:
            continue

    print(
        "TabPFNRegressor import/install failed for both local and API backends; "
        "the run will continue without TabPFN.",
        flush=True,
    )
    TabPFNRegressor = None
    TABPFN_REGRESSOR_SOURCE = "unavailable"
    return False


def ensure_tabpfn_client_installed() -> bool:
    if importlib.util.find_spec("tabpfn_client") is not None:
        return True
    print("[installing] tabpfn-client", flush=True)
    install_cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-input", "tabpfn-client"]
    install_result = subprocess.run(install_cmd, capture_output=True, text=True)
    if install_result.returncode != 0:
        install_logs = ((install_result.stdout or "") + "\n" + (install_result.stderr or "")).strip()
        print(
            "tabpfn-client auto-install failed; token setup via tabpfn_client will be skipped.\n"
            f"Command: {' '.join(install_cmd)}\n"
            f"Installer output (tail):\n{install_logs[-800:]}",
            flush=True,
        )
        return False
    importlib.invalidate_caches()
    return importlib.util.find_spec("tabpfn_client") is not None


def configure_tabpfn_access_token_from_env() -> tuple[bool, str]:
    api_key = str(os.environ.get("PRIORLABS_API_KEY", os.environ.get("TABPFN_API_KEY", ""))).strip()
    if not api_key:
        return False, "No PRIORLABS_API_KEY/TABPFN_API_KEY found in environment."
    if not ensure_tabpfn_client_installed():
        return False, "tabpfn-client is unavailable; could not apply access token."
    try:
        from tabpfn_client import set_access_token

        set_access_token(api_key)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _probe_tabpfn_runtime_ready() -> tuple[bool, str]:
    if TabPFNRegressor is None:
        return False, "TabPFNRegressor is unavailable."
    try:
        probe_model = TabPFNRegressor()
        probe_X = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=float,
        )
        probe_y = np.asarray([0.0, 1.0, 0.5, 1.5, 1.0, 2.0], dtype=float)
        probe_model.fit(probe_X, probe_y)
        _ = probe_model.predict(probe_X[:2])
        return True, ""
    except Exception as exc:
        return False, str(exc)


def prepare_tabpfn_auth(args: argparse.Namespace) -> tuple[bool, str]:
    if not bool(getattr(args, "run_tabpfn", False)):
        return True, "TabPFN is disabled for this run."
    if str(TABPFN_REGRESSOR_SOURCE).strip().lower() != "tabpfn_client":
        return True, f"TabPFN backend source: {TABPFN_REGRESSOR_SOURCE} (no Prior Labs token setup required)."

    token_ok, token_error = configure_tabpfn_access_token_from_env()
    if token_ok:
        probe_ok, probe_error = _probe_tabpfn_runtime_ready()
        if probe_ok:
            return True, "TabPFN authentication verified via PRIORLABS_API_KEY/TABPFN_API_KEY."
        if is_tabpfn_token_limit_error(probe_error):
            return False, format_tabpfn_token_limit_notice(
                f"TabPFN access token was loaded, but runtime preflight hit a token limit. Error: {probe_error}"
            )
        return False, (
            "TabPFN access token was loaded from environment, but runtime preflight failed. "
            f"Error: {probe_error}"
        )

    has_env_key = bool(str(os.environ.get("PRIORLABS_API_KEY", os.environ.get("TABPFN_API_KEY", ""))).strip())
    if not has_env_key:
        probe_ok, probe_error = _probe_tabpfn_runtime_ready()
        if probe_ok:
            return True, "TabPFN preflight passed using an existing local client token."
        if is_tabpfn_token_limit_error(probe_error):
            return False, format_tabpfn_token_limit_notice(probe_error)
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                entered = getpass.getpass(
                    "TabPFN is enabled and no PRIORLABS_API_KEY/TABPFN_API_KEY was found. "
                    "Enter Prior Labs API key (input hidden), or press Enter to disable TabPFN for this run: "
                ).strip()
            except Exception:
                entered = ""
            if entered:
                os.environ["PRIORLABS_API_KEY"] = entered
                token_ok, token_error = configure_tabpfn_access_token_from_env()
                if token_ok:
                    probe_ok, probe_error = _probe_tabpfn_runtime_ready()
                    if probe_ok:
                        return True, "TabPFN authentication verified from interactively provided API key."
                    if is_tabpfn_token_limit_error(probe_error):
                        return False, format_tabpfn_token_limit_notice(
                            f"TabPFN runtime preflight hit a token limit after interactive key setup. Error: {probe_error}"
                        )
                    return False, f"TabPFN runtime preflight failed after interactive key setup: {probe_error}"
                return False, f"TabPFN authentication/setup failed after interactive key entry: {token_error}"
        return False, (
            "No PRIORLABS_API_KEY/TABPFN_API_KEY was provided and no existing local client token could be used. "
            "TabPFN will be disabled for this run."
        )

    probe_ok, probe_error = _probe_tabpfn_runtime_ready()
    if probe_ok:
        return True, "TabPFN preflight passed using the active TabPFN client credentials."
    if is_tabpfn_token_limit_error(token_error) or is_tabpfn_token_limit_error(probe_error):
        return False, format_tabpfn_token_limit_notice(token_error or probe_error)
    return False, f"TabPFN authentication/setup failed: {token_error or probe_error}"


TABPFN_DAILY_TOKEN_BUDGET = 100_000_000
TABPFN_DAILY_RESET_NOTE = (
    "Prior Labs TabPFN API budget: 100,000,000 tokens per user per day "
    "(tokens ~ rows * columns * estimators), with a daily reset."
)


def tabpfn_estimators_per_dataset_run(args: argparse.Namespace) -> int:
    # evaluate_model() performs cross_validate (k fits) + cross_val_predict (k fits) + final fit (1 fit)
    return max(1, (2 * int(getattr(args, "cv_folds", 5))) + 1)


def estimate_tabpfn_tokens(rows: int, columns: int, estimators: int) -> int:
    return int(max(0, int(rows)) * max(0, int(columns)) * max(1, int(estimators)))


def is_tabpfn_token_limit_error(error_like: Any) -> bool:
    text = str(error_like or "").strip().lower()
    markers = (
        "token",
        "quota",
        "budget",
        "limit exceeded",
        "rate limit",
        "429",
        "too many requests",
    )
    return any(marker in text for marker in markers)


def format_tabpfn_token_limit_notice(base_error: str) -> str:
    base_text = str(base_error or "").strip()
    if not base_text:
        base_text = "TabPFN request failed due to API token budget limits."
    return f"{base_text} {TABPFN_DAILY_RESET_NOTE}"


def estimate_tabpfn_daily_dataset_capacity(
    datasets: list["DatasetSpec"],
    args: argparse.Namespace,
) -> dict[str, Any]:
    estimators = tabpfn_estimators_per_dataset_run(args)
    rows_payload = []
    sortable_tokens = []
    for spec in datasets:
        n_rows = int(len(spec.frame))
        est_train_rows = max(1, int(round(n_rows * (1.0 - float(getattr(args, "test_fraction", 0.2))))))
        max_features_cfg = int(getattr(args, "max_selected_features", 512))
        if max_features_cfg > 0:
            est_columns = int(max_features_cfg)
        else:
            est_columns = max(32, int(round(0.1 * est_train_rows)))
        est_tokens = estimate_tabpfn_tokens(est_train_rows, est_columns, estimators)
        rows_payload.append(
            {
                "dataset": str(spec.name),
                "estimated_train_rows": est_train_rows,
                "estimated_columns": est_columns,
                "estimated_estimators": estimators,
                "estimated_tabpfn_tokens": est_tokens,
                "fits_single_day_budget": bool(est_tokens <= TABPFN_DAILY_TOKEN_BUDGET),
            }
        )
        sortable_tokens.append(est_tokens)
    individually_fit_count = sum(1 for value in sortable_tokens if value <= TABPFN_DAILY_TOKEN_BUDGET)
    cumulative = 0
    cumulative_count = 0
    for value in sorted(sortable_tokens):
        if cumulative + value > TABPFN_DAILY_TOKEN_BUDGET:
            break
        cumulative += value
        cumulative_count += 1
    return {
        "table": pd.DataFrame(rows_payload),
        "individually_fit_count": int(individually_fit_count),
        "smallest_first_count": int(cumulative_count),
        "estimators_per_dataset": int(estimators),
    }


@dataclass
class DatasetSpec:
    name: str
    source: str
    frame: pd.DataFrame
    smiles_column: str
    target_column: str
    recommended_split: str | None = None
    recommended_metric: str | None = None
    benchmark_suite: str | None = None
    benchmark_id: str | None = None
    leaderboard_url: str | None = None
    leaderboard_summary: dict[str, Any] | None = None
    predefined_split_column: str | None = None
    auxiliary_feature_columns: list[str] | None = None


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


def order_datasets_smallest_first(datasets: list["DatasetSpec"]) -> list["DatasetSpec"]:
    def _dataset_size(spec: "DatasetSpec") -> int:
        frame = getattr(spec, "frame", None)
        if isinstance(frame, pd.DataFrame):
            return int(len(frame))
        try:
            return int(len(frame))
        except Exception:
            return int(10**12)

    return sorted(
        list(datasets),
        key=lambda spec: (
            _dataset_size(spec),
            str(getattr(spec, "name", "")).strip().lower(),
        ),
    )


def local_timestamp_text() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def estimate_elasticnet_selector_seconds_from_dataset_size(
    dataset_size: int,
    *,
    log10_slope: float,
    log10_intercept: float,
) -> float:
    size = int(dataset_size)
    if size <= 1:
        return 0.0
    slope = float(log10_slope)
    intercept = float(log10_intercept)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.0:
        return float("nan")
    try:
        return float(10 ** (intercept + slope * math.log10(float(size))))
    except Exception:
        return float("nan")


def elasticnet_selector_timeout_dataset_size_threshold(
    timeout_seconds: float,
    *,
    log10_slope: float,
    log10_intercept: float,
) -> float:
    timeout = float(timeout_seconds)
    slope = float(log10_slope)
    intercept = float(log10_intercept)
    if timeout <= 0.0 or not np.isfinite(timeout):
        return float("nan")
    if slope <= 0.0 or not np.isfinite(slope) or not np.isfinite(intercept):
        return float("nan")
    try:
        return float(10 ** ((math.log10(timeout) - intercept) / slope))
    except Exception:
        return float("nan")


def smiles_hash(smiles_values: pd.Series | list[str]) -> str:
    smiles_list = [str(item).strip() for item in list(smiles_values)]
    payload = "\n".join(smiles_list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_split_signature(smiles_train: pd.Series, smiles_test: pd.Series) -> dict[str, Any]:
    train_series = pd.Series(smiles_train, dtype=str).str.strip().reset_index(drop=True)
    test_series = pd.Series(smiles_test, dtype=str).str.strip().reset_index(drop=True)
    return {
        "train_count": int(len(train_series)),
        "test_count": int(len(test_series)),
        "train_hash": smiles_hash(train_series),
        "test_hash": smiles_hash(test_series),
    }


def benchmark_config_signature(args: argparse.Namespace) -> str:
    config = vars(args).copy()
    config.pop("output_dir", None)
    config.pop("dry_run", None)
    config.pop("ga_resolution", None)
    config["default_feature_families"] = list(DEFAULT_BENCHMARK_FEATURE_FAMILIES)
    return json.dumps(config, sort_keys=True, default=str)


STAGE23_RESUME_CACHE_VERSION = 1
SHARED_FEATURE_MATRIX_CACHE_VERSION = 1
MAPLIGHT_CLASSIC_PREFIXES = ("maplight_morgan_", "avalon_count_", "erg_", "maplight_desc_")
MAPLIGHT_CATBOOST_LABEL_LEGACY = "MapLight CatBoost"
MAPLIGHT_GNN_LABEL_LEGACY = "MapLight + GNN (CatBoost)"
MAPLIGHT_CATBOOST_LABEL_STRICT = "MapLight CatBoost (Strict Parity)"
MAPLIGHT_GNN_LABEL_STRICT = "MapLight + GNN (CatBoost, Strict Parity)"
MAPLIGHT_PARITY_SEEDS_DEFAULT = [1, 2, 3, 4, 5]
_MAPLIGHT_EMBEDDER_CACHE: dict[str, tuple[Callable[[list[str]], list[Any]], str]] = {}


def _stable_float_text(value: Any) -> str:
    try:
        value_float = float(value)
        if not np.isfinite(value_float):
            return "nan"
        return f"{value_float:.12g}"
    except Exception:
        return str(value)


def dataset_content_signature(
    canonical_smiles: pd.Series,
    target_values: pd.Series,
    predefined_split: pd.Series | None = None,
) -> str:
    smiles_series = pd.Series(canonical_smiles, dtype=str).str.strip().reset_index(drop=True)
    target_series = pd.Series(target_values, dtype=float).reset_index(drop=True)
    split_series = None
    if predefined_split is not None:
        split_series = pd.Series(predefined_split, dtype=str).str.strip().str.lower().reset_index(drop=True)
        if len(split_series) != len(smiles_series):
            split_series = None

    hasher = hashlib.sha256()
    for idx in range(len(smiles_series)):
        hasher.update(str(smiles_series.iloc[idx]).encode("utf-8", errors="replace"))
        hasher.update(b"\t")
        hasher.update(_stable_float_text(target_series.iloc[idx]).encode("utf-8", errors="replace"))
        if split_series is not None:
            hasher.update(b"\t")
            hasher.update(str(split_series.iloc[idx]).encode("utf-8", errors="replace"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def stage23_resume_signature(
    *,
    args: argparse.Namespace,
    spec: "DatasetSpec",
    canonical_df: pd.DataFrame,
    input_meta: dict[str, Any],
    predefined_split: pd.Series | None,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "cache_version": int(STAGE23_RESUME_CACHE_VERSION),
        "dataset_name": str(spec.name),
        "dataset_source": str(spec.source),
        "dataset_rows": int(len(canonical_df)),
        "dataset_content_hash": dataset_content_signature(
            canonical_df["canonical_smiles"],
            canonical_df["target"],
            predefined_split=predefined_split,
        ),
        "target_transform": str(input_meta.get("target_transform", "")),
        "smiles_column": str(input_meta.get("smiles_column", "")),
        "target_column": str(input_meta.get("target_column", "")),
        "feature_families": list(DEFAULT_BENCHMARK_FEATURE_FAMILIES),
        "fingerprint_bits": int(getattr(args, "fingerprint_bits", 1024)),
        "enable_persistent_feature_store": bool(getattr(args, "enable_persistent_feature_store", True)),
        "reuse_persistent_feature_store": bool(getattr(args, "reuse_persistent_feature_store", True)),
        "persistent_feature_store_path": str(getattr(args, "persistent_feature_store_path", "AUTO")),
        "split_strategy_requested": str(getattr(args, "split_strategy", "target_quartiles")),
        "test_fraction": float(getattr(args, "test_fraction", 0.2)),
        "random_seed": int(getattr(args, "random_seed", 13)),
        "selector_method": str(getattr(args, "selector_method", "elasticnet_cv")),
        "selector_l1_ratio_grid": str(getattr(args, "selector_l1_ratio_grid", "")),
        "selector_alpha_min_log10": float(getattr(args, "selector_alpha_min_log10", -5)),
        "selector_alpha_max_log10": float(getattr(args, "selector_alpha_max_log10", -1)),
        "selector_alpha_grid_size": int(getattr(args, "selector_alpha_grid_size", 12)),
        "selector_cv_folds": int(getattr(args, "selector_cv_folds", 3)),
        "selector_max_iter": int(getattr(args, "selector_max_iter", 10000)),
        "selector_coefficient_threshold": float(getattr(args, "selector_coefficient_threshold", 1e-10)),
        "selector_auto_rf_by_dataset_size": bool(getattr(args, "selector_auto_rf_by_dataset_size", True)),
        "selector_auto_rf_threshold_seconds": float(getattr(args, "selector_auto_rf_threshold_seconds", 7200.0)),
        "selector_auto_rf_log10_slope": float(getattr(args, "selector_auto_rf_log10_slope", 1.225)),
        "selector_auto_rf_log10_intercept": float(getattr(args, "selector_auto_rf_log10_intercept", -0.658)),
        "selector_elasticnet_timeout_seconds": float(getattr(args, "selector_elasticnet_timeout_seconds", 7200.0)),
        "selector_rf_fallback_n_estimators": int(getattr(args, "selector_rf_fallback_n_estimators", 400)),
        "max_selected_features": int(getattr(args, "max_selected_features", 512)),
        "dedup_variance_threshold": 1e-8,
        "dedup_binary_prevalence_min": 0.005,
        "dedup_binary_prevalence_max": 0.995,
        "predefined_split_column": str(spec.predefined_split_column or ""),
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return signature, payload


def stage23_resume_cache_path(dataset_dir: Path) -> Path:
    return dataset_dir / "stage23_resume_cache.pkl"


def load_stage23_resume_cache(dataset_dir: Path, expected_signature: str) -> dict[str, Any] | None:
    cache_path = stage23_resume_cache_path(dataset_dir)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("cache_version", -1)) != int(STAGE23_RESUME_CACHE_VERSION):
        return None
    if str(payload.get("stage23_signature", "")).strip() != str(expected_signature).strip():
        return None
    required_keys = {
        "split",
        "X_train_selected",
        "X_test_selected",
        "feature_meta",
        "selector_meta",
        "feature_dedup_meta",
        "maplight_feature_cols",
        "cv_strategy_for_workflows",
    }
    if not required_keys.issubset(set(payload.keys())):
        return None
    return payload


def write_stage23_resume_cache(
    dataset_dir: Path,
    *,
    stage23_signature_value: str,
    signature_payload: dict[str, Any],
    split: dict[str, Any],
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
    feature_meta: dict[str, Any],
    selector_meta: dict[str, Any],
    feature_dedup_meta: dict[str, Any],
    maplight_feature_cols: list[str],
    cv_strategy_for_workflows: str,
) -> None:
    payload = {
        "cache_version": int(STAGE23_RESUME_CACHE_VERSION),
        "stage23_signature": str(stage23_signature_value),
        "signature_payload": dict(signature_payload),
        "split": {
            "X_train": split["X_train"].copy(),
            "X_test": split["X_test"].copy(),
            "y_train": pd.Series(split["y_train"]).reset_index(drop=True),
            "y_test": pd.Series(split["y_test"]).reset_index(drop=True),
            "smiles_train": pd.Series(split["smiles_train"], dtype=str).reset_index(drop=True),
            "smiles_test": pd.Series(split["smiles_test"], dtype=str).reset_index(drop=True),
            "split_strategy_used": str(split.get("split_strategy_used", "")),
            "split_signature": dict(split.get("split_signature", {})),
        },
        "X_train_selected": X_train_selected.copy(),
        "X_test_selected": X_test_selected.copy(),
        "feature_meta": dict(feature_meta),
        "selector_meta": dict(selector_meta),
        "feature_dedup_meta": dict(feature_dedup_meta),
        "maplight_feature_cols": [str(col) for col in list(maplight_feature_cols)],
        "cv_strategy_for_workflows": str(cv_strategy_for_workflows),
        "created_at": local_timestamp_text(),
    }
    cache_path = stage23_resume_cache_path(dataset_dir)
    try:
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        print(f"[warn] failed to write stage 2/3 resume cache for {dataset_dir.name}: {exc}", flush=True)


def default_shared_feature_matrix_cache_path() -> Path:
    return Path(__file__).resolve().parents[1] / "model_cache" / "benchmark_feature_matrix_cache"


def resolve_shared_feature_matrix_cache_path(cache_path: str | Path = "AUTO") -> Path:
    path_text = str(cache_path).strip()
    if not path_text or path_text.upper() == "AUTO":
        return default_shared_feature_matrix_cache_path()
    return Path(path_text)


def shared_feature_matrix_signature(
    *,
    smiles_values: pd.Series,
    selected_families: list[str],
    radius: int,
    n_bits: int,
) -> tuple[str, dict[str, Any]]:
    smiles_series = pd.Series(smiles_values, dtype=str).reset_index(drop=True)
    payload = {
        "cache_version": int(SHARED_FEATURE_MATRIX_CACHE_VERSION),
        "feature_families": [str(item) for item in list(selected_families)],
        "radius": int(radius),
        "fingerprint_bits": int(n_bits),
        "n_rows": int(len(smiles_series)),
        "smiles_hash": smiles_hash(smiles_series),
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return signature, payload


def shared_feature_matrix_cache_file(cache_root: Path, signature: str) -> Path:
    return Path(cache_root) / f"{str(signature).strip()}.pkl"


def load_shared_feature_matrix_cache(
    cache_root: Path,
    signature: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    cache_file = shared_feature_matrix_cache_file(cache_root, signature)
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("cache_version", -1)) != int(SHARED_FEATURE_MATRIX_CACHE_VERSION):
        return None
    if str(payload.get("signature", "")).strip() != str(signature).strip():
        return None
    feature_matrix = payload.get("feature_matrix")
    feature_meta = payload.get("feature_meta")
    if not isinstance(feature_matrix, pd.DataFrame):
        return None
    if not isinstance(feature_meta, dict):
        feature_meta = {}
    feature_meta = dict(feature_meta)
    feature_meta["shared_feature_matrix_cache_hit"] = True
    feature_meta["shared_feature_matrix_cache_file"] = str(cache_file)
    return feature_matrix.copy(), feature_meta


def write_shared_feature_matrix_cache(
    cache_root: Path,
    signature: str,
    signature_payload: dict[str, Any],
    feature_matrix: pd.DataFrame,
    feature_meta: dict[str, Any],
) -> None:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = shared_feature_matrix_cache_file(cache_root, signature)
    payload = {
        "cache_version": int(SHARED_FEATURE_MATRIX_CACHE_VERSION),
        "signature": str(signature),
        "signature_payload": dict(signature_payload),
        "feature_matrix": feature_matrix.copy(),
        "feature_meta": dict(feature_meta),
        "created_at": local_timestamp_text(),
    }
    try:
        with cache_file.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        print(f"[warn] failed to write shared feature matrix cache ({cache_file}): {exc}", flush=True)


def parse_comma_list(text: Any) -> list[str]:
    return [str(token).strip() for token in str(text or "").split(",") if str(token).strip()]


def _normalize_workflow_label(workflow_name: Any) -> str:
    return str(workflow_name or "").strip().lower()


def cfa_source_workflow_filter(args: argparse.Namespace) -> set[str] | None:
    tokens = [_normalize_workflow_label(token) for token in parse_comma_list(getattr(args, "cfa_source_workflows", "all"))]
    tokens = [token for token in tokens if token]
    if not tokens or any(token in {"all", "*", "any"} for token in tokens):
        return None
    return set(tokens)


def maplight_catboost_model_label(args: argparse.Namespace) -> str:
    if bool(getattr(args, "maplight_leaderboard_parity_mode", True)):
        return MAPLIGHT_CATBOOST_LABEL_STRICT
    return MAPLIGHT_CATBOOST_LABEL_LEGACY


def maplight_gnn_model_label(args: argparse.Namespace) -> str:
    if bool(getattr(args, "maplight_leaderboard_parity_mode", True)):
        return MAPLIGHT_GNN_LABEL_STRICT
    return MAPLIGHT_GNN_LABEL_LEGACY


def maplight_parity_seed_values(args: argparse.Namespace) -> list[int]:
    seed_text = str(getattr(args, "maplight_parity_seeds", "")).strip()
    values: list[int] = []
    for token in parse_comma_list(seed_text):
        try:
            values.append(int(token))
        except Exception:
            continue
    if not values:
        values = list(MAPLIGHT_PARITY_SEEDS_DEFAULT)
    # Preserve input ordering but drop duplicates.
    deduped: list[int] = []
    seen: set[int] = set()
    for seed_value in values:
        if seed_value in seen:
            continue
        seen.add(seed_value)
        deduped.append(int(seed_value))
    return deduped


def build_maplight_parity_matrix(smiles_values: pd.Series, args: argparse.Namespace) -> pd.DataFrame:
    smiles_series = pd.Series(smiles_values, dtype=str).reset_index(drop=True)
    matrix = make_maplight_classic_matrix(
        smiles_series.tolist(),
        radius=2,
        n_bits=int(getattr(args, "fingerprint_bits", 1024)),
    )
    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    matrix.reset_index(drop=True, inplace=True)
    return matrix


def evaluate_maplight_seeded_catboost(
    *,
    model_name: str,
    workflow_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed_values: list[int],
    feature_source: str,
    primary_metric: str = "mae",
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost is unavailable.")
    if not seed_values:
        raise ValueError("MapLight parity requires at least one seed.")

    train_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    for seed_value in seed_values:
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    CatBoostRegressor(
                        loss_function="MAE",
                        eval_metric="MAE",
                        random_seed=int(seed_value),
                        verbose=False,
                    ),
                ),
            ]
        )
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        train_preds.append(np.asarray(fitted.predict(X_train), dtype=float).reshape(-1))
        test_preds.append(np.asarray(fitted.predict(X_test), dtype=float).reshape(-1))

    pred_train = np.mean(np.vstack(train_preds), axis=0)
    pred_test = np.mean(np.vstack(test_preds), axis=0)
    primary_metric_value = compute_primary_metric(str(primary_metric), y_test, pred_test)
    row: dict[str, Any] = {
        "model": str(model_name),
        "workflow": str(workflow_name),
        "cv_folds": int(max(1, len(seed_values))),
        "cv_split_strategy": "seed_ensemble_no_cv",
        "cv_r2": np.nan,
        "cv_rmse": np.nan,
        "cv_mae": np.nan,
        "primary_metric": str(primary_metric),
        "cv_primary": np.nan,
        "primary_metric_value": float(primary_metric_value),
        "maplight_parity_mode": "strict",
        "maplight_seed_count": int(len(seed_values)),
        "maplight_seed_values": ",".join(str(int(seed_value)) for seed_value in seed_values),
        "maplight_loss_function": "MAE",
        "maplight_scaler": "StandardScaler",
        "maplight_feature_source": str(feature_source),
    }
    row.update(regression_metrics(y_train, pred_train, y_test, pred_test))
    row = add_leaderboard_reference_columns(
        row,
        primary_metric=str(primary_metric),
        primary_metric_value=float(primary_metric_value),
    )
    return row, pred_train, pred_test


def discover_recent_benchmark_runs(root: Path, *, exclude_dir: Path | None = None) -> list[Path]:
    benchmark_root = root / "benchmark_results"
    if not benchmark_root.exists():
        return []
    excluded = exclude_dir.resolve() if exclude_dir is not None else None
    candidates: list[tuple[float, Path]] = []
    for candidate in benchmark_root.iterdir():
        if not candidate.is_dir():
            continue
        if excluded is not None:
            try:
                if candidate.resolve() == excluded:
                    continue
            except Exception:
                pass
        run_config = candidate / "run_config.json"
        summary = candidate / "summary_metrics.csv"
        if run_config.exists():
            timestamp = run_config.stat().st_mtime
        elif summary.exists():
            timestamp = summary.stat().st_mtime
        else:
            metrics_files = list(candidate.glob("*/metrics.csv"))
            if not metrics_files:
                continue
            timestamp = max(path.stat().st_mtime for path in metrics_files)
        candidates.append((float(timestamp), candidate))
    return [path for _timestamp, path in sorted(candidates, key=lambda item: item[0], reverse=True)]


def load_run_config_payload(run_dir: Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_run_metrics_dataframe(run_dir: Path) -> pd.DataFrame:
    run_root = Path(run_dir)
    summary_path = run_root / "summary_metrics.csv"
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            if not summary_df.empty:
                return summary_df
        except Exception:
            pass

    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(run_root.glob("*/metrics.csv")):
        try:
            dataset_df = pd.read_csv(metrics_path)
        except Exception:
            continue
        dataset_name = metrics_path.parent.name
        if "dataset" not in dataset_df.columns:
            dataset_df.insert(0, "dataset", dataset_name)
        else:
            dataset_df["dataset"] = dataset_df["dataset"].where(
                dataset_df["dataset"].astype(str).str.strip().ne(""),
                dataset_name,
            )
        rows.append(dataset_df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _error_rank_from_metrics(metrics_df: pd.DataFrame) -> pd.Series:
    if metrics_df.empty or "error" not in metrics_df.columns:
        return pd.Series(0, index=metrics_df.index, dtype=int)
    error_series = metrics_df["error"].fillna("").astype(str).str.strip().str.lower()
    has_error = error_series.ne("") & ~error_series.isin({"nan", "none"})
    return has_error.astype(int)


def _status_rank_from_metrics(metrics_df: pd.DataFrame) -> pd.Series:
    if metrics_df.empty or "status" not in metrics_df.columns:
        return pd.Series(0, index=metrics_df.index, dtype=int)
    status_series = metrics_df["status"].fillna("").astype(str).str.strip().str.lower()
    is_bad_status = status_series.str.startswith("skipped") | status_series.str.contains("error", regex=False)
    return is_bad_status.astype(int)


def _rmse_rank_from_metrics(metrics_df: pd.DataFrame) -> pd.Series:
    if metrics_df.empty:
        return pd.Series(np.inf, index=metrics_df.index, dtype=float)
    for candidate in ["test_rmse", "rmse", "primary_metric_value", "cv_rmse", "train_rmse"]:
        if candidate in metrics_df.columns:
            values = pd.to_numeric(metrics_df[candidate], errors="coerce")
            return values.fillna(np.inf)
    return pd.Series(np.inf, index=metrics_df.index, dtype=float)


def deduplicate_metrics_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df.copy()
    working = metrics_df.copy()
    if "dataset" not in working.columns or "model" not in working.columns:
        return working
    if "workflow" not in working.columns:
        working["workflow"] = ""
    working["dataset"] = working["dataset"].fillna("").astype(str).str.strip()
    working["model"] = working["model"].fillna("").astype(str).str.strip()
    working["workflow"] = working["workflow"].fillna("").astype(str).str.strip()
    working = working.loc[working["dataset"].ne("") & working["model"].ne("")].copy()
    if working.empty:
        return working
    working["_error_rank"] = _error_rank_from_metrics(working)
    working["_status_rank"] = _status_rank_from_metrics(working)
    working["_rmse_rank"] = _rmse_rank_from_metrics(working)
    working["_row_order"] = np.arange(len(working), dtype=int)
    # Prefer: non-error rows, then non-skipped status rows, then finite/lower RMSE, then latest row.
    working = working.sort_values(
        by=["dataset", "model", "workflow", "_error_rank", "_status_rank", "_rmse_rank", "_row_order"],
        ascending=[True, True, True, True, True, True, False],
    )
    deduped = working.groupby(["dataset", "model", "workflow"], as_index=False, dropna=False).first()
    deduped = deduped.drop(columns=["_error_rank", "_status_rank", "_rmse_rank", "_row_order"], errors="ignore")
    return deduped


def build_summary_from_dataset_metrics(output_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(Path(output_dir).glob("*/metrics.csv")):
        try:
            dataset_df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if dataset_df.empty:
            continue
        dataset_name = metrics_path.parent.name
        if "dataset" not in dataset_df.columns:
            dataset_df.insert(0, "dataset", dataset_name)
        else:
            dataset_df["dataset"] = dataset_df["dataset"].where(
                dataset_df["dataset"].astype(str).str.strip().ne(""),
                dataset_name,
            )
        rows.append(dataset_df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = deduplicate_metrics_rows(combined)
    if not combined.empty and {"dataset", "model", "workflow"}.issubset(set(combined.columns)):
        combined = combined.sort_values(["dataset", "model", "workflow"], kind="stable").reset_index(drop=True)
    return combined


def error_mask(metrics_df: pd.DataFrame) -> pd.Series:
    if metrics_df.empty:
        return pd.Series([], dtype=bool)
    if "error" not in metrics_df.columns:
        return pd.Series(False, index=metrics_df.index)
    error_series = metrics_df["error"].fillna("").astype(str).str.strip()
    return error_series.ne("") & ~error_series.str.lower().isin({"nan", "none"})


def meaningful_ga_models_from_reference(
    metrics_df: pd.DataFrame,
    *,
    min_relative_improvement: float = 0.005,
    min_dataset_wins: int = 1,
    min_improving_datasets: int = 1,
) -> tuple[list[str], dict[str, Any]]:
    if metrics_df.empty or "dataset" not in metrics_df.columns or "model" not in metrics_df.columns:
        return [], {"reason": "missing_metrics"}

    working = metrics_df.copy()
    working = working.loc[~error_mask(working)].copy()
    if working.empty:
        return [], {"reason": "all_rows_have_errors"}
    if "primary_metric_value" not in working.columns:
        return [], {"reason": "missing_primary_metric_value"}

    working["primary_metric_value"] = pd.to_numeric(working["primary_metric_value"], errors="coerce")
    working = working.loc[working["primary_metric_value"].notna()].copy()
    if working.empty:
        return [], {"reason": "no_numeric_primary_metric_values"}

    wins_by_model: Counter[str] = Counter()
    improvements_by_model: dict[str, set[str]] = defaultdict(set)
    improvement_details: list[dict[str, Any]] = []

    for dataset_name, dataset_df in working.groupby("dataset", sort=False):
        if dataset_df.empty:
            continue
        metric_name = normalize_benchmark_metric(
            str(dataset_df.get("primary_metric", pd.Series(["rmse"])).dropna().iloc[0]),
            fallback="rmse",
        )
        lower_is_better = metric_lower_is_better(metric_name)
        if lower_is_better is None:
            lower_is_better = True

        values = pd.to_numeric(dataset_df["primary_metric_value"], errors="coerce")
        if values.notna().sum() == 0:
            continue
        best_idx = values.idxmin() if lower_is_better else values.idxmax()
        best_row = dataset_df.loc[best_idx]
        best_workflow = str(best_row.get("workflow", "")).strip().lower()
        best_model = str(best_row.get("model", "")).strip()
        if best_workflow == "ga_tuned" and best_model.endswith(" GA"):
            wins_by_model[best_model[:-3].strip()] += 1

        ga_rows = dataset_df.loc[
            dataset_df.get("workflow", pd.Series("", index=dataset_df.index)).astype(str).str.strip().str.lower().eq("ga_tuned")
        ].copy()
        if ga_rows.empty:
            continue

        for _idx, ga_row in ga_rows.iterrows():
            ga_model_label = str(ga_row.get("model", "")).strip()
            if not ga_model_label.endswith(" GA"):
                continue
            base_model = ga_model_label[:-3].strip()
            if not base_model:
                continue
            base_rows = dataset_df.loc[
                dataset_df.get("model", pd.Series("", index=dataset_df.index)).astype(str).str.strip().eq(base_model)
            ].copy()
            if base_rows.empty:
                continue
            base_values = pd.to_numeric(base_rows["primary_metric_value"], errors="coerce").dropna()
            if base_values.empty:
                continue
            base_best = float(base_values.min() if lower_is_better else base_values.max())
            ga_value = float(pd.to_numeric(ga_row.get("primary_metric_value"), errors="coerce"))
            if not math.isfinite(base_best) or not math.isfinite(ga_value):
                continue
            improvement = (base_best - ga_value) if lower_is_better else (ga_value - base_best)
            denom = abs(base_best) if abs(base_best) > 1e-12 else 1.0
            relative_improvement = float(improvement / denom)
            if improvement > 0 and relative_improvement >= float(min_relative_improvement):
                improvements_by_model[base_model].add(str(dataset_name))
                improvement_details.append(
                    {
                        "dataset": str(dataset_name),
                        "model": base_model,
                        "base_value": base_best,
                        "ga_value": ga_value,
                        "relative_improvement": relative_improvement,
                        "metric": metric_name,
                    }
                )

    candidate_models = sorted(
        {
            model_name
            for model_name in set(wins_by_model.keys()) | set(improvements_by_model.keys())
            if int(wins_by_model.get(model_name, 0)) >= int(min_dataset_wins)
            or int(len(improvements_by_model.get(model_name, set()))) >= int(min_improving_datasets)
        }
    )
    diagnostics = {
        "wins_by_model": {key: int(value) for key, value in wins_by_model.items()},
        "improving_dataset_counts": {key: int(len(value)) for key, value in improvements_by_model.items()},
        "min_relative_improvement": float(min_relative_improvement),
        "improvement_examples": improvement_details[:50],
    }
    return candidate_models, diagnostics


def resolve_requested_ga_models(args: argparse.Namespace, root: Path, *, exclude_dir: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    requested_text = str(getattr(args, "ga_models", "")).strip()
    if not requested_text:
        return [], {"mode": "disabled", "reason": "empty_ga_models"}
    if requested_text.lower() != "auto":
        return parse_comma_list(requested_text), {"mode": "manual"}

    explicit_reference_dir = Path(str(getattr(args, "ga_auto_reference_run_dir", "")).strip()) if str(getattr(args, "ga_auto_reference_run_dir", "")).strip() else None
    reference_metrics = pd.DataFrame()
    if explicit_reference_dir is not None and explicit_reference_dir.exists():
        reference_dir = explicit_reference_dir
        reference_metrics = load_run_metrics_dataframe(reference_dir)
    else:
        recent_runs = discover_recent_benchmark_runs(root, exclude_dir=exclude_dir)
        reference_dir = None
        min_reference_datasets = int(getattr(args, "ga_auto_min_reference_datasets", 5))
        for candidate in recent_runs:
            candidate_metrics = load_run_metrics_dataframe(candidate)
            dataset_count = (
                int(candidate_metrics["dataset"].astype(str).str.strip().nunique())
                if (not candidate_metrics.empty and "dataset" in candidate_metrics.columns)
                else 0
            )
            ga_row_count = (
                int(
                    candidate_metrics.get("workflow", pd.Series("", index=candidate_metrics.index))
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("ga_tuned")
                    .sum()
                )
                if not candidate_metrics.empty
                else 0
            )
            if dataset_count >= min_reference_datasets and ga_row_count > 0:
                reference_dir = candidate
                reference_metrics = candidate_metrics
                break
        if reference_dir is None and recent_runs:
            reference_dir = recent_runs[0]
            reference_metrics = load_run_metrics_dataframe(reference_dir)

    if reference_dir is None:
        return [], {"mode": "auto", "reason": "no_reference_run_found"}
    selected_models, diagnostics = meaningful_ga_models_from_reference(
        reference_metrics,
        min_relative_improvement=float(getattr(args, "ga_auto_min_relative_improvement", 0.005)),
        min_dataset_wins=int(getattr(args, "ga_auto_min_dataset_wins", 1)),
        min_improving_datasets=int(getattr(args, "ga_auto_min_improving_datasets", 1)),
    )
    diagnostics.update(
        {
            "mode": "auto",
            "reference_run_dir": str(reference_dir),
            "reference_rows": int(len(reference_metrics)),
            "reference_dataset_count": int(reference_metrics["dataset"].astype(str).str.strip().nunique()) if (not reference_metrics.empty and "dataset" in reference_metrics.columns) else 0,
            "reference_ga_row_count": int(
                reference_metrics.get("workflow", pd.Series("", index=reference_metrics.index))
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("ga_tuned")
                .sum()
            )
            if not reference_metrics.empty
            else 0,
            "selected_models": selected_models,
        }
    )
    return selected_models, diagnostics


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
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
            exc_text = str(exc)
            if name == "xyz_polarizability" and "openbabel" in exc_text.lower():
                print(
                    "[info] ChemML xyz_polarizability dataset unavailable (optional OpenBabel dependency missing); continuing.",
                    flush=True,
                )
            else:
                print(f"[skip] ChemML {name}: {exc_text}")
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

    def _normalize_tdc_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        rename_map = {}
        smiles_col = infer_column(list(normalized.columns), ["smiles", "SMILES", "Drug"])
        target_col = infer_column(list(normalized.columns), ["target", "TARGET", "Y", "y", "label", "Label"])
        if smiles_col is None:
            raise ValueError(f"could not infer SMILES column from {list(normalized.columns)}")
        if target_col is None:
            raise ValueError(f"could not infer target column from {list(normalized.columns)}")
        rename_map[smiles_col] = "smiles"
        rename_map[target_col] = "target"
        normalized = normalized.rename(columns=rename_map)
        return normalized[["smiles", "target"]].copy()

    benchmark_group = None
    benchmark_group_names: dict[str, str] = {}
    benchmark_group_error: Exception | None = None
    try:
        from tdc.benchmark_group import admet_group

        benchmark_group = admet_group(path=path)
        names = list(getattr(benchmark_group, "dataset_names", []))
        benchmark_group_names = {str(name).strip().lower(): str(name) for name in names}
    except Exception as exc:
        benchmark_group_error = exc
        benchmark_group = None

    if benchmark_group is None and benchmark_group_error is not None:
        print(
            "[info] PyTDC benchmark-group split loader unavailable; "
            f"falling back to single_pred datasets ({type(benchmark_group_error).__name__}: {benchmark_group_error})",
            flush=True,
        )

    loader_by_task = {"ADME": ADME, "Tox": Tox}
    for dataset_name, config in TDC_QSAR_OPTIONS.items():
        dataset_key = dataset_name.lower().strip()
        recommended_metric = admet_metrics.get(dataset_key) if isinstance(admet_metrics, dict) else None
        recommended_split = admet_splits.get(dataset_key) if isinstance(admet_splits, dict) else None
        frame: pd.DataFrame | None = None
        source_label = f"PyTDC {config['task']} benchmark: {dataset_name}"
        predefined_split_column: str | None = None

        benchmark_name = benchmark_group_names.get(dataset_key)
        if benchmark_group is not None and benchmark_name:
            try:
                benchmark_payload = benchmark_group.get(benchmark_name)
                train_val_df = benchmark_payload.get("train_val")
                test_df = benchmark_payload.get("test")
                if train_val_df is None or test_df is None:
                    raise ValueError("benchmark group payload missing train_val/test splits")
                train_norm = _normalize_tdc_frame_columns(pd.DataFrame(train_val_df))
                test_norm = _normalize_tdc_frame_columns(pd.DataFrame(test_df))
                train_norm["__benchmark_split"] = "train"
                test_norm["__benchmark_split"] = "test"
                frame = pd.concat([train_norm, test_norm], ignore_index=True)
                source_label = f"PyTDC {config['task']} benchmark group: {dataset_name} (official train_val/test split)"
                recommended_split = "predefined"
                predefined_split_column = "__benchmark_split"
            except Exception as exc:
                print(
                    f"[warn] PyTDC benchmark group split failed for {dataset_name}; "
                    f"falling back to single_pred loader ({type(exc).__name__}: {exc})",
                    flush=True,
                )

        try:
            if frame is None:
                loader = loader_by_task[config["task"]](name=dataset_name, path=path, print_stats=False)
                frame = loader.get_data().copy()
                frame = _normalize_tdc_frame_columns(frame)
            datasets.append(
                DatasetSpec(
                    f"tdc_{dataset_name}",
                    source_label,
                    frame,
                    "smiles",
                    "target",
                    recommended_split=recommended_split,
                    recommended_metric=recommended_metric,
                    benchmark_suite="tdc",
                    benchmark_id=dataset_name,
                    leaderboard_url=TDC_LEADERBOARD_URLS.get(dataset_key),
                    predefined_split_column=predefined_split_column,
                )
            )
        except Exception as exc:
            print(f"[skip] PyTDC {dataset_name}: {exc}")
    return datasets


def download_csv_with_cache(url: str, destination: Path) -> pd.DataFrame:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    return pd.read_csv(destination, low_memory=False)


def _normalize_freesolv_split_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"train", "training", "validation", "valid", "val"}:
        return "train"
    if text in {"test", "testing", "holdout"}:
        return "test"
    return ""


def extract_freesolv_expanded_scaled_paper_summary(workbook_path: Path) -> dict[str, Any] | None:
    try:
        model_benchmarking = pd.read_excel(workbook_path, sheet_name=str(FREESOLV_EXPANDED_SCALED_OPTION.get("sheet_name", "Model Benchmarking")))
    except Exception:
        return None
    if model_benchmarking is None or model_benchmarking.empty or model_benchmarking.shape[1] < 13:
        return None

    metrics_block = model_benchmarking.iloc[:, 10:13].copy()
    metrics_block.columns = ["r2", "mae", "percent_ard"]
    metrics_numeric = metrics_block.apply(pd.to_numeric, errors="coerce")

    non_scaled_idx = metrics_numeric["mae"].dropna().index.min()
    if non_scaled_idx is None:
        return None

    scaled_idx = None
    label_series = model_benchmarking.iloc[:, 10].astype(str)
    for idx, label_text in label_series.items():
        normalized = str(label_text).strip().lower()
        if "ln(-" in normalized and "1/t" in normalized:
            following = metrics_numeric.loc[idx + 1 :, "mae"].dropna()
            if not following.empty:
                scaled_idx = int(following.index[0])
            break
    if scaled_idx is None:
        numeric_indices = list(metrics_numeric["mae"].dropna().index)
        if len(numeric_indices) >= 2:
            scaled_idx = int(numeric_indices[1])
        else:
            scaled_idx = int(non_scaled_idx)

    references: list[dict[str, Any]] = []
    for model_name, idx in [
        ("Temperature-Dependent Model (scaled; ln(-Δsolvgpuresat) vs 1/T)", int(scaled_idx)),
        ("Temperature-Dependent Model (non-scaled features)", int(non_scaled_idx)),
    ]:
        row = metrics_numeric.loc[idx]
        mae = float(row["mae"]) if pd.notna(row["mae"]) else np.nan
        r2 = float(row["r2"]) if pd.notna(row["r2"]) else np.nan
        if not np.isfinite(mae):
            continue
        model_text = model_name if not np.isfinite(r2) else f"{model_name} (R2={r2:.4f})"
        references.append(
            {
                "rank": "",
                "model": model_text,
                "metric_name": "MAE / kcal mol-1",
                "metric_value": f"{mae:.6f}",
            }
        )

    if not references:
        return None
    references = sorted(
        references,
        key=lambda item: parse_first_float(item.get("metric_value")) if parse_first_float(item.get("metric_value")) is not None else float("inf"),
    )
    for rank_idx, entry in enumerate(references, start=1):
        entry["rank"] = str(rank_idx)

    best = references[0]
    return {
        "url": str(FREESOLV_EXPANDED_SCALED_OPTION.get("benchmark_url", "")).strip(),
        "rank": str(best.get("rank", "1")),
        "model": str(best.get("model", "")).strip(),
        "metric_name": "MAE / kcal mol-1",
        "metric_value": str(best.get("metric_value", "")).strip(),
        "dataset_split": "training/validation/test (paper split; benchmark uses test holdout)",
        "top10": references[:10],
        "source": "literature_xlsx",
    }


def load_freesolv_expanded_scaled_dataset(path: str = "./data/free_solv") -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    source_file = Path(str(FREESOLV_EXPANDED_SCALED_OPTION.get("source_file", "")).strip())
    workbook_name = source_file.name if source_file.name else "1-s2.0-S0378381225000068-mmc1.xlsx"
    workbook_path = Path(path) / workbook_name
    if not workbook_path.exists():
        candidate = source_file if source_file.is_absolute() else Path(".") / source_file
        if candidate.exists():
            workbook_path = candidate
    if not workbook_path.exists():
        print(
            f"[skip] {FREESOLV_EXPANDED_SCALED_OPTION.get('dataset_name', 'freesolv_expanded_scaled_2025')}: "
            f"workbook not found at {workbook_path}"
        )
        return datasets

    benchmark_sheet = str(FREESOLV_EXPANDED_SCALED_OPTION.get("sheet_name", "Model Benchmarking"))
    split_sheet = str(FREESOLV_EXPANDED_SCALED_OPTION.get("split_sheet_name", "Temperature-Dependent Model"))
    try:
        benchmark_df = pd.read_excel(workbook_path, sheet_name=benchmark_sheet)
        split_df = pd.read_excel(workbook_path, sheet_name=split_sheet)
    except Exception as exc:
        print(f"[skip] {FREESOLV_EXPANDED_SCALED_OPTION.get('dataset_name', 'freesolv_expanded_scaled_2025')}: {exc}")
        return datasets

    if benchmark_df.shape[1] < 8 or split_df.shape[1] < 5:
        print(f"[skip] {FREESOLV_EXPANDED_SCALED_OPTION.get('dataset_name', 'freesolv_expanded_scaled_2025')}: workbook schema is not compatible.")
        return datasets

    # Scaled benchmark block (columns F-H in the supplemental workbook):
    # SMILES, 1/T, ln(-Δsolvgpuresat).
    frame = pd.DataFrame(
        {
            "smiles": benchmark_df.iloc[:, 5],
            "inverse_temperature_k_inv": pd.to_numeric(benchmark_df.iloc[:, 6], errors="coerce"),
            "target": pd.to_numeric(benchmark_df.iloc[:, 7], errors="coerce"),
            "temperature_k": pd.to_numeric(benchmark_df.iloc[:, 1], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["smiles", "inverse_temperature_k_inv", "target", "temperature_k"]).copy()
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    frame = frame.loc[~frame["smiles"].str.lower().isin({"", "none", "nan", "smiles"})].copy()

    split_frame = pd.DataFrame(
        {
            "smiles": split_df.iloc[:, 0],
            "temperature_k": pd.to_numeric(split_df.iloc[:, 1], errors="coerce"),
            "subset_raw": split_df.iloc[:, 4],
        }
    )
    split_frame = split_frame.dropna(subset=["smiles", "temperature_k", "subset_raw"]).copy()
    split_frame["smiles"] = split_frame["smiles"].astype(str).str.strip()
    split_frame = split_frame.loc[~split_frame["smiles"].str.lower().isin({"", "none", "nan", "smiles"})].copy()
    split_frame["__benchmark_split"] = split_frame["subset_raw"].apply(_normalize_freesolv_split_label)
    split_frame = split_frame.loc[split_frame["__benchmark_split"].isin({"train", "test"})].copy()

    frame["__join_key"] = frame["smiles"].astype(str) + "|" + frame["temperature_k"].round(6).astype(str)
    split_frame["__join_key"] = split_frame["smiles"].astype(str) + "|" + split_frame["temperature_k"].round(6).astype(str)
    split_lookup = split_frame[["__join_key", "__benchmark_split"]].drop_duplicates(subset=["__join_key"], keep="first")
    frame = frame.merge(split_lookup, on="__join_key", how="left")
    frame = frame.dropna(subset=["__benchmark_split"]).copy()
    frame = frame.drop(columns=["__join_key", "temperature_k"], errors="ignore")
    frame = frame.reset_index(drop=True)

    if frame.empty:
        print(f"[skip] {FREESOLV_EXPANDED_SCALED_OPTION.get('dataset_name', 'freesolv_expanded_scaled_2025')}: no valid rows after split alignment.")
        return datasets

    summary = extract_freesolv_expanded_scaled_paper_summary(workbook_path)
    datasets.append(
        DatasetSpec(
            name=str(FREESOLV_EXPANDED_SCALED_OPTION.get("dataset_name", "freesolv_expanded_scaled_2025")),
            source=str(FREESOLV_EXPANDED_SCALED_OPTION.get("source_label", "Expanded Free Solvation Energy dataset (scaled)")),
            frame=frame,
            smiles_column="smiles",
            target_column="target",
            recommended_split=str(FREESOLV_EXPANDED_SCALED_OPTION.get("recommended_split", "predefined")),
            recommended_metric=str(FREESOLV_EXPANDED_SCALED_OPTION.get("recommended_metric", "mae")),
            benchmark_suite=str(FREESOLV_EXPANDED_SCALED_OPTION.get("benchmark_suite", "literature")),
            benchmark_id=str(FREESOLV_EXPANDED_SCALED_OPTION.get("benchmark_id", "freesolv_expanded_scaled_2025")),
            leaderboard_url=str(FREESOLV_EXPANDED_SCALED_OPTION.get("benchmark_url", "")),
            leaderboard_summary=summary,
            predefined_split_column="__benchmark_split",
            auxiliary_feature_columns=["inverse_temperature_k_inv"],
        )
    )
    return datasets


def load_moleculenet_physchem_datasets(path: str = "./data/moleculenet") -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    data_root = Path(path)
    for config in MOLECULENET_PHYSCHEM_OPTIONS.values():
        try:
            csv_name = Path(str(config["source_url"]).split("?")[0]).name
            cached_csv = data_root / csv_name
            frame = download_csv_with_cache(str(config["source_url"]), cached_csv)
            smiles_column = infer_column(list(frame.columns), [str(config.get("smiles_column", "smiles"))]) or "smiles"
            target_candidates = list(config.get("target_candidates", ["target"]))
            target_column = infer_column(list(frame.columns), target_candidates)
            if target_column is None:
                raise ValueError(f"target column not found; tried {target_candidates}")
            frame = frame.rename(columns={smiles_column: "smiles", target_column: "target"})
            datasets.append(
                DatasetSpec(
                    str(config["dataset_name"]),
                    str(config["source_label"]),
                    frame,
                    "smiles",
                    "target",
                    recommended_split=str(config.get("recommended_split", "random")),
                    recommended_metric=str(config.get("recommended_metric", "rmse")),
                    benchmark_suite="moleculenet",
                    benchmark_id=str(config["dataset_name"]),
                    leaderboard_url=str(config.get("leaderboard_url") or ""),
                )
            )
        except Exception as exc:
            print(f"[skip] MoleculeNet {config.get('dataset_name', 'unknown')}: {exc}")
    return datasets


def load_polaris_adme_datasets(path: str = "./data/polaris_adme") -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    data_root = Path(path)
    for config in POLARIS_ADME_OPTIONS.values():
        try:
            train_name = Path(str(config["train_url"]).split("?")[0]).name
            test_name = Path(str(config["test_url"]).split("?")[0]).name
            train_df = download_csv_with_cache(str(config["train_url"]), data_root / train_name)
            test_df = download_csv_with_cache(str(config["test_url"]), data_root / test_name)
            smiles_column = infer_column(list(train_df.columns), [str(config.get("smiles_column", "smiles"))]) or "smiles"
            target_column = infer_column(list(train_df.columns), [str(config.get("target_column", "activity"))]) or "activity"
            if smiles_column not in test_df.columns or target_column not in test_df.columns:
                raise ValueError("train/test CSV schema mismatch for Polaris benchmark mirror")
            train_frame = train_df[[smiles_column, target_column]].copy()
            test_frame = test_df[[smiles_column, target_column]].copy()
            train_frame["__benchmark_split"] = "train"
            test_frame["__benchmark_split"] = "test"
            frame = pd.concat([train_frame, test_frame], ignore_index=True)
            frame = frame.rename(columns={smiles_column: "smiles", target_column: "target"})
            datasets.append(
                DatasetSpec(
                    str(config["dataset_name"]),
                    str(config["source_label"]),
                    frame,
                    "smiles",
                    "target",
                    recommended_split=str(config.get("recommended_split", "predefined")),
                    recommended_metric=str(config.get("recommended_metric", "mean_squared_error")),
                    benchmark_suite="polaris",
                    benchmark_id=str(config.get("benchmark_id", config["dataset_name"])),
                    leaderboard_url=str(config.get("benchmark_url") or ""),
                    predefined_split_column="__benchmark_split",
                )
            )
        except Exception as exc:
            print(f"[skip] Polaris {config.get('benchmark_id', 'unknown')}: {exc}")
    return datasets


def leaderboard_topk_rows(
    table: pd.DataFrame,
    *,
    rank_candidates: list[str],
    model_candidates: list[str],
    metric_name: str | None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if table is None or table.empty:
        return []
    rank_col = next((candidate for candidate in rank_candidates if candidate in table.columns), None)
    model_col = next((candidate for candidate in model_candidates if candidate in table.columns), None)
    metric_col = metric_name if metric_name in table.columns else None
    rows: list[dict[str, Any]] = []
    for _, row in table.head(int(max(1, top_k))).iterrows():
        rows.append(
            {
                "rank": str(row.get(rank_col, "")).strip() if rank_col else "",
                "model": str(row.get(model_col, "")).strip() if model_col else "",
                "metric_name": str(metric_name or "").strip(),
                "metric_value": str(row.get(metric_col, "")).strip() if metric_col else "",
            }
        )
    return rows


def fetch_tdc_leaderboard_best(dataset_name: str, timeout: int = 20) -> dict[str, Any] | None:
    dataset_key = str(dataset_name).strip().lower()
    leaderboard_url = TDC_LEADERBOARD_URLS.get(dataset_key)
    if not leaderboard_url:
        return None
    request = urllib.request.Request(leaderboard_url, headers={"User-Agent": "AutoQSAR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=int(timeout)) as response:
        html = response.read().decode("utf-8", errors="replace")
    tables = pd.read_html(io.StringIO(html))
    summary_table = None
    leaderboard_table = None
    for table in tables:
        cols = {str(col).strip() for col in table.columns}
        if {"Dataset", "Metric", "Dataset Split"}.issubset(cols):
            summary_table = table.copy()
        if {"Rank", "Model"}.issubset(cols):
            leaderboard_table = table.copy()
            break
    if leaderboard_table is None:
        return None
    leaderboard_table.columns = [str(col).strip() for col in leaderboard_table.columns]
    leaderboard_table = leaderboard_table.dropna(how="all")
    if leaderboard_table.empty:
        return None
    top_row = leaderboard_table.iloc[0]
    metric_name = None
    dataset_split = None
    if summary_table is not None and not summary_table.empty:
        summary_table.columns = [str(col).strip() for col in summary_table.columns]
        metric_name = str(summary_table.iloc[0].get("Metric", "")).strip() or None
        dataset_split = str(summary_table.iloc[0].get("Dataset Split", "")).strip() or None
    if metric_name is None:
        metric_name = next((col for col in leaderboard_table.columns if col not in {"Rank", "Model", "Contact", "Link", "#Params"}), None)
    metric_value = None if metric_name is None else str(top_row.get(metric_name, "")).strip()
    top10_rows = leaderboard_topk_rows(
        leaderboard_table,
        rank_candidates=["Rank"],
        model_candidates=["Model"],
        metric_name=metric_name,
        top_k=10,
    )
    return {
        "url": leaderboard_url,
        "rank": str(top_row.get("Rank", "")).strip(),
        "model": str(top_row.get("Model", "")).strip(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "dataset_split": dataset_split,
        "top10": top10_rows,
        "source": "tdc",
    }


def fetch_moleculenet_leaderboard_best(dataset_key: str, timeout: int = 20) -> dict[str, Any] | None:
    config = MOLECULENET_PHYSCHEM_OPTIONS.get(str(dataset_key))
    if not config:
        return None
    section_name = str(config.get("leaderboard_section") or "").strip()
    section_candidates = []
    if section_name:
        section_candidates.append(section_name)
    for candidate in list(config.get("leaderboard_section_candidates", [])):
        candidate_text = str(candidate).strip()
        if candidate_text and candidate_text not in section_candidates:
            section_candidates.append(candidate_text)
    if not section_candidates:
        return None
    request = urllib.request.Request(MOLECULENET_LEADERBOARD_README_URL, headers={"User-Agent": "AutoQSAR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=int(timeout)) as response:
        markdown = response.read().decode("utf-8", errors="replace")
    section_match = None
    for candidate_name in section_candidates:
        section_pattern = rf"###\s*{re.escape(candidate_name)}\s*(?P<section>.*?)(?:\n###\s+|\Z)"
        section_match = re.search(section_pattern, markdown, flags=re.IGNORECASE | re.DOTALL)
        if section_match is not None:
            break
    if section_match is None:
        return None
    section_text = str(section_match.group("section") or "")
    table_lines = [line.strip() for line in section_text.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return None
    header_tokens = [token.strip() for token in table_lines[0].strip("|").split("|")]
    raw_rows: list[list[str]] = []
    for line in table_lines[2:]:
        values = [token.strip() for token in line.strip("|").split("|")]
        if len(values) != len(header_tokens):
            continue
        raw_rows.append(values)
    if not raw_rows:
        return None
    leaderboard_table = pd.DataFrame(raw_rows, columns=header_tokens)
    rank_column = next((col for col in leaderboard_table.columns if str(col).strip().lower() == "rank"), None)
    model_column = next((col for col in leaderboard_table.columns if str(col).strip().lower() == "model"), None)
    metric_name = next((col for col in leaderboard_table.columns if "rmse" in str(col).strip().lower()), "Test RMSE")
    if rank_column and rank_column in leaderboard_table.columns:
        try:
            leaderboard_table = leaderboard_table.sort_values(
                by=rank_column,
                key=lambda series: pd.to_numeric(series, errors="coerce"),
                ascending=True,
            ).reset_index(drop=True)
        except Exception:
            leaderboard_table = leaderboard_table.reset_index(drop=True)
    top_row = leaderboard_table.iloc[0]
    top10_rows = leaderboard_topk_rows(
        leaderboard_table,
        rank_candidates=[rank_column or "Rank"],
        model_candidates=[model_column or "Model"],
        metric_name=metric_name,
        top_k=10,
    )
    return {
        "url": str(config.get("leaderboard_url") or ""),
        "rank": str(top_row.get(rank_column or "", "")).strip() if rank_column else "1",
        "model": str(top_row.get(model_column or "", "")).strip(),
        "metric_name": metric_name,
        "metric_value": str(top_row.get(metric_name, "")).strip(),
        "dataset_split": str(config.get("recommended_split", "random")),
        "top10": top10_rows,
        "source": "moleculenet",
    }


def fetch_polaris_leaderboard_best(leaderboard_url: str, timeout: int = 20) -> dict[str, Any] | None:
    if not leaderboard_url:
        return None
    request = urllib.request.Request(leaderboard_url, headers={"User-Agent": "AutoQSAR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=int(timeout)) as response:
        html = response.read().decode("utf-8", errors="replace")
    tables = pd.read_html(io.StringIO(html))
    leaderboard_table = None
    for table in tables:
        columns = [str(col).strip() for col in table.columns]
        cols = set(columns)
        if {"#", "Name"}.issubset(cols) or {"Name", "mean_squared_error"}.issubset(cols):
            leaderboard_table = table.copy()
            leaderboard_table.columns = columns
            break
    if leaderboard_table is None:
        return None
    leaderboard_table = leaderboard_table.dropna(how="all")
    if leaderboard_table.empty:
        return None
    top_row = leaderboard_table.iloc[0]
    metric_name = None
    for candidate in ["mean_squared_error", "r2", "spearmanr", "pearsonr", "mean_absolute_error"]:
        if candidate in leaderboard_table.columns:
            metric_name = candidate
            break
    if metric_name is None:
        metric_name = next((col for col in leaderboard_table.columns if col not in {"#", "Name", "Contributors", "References"}), None)
    metric_value = None if metric_name is None else str(top_row.get(metric_name, "")).strip()
    top10_rows = leaderboard_topk_rows(
        leaderboard_table,
        rank_candidates=["#", "Rank"],
        model_candidates=["Name", "Model"],
        metric_name=metric_name,
        top_k=10,
    )
    return {
        "url": leaderboard_url,
        "rank": str(top_row.get("#", "")).strip() or str(top_row.get("Rank", "")).strip(),
        "model": str(top_row.get("Name", "")).strip() or str(top_row.get("Model", "")).strip(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "dataset_split": "predefined",
        "top10": top10_rows,
        "source": "polaris",
    }


def attach_leaderboard_summary(spec: DatasetSpec) -> DatasetSpec:
    if isinstance(spec.leaderboard_summary, dict) and spec.leaderboard_summary:
        return spec
    summary: dict[str, Any] | None = None
    try:
        if spec.benchmark_suite == "tdc" and spec.benchmark_id:
            summary = fetch_tdc_leaderboard_best(spec.benchmark_id)
        elif spec.benchmark_suite == "moleculenet" and spec.benchmark_id:
            summary = fetch_moleculenet_leaderboard_best(spec.benchmark_id)
        elif spec.benchmark_suite == "polaris":
            summary = fetch_polaris_leaderboard_best(spec.leaderboard_url or "")
    except Exception as exc:
        print(f"[warn] leaderboard lookup failed for {spec.name}: {exc}")
    if not summary:
        cached_summary = cached_leaderboard_summary_for_spec(spec)
        if cached_summary:
            summary = cached_summary
            print(f"[info] leaderboard lookup fallback: using cached top10 reference for {spec.name}")
    spec.leaderboard_summary = summary
    return spec


def discover_default_example_datasets(root: Path) -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    datasets.extend(load_chemml_datasets())
    datasets.extend(load_tdc_datasets(path=str(root / "data")))
    datasets.extend(load_moleculenet_physchem_datasets(path=str(root / "data" / "moleculenet")))
    datasets.extend(load_polaris_adme_datasets(path=str(root / "data" / "polaris_adme")))
    return [attach_leaderboard_summary(item) for item in datasets]


def discover_benchmark_catalog_for_leaderboards() -> list[DatasetSpec]:
    placeholder = pd.DataFrame({"smiles": [], "target": []})
    specs: list[DatasetSpec] = []
    for dataset_name, cfg in TDC_QSAR_OPTIONS.items():
        dataset_key = str(dataset_name).strip().lower()
        specs.append(
            DatasetSpec(
                name=f"tdc_{dataset_name}",
                source=f"PyTDC {cfg['task']} benchmark: {dataset_name}",
                frame=placeholder.copy(),
                smiles_column="smiles",
                target_column="target",
                recommended_split="scaffold",
                benchmark_suite="tdc",
                benchmark_id=str(dataset_name),
                leaderboard_url=TDC_LEADERBOARD_URLS.get(dataset_key, ""),
            )
        )
    for cfg in MOLECULENET_PHYSCHEM_OPTIONS.values():
        specs.append(
            DatasetSpec(
                name=str(cfg.get("dataset_name")),
                source=str(cfg.get("source_label")),
                frame=placeholder.copy(),
                smiles_column="smiles",
                target_column="target",
                recommended_split=str(cfg.get("recommended_split", "random")),
                benchmark_suite="moleculenet",
                benchmark_id=str(cfg.get("dataset_name")),
                leaderboard_url=str(cfg.get("leaderboard_url", "")),
            )
        )
    for cfg in POLARIS_ADME_OPTIONS.values():
        specs.append(
            DatasetSpec(
                name=str(cfg.get("dataset_name")),
                source=str(cfg.get("source_label")),
                frame=placeholder.copy(),
                smiles_column="smiles",
                target_column="target",
                recommended_split="predefined",
                benchmark_suite="polaris",
                benchmark_id=str(cfg.get("benchmark_id", cfg.get("dataset_name"))),
                leaderboard_url=str(cfg.get("benchmark_url", "")),
            )
        )
    return [attach_leaderboard_summary(item) for item in specs]


def normalize_benchmark_split(recommended_split: str | None, fallback: str, allow_predefined: bool = False) -> str:
    if not recommended_split:
        return fallback
    text = str(recommended_split).strip().lower()
    if "predefined" in text or "fixed" in text:
        return "predefined" if allow_predefined else fallback
    if "scaffold" in text:
        return "scaffold"
    if "random" in text:
        return "random"
    if "strat" in text or "stratif" in text:
        return "target_quartiles"
    return fallback


def effective_split_strategy_for_dataset(
    requested_split: str,
    *,
    allow_predefined: bool,
) -> str:
    effective = str(requested_split).strip().lower()
    if CURRENT_DATASET_SPEC is not None and CURRENT_DATASET_SPEC.recommended_split:
        effective = normalize_benchmark_split(
            CURRENT_DATASET_SPEC.recommended_split,
            effective,
            allow_predefined=allow_predefined,
        )
    return effective


def effective_cv_split_strategy(split_strategy_used: str) -> str:
    split_text = str(split_strategy_used).strip().lower()
    if split_text == "predefined":
        return "random"
    return split_text


def normalize_benchmark_metric(recommended_metric: str | None, fallback: str = "rmse") -> str:
    if not recommended_metric:
        return fallback
    text = str(recommended_metric).strip().lower()
    if "mean_squared_error" in text or text == "mse":
        return "mse"
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


def current_dataset_primary_metric(fallback: str = "rmse") -> str:
    if CURRENT_DATASET_SPEC is not None and CURRENT_DATASET_SPEC.recommended_metric:
        return normalize_benchmark_metric(CURRENT_DATASET_SPEC.recommended_metric, fallback=fallback)
    return str(fallback).strip().lower()


def parse_first_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_first_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def metric_lower_is_better(metric_name: str) -> bool | None:
    metric = str(metric_name or "").strip().lower()
    if metric in {"rmse", "mae", "mse", "mean_squared_error", "mean_absolute_error"}:
        return True
    if metric in {"r2", "pearson", "pearsonr", "spearman", "spearmanr"}:
        return False
    return None


def comparable_top10_entries(leaderboard_summary: dict[str, Any], metric_norm: str) -> list[dict[str, Any]]:
    top10_entries = leaderboard_summary.get("top10", [])
    if not isinstance(top10_entries, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in top10_entries:
        if not isinstance(entry, dict):
            continue
        entry_metric = normalize_benchmark_metric(entry.get("metric_name"), fallback=metric_norm)
        if metric_norm and entry_metric != metric_norm:
            continue
        metric_value = parse_first_float(entry.get("metric_value"))
        if metric_value is None:
            continue
        rows.append(
            {
                "rank": parse_first_int(entry.get("rank")),
                "model": str(entry.get("model", "")).strip(),
                "metric_name": entry_metric,
                "metric_value": float(metric_value),
            }
        )
    return rows


def estimate_rank_vs_top10(primary_value: float, comparable_top10_values: list[float], *, lower_is_better: bool) -> tuple[int, bool, float]:
    values = [float(value) for value in comparable_top10_values if value is not None and np.isfinite(value)]
    if not values:
        return (0, False, np.nan)
    if lower_is_better:
        sorted_values = sorted(values)
        better_count = sum(1 for value in sorted_values if value < float(primary_value))
        top10_cutoff = sorted_values[min(9, len(sorted_values) - 1)]
        gap_to_cutoff = float(primary_value) - float(top10_cutoff)
    else:
        sorted_values = sorted(values, reverse=True)
        better_count = sum(1 for value in sorted_values if value > float(primary_value))
        top10_cutoff = sorted_values[min(9, len(sorted_values) - 1)]
        gap_to_cutoff = float(top10_cutoff) - float(primary_value)
    estimated_rank = int(better_count + 1)
    return (estimated_rank, bool(estimated_rank <= 10), float(gap_to_cutoff))


def cached_leaderboard_summary_for_spec(spec: DatasetSpec, cache_csv_path: Path | None = None) -> dict[str, Any] | None:
    cache_path = cache_csv_path or Path("data/benchmark_leaderboards/leaderboard_top10_reference_latest.csv")
    if not cache_path.exists():
        return None
    try:
        cache_df = pd.read_csv(cache_path)
    except Exception:
        return None
    if cache_df.empty:
        return None

    benchmark_suite = str(spec.benchmark_suite or "").strip().lower()
    benchmark_id = str(spec.benchmark_id or "").strip().lower()
    dataset_name = str(spec.name or "").strip().lower()
    if "benchmark_suite" not in cache_df.columns or "benchmark_id" not in cache_df.columns:
        return None

    cache_df["benchmark_suite"] = cache_df["benchmark_suite"].astype(str).str.strip().str.lower()
    cache_df["benchmark_id"] = cache_df["benchmark_id"].astype(str).str.strip().str.lower()
    rows = cache_df.loc[
        (cache_df["benchmark_suite"] == benchmark_suite)
        & (
            (cache_df["benchmark_id"] == benchmark_id)
            | (cache_df["benchmark_id"] == dataset_name)
        )
    ].copy()
    if rows.empty:
        return None

    rows["rank_numeric"] = rows.get("rank", pd.Series(index=rows.index, dtype=object)).apply(parse_first_int)
    rows = rows.sort_values(by=["rank_numeric"], ascending=True, na_position="last").reset_index(drop=True)
    top10_rows = rows.head(10).copy()
    if top10_rows.empty:
        return None
    top_row = top10_rows.iloc[0]
    metric_name = str(top_row.get("leaderboard_metric_name", "")).strip()
    top10_payload = []
    for _, row in top10_rows.iterrows():
        top10_payload.append(
            {
                "rank": str(row.get("rank", "")).strip(),
                "model": str(row.get("model", "")).strip(),
                "metric_name": metric_name,
                "metric_value": str(row.get("metric_value", "")).strip(),
            }
        )
    return {
        "url": str(top_row.get("leaderboard_url", spec.leaderboard_url or "")).strip(),
        "rank": str(top_row.get("rank", "")).strip(),
        "model": str(top_row.get("model", "")).strip(),
        "metric_name": metric_name,
        "metric_value": str(top_row.get("metric_value", "")).strip(),
        "dataset_split": str(top_row.get("leaderboard_dataset_split", "")).strip(),
        "top10": top10_payload,
        "source": "cache",
    }


def leaderboard_reference_rows_for_dataset(spec: DatasetSpec, captured_at: str) -> list[dict[str, Any]]:
    summary = (spec.leaderboard_summary or {}).copy()
    if not summary:
        return []
    metric_name = str(summary.get("metric_name", "")).strip()
    dataset_split = str(summary.get("dataset_split", "")).strip()
    leaderboard_url = str(summary.get("url", spec.leaderboard_url or "")).strip()
    benchmark_suite = str(spec.benchmark_suite or "").strip()
    benchmark_id = str(spec.benchmark_id or spec.name or "").strip()
    source = str(summary.get("source", "")).strip()
    rows: list[dict[str, Any]] = []

    top10_entries = summary.get("top10", [])
    if isinstance(top10_entries, list) and top10_entries:
        for entry in top10_entries:
            if not isinstance(entry, dict):
                continue
            metric_value_text = str(entry.get("metric_value", "")).strip()
            rows.append(
                {
                    "dataset": str(spec.name),
                    "dataset_source": str(spec.source),
                    "benchmark_suite": benchmark_suite,
                    "benchmark_id": benchmark_id,
                    "leaderboard_url": leaderboard_url,
                    "leaderboard_dataset_split": dataset_split,
                    "leaderboard_metric_name": metric_name,
                    "rank": str(entry.get("rank", "")).strip(),
                    "rank_numeric": parse_first_int(entry.get("rank")),
                    "model": str(entry.get("model", "")).strip(),
                    "metric_value": metric_value_text,
                    "metric_value_numeric": parse_first_float(metric_value_text),
                    "is_top10_entry": True,
                    "reference_source": source,
                    "captured_at": str(captured_at),
                }
            )
    else:
        metric_value_text = str(summary.get("metric_value", "")).strip()
        if metric_name and metric_value_text:
            rows.append(
                {
                    "dataset": str(spec.name),
                    "dataset_source": str(spec.source),
                    "benchmark_suite": benchmark_suite,
                    "benchmark_id": benchmark_id,
                    "leaderboard_url": leaderboard_url,
                    "leaderboard_dataset_split": dataset_split,
                    "leaderboard_metric_name": metric_name,
                    "rank": str(summary.get("rank", "")).strip(),
                    "rank_numeric": parse_first_int(summary.get("rank")),
                    "model": str(summary.get("model", "")).strip(),
                    "metric_value": metric_value_text,
                    "metric_value_numeric": parse_first_float(metric_value_text),
                    "is_top10_entry": False,
                    "reference_source": source,
                    "captured_at": str(captured_at),
                }
            )
    return rows


def leaderboard_reference_table(datasets: list[DatasetSpec], captured_at: str | None = None) -> pd.DataFrame:
    captured_text = str(captured_at or time.strftime("%Y-%m-%d %H:%M:%S"))
    rows: list[dict[str, Any]] = []
    for spec in datasets:
        rows.extend(leaderboard_reference_rows_for_dataset(spec, captured_at=captured_text))
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "dataset_source",
                "benchmark_suite",
                "benchmark_id",
                "leaderboard_url",
                "leaderboard_dataset_split",
                "leaderboard_metric_name",
                "rank",
                "rank_numeric",
                "model",
                "metric_value",
                "metric_value_numeric",
                "is_top10_entry",
                "reference_source",
                "captured_at",
            ]
        )
    return pd.DataFrame(rows)


def bootstrap_leaderboard_cache_from_history(root: Path) -> pd.DataFrame:
    benchmark_root = root / "benchmark_results"
    rows: list[dict[str, Any]] = []
    if not benchmark_root.exists():
        return pd.DataFrame()
    for metrics_path in benchmark_root.glob("**/metrics.csv"):
        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if df.empty:
            continue
        required_cols = {"dataset", "benchmark_suite", "benchmark_id", "leaderboard_top10_json", "leaderboard_metric_name"}
        if not required_cols.issubset(set(df.columns)):
            continue
        for _, row in df.iterrows():
            top10_json = str(row.get("leaderboard_top10_json", "")).strip()
            if not top10_json or top10_json in {"[]", "nan", "None"}:
                continue
            try:
                top10_entries = json.loads(top10_json)
            except Exception:
                continue
            if not isinstance(top10_entries, list) or not top10_entries:
                continue
            metric_name = str(row.get("leaderboard_metric_name", "")).strip()
            captured_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metrics_path.stat().st_mtime))
            for entry in top10_entries[:10]:
                if not isinstance(entry, dict):
                    continue
                metric_value_text = str(entry.get("metric_value", "")).strip()
                rows.append(
                    {
                        "dataset": str(row.get("dataset", metrics_path.parent.name)),
                        "dataset_source": str(row.get("dataset_source", "")),
                        "benchmark_suite": str(row.get("benchmark_suite", "")),
                        "benchmark_id": str(row.get("benchmark_id", row.get("dataset", ""))),
                        "leaderboard_url": str(row.get("leaderboard_url", "")),
                        "leaderboard_dataset_split": str(row.get("leaderboard_dataset_split", "")),
                        "leaderboard_metric_name": metric_name,
                        "rank": str(entry.get("rank", "")).strip(),
                        "rank_numeric": parse_first_int(entry.get("rank")),
                        "model": str(entry.get("model", "")).strip(),
                        "metric_value": metric_value_text,
                        "metric_value_numeric": parse_first_float(metric_value_text),
                        "is_top10_entry": True,
                        "reference_source": "historical_metrics",
                        "captured_at": captured_at,
                    }
                )
    if not rows:
        return pd.DataFrame()
    history_df = pd.DataFrame(rows)
    history_df = history_df.drop_duplicates(
        subset=[
            "benchmark_suite",
            "benchmark_id",
            "rank",
            "model",
            "leaderboard_metric_name",
            "metric_value",
        ],
        keep="last",
    ).reset_index(drop=True)
    return history_df


def estimate_rank_columns_for_row(
    *,
    primary_metric: str,
    primary_value: float,
    leaderboard_summary: dict[str, Any],
) -> dict[str, Any]:
    metric_norm = normalize_benchmark_metric(leaderboard_summary.get("metric_name"), fallback="")
    lower_better = metric_lower_is_better(metric_norm)
    if metric_norm != str(primary_metric).strip().lower() or lower_better is None or not np.isfinite(float(primary_value)):
        return {
            "leaderboard_estimated_rank_vs_top10": np.nan,
            "leaderboard_estimated_in_top10": np.nan,
            "leaderboard_gap_to_top10_cutoff": np.nan,
            "leaderboard_top10_cutoff_value": np.nan,
        }
    comparable_entries = comparable_top10_entries(leaderboard_summary, metric_norm)
    comparable_values = [entry["metric_value"] for entry in comparable_entries if "metric_value" in entry]
    if not comparable_values:
        return {
            "leaderboard_estimated_rank_vs_top10": np.nan,
            "leaderboard_estimated_in_top10": np.nan,
            "leaderboard_gap_to_top10_cutoff": np.nan,
            "leaderboard_top10_cutoff_value": np.nan,
        }
    rank_estimate, in_top10, gap_to_cutoff = estimate_rank_vs_top10(
        float(primary_value),
        comparable_values,
        lower_is_better=bool(lower_better),
    )
    cutoff = sorted(comparable_values)[min(9, len(comparable_values) - 1)] if lower_better else sorted(comparable_values, reverse=True)[min(9, len(comparable_values) - 1)]
    return {
        "leaderboard_estimated_rank_vs_top10": int(rank_estimate),
        "leaderboard_estimated_in_top10": bool(in_top10),
        "leaderboard_gap_to_top10_cutoff": float(gap_to_cutoff),
        "leaderboard_top10_cutoff_value": float(cutoff),
    }


def annotate_metrics_with_leaderboard(metrics_df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return metrics_df
    annotated = metrics_df.copy()
    summary = (spec.leaderboard_summary or {}).copy()
    for required_col in [
        "leaderboard_model",
        "leaderboard_metric_name",
        "leaderboard_metric_value",
        "leaderboard_rank",
        "leaderboard_url",
        "leaderboard_top10_count",
        "leaderboard_top10_json",
    ]:
        if required_col not in annotated.columns:
            annotated[required_col] = np.nan
    annotated["leaderboard_model"] = annotated["leaderboard_model"].where(annotated["leaderboard_model"].notna(), summary.get("model", ""))
    annotated["leaderboard_metric_name"] = annotated["leaderboard_metric_name"].where(annotated["leaderboard_metric_name"].notna(), summary.get("metric_name", ""))
    annotated["leaderboard_metric_value"] = annotated["leaderboard_metric_value"].where(annotated["leaderboard_metric_value"].notna(), summary.get("metric_value", ""))
    annotated["leaderboard_rank"] = annotated["leaderboard_rank"].where(annotated["leaderboard_rank"].notna(), summary.get("rank", ""))
    annotated["leaderboard_url"] = annotated["leaderboard_url"].where(annotated["leaderboard_url"].notna(), summary.get("url", spec.leaderboard_url or ""))
    top10_entries = summary.get("top10", [])
    top10_json = json.dumps(top10_entries, default=str) if isinstance(top10_entries, list) else "[]"
    top10_count = int(len(top10_entries)) if isinstance(top10_entries, list) else 0
    annotated["leaderboard_top10_count"] = annotated["leaderboard_top10_count"].where(annotated["leaderboard_top10_count"].notna(), top10_count)
    annotated["leaderboard_top10_json"] = annotated["leaderboard_top10_json"].where(annotated["leaderboard_top10_json"].notna(), top10_json)

    for idx, row in annotated.iterrows():
        primary_metric = str(row.get("primary_metric", "")).strip().lower()
        primary_value = row.get("primary_metric_value", np.nan)
        rank_cols = estimate_rank_columns_for_row(
            primary_metric=primary_metric,
            primary_value=primary_value if pd.notna(primary_value) else np.nan,
            leaderboard_summary=summary,
        )
        for key, value in rank_cols.items():
            annotated.loc[idx, key] = value
    return annotated


def write_leaderboard_reference_artifacts(root: Path, output_dir: Path, datasets: list[DatasetSpec]) -> pd.DataFrame:
    captured_at = time.strftime("%Y-%m-%d %H:%M:%S")
    reference_df = leaderboard_reference_table(datasets, captured_at=captured_at)
    run_reference_path = output_dir / "leaderboard_top10_reference.csv"
    reference_df.to_csv(run_reference_path, index=False)

    cache_dir = root / "data" / "benchmark_leaderboards"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latest_cache_path = cache_dir / "leaderboard_top10_reference_latest.csv"
    timestamped_cache_path = cache_dir / f"leaderboard_top10_reference_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if not reference_df.empty:
        reference_df.to_csv(latest_cache_path, index=False)
        reference_df.to_csv(timestamped_cache_path, index=False)

    json_path = output_dir / "leaderboard_top10_reference.json"
    json_path.write_text(reference_df.to_json(orient="records", indent=2), encoding="utf-8")
    return reference_df


def leaderboard_comparison_by_dataset(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "dataset" not in summary_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for dataset_name, group in summary_df.groupby("dataset", sort=True):
        if group.empty:
            continue
        group = group.copy()
        group = group[group["primary_metric_value"].notna()]
        if group.empty:
            continue
        group["primary_metric_value"] = pd.to_numeric(group["primary_metric_value"], errors="coerce")
        group = group[group["primary_metric_value"].notna()].copy()
        if group.empty:
            continue

        group["__primary_metric_norm"] = group["primary_metric"].apply(
            lambda value: normalize_benchmark_metric(value, fallback="rmse")
        )

        target_metric = ""
        if "leaderboard_metric_name" in group.columns:
            lb_metric_norm = group["leaderboard_metric_name"].apply(
                lambda value: normalize_benchmark_metric(value, fallback="")
            )
            lb_metric_norm = lb_metric_norm[lb_metric_norm.astype(str).str.strip() != ""]
            if not lb_metric_norm.empty:
                target_metric = str(lb_metric_norm.iloc[0]).strip().lower()

        if not target_metric:
            target_metric = str(group["__primary_metric_norm"].dropna().iloc[0]).strip().lower()

        metric_matched = group[group["__primary_metric_norm"] == target_metric].copy()
        working = metric_matched if not metric_matched.empty else group.copy()
        if working.empty:
            continue

        rank_values = pd.to_numeric(
            working.get("leaderboard_estimated_rank_vs_top10", pd.Series(index=working.index, dtype=float)),
            errors="coerce",
        )
        gap_values = pd.to_numeric(
            working.get("leaderboard_gap_to_top10_cutoff", pd.Series(index=working.index, dtype=float)),
            errors="coerce",
        )
        comparable_mask = rank_values.notna() | gap_values.notna()
        if bool(comparable_mask.any()):
            working = working.loc[comparable_mask].copy()

        lower_is_better = metric_lower_is_better(target_metric)
        ascending = True if lower_is_better is None else bool(lower_is_better)
        best = working.sort_values("primary_metric_value", ascending=ascending).iloc[0]

        leaderboard_metric_name = str(best.get("leaderboard_metric_name", "")).strip()
        if not leaderboard_metric_name and "leaderboard_metric_name" in working.columns:
            metric_name_candidates = working["leaderboard_metric_name"].astype(str).str.strip()
            metric_name_candidates = metric_name_candidates[metric_name_candidates != ""]
            if not metric_name_candidates.empty:
                leaderboard_metric_name = str(metric_name_candidates.iloc[0]).strip()

        leaderboard_metric_reference = pd.to_numeric(best.get("leaderboard_metric_reference", np.nan), errors="coerce")
        if pd.isna(leaderboard_metric_reference) and "leaderboard_metric_reference" in working.columns:
            reference_candidates = pd.to_numeric(working["leaderboard_metric_reference"], errors="coerce").dropna()
            if not reference_candidates.empty:
                leaderboard_metric_reference = float(reference_candidates.iloc[0])

        rows.append(
            {
                "dataset": str(dataset_name),
                "best_model": str(best.get("model", "")),
                "best_workflow": str(best.get("workflow", "")),
                "primary_metric": target_metric or str(best.get("primary_metric", "")).strip().lower(),
                "primary_metric_value": best.get("primary_metric_value", np.nan),
                "leaderboard_metric_name": leaderboard_metric_name,
                "leaderboard_metric_reference": leaderboard_metric_reference,
                "leaderboard_delta_primary": best.get("leaderboard_delta_primary", np.nan),
                "leaderboard_estimated_rank_vs_top10": best.get("leaderboard_estimated_rank_vs_top10", np.nan),
                "leaderboard_estimated_in_top10": best.get("leaderboard_estimated_in_top10", np.nan),
                "leaderboard_gap_to_top10_cutoff": best.get("leaderboard_gap_to_top10_cutoff", np.nan),
                "leaderboard_url": best.get("leaderboard_url", ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["dataset"]).reset_index(drop=True)


def compute_primary_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric = str(metric_name).strip().lower()
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if metric in {"mse", "mean_squared_error"}:
        return float(mean_squared_error(y_true, y_pred))
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
    if metric in {"mse", "mean_squared_error"}:
        return make_scorer(mean_squared_error, greater_is_better=False)
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
    keep_columns = [spec.smiles_column, spec.target_column]
    include_predefined_split = bool(spec.predefined_split_column and spec.predefined_split_column in spec.frame.columns)
    if include_predefined_split:
        keep_columns.append(str(spec.predefined_split_column))
    df = spec.frame[keep_columns].copy()
    rename_map = {spec.smiles_column: "smiles", spec.target_column: "target"}
    if include_predefined_split:
        rename_map[str(spec.predefined_split_column)] = "__predefined_split"
    df = df.rename(columns=rename_map)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if "__predefined_split" in df.columns:
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["smiles", "target", "__predefined_split"])
    else:
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
    if "__predefined_split" in df.columns and spec.predefined_split_column:
        df[str(spec.predefined_split_column)] = df["__predefined_split"].astype(str)
        df = df.drop(columns=["__predefined_split"])
    transform = "raw"
    if log10_target:
        if (df["target"] <= 0).any():
            print(f"[info] {spec.name}: non-positive target values; using raw target.")
        else:
            df["target"] = np.log10(df["target"].astype(float))
            transform = "log10"
    return df, {"target_transform": transform, "smiles_column": spec.smiles_column, "target_column": spec.target_column}


def resolve_dataset_log10_target(spec: DatasetSpec, args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "target_transform", "auto")).strip().lower()
    if mode == "log10":
        return True
    if mode == "raw":
        return False
    if str(spec.benchmark_suite or "").strip().lower() in {"tdc", "moleculenet", "polaris"}:
        return False
    return bool(getattr(args, "log10_target", True))


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
    predefined_split: pd.Series | None = None,
) -> dict[str, Any]:
    split_strategy = effective_split_strategy_for_dataset(
        str(split_strategy_override or args.split_strategy),
        allow_predefined=True,
    )
    if split_strategy not in {"random", "target_quartiles", "scaffold", "predefined"}:
        raise ValueError("split_strategy must be one of: random, target_quartiles, scaffold, predefined")
    if split_strategy == "predefined":
        if predefined_split is None:
            raise ValueError("predefined split strategy requires a predefined split series.")
        split_labels = predefined_split.astype(str).str.strip().str.lower().reset_index(drop=True)
        if len(split_labels) != len(X):
            raise ValueError("predefined split labels length does not match feature matrix rows")
        train_mask = split_labels.isin({"train", "training"})
        test_mask = split_labels.isin({"test", "holdout"})
        if not bool(train_mask.any()) or not bool(test_mask.any()):
            raise ValueError("predefined split labels must include both train and test rows.")
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]
        smiles_train = smiles.loc[train_mask]
        smiles_test = smiles.loc[test_mask]
    elif split_strategy in {"random", "target_quartiles"}:
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
        "split_signature": build_split_signature(smiles_train, smiles_test),
    }


def run_timed_elasticnet_selector_fit(
    *,
    X_scaled: np.ndarray,
    y_train: np.ndarray,
    l1_ratio: list[float],
    alphas: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    max_iter: int,
    random_seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "X_scaled": np.asarray(X_scaled, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "l1_ratio": [float(value) for value in list(l1_ratio)],
        "alphas": np.asarray(alphas, dtype=float),
        "cv_splits": [
            (np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int))
            for train_idx, valid_idx in list(cv_splits)
        ],
        "max_iter": int(max_iter),
        "random_seed": int(random_seed),
    }
    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0:
        timeout_seconds = 1.0

    worker_code = """
import json
import pickle
import sys
import numpy as np
from sklearn.linear_model import ElasticNetCV

def main():
    payload_path = sys.argv[1]
    result_path = sys.argv[2]
    result = {"ok": False, "error": "unknown"}
    try:
        with open(payload_path, "rb") as handle:
            payload = pickle.load(handle)
        selector = ElasticNetCV(
            l1_ratio=[float(value) for value in payload["l1_ratio"]],
            alphas=np.asarray(payload["alphas"], dtype=float),
            cv=payload["cv_splits"],
            max_iter=int(payload["max_iter"]),
            n_jobs=1,
            random_state=int(payload["random_seed"]),
        )
        selector.fit(np.asarray(payload["X_scaled"], dtype=float), np.asarray(payload["y_train"], dtype=float))
        coef = np.asarray(selector.coef_, dtype=float)
        result = {
            "ok": True,
            "coef": coef.tolist(),
            "alpha": float(selector.alpha_),
            "l1_ratio": float(selector.l1_ratio_),
            "n_iter": int(np.max(np.atleast_1d(selector.n_iter_))),
        }
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)

if __name__ == "__main__":
    main()
"""
    selector_tmp_root = Path.cwd() / ".autoqsar_tmp"
    selector_tmp_root.mkdir(parents=True, exist_ok=True)
    temp_path = selector_tmp_root / f"selector_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    temp_path.mkdir(parents=True, exist_ok=True)
    try:
        payload_path = temp_path / "selector_payload.pkl"
        result_path = temp_path / "selector_result.json"
        worker_path = temp_path / "selector_worker.py"
        with payload_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        worker_path.write_text(worker_code, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(worker_path), str(payload_path), str(result_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "timed_out": True,
                "error": f"ElasticNetCV selector exceeded timeout ({timeout_seconds:.1f} seconds)",
            }

        if completed.returncode != 0:
            stderr_tail = (completed.stderr or completed.stdout or "").strip()
            stderr_tail = stderr_tail.splitlines()[-1] if stderr_tail else "subprocess returned non-zero exit code"
            return {
                "ok": False,
                "timed_out": False,
                "error": f"ElasticNetCV selector subprocess failed (exitcode={completed.returncode}): {stderr_tail}",
            }

        if not result_path.exists():
            return {
                "ok": False,
                "timed_out": False,
                "error": "ElasticNetCV selector subprocess finished without writing a result payload",
            }
        try:
            message = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "ok": False,
                "timed_out": False,
                "error": f"ElasticNetCV selector result payload could not be parsed: {type(exc).__name__}: {exc}",
            }
    finally:
        try:
            shutil.rmtree(temp_path, ignore_errors=True)
        except Exception:
            pass

    if not bool(message.get("ok")):
        return {
            "ok": False,
            "timed_out": False,
            "error": str(message.get("error", "ElasticNetCV selector worker failed")),
        }
    return {
        "ok": True,
        "timed_out": False,
        "coef": np.asarray(message["coef"], dtype=float),
        "alpha": float(message["alpha"]),
        "l1_ratio": float(message["l1_ratio"]),
        "n_iter": int(message["n_iter"]),
    }


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    smiles_train: pd.Series,
    args: argparse.Namespace,
    split_strategy_for_cv: str | None = None,
):
    if args.selector_method == "none":
        return X_train.copy(), X_test.copy(), {
            "selector_method": "none",
            "selector_timed_out": False,
            "selected_feature_count": int(X_train.shape[1]),
            "original_feature_count": int(X_train.shape[1]),
            "selected_features": list(X_train.columns),
        }
    max_features = args.max_selected_features if args.max_selected_features > 0 else max(1, math.ceil(0.10 * len(y_train)))

    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)
    selector_cv, selector_cv_folds, selector_cv_strategy = make_qsar_cv_splitter(
        X_train,
        y_train,
        smiles_train,
        split_strategy=str(split_strategy_for_cv or args.split_strategy),
        cv_folds=args.selector_cv_folds,
        random_seed=args.random_seed,
    )
    cv_splits = list(selector_cv) if isinstance(selector_cv, list) else list(selector_cv.split(X_train, y_train))
    selector_fit: dict[str, Any]
    auto_rf_large_dataset = False
    predicted_selector_seconds = estimate_elasticnet_selector_seconds_from_dataset_size(
        int(len(y_train)),
        log10_slope=float(getattr(args, "selector_auto_rf_log10_slope", 1.225)),
        log10_intercept=float(getattr(args, "selector_auto_rf_log10_intercept", -0.658)),
    )
    auto_rf_threshold_seconds = float(getattr(args, "selector_auto_rf_threshold_seconds", 7200.0))
    auto_rf_threshold_size = elasticnet_selector_timeout_dataset_size_threshold(
        auto_rf_threshold_seconds,
        log10_slope=float(getattr(args, "selector_auto_rf_log10_slope", 1.225)),
        log10_intercept=float(getattr(args, "selector_auto_rf_log10_intercept", -0.658)),
    )
    if bool(getattr(args, "selector_auto_rf_by_dataset_size", True)):
        if (
            np.isfinite(predicted_selector_seconds)
            and np.isfinite(auto_rf_threshold_seconds)
            and predicted_selector_seconds > auto_rf_threshold_seconds
        ):
            auto_rf_large_dataset = True
            selector_fit = {
                "ok": False,
                "timed_out": False,
                "error": (
                    "ElasticNetCV selector preemptively skipped due to dataset-size runtime estimate "
                    f"({predicted_selector_seconds:,.0f}s > {auto_rf_threshold_seconds:,.0f}s threshold)."
                ),
            }
            threshold_size_text = (
                f"{int(round(auto_rf_threshold_size)):,}"
                if np.isfinite(auto_rf_threshold_size)
                else "unknown"
            )
            print(
                "[selector] ElasticNetCV predicted runtime exceeds threshold; "
                f"using RandomForest feature importance by default "
                f"(n={int(len(y_train)):,}, estimate={predicted_selector_seconds:,.0f}s, "
                f"threshold={auto_rf_threshold_seconds:,.0f}s, threshold_n~{threshold_size_text}).",
                flush=True,
            )
        else:
            selector_fit = run_timed_elasticnet_selector_fit(
                X_scaled=X_scaled,
                y_train=y_train.to_numpy(dtype=float),
                l1_ratio=parse_l1_grid(args.selector_l1_ratio_grid),
                alphas=regularization_grid(args.selector_alpha_min_log10, args.selector_alpha_max_log10, args.selector_alpha_grid_size),
                cv_splits=cv_splits,
                max_iter=args.selector_max_iter,
                random_seed=args.random_seed,
                timeout_seconds=float(args.selector_elasticnet_timeout_seconds),
            )
    else:
        selector_fit = run_timed_elasticnet_selector_fit(
            X_scaled=X_scaled,
            y_train=y_train.to_numpy(dtype=float),
            l1_ratio=parse_l1_grid(args.selector_l1_ratio_grid),
            alphas=regularization_grid(args.selector_alpha_min_log10, args.selector_alpha_max_log10, args.selector_alpha_grid_size),
            cv_splits=cv_splits,
            max_iter=args.selector_max_iter,
            random_seed=args.random_seed,
            timeout_seconds=float(args.selector_elasticnet_timeout_seconds),
        )

    if bool(selector_fit.get("ok")):
        coef = np.asarray(selector_fit["coef"], dtype=float)
        abs_coef = np.abs(coef)
        mask = abs_coef > args.selector_coefficient_threshold
        if not mask.any():
            mask[int(np.argmax(abs_coef))] = True
        if mask.sum() > max_features:
            selected = np.flatnonzero(mask)
            keep = selected[np.argsort(abs_coef[selected])[::-1]][:max_features]
            mask = np.zeros_like(mask, dtype=bool)
            mask[keep] = True
        columns = X_train.columns[mask].tolist()
        coef_df = pd.DataFrame({"feature": X_train.columns, "coefficient": coef, "abs_coefficient": abs_coef})
        return X_train[columns].copy(), X_test[columns].copy(), {
            "selector_method": "elasticnet_cv",
            "selector_timed_out": False,
            "selector_auto_rf_large_dataset_triggered": bool(auto_rf_large_dataset),
            "selected_feature_count": int(len(columns)),
            "original_feature_count": int(X_train.shape[1]),
            "selector_alpha": float(selector_fit["alpha"]),
            "selector_l1_ratio": float(selector_fit["l1_ratio"]),
            "selector_n_iter": int(selector_fit["n_iter"]),
            "selector_cv_folds": int(selector_cv_folds),
            "selector_cv_split_strategy": selector_cv_strategy,
            "selector_predicted_elasticnet_seconds": (
                float(predicted_selector_seconds) if np.isfinite(predicted_selector_seconds) else np.nan
            ),
            "selector_auto_rf_threshold_seconds": float(auto_rf_threshold_seconds),
            "selector_auto_rf_threshold_dataset_size": (
                float(auto_rf_threshold_size) if np.isfinite(auto_rf_threshold_size) else np.nan
            ),
            "selected_features": columns,
            "selector_coefficients": coef_df.sort_values("abs_coefficient", ascending=False),
            "max_selected_features": int(max_features),
        }

    fallback_reason = str(selector_fit.get("error", "ElasticNetCV selector failed"))
    if bool(selector_fit.get("timed_out")):
        print(
            f"[selector] ElasticNetCV exceeded {float(args.selector_elasticnet_timeout_seconds):,.0f}s; "
            "falling back to RandomForest feature importance.",
            flush=True,
        )
    else:
        print(
            f"[selector] ElasticNetCV unavailable ({fallback_reason}); "
            "falling back to RandomForest feature importance.",
            flush=True,
        )

    rf_selector = RandomForestRegressor(
        n_estimators=int(args.selector_rf_fallback_n_estimators),
        random_state=args.random_seed,
        n_jobs=1,
    )
    rf_selector.fit(X_imputed, y_train.to_numpy(dtype=float))
    importances = np.asarray(getattr(rf_selector, "feature_importances_", np.zeros(X_train.shape[1])), dtype=float)
    if importances.ndim != 1 or int(importances.shape[0]) != int(X_train.shape[1]):
        importances = np.zeros(X_train.shape[1], dtype=float)
    importances = np.where(np.isfinite(importances), importances, 0.0)
    mask = importances > 0.0
    if not mask.any():
        mask[int(np.argmax(importances))] = True
    if mask.sum() > max_features:
        selected = np.flatnonzero(mask)
        keep = selected[np.argsort(importances[selected])[::-1]][:max_features]
        mask = np.zeros_like(mask, dtype=bool)
        mask[keep] = True
    columns = X_train.columns[mask].tolist()
    imp_df = pd.DataFrame({"feature": X_train.columns, "importance": importances})
    return X_train[columns].copy(), X_test[columns].copy(), {
        "selector_method": "random_forest_importance_fallback",
        "selector_timed_out": bool(selector_fit.get("timed_out", False)),
        "selector_auto_rf_large_dataset_triggered": bool(auto_rf_large_dataset),
        "selected_feature_count": int(len(columns)),
        "original_feature_count": int(X_train.shape[1]),
        "selector_alpha": np.nan,
        "selector_l1_ratio": np.nan,
        "selector_n_iter": np.nan,
        "selector_cv_folds": int(selector_cv_folds),
        "selector_cv_split_strategy": selector_cv_strategy,
        "selector_predicted_elasticnet_seconds": (
            float(predicted_selector_seconds) if np.isfinite(predicted_selector_seconds) else np.nan
        ),
        "selector_auto_rf_threshold_seconds": float(auto_rf_threshold_seconds),
        "selector_auto_rf_threshold_dataset_size": (
            float(auto_rf_threshold_size) if np.isfinite(auto_rf_threshold_size) else np.nan
        ),
        "selected_features": columns,
        "selector_coefficients": imp_df.sort_values("importance", ascending=False),
        "max_selected_features": int(max_features),
        "selector_fallback_reason": fallback_reason,
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


def add_leaderboard_reference_columns(
    row: dict[str, Any],
    *,
    primary_metric: str,
    primary_metric_value: float,
) -> dict[str, Any]:
    leaderboard_summary = (CURRENT_DATASET_SPEC.leaderboard_summary if CURRENT_DATASET_SPEC is not None else None) or {}
    row["leaderboard_model"] = leaderboard_summary.get("model", "")
    row["leaderboard_metric_name"] = leaderboard_summary.get("metric_name", "")
    row["leaderboard_metric_value"] = leaderboard_summary.get("metric_value", "")
    row["leaderboard_rank"] = leaderboard_summary.get("rank", "")
    row["leaderboard_url"] = leaderboard_summary.get(
        "url",
        CURRENT_DATASET_SPEC.leaderboard_url if CURRENT_DATASET_SPEC is not None else "",
    )
    top10_entries = leaderboard_summary.get("top10", [])
    row["leaderboard_top10_count"] = int(len(top10_entries)) if isinstance(top10_entries, list) else 0
    row["leaderboard_top10_json"] = json.dumps(top10_entries, default=str) if isinstance(top10_entries, list) else "[]"
    leaderboard_metric_norm = normalize_benchmark_metric(leaderboard_summary.get("metric_name"), fallback="")
    leaderboard_value = parse_first_float(leaderboard_summary.get("metric_value"))
    if leaderboard_metric_norm and leaderboard_value is not None and leaderboard_metric_norm == str(primary_metric).strip().lower():
        row["leaderboard_metric_normalized"] = leaderboard_metric_norm
        row["leaderboard_metric_reference"] = float(leaderboard_value)
        row["leaderboard_delta_primary"] = float(primary_metric_value - leaderboard_value)
        row["leaderboard_lower_is_better"] = metric_lower_is_better(leaderboard_metric_norm)
    else:
        row["leaderboard_metric_normalized"] = leaderboard_metric_norm
        row["leaderboard_metric_reference"] = np.nan
        row["leaderboard_delta_primary"] = np.nan
        row["leaderboard_lower_is_better"] = metric_lower_is_better(leaderboard_metric_norm)
    if isinstance(top10_entries, list) and top10_entries:
        comparable_values = [
            parse_first_float(entry.get("metric_value"))
            for entry in top10_entries
            if isinstance(entry, dict)
            and normalize_benchmark_metric(entry.get("metric_name"), fallback=leaderboard_metric_norm) == leaderboard_metric_norm
        ]
        comparable_values = [value for value in comparable_values if value is not None]
        if comparable_values and leaderboard_metric_norm == str(primary_metric).strip().lower():
            top10_best = min(comparable_values) if metric_lower_is_better(leaderboard_metric_norm) else max(comparable_values)
            row["leaderboard_top10_best_reference"] = float(top10_best)
            row["leaderboard_delta_top10_best"] = float(primary_metric_value - float(top10_best))
        else:
            row["leaderboard_top10_best_reference"] = np.nan
            row["leaderboard_delta_top10_best"] = np.nan
    else:
        row["leaderboard_top10_best_reference"] = np.nan
        row["leaderboard_delta_top10_best"] = np.nan
    rank_columns = estimate_rank_columns_for_row(
        primary_metric=str(primary_metric),
        primary_value=float(primary_metric_value),
        leaderboard_summary=leaderboard_summary,
    )
    row.update(rank_columns)
    return row


def conventional_models(
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    smiles_train: pd.Series,
    split_strategy_for_cv: str | None = None,
) -> dict[str, Any]:
    finite_numeric = FunctionTransformer(
        np.nan_to_num,
        kw_args={"nan": np.nan, "posinf": np.nan, "neginf": np.nan},
        validate=False,
    )
    elasticnet_cv, elasticnet_cv_folds, elasticnet_cv_strategy = make_reusable_inner_cv_splitter(
        split_strategy=str(split_strategy_for_cv or args.split_strategy),
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
                        n_jobs=1,
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
        "Random forest": RandomForestRegressor(n_estimators=400, random_state=args.random_seed, n_jobs=1),
        "Extra trees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=args.random_seed,
            n_jobs=1,
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("finite", finite_numeric),
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=500,
                        max_depth=8,
                        random_state=args.random_seed,
                    ),
                ),
            ]
        ),
        "Voting Regressor (KNN, SVM)": VotingRegressor(
            estimators=[
                (
                    "knn",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("model", KNeighborsRegressor(n_neighbors=15, weights="distance")),
                        ]
                    ),
                ),
                (
                    "svr",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("model", SVR(C=10.0, epsilon=0.1, gamma="scale")),
                        ]
                    ),
                ),
                ]
            ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=args.random_seed,
        ),
        "Tabular MLP": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(512, 256),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        random_state=args.random_seed,
                    ),
                ),
            ]
        ),
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
            n_jobs=1,
        )
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_seed,
            n_jobs=1,
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
        if bool(getattr(args, "maplight_leaderboard_parity_mode", True)):
            # Strict parity mode uses MAE + scaling + 5-seed averaging in the
            # dedicated MapLight evaluation block.
            models[maplight_catboost_model_label(args)] = None
        else:
            models[MAPLIGHT_CATBOOST_LABEL_LEGACY] = CatBoostRegressor(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=args.random_seed,
                verbose=False,
            )
    if (
        bool(getattr(args, "run_tabpfn", False))
        and TabPFNRegressor is not None
        and int(X_train.shape[0]) <= int(getattr(args, "tabpfn_max_train_rows", 1000))
    ):
        models["TabPFNRegressor"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", TabPFNRegressor()),
            ]
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
    split_strategy_for_cv: str | None = None,
):
    effective_split_strategy = effective_split_strategy_for_dataset(
        str(split_strategy_for_cv or args.split_strategy),
        allow_predefined=False,
    )
    if primary_metric is None:
        if CURRENT_DATASET_SPEC is not None and CURRENT_DATASET_SPEC.recommended_metric:
            primary_metric = normalize_benchmark_metric(CURRENT_DATASET_SPEC.recommended_metric, fallback="rmse")
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
        if primary_metric in {"rmse", "mse", "mean_squared_error", "mae"}:
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
    row = add_leaderboard_reference_columns(
        row,
        primary_metric=primary_metric,
        primary_metric_value=primary_test_value,
    )
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


def run_simple_ga(
    name: str,
    build_estimator: Callable,
    space: dict[str, dict[str, Any]],
    decode: Callable,
    X_train,
    X_test,
    y_train,
    y_test,
    smiles_train,
    args,
    split_strategy_for_cv: str | None = None,
):
    rng = random.Random(args.random_seed)
    keys = list(space)
    cv, cv_folds, cv_split_strategy = make_qsar_cv_splitter(
        X_train,
        y_train,
        smiles_train,
        split_strategy=str(split_strategy_for_cv or args.split_strategy),
        cv_folds=args.ga_cv_folds,
        random_seed=args.random_seed,
    )
    cv_splits = list(cv) if isinstance(cv, list) else list(cv.split(X_train, y_train))
    if not cv_splits:
        raise ValueError("GA tuning could not build any cross-validation splits.")

    def random_individual():
        return {key: sample_value(space[key], rng) for key in keys}

    def score(individual):
        estimator = build_estimator(individual)
        rmses = []
        for train_idx, valid_idx in cv_splits:
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


def prediction_payload(
    *,
    workflow: str,
    train_smiles: pd.Series,
    test_smiles: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
) -> dict[str, Any]:
    return {
        "workflow": str(workflow),
        "train_smiles": pd.Series(train_smiles, dtype=str).reset_index(drop=True),
        "test_smiles": pd.Series(test_smiles, dtype=str).reset_index(drop=True),
        "train_observed": np.asarray(y_train, dtype=float).reshape(-1),
        "test_observed": np.asarray(y_test, dtype=float).reshape(-1),
        "train": np.asarray(pred_train, dtype=float).reshape(-1),
        "test": np.asarray(pred_test, dtype=float).reshape(-1),
    }


def _version_key(version_text: str) -> tuple[int, int, int]:
    parts = re.findall(r"\d+", str(version_text))
    values = [int(part) for part in parts[:3]]
    while len(values) < 3:
        values.append(0)
    return tuple(values)


def _patch_torchdata_dill_available() -> bool:
    try:
        import torch.utils.data.datapipes.utils.common as torch_datapipe_common

        if not hasattr(torch_datapipe_common, "DILL_AVAILABLE"):
            torch_datapipe_common.DILL_AVAILABLE = False
            return True
    except Exception:
        return False
    return False


def _is_maplight_store_access_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = [
        "molfeat-store-prod",
        "storage.objects.list",
        "anonymous caller",
        "default credentials",
        "httperror",
        "cannot connect to host storage.googleapis.com",
    ]
    return any(marker in message for marker in markers)


@contextlib.contextmanager
def _suppress_maplight_store_probe_noise():
    noisy_loggers = ("gcsfs", "fsspec", "google.auth", "google.cloud")
    prior_states: list[tuple[logging.Logger, int, bool]] = []
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        prior_states.append((logger, logger.level, logger.disabled))
        logger.disabled = True
    probe_stderr = io.StringIO()
    probe_stdout = io.StringIO()
    try:
        with contextlib.redirect_stderr(probe_stderr), contextlib.redirect_stdout(probe_stdout):
            yield
    finally:
        for logger, level, disabled in prior_states:
            logger.setLevel(level)
            logger.disabled = disabled


def default_maplight_pretrained_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model_cache" / "maplight_gnn_pretrained"


def _build_maplight_gnn_embedder(kind: str = "gin_supervised_masking") -> tuple[Callable[[list[str]], list[Any]], str]:
    cache_key = str(kind).strip().lower()
    if cache_key in _MAPLIGHT_EMBEDDER_CACHE:
        return _MAPLIGHT_EMBEDDER_CACHE[cache_key]

    _patch_torchdata_dill_available()
    from molfeat.trans.pretrained import PretrainedDGLTransformer

    try:
        with _suppress_maplight_store_probe_noise():
            transformer = PretrainedDGLTransformer(kind=kind, dtype=float)

        def _embed_with_molfeat(smiles_values: list[str]):
            return transformer(smiles_values)

        result = (_embed_with_molfeat, "molfeat-store")
        _MAPLIGHT_EMBEDDER_CACHE[cache_key] = result
        return result
    except Exception as primary_exc:
        if not _is_maplight_store_access_error(primary_exc):
            raise
        print(
            "MapLight + GNN note: molfeat store access unavailable; switching to dgllife-direct fallback.",
            flush=True,
        )
        try:
            import dgl
            import dgllife
            import torch
            from torch.utils.data import DataLoader

            pretrained_cache_dir = default_maplight_pretrained_cache_dir()
            pretrained_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("DGL_DOWNLOAD_DIR", str(pretrained_cache_dir.resolve()))
            cached_weight_file = pretrained_cache_dir / f"{kind}_pre_trained.pth"
            if cached_weight_file.exists():
                print(
                    f"MapLight + GNN note: reusing cached pretrained weights at {cached_weight_file}.",
                    flush=True,
                )
            with contextlib.chdir(pretrained_cache_dir):
                fallback_model = dgllife.model.load_pretrained(kind)
            fallback_model.eval()
            pooling = PretrainedDGLTransformer.get_pooling("mean")

            def _embed_with_dgllife(smiles_values: list[str]):
                smiles_list = [str(value) for value in list(smiles_values)]
                dataset, successes = PretrainedDGLTransformer.graph_featurizer(smiles_list, kind=kind)
                data_loader = DataLoader(
                    dataset,
                    batch_size=32,
                    collate_fn=dgl.batch,
                    shuffle=False,
                    drop_last=False,
                )
                mol_emb: list[np.ndarray] = []
                for bg in data_loader:
                    nfeats = [
                        bg.ndata.pop("atomic_number").to(torch.device("cpu")),
                        bg.ndata.pop("chirality_type").to(torch.device("cpu")),
                    ]
                    efeats = [
                        bg.edata.pop("bond_type").to(torch.device("cpu")),
                        bg.edata.pop("bond_direction_type").to(torch.device("cpu")),
                    ]
                    with torch.no_grad():
                        node_repr = fallback_model(bg, nfeats, efeats)
                    mol_emb.append(pooling(bg, node_repr).detach().cpu().numpy())

                if mol_emb:
                    emb = np.concatenate(mol_emb, axis=0)
                else:
                    emb = np.empty((0, 0), dtype=float)

                out: list[Any] = []
                emb_idx = 0
                for success in successes:
                    if success:
                        out.append(emb[emb_idx])
                        emb_idx += 1
                    else:
                        out.append(None)
                return out

            result = (_embed_with_dgllife, "dgllife-direct")
            _MAPLIGHT_EMBEDDER_CACHE[cache_key] = result
            return result
        except Exception as fallback_exc:
            raise RuntimeError(
                "MapLight + GNN could not load pretrained GIN weights. "
                "The molfeat store backend failed and dgllife direct fallback download/load also failed.\n"
                f"molfeat-store error: {primary_exc}\n"
                f"dgllife direct fallback error: {fallback_exc}"
            ) from fallback_exc


def _embeddings_to_matrix(
    embeddings: list[Any],
    *,
    reference_width: int | None = None,
) -> np.ndarray:
    vectors: list[np.ndarray | None] = []
    inferred_width = int(reference_width or 0)
    for item in embeddings:
        if item is None:
            vectors.append(None)
            continue
        arr = np.asarray(item, dtype=float).reshape(-1)
        if arr.size == 0:
            vectors.append(None)
            continue
        inferred_width = max(inferred_width, int(arr.size))
        vectors.append(arr)
    if inferred_width <= 0:
        raise ValueError("MapLight + GNN embeddings are empty for all molecules.")
    matrix = np.full((len(vectors), inferred_width), np.nan, dtype=float)
    for idx, arr in enumerate(vectors):
        if arr is None:
            continue
        clipped = arr[:inferred_width]
        matrix[idx, : len(clipped)] = clipped
    return matrix


def _fit_chemml_mlp(
    *,
    engine_name: str,
    X_fit_raw: np.ndarray,
    y_fit_raw: np.ndarray,
    X_predict_raw: np.ndarray,
    hidden_layers: int,
    hidden_width: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    random_seed: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    from chemml.models.mlp import MLP

    x_scaler_local = StandardScaler()
    y_scaler_local = StandardScaler()
    X_fit_scaled = x_scaler_local.fit_transform(np.asarray(X_fit_raw, dtype=np.float32)).astype(np.float32)
    X_predict_scaled = x_scaler_local.transform(np.asarray(X_predict_raw, dtype=np.float32)).astype(np.float32)
    y_fit_scaled = y_scaler_local.fit_transform(np.asarray(y_fit_raw, dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    mlp = MLP(
        engine=str(engine_name),
        nfeatures=int(X_fit_scaled.shape[1]),
        nneurons=[int(hidden_width)] * int(hidden_layers),
        activations=["ReLU"] * int(hidden_layers),
        learning_rate=float(learning_rate),
        alpha=0.0001,
        nepochs=int(epochs),
        batch_size=int(batch_size),
        loss="mean_squared_error",
        is_regression=True,
        nclasses=None,
        layer_config_file=None,
        opt_config="Adam",
        random_seed=int(random_seed),
    )
    mlp.fit(X_fit_scaled, y_fit_scaled)
    fit_pred = y_scaler_local.inverse_transform(np.asarray(mlp.predict(X_fit_scaled)).reshape(-1, 1)).reshape(-1)
    predict_pred = y_scaler_local.inverse_transform(np.asarray(mlp.predict(X_predict_scaled)).reshape(-1, 1)).reshape(-1)
    return mlp, np.asarray(fit_pred, dtype=float), np.asarray(predict_pred, dtype=float)


def _compute_cv_metrics_from_oof(y_true: np.ndarray, oof_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    oof_arr = np.asarray(oof_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true_arr) & np.isfinite(oof_arr)
    if int(mask.sum()) < 2:
        return {"cv_r2": np.nan, "cv_rmse": np.nan, "cv_mae": np.nan}
    y_masked = y_true_arr[mask]
    pred_masked = oof_arr[mask]
    return {
        "cv_r2": float(r2_score(y_masked, pred_masked)),
        "cv_rmse": float(math.sqrt(mean_squared_error(y_masked, pred_masked))),
        "cv_mae": float(mean_absolute_error(y_masked, pred_masked)),
    }


def train_chemml_model(
    *,
    label: str,
    engine_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    smiles_train: pd.Series,
    args: argparse.Namespace,
    split_strategy_for_cv: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _, pred_train, pred_test = _fit_chemml_mlp(
        engine_name=engine_name,
        X_fit_raw=X_train.to_numpy(dtype=np.float32),
        y_fit_raw=y_train.to_numpy(dtype=np.float32),
        X_predict_raw=X_test.to_numpy(dtype=np.float32),
        hidden_layers=int(args.chemml_hidden_layers),
        hidden_width=int(args.chemml_hidden_width),
        learning_rate=float(args.chemml_learning_rate),
        epochs=int(args.chemml_training_epochs),
        batch_size=int(args.chemml_batch_size),
        random_seed=int(args.random_seed),
    )

    cv_strategy = str(split_strategy_for_cv)
    primary_metric = current_dataset_primary_metric("rmse")
    cv_r2 = np.nan
    cv_rmse = np.nan
    cv_mae = np.nan
    cv_primary = np.nan
    if bool(args.chemml_use_cross_validation):
        cv_splitter, cv_folds, cv_split_strategy = make_qsar_cv_splitter(
            X_train,
            y_train,
            smiles_train,
            split_strategy=cv_strategy,
            cv_folds=int(args.chemml_cv_folds),
            random_seed=int(args.random_seed),
        )
        y_train_arr = y_train.to_numpy(dtype=float)
        oof_predictions = np.full(len(y_train_arr), np.nan, dtype=float)
        splits = list(cv_splitter) if isinstance(cv_splitter, list) else list(cv_splitter.split(X_train, y_train_arr))
        for fold_idx, (fit_idx, val_idx) in enumerate(splits):
            _fold_model, _fit_pred, fold_val_pred = _fit_chemml_mlp(
                engine_name=engine_name,
                X_fit_raw=X_train.iloc[fit_idx].to_numpy(dtype=np.float32),
                y_fit_raw=y_train.iloc[fit_idx].to_numpy(dtype=np.float32),
                X_predict_raw=X_train.iloc[val_idx].to_numpy(dtype=np.float32),
                hidden_layers=int(args.chemml_hidden_layers),
                hidden_width=int(args.chemml_hidden_width),
                learning_rate=float(args.chemml_learning_rate),
                epochs=int(args.chemml_training_epochs),
                batch_size=int(args.chemml_batch_size),
                random_seed=int(args.random_seed) + int(fold_idx),
            )
            oof_predictions[np.asarray(val_idx, dtype=int)] = np.asarray(fold_val_pred, dtype=float).reshape(-1)
        cv_metrics = _compute_cv_metrics_from_oof(y_train_arr, oof_predictions)
        cv_r2, cv_rmse, cv_mae = cv_metrics["cv_r2"], cv_metrics["cv_rmse"], cv_metrics["cv_mae"]
        oof_mask = np.isfinite(y_train_arr) & np.isfinite(oof_predictions)
        if int(oof_mask.sum()) >= 2:
            cv_primary = compute_primary_metric(
                primary_metric,
                y_train_arr[oof_mask],
                oof_predictions[oof_mask],
            )
    else:
        cv_folds = np.nan
        cv_split_strategy = ""

    row = {
        "model": str(label),
        "workflow": "ChemML deep learning",
        "cv_folds": int(cv_folds) if pd.notna(cv_folds) else np.nan,
        "cv_split_strategy": str(cv_split_strategy),
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "primary_metric": primary_metric,
        "cv_primary": cv_primary if pd.notna(cv_primary) else (cv_rmse if primary_metric == "rmse" else np.nan),
    }
    row.update(regression_metrics(y_train, pred_train, y_test, pred_test))
    row["primary_metric_value"] = float(
        compute_primary_metric(
            primary_metric,
            y_test.to_numpy(dtype=float),
            np.asarray(pred_test, dtype=float),
        )
    )
    return row, np.asarray(pred_train, dtype=float), np.asarray(pred_test, dtype=float)


def _resolve_chemprop_command() -> list[str]:
    exe_parent = Path(sys.executable).resolve().parent
    candidates: list[list[str]] = [[sys.executable, "-m", "chemprop"]]
    for candidate_dir in [exe_parent / "Scripts", exe_parent / "bin", exe_parent]:
        candidates.append([str(candidate_dir / "chemprop.exe")])
        candidates.append([str(candidate_dir / "chemprop")])
    candidates.append(["chemprop"])
    for candidate in candidates:
        try:
            probe = subprocess.run(candidate + ["--help"], capture_output=True, text=True)
        except FileNotFoundError:
            continue
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(
        "Could not locate a working Chemprop CLI command. "
        "Install chemprop>=2 and ensure the CLI is available in the active environment."
    )


def _run_chemprop_command(command_prefix: list[str], command_args: list[str], description: str) -> None:
    cmd = list(command_prefix) + list(command_args)
    print(f"[Chemprop] {description}: {' '.join(str(part) for part in cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stdout_tail = (result.stdout or "").strip()[-1500:]
        stderr_tail = (result.stderr or "").strip()[-1500:]
        raise RuntimeError(
            f"Chemprop command failed during {description} (exit={result.returncode}).\n"
            f"Command: {' '.join(str(part) for part in cmd)}\n"
            f"stdout tail:\n{stdout_tail}\n\n"
            f"stderr tail:\n{stderr_tail}"
        )


def _align_predictions_by_smiles_occurrence(
    expected_smiles: pd.Series,
    actual_smiles: pd.Series,
    pred_values: pd.Series,
) -> np.ndarray | None:
    expected_df = pd.DataFrame({"smiles": pd.Series(expected_smiles, dtype=str).str.strip()})
    actual_df = pd.DataFrame(
        {
            "smiles": pd.Series(actual_smiles, dtype=str).str.strip(),
            "predicted": pd.to_numeric(pred_values, errors="coerce"),
        }
    )
    expected_df["_occurrence"] = expected_df.groupby("smiles", sort=False).cumcount()
    expected_df["_expected_row"] = np.arange(len(expected_df), dtype=int)
    actual_df["_occurrence"] = actual_df.groupby("smiles", sort=False).cumcount()
    merged = expected_df.merge(
        actual_df,
        on=["smiles", "_occurrence"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if merged["predicted"].isna().any():
        return None
    aligned = merged.sort_values("_expected_row")["predicted"].to_numpy(dtype=float)
    if len(aligned) != len(expected_df):
        return None
    return np.asarray(aligned, dtype=float)


def _extract_chemprop_predictions(preds_csv_path: Path, expected_smiles: pd.Series) -> np.ndarray:
    preds_df = pd.read_csv(preds_csv_path)
    if preds_df.empty:
        raise ValueError(f"Chemprop prediction file is empty: {preds_csv_path}")
    smiles_col = None
    for candidate in ["SMILES", "smiles", "Smiles"]:
        if candidate in preds_df.columns:
            smiles_col = candidate
            break
    prediction_cols = [col for col in preds_df.columns if "pred" in str(col).lower()]
    if not prediction_cols:
        non_smiles_cols = [col for col in preds_df.columns if str(col) not in {"SMILES", "smiles", "Smiles"}]
        if len(non_smiles_cols) == 1:
            prediction_cols = non_smiles_cols
        else:
            prediction_cols = [col for col in non_smiles_cols if str(col) != "split"]
    if not prediction_cols:
        raise ValueError(f"Could not identify a prediction column in Chemprop output: {list(preds_df.columns)}")
    prediction_col = prediction_cols[0]
    pred_values = pd.to_numeric(preds_df[prediction_col], errors="coerce")

    expected_smiles = pd.Series(expected_smiles, dtype=str).reset_index(drop=True)
    if smiles_col is not None:
        actual_smiles = preds_df[smiles_col].astype(str).str.strip().reset_index(drop=True)
        expected_smiles = expected_smiles.astype(str).str.strip().reset_index(drop=True)

        valid_mask = pred_values.notna()
        if not bool(valid_mask.all()):
            preds_df = preds_df.loc[valid_mask].reset_index(drop=True)
            pred_values = pred_values.loc[valid_mask].reset_index(drop=True)
            actual_smiles = actual_smiles.loc[valid_mask].reset_index(drop=True)

        if len(actual_smiles) == len(expected_smiles) and actual_smiles.equals(expected_smiles):
            return pred_values.to_numpy(dtype=float)

        aligned = _align_predictions_by_smiles_occurrence(
            expected_smiles=expected_smiles,
            actual_smiles=actual_smiles,
            pred_values=pred_values,
        )
        if aligned is not None:
            return aligned

        # Fallback for occasional Chemprop CSV append behavior: preserve order and clip
        # when prefix rows still align exactly to the expected split.
        if len(actual_smiles) >= len(expected_smiles):
            actual_prefix = actual_smiles.iloc[: len(expected_smiles)].reset_index(drop=True)
            if actual_prefix.equals(expected_smiles):
                return pred_values.iloc[: len(expected_smiles)].to_numpy(dtype=float)

    if pred_values.isna().any():
        raise ValueError(
            "Chemprop prediction output contains non-numeric values "
            f"(file={preds_csv_path}, prediction_col={prediction_col})."
        )
    if len(pred_values) > len(expected_smiles):
        pred_values = pred_values.iloc[: len(expected_smiles)].reset_index(drop=True)
    if len(pred_values) != len(expected_smiles):
        raise ValueError(
            f"Chemprop prediction length mismatch: got {len(pred_values)} rows, expected {len(expected_smiles)} "
            f"(file={preds_csv_path}, prediction_col={prediction_col})."
        )
    return pred_values.to_numpy(dtype=float)


def _is_chemprop_descriptor_scale_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = [
        "input x contains infinity",
        "value too large for dtype",
        "standardscaler",
        "check_array",
    ]
    return any(marker in message for marker in markers)


def train_chemprop_model(
    *,
    label: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    smiles_train: pd.Series,
    smiles_test: pd.Series,
    args: argparse.Namespace,
    dataset_dir: Path,
    split_strategy_for_cv: str,
    featurizers: list[str] | None = None,
    variant_tag: str = "base",
    architecture_key: str = "dmpnn",
    workflow_label: str = "Chemprop v2",
    extra_train_args: list[str] | None = None,
    use_selected_descriptors: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    command_prefix = _resolve_chemprop_command()
    featurizer_list = [str(item).strip() for item in list(featurizers or []) if str(item).strip()]
    train_extra_args = [str(item).strip() for item in list(extra_train_args or []) if str(item).strip()]
    architecture_slug = slugify(str(architecture_key or "dmpnn"))
    variant_slug = slugify(str(variant_tag or "base"))
    save_dir = dataset_dir / "chemprop_v2" / (
        f"{architecture_slug}_{variant_slug}_ensemble_{int(args.chemprop_ensemble_size)}_seed_{int(args.chemprop_random_seed)}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    train_csv = save_dir / "train.csv"
    test_csv = save_dir / "test.csv"
    train_preds_path = save_dir / "train_predictions.csv"
    test_preds_path = save_dir / "test_predictions.csv"
    train_df = pd.DataFrame({"SMILES": smiles_train.astype(str).reset_index(drop=True), "TARGET": y_train.to_numpy(dtype=float)})
    test_df = pd.DataFrame({"SMILES": smiles_test.astype(str).reset_index(drop=True), "TARGET": y_test.to_numpy(dtype=float)})
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    descriptor_args_train: list[str] = []
    descriptor_args_train_predict: list[str] = []
    descriptor_args_test_predict: list[str] = []
    selected_descriptor_count = 0
    selected_descriptor_columns_sha256 = ""
    if bool(use_selected_descriptors):
        train_descriptor_path = save_dir / "train_descriptors.npz"
        test_descriptor_path = save_dir / "test_descriptors.npz"
        selected_descriptor_columns = [str(col) for col in X_train.columns]
        selected_descriptor_count = int(len(selected_descriptor_columns))
        selected_descriptor_columns_sha256 = hashlib.sha256(
            "||".join(selected_descriptor_columns).encode("utf-8")
        ).hexdigest()
        train_descriptor_values = np.nan_to_num(
            X_train.to_numpy(dtype=np.float32, copy=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        test_descriptor_values = np.nan_to_num(
            X_test.to_numpy(dtype=np.float32, copy=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.savez(train_descriptor_path, train_descriptor_values)
        np.savez(test_descriptor_path, test_descriptor_values)
        descriptor_args_train = ["--descriptors-path", str(train_descriptor_path)]
        descriptor_args_train_predict = ["--descriptors-path", str(train_descriptor_path)]
        descriptor_args_test_predict = ["--descriptors-path", str(test_descriptor_path)]

    pred_train: np.ndarray | None = None
    pred_test: np.ndarray | None = None
    feature_args: list[str] = []
    if featurizer_list:
        feature_args = ["--molecule-featurizers", *featurizer_list]
    chemprop_split_mode = "SCAFFOLD_BALANCED" if str(split_strategy_for_cv).strip().lower() == "scaffold" else "RANDOM"
    effective_featurizers = list(featurizer_list)
    descriptor_fallback_applied = False

    if bool(args.chemprop_reuse_model_cache) and train_preds_path.exists() and test_preds_path.exists():
        try:
            pred_train = _extract_chemprop_predictions(train_preds_path, train_df["SMILES"].astype(str))
            pred_test = _extract_chemprop_predictions(test_preds_path, test_df["SMILES"].astype(str))
            print(f"[Chemprop] loaded cached predictions from {save_dir}", flush=True)
        except Exception:
            pred_train = None
            pred_test = None

    if pred_train is None or pred_test is None:
        def _remove_stale_prediction_file(path: Path) -> None:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        if bool(args.chemprop_reuse_model_cache):
            try:
                _remove_stale_prediction_file(train_preds_path)
                _run_chemprop_command(
                    command_prefix,
                    [
                        "predict",
                        "--test-path", str(train_csv),
                        "--smiles-columns", "SMILES",
                        "--model-paths", str(save_dir),
                        "--preds-path", str(train_preds_path),
                        *feature_args,
                        *descriptor_args_train_predict,
                    ],
                    description="train-set prediction (cached model)",
                )
                _remove_stale_prediction_file(test_preds_path)
                _run_chemprop_command(
                    command_prefix,
                    [
                        "predict",
                        "--test-path", str(test_csv),
                        "--smiles-columns", "SMILES",
                        "--model-paths", str(save_dir),
                        "--preds-path", str(test_preds_path),
                        *feature_args,
                        *descriptor_args_test_predict,
                    ],
                    description="test-set prediction (cached model)",
                )
                pred_train = _extract_chemprop_predictions(train_preds_path, train_df["SMILES"].astype(str))
                pred_test = _extract_chemprop_predictions(test_preds_path, test_df["SMILES"].astype(str))
            except Exception:
                pred_train = None
                pred_test = None

    if pred_train is None or pred_test is None:
        def _train_and_predict_with_feature_args(active_feature_args: list[str], *, retry_suffix: str = "") -> tuple[np.ndarray, np.ndarray]:
            _run_chemprop_command(
                command_prefix,
                [
                    "train",
                    "--data-path", str(train_csv),
                    "--smiles-columns", "SMILES",
                    "--target-columns", "TARGET",
                    "--task-type", "regression",
                    "--epochs", str(int(args.chemprop_epochs)),
                    "--batch-size", str(int(args.chemprop_batch_size)),
                    "--num-workers", str(int(args.chemprop_num_workers)),
                    "--ensemble-size", str(int(args.chemprop_ensemble_size)),
                    "--pytorch-seed", str(int(args.chemprop_random_seed)),
                    "--data-seed", str(int(args.chemprop_random_seed)),
                    "--split", str(chemprop_split_mode),
                    "--split-sizes", "0.9", "0.1", "0.0",
                    "--output-dir", str(save_dir),
                    *train_extra_args,
                    *active_feature_args,
                    *descriptor_args_train,
                ],
                description=f"training{retry_suffix}",
            )
            _remove_stale_prediction_file(train_preds_path)
            _run_chemprop_command(
                command_prefix,
                [
                    "predict",
                    "--test-path", str(train_csv),
                    "--smiles-columns", "SMILES",
                    "--model-paths", str(save_dir),
                    "--preds-path", str(train_preds_path),
                    *active_feature_args,
                    *descriptor_args_train_predict,
                ],
                description=f"train-set prediction{retry_suffix}",
            )
            _remove_stale_prediction_file(test_preds_path)
            _run_chemprop_command(
                command_prefix,
                [
                    "predict",
                    "--test-path", str(test_csv),
                    "--smiles-columns", "SMILES",
                    "--model-paths", str(save_dir),
                    "--preds-path", str(test_preds_path),
                    *active_feature_args,
                    *descriptor_args_test_predict,
                ],
                description=f"test-set prediction{retry_suffix}",
            )
            return (
                _extract_chemprop_predictions(train_preds_path, train_df["SMILES"].astype(str)),
                _extract_chemprop_predictions(test_preds_path, test_df["SMILES"].astype(str)),
            )

        try:
            pred_train, pred_test = _train_and_predict_with_feature_args(feature_args)
        except Exception as exc:
            can_retry_without_rdkit2d = (
                "rdkit_2d" in {item.strip() for item in featurizer_list}
                and bool(feature_args)
                and _is_chemprop_descriptor_scale_error(exc)
            )
            if not can_retry_without_rdkit2d:
                raise
            descriptor_fallback_applied = True
            effective_featurizers = [item for item in featurizer_list if str(item).strip() != "rdkit_2d"]
            feature_args = ["--molecule-featurizers", *effective_featurizers] if effective_featurizers else []
            print(
                "[Chemprop] detected non-finite RDKit2D descriptor values; retrying training without rdkit_2d features.",
                flush=True,
            )
            pred_train, pred_test = _train_and_predict_with_feature_args(
                feature_args,
                retry_suffix=" (retry without rdkit_2d)",
            )

    pred_train = np.asarray(pred_train, dtype=float).reshape(-1)
    pred_test = np.asarray(pred_test, dtype=float).reshape(-1)
    execution_mode = "CPU"
    try:
        import torch

        if bool(torch.cuda.is_available()):
            execution_mode = "GPU"
    except Exception:
        execution_mode = "CPU"

    primary_metric = current_dataset_primary_metric("rmse")
    row = {
        "model": str(label),
        "workflow": str(workflow_label),
        "execution_mode": execution_mode,
        "cv_folds": np.nan,
        "cv_split_strategy": "",
        "cv_r2": np.nan,
        "cv_rmse": np.nan,
        "cv_mae": np.nan,
        "primary_metric": primary_metric,
        "cv_primary": np.nan,
        "chemprop_model_dir": str(save_dir),
        "chemprop_epochs": int(args.chemprop_epochs),
        "chemprop_batch_size": int(args.chemprop_batch_size),
        "chemprop_num_workers": int(args.chemprop_num_workers),
        "chemprop_ensemble_size": int(args.chemprop_ensemble_size),
        "chemprop_random_seed": int(args.chemprop_random_seed),
        "chemprop_split_mode": str(chemprop_split_mode),
        "chemprop_architecture": str(architecture_slug),
        "chemprop_variant_tag": str(variant_slug),
        "chemprop_train_extra_args": " ".join(train_extra_args),
        "chemprop_molecule_featurizers": ",".join(effective_featurizers),
        "chemprop_rdkit2d_fallback_applied": bool(descriptor_fallback_applied),
        "chemprop_uses_selected_descriptors": bool(use_selected_descriptors),
        "chemprop_selected_descriptor_count": int(selected_descriptor_count),
        "chemprop_selected_descriptor_columns_sha256": str(selected_descriptor_columns_sha256),
    }
    row.update(regression_metrics(y_train, pred_train, y_test, pred_test))
    row["primary_metric_value"] = float(
        compute_primary_metric(
            primary_metric,
            y_test.to_numpy(dtype=float),
            np.asarray(pred_test, dtype=float),
        )
    )
    return row, pred_train, pred_test


def train_unimol_v1_model(
    *,
    label: str,
    y_train: pd.Series,
    y_test: pd.Series,
    smiles_train: pd.Series,
    smiles_test: pd.Series,
    args: argparse.Namespace,
    dataset_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    try:
        from unimol_tools import MolPredict, MolTrain
    except Exception as exc:
        raise RuntimeError(
            "Uni-Mol V1 requires `unimol_tools` in the active environment."
        ) from exc

    save_dir = dataset_dir / "unimol_v1" / f"seed_{int(args.random_seed)}"
    save_dir.mkdir(parents=True, exist_ok=True)
    train_csv = save_dir / "train_unimol_v1.csv"
    test_csv = save_dir / "test_unimol_v1.csv"
    train_pred_cache = save_dir / "pred_train.npy"
    test_pred_cache = save_dir / "pred_test.npy"

    train_df = pd.DataFrame(
        {
            "SMILES": pd.Series(smiles_train, dtype=str).astype(str).reset_index(drop=True),
            "TARGET": pd.Series(y_train, dtype=float).reset_index(drop=True),
        }
    )
    test_df = pd.DataFrame(
        {
            "SMILES": pd.Series(smiles_test, dtype=str).astype(str).reset_index(drop=True),
            "TARGET": pd.Series(y_test, dtype=float).reset_index(drop=True),
        }
    )
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    pred_train: np.ndarray | None = None
    pred_test: np.ndarray | None = None
    if bool(getattr(args, "unimol_reuse_model_cache", True)):
        if train_pred_cache.exists() and test_pred_cache.exists():
            try:
                pred_train = np.asarray(np.load(train_pred_cache), dtype=float).reshape(-1)
                pred_test = np.asarray(np.load(test_pred_cache), dtype=float).reshape(-1)
                if len(pred_train) != len(train_df) or len(pred_test) != len(test_df):
                    pred_train, pred_test = None, None
            except Exception:
                pred_train, pred_test = None, None

    if pred_train is None or pred_test is None:
        trainer = MolTrain(
            task="regression",
            data_type="molecule",
            model_name="unimolv1",
            epochs=int(args.unimol_epochs),
            learning_rate=float(args.unimol_learning_rate),
            batch_size=int(args.unimol_batch_size),
            early_stopping=int(args.unimol_early_stopping),
            metrics="mse",
            split=str(args.unimol_internal_split),
            save_path=str(save_dir),
            num_workers=int(args.unimol_num_workers),
        )
        trainer.fit(str(train_csv))
        predictor = MolPredict(load_model=str(save_dir))
        pred_train = np.asarray(predictor.predict(str(train_csv))).reshape(-1)
        pred_test = np.asarray(predictor.predict(str(test_csv))).reshape(-1)
        try:
            np.save(train_pred_cache, np.asarray(pred_train, dtype=float))
            np.save(test_pred_cache, np.asarray(pred_test, dtype=float))
        except Exception:
            pass

    execution_mode = "GPU" if detect_gpu_available() else "CPU"
    primary_metric = current_dataset_primary_metric("rmse")
    row = {
        "model": str(label),
        "workflow": "Uni-Mol",
        "execution_mode": execution_mode,
        "cv_folds": np.nan,
        "cv_split_strategy": "",
        "cv_r2": np.nan,
        "cv_rmse": np.nan,
        "cv_mae": np.nan,
        "primary_metric": primary_metric,
        "cv_primary": np.nan,
        "unimol_model_dir": str(save_dir),
        "unimol_internal_split": str(args.unimol_internal_split),
        "unimol_epochs": int(args.unimol_epochs),
        "unimol_learning_rate": float(args.unimol_learning_rate),
        "unimol_batch_size": int(args.unimol_batch_size),
        "unimol_early_stopping": int(args.unimol_early_stopping),
        "unimol_num_workers": int(args.unimol_num_workers),
    }
    row.update(regression_metrics(y_train, pred_train, y_test, pred_test))
    row["primary_metric_value"] = float(
        compute_primary_metric(
            primary_metric,
            pd.Series(y_test, dtype=float).to_numpy(dtype=float),
            np.asarray(pred_test, dtype=float),
        )
    )
    return row, np.asarray(pred_train, dtype=float), np.asarray(pred_test, dtype=float)


def build_ensemble_result(
    *,
    payloads: dict[str, dict[str, Any]],
    method: str,
    stacking_cv_folds: int,
    random_seed: int,
    drop_highly_correlated_members: bool,
    max_train_prediction_correlation: float,
    exclude_negative_test_r2_members: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str], Any]:
    if len(payloads) < 2:
        raise ValueError("At least two model predictions are required to build an ensemble.")

    def build_split_frame(split_name: str) -> tuple[pd.DataFrame, list[str]]:
        merged: pd.DataFrame | None = None
        prediction_columns: list[str] = []
        for model_name, payload in payloads.items():
            split_df = pd.DataFrame(
                {
                    "SMILES": pd.Series(payload[f"{split_name}_smiles"], dtype=str).str.strip(),
                    "Observed": np.asarray(payload[f"{split_name}_observed"], dtype=float),
                    str(model_name): np.asarray(payload[split_name], dtype=float),
                }
            )
            split_df = split_df.drop_duplicates(subset=["SMILES"], keep="first")
            if merged is None:
                merged = split_df
            else:
                merged = merged.merge(split_df, on=["SMILES"], how="inner", suffixes=("", "__new_obs"))
                if "Observed__new_obs" in merged.columns:
                    merged = merged.drop(columns=["Observed__new_obs"])
            prediction_columns.append(str(model_name))
        if merged is None:
            raise ValueError("No predictions were available for ensemble alignment.")
        return merged.reset_index(drop=True), prediction_columns

    aligned_train, prediction_columns = build_split_frame("train")
    aligned_test, _ = build_split_frame("test")
    if aligned_train.empty or aligned_test.empty:
        raise ValueError("Selected models do not share molecules for ensemble alignment.")

    member_metrics: dict[str, dict[str, float]] = {}
    for model_name in prediction_columns:
        split_metrics = {
            "Train RMSE": float(math.sqrt(mean_squared_error(aligned_train["Observed"], aligned_train[model_name]))),
            "Test RMSE": float(math.sqrt(mean_squared_error(aligned_test["Observed"], aligned_test[model_name]))),
            "Train R2": float(r2_score(aligned_train["Observed"], aligned_train[model_name])),
            "Test R2": float(r2_score(aligned_test["Observed"], aligned_test[model_name])),
            "Train MAE": float(mean_absolute_error(aligned_train["Observed"], aligned_train[model_name])),
            "Test MAE": float(mean_absolute_error(aligned_test["Observed"], aligned_test[model_name])),
            "Train Spearman": float(
            pd.Series(aligned_train["Observed"]).corr(pd.Series(aligned_train[model_name]), method="spearman")
            ),
            "Test Spearman": float(
            pd.Series(aligned_test["Observed"]).corr(pd.Series(aligned_test[model_name]), method="spearman")
            ),
        }
        member_metrics[model_name] = split_metrics

    member_filter_notes: list[str] = []
    active_columns = list(prediction_columns)
    if bool(exclude_negative_test_r2_members):
        positive_test_columns = [name for name in active_columns if float(member_metrics[name]["Test R2"]) > 0.0]
        removed_negative = [name for name in active_columns if name not in positive_test_columns]
        if removed_negative and len(positive_test_columns) >= 2:
            active_columns = positive_test_columns
            member_filter_notes.append("Dropped members with non-positive overlap Test R2: " + ", ".join(removed_negative))

    if bool(drop_highly_correlated_members) and len(active_columns) > 2:
        threshold = float(min(max(max_train_prediction_correlation, 0.0), 0.999999))
        removed_correlated = []
        while len(active_columns) > 2:
            corr_matrix = (
                aligned_train[active_columns]
                .corr()
                .abs()
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            corr_values = corr_matrix.to_numpy(dtype=float, copy=True)
            np.fill_diagonal(corr_values, 0.0)
            max_corr = float(corr_values.max())
            if max_corr <= threshold:
                break
            max_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
            model_a = str(corr_matrix.index[max_idx[0]])
            model_b = str(corr_matrix.columns[max_idx[1]])
            rmse_a = float(member_metrics[model_a]["Test RMSE"])
            rmse_b = float(member_metrics[model_b]["Test RMSE"])
            r2_a = float(member_metrics[model_a]["Test R2"])
            r2_b = float(member_metrics[model_b]["Test R2"])
            if rmse_a > rmse_b:
                drop_model = model_a
            elif rmse_b > rmse_a:
                drop_model = model_b
            else:
                drop_model = model_a if r2_a < r2_b else model_b
            removed_correlated.append((model_a, model_b, drop_model, max_corr))
            active_columns = [name for name in active_columns if name != drop_model]
        if removed_correlated:
            details = [f"{drop} (pair={a}/{b}, corr={corr:.3f})" for a, b, drop, corr in removed_correlated]
            member_filter_notes.append("Dropped highly correlated members using overlap train predictions: " + "; ".join(details))

    if len(active_columns) < 2:
        raise ValueError("Ensemble filtering left fewer than two members.")

    prediction_columns = list(active_columns)
    X_meta_train = aligned_train[prediction_columns].to_numpy(dtype=float)
    X_meta_test = aligned_test[prediction_columns].to_numpy(dtype=float)
    y_meta_train = aligned_train["Observed"].to_numpy(dtype=float)
    y_meta_test = aligned_test["Observed"].to_numpy(dtype=float)

    meta_model = None
    ensemble_method_label = str(method)
    if str(method) == "OOF Stacking (RidgeCV)":
        n_splits = int(min(max(2, int(stacking_cv_folds)), len(aligned_train)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
        meta_model = RidgeCV(alphas=np.logspace(-6, 3, 30), fit_intercept=True)
        ensemble_train_pred = np.asarray(cross_val_predict(meta_model, X_meta_train, y_meta_train, cv=cv, method="predict")).reshape(-1)
        meta_model.fit(X_meta_train, y_meta_train)
        ensemble_test_pred = np.asarray(meta_model.predict(X_meta_test)).reshape(-1)
        ensemble_method_label = f"OOF Stacking (RidgeCV, {n_splits}-fold)"
        raw_coeffs = np.asarray(getattr(meta_model, "coef_", np.zeros(len(prediction_columns))), dtype=float).reshape(-1)
        abs_total = float(np.abs(raw_coeffs).sum())
        norm_contrib = np.abs(raw_coeffs) / abs_total if abs_total > 0 else np.zeros_like(raw_coeffs)
        weight_df = pd.DataFrame(
            {
                "Model": prediction_columns,
                "Weight": raw_coeffs,
                "Abs normalized contribution": norm_contrib,
                "Workflow": [payloads[name]["workflow"] for name in prediction_columns],
            }
        )
    elif str(method) == "Weighted average (inverse train RMSE)":
        raw_weights = np.asarray([1.0 / max(float(member_metrics[name]["Train RMSE"]), 1e-8) for name in prediction_columns], dtype=float)
        weights = raw_weights / raw_weights.sum()
        ensemble_train_pred = np.dot(X_meta_train, weights)
        ensemble_test_pred = np.dot(X_meta_test, weights)
        weight_df = pd.DataFrame({"Model": prediction_columns, "Weight": weights, "Workflow": [payloads[name]["workflow"] for name in prediction_columns]})
    else:
        weights = np.ones(len(prediction_columns), dtype=float) / float(len(prediction_columns))
        ensemble_train_pred = np.dot(X_meta_train, weights)
        ensemble_test_pred = np.dot(X_meta_test, weights)
        weight_df = pd.DataFrame({"Model": prediction_columns, "Weight": weights, "Workflow": [payloads[name]["workflow"] for name in prediction_columns]})

    ensemble_rows: list[dict[str, Any]] = []
    primary_metric = current_dataset_primary_metric("rmse")
    for model_name in prediction_columns:
        row = {"model": model_name, "workflow": str(payloads[model_name]["workflow"])}
        row.update(regression_metrics(y_meta_train, aligned_train[model_name], y_meta_test, aligned_test[model_name]))
        row["primary_metric"] = primary_metric
        row["primary_metric_value"] = float(
            compute_primary_metric(
                primary_metric,
                np.asarray(y_meta_test, dtype=float),
                np.asarray(aligned_test[model_name], dtype=float),
            )
        )
        ensemble_rows.append(row)

    ensemble_row = {"model": f"Ensemble ({ensemble_method_label})", "workflow": "ensemble"}
    ensemble_row.update(regression_metrics(y_meta_train, ensemble_train_pred, y_meta_test, ensemble_test_pred))
    ensemble_row["primary_metric"] = primary_metric
    ensemble_row["primary_metric_value"] = float(
        compute_primary_metric(
            primary_metric,
            np.asarray(y_meta_test, dtype=float),
            np.asarray(ensemble_test_pred, dtype=float),
        )
    )
    ensemble_rows.append(ensemble_row)
    ensemble_results = pd.DataFrame(ensemble_rows).sort_values(["test_rmse", "test_mae"], ascending=True).reset_index(drop=True)
    return (
        ensemble_results,
        weight_df,
        np.asarray(ensemble_train_pred, dtype=float),
        np.asarray(ensemble_test_pred, dtype=float),
        prediction_columns,
        member_filter_notes,
        meta_model,
    )


def write_selector_outputs(dataset_dir: Path, selector_meta: dict[str, Any]) -> None:
    selected_features = selector_meta.get("selected_features", [])
    pd.DataFrame({"feature": selected_features}).to_csv(dataset_dir / "selected_features.csv", index=False)
    coefficients = selector_meta.get("selector_coefficients")
    if isinstance(coefficients, pd.DataFrame):
        coefficients.to_csv(dataset_dir / "selector_coefficients.csv", index=False)


def write_feature_dedup_outputs(dataset_dir: Path, dedup_meta: dict[str, Any]) -> None:
    dropped_rows = []
    for feature_name in list(dedup_meta.get("dropped_low_variance_columns", [])):
        dropped_rows.append({"feature": str(feature_name), "reason": "low_variance"})
    for feature_name in list(dedup_meta.get("dropped_binary_prevalence_columns", [])):
        dropped_rows.append({"feature": str(feature_name), "reason": "binary_prevalence"})
    for feature_name in list(dedup_meta.get("dropped_exact_columns", [])):
        dropped_rows.append({"feature": str(feature_name), "reason": "exact_duplicate"})
    for feature_name in list(dedup_meta.get("dropped_near_columns", [])):
        dropped_rows.append({"feature": str(feature_name), "reason": "near_duplicate"})
    for feature_name in list(dedup_meta.get("dropped_moderate_columns", [])):
        dropped_rows.append({"feature": str(feature_name), "reason": "moderate_correlation"})
    pd.DataFrame(dropped_rows, columns=["feature", "reason"]).to_csv(
        dataset_dir / "dropped_duplicate_features.csv",
        index=False,
    )


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


def load_partial_dataset_artifacts(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    _status_path, metrics_path = dataset_status_paths(dataset_dir)
    predictions_path = dataset_dir / "predictions.csv"
    ga_history_path = dataset_dir / "ga_history.csv"
    metrics_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    ga_history_tables: list[pd.DataFrame] = []
    if metrics_path.exists():
        try:
            metrics_rows = pd.read_csv(metrics_path).to_dict(orient="records")
        except Exception:
            metrics_rows = []
    if predictions_path.exists():
        try:
            prediction_tables = [pd.read_csv(predictions_path)]
        except Exception:
            prediction_tables = []
    if ga_history_path.exists():
        try:
            ga_history_tables = [pd.read_csv(ga_history_path)]
        except Exception:
            ga_history_tables = []
    return metrics_rows, prediction_tables, ga_history_tables


def rebuild_prediction_payloads(prediction_tables: list[pd.DataFrame]) -> dict[str, dict[str, Any]]:
    if not prediction_tables:
        return {}
    combined = pd.concat(prediction_tables, ignore_index=True)
    required_cols = {"model", "workflow", "split", "smiles", "observed", "predicted"}
    if not required_cols.issubset(set(combined.columns)):
        return {}
    payloads: dict[str, dict[str, Any]] = {}
    for model_name, model_df in combined.groupby("model", sort=False):
        workflow_name = str(model_df["workflow"].dropna().iloc[0]) if not model_df["workflow"].dropna().empty else ""
        split_tables: dict[str, pd.DataFrame] = {}
        for split_name in ("train", "test"):
            split_df = model_df.loc[model_df["split"].astype(str).str.lower() == split_name].copy()
            if split_df.empty:
                continue
            split_df["smiles"] = split_df["smiles"].astype(str).str.strip()
            split_df = split_df.drop_duplicates(subset=["smiles"], keep="first")
            split_tables[split_name] = split_df
        if "train" not in split_tables or "test" not in split_tables:
            continue
        payloads[str(model_name)] = {
            "workflow": workflow_name,
            "train_smiles": split_tables["train"]["smiles"].reset_index(drop=True),
            "test_smiles": split_tables["test"]["smiles"].reset_index(drop=True),
            "train_observed": split_tables["train"]["observed"].to_numpy(dtype=float),
            "test_observed": split_tables["test"]["observed"].to_numpy(dtype=float),
            "train": split_tables["train"]["predicted"].to_numpy(dtype=float),
            "test": split_tables["test"]["predicted"].to_numpy(dtype=float),
        }
    return payloads


def is_ensemble_result_row(model_name: Any, workflow_name: Any = "") -> bool:
    model_text = str(model_name or "").strip().lower()
    workflow_text = str(workflow_name or "").strip().lower()
    if workflow_text == "ensemble":
        return True
    if model_text == "ensemble":
        return True
    if model_text.startswith("ensemble ("):
        return True
    return False


def write_dataset_status(dataset_dir: Path, payload: dict[str, Any]) -> None:
    status_path, _metrics_path = dataset_status_paths(dataset_dir)
    status_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def selected_conventional_model_names(args: argparse.Namespace) -> list[str]:
    names = [
        "ElasticNetCV",
        "SVR",
        "Random forest",
        "Extra trees",
        "HistGradientBoosting",
        "Voting Regressor (KNN, SVM)",
        "AdaBoost",
        "Tabular MLP",
    ]
    if XGBRegressor is not None:
        names.append("XGBoost")
    if LGBMRegressor is not None:
        names.append("LightGBM")
    if CatBoostRegressor is not None:
        names.append("CatBoost")
        names.append(maplight_catboost_model_label(args))
    if bool(getattr(args, "run_tabpfn", False)) and TabPFNRegressor is not None:
        names.append("TabPFNRegressor")
    return names


def _row_has_error_text(row: dict[str, Any]) -> bool:
    error_text = str(row.get("error", "")).strip().lower()
    return bool(error_text and error_text not in {"nan", "none"})


def successful_model_names_from_metrics_rows(metrics_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("model", "")).strip()
        for row in metrics_rows
        if str(row.get("model", "")).strip() and not _row_has_error_text(row)
    }


def expected_model_targets_for_args(args: argparse.Namespace) -> tuple[set[str], bool]:
    expected_names: set[str] = set(selected_conventional_model_names(args))
    if bool(getattr(args, "run_cfa", False)):
        expected_names.add("CFA (Combinatorial Fusion)")
    requested_ga_models = parse_comma_list(
        getattr(args, "ga_models_resolved", getattr(args, "ga_models", ""))
    )
    expected_names.update({f"{model_name} GA" for model_name in requested_ga_models})
    if bool(getattr(args, "run_chemml_pytorch", False)):
        expected_names.add("ChemML MLP (PyTorch)")
    if bool(getattr(args, "run_chemml_tensorflow", False)):
        expected_names.add("ChemML MLP (TensorFlow)")
    if bool(getattr(args, "run_chemprop_mpnn", False)):
        expected_names.update(
            {
                str(spec.get("label", "")).strip()
                for spec in chemprop_variant_specs(args)
                if str(spec.get("label", "")).strip()
            }
        )
    if bool(getattr(args, "run_unimol_v1", False)):
        expected_names.add("Uni-Mol V1")
    if bool(getattr(args, "run_maplight_gnn", False)):
        expected_names.add(maplight_gnn_model_label(args))
    return expected_names, bool(getattr(args, "run_ensemble", False))


def missing_requested_models_for_dataset(
    metrics_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[str], bool]:
    completed_names = successful_model_names_from_metrics_rows(metrics_rows)
    expected_names, ensemble_expected = expected_model_targets_for_args(args)
    missing = sorted(name for name in expected_names if name not in completed_names)
    ensemble_completed = any(
        is_ensemble_result_row(row.get("model", ""), row.get("workflow", ""))
        and not _row_has_error_text(row)
        for row in metrics_rows
    )
    missing_ensemble = bool(ensemble_expected and not ensemble_completed)
    return missing, missing_ensemble


def has_successful_ensemble_result(metrics_rows: list[dict[str, Any]]) -> bool:
    return any(
        is_ensemble_result_row(row.get("model", ""), row.get("workflow", ""))
        and not _row_has_error_text(row)
        for row in metrics_rows
    )


def model_stage_type_label(model_name: str) -> str:
    text = str(model_name or "").strip()
    lower = text.lower()
    if lower == "ensemble" or lower.startswith("ensemble ("):
        return "Ensemble"
    if lower.startswith("cfa (") or lower == "cfa":
        return "CFA fusion"
    if text.endswith(" GA"):
        return "GA tuned conventional"
    if lower.startswith("chemml mlp"):
        return "ChemML deep learning"
    if lower.startswith("chemprop v2"):
        return "Chemprop deep learning"
    if lower.startswith("uni-mol"):
        return "Uni-Mol deep learning"
    if lower.startswith("maplight + gnn"):
        return "MapLight + GNN deep learning"
    return "Conventional ML"


def build_resume_execution_plan(
    datasets: list["DatasetSpec"],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    datasets_pending = 0
    datasets_reused = 0
    model_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()

    for spec in datasets:
        dataset_id = slugify(spec.name)
        dataset_dir = output_dir / dataset_id
        completed_result = load_completed_dataset_result(dataset_dir)
        metrics_rows: list[dict[str, Any]] = []
        if completed_result is not None:
            metrics_rows = list(completed_result.metrics_rows)
        elif dataset_dir.exists():
            metrics_rows, _prediction_tables, _ga_history_tables = load_partial_dataset_artifacts(dataset_dir)

        missing_models, missing_ensemble = missing_requested_models_for_dataset(metrics_rows, args)
        pending_models = list(missing_models)
        if (
            bool(getattr(args, "run_ensemble", False))
            and bool(missing_models)
            and has_successful_ensemble_result(metrics_rows)
        ):
            if "Ensemble" not in pending_models:
                pending_models.append("Ensemble")
            missing_ensemble = True
        if bool(getattr(args, "run_ensemble", False)) and bool(getattr(args, "rebuild_ensemble", False)):
            if "Ensemble" not in pending_models:
                pending_models.append("Ensemble")
            missing_ensemble = True
        elif missing_ensemble and "Ensemble" not in pending_models:
            pending_models.append("Ensemble")

        if completed_result is not None and not bool(getattr(args, "revisit_completed_datasets", False)) and not pending_models:
            datasets_reused += 1
            continue

        if pending_models:
            datasets_pending += 1
            for model_name in pending_models:
                model_text = str(model_name).strip()
                if not model_text:
                    continue
                model_counter[model_text] += 1
                type_counter[model_stage_type_label(model_text)] += 1

    return {
        "datasets_total": int(len(datasets)),
        "datasets_pending": int(datasets_pending),
        "datasets_reused": int(datasets_reused),
        "model_counter": model_counter,
        "type_counter": type_counter,
        "total_model_stage_executions": int(sum(model_counter.values())),
    }


def print_resume_execution_plan(plan: dict[str, Any]) -> None:
    datasets_total = int(plan.get("datasets_total", 0))
    datasets_pending = int(plan.get("datasets_pending", 0))
    datasets_reused = int(plan.get("datasets_reused", 0))
    total_model_stage_executions = int(plan.get("total_model_stage_executions", 0))
    type_counter = Counter(plan.get("type_counter", {}))
    model_counter = Counter(plan.get("model_counter", {}))

    print(
        "Resume execution plan: "
        f"{datasets_pending}/{datasets_total} dataset(s) require model execution; "
        f"{datasets_reused} dataset(s) can be reused as-completed.",
        flush=True,
    )
    print(
        f"Planned model-stage executions across pending datasets: {total_model_stage_executions}",
        flush=True,
    )
    if type_counter:
        type_summary = ", ".join(
            f"{stage_type}={int(count)}"
            for stage_type, count in sorted(type_counter.items(), key=lambda item: (-item[1], item[0]))
        )
        print(f"Planned stage types: {type_summary}", flush=True)
    if model_counter:
        top_models = ", ".join(
            f"{name}={int(count)}"
            for name, count in sorted(model_counter.items(), key=lambda item: (-item[1], item[0]))[:12]
        )
        print(f"Most frequent pending models: {top_models}", flush=True)


def chemprop_variant_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    architecture_keys: list[str] = []
    if bool(getattr(args, "run_chemprop_dmpnn", True)):
        architecture_keys.append("dmpnn")
    if bool(getattr(args, "run_chemprop_cmpnn", True)):
        architecture_keys.append("cmpnn")
    if bool(getattr(args, "run_chemprop_attentivefp", True)):
        architecture_keys.append("attentivefp")
    if not architecture_keys:
        return []
    include_selected_feature_variant = bool(getattr(args, "run_chemprop_selected_features", False))
    return resolve_chemprop_architecture_specs(
        architecture_keys,
        ensemble_size=int(args.chemprop_ensemble_size),
        include_rdkit2d_extra=bool(getattr(args, "run_chemprop_rdkit2d", False)),
        include_selected_feature_variant=include_selected_feature_variant,
    )


def run_dataset(spec: DatasetSpec, output_dir: Path, args: argparse.Namespace, dataset_position: int | None = None, dataset_total: int | None = None) -> DatasetRunResult:
    start = time.time()
    dataset_id = slugify(spec.name)
    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = dataset_dir / "metrics.csv"
    predictions_path = dataset_dir / "predictions.csv"
    ga_history_path = dataset_dir / "ga_history.csv"
    rebuild_ensemble_requested = bool(getattr(args, "rebuild_ensemble", False)) and bool(getattr(args, "run_ensemble", False))
    completed_result = load_completed_dataset_result(dataset_dir)
    if completed_result is not None:
        prefix = f"[{dataset_position}/{dataset_total}] " if dataset_position is not None and dataset_total is not None else ""
        missing_models, missing_ensemble = missing_requested_models_for_dataset(
            completed_result.metrics_rows,
            args,
        )
        auto_rebuild_for_missing_coverage = bool(
            bool(getattr(args, "run_ensemble", False))
            and bool(missing_models)
            and has_successful_ensemble_result(completed_result.metrics_rows)
            and not rebuild_ensemble_requested
        )
        if auto_rebuild_for_missing_coverage:
            rebuild_ensemble_requested = True
            print(
                f"[resume] {dataset_id}: base-model coverage is incomplete and a prior ensemble exists; "
                "ensemble will be rebuilt after missing stages finish.",
                flush=True,
            )
        has_missing_targets = bool(missing_models or missing_ensemble)
        if (
            not bool(getattr(args, "revisit_completed_datasets", False))
            and not rebuild_ensemble_requested
            and not has_missing_targets
        ):
            print(f"\n{prefix}{dataset_id}: already completed in {format_seconds(completed_result.elapsed_seconds)}; reusing saved outputs")
            return completed_result
        if has_missing_targets and not bool(getattr(args, "revisit_completed_datasets", False)) and not rebuild_ensemble_requested:
            missing_label = ", ".join(missing_models[:6]) + (" ..." if len(missing_models) > 6 else "")
            if missing_ensemble:
                missing_label = (missing_label + ", ensemble").strip(", ")
            print(
                f"\n{prefix}{dataset_id}: previously completed in {format_seconds(completed_result.elapsed_seconds)}; "
                f"resuming missing requested model stages ({missing_label}).",
                flush=True,
            )
        elif rebuild_ensemble_requested and not bool(getattr(args, "revisit_completed_datasets", False)):
            print(
                f"\n{prefix}{dataset_id}: previously completed in {format_seconds(completed_result.elapsed_seconds)}; "
                "revisiting to rebuild the ensemble.",
                flush=True,
            )
        if rebuild_ensemble_requested and not bool(getattr(args, "revisit_completed_datasets", False)):
            pass
        elif not has_missing_targets:
            print(
                f"\n{prefix}{dataset_id}: previously completed in {format_seconds(completed_result.elapsed_seconds)}; "
                "revisiting to run any newly requested model stages.",
                flush=True,
            )

    requested_ga_models = parse_comma_list(getattr(args, "ga_models_resolved", getattr(args, "ga_models", "")))
    requested_deep_stages = (
        int(bool(args.run_chemml_pytorch))
        + int(bool(args.run_chemml_tensorflow))
        + (len(chemprop_variant_specs(args)) if bool(args.run_chemprop_mpnn) else 0)
        + int(bool(getattr(args, "run_unimol_v1", False)))
        + int(bool(args.run_maplight_gnn))
    )
    requested_cfa_stage = int(bool(getattr(args, "run_cfa", False)))
    requested_ensemble_stage = int(bool(args.run_ensemble))
    total_stages = (
        3
        + len(selected_conventional_model_names(args))
        + requested_cfa_stage
        + len(requested_ga_models)
        + requested_deep_stages
        + requested_ensemble_stage
    )
    step_runtime_csv_path = dataset_dir / "step_runtime.csv"
    step_runtime_json_path = dataset_dir / "step_runtime.json"
    stage_records: list[dict[str, Any]] = []
    active_stage_record: dict[str, Any] | None = None

    def _local_timestamp() -> str:
        return local_timestamp_text()

    def _close_active_stage(default_status: str = "completed") -> None:
        nonlocal active_stage_record
        if active_stage_record is None:
            return
        ended_epoch = time.time()
        active_stage_record["ended_at"] = _local_timestamp()
        active_stage_record["ended_epoch"] = ended_epoch
        active_stage_record["duration_seconds"] = round(
            ended_epoch - float(active_stage_record.get("_started_epoch", ended_epoch)),
            3,
        )
        current_status = str(active_stage_record.get("status", "")).strip().lower()
        if current_status in {"", "running"}:
            active_stage_record["status"] = str(default_status)
        stage_records.append(active_stage_record)
        active_stage_record = None

    def _set_active_stage_status(status: str, error_text: str = "") -> None:
        if active_stage_record is None:
            return
        active_stage_record["status"] = str(status)
        if str(error_text).strip():
            active_stage_record["error"] = str(error_text).strip()

    def _stage_runtime_rows_snapshot() -> list[dict[str, Any]]:
        rows = list(stage_records)
        if active_stage_record is not None:
            now_epoch = time.time()
            running_copy = dict(active_stage_record)
            running_copy["ended_at"] = _local_timestamp()
            running_copy["ended_epoch"] = now_epoch
            running_copy["duration_seconds"] = round(
                now_epoch - float(running_copy.get("_started_epoch", now_epoch)),
                3,
            )
            if not str(running_copy.get("status", "")).strip():
                running_copy["status"] = "running"
            rows.append(running_copy)
        clean_rows: list[dict[str, Any]] = []
        for row in rows:
            clean_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if not str(key).startswith("_")
                }
            )
        return clean_rows

    def _write_stage_runtime_outputs() -> None:
        runtime_rows = _stage_runtime_rows_snapshot()
        pd.DataFrame(runtime_rows).to_csv(step_runtime_csv_path, index=False)
        step_runtime_json_path.write_text(
            json.dumps(runtime_rows, indent=2, default=str),
            encoding="utf-8",
        )

    def stage_message(stage_index: int, label: str) -> None:
        nonlocal active_stage_record
        _close_active_stage(default_status="completed")
        stage_started_at = _local_timestamp()
        stage_started_epoch = time.time()
        active_stage_record = {
            "dataset": dataset_id,
            "stage_index": int(stage_index),
            "total_stages": int(total_stages),
            "stage_label": str(label),
            "started_at": stage_started_at,
            "started_epoch": stage_started_epoch,
            "_started_epoch": stage_started_epoch,
            "status": "running",
        }
        elapsed = time.time() - start
        avg_stage = elapsed / max(1, stage_index - 1) if stage_index > 1 else 0.0
        remaining = max(0, total_stages - stage_index + 1)
        eta = avg_stage * remaining if avg_stage else 0.0
        prefix = f"[{dataset_position}/{dataset_total}] " if dataset_position is not None and dataset_total is not None else ""
        print(
            f"\n{prefix}{dataset_id} | stage {stage_index}/{total_stages}: {label} "
            f"| started {stage_started_at} | elapsed {format_seconds(elapsed)} | dataset ETA {format_seconds(eta)}"
        )
        _write_stage_runtime_outputs()

    metrics_rows, prediction_tables, ga_history_tables = load_partial_dataset_artifacts(dataset_dir)
    prediction_payloads = rebuild_prediction_payloads(prediction_tables)
    if (
        bool(getattr(args, "run_ensemble", False))
        and bool(metrics_rows)
        and not rebuild_ensemble_requested
    ):
        partial_missing_models, _partial_missing_ensemble = missing_requested_models_for_dataset(metrics_rows, args)
        if bool(partial_missing_models) and has_successful_ensemble_result(metrics_rows):
            rebuild_ensemble_requested = True
            print(
                f"[resume] {dataset_id}: detected missing model coverage with an existing ensemble row in checkpoint artifacts; "
                "ensemble outputs will be rebuilt.",
                flush=True,
            )

    completed_model_names = {
        str(row.get("model", "")).strip()
        for row in metrics_rows
        if str(row.get("model", "")).strip() and not _row_has_error_text(row)
    }
    if rebuild_ensemble_requested:
        original_metric_count = len(metrics_rows)
        metrics_rows = [
            row
            for row in metrics_rows
            if not is_ensemble_result_row(row.get("model", ""), row.get("workflow", ""))
        ]
        removed_metric_rows = int(original_metric_count - len(metrics_rows))

        trimmed_prediction_tables: list[pd.DataFrame] = []
        removed_prediction_rows = 0
        for table in prediction_tables:
            if not isinstance(table, pd.DataFrame) or table.empty:
                continue
            model_col = "model" if "model" in table.columns else None
            workflow_col = "workflow" if "workflow" in table.columns else None
            if model_col is None and workflow_col is None:
                trimmed_prediction_tables.append(table)
                continue
            keep_mask = np.ones(len(table), dtype=bool)
            if model_col is not None:
                keep_mask &= ~table[model_col].map(lambda value: is_ensemble_result_row(value, ""))
            if workflow_col is not None:
                keep_mask &= ~table[workflow_col].astype(str).str.strip().str.lower().eq("ensemble")
            removed_prediction_rows += int((~keep_mask).sum())
            trimmed = table.loc[keep_mask].reset_index(drop=True)
            if not trimmed.empty:
                trimmed_prediction_tables.append(trimmed)
        prediction_tables = trimmed_prediction_tables
        prediction_payloads = rebuild_prediction_payloads(prediction_tables)
        completed_model_names = {
            str(row.get("model", "")).strip()
            for row in metrics_rows
            if str(row.get("model", "")).strip() and not _row_has_error_text(row)
        }

        removed_cached_files: list[str] = []
        for ensemble_artifact in [dataset_dir / "ensemble_results.csv", dataset_dir / "ensemble_weights.csv"]:
            if ensemble_artifact.exists():
                try:
                    ensemble_artifact.unlink()
                    removed_cached_files.append(str(ensemble_artifact.name))
                except Exception:
                    pass
        print(
            f"[resume] {dataset_id}: --rebuild-ensemble removed {removed_metric_rows} ensemble metric row(s) and "
            f"{removed_prediction_rows} ensemble prediction row(s). "
            f"{'Removed cached files: ' + ', '.join(removed_cached_files) if removed_cached_files else 'No standalone ensemble artifact files needed removal.'}",
            flush=True,
        )
    if metrics_rows:
        print(f"[resume] {dataset_id}: restored {len(metrics_rows)} metric rows from checkpoint files", flush=True)

    def current_annotated_metrics_rows() -> list[dict[str, Any]]:
        if not metrics_rows:
            return []
        annotated_df = annotate_metrics_with_leaderboard(pd.DataFrame(metrics_rows), spec)
        return annotated_df.to_dict(orient="records")

    def persist_partial(stage_label: str) -> None:
        annotated_rows = current_annotated_metrics_rows()
        if annotated_rows:
            pd.DataFrame(annotated_rows).to_csv(metrics_path, index=False)
        if prediction_tables:
            pd.concat(prediction_tables, ignore_index=True).to_csv(predictions_path, index=False)
        if ga_history_tables:
            pd.concat(ga_history_tables, ignore_index=True).to_csv(ga_history_path, index=False)
        _write_stage_runtime_outputs()
        write_dataset_status(
            dataset_dir,
            {
                "status": "running",
                "dataset": dataset_id,
                "source": spec.source,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_stage": stage_label,
                "n_metrics_rows": len(annotated_rows),
                "n_prediction_tables": len(prediction_tables),
                "n_stage_records": len(_stage_runtime_rows_snapshot()),
            },
        )

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
    use_log10_target = resolve_dataset_log10_target(spec, args)
    df, input_meta = canonicalize_frame(spec, use_log10_target)
    if len(df) < args.minimum_rows:
        print(f"[skip] {dataset_id}: only {len(df)} valid rows after cleanup")
        _set_active_stage_status("skipped")
        _close_active_stage(default_status="skipped")
        _write_stage_runtime_outputs()
        write_dataset_status(dataset_dir, {"status": "skipped", "reason": "too_few_rows_after_cleanup", "n_rows": int(len(df))})
        return DatasetRunResult([], [], [], "skipped", time.time() - start)
    if args.row_limit and len(df) > args.row_limit:
        df = df.sample(n=args.row_limit, random_state=args.random_seed).reset_index(drop=True)

    predefined_split = None
    if spec.predefined_split_column and spec.predefined_split_column in df.columns:
        predefined_split = df[spec.predefined_split_column].reset_index(drop=True)
    selector_args = argparse.Namespace(**vars(args))
    if selector_args.max_selected_features <= 0:
        selector_args.max_selected_features = max(1, math.ceil(0.10 * len(df)))
    stage23_signature_value, stage23_signature_payload = stage23_resume_signature(
        args=selector_args,
        spec=spec,
        canonical_df=df,
        input_meta=input_meta,
        predefined_split=predefined_split,
    )
    stage23_cache_payload = load_stage23_resume_cache(dataset_dir, expected_signature=stage23_signature_value)
    maplight_parity_mode = bool(getattr(args, "maplight_leaderboard_parity_mode", True))
    maplight_seed_values = maplight_parity_seed_values(args)
    maplight_direct_X_train = pd.DataFrame()
    maplight_direct_X_test = pd.DataFrame()
    shared_feature_cache_enabled = bool(getattr(args, "enable_shared_feature_matrix_cache", True))
    shared_feature_cache_reuse = bool(getattr(args, "reuse_shared_feature_matrix_cache", True))
    shared_feature_cache_path = resolve_shared_feature_matrix_cache_path(
        getattr(args, "shared_feature_matrix_cache_path", "AUTO")
    )

    if stage23_cache_payload is not None:
        stage_message(2, "building molecular features (cached)")
        _set_active_stage_status("resumed_cache")
        print(
            f"[resume] {dataset_id}: stage 2/3 cache hit (signature match); "
            "reusing split + selected feature matrices.",
            flush=True,
        )
        split_payload = dict(stage23_cache_payload.get("split", {}))
        split = {
            "X_train": pd.DataFrame(split_payload.get("X_train")).copy(),
            "X_test": pd.DataFrame(split_payload.get("X_test")).copy(),
            "y_train": pd.Series(split_payload.get("y_train"), dtype=float).reset_index(drop=True),
            "y_test": pd.Series(split_payload.get("y_test"), dtype=float).reset_index(drop=True),
            "smiles_train": pd.Series(split_payload.get("smiles_train"), dtype=str).reset_index(drop=True),
            "smiles_test": pd.Series(split_payload.get("smiles_test"), dtype=str).reset_index(drop=True),
            "split_strategy_used": str(split_payload.get("split_strategy_used", "")),
            "split_signature": dict(split_payload.get("split_signature", {})),
        }
        feature_meta = dict(stage23_cache_payload.get("feature_meta", {}))
        X_train = pd.DataFrame(stage23_cache_payload.get("X_train_selected")).copy()
        X_test = pd.DataFrame(stage23_cache_payload.get("X_test_selected")).copy()
        selector_meta = dict(stage23_cache_payload.get("selector_meta", {}))
        feature_dedup_meta = dict(stage23_cache_payload.get("feature_dedup_meta", {}))
        maplight_feature_cols = [str(col) for col in stage23_cache_payload.get("maplight_feature_cols", [])]
        cv_strategy_for_workflows = str(stage23_cache_payload.get("cv_strategy_for_workflows", "random"))
        stage_message(3, "splitting data and selecting features (cached)")
        _set_active_stage_status("resumed_cache")
        print(
            f"[resume] {dataset_id}: restored {int(X_train.shape[1]):,} selected features from stage 2/3 cache.",
            flush=True,
        )
        write_selector_outputs(dataset_dir, selector_meta)
        write_feature_dedup_outputs(dataset_dir, feature_dedup_meta)
    else:
        stage_message(2, "building molecular features")
        shared_signature, shared_signature_payload = shared_feature_matrix_signature(
            smiles_values=df["canonical_smiles"],
            selected_families=list(DEFAULT_BENCHMARK_FEATURE_FAMILIES),
            radius=2,
            n_bits=int(args.fingerprint_bits),
        )
        shared_cache_hit = False
        X = pd.DataFrame()
        feature_meta: dict[str, Any] = {}
        if shared_feature_cache_enabled and shared_feature_cache_reuse:
            loaded = load_shared_feature_matrix_cache(shared_feature_cache_path, shared_signature)
            if loaded is not None:
                X, feature_meta = loaded
                shared_cache_hit = True
                print(
                    f"[feature-cache] {dataset_id}: loaded shared feature matrix cache "
                    f"({len(X):,} rows, {int(X.shape[1]):,} columns) from {shared_feature_cache_path}.",
                    flush=True,
                )
        if not shared_cache_hit:
            X, feature_meta = build_feature_matrix_from_smiles(
                df["canonical_smiles"].tolist(),
                selected_feature_families=list(DEFAULT_BENCHMARK_FEATURE_FAMILIES),
                radius=2,
                n_bits=args.fingerprint_bits,
                enable_persistent_feature_store=args.enable_persistent_feature_store,
                reuse_persistent_feature_store=args.reuse_persistent_feature_store,
                persistent_feature_store_path=args.persistent_feature_store_path,
            )
            feature_meta = dict(feature_meta)
            feature_meta["shared_feature_matrix_cache_hit"] = False
            feature_meta["shared_feature_matrix_cache_file"] = str(
                shared_feature_matrix_cache_file(shared_feature_cache_path, shared_signature)
            )
            if shared_feature_cache_enabled:
                write_shared_feature_matrix_cache(
                    shared_feature_cache_path,
                    shared_signature,
                    shared_signature_payload,
                    X,
                    feature_meta,
                )
        y = df["target"].astype(float).reset_index(drop=True)
        smiles = df["canonical_smiles"].reset_index(drop=True)

        if len(df) < args.minimum_rows:
            print(f"[skip] {dataset_id}: only {len(df)} valid rows after feature generation")
            _set_active_stage_status("skipped")
            _close_active_stage(default_status="skipped")
            _write_stage_runtime_outputs()
            write_dataset_status(dataset_dir, {"status": "skipped", "reason": "too_few_rows_after_features", "n_rows": int(len(df))})
            return DatasetRunResult([], [], [], "skipped", time.time() - start)

        stage_message(3, "splitting data and selecting features")
        split = split_data(X, y, smiles, args, predefined_split=predefined_split)
        split["X_train"], split["X_test"], feature_dedup_meta = drop_exact_and_near_duplicate_features(
            split["X_train"],
            split["X_test"],
            variance_threshold=1e-8,
            binary_prevalence_min=0.005,
            binary_prevalence_max=0.995,
        )
        dedup_dropped_count = int(feature_dedup_meta.get("dropped_feature_count", 0))
        if dedup_dropped_count > 0:
            print(
                "[feature-dedup] dropped "
                f"{dedup_dropped_count:,} feature(s) before selector/model fitting "
                f"({int(feature_dedup_meta.get('dropped_low_variance_count', 0)):,} low-variance, "
                f"{int(feature_dedup_meta.get('dropped_binary_prevalence_count', 0)):,} binary-prevalence, "
                f"{int(feature_dedup_meta.get('dropped_exact_count', 0)):,} exact-duplicate).",
                flush=True,
            )
        if str(feature_dedup_meta.get("near_duplicate_scan_error", "")).strip():
            print(
                "[feature-dedup] near-duplicate scan warning: "
                f"{str(feature_dedup_meta.get('near_duplicate_scan_error')).strip()}",
                flush=True,
            )
        cv_strategy_for_workflows = effective_cv_split_strategy(split["split_strategy_used"])
        X_train, X_test, selector_meta = select_features(
            split["X_train"],
            split["X_test"],
            split["y_train"],
            split["smiles_train"],
            selector_args,
            split_strategy_for_cv=cv_strategy_for_workflows,
        )
        write_selector_outputs(dataset_dir, selector_meta)
        write_feature_dedup_outputs(dataset_dir, feature_dedup_meta)
        maplight_feature_cols = [
            col for col in split["X_train"].columns if str(col).startswith(MAPLIGHT_CLASSIC_PREFIXES)
        ]
        write_stage23_resume_cache(
            dataset_dir,
            stage23_signature_value=stage23_signature_value,
            signature_payload=stage23_signature_payload,
            split=split,
            X_train_selected=X_train,
            X_test_selected=X_test,
            feature_meta=feature_meta,
            selector_meta=selector_meta,
            feature_dedup_meta=feature_dedup_meta,
            maplight_feature_cols=maplight_feature_cols,
            cv_strategy_for_workflows=cv_strategy_for_workflows,
        )

    if maplight_parity_mode:
        # Build MapLight classic features directly from split SMILES so parity
        # mode is not affected by global feature de-duplication/pruning.
        maplight_direct_X_train = build_maplight_parity_matrix(split["smiles_train"], args)
        maplight_direct_X_test = build_maplight_parity_matrix(split["smiles_test"], args)
        if list(maplight_direct_X_test.columns) != list(maplight_direct_X_train.columns):
            maplight_direct_X_test = maplight_direct_X_test.reindex(columns=list(maplight_direct_X_train.columns), fill_value=np.nan)

    base_meta = {
        "dataset": dataset_id,
        "dataset_source": spec.source,
        "benchmark_suite": spec.benchmark_suite or "",
        "benchmark_id": spec.benchmark_id or "",
        "leaderboard_url": spec.leaderboard_url or "",
        "leaderboard_model": (spec.leaderboard_summary or {}).get("model", ""),
        "leaderboard_metric_name": (spec.leaderboard_summary or {}).get("metric_name", ""),
        "leaderboard_metric_value": (spec.leaderboard_summary or {}).get("metric_value", ""),
        "leaderboard_dataset_split": (spec.leaderboard_summary or {}).get("dataset_split", ""),
        "leaderboard_top10_count": int(len((spec.leaderboard_summary or {}).get("top10", []))),
        "leaderboard_top10_json": json.dumps((spec.leaderboard_summary or {}).get("top10", []), default=str),
        "n_molecules": int(len(df)),
        "n_train": int(len(split["y_train"])),
        "n_test": int(len(split["y_test"])),
        "target_transform": input_meta["target_transform"],
        "smiles_column": input_meta["smiles_column"],
        "target_column": input_meta["target_column"],
        "split_strategy": split["split_strategy_used"],
        "split_train_hash": split["split_signature"]["train_hash"],
        "split_test_hash": split["split_signature"]["test_hash"],
        "original_feature_count": int(feature_dedup_meta.get("original_feature_count", selector_meta["original_feature_count"])),
        "post_dedup_feature_count": int(feature_dedup_meta.get("post_dedup_feature_count", selector_meta["original_feature_count"])),
        "feature_dedup_dropped_count": int(feature_dedup_meta.get("dropped_feature_count", 0)),
        "feature_dedup_low_variance_count": int(feature_dedup_meta.get("dropped_low_variance_count", 0)),
        "feature_dedup_binary_prevalence_count": int(feature_dedup_meta.get("dropped_binary_prevalence_count", 0)),
        "feature_dedup_exact_count": int(feature_dedup_meta.get("dropped_exact_count", 0)),
        "feature_dedup_near_count": int(feature_dedup_meta.get("dropped_near_count", 0)),
        "feature_dedup_moderate_count": int(feature_dedup_meta.get("dropped_moderate_count", 0)),
        "feature_dedup_threshold": float(feature_dedup_meta.get("correlation_threshold", 0.999)),
        "feature_dedup_moderate_threshold": float(feature_dedup_meta.get("moderate_correlation_threshold", np.nan)),
        "feature_dedup_variance_threshold": float(feature_dedup_meta.get("variance_threshold", np.nan)),
        "feature_dedup_binary_prevalence_min": float(feature_dedup_meta.get("binary_prevalence_min", np.nan)),
        "feature_dedup_binary_prevalence_max": float(feature_dedup_meta.get("binary_prevalence_max", np.nan)),
        "feature_dedup_scan_warning": str(feature_dedup_meta.get("near_duplicate_scan_error", "")),
        "selected_feature_count": int(selector_meta["selected_feature_count"]),
        "selector_method": selector_meta["selector_method"],
        "selector_timed_out": bool(selector_meta.get("selector_timed_out", False)),
        "selector_alpha": selector_meta.get("selector_alpha", np.nan),
        "selector_l1_ratio": selector_meta.get("selector_l1_ratio", np.nan),
        "selector_n_iter": selector_meta.get("selector_n_iter", np.nan),
        "selector_cv_folds": selector_meta.get("selector_cv_folds", np.nan),
        "selector_cv_split_strategy": selector_meta.get("selector_cv_split_strategy", ""),
        "selector_max_selected_features": selector_meta.get("max_selected_features", np.nan),
        "selector_auto_rf_large_dataset_triggered": bool(
            selector_meta.get("selector_auto_rf_large_dataset_triggered", False)
        ),
        "selector_predicted_elasticnet_seconds": selector_meta.get(
            "selector_predicted_elasticnet_seconds",
            np.nan,
        ),
        "selector_auto_rf_threshold_seconds": selector_meta.get(
            "selector_auto_rf_threshold_seconds",
            np.nan,
        ),
        "selector_auto_rf_threshold_dataset_size": selector_meta.get(
            "selector_auto_rf_threshold_dataset_size",
            np.nan,
        ),
        "selected_feature_families_json": json.dumps(feature_meta.get("selected_feature_families", []), default=str),
        "built_feature_families_json": json.dumps(feature_meta.get("built_feature_families", []), default=str),
        "feature_store_path": feature_meta.get("feature_store_path", ""),
        "feature_store_shard_format": feature_meta.get("feature_store_shard_format", ""),
        "feature_store_cached_rows": feature_meta.get("cached_rows_loaded", np.nan),
        "feature_store_generated_rows": feature_meta.get("generated_rows_added", np.nan),
        "shared_feature_matrix_cache_hit": bool(feature_meta.get("shared_feature_matrix_cache_hit", False)),
        "shared_feature_matrix_cache_file": str(feature_meta.get("shared_feature_matrix_cache_file", "")),
        "representation_key": feature_meta.get("representation_key", ""),
    }

    stage_index = 4
    model_bundle = conventional_models(
        args,
        X_train,
        split["y_train"],
        split["smiles_train"],
        split_strategy_for_cv=cv_strategy_for_workflows,
    )
    if bool(getattr(args, "run_tabpfn", False)):
        if TabPFNRegressor is None:
            print(
                f"[info] {dataset_id} TabPFNRegressor unavailable: install `tabpfn` to enable this model.",
                flush=True,
            )
        elif int(X_train.shape[0]) > int(getattr(args, "tabpfn_max_train_rows", 1000)):
            print(
                f"[skip] {dataset_id} TabPFNRegressor: train rows={int(X_train.shape[0])} exceeds "
                f"--tabpfn-max-train-rows={int(getattr(args, 'tabpfn_max_train_rows', 1000))}.",
                flush=True,
            )
            if "TabPFNRegressor" not in completed_model_names:
                tabpfn_skip_row = {
                    **base_meta,
                    "model": "TabPFNRegressor",
                    "workflow": "conventional",
                    "status": "skipped_tabpfn_max_train_rows",
                    "tabpfn_max_train_rows": int(getattr(args, "tabpfn_max_train_rows", 1000)),
                    "tabpfn_train_rows": int(X_train.shape[0]),
                    "elapsed_seconds": round(time.time() - start, 3),
                }
                metrics_rows.append(tabpfn_skip_row)
                completed_model_names.add("TabPFNRegressor")
                persist_partial("conventional:TabPFNRegressor:skipped_max_train_rows")
    elasticnet_cv_meta = model_bundle.pop("_elasticnet_cv_meta", {})
    model_items = list(model_bundle.items())
    maplight_catboost_label = maplight_catboost_model_label(args)
    for model_name, estimator in model_items:
        if str(model_name) in completed_model_names:
            stage_message(stage_index, f"conventional model {model_name} (cached)")
            stage_index += 1
            continue
        if (
            model_name == "ElasticNetCV"
            and bool(args.skip_elasticnetcv_if_selector_timeout)
            and bool(selector_meta.get("selector_timed_out", False))
        ):
            stage_message(stage_index, f"conventional model {model_name} (skipped: selector timeout)")
            skip_reason = (
                f"Skipped {model_name}: selector ElasticNetCV exceeded timeout "
                f"({float(args.selector_elasticnet_timeout_seconds):,.0f}s) and fell back to RandomForest importance."
            )
            print(f"[skip] {dataset_id} {model_name}: {skip_reason}", flush=True)
            row = {
                "model": model_name,
                "workflow": "conventional",
                "error": skip_reason,
                "status": "skipped_selector_timeout",
            }
            row.update(elasticnet_cv_meta)
            row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
            metrics_rows.append(row)
            completed_model_names.add(str(model_name))
            persist_partial(f"conventional:{model_name}")
            stage_index += 1
            continue
        stage_message(stage_index, f"conventional model {model_name}")
        if model_name == maplight_catboost_label:
            if maplight_parity_mode:
                if maplight_direct_X_train.empty:
                    print(f"[skip] {dataset_id} {model_name}: strict parity MapLight features are unavailable")
                    stage_index += 1
                    continue
                model_X_train = maplight_direct_X_train
                model_X_test = maplight_direct_X_test
            else:
                if not maplight_feature_cols:
                    print(f"[skip] {dataset_id} {model_name}: MapLight classic features were not found in the feature matrix")
                    stage_index += 1
                    continue
                model_X_train = split["X_train"].loc[:, maplight_feature_cols].reset_index(drop=True)
                model_X_test = split["X_test"].loc[:, maplight_feature_cols].reset_index(drop=True)
        else:
            model_X_train = X_train
            model_X_test = X_test
        if model_name == "TabPFNRegressor" and str(TABPFN_REGRESSOR_SOURCE).strip().lower() == "tabpfn_client":
            estimated_tokens = estimate_tabpfn_tokens(
                rows=int(model_X_train.shape[0]),
                columns=int(model_X_train.shape[1]),
                estimators=tabpfn_estimators_per_dataset_run(args),
            )
            if estimated_tokens > TABPFN_DAILY_TOKEN_BUDGET:
                skip_reason = (
                    "Preflight budget guard skipped TabPFNRegressor: estimated usage "
                    f"{estimated_tokens:,} tokens exceeds configured API budget guardrail "
                    f"{TABPFN_DAILY_TOKEN_BUDGET:,}. This is an estimate, not an API-denied request. "
                    f"{TABPFN_DAILY_RESET_NOTE}"
                )
                print(f"[skip] {dataset_id} TabPFNRegressor: {skip_reason}", flush=True)
                row = {
                    "model": model_name,
                    "workflow": "conventional",
                    "error": skip_reason,
                    "status": "skipped_tabpfn_token_budget_estimate",
                    "tabpfn_estimated_tokens": int(estimated_tokens),
                    "tabpfn_daily_token_budget": int(TABPFN_DAILY_TOKEN_BUDGET),
                    "tabpfn_estimated_estimators": int(tabpfn_estimators_per_dataset_run(args)),
                }
                row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
                metrics_rows.append(row)
                completed_model_names.add(str(model_name))
                persist_partial(f"conventional:{model_name}")
                stage_index += 1
                continue
        try:
            if model_name == maplight_catboost_label and maplight_parity_mode:
                row, pred_train, pred_test = evaluate_maplight_seeded_catboost(
                    model_name=model_name,
                    workflow_name="conventional",
                    X_train=model_X_train,
                    X_test=model_X_test,
                    y_train=split["y_train"],
                    y_test=split["y_test"],
                    seed_values=maplight_seed_values,
                    feature_source="direct_maplight_classic_from_smiles_no_dedup",
                    primary_metric="mae",
                )
            else:
                row, pred_train, pred_test = evaluate_model(
                    model_name,
                    estimator,
                    model_X_train,
                    model_X_test,
                    split["y_train"],
                    split["y_test"],
                    split["smiles_train"],
                    args,
                    split_strategy_for_cv=cv_strategy_for_workflows,
                )
        except Exception as exc:
            error_text = str(exc)
            if model_name == "TabPFNRegressor" and is_tabpfn_token_limit_error(error_text):
                error_text = format_tabpfn_token_limit_notice(error_text)
                print(f"[warn] {dataset_id} TabPFNRegressor token budget event: {error_text}", flush=True)
            row = {"model": model_name, "workflow": "conventional", "error": error_text}
            pred_train, pred_test = np.array([]), np.array([])
        if model_name == "ElasticNetCV":
            row.update(elasticnet_cv_meta)
        row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
        metrics_rows.append(row)
        completed_model_names.add(str(model_name))
        if len(pred_train):
            prediction_tables.extend(
                [
                    prediction_frame(dataset_id, model_name, "conventional", "train", split["smiles_train"], split["y_train"], pred_train),
                    prediction_frame(dataset_id, model_name, "conventional", "test", split["smiles_test"], split["y_test"], pred_test),
                ]
            )
            prediction_payloads[model_name] = prediction_payload(
                workflow="Conventional ML",
                train_smiles=split["smiles_train"],
                test_smiles=split["smiles_test"],
                y_train=split["y_train"],
                y_test=split["y_test"],
                pred_train=pred_train,
                pred_test=pred_test,
            )
        persist_partial(f"conventional:{model_name}")
        stage_index += 1

    ga_specs = ga_model_specs(args)
    for model_name in requested_ga_models:
        ga_label = f"{model_name} GA"
        if ga_label in completed_model_names:
            stage_message(stage_index, f"GA tuning {model_name} (cached)")
            stage_index += 1
            continue
        if model_name not in ga_specs:
            print(f"[skip] {dataset_id} GA {model_name}: model unavailable")
            stage_index += 1
            continue
        stage_message(stage_index, f"GA tuning {model_name}")
        try:
            row, history, pred_train, pred_test = run_simple_ga(
                model_name,
                *ga_specs[model_name],
                X_train,
                X_test,
                split["y_train"],
                split["y_test"],
                split["smiles_train"],
                args,
                split_strategy_for_cv=cv_strategy_for_workflows,
            )
        except Exception as exc:
            row = {"model": ga_label, "workflow": "ga_tuned", "error": str(exc)}
            history, pred_train, pred_test = pd.DataFrame(), np.array([]), np.array([])
        row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
        metrics_rows.append(row)
        completed_model_names.add(ga_label)
        if not history.empty:
            history.insert(0, "dataset", dataset_id)
            ga_history_tables.append(history)
        if len(pred_train):
            prediction_tables.extend(
                [
                    prediction_frame(dataset_id, ga_label, "ga_tuned", "train", split["smiles_train"], split["y_train"], pred_train),
                    prediction_frame(dataset_id, ga_label, "ga_tuned", "test", split["smiles_test"], split["y_test"], pred_test),
                ]
            )
            prediction_payloads[ga_label] = prediction_payload(
                workflow="Tuned conventional ML",
                train_smiles=split["smiles_train"],
                test_smiles=split["smiles_test"],
                y_train=split["y_train"],
                y_test=split["y_test"],
                pred_train=pred_train,
                pred_test=pred_test,
            )
        persist_partial(f"ga:{model_name}")
        stage_index += 1

    deep_model_specs: list[tuple[str, str]] = []
    if bool(args.run_chemml_pytorch):
        deep_model_specs.append(("ChemML MLP (PyTorch)", "pytorch"))
    if bool(args.run_chemml_tensorflow):
        deep_model_specs.append(("ChemML MLP (TensorFlow)", "tensorflow"))
    for deep_label, deep_engine in deep_model_specs:
        if deep_label in completed_model_names:
            stage_message(stage_index, f"deep model {deep_label} (cached)")
            stage_index += 1
            continue
        stage_message(stage_index, f"deep model {deep_label}")
        try:
            row, pred_train, pred_test = train_chemml_model(
                label=deep_label,
                engine_name=deep_engine,
                X_train=X_train,
                X_test=X_test,
                y_train=split["y_train"],
                y_test=split["y_test"],
                smiles_train=split["smiles_train"],
                args=args,
                split_strategy_for_cv=cv_strategy_for_workflows,
            )
        except Exception as exc:
            row = {"model": deep_label, "workflow": "ChemML deep learning", "error": str(exc)}
            pred_train, pred_test = np.array([]), np.array([])
        row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
        metrics_rows.append(row)
        completed_model_names.add(str(deep_label))
        if len(pred_train):
            prediction_tables.extend(
                [
                    prediction_frame(dataset_id, deep_label, "ChemML deep learning", "train", split["smiles_train"], split["y_train"], pred_train),
                    prediction_frame(dataset_id, deep_label, "ChemML deep learning", "test", split["smiles_test"], split["y_test"], pred_test),
                ]
            )
            prediction_payloads[deep_label] = prediction_payload(
                workflow="ChemML deep learning",
                train_smiles=split["smiles_train"],
                test_smiles=split["smiles_test"],
                y_train=split["y_train"],
                y_test=split["y_test"],
                pred_train=pred_train,
                pred_test=pred_test,
            )
        persist_partial(f"deep:{deep_label}")
        stage_index += 1

    if bool(args.run_chemprop_mpnn):
        for chemprop_variant in chemprop_variant_specs(args):
            chemprop_label = str(chemprop_variant.get("label", f"Chemprop v2 (ensemble={int(args.chemprop_ensemble_size)})"))
            chemprop_workflow = str(chemprop_variant.get("workflow", "Chemprop v2"))
            chemprop_architecture = str(chemprop_variant.get("architecture_key", "dmpnn"))
            chemprop_variant_tag = str(chemprop_variant.get("variant_tag", "base"))
            chemprop_train_args = [str(item).strip() for item in list(chemprop_variant.get("train_args", [])) if str(item).strip()]
            chemprop_featurizers = [str(item).strip() for item in list(chemprop_variant.get("featurizers", [])) if str(item).strip()]
            chemprop_use_selected_descriptors = bool(chemprop_variant.get("use_selected_descriptors", False))
            if chemprop_label in completed_model_names:
                stage_message(stage_index, f"deep model {chemprop_label} (cached)")
                stage_index += 1
                continue
            stage_message(stage_index, f"deep model {chemprop_label}")
            try:
                row, pred_train, pred_test = train_chemprop_model(
                    label=chemprop_label,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=split["y_train"],
                    y_test=split["y_test"],
                    smiles_train=split["smiles_train"],
                    smiles_test=split["smiles_test"],
                    args=args,
                    dataset_dir=dataset_dir,
                    split_strategy_for_cv=cv_strategy_for_workflows,
                    featurizers=chemprop_featurizers,
                    variant_tag=chemprop_variant_tag,
                    architecture_key=chemprop_architecture,
                    workflow_label=chemprop_workflow,
                    extra_train_args=chemprop_train_args,
                    use_selected_descriptors=chemprop_use_selected_descriptors,
                )
            except Exception as exc:
                row = {
                    "model": chemprop_label,
                    "workflow": chemprop_workflow,
                    "error": str(exc),
                    "chemprop_architecture": chemprop_architecture,
                    "chemprop_variant_tag": chemprop_variant_tag,
                    "chemprop_train_extra_args": " ".join(chemprop_train_args),
                    "chemprop_molecule_featurizers": ",".join(chemprop_featurizers),
                    "chemprop_uses_selected_descriptors": bool(chemprop_use_selected_descriptors),
                    "chemprop_selected_descriptor_count": int(X_train.shape[1]) if chemprop_use_selected_descriptors else 0,
                }
                pred_train, pred_test = np.array([]), np.array([])
            row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
            metrics_rows.append(row)
            completed_model_names.add(str(chemprop_label))
            if len(pred_train):
                prediction_tables.extend(
                    [
                        prediction_frame(dataset_id, chemprop_label, chemprop_workflow, "train", split["smiles_train"], split["y_train"], pred_train),
                        prediction_frame(dataset_id, chemprop_label, chemprop_workflow, "test", split["smiles_test"], split["y_test"], pred_test),
                    ]
                )
                prediction_payloads[chemprop_label] = prediction_payload(
                    workflow=chemprop_workflow,
                    train_smiles=split["smiles_train"],
                    test_smiles=split["smiles_test"],
                    y_train=split["y_train"],
                    y_test=split["y_test"],
                    pred_train=pred_train,
                    pred_test=pred_test,
                )
            persist_partial(f"deep:Chemprop-v2:{chemprop_variant_tag}")
            stage_index += 1

    if bool(getattr(args, "run_unimol_v1", False)):
        unimol_label = "Uni-Mol V1"
        if unimol_label in completed_model_names:
            stage_message(stage_index, f"deep model {unimol_label} (cached)")
            stage_index += 1
        else:
            stage_message(stage_index, f"deep model {unimol_label}")
            try:
                row, pred_train, pred_test = train_unimol_v1_model(
                    label=unimol_label,
                    y_train=split["y_train"],
                    y_test=split["y_test"],
                    smiles_train=split["smiles_train"],
                    smiles_test=split["smiles_test"],
                    args=args,
                    dataset_dir=dataset_dir,
                )
            except Exception as exc:
                row = {"model": unimol_label, "workflow": "Uni-Mol", "error": str(exc)}
                pred_train, pred_test = np.array([]), np.array([])
            row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
            metrics_rows.append(row)
            completed_model_names.add(str(unimol_label))
            if len(pred_train):
                prediction_tables.extend(
                    [
                        prediction_frame(dataset_id, unimol_label, "Uni-Mol", "train", split["smiles_train"], split["y_train"], pred_train),
                        prediction_frame(dataset_id, unimol_label, "Uni-Mol", "test", split["smiles_test"], split["y_test"], pred_test),
                    ]
                )
                prediction_payloads[unimol_label] = prediction_payload(
                    workflow="Uni-Mol",
                    train_smiles=split["smiles_train"],
                    test_smiles=split["smiles_test"],
                    y_train=split["y_train"],
                    y_test=split["y_test"],
                    pred_train=pred_train,
                    pred_test=pred_test,
                )
            persist_partial("deep:Uni-Mol-V1")
            stage_index += 1

    if bool(args.run_maplight_gnn):
        maplight_label = maplight_gnn_model_label(args)
        if maplight_label in completed_model_names:
            stage_message(stage_index, f"deep model {maplight_label} (cached)")
            stage_index += 1
        else:
            stage_message(stage_index, f"deep model {maplight_label}")
            if CatBoostRegressor is None:
                row = {"model": maplight_label, "workflow": "MapLight + GNN", "error": "CatBoost is unavailable"}
                metrics_rows.append({**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)})
                completed_model_names.add(maplight_label)
            elif maplight_parity_mode and maplight_direct_X_train.empty:
                row = {"model": maplight_label, "workflow": "MapLight + GNN", "error": "MapLight strict parity features are unavailable"}
                metrics_rows.append({**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)})
                completed_model_names.add(maplight_label)
            elif (not maplight_parity_mode) and (not maplight_feature_cols):
                row = {"model": maplight_label, "workflow": "MapLight + GNN", "error": "MapLight classic features are unavailable"}
                metrics_rows.append({**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)})
                completed_model_names.add(maplight_label)
            else:
                try:
                    embedder, embedding_backend = _build_maplight_gnn_embedder(str(args.maplight_gnn_kind))
                    if str(embedding_backend) != "molfeat-store":
                        print(
                            f"[info] {dataset_id} MapLight + GNN embedding backend: {embedding_backend}",
                            flush=True,
                        )
                    train_embeddings_raw = embedder(split["smiles_train"].astype(str).tolist())
                    test_embeddings_raw = embedder(split["smiles_test"].astype(str).tolist())
                    emb_train = _embeddings_to_matrix(train_embeddings_raw)
                    emb_test = _embeddings_to_matrix(test_embeddings_raw, reference_width=emb_train.shape[1])
                    train_fill = np.nanmean(emb_train, axis=0)
                    train_fill = np.where(np.isfinite(train_fill), train_fill, 0.0)
                    emb_train = np.where(np.isfinite(emb_train), emb_train, train_fill)
                    emb_test = np.where(np.isfinite(emb_test), emb_test, train_fill)
                    if maplight_parity_mode:
                        maplight_base_train = maplight_direct_X_train.to_numpy(dtype=float)
                        maplight_base_test = maplight_direct_X_test.to_numpy(dtype=float)
                        maplight_columns = [str(col) for col in maplight_direct_X_train.columns]
                    else:
                        maplight_base_train = split["X_train"].loc[:, maplight_feature_cols].to_numpy(dtype=float)
                        maplight_base_test = split["X_test"].loc[:, maplight_feature_cols].to_numpy(dtype=float)
                        maplight_columns = list(maplight_feature_cols)
                    gnn_feature_names = [f"gin_emb_{idx:03d}" for idx in range(int(emb_train.shape[1]))]
                    fusion_train = pd.DataFrame(
                        np.concatenate([maplight_base_train, emb_train], axis=1),
                        columns=maplight_columns + gnn_feature_names,
                    )
                    fusion_test = pd.DataFrame(
                        np.concatenate([maplight_base_test, emb_test], axis=1),
                        columns=maplight_columns + gnn_feature_names,
                    )
                    if maplight_parity_mode:
                        row, pred_train, pred_test = evaluate_maplight_seeded_catboost(
                            model_name=maplight_label,
                            workflow_name="MapLight + GNN",
                            X_train=fusion_train,
                            X_test=fusion_test,
                            y_train=split["y_train"],
                            y_test=split["y_test"],
                            seed_values=maplight_seed_values,
                            feature_source="direct_maplight_classic_plus_gnn_no_dedup",
                            primary_metric="mae",
                        )
                    else:
                        maplight_estimator = CatBoostRegressor(
                            iterations=400,
                            depth=6,
                            learning_rate=0.05,
                            loss_function="RMSE",
                            random_seed=args.random_seed,
                            verbose=False,
                        )
                        row, pred_train, pred_test = evaluate_model(
                            maplight_label,
                            maplight_estimator,
                            fusion_train,
                            fusion_test,
                            split["y_train"],
                            split["y_test"],
                            split["smiles_train"],
                            args,
                            split_strategy_for_cv=cv_strategy_for_workflows,
                        )
                    row["workflow"] = "MapLight + GNN"
                    row["maplight_embedding_backend"] = str(embedding_backend)
                    row["maplight_embedding_kind"] = str(args.maplight_gnn_kind)
                except Exception as exc:
                    row = {"model": maplight_label, "workflow": "MapLight + GNN", "error": str(exc)}
                    pred_train, pred_test = np.array([]), np.array([])
                row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
                metrics_rows.append(row)
                completed_model_names.add(maplight_label)
                if len(pred_train):
                    prediction_tables.extend(
                        [
                            prediction_frame(dataset_id, maplight_label, "MapLight + GNN", "train", split["smiles_train"], split["y_train"], pred_train),
                            prediction_frame(dataset_id, maplight_label, "MapLight + GNN", "test", split["smiles_test"], split["y_test"], pred_test),
                        ]
                    )
                    prediction_payloads[maplight_label] = prediction_payload(
                        workflow="MapLight + GNN",
                        train_smiles=split["smiles_train"],
                        test_smiles=split["smiles_test"],
                        y_train=split["y_train"],
                        y_test=split["y_test"],
                        pred_train=pred_train,
                        pred_test=pred_test,
                    )
            persist_partial("deep:MapLight+GNN")
            stage_index += 1

    cfa_label = "CFA (Combinatorial Fusion)"
    if bool(getattr(args, "run_cfa", False)):
        if cfa_label in completed_model_names:
            stage_message(stage_index, "CFA combinatorial fusion (cached)")
            stage_index += 1
        else:
            stage_message(stage_index, "CFA combinatorial fusion")
            allowed_workflows = cfa_source_workflow_filter(args)
            cfa_base_payloads = {
                name: payload
                for name, payload in prediction_payloads.items()
                if allowed_workflows is None
                or _normalize_workflow_label(payload.get("workflow", "")) in allowed_workflows
            }
            min_required_models = int(max(1, int(getattr(args, "cfa_min_models", 2))))
            if len(cfa_base_payloads) < min_required_models:
                available_workflows = sorted(
                    {
                        str(payload.get("workflow", "")).strip()
                        for payload in prediction_payloads.values()
                        if str(payload.get("workflow", "")).strip()
                    }
                )
                workflow_filter_text = (
                    "all"
                    if allowed_workflows is None
                    else ", ".join(sorted(allowed_workflows))
                )
                skip_reason = (
                    "CFA skipped: not enough successfully predicted models after source-workflow filtering "
                    f"({len(cfa_base_payloads)} available; requires >= {min_required_models}; "
                    f"filter={workflow_filter_text}; available_workflows={available_workflows})."
                )
                print(f"[skip] {dataset_id} {cfa_label}: {skip_reason}", flush=True)
                row = {
                    "model": cfa_label,
                    "workflow": "cfa",
                    "status": "skipped_insufficient_models",
                    "cfa_skip_reason": skip_reason,
                }
                pred_train, pred_test = np.array([]), np.array([])
            else:
                try:
                    cfa_result = run_cfa_regression_fusion(
                        train_prediction_map={name: payload["train"] for name, payload in cfa_base_payloads.items()},
                        test_prediction_map={name: payload["test"] for name, payload in cfa_base_payloads.items()},
                        y_train=split["y_train"],
                        min_models=min_required_models,
                        max_models=int(max(1, int(getattr(args, "cfa_max_models", 4)))),
                        optimize_metric=str(getattr(args, "cfa_optimize_metric", "mae")),
                        include_rank_combinations=bool(getattr(args, "cfa_include_rank_combinations", True)),
                        rank_prefer_when_diverse=bool(getattr(args, "cfa_rank_prefer_when_diverse", True)),
                        rank_diversity_threshold=float(getattr(args, "cfa_rank_diversity_threshold", 0.15)),
                        rank_metric_discount=float(getattr(args, "cfa_rank_metric_discount", 0.98)),
                    )
                    pred_train = np.asarray(cfa_result["pred_train"], dtype=float).reshape(-1)
                    pred_test = np.asarray(cfa_result["pred_test"], dtype=float).reshape(-1)
                    primary_metric = (
                        normalize_benchmark_metric(CURRENT_DATASET_SPEC.recommended_metric, fallback="rmse")
                        if CURRENT_DATASET_SPEC is not None and CURRENT_DATASET_SPEC.recommended_metric
                        else "rmse"
                    )
                    primary_value = compute_primary_metric(primary_metric, split["y_test"], pred_test)
                    row = {
                        "model": cfa_label,
                        "workflow": "cfa",
                        "cv_folds": np.nan,
                        "cv_split_strategy": "",
                        "cv_r2": np.nan,
                        "cv_rmse": np.nan,
                        "cv_mae": np.nan,
                        "primary_metric": primary_metric,
                        "cv_primary": np.nan,
                        "primary_metric_value": float(primary_value),
                        "cfa_variant": str(cfa_result.get("variant", "")),
                        "cfa_combination_space": str(cfa_result.get("combination_space", "")),
                        "cfa_optimize_metric": str(cfa_result.get("optimize_metric", "")),
                        "cfa_train_metric": float(cfa_result.get("train_metric", np.nan)),
                        "cfa_adjusted_metric": float(cfa_result.get("adjusted_metric", np.nan)),
                        "cfa_subset_diversity_mean": float(cfa_result.get("subset_diversity_mean", np.nan)),
                        "cfa_selected_models": "; ".join([str(name) for name in cfa_result.get("selected_models", [])]),
                        "cfa_weights_json": json.dumps(cfa_result.get("weights", {}), default=float),
                        "cfa_candidate_count": int(len(cfa_result.get("candidate_table", []))),
                    }
                    row.update(regression_metrics(split["y_train"], pred_train, split["y_test"], pred_test))
                    row = add_leaderboard_reference_columns(
                        row,
                        primary_metric=primary_metric,
                        primary_metric_value=float(primary_value),
                    )
                    candidate_df = cfa_result.get("candidate_table")
                    if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
                        candidate_df.to_csv(dataset_dir / "cfa_candidate_table.csv", index=False)
                    strengths_df = cfa_result.get("base_strengths")
                    if isinstance(strengths_df, pd.DataFrame) and not strengths_df.empty:
                        strengths_df.to_csv(dataset_dir / "cfa_base_strengths.csv", index=False)
                except Exception as exc:
                    row = {"model": cfa_label, "workflow": "cfa", "error": str(exc)}
                    pred_train, pred_test = np.array([]), np.array([])
            row = {**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)}
            metrics_rows.append(row)
            completed_model_names.add(cfa_label)
            if len(pred_train):
                prediction_tables.extend(
                    [
                        prediction_frame(dataset_id, cfa_label, "cfa", "train", split["smiles_train"], split["y_train"], pred_train),
                        prediction_frame(dataset_id, cfa_label, "cfa", "test", split["smiles_test"], split["y_test"], pred_test),
                    ]
                )
                prediction_payloads[cfa_label] = prediction_payload(
                    workflow="CFA fusion",
                    train_smiles=split["smiles_train"],
                    test_smiles=split["smiles_test"],
                    y_train=split["y_train"],
                    y_test=split["y_test"],
                    pred_train=pred_train,
                    pred_test=pred_test,
                )
            persist_partial("cfa")
            stage_index += 1

    if bool(args.run_ensemble):
        existing_ensemble_models = [
            name
            for name in completed_model_names
            if str(name).startswith("Ensemble (") or str(name).strip().lower() == "ensemble"
        ]
        if existing_ensemble_models:
            stage_message(stage_index, "ensemble (cached)")
            stage_index += 1
        else:
            stage_message(stage_index, "ensemble")
            if len(prediction_payloads) < 2:
                print(f"[skip] {dataset_id} ensemble: fewer than two models produced predictions")
            else:
                try:
                    (
                        ensemble_results,
                        weight_df,
                        ensemble_train_pred,
                        ensemble_test_pred,
                        ensemble_members,
                        member_filter_notes,
                        _meta_model,
                    ) = build_ensemble_result(
                        payloads=prediction_payloads,
                        method=str(args.ensemble_method),
                        stacking_cv_folds=int(args.ensemble_stacking_cv_folds),
                        random_seed=int(args.random_seed),
                        drop_highly_correlated_members=bool(args.ensemble_drop_highly_correlated_members),
                        max_train_prediction_correlation=float(args.ensemble_max_train_correlation),
                        exclude_negative_test_r2_members=bool(args.ensemble_exclude_negative_test_r2_members),
                    )
                    final_ensemble_row = ensemble_results.loc[ensemble_results["workflow"].astype(str) == "ensemble"].iloc[0].to_dict()
                    final_ensemble_row["ensemble_method"] = str(args.ensemble_method)
                    final_ensemble_row["ensemble_member_count"] = int(len(ensemble_members))
                    final_ensemble_row["ensemble_members"] = ", ".join(ensemble_members)
                    final_ensemble_row["ensemble_member_filter_notes"] = " | ".join(member_filter_notes)
                    metrics_rows.append({**base_meta, **final_ensemble_row, "elapsed_seconds": round(time.time() - start, 3)})
                    completed_model_names.add(str(final_ensemble_row.get("model", "Ensemble")))
                    prediction_tables.extend(
                        [
                            prediction_frame(
                                dataset_id,
                                str(final_ensemble_row["model"]),
                                "ensemble",
                                "train",
                                split["smiles_train"],
                                split["y_train"],
                                ensemble_train_pred,
                            ),
                            prediction_frame(
                                dataset_id,
                                str(final_ensemble_row["model"]),
                                "ensemble",
                                "test",
                                split["smiles_test"],
                                split["y_test"],
                                ensemble_test_pred,
                            ),
                        ]
                    )
                    weight_df.to_csv(dataset_dir / "ensemble_weights.csv", index=False)
                    ensemble_results.to_csv(dataset_dir / "ensemble_results.csv", index=False)
                except Exception as exc:
                    metrics_rows.append(
                        {
                            **base_meta,
                            "model": "Ensemble",
                            "workflow": "ensemble",
                            "error": str(exc),
                            "elapsed_seconds": round(time.time() - start, 3),
                        }
                    )
                    completed_model_names.add("Ensemble")
            persist_partial("ensemble")
            stage_index += 1
        if existing_ensemble_models:
            persist_partial("ensemble:cached")

    final_metrics_rows = current_annotated_metrics_rows()
    if final_metrics_rows:
        pd.DataFrame(final_metrics_rows).to_csv(metrics_path, index=False)
    if prediction_tables:
        pd.concat(prediction_tables, ignore_index=True).to_csv(predictions_path, index=False)
    if ga_history_tables:
        pd.concat(ga_history_tables, ignore_index=True).to_csv(ga_history_path, index=False)
    _set_active_stage_status("completed")
    _close_active_stage(default_status="completed")
    _write_stage_runtime_outputs()
    elapsed_seconds = time.time() - start
    write_dataset_status(
        dataset_dir,
        {
            "status": "completed",
            "dataset": dataset_id,
            "source": spec.source,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "n_metrics_rows": len(final_metrics_rows),
            "n_stage_records": len(stage_records),
        },
    )
    return DatasetRunResult(final_metrics_rows, prediction_tables, ga_history_tables, "completed", elapsed_seconds)


def load_summary_metrics_for_output_dir(output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "summary_metrics.csv"
    if summary_path.exists():
        try:
            return pd.read_csv(summary_path)
        except Exception:
            pass
    return load_run_metrics_dataframe(output_dir)


def best_rows_by_dataset(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or "dataset" not in metrics_df.columns:
        return pd.DataFrame()
    working = metrics_df.copy()
    working = working.loc[~error_mask(working)].copy()
    if working.empty:
        return pd.DataFrame()
    if "primary_metric_value" not in working.columns:
        return pd.DataFrame()
    working["primary_metric_value"] = pd.to_numeric(working["primary_metric_value"], errors="coerce")
    working = working.loc[working["primary_metric_value"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    selected_rows: list[pd.Series] = []
    for dataset_name, dataset_df in working.groupby("dataset", sort=False):
        if dataset_df.empty:
            continue
        primary_metric = normalize_benchmark_metric(
            str(dataset_df.get("primary_metric", pd.Series(["rmse"])).dropna().iloc[0]),
            fallback="rmse",
        )
        lower_is_better = metric_lower_is_better(primary_metric)
        if lower_is_better is None:
            lower_is_better = True
        values = pd.to_numeric(dataset_df["primary_metric_value"], errors="coerce")
        if values.notna().sum() == 0:
            continue
        best_idx = values.idxmin() if lower_is_better else values.idxmax()
        selected_rows.append(dataset_df.loc[best_idx])
    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def extract_split_signature_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "split_strategy", "split_train_hash", "split_test_hash", "n_train", "n_test"}
    if metrics_df.empty or not required.issubset(set(metrics_df.columns)):
        return pd.DataFrame(columns=["dataset", "split_strategy", "split_train_hash", "split_test_hash", "n_train", "n_test"])
    subset = metrics_df.loc[:, list(required)].copy()
    subset["dataset"] = subset["dataset"].astype(str).str.strip()
    subset = subset.loc[subset["dataset"].ne("")].copy()
    if subset.empty:
        return pd.DataFrame(columns=["dataset", "split_strategy", "split_train_hash", "split_test_hash", "n_train", "n_test"])

    def _first_non_empty(series: pd.Series) -> Any:
        for value in series:
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none"}:
                return value
        return series.iloc[0] if len(series) else ""

    grouped = (
        subset.groupby("dataset", as_index=False)
        .agg(
            {
                "split_strategy": _first_non_empty,
                "split_train_hash": _first_non_empty,
                "split_test_hash": _first_non_empty,
                "n_train": _first_non_empty,
                "n_test": _first_non_empty,
            }
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    return grouped


def classify_error_transition(previous_error: str, current_error: str) -> str:
    prev = str(previous_error or "").strip()
    curr = str(current_error or "").strip()
    prev_ok = (not prev) or prev.lower() in {"nan", "none"}
    curr_ok = (not curr) or curr.lower() in {"nan", "none"}
    if prev_ok and curr_ok:
        return "ok_both"
    if prev_ok and not curr_ok:
        return "new_error"
    if not prev_ok and curr_ok:
        return "resolved"
    if prev == curr:
        return "persistent_same"
    return "changed_error"


def config_diff_table(current_config: dict[str, Any], reference_config: dict[str, Any]) -> pd.DataFrame:
    all_keys = sorted(set(current_config.keys()) | set(reference_config.keys()))
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        current_value = current_config.get(key, "")
        reference_value = reference_config.get(key, "")
        current_text = json.dumps(current_value, sort_keys=True, default=str) if isinstance(current_value, (dict, list)) else str(current_value)
        reference_text = json.dumps(reference_value, sort_keys=True, default=str) if isinstance(reference_value, (dict, list)) else str(reference_value)
        rows.append(
            {
                "config_key": key,
                "current_value": current_text,
                "reference_value": reference_text,
                "different": bool(current_text != reference_text),
            }
        )
    return pd.DataFrame(rows)


def error_diff_table(current_metrics: pd.DataFrame, reference_metrics: pd.DataFrame) -> pd.DataFrame:
    def _ensure_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in ["dataset", "model", "workflow", "error"]:
            if column not in out.columns:
                out[column] = ""
        return out

    current_frame = _ensure_error_columns(current_metrics)
    reference_frame = _ensure_error_columns(reference_metrics)
    current_errors = (
        current_frame.loc[:, ["dataset", "model", "workflow", "error"]]
        .copy()
        .rename(columns={"workflow": "current_workflow", "error": "current_error"})
    )
    reference_errors = (
        reference_frame.loc[:, ["dataset", "model", "workflow", "error"]]
        .copy()
        .rename(columns={"workflow": "reference_workflow", "error": "reference_error"})
    )
    merged = current_errors.merge(
        reference_errors,
        on=["dataset", "model"],
        how="outer",
        sort=True,
    )
    merged["current_error"] = merged["current_error"].fillna("").astype(str)
    merged["reference_error"] = merged["reference_error"].fillna("").astype(str)
    merged["transition"] = merged.apply(
        lambda row: classify_error_transition(row.get("reference_error", ""), row.get("current_error", "")),
        axis=1,
    )
    return merged.sort_values(["transition", "dataset", "model"]).reset_index(drop=True)


def leaderboard_comparability_table(current_metrics: pd.DataFrame, reference_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    current_best = best_rows_by_dataset(current_metrics)
    if current_best.empty:
        return pd.DataFrame()
    current_cols = {
        "dataset": "dataset",
        "model": "current_best_model",
        "workflow": "current_best_workflow",
        "primary_metric": "primary_metric",
        "primary_metric_value": "current_primary_metric_value",
        "leaderboard_metric_name": "leaderboard_metric_name",
        "leaderboard_metric_reference": "leaderboard_metric_reference",
        "leaderboard_estimated_rank_vs_top10": "current_estimated_rank_vs_top10",
        "leaderboard_estimated_in_top10": "current_estimated_in_top10",
        "leaderboard_gap_to_top10_cutoff": "current_gap_to_top10_cutoff",
        "leaderboard_top10_cutoff_value": "leaderboard_top10_cutoff_value",
        "leaderboard_url": "leaderboard_url",
    }
    for key in list(current_cols):
        if key not in current_best.columns:
            current_best[key] = np.nan if key != "dataset" else ""
    output = current_best.loc[:, list(current_cols.keys())].rename(columns=current_cols)
    output["leaderboard_metric_name"] = output["leaderboard_metric_name"].fillna("").astype(str)
    output["primary_metric"] = output["primary_metric"].fillna("").astype(str)
    output["leaderboard_metric_reference"] = pd.to_numeric(output["leaderboard_metric_reference"], errors="coerce")
    output["leaderboard_comparable"] = output.apply(
        lambda row: bool(
            normalize_benchmark_metric(row.get("leaderboard_metric_name"), fallback="")
            == normalize_benchmark_metric(row.get("primary_metric"), fallback="")
            and pd.notna(row.get("leaderboard_metric_reference"))
        ),
        axis=1,
    )

    if reference_metrics is not None and not reference_metrics.empty:
        reference_best = best_rows_by_dataset(reference_metrics)
        if not reference_best.empty:
            if "primary_metric_value" not in reference_best.columns:
                reference_best["primary_metric_value"] = np.nan
            reference_view = reference_best.loc[:, ["dataset", "model", "primary_metric_value"]].rename(
                columns={"model": "reference_best_model", "primary_metric_value": "reference_primary_metric_value"}
            )
            output = output.merge(reference_view, on="dataset", how="left")
            output["current_minus_reference"] = (
                pd.to_numeric(output["current_primary_metric_value"], errors="coerce")
                - pd.to_numeric(output["reference_primary_metric_value"], errors="coerce")
            )
    return output.sort_values("dataset").reset_index(drop=True)


def write_model_value_report(output_dir: Path, metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty or "model" not in metrics_df.columns:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "model_value_report.csv", index=False)
        empty.to_csv(output_dir / "model_zero_value_candidates.csv", index=False)
        return empty, empty

    working = metrics_df.copy()
    if "dataset" not in working.columns:
        working["dataset"] = ""
    if "workflow" not in working.columns:
        working["workflow"] = ""
    if "primary_metric_value" not in working.columns:
        working["primary_metric_value"] = np.nan
    working["primary_metric_value"] = pd.to_numeric(working["primary_metric_value"], errors="coerce")
    ok_mask = ~error_mask(working)

    top_counter: Counter[str] = Counter()
    best_rows = best_rows_by_dataset(working)
    if not best_rows.empty and "model" in best_rows.columns:
        top_counter.update(best_rows["model"].astype(str).tolist())

    ensemble_weights_rows: list[dict[str, Any]] = []
    for weights_path in sorted(output_dir.glob("*/ensemble_weights.csv")):
        dataset_name = weights_path.parent.name
        try:
            weight_df = pd.read_csv(weights_path)
        except Exception:
            continue
        if weight_df.empty or "Model" not in weight_df.columns or "Weight" not in weight_df.columns:
            continue
        for _idx, row in weight_df.iterrows():
            model_name = str(row.get("Model", "")).strip()
            if not model_name:
                continue
            weight_value = float(pd.to_numeric(row.get("Weight"), errors="coerce"))
            if not math.isfinite(weight_value):
                continue
            ensemble_weights_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "weight": weight_value,
                    "abs_weight": abs(weight_value),
                }
            )
    ensemble_weights_df = pd.DataFrame(ensemble_weights_rows)

    records: list[dict[str, Any]] = []
    for model_name, model_df in working.groupby("model", sort=False):
        model_df = model_df.copy()
        model_ok = model_df.loc[ok_mask.reindex(model_df.index, fill_value=False)].copy()
        attempted_datasets = int(model_df["dataset"].astype(str).str.strip().nunique())
        successful_datasets = int(model_ok["dataset"].astype(str).str.strip().nunique())
        workflows = [str(item).strip() for item in model_df["workflow"].dropna().tolist() if str(item).strip()]
        workflow = Counter(workflows).most_common(1)[0][0] if workflows else ""
        ensemble_subset = (
            ensemble_weights_df.loc[ensemble_weights_df["model"].astype(str).str.strip().eq(str(model_name).strip())].copy()
            if not ensemble_weights_df.empty
            else pd.DataFrame(columns=["dataset", "model", "weight", "abs_weight"])
        )
        ensemble_member_count = int(ensemble_subset["dataset"].nunique()) if not ensemble_subset.empty else 0
        ensemble_nonzero_count = int((ensemble_subset["abs_weight"] > 1e-12).sum()) if not ensemble_subset.empty else 0
        ensemble_mean_abs = float(ensemble_subset["abs_weight"].mean()) if not ensemble_subset.empty else 0.0
        top_count = int(top_counter.get(str(model_name), 0))
        records.append(
            {
                "model": str(model_name),
                "workflow": workflow,
                "datasets_attempted": attempted_datasets,
                "datasets_successful": successful_datasets,
                "top1_count": top_count,
                "ensemble_member_count": ensemble_member_count,
                "ensemble_nonzero_weight_count": ensemble_nonzero_count,
                "ensemble_mean_abs_weight": ensemble_mean_abs,
                "never_top_or_ensemble": bool(
                    successful_datasets > 0
                    and top_count == 0
                    and ensemble_nonzero_count == 0
                    and not is_ensemble_result_row(model_name, workflow)
                ),
            }
        )
    report_df = pd.DataFrame(records).sort_values(
        ["never_top_or_ensemble", "top1_count", "ensemble_nonzero_weight_count", "datasets_successful", "model"],
        ascending=[False, True, True, False, True],
    ).reset_index(drop=True)
    zero_value_df = report_df.loc[report_df["never_top_or_ensemble"]].copy()
    report_df.to_csv(output_dir / "model_value_report.csv", index=False)
    zero_value_df.to_csv(output_dir / "model_zero_value_candidates.csv", index=False)
    return report_df, zero_value_df


def collect_step_runtime_summary(output_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for runtime_path in sorted(output_dir.glob("*/step_runtime.csv")):
        try:
            runtime_df = pd.read_csv(runtime_path)
        except Exception:
            continue
        if runtime_df.empty:
            continue
        if "dataset" not in runtime_df.columns:
            runtime_df.insert(0, "dataset", runtime_path.parent.name)
        rows.append(runtime_df)
    if not rows:
        return pd.DataFrame()
    summary_df = pd.concat(rows, ignore_index=True)
    summary_df.to_csv(output_dir / "step_runtime_summary.csv", index=False)
    return summary_df


def write_run_vs_run_attribution_report(
    *,
    root: Path,
    output_dir: Path,
    current_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference_dir: Path | None = None
    requested_reference = getattr(args, "compare_run_dir", None)
    if requested_reference is not None:
        candidate = Path(requested_reference)
        if candidate.exists():
            reference_dir = candidate
    if reference_dir is None:
        recent_runs = discover_recent_benchmark_runs(root, exclude_dir=output_dir)
        if recent_runs:
            reference_dir = recent_runs[0]

    current_config = load_run_config_payload(output_dir)
    summary_payload: dict[str, Any] = {
        "generated_at": local_timestamp_text(),
        "current_run_dir": str(output_dir),
        "reference_run_dir": str(reference_dir) if reference_dir is not None else "",
        "has_reference_run": bool(reference_dir is not None),
    }

    current_split = extract_split_signature_table(current_metrics)
    if reference_dir is not None:
        reference_metrics = load_summary_metrics_for_output_dir(reference_dir)
        reference_config = load_run_config_payload(reference_dir)
        reference_split = extract_split_signature_table(reference_metrics)
        split_diff = current_split.merge(
            reference_split,
            on="dataset",
            how="outer",
            suffixes=("_current", "_reference"),
        )
        for field in ["split_strategy", "split_train_hash", "split_test_hash", "n_train", "n_test"]:
            split_diff[f"{field}_matches"] = (
                split_diff[f"{field}_current"].astype(str).fillna("").str.strip()
                == split_diff[f"{field}_reference"].astype(str).fillna("").str.strip()
            )
        split_diff["all_match"] = split_diff[
            [f"{field}_matches" for field in ["split_strategy", "split_train_hash", "split_test_hash", "n_train", "n_test"]]
        ].all(axis=1)
        split_diff.to_csv(output_dir / "run_vs_run_split_match.csv", index=False)

        cfg_diff = config_diff_table(current_config, reference_config)
        cfg_diff.to_csv(output_dir / "run_vs_run_config_diff.csv", index=False)

        err_diff = error_diff_table(current_metrics, reference_metrics)
        err_diff.to_csv(output_dir / "run_vs_run_error_diff.csv", index=False)

        lb_comp = leaderboard_comparability_table(current_metrics, reference_metrics)
        lb_comp.to_csv(output_dir / "run_vs_run_leaderboard_comparability.csv", index=False)

        summary_payload.update(
            {
                "split_datasets_compared": int(len(split_diff)),
                "split_all_match_count": int(split_diff["all_match"].sum()) if not split_diff.empty else 0,
                "config_diff_count": int(cfg_diff["different"].sum()) if not cfg_diff.empty else 0,
                "new_error_count": int(err_diff["transition"].astype(str).eq("new_error").sum()) if not err_diff.empty else 0,
                "resolved_error_count": int(err_diff["transition"].astype(str).eq("resolved").sum()) if not err_diff.empty else 0,
                "leaderboard_comparable_dataset_count": int(lb_comp["leaderboard_comparable"].sum()) if not lb_comp.empty else 0,
            }
        )
    else:
        pd.DataFrame().to_csv(output_dir / "run_vs_run_split_match.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "run_vs_run_config_diff.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "run_vs_run_error_diff.csv", index=False)
        lb_comp = leaderboard_comparability_table(current_metrics, None)
        lb_comp.to_csv(output_dir / "run_vs_run_leaderboard_comparability.csv", index=False)
        summary_payload.update(
            {
                "split_datasets_compared": 0,
                "split_all_match_count": 0,
                "config_diff_count": 0,
                "new_error_count": 0,
                "resolved_error_count": 0,
                "leaderboard_comparable_dataset_count": int(lb_comp["leaderboard_comparable"].sum()) if not lb_comp.empty else 0,
                "note": "No reference run found for run-vs-run attribution.",
            }
        )

    (output_dir / "run_vs_run_attribution_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str),
        encoding="utf-8",
    )
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="CSV dataset path. Repeat to override the default notebook example set with local CSVs only.")
    parser.add_argument(
        "--refresh-leaderboards-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fetch benchmark leaderboard top10 references and write auxiliary reference files without running model training.",
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        help=(
            "Filter discovered datasets by built-in dataset name/id. Repeat to run multiple. "
            "Examples: tdc_caco2_wang, esol_delaney, polaris_adme_fang_perm_1."
        ),
    )
    parser.add_argument("--include-local-csv", action="append", help="Optional extra local CSV dataset path to add on top of the default benchmark example set.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to benchmark_results/autoqsar_benchmark_<timestamp>.")
    parser.add_argument(
        "--benchmark-profile",
        choices=["cost_optimized", "full"],
        default="cost_optimized",
        help=(
            "Runtime/performance profile. cost_optimized disables historically low-value expensive model variants by default; "
            "full restores the broader model set."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="List discovered datasets and planned configuration without fitting models.")
    parser.add_argument(
        "--revisit-completed-datasets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When resuming into an existing output directory, revisit datasets marked completed and "
            "run only missing model stages (instead of skipping the dataset entirely)."
        ),
    )

    parser.add_argument("--minimum-rows", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=0, help="Optional deterministic row cap for smoke tests. 0 uses all rows.")
    parser.add_argument(
        "--fingerprint-bits",
        type=int,
        default=1024,
        help="Bit length for hash-based fingerprint families (Morgan/ECFP/FCFP/Layered/AtomPair/TopologicalTorsion/RDKit path).",
    )
    parser.add_argument(
        "--target-transform",
        choices=["auto", "raw", "log10"],
        default="auto",
        help="Target transform policy. auto keeps benchmark/leaderboard datasets on raw target scale.",
    )
    parser.add_argument("--log10-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-strategy", choices=["target_quartiles", "random", "scaffold", "predefined"], default="target_quartiles")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--enable-persistent-feature-store", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-persistent-feature-store", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-feature-store-path", default="AUTO")
    parser.add_argument(
        "--enable-shared-feature-matrix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache fully built benchmark feature matrices (post family assembly, pre split/selection) "
            "for cross-run reuse."
        ),
    )
    parser.add_argument(
        "--reuse-shared-feature-matrix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse shared feature matrix cache entries before generating new feature matrices.",
    )
    parser.add_argument(
        "--shared-feature-matrix-cache-path",
        default="AUTO",
        help="Shared cache directory for full feature matrices. AUTO resolves to model_cache/benchmark_feature_matrix_cache.",
    )

    parser.add_argument("--selector-method", choices=["elasticnet_cv", "none"], default="elasticnet_cv")
    parser.add_argument("--selector-l1-ratio-grid", default="0.3,0.7")
    parser.add_argument("--selector-alpha-min-log10", type=float, default=-5)
    parser.add_argument("--selector-alpha-max-log10", type=float, default=-1)
    parser.add_argument("--selector-alpha-grid-size", type=int, default=12)
    parser.add_argument("--selector-cv-folds", type=int, default=3)
    parser.add_argument("--selector-max-iter", type=int, default=10000)
    parser.add_argument("--selector-coefficient-threshold", type=float, default=1e-10)
    parser.add_argument(
        "--selector-auto-rf-by-dataset-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, preemptively skip ElasticNetCV selector and use RF-importance selector if dataset-size runtime "
            "estimate exceeds --selector-auto-rf-threshold-seconds."
        ),
    )
    parser.add_argument(
        "--selector-auto-rf-threshold-seconds",
        type=float,
        default=7200.0,
        help="ElasticNetCV selector runtime threshold used by the dataset-size precheck (default: 2 hours).",
    )
    parser.add_argument(
        "--selector-auto-rf-log10-slope",
        type=float,
        default=1.225,
        help="Slope for log10(runtime_seconds) ~ intercept + slope*log10(dataset_size) used by selector auto-RF precheck.",
    )
    parser.add_argument(
        "--selector-auto-rf-log10-intercept",
        type=float,
        default=-0.658,
        help="Intercept for log10(runtime_seconds) ~ intercept + slope*log10(dataset_size) used by selector auto-RF precheck.",
    )
    parser.add_argument(
        "--selector-elasticnet-timeout-seconds",
        type=float,
        default=7200.0,
        help="Maximum wall-clock time for ElasticNetCV feature selection. If exceeded, fallback to RF importance.",
    )
    parser.add_argument(
        "--selector-rf-fallback-n-estimators",
        type=int,
        default=400,
        help="Number of trees for RF-importance fallback feature selection.",
    )
    parser.add_argument(
        "--skip-elasticnetcv-if-selector-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip the conventional ElasticNetCV training stage when selector ElasticNetCV timed out.",
    )
    parser.add_argument("--max-selected-features", type=int, default=512, help="0 means auto cap at 10%% of valid molecules; positive values override.")

    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--elasticnet-l1-ratio-grid", default="0.4,0.8")
    parser.add_argument("--elasticnet-alpha-min-log10", type=float, default=-4)
    parser.add_argument("--elasticnet-alpha-max-log10", type=float, default=0)
    parser.add_argument("--elasticnet-alpha-grid-size", type=int, default=12)
    parser.add_argument("--elasticnet-cv-folds", type=int, default=3)
    parser.add_argument("--elasticnet-max-iter", type=int, default=15000)

    parser.add_argument(
        "--ga-models",
        default="",
        help=(
            "Comma-separated GA models to tune (example: ElasticNet,CatBoost). "
            "Default is disabled (empty). "
            "Use 'auto' to enable only GA models that showed meaningful value in the most recent comparable run."
        ),
    )
    parser.add_argument(
        "--ga-auto-reference-run-dir",
        type=Path,
        default=None,
        help="Optional benchmark run directory used as evidence for --ga-models auto.",
    )
    parser.add_argument("--ga-auto-min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--ga-auto-min-dataset-wins", type=int, default=1)
    parser.add_argument("--ga-auto-min-improving-datasets", type=int, default=1)
    parser.add_argument("--ga-auto-min-reference-datasets", type=int, default=5)
    parser.add_argument("--ga-generations", type=int, default=12)
    parser.add_argument("--ga-population-size", type=int, default=16)
    parser.add_argument("--ga-elites", type=int, default=2)
    parser.add_argument("--ga-cv-folds", type=int, default=5)
    parser.add_argument("--ga-mutation-probability", type=float, default=0.35)

    parser.add_argument("--run-chemml-pytorch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemml-tensorflow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--run-tabpfn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable TabPFNRegressor (tabular foundation model) on selected descriptors. "
            "When GPU is detected, this script prefers local `tabpfn` inference before API-client fallback."
        ),
    )
    parser.add_argument(
        "--tabpfn-max-train-rows",
        type=int,
        default=1000,
        help="Maximum training rows allowed for TabPFNRegressor (CPU guardrail).",
    )
    parser.add_argument(
        "--run-cfa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run CFA (Combinatorial Fusion Analysis) as an additional model stage "
            "that fuses predictions from selected workflow sources using CFA-style performance/diversity weighting."
        ),
    )
    parser.add_argument(
        "--cfa-min-models",
        type=int,
        default=2,
        help="Minimum number of base models required to run CFA fusion.",
    )
    parser.add_argument(
        "--cfa-max-models",
        type=int,
        default=4,
        help="Maximum number of base models per CFA candidate combination.",
    )
    parser.add_argument(
        "--cfa-optimize-metric",
        choices=["mae", "rmse"],
        default="mae",
        help="Train-side metric used to select the best CFA fusion candidate.",
    )
    parser.add_argument(
        "--cfa-source-workflows",
        default="all",
        help=(
            "Comma-separated workflow labels to allow as CFA base models. "
            "Use 'all' to include every successful workflow prediction available before ensembling "
            "(e.g., Conventional ML, Tuned conventional ML, ChemML deep learning, Chemprop v2, MapLight + GNN, Uni-Mol)."
        ),
    )
    parser.add_argument(
        "--cfa-include-rank-combinations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CFA rank-combination families (AC/WCP/WCDS) in addition to score combinations.",
    )
    parser.add_argument(
        "--cfa-rank-prefer-when-diverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a small train-metric preference to rank-based candidates when model diversity exceeds threshold.",
    )
    parser.add_argument(
        "--cfa-rank-diversity-threshold",
        type=float,
        default=0.15,
        help="Diversity threshold for rank-combination preference.",
    )
    parser.add_argument(
        "--cfa-rank-metric-discount",
        type=float,
        default=0.98,
        help="Multiplicative train-metric discount applied to rank candidates when diversity threshold is met.",
    )
    parser.add_argument("--run-chemprop-mpnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemprop-dmpnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-chemprop-cmpnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-chemprop-attentivefp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-chemprop-selected-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an additional Chemprop variant that consumes the train-only selected descriptor matrix via descriptors-path.",
    )
    parser.add_argument(
        "--run-unimol-v1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run Uni-Mol V1 benchmarking stage. Default is auto: enabled only when a GPU is detected. "
            "Override with --run-unimol-v1 or --no-run-unimol-v1."
        ),
    )
    parser.add_argument("--unimol-internal-split", choices=["random", "scaffold"], default="random")
    parser.add_argument("--unimol-epochs", type=int, default=10)
    parser.add_argument("--unimol-learning-rate", type=float, default=1e-4)
    parser.add_argument("--unimol-batch-size", type=int, default=16)
    parser.add_argument("--unimol-early-stopping", type=int, default=5)
    parser.add_argument("--unimol-num-workers", type=int, default=0)
    parser.add_argument("--unimol-reuse-model-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-maplight-gnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--maplight-leaderboard-parity-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable strict MapLight leaderboard parity mode for MapLight CatBoost and MapLight + GNN: "
            "MAE objective, StandardScaler, 5-seed averaging, and direct MapLight features built from split SMILES "
            "(no feature-dedup pruning on MapLight columns)."
        ),
    )
    parser.add_argument(
        "--maplight-parity-seeds",
        default="1,2,3,4,5",
        help="Comma-separated seed list for strict MapLight parity mode.",
    )
    parser.add_argument("--chemml-hidden-layers", type=int, default=2)
    parser.add_argument("--chemml-hidden-width", type=int, default=256)
    parser.add_argument("--chemml-training-epochs", type=int, default=80)
    parser.add_argument("--chemml-batch-size", type=int, default=64)
    parser.add_argument("--chemml-learning-rate", type=float, default=1e-3)
    parser.add_argument("--chemml-use-cross-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chemml-cv-folds", type=int, default=5)
    parser.add_argument("--chemprop-epochs", type=int, default=15)
    parser.add_argument("--chemprop-batch-size", type=int, default=32)
    parser.add_argument("--chemprop-num-workers", type=int, default=0)
    parser.add_argument("--chemprop-ensemble-size", type=int, default=1)
    parser.add_argument("--chemprop-random-seed", type=int, default=42)
    parser.add_argument("--chemprop-reuse-model-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemprop-rdkit2d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--maplight-gnn-kind", default="gin_supervised_masking")
    parser.add_argument("--run-ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rebuild-ensemble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When resuming into an existing output directory, remove previously saved ensemble rows "
            "for each dataset and recompute the ensemble from currently available member predictions."
        ),
    )
    parser.add_argument(
        "--ensemble-method",
        choices=["OOF Stacking (RidgeCV)", "Simple average", "Weighted average (inverse train RMSE)"],
        default="OOF Stacking (RidgeCV)",
    )
    parser.add_argument("--ensemble-stacking-cv-folds", type=int, default=5)
    parser.add_argument("--ensemble-drop-highly-correlated-members", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ensemble-max-train-correlation", type=float, default=0.995)
    parser.add_argument("--ensemble-exclude-negative-test-r2-members", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-run-dir", type=Path, default=None, help="Optional previous run directory for automatic run-vs-run attribution diagnostics.")
    parser.add_argument("--emit-run-vs-run-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume a compatible incomplete run when possible.")
    return parser.parse_args()


def _cli_option_provided(argv_tokens: list[str], option_name: str) -> bool:
    option = f"--{str(option_name).strip().replace('_', '-')}"
    negated = f"--no-{option[2:]}"
    for token in argv_tokens:
        token_text = str(token).strip()
        if (
            token_text == option
            or token_text == negated
            or token_text.startswith(option + "=")
            or token_text.startswith(negated + "=")
        ):
            return True
    return False


def apply_benchmark_profile_defaults(args: argparse.Namespace, argv_tokens: list[str]) -> None:
    profile = str(getattr(args, "benchmark_profile", "cost_optimized")).strip().lower()
    if profile not in {"cost_optimized", "full"}:
        profile = "cost_optimized"

    profile_defaults: dict[str, Any]
    if profile == "full":
        profile_defaults = {
            "run_chemml_tensorflow": True,
            "run_tabpfn": True,
            "run_chemprop_dmpnn": True,
            "run_chemprop_cmpnn": True,
            "run_chemprop_rdkit2d": True,
            "chemprop_epochs": 40,
            "chemprop_ensemble_size": 3,
            "selector_auto_rf_by_dataset_size": False,
        }
    else:
        profile_defaults = {
            "run_chemml_tensorflow": False,
            "run_tabpfn": True,
            "run_chemprop_dmpnn": False,
            "run_chemprop_cmpnn": False,
            "run_chemprop_rdkit2d": False,
            "selector_auto_rf_by_dataset_size": True,
        }

    for arg_name, value in profile_defaults.items():
        if not _cli_option_provided(argv_tokens, arg_name):
            setattr(args, arg_name, value)


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
    apply_benchmark_profile_defaults(args, sys.argv[1:])
    gpu_available = detect_gpu_available()
    args.gpu_available = bool(gpu_available)
    if getattr(args, "run_unimol_v1", None) is None:
        args.run_unimol_v1 = bool(gpu_available)
    elif bool(args.run_unimol_v1) and not gpu_available:
        print(
            "Uni-Mol V1 was explicitly enabled without detected GPU; attempting CPU fallback.",
            flush=True,
        )
    root = Path(__file__).resolve().parents[1]
    if str(getattr(args, "persistent_feature_store_path", "AUTO")).strip().upper() == "AUTO":
        args.persistent_feature_store_path = str((root / "model_cache" / "feature_store_parquet").resolve())
    if str(getattr(args, "shared_feature_matrix_cache_path", "AUTO")).strip().upper() == "AUTO":
        args.shared_feature_matrix_cache_path = str(
            (root / "model_cache" / "benchmark_feature_matrix_cache").resolve()
        )
    if bool(getattr(args, "enable_persistent_feature_store", True)):
        Path(str(args.persistent_feature_store_path)).mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "enable_shared_feature_matrix_cache", True)):
        Path(str(args.shared_feature_matrix_cache_path)).mkdir(parents=True, exist_ok=True)
    default_maplight_pretrained_cache_dir().mkdir(parents=True, exist_ok=True)

    resolved_ga_models, ga_resolution = resolve_requested_ga_models(
        args,
        root,
        exclude_dir=Path(args.output_dir) if getattr(args, "output_dir", None) is not None else None,
    )
    args.ga_models_resolved = ",".join(resolved_ga_models)
    args.ga_resolution = ga_resolution
    if str(getattr(args, "ga_models", "")).strip().lower() == "auto":
        reference_text = str(ga_resolution.get("reference_run_dir", "")).strip()
        if resolved_ga_models:
            print(
                "GA auto-selection: enabled model(s) "
                f"{', '.join(resolved_ga_models)}"
                f"{' from reference run ' + reference_text if reference_text else ''}.",
                flush=True,
            )
        else:
            print(
                "GA auto-selection: no models met the meaningful-value threshold; GA tuning will be skipped."
                + (f" Reference run: {reference_text}" if reference_text else ""),
                flush=True,
            )
    cache_dir = root / "data" / "benchmark_leaderboards"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latest_cache_path = cache_dir / "leaderboard_top10_reference_latest.csv"
    historical_cache_df = bootstrap_leaderboard_cache_from_history(root)
    if not historical_cache_df.empty:
        if latest_cache_path.exists():
            try:
                existing_cache_df = pd.read_csv(latest_cache_path)
            except Exception:
                existing_cache_df = pd.DataFrame()
            if not existing_cache_df.empty:
                historical_cache_df = pd.concat([existing_cache_df, historical_cache_df], ignore_index=True)
                historical_cache_df = historical_cache_df.drop_duplicates(
                    subset=[
                        "benchmark_suite",
                        "benchmark_id",
                        "rank",
                        "model",
                        "leaderboard_metric_name",
                        "metric_value",
                    ],
                    keep="last",
                ).reset_index(drop=True)
        historical_cache_df.to_csv(latest_cache_path, index=False)
    output_dir = select_output_dir(root, args)

    if bool(args.refresh_leaderboards_only):
        output_dir.mkdir(parents=True, exist_ok=True)
        specs = discover_benchmark_catalog_for_leaderboards()
        reference_df = write_leaderboard_reference_artifacts(root, output_dir, specs)
        print(f"Leaderboard refresh completed. Output directory: {output_dir}")
        print(f"Top10 reference rows stored: {len(reference_df)}")
        if not reference_df.empty:
            by_suite = reference_df.groupby("benchmark_suite", dropna=False).size().to_dict()
            print(f"Rows by suite: {by_suite}")
            print(f"Auxiliary CSV: {output_dir / 'leaderboard_top10_reference.csv'}")
            print(f"Shared cache CSV: {root / 'data' / 'benchmark_leaderboards' / 'leaderboard_top10_reference_latest.csv'}")
        return 0

    if args.dataset:
        datasets = discover_local_datasets(root, args.dataset)
    else:
        datasets = discover_default_example_datasets(root)
        if args.include_local_csv:
            datasets.extend(discover_local_datasets(root, args.include_local_csv))
    if args.dataset_name:
        requested = {str(item).strip().lower() for item in args.dataset_name if str(item).strip()}
        datasets = [item for item in datasets if item.name.strip().lower() in requested]
        if not datasets:
            print(
                "No datasets matched --dataset-name. "
                "Use --dry-run without --dataset-name to list discovered dataset names."
            )
            return 1
    if not datasets:
        print("No datasets found. Provide --dataset PATH or verify benchmark data access/dependencies are available.")
        return 1

    datasets = order_datasets_smallest_first(datasets)

    tabpfn_budget_estimate = None
    if bool(getattr(args, "run_tabpfn", False)) and str(TABPFN_REGRESSOR_SOURCE).strip().lower() == "tabpfn_client":
        tabpfn_budget_estimate = estimate_tabpfn_daily_dataset_capacity(datasets, args)

    resume_plan = None
    if bool(getattr(args, "resume", False)):
        resume_plan = build_resume_execution_plan(datasets, output_dir, args)

    if args.dry_run:
        selector_threshold_size = elasticnet_selector_timeout_dataset_size_threshold(
            float(args.selector_auto_rf_threshold_seconds),
            log10_slope=float(args.selector_auto_rf_log10_slope),
            log10_intercept=float(args.selector_auto_rf_log10_intercept),
        )
        selector_threshold_size_text = (
            f"{int(round(selector_threshold_size)):,}"
            if np.isfinite(selector_threshold_size)
            else "unknown"
        )
        print(f"Planned output directory: {output_dir}")
        print(f"Benchmark profile: {args.benchmark_profile}")
        print(f"Default molecular feature families: {', '.join(DEFAULT_BENCHMARK_FEATURE_FAMILIES)}")
        print(
            "Persistent feature store: "
            f"{'on' if bool(getattr(args, 'enable_persistent_feature_store', True)) else 'off'} "
            f"(reuse={'on' if bool(getattr(args, 'reuse_persistent_feature_store', True)) else 'off'}, "
            f"path={args.persistent_feature_store_path})"
        )
        print(
            "Shared feature matrix cache: "
            f"{'on' if bool(getattr(args, 'enable_shared_feature_matrix_cache', True)) else 'off'} "
            f"(reuse={'on' if bool(getattr(args, 'reuse_shared_feature_matrix_cache', True)) else 'off'}, "
            f"path={args.shared_feature_matrix_cache_path})"
        )
        print(f"MapLight pretrained cache dir: {default_maplight_pretrained_cache_dir()}")
        print(
            "Selector auto-RF by dataset size: "
            f"{'on' if bool(args.selector_auto_rf_by_dataset_size) else 'off'} "
            f"(threshold={float(args.selector_auto_rf_threshold_seconds):,.0f}s, threshold_n~{selector_threshold_size_text})"
        )
        print(
            "GA models for this run: "
            + (", ".join(parse_comma_list(getattr(args, "ga_models_resolved", ""))) if parse_comma_list(getattr(args, "ga_models_resolved", "")) else "(none)")
        )
        print(
            "CFA stage: "
            f"{'on' if bool(getattr(args, 'run_cfa', False)) else 'off'} "
            f"(min_models={int(getattr(args, 'cfa_min_models', 2))}, max_models={int(getattr(args, 'cfa_max_models', 4))}, "
            f"opt_metric={str(getattr(args, 'cfa_optimize_metric', 'mae'))}, "
            f"sources={str(getattr(args, 'cfa_source_workflows', 'all'))}, "
            f"rank={'on' if bool(getattr(args, 'cfa_include_rank_combinations', True)) else 'off'}, "
            f"rank_pref={'on' if bool(getattr(args, 'cfa_rank_prefer_when_diverse', True)) else 'off'}, "
            f"rank_threshold={float(getattr(args, 'cfa_rank_diversity_threshold', 0.15)):.3f}, "
            f"rank_discount={float(getattr(args, 'cfa_rank_metric_discount', 0.98)):.3f})"
        )
        print(
            "MapLight parity mode: "
            f"{'strict' if bool(getattr(args, 'maplight_leaderboard_parity_mode', True)) else 'legacy'} "
            f"(seeds={','.join(str(seed) for seed in maplight_parity_seed_values(args))})"
        )
        if resume_plan is not None:
            print_resume_execution_plan(resume_plan)
        if tabpfn_budget_estimate is not None:
            print(
                "TabPFN daily-budget estimate: "
                f"{tabpfn_budget_estimate['individually_fit_count']}/{len(datasets)} dataset(s) are estimated to fit "
                f"individually within {TABPFN_DAILY_TOKEN_BUDGET:,} tokens/day. "
                f"At most {tabpfn_budget_estimate['smallest_first_count']}/{len(datasets)} dataset(s) are estimated to fit "
                f"if run smallest-first in one day. "
                f"Estimator multiplier={int(tabpfn_budget_estimate['estimators_per_dataset'])}. "
                f"{TABPFN_DAILY_RESET_NOTE}"
            )
        leaderboard_refs = leaderboard_reference_table(datasets)
        print(f"Leaderboard top10 reference rows fetched: {len(leaderboard_refs)}")
        print("Datasets:")
        for dataset in datasets:
            leaderboard_metric = ((dataset.leaderboard_summary or {}).get("metric_name") or "").strip()
            leaderboard_value = ((dataset.leaderboard_summary or {}).get("metric_value") or "").strip()
            leaderboard_note = f", leaderboard={leaderboard_metric} {leaderboard_value}".strip()
            target_transform_note = "log10" if resolve_dataset_log10_target(dataset, args) else "raw"
            print(
                f"  - {dataset.name}: smiles={dataset.smiles_column}, target={dataset.target_column}, "
                f"source={dataset.source}, split={dataset.recommended_split or args.split_strategy}"
                f", target_transform={target_transform_note}"
                f"{leaderboard_note if leaderboard_metric or leaderboard_value else ''}"
            )
        return 0

    if bool(getattr(args, "run_tabpfn", False)):
        if not ensure_tabpfn_installed(prefer_local_backend=bool(getattr(args, "gpu_available", False))):
            args.run_tabpfn = False
            print(
                "TabPFNRegressor has been automatically disabled for this run. "
                "You can still run the rest of the workflow. "
                "If you later set up TabPFN authentication/access, rerun with --run-tabpfn.",
                flush=True,
            )
        else:
            auth_ready, auth_message = prepare_tabpfn_auth(args)
            if auth_ready:
                print(auth_message, flush=True)
            else:
                args.run_tabpfn = False
                print(
                    "TabPFNRegressor has been disabled for this run because authentication/preflight did not pass.\n"
                    f"Reason: {auth_message}",
                    flush=True,
                )

    if bool(getattr(args, "run_tabpfn", False)) and str(TABPFN_REGRESSOR_SOURCE).strip().lower() == "tabpfn_client":
        tabpfn_budget_estimate = estimate_tabpfn_daily_dataset_capacity(datasets, args)
    else:
        tabpfn_budget_estimate = None
    if bool(getattr(args, "resume", False)):
        resume_plan = build_resume_execution_plan(datasets, output_dir, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["default_feature_families"] = list(DEFAULT_BENCHMARK_FEATURE_FAMILIES)
    config["config_signature"] = benchmark_config_signature(args)
    config["datasets"] = [{"name": item.name, "source": item.source, "smiles_column": item.smiles_column, "target_column": item.target_column} for item in datasets]
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    leaderboard_ref_df = write_leaderboard_reference_artifacts(root, output_dir, datasets)
    if bool(getattr(args, "run_tabpfn", False)) and tabpfn_budget_estimate is not None:
        tabpfn_budget_estimate["table"].to_csv(output_dir / "tabpfn_daily_budget_estimate.csv", index=False)

    print(f"Output directory: {output_dir}")
    selector_threshold_size = elasticnet_selector_timeout_dataset_size_threshold(
        float(args.selector_auto_rf_threshold_seconds),
        log10_slope=float(args.selector_auto_rf_log10_slope),
        log10_intercept=float(args.selector_auto_rf_log10_intercept),
    )
    selector_threshold_size_text = (
        f"{int(round(selector_threshold_size)):,}"
        if np.isfinite(selector_threshold_size)
        else "unknown"
    )
    print(f"Benchmark profile: {args.benchmark_profile}")
    print(f"Default molecular feature families: {', '.join(DEFAULT_BENCHMARK_FEATURE_FAMILIES)}")
    print(
        "Persistent feature store: "
        f"{'on' if bool(getattr(args, 'enable_persistent_feature_store', True)) else 'off'} "
        f"(reuse={'on' if bool(getattr(args, 'reuse_persistent_feature_store', True)) else 'off'}, "
        f"path={args.persistent_feature_store_path})"
    )
    print(
        "Shared feature matrix cache: "
        f"{'on' if bool(getattr(args, 'enable_shared_feature_matrix_cache', True)) else 'off'} "
        f"(reuse={'on' if bool(getattr(args, 'reuse_shared_feature_matrix_cache', True)) else 'off'}, "
        f"path={args.shared_feature_matrix_cache_path})"
    )
    print(f"MapLight pretrained cache dir: {default_maplight_pretrained_cache_dir()}")
    print(
        "Selector auto-RF by dataset size: "
        f"{'on' if bool(args.selector_auto_rf_by_dataset_size) else 'off'} "
        f"(threshold={float(args.selector_auto_rf_threshold_seconds):,.0f}s, threshold_n~{selector_threshold_size_text})"
    )
    if bool(getattr(args, "run_tabpfn", False)) and tabpfn_budget_estimate is not None:
        print(
            "TabPFN daily-budget estimate: "
            f"{tabpfn_budget_estimate['individually_fit_count']}/{len(datasets)} dataset(s) are estimated to fit "
            f"individually within {TABPFN_DAILY_TOKEN_BUDGET:,} tokens/day. "
            f"At most {tabpfn_budget_estimate['smallest_first_count']}/{len(datasets)} dataset(s) are estimated to fit "
            f"if run smallest-first in one day. "
            f"Estimator multiplier={int(tabpfn_budget_estimate['estimators_per_dataset'])}. "
            f"{TABPFN_DAILY_RESET_NOTE}"
        )
        print(f"TabPFN budget detail CSV: {output_dir / 'tabpfn_daily_budget_estimate.csv'}")
    print(
        "GA models for this run: "
        + (", ".join(parse_comma_list(getattr(args, "ga_models_resolved", ""))) if parse_comma_list(getattr(args, "ga_models_resolved", "")) else "(none)")
    )
    print(
        "CFA stage: "
        f"{'on' if bool(getattr(args, 'run_cfa', False)) else 'off'} "
        f"(min_models={int(getattr(args, 'cfa_min_models', 2))}, max_models={int(getattr(args, 'cfa_max_models', 4))}, "
        f"opt_metric={str(getattr(args, 'cfa_optimize_metric', 'mae'))}, "
        f"sources={str(getattr(args, 'cfa_source_workflows', 'all'))}, "
        f"rank={'on' if bool(getattr(args, 'cfa_include_rank_combinations', True)) else 'off'}, "
        f"rank_pref={'on' if bool(getattr(args, 'cfa_rank_prefer_when_diverse', True)) else 'off'}, "
        f"rank_threshold={float(getattr(args, 'cfa_rank_diversity_threshold', 0.15)):.3f}, "
        f"rank_discount={float(getattr(args, 'cfa_rank_metric_discount', 0.98)):.3f})"
    )
    print(
        "MapLight parity mode: "
        f"{'strict' if bool(getattr(args, 'maplight_leaderboard_parity_mode', True)) else 'legacy'} "
        f"(seeds={','.join(str(seed) for seed in maplight_parity_seed_values(args))})"
    )
    if resume_plan is not None:
        print_resume_execution_plan(resume_plan)
    if bool(getattr(args, "run_tabpfn", False)):
        print(f"TabPFN backend source: {TABPFN_REGRESSOR_SOURCE}")
    print(
        "Uni-Mol V1 stage: "
        f"{'on' if bool(getattr(args, 'run_unimol_v1', False)) else 'off'} "
        f"(gpu_detected={'yes' if bool(getattr(args, 'gpu_available', False)) else 'no'})"
    )
    print(f"Leaderboard reference rows stored: {len(leaderboard_ref_df)}")
    print("Datasets:")
    for dataset in datasets:
        leaderboard_metric = ((dataset.leaderboard_summary or {}).get("metric_name") or "").strip()
        leaderboard_value = ((dataset.leaderboard_summary or {}).get("metric_value") or "").strip()
        leaderboard_note = f", leaderboard={leaderboard_metric} {leaderboard_value}".strip()
        target_transform_note = "log10" if resolve_dataset_log10_target(dataset, args) else "raw"
        print(
            f"  - {dataset.name}: smiles={dataset.smiles_column}, target={dataset.target_column}, "
            f"source={dataset.source}, split={dataset.recommended_split or args.split_strategy}"
            f", target_transform={target_transform_note}"
            f"{leaderboard_note if leaderboard_metric or leaderboard_value else ''}"
        )

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

    summary = build_summary_from_dataset_metrics(output_dir)
    if summary.empty and all_metrics:
        summary = deduplicate_metrics_rows(pd.DataFrame(all_metrics))
    if not summary.empty:
        summary.to_csv(output_dir / "summary_metrics.csv", index=False)
        leaderboard_comparison_df = leaderboard_comparison_by_dataset(summary)
        if not leaderboard_comparison_df.empty:
            leaderboard_comparison_df.to_csv(output_dir / "leaderboard_comparison_by_dataset.csv", index=False)
        if {"dataset", "model", "test_rmse"}.issubset(summary.columns):
            pivot = summary.pivot_table(index="dataset", columns="model", values="test_rmse", aggfunc="min")
            pivot.to_csv(output_dir / "test_rmse_pivot.csv")
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    if all_histories:
        pd.concat(all_histories, ignore_index=True).to_csv(output_dir / "ga_history.csv", index=False)

    if summary.empty:
        summary = load_summary_metrics_for_output_dir(output_dir)

    step_runtime_summary = collect_step_runtime_summary(output_dir)
    if not step_runtime_summary.empty:
        print(
            "Step runtime rows captured: "
            f"{len(step_runtime_summary):,} (saved to {output_dir / 'step_runtime_summary.csv'})",
            flush=True,
        )

    model_value_report, zero_value_candidates = write_model_value_report(output_dir, summary)
    if not model_value_report.empty:
        print(
            "Model value report saved: "
            f"{output_dir / 'model_value_report.csv'} "
            f"(zero-value candidates: {len(zero_value_candidates)})",
            flush=True,
        )

    attribution_payload = {}
    if bool(getattr(args, "emit_run_vs_run_report", True)):
        attribution_payload = write_run_vs_run_attribution_report(
            root=root,
            output_dir=output_dir,
            current_metrics=summary,
            args=args,
        )
        print(
            "Run-vs-run attribution report saved: "
            f"{output_dir / 'run_vs_run_attribution_summary.json'}",
            flush=True,
        )
    (output_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - overall_start, 3),
                "dataset_count": len(datasets),
                "ga_models_resolved": parse_comma_list(getattr(args, "ga_models_resolved", "")),
                "run_vs_run_attribution": attribution_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    primary_files = ["summary_metrics.csv", "test_rmse_pivot.csv", "predictions.csv", "run_config.json"]
    if all_histories:
        primary_files.append("ga_history.csv")
    if not step_runtime_summary.empty:
        primary_files.append("step_runtime_summary.csv")
    if not model_value_report.empty:
        primary_files.extend(["model_value_report.csv", "model_zero_value_candidates.csv"])
    if bool(getattr(args, "emit_run_vs_run_report", True)):
        primary_files.extend(
            [
                "run_vs_run_attribution_summary.json",
                "run_vs_run_split_match.csv",
                "run_vs_run_config_diff.csv",
                "run_vs_run_error_diff.csv",
                "run_vs_run_leaderboard_comparability.csv",
            ]
        )
    print(f"\nWrote benchmark outputs to {output_dir}")
    print("Primary files: " + ", ".join(primary_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
