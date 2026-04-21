#!/usr/bin/env python
"""Benchmark the AutoQSAR notebook workflow (CPU-focused).

This script benchmarks the notebook's example SMILES-based datasets: runnable
ChemML bundled examples plus QSAR benchmark suites (PyTDC, MoleculeNet
physchem, and Polaris ADME Fang splits). It builds
molecular features, runs the train/test split plus train-only ElasticNetCV
feature selection, evaluates conventional models, optionally runs a small GA
tuning pass, runs deep workflows (ChemML backends, Chemprop v2 graph variants,
and MapLight + GNN), builds an optional ensemble over available members, and
writes cross-dataset performance tables.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import math
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
from pathlib import Path
from typing import Any, Callable
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, RidgeCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

try:
    from portable_colab_qsar_bundle.qsar_workflow_core import (
        build_feature_matrix_from_smiles,
        drop_exact_and_near_duplicate_features,
        make_qsar_cv_splitter,
        make_reusable_inner_cv_splitter,
        resolve_chemprop_architecture_specs,
        scaffold_train_test_split,
        target_quartile_labels,
    )
    from portable_colab_qsar_bundle.benchmark_registry import (
        CHEMML_EXAMPLE_OPTIONS,
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
        make_qsar_cv_splitter,
        make_reusable_inner_cv_splitter,
        resolve_chemprop_architecture_specs,
        scaffold_train_test_split,
        target_quartile_labels,
    )
    from portable_colab_qsar_bundle.benchmark_registry import (
        CHEMML_EXAMPLE_OPTIONS,
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
    config["default_feature_families"] = list(DEFAULT_BENCHMARK_FEATURE_FAMILIES)
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
    section_name = config.get("leaderboard_section")
    if not section_name:
        return None
    request = urllib.request.Request(MOLECULENET_LEADERBOARD_README_URL, headers={"User-Agent": "AutoQSAR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=int(timeout)) as response:
        markdown = response.read().decode("utf-8", errors="replace")
    section_pattern = rf"###\s*{re.escape(str(section_name))}\s*(?P<section>.*?)(?:\n###\s+|\Z)"
    section_match = re.search(section_pattern, markdown, flags=re.IGNORECASE | re.DOTALL)
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
        return "rmse"
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
        primary_metric = str(group["primary_metric"].dropna().iloc[0]).strip().lower() if group["primary_metric"].notna().any() else "rmse"
        lower_is_better = metric_lower_is_better(primary_metric)
        ascending = True if lower_is_better is None else bool(lower_is_better)
        best = group.sort_values("primary_metric_value", ascending=ascending).iloc[0]
        rows.append(
            {
                "dataset": str(dataset_name),
                "best_model": str(best.get("model", "")),
                "best_workflow": str(best.get("workflow", "")),
                "primary_metric": primary_metric,
                "primary_metric_value": best.get("primary_metric_value", np.nan),
                "leaderboard_metric_name": best.get("leaderboard_metric_name", ""),
                "leaderboard_metric_reference": best.get("leaderboard_metric_reference", np.nan),
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
            "selected_feature_count": int(len(columns)),
            "original_feature_count": int(X_train.shape[1]),
            "selector_alpha": float(selector_fit["alpha"]),
            "selector_l1_ratio": float(selector_fit["l1_ratio"]),
            "selector_n_iter": int(selector_fit["n_iter"]),
            "selector_cv_folds": int(selector_cv_folds),
            "selector_cv_split_strategy": selector_cv_strategy,
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
        "selected_feature_count": int(len(columns)),
        "original_feature_count": int(X_train.shape[1]),
        "selector_alpha": np.nan,
        "selector_l1_ratio": np.nan,
        "selector_n_iter": np.nan,
        "selector_cv_folds": int(selector_cv_folds),
        "selector_cv_split_strategy": selector_cv_strategy,
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


def conventional_models(
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    smiles_train: pd.Series,
    split_strategy_for_cv: str | None = None,
) -> dict[str, Any]:
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
    leaderboard_summary = (CURRENT_DATASET_SPEC.leaderboard_summary if CURRENT_DATASET_SPEC is not None else None) or {}
    row["leaderboard_model"] = leaderboard_summary.get("model", "")
    row["leaderboard_metric_name"] = leaderboard_summary.get("metric_name", "")
    row["leaderboard_metric_value"] = leaderboard_summary.get("metric_value", "")
    row["leaderboard_rank"] = leaderboard_summary.get("rank", "")
    row["leaderboard_url"] = leaderboard_summary.get("url", CURRENT_DATASET_SPEC.leaderboard_url if CURRENT_DATASET_SPEC is not None else "")
    top10_entries = leaderboard_summary.get("top10", [])
    row["leaderboard_top10_count"] = int(len(top10_entries)) if isinstance(top10_entries, list) else 0
    row["leaderboard_top10_json"] = json.dumps(top10_entries, default=str) if isinstance(top10_entries, list) else "[]"
    leaderboard_metric_norm = normalize_benchmark_metric(leaderboard_summary.get("metric_name"), fallback="")
    leaderboard_value = parse_first_float(leaderboard_summary.get("metric_value"))
    if leaderboard_metric_norm and leaderboard_value is not None and leaderboard_metric_norm == str(primary_metric).strip().lower():
        row["leaderboard_metric_normalized"] = leaderboard_metric_norm
        row["leaderboard_metric_reference"] = float(leaderboard_value)
        row["leaderboard_delta_primary"] = float(primary_test_value - leaderboard_value)
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
            if isinstance(entry, dict) and normalize_benchmark_metric(entry.get("metric_name"), fallback=leaderboard_metric_norm) == leaderboard_metric_norm
        ]
        comparable_values = [value for value in comparable_values if value is not None]
        if comparable_values and leaderboard_metric_norm == str(primary_metric).strip().lower():
            top10_best = min(comparable_values) if metric_lower_is_better(leaderboard_metric_norm) else max(comparable_values)
            row["leaderboard_top10_best_reference"] = float(top10_best)
            row["leaderboard_delta_top10_best"] = float(primary_test_value - float(top10_best))
        else:
            row["leaderboard_top10_best_reference"] = np.nan
            row["leaderboard_delta_top10_best"] = np.nan
    else:
        row["leaderboard_top10_best_reference"] = np.nan
        row["leaderboard_delta_top10_best"] = np.nan
    rank_columns = estimate_rank_columns_for_row(
        primary_metric=str(primary_metric),
        primary_value=float(primary_test_value),
        leaderboard_summary=leaderboard_summary,
    )
    row.update(rank_columns)
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


def _build_maplight_gnn_embedder(kind: str = "gin_supervised_masking") -> tuple[Callable[[list[str]], list[Any]], str]:
    _patch_torchdata_dill_available()
    from molfeat.trans.pretrained import PretrainedDGLTransformer

    try:
        transformer = PretrainedDGLTransformer(kind=kind, dtype=float)

        def _embed_with_molfeat(smiles_values: list[str]):
            return transformer(smiles_values)

        return _embed_with_molfeat, "molfeat-store"
    except Exception as primary_exc:
        if not _is_maplight_store_access_error(primary_exc):
            raise
        try:
            import dgl
            import dgllife
            import torch
            from torch.utils.data import DataLoader

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

            return _embed_with_dgllife, "dgllife-direct"
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


def _extract_chemprop_predictions(preds_csv_path: Path, expected_smiles: pd.Series) -> np.ndarray:
    preds_df = pd.read_csv(preds_csv_path)
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
    pred_values = pd.to_numeric(preds_df[prediction_cols[0]], errors="coerce")
    if pred_values.isna().any():
        raise ValueError("Chemprop prediction output contains non-numeric values.")

    expected_smiles = pd.Series(expected_smiles, dtype=str).reset_index(drop=True)
    if smiles_col is not None:
        actual_smiles = preds_df[smiles_col].astype(str).reset_index(drop=True)
        if len(actual_smiles) == len(expected_smiles) and set(actual_smiles.tolist()) == set(expected_smiles.tolist()):
            aligned = (
                pd.DataFrame({"smiles": actual_smiles, "predicted": pred_values.to_numpy(dtype=float)})
                .set_index("smiles")
                .loc[expected_smiles.tolist(), "predicted"]
                .to_numpy(dtype=float)
            )
            return np.asarray(aligned, dtype=float)
    if len(pred_values) != len(expected_smiles):
        raise ValueError(
            f"Chemprop prediction length mismatch: got {len(pred_values)} rows, expected {len(expected_smiles)}."
        )
    return pred_values.to_numpy(dtype=float)


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

    pred_train: np.ndarray | None = None
    pred_test: np.ndarray | None = None
    feature_args: list[str] = []
    if featurizer_list:
        feature_args = ["--molecule-featurizers", *featurizer_list]
    chemprop_split_mode = "SCAFFOLD_BALANCED" if str(split_strategy_for_cv).strip().lower() == "scaffold" else "RANDOM"

    if bool(args.chemprop_reuse_model_cache) and train_preds_path.exists() and test_preds_path.exists():
        try:
            pred_train = _extract_chemprop_predictions(train_preds_path, train_df["SMILES"].astype(str))
            pred_test = _extract_chemprop_predictions(test_preds_path, test_df["SMILES"].astype(str))
            print(f"[Chemprop] loaded cached predictions from {save_dir}", flush=True)
        except Exception:
            pred_train = None
            pred_test = None

    if pred_train is None or pred_test is None:
        if bool(args.chemprop_reuse_model_cache):
            try:
                _run_chemprop_command(
                    command_prefix,
                    [
                        "predict",
                        "--test-path", str(train_csv),
                        "--smiles-columns", "SMILES",
                        "--model-paths", str(save_dir),
                        "--preds-path", str(train_preds_path),
                        *feature_args,
                    ],
                    description="train-set prediction (cached model)",
                )
                _run_chemprop_command(
                    command_prefix,
                    [
                        "predict",
                        "--test-path", str(test_csv),
                        "--smiles-columns", "SMILES",
                        "--model-paths", str(save_dir),
                        "--preds-path", str(test_preds_path),
                        *feature_args,
                    ],
                    description="test-set prediction (cached model)",
                )
                pred_train = _extract_chemprop_predictions(train_preds_path, train_df["SMILES"].astype(str))
                pred_test = _extract_chemprop_predictions(test_preds_path, test_df["SMILES"].astype(str))
            except Exception:
                pred_train = None
                pred_test = None

    if pred_train is None or pred_test is None:
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
                *feature_args,
            ],
            description="training",
        )
        _run_chemprop_command(
            command_prefix,
            [
                "predict",
                "--test-path", str(train_csv),
                "--smiles-columns", "SMILES",
                "--model-paths", str(save_dir),
                "--preds-path", str(train_preds_path),
                *feature_args,
            ],
            description="train-set prediction",
        )
        _run_chemprop_command(
            command_prefix,
            [
                "predict",
                "--test-path", str(test_csv),
                "--smiles-columns", "SMILES",
                "--model-paths", str(save_dir),
                "--preds-path", str(test_preds_path),
                *feature_args,
            ],
            description="test-set prediction",
        )
        pred_train = _extract_chemprop_predictions(train_preds_path, train_df["SMILES"].astype(str))
        pred_test = _extract_chemprop_predictions(test_preds_path, test_df["SMILES"].astype(str))

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
        "chemprop_molecule_featurizers": ",".join(featurizer_list),
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
    for model_name in prediction_columns:
        row = {"model": model_name, "workflow": str(payloads[model_name]["workflow"])}
        row.update(regression_metrics(y_meta_train, aligned_train[model_name], y_meta_test, aligned_test[model_name]))
        row["primary_metric"] = "rmse"
        row["primary_metric_value"] = float(math.sqrt(mean_squared_error(y_meta_test, aligned_test[model_name])))
        ensemble_rows.append(row)

    ensemble_row = {"model": f"Ensemble ({ensemble_method_label})", "workflow": "ensemble"}
    ensemble_row.update(regression_metrics(y_meta_train, ensemble_train_pred, y_meta_test, ensemble_test_pred))
    ensemble_row["primary_metric"] = "rmse"
    ensemble_row["primary_metric_value"] = float(math.sqrt(mean_squared_error(y_meta_test, ensemble_test_pred)))
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
    for feature_name in list(dedup_meta.get("dropped_exact_columns", [])):
        dropped_rows.append({"feature": str(feature_name), "reason": "exact_duplicate"})
    for feature_name in list(dedup_meta.get("dropped_near_columns", [])):
        dropped_rows.append({"feature": str(feature_name), "reason": "near_duplicate"})
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
    names = ["ElasticNetCV", "SVR", "Random forest"]
    if XGBRegressor is not None:
        names.append("XGBoost")
    if CatBoostRegressor is not None:
        names.append("CatBoost")
        names.append("MapLight CatBoost")
    return names


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
    return resolve_chemprop_architecture_specs(
        architecture_keys,
        ensemble_size=int(args.chemprop_ensemble_size),
        include_rdkit2d_extra=bool(getattr(args, "run_chemprop_rdkit2d", False)),
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
        if not bool(getattr(args, "revisit_completed_datasets", False)) and not rebuild_ensemble_requested:
            print(f"\n{prefix}{dataset_id}: already completed in {format_seconds(completed_result.elapsed_seconds)}; reusing saved outputs")
            return completed_result
        if rebuild_ensemble_requested and not bool(getattr(args, "revisit_completed_datasets", False)):
            print(
                f"\n{prefix}{dataset_id}: previously completed in {format_seconds(completed_result.elapsed_seconds)}; "
                "revisiting to rebuild the ensemble.",
                flush=True,
            )
        else:
            print(
                f"\n{prefix}{dataset_id}: previously completed in {format_seconds(completed_result.elapsed_seconds)}; "
                "revisiting to run any newly requested model stages.",
                flush=True,
            )

    requested_ga_models = [model.strip() for model in args.ga_models.split(",") if model.strip()]
    requested_deep_stages = (
        int(bool(args.run_chemml_pytorch))
        + int(bool(args.run_chemml_tensorflow))
        + (len(chemprop_variant_specs(args)) if bool(args.run_chemprop_mpnn) else 0)
        + int(bool(args.run_maplight_gnn))
    )
    requested_ensemble_stage = int(bool(args.run_ensemble))
    total_stages = 3 + len(selected_conventional_model_names(args)) + len(requested_ga_models) + requested_deep_stages + requested_ensemble_stage

    def stage_message(stage_index: int, label: str) -> None:
        stage_started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        elapsed = time.time() - start
        avg_stage = elapsed / max(1, stage_index - 1) if stage_index > 1 else 0.0
        remaining = max(0, total_stages - stage_index + 1)
        eta = avg_stage * remaining if avg_stage else 0.0
        prefix = f"[{dataset_position}/{dataset_total}] " if dataset_position is not None and dataset_total is not None else ""
        print(
            f"\n{prefix}{dataset_id} | stage {stage_index}/{total_stages}: {label} "
            f"| started {stage_started_at} | elapsed {format_seconds(elapsed)} | dataset ETA {format_seconds(eta)}"
        )

    metrics_rows, prediction_tables, ga_history_tables = load_partial_dataset_artifacts(dataset_dir)
    prediction_payloads = rebuild_prediction_payloads(prediction_tables)
    completed_model_names = {
        str(row.get("model", "")).strip()
        for row in metrics_rows
        if str(row.get("model", "")).strip()
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
            if str(row.get("model", "")).strip()
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
        write_dataset_status(dataset_dir, {"status": "skipped", "reason": "too_few_rows_after_cleanup", "n_rows": int(len(df))})
        return DatasetRunResult([], [], [], "skipped", time.time() - start)
    if args.row_limit and len(df) > args.row_limit:
        df = df.sample(n=args.row_limit, random_state=args.random_seed).reset_index(drop=True)

    stage_message(2, "building molecular features")
    X, feature_meta = build_feature_matrix_from_smiles(
        df["canonical_smiles"].tolist(),
        selected_feature_families=list(DEFAULT_BENCHMARK_FEATURE_FAMILIES),
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
    predefined_split = None
    if spec.predefined_split_column and spec.predefined_split_column in df.columns:
        predefined_split = df[spec.predefined_split_column].reset_index(drop=True)
    split = split_data(X, y, smiles, args, predefined_split=predefined_split)
    split["X_train"], split["X_test"], feature_dedup_meta = drop_exact_and_near_duplicate_features(
        split["X_train"],
        split["X_test"],
        correlation_threshold=0.999,
    )
    dedup_dropped_count = int(feature_dedup_meta.get("dropped_feature_count", 0))
    if dedup_dropped_count > 0:
        print(
            "[feature-dedup] dropped "
            f"{dedup_dropped_count:,} feature(s) before selector/model fitting "
            f"({int(feature_dedup_meta.get('dropped_exact_count', 0)):,} exact, "
            f"{int(feature_dedup_meta.get('dropped_near_count', 0)):,} near-duplicate at "
            f"|r|>{float(feature_dedup_meta.get('correlation_threshold', 0.999)):.3f}).",
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
    maplight_prefixes = ("maplight_morgan_", "avalon_count_", "erg_", "maplight_desc_")
    maplight_feature_cols = [
        col for col in split["X_train"].columns if str(col).startswith(maplight_prefixes)
    ]

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
        "feature_dedup_exact_count": int(feature_dedup_meta.get("dropped_exact_count", 0)),
        "feature_dedup_near_count": int(feature_dedup_meta.get("dropped_near_count", 0)),
        "feature_dedup_threshold": float(feature_dedup_meta.get("correlation_threshold", 0.999)),
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
        "selected_feature_families_json": json.dumps(feature_meta.get("selected_feature_families", []), default=str),
        "built_feature_families_json": json.dumps(feature_meta.get("built_feature_families", []), default=str),
        "feature_store_path": feature_meta.get("feature_store_path", ""),
        "feature_store_shard_format": feature_meta.get("feature_store_shard_format", ""),
        "feature_store_cached_rows": feature_meta.get("cached_rows_loaded", np.nan),
        "feature_store_generated_rows": feature_meta.get("generated_rows_added", np.nan),
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
    elasticnet_cv_meta = model_bundle.pop("_elasticnet_cv_meta", {})
    model_items = list(model_bundle.items())
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
            row = {"model": model_name, "workflow": "conventional", "error": str(exc)}
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

    if bool(args.run_maplight_gnn):
        maplight_label = "MapLight + GNN (CatBoost)"
        if maplight_label in completed_model_names:
            stage_message(stage_index, f"deep model {maplight_label} (cached)")
            stage_index += 1
        else:
            stage_message(stage_index, f"deep model {maplight_label}")
            if CatBoostRegressor is None:
                row = {"model": maplight_label, "workflow": "MapLight + GNN", "error": "CatBoost is unavailable"}
                metrics_rows.append({**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)})
                completed_model_names.add(maplight_label)
            elif not maplight_feature_cols:
                row = {"model": maplight_label, "workflow": "MapLight + GNN", "error": "MapLight classic features are unavailable"}
                metrics_rows.append({**base_meta, **row, "elapsed_seconds": round(time.time() - start, 3)})
                completed_model_names.add(maplight_label)
            else:
                try:
                    embedder, embedding_backend = _build_maplight_gnn_embedder(str(args.maplight_gnn_kind))
                    train_embeddings_raw = embedder(split["smiles_train"].astype(str).tolist())
                    test_embeddings_raw = embedder(split["smiles_test"].astype(str).tolist())
                    emb_train = _embeddings_to_matrix(train_embeddings_raw)
                    emb_test = _embeddings_to_matrix(test_embeddings_raw, reference_width=emb_train.shape[1])
                    train_fill = np.nanmean(emb_train, axis=0)
                    train_fill = np.where(np.isfinite(train_fill), train_fill, 0.0)
                    emb_train = np.where(np.isfinite(emb_train), emb_train, train_fill)
                    emb_test = np.where(np.isfinite(emb_test), emb_test, train_fill)
                    maplight_base_train = split["X_train"].loc[:, maplight_feature_cols].to_numpy(dtype=float)
                    maplight_base_test = split["X_test"].loc[:, maplight_feature_cols].to_numpy(dtype=float)
                    gnn_feature_names = [f"gin_emb_{idx:03d}" for idx in range(int(emb_train.shape[1]))]
                    fusion_train = pd.DataFrame(
                        np.concatenate([maplight_base_train, emb_train], axis=1),
                        columns=list(maplight_feature_cols) + gnn_feature_names,
                    )
                    fusion_test = pd.DataFrame(
                        np.concatenate([maplight_base_test, emb_test], axis=1),
                        columns=list(maplight_feature_cols) + gnn_feature_names,
                    )
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
        },
    )
    return DatasetRunResult(final_metrics_rows, prediction_tables, ga_history_tables, "completed", elapsed_seconds)


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
        default=2048,
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

    parser.add_argument("--selector-method", choices=["elasticnet_cv", "none"], default="elasticnet_cv")
    parser.add_argument("--selector-l1-ratio-grid", default="0.1, 0.3, 0.5, 0.7, 0.9")
    parser.add_argument("--selector-alpha-min-log10", type=float, default=-5)
    parser.add_argument("--selector-alpha-max-log10", type=float, default=-1)
    parser.add_argument("--selector-alpha-grid-size", type=int, default=40)
    parser.add_argument("--selector-cv-folds", type=int, default=5)
    parser.add_argument("--selector-max-iter", type=int, default=50000)
    parser.add_argument("--selector-coefficient-threshold", type=float, default=1e-10)
    parser.add_argument(
        "--selector-elasticnet-timeout-seconds",
        type=float,
        default=10800.0,
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

    parser.add_argument("--run-chemml-pytorch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemml-tensorflow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemprop-mpnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemprop-dmpnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemprop-cmpnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-chemprop-attentivefp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-maplight-gnn", action=argparse.BooleanOptionalAction, default=True)
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

    if args.dry_run:
        print(f"Planned output directory: {output_dir}")
        print(f"Default molecular feature families: {', '.join(DEFAULT_BENCHMARK_FEATURE_FAMILIES)}")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["default_feature_families"] = list(DEFAULT_BENCHMARK_FEATURE_FAMILIES)
    config["config_signature"] = benchmark_config_signature(args)
    config["datasets"] = [{"name": item.name, "source": item.source, "smiles_column": item.smiles_column, "target_column": item.target_column} for item in datasets]
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    leaderboard_ref_df = write_leaderboard_reference_artifacts(root, output_dir, datasets)

    print(f"Output directory: {output_dir}")
    print(f"Default molecular feature families: {', '.join(DEFAULT_BENCHMARK_FEATURE_FAMILIES)}")
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

    if all_metrics:
        summary = pd.DataFrame(all_metrics)
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
