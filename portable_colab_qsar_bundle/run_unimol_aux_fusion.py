#!/usr/bin/env python
"""Grouped Uni-Mol plus auxiliary-feature fusion for PFAS workbook sheets.

This script runs a stacked regression workflow for workbook sheets that contain
SMILES, TARGET_log10, a predefined split column, and numeric auxiliary features.
It trains Uni-Mol on SMILES-only data, creates out-of-fold training predictions,
then feeds Uni-Mol predictions plus auxiliary variables into a tabular fusion
model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IDENTIFIER_COLUMNS = {
    "target",
    "smiles_subset",
    "split",
    "QSAR_READY_SMILES",
    "SMILES",
    "smiles",
    "canonical_smiles",
    "TARGET_log10",
    "target_value",
    "source_sheet",
}


@dataclass(frozen=True)
class UniMolPredictionResult:
    predictions: np.ndarray
    model_dir: Path
    elapsed_seconds: float
    cached: bool = False


class StatusPrinter:
    COLORS = {
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }

    def __init__(self, *, use_color: bool) -> None:
        self.use_color = bool(use_color)

    def color(self, text: str, name: str) -> str:
        if not self.use_color:
            return text
        return f"{self.COLORS.get(name, '')}{text}{self.COLORS['reset']}"

    def line(self, tag: str, message: str, color: str = "cyan") -> None:
        print(f"{self.color(tag, color)} {message}", flush=True)

    def section(self, message: str) -> None:
        self.line("[fusion]", message, "bold")

    def info(self, message: str) -> None:
        self.line("[info]", message, "cyan")

    def ok(self, message: str) -> None:
        self.line("[done]", message, "green")

    def warn(self, message: str) -> None:
        self.line("[skip]", message, "yellow")


class ProgressReporter:
    def __init__(self, *, total: int, status: StatusPrinter) -> None:
        self.total = max(int(total), 0)
        self.completed = 0
        self.status = status

    def start(self, label: str, *, train_rows: int, predict_rows: int, unique_smiles: int) -> None:
        next_index = min(self.completed + 1, self.total)
        self.status.line(
            "[unimol]",
            (
                f"job {next_index}/{self.total}: {label} "
                f"(train_rows={train_rows:,}, unique_smiles={unique_smiles:,}, predict_rows={predict_rows:,})"
            ),
            "blue",
        )

    def finish(self, label: str, result: UniMolPredictionResult) -> None:
        self.completed += 1
        mode = "cache" if result.cached else f"{result.elapsed_seconds:.1f}s"
        self.status.ok(f"job {self.completed}/{self.total}: {label} complete ({mode})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True, help="PFAS auxiliary workbook.")
    parser.add_argument("--sheet", required=True, help="Workbook sheet to run, for example VDss_all_aux.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for fusion outputs.")
    parser.add_argument(
        "--group-prefix",
        action="append",
        default=None,
        help="One-hot auxiliary column prefix used to define Uni-Mol groups. Repeatable. Default: species_, sex_.",
    )
    parser.add_argument("--min-group-rows", type=int, default=30, help="Minimum training rows for a grouped Uni-Mol model.")
    parser.add_argument("--min-group-unique-smiles", type=int, default=20, help="Minimum unique training SMILES for a grouped Uni-Mol model.")
    parser.add_argument("--oof-folds", type=int, default=5, help="Maximum OOF folds for global and grouped Uni-Mol models.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--unimol-epochs", type=int, default=10)
    parser.add_argument("--unimol-learning-rate", type=float, default=1e-4)
    parser.add_argument("--unimol-batch-size", type=int, default=16)
    parser.add_argument("--unimol-early-stopping", type=int, default=5)
    parser.add_argument("--unimol-num-workers", type=int, default=0)
    parser.add_argument(
        "--fusion-model",
        choices=["catboost", "random_forest", "ridge"],
        default="catboost",
        help="Second-stage model. CatBoost is used when installed; otherwise the script falls back to random forest.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing completed output directory.")
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Colorize progress messages. Default: auto.",
    )
    return parser.parse_args()


def normalize_smiles(value: Any) -> str:
    return str(value).strip()


def load_sheet(workbook: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    required = {"split", "QSAR_READY_SMILES", "TARGET_log10"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{sheet} is missing required column(s): {', '.join(missing)}")
    frame = frame.copy()
    frame["QSAR_READY_SMILES"] = frame["QSAR_READY_SMILES"].map(normalize_smiles)
    frame["TARGET_log10"] = pd.to_numeric(frame["TARGET_log10"], errors="coerce")
    frame = frame.dropna(subset=["QSAR_READY_SMILES", "TARGET_log10"])
    frame = frame[frame["QSAR_READY_SMILES"] != ""].reset_index(drop=True)
    return frame


def numeric_auxiliary_columns(frame: pd.DataFrame) -> list[str]:
    aux_columns: list[str] = []
    for column in frame.columns:
        if str(column) in IDENTIFIER_COLUMNS:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            frame[column] = numeric
            aux_columns.append(str(column))
    return aux_columns


def active_one_hot_label(row: pd.Series, prefix: str) -> str:
    matches = []
    for column, value in row.items():
        if not str(column).startswith(prefix):
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric) > 0:
            matches.append(str(column)[len(prefix) :].strip() or "unknown")
    if not matches:
        return "unknown"
    return "+".join(sorted(matches))


def add_group_labels(frame: pd.DataFrame, group_prefixes: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    labels = []
    for _, row in frame.iterrows():
        parts = [f"{prefix.rstrip('_')}={active_one_hot_label(row, prefix)}" for prefix in group_prefixes]
        labels.append("|".join(parts))
    frame["unimol_group"] = labels
    return frame


def collapsed_unimol_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame[["QSAR_READY_SMILES", "TARGET_log10"]].copy()
    working["TARGET_log10"] = pd.to_numeric(working["TARGET_log10"], errors="coerce")
    working = working.dropna(subset=["QSAR_READY_SMILES", "TARGET_log10"])
    collapsed = (
        working.groupby("QSAR_READY_SMILES", as_index=False, sort=True)
        .agg(TARGET=("TARGET_log10", "mean"))
        .rename(columns={"QSAR_READY_SMILES": "SMILES"})
    )
    return collapsed[["SMILES", "TARGET"]]


def write_prediction_csv(smiles: Sequence[str], path: Path) -> None:
    frame = pd.DataFrame({"SMILES": pd.Series(smiles, dtype=str), "TARGET": 0.0})
    frame.to_csv(path, index=False)


def color_enabled(args: argparse.Namespace) -> bool:
    setting = str(getattr(args, "color", "auto")).strip().lower()
    if setting == "always":
        return True
    if setting == "never":
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def run_unimol_predict(
    *,
    train_rows: pd.DataFrame,
    predict_smiles: Sequence[str],
    model_dir: Path,
    args: argparse.Namespace,
) -> UniMolPredictionResult:
    try:
        from unimol_tools import MolPredict, MolTrain
    except Exception as exc:
        raise RuntimeError("Uni-Mol fusion requires `unimol_tools` in the active environment.") from exc

    start = time.perf_counter()
    model_dir.mkdir(parents=True, exist_ok=True)
    train_csv = model_dir / "train.csv"
    predict_csv = model_dir / "predict.csv"
    pred_cache = model_dir / "predictions.npy"

    if bool(args.resume) and pred_cache.exists() and not bool(args.force):
        predictions = np.asarray(np.load(pred_cache), dtype=float).reshape(-1)
        if len(predictions) == len(predict_smiles):
            return UniMolPredictionResult(predictions, model_dir, 0.0, cached=True)

    collapsed = collapsed_unimol_training_frame(train_rows)
    if collapsed["SMILES"].nunique() < 2:
        raise ValueError(f"Need at least two unique SMILES to train Uni-Mol in {model_dir}")
    collapsed.to_csv(train_csv, index=False)
    write_prediction_csv(predict_smiles, predict_csv)

    trainer = MolTrain(
        task="regression",
        data_type="molecule",
        model_name="unimolv1",
        epochs=int(args.unimol_epochs),
        learning_rate=float(args.unimol_learning_rate),
        batch_size=int(args.unimol_batch_size),
        early_stopping=int(args.unimol_early_stopping),
        metrics="mse",
        split="random",
        save_path=str(model_dir),
        num_workers=int(args.unimol_num_workers),
    )
    trainer.fit(str(train_csv))
    predictor = MolPredict(load_model=str(model_dir))
    predictions = np.asarray(predictor.predict(str(predict_csv)), dtype=float).reshape(-1)
    np.save(pred_cache, predictions)
    return UniMolPredictionResult(predictions, model_dir, float(time.perf_counter() - start))


def group_kfold_indices(frame: pd.DataFrame, max_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["QSAR_READY_SMILES"].astype(str).to_numpy()
    unique_count = int(pd.Series(groups).nunique())
    n_splits = min(int(max_folds), unique_count, len(frame))
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(frame))
    return list(splitter.split(indices, groups=groups))


def build_oof_predictions(
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    status: StatusPrinter,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pred = train_frame[["QSAR_READY_SMILES", "TARGET_log10", "unimol_group"]].copy()
    test_pred = test_frame[["QSAR_READY_SMILES", "TARGET_log10", "unimol_group"]].copy()
    train_pred["unimol_global_oof"] = np.nan
    train_pred["unimol_group_oof"] = np.nan
    test_pred["unimol_global_pred"] = np.nan
    test_pred["unimol_group_pred"] = np.nan
    group_rows: list[dict[str, Any]] = []

    global_folds = group_kfold_indices(train_frame, args.oof_folds)
    if not global_folds:
        raise ValueError("Global OOF Uni-Mol requires at least two unique training SMILES.")

    group_plan: list[dict[str, Any]] = []
    for group_label, group_train in train_frame.groupby("unimol_group", sort=True):
        group_test = test_frame[test_frame["unimol_group"].eq(group_label)]
        unique_smiles = int(group_train["QSAR_READY_SMILES"].nunique())
        viable = len(group_train) >= int(args.min_group_rows) and unique_smiles >= int(args.min_group_unique_smiles)
        if viable:
            folds = group_kfold_indices(group_train.reset_index(drop=True), args.oof_folds)
            viable = bool(folds)
        else:
            folds = []
        group_plan.append(
            {
                "group_label": group_label,
                "group_train": group_train,
                "group_test": group_test,
                "unique_smiles": unique_smiles,
                "viable": viable,
                "folds": folds,
            }
        )

    total_jobs = len(global_folds) + 1 + sum(
        len(item["folds"]) + (1 if item["viable"] and len(item["group_test"]) else 0)
        for item in group_plan
    )
    progress = ProgressReporter(total=total_jobs, status=status)
    status.section(
        f"Uni-Mol plan: {total_jobs:,} model/prediction job(s) "
        f"({len(global_folds)} global OOF, 1 global final, "
        f"{sum(1 for item in group_plan if item['viable'])} trained subgroup(s), "
        f"{sum(1 for item in group_plan if not item['viable'])} global-fallback subgroup(s))."
    )

    for fold_index, (fit_idx, valid_idx) in enumerate(global_folds, start=1):
        valid_smiles = train_frame.iloc[valid_idx]["QSAR_READY_SMILES"].astype(str).tolist()
        label = f"global OOF fold {fold_index}/{len(global_folds)}"
        progress.start(
            label,
            train_rows=len(fit_idx),
            predict_rows=len(valid_idx),
            unique_smiles=int(train_frame.iloc[fit_idx]["QSAR_READY_SMILES"].nunique()),
        )
        result = run_unimol_predict(
            train_rows=train_frame.iloc[fit_idx],
            predict_smiles=valid_smiles,
            model_dir=output_dir / "unimol_models" / "global_oof" / f"fold_{fold_index}",
            args=args,
        )
        progress.finish(label, result)
        train_pred.loc[train_frame.index[valid_idx], "unimol_global_oof"] = result.predictions

    label = "global final test predictor"
    progress.start(
        label,
        train_rows=len(train_frame),
        predict_rows=len(test_frame),
        unique_smiles=int(train_frame["QSAR_READY_SMILES"].nunique()),
    )
    global_final = run_unimol_predict(
        train_rows=train_frame,
        predict_smiles=test_frame["QSAR_READY_SMILES"].astype(str).tolist(),
        model_dir=output_dir / "unimol_models" / "global_final",
        args=args,
    )
    progress.finish(label, global_final)
    test_pred["unimol_global_pred"] = global_final.predictions

    for item in group_plan:
        group_label = item["group_label"]
        group_train = item["group_train"]
        group_test = item["group_test"]
        unique_smiles = int(item["unique_smiles"])
        viable = bool(item["viable"])
        folds = item["folds"]
        group_status = "trained" if viable else "fallback_global"

        if viable:
            for fold_index, (fit_idx, valid_idx) in enumerate(folds, start=1):
                fit_rows = group_train.reset_index(drop=True).iloc[fit_idx]
                valid_rows = group_train.reset_index(drop=True).iloc[valid_idx]
                label = f"group OOF {group_label} fold {fold_index}/{len(folds)}"
                progress.start(
                    label,
                    train_rows=len(fit_rows),
                    predict_rows=len(valid_rows),
                    unique_smiles=int(fit_rows["QSAR_READY_SMILES"].nunique()),
                )
                result = run_unimol_predict(
                    train_rows=fit_rows,
                    predict_smiles=valid_rows["QSAR_READY_SMILES"].astype(str).tolist(),
                    model_dir=output_dir / "unimol_models" / "group_oof" / safe_name(group_label) / f"fold_{fold_index}",
                    args=args,
                )
                progress.finish(label, result)
                train_pred.loc[valid_rows.index.map(lambda i: group_train.index[i]), "unimol_group_oof"] = result.predictions
            if len(group_test):
                label = f"group final {group_label}"
                progress.start(
                    label,
                    train_rows=len(group_train),
                    predict_rows=len(group_test),
                    unique_smiles=unique_smiles,
                )
                final_result = run_unimol_predict(
                    train_rows=group_train,
                    predict_smiles=group_test["QSAR_READY_SMILES"].astype(str).tolist(),
                    model_dir=output_dir / "unimol_models" / "group_final" / safe_name(group_label),
                    args=args,
                )
                progress.finish(label, final_result)
                test_pred.loc[group_test.index, "unimol_group_pred"] = final_result.predictions
        else:
            status.warn(
                f"{group_label}: using global Uni-Mol fallback "
                f"(train_rows={len(group_train):,}, unique_smiles={unique_smiles:,}, test_rows={len(group_test):,}; "
                f"thresholds rows>={int(args.min_group_rows):,}, unique_smiles>={int(args.min_group_unique_smiles):,})."
            )

        group_rows.append(
            {
                "unimol_group": group_label,
                "status": group_status,
                "train_rows": int(len(group_train)),
                "train_unique_smiles": unique_smiles,
                "test_rows": int(len(group_test)),
                "oof_folds": int(len(folds)),
            }
        )

    train_pred["unimol_stack_pred"] = train_pred["unimol_group_oof"].where(
        train_pred["unimol_group_oof"].notna(),
        train_pred["unimol_global_oof"],
    )
    train_pred["unimol_group_available"] = train_pred["unimol_group_oof"].notna().astype(int)
    test_pred["unimol_stack_pred"] = test_pred["unimol_group_pred"].where(
        test_pred["unimol_group_pred"].notna(),
        test_pred["unimol_global_pred"],
    )
    test_pred["unimol_group_available"] = test_pred["unimol_group_pred"].notna().astype(int)
    return train_pred, test_pred, pd.DataFrame(group_rows)


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return safe[:120] or "unknown"


def make_fusion_estimator(args: argparse.Namespace) -> Any:
    if str(args.fusion_model) == "catboost":
        try:
            from catboost import CatBoostRegressor

            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        CatBoostRegressor(
                            loss_function="RMSE",
                            iterations=600,
                            learning_rate=0.04,
                            depth=6,
                            random_seed=int(args.random_seed),
                            verbose=False,
                            allow_writing_files=False,
                        ),
                    ),
                ]
            )
        except Exception:
            pass
    if str(args.fusion_model) == "ridge":
        from sklearn.linear_model import RidgeCV

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 17))),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=600,
                    min_samples_leaf=2,
                    random_state=int(args.random_seed),
                    n_jobs=-1,
                ),
            ),
        ]
    )


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else np.nan,
    }


def run(args: argparse.Namespace) -> None:
    status = StatusPrinter(use_color=color_enabled(args))
    output_dir = args.output_dir.resolve()
    complete_path = output_dir / "run_complete.json"
    if output_dir.exists() and bool(args.force):
        shutil.rmtree(output_dir)
    if complete_path.exists() and bool(args.resume) and not bool(args.force):
        status.ok(f"Existing completed fusion run: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    group_prefixes = args.group_prefix or ["species_", "sex_"]
    frame = add_group_labels(load_sheet(args.workbook, args.sheet), group_prefixes)
    aux_columns = numeric_auxiliary_columns(frame)
    train_mask = frame["split"].astype(str).str.lower().eq("train")
    test_mask = frame["split"].astype(str).str.lower().eq("test")
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("The selected sheet must contain both train and test rows.")

    status.section(f"Fusion dataset 1/1: {args.sheet}")
    status.info(
        f"Rows: total={len(frame):,}, train={len(train_frame):,}, test={len(test_frame):,}; "
        f"auxiliary_features={len(aux_columns):,}; groups={frame['unimol_group'].nunique():,}; "
        f"output={output_dir}"
    )

    train_pred, test_pred, group_manifest = build_oof_predictions(
        train_frame=train_frame,
        test_frame=test_frame,
        output_dir=output_dir,
        args=args,
        status=status,
    )
    if train_pred["unimol_stack_pred"].isna().any() or test_pred["unimol_stack_pred"].isna().any():
        raise RuntimeError("Missing Uni-Mol stack predictions after OOF/final prediction generation.")

    fusion_train = pd.concat(
        [
            train_frame[aux_columns].reset_index(drop=True),
            train_pred[["unimol_global_oof", "unimol_group_oof", "unimol_stack_pred", "unimol_group_available"]].reset_index(drop=True),
        ],
        axis=1,
    )
    fusion_test = pd.concat(
        [
            test_frame[aux_columns].reset_index(drop=True),
            test_pred[["unimol_global_pred", "unimol_group_pred", "unimol_stack_pred", "unimol_group_available"]]
            .rename(columns={"unimol_global_pred": "unimol_global_oof", "unimol_group_pred": "unimol_group_oof"})
            .reset_index(drop=True),
        ],
        axis=1,
    )
    fusion_test = fusion_test.reindex(columns=fusion_train.columns)

    estimator = make_fusion_estimator(args)
    status.section(f"Training fusion meta-model: {type(estimator).__name__}")
    estimator.fit(fusion_train, train_frame["TARGET_log10"].astype(float))
    train_fusion_pred = np.asarray(estimator.predict(fusion_train), dtype=float).reshape(-1)
    test_fusion_pred = np.asarray(estimator.predict(fusion_test), dtype=float).reshape(-1)

    train_metrics = regression_metrics(train_frame["TARGET_log10"], train_fusion_pred)
    test_metrics = regression_metrics(test_frame["TARGET_log10"], test_fusion_pred)
    metrics = pd.DataFrame(
        [
            {
                "model": "Uni-Mol V1 + Aux Fusion (Grouped)",
                "workflow": "Uni-Mol grouped aux fusion",
                "sheet": args.sheet,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "auxiliary_feature_count": int(len(aux_columns)),
                "trained_group_unimol_models": int(group_manifest["status"].eq("trained").sum()),
                "fallback_group_count": int(group_manifest["status"].eq("fallback_global").sum()),
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_r2": train_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
            }
        ]
    )

    prediction_rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "train",
                    "row_index": train_frame.index,
                    "QSAR_READY_SMILES": train_frame["QSAR_READY_SMILES"],
                    "unimol_group": train_frame["unimol_group"],
                    "observed": train_frame["TARGET_log10"].astype(float),
                    "unimol_global_pred": train_pred["unimol_global_oof"],
                    "unimol_group_pred": train_pred["unimol_group_oof"],
                    "unimol_stack_pred": train_pred["unimol_stack_pred"],
                    "fusion_pred": train_fusion_pred,
                }
            ),
            pd.DataFrame(
                {
                    "split": "test",
                    "row_index": test_frame.index,
                    "QSAR_READY_SMILES": test_frame["QSAR_READY_SMILES"],
                    "unimol_group": test_frame["unimol_group"],
                    "observed": test_frame["TARGET_log10"].astype(float),
                    "unimol_global_pred": test_pred["unimol_global_pred"],
                    "unimol_group_pred": test_pred["unimol_group_pred"],
                    "unimol_stack_pred": test_pred["unimol_stack_pred"],
                    "fusion_pred": test_fusion_pred,
                }
            ),
        ],
        ignore_index=True,
    )

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    prediction_rows.to_csv(output_dir / "predictions.csv", index=False)
    group_manifest.to_csv(output_dir / "group_manifest.csv", index=False)
    pd.DataFrame({"auxiliary_feature": aux_columns}).to_csv(output_dir / "auxiliary_features.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "workbook": str(args.workbook),
                "sheet": str(args.sheet),
                "group_prefixes": list(group_prefixes),
                "min_group_rows": int(args.min_group_rows),
                "min_group_unique_smiles": int(args.min_group_unique_smiles),
                "oof_folds": int(args.oof_folds),
                "fusion_model": str(args.fusion_model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    complete_path.write_text(json.dumps({"status": "completed", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
    status.ok("Fusion meta-model evaluation complete.")
    print(metrics.to_string(index=False), flush=True)
    status.ok(f"Fusion outputs written to: {output_dir}")


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
