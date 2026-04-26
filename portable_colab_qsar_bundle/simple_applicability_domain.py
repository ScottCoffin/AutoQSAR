#!/usr/bin/env python
"""
Applicability domain assessment for a query chemical SMILES.

This script supports:

1. Lightweight feature-space AD diagnostics:
   - kNN mean distance
   - Ledoit-Wolf Mahalanobis distance

2. Optional MAST-ML/MADML applicability-domain workflow:
   - mastml.domain.Domain("madml")
   - internal RandomForestRegressor
   - MADML domain rules

Example PowerShell usage:

python portable_colab_qsar_bundle/simple_applicability_domain.py `
  --train_csv data_pod_nc.csv `
  --smiles_col SMILES `
  --target_col POD `
  --query_smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" `
  --use_mastml `
  --output_csv ad_result.csv `
  --metadata_json ad_metadata.json
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# -----------------------------------------------------------------------------
# SMILES + feature generation
# -----------------------------------------------------------------------------

def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mol_to_features(
    smiles: str,
    radius: int = 2,
    n_bits: int = 1024,
    include_descriptors: bool = True,
    include_fingerprint: bool = True,
) -> pd.Series:
    """
    Convert a SMILES string to a feature vector.

    Important:
    For production QSAR work, this feature builder should match the exact
    feature space used by the fitted conventional ML model.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    features = {}

    if include_descriptors:
        features.update({
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
        })

    if include_fingerprint:
        generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)

        arr = np.zeros((n_bits,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)

        for i, value in enumerate(arr):
            features[f"morgan_{radius}_{n_bits}_{i}"] = int(value)

    return pd.Series(features, dtype=float)


def build_feature_matrix(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 1024,
    include_descriptors: bool = True,
    include_fingerprint: bool = True,
) -> pd.DataFrame:
    """Build a feature matrix from a list of SMILES."""
    rows = []
    valid_smiles = []

    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)

        if canonical is None:
            warnings.warn(f"Skipping invalid SMILES: {smiles}")
            continue

        rows.append(
            mol_to_features(
                canonical,
                radius=radius,
                n_bits=n_bits,
                include_descriptors=include_descriptors,
                include_fingerprint=include_fingerprint,
            )
        )
        valid_smiles.append(canonical)

    if not rows:
        raise ValueError("No valid SMILES were available for featurization.")

    X = pd.DataFrame(rows)
    X.insert(0, "canonical_smiles", valid_smiles)

    return X


def clean_numeric_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Convert feature matrix to numeric, impute missing values, and replace infinities."""
    X_clean = X.copy()
    X_clean = X_clean.apply(pd.to_numeric, errors="coerce")
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    medians = X_clean.median(numeric_only=True)
    X_clean = X_clean.fillna(medians)
    X_clean = X_clean.fillna(0.0)

    return X_clean


def align_query_to_training(
    X_query: pd.DataFrame,
    training_columns: list[str],
    training_medians: dict,
) -> pd.DataFrame:
    """
    Align query features to training columns.

    Missing columns are added.
    Extra columns are dropped.
    Missing values are imputed using training medians.
    """
    X = X_query.copy()

    for col in training_columns:
        if col not in X.columns:
            X[col] = np.nan

    X = X[training_columns]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in training_columns:
        X[col] = X[col].fillna(training_medians.get(col, 0.0))

    X = X.fillna(0.0)

    return X


# -----------------------------------------------------------------------------
# Lightweight AD diagnostics
# -----------------------------------------------------------------------------

def fit_feature_space_ad_reference(
    X_train: pd.DataFrame,
    knn_neighbors: int = 5,
    knn_quantile: float = 0.95,
    mahalanobis_quantile: float = 0.975,
) -> dict:
    """
    Fit kNN and Mahalanobis AD reference objects and thresholds.
    """
    X_train_numeric = clean_numeric_feature_matrix(X_train)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train_numeric)
    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    n_train = train_scaled.shape[0]

    if n_train < 3:
        raise ValueError("At least 3 valid training chemicals are recommended for AD assessment.")

    k = max(1, min(int(knn_neighbors), n_train - 1))

    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(train_scaled)

    train_distances, _ = knn.kneighbors(train_scaled, n_neighbors=k + 1)

    # First neighbor is the molecule itself.
    train_knn_mean = np.mean(train_distances[:, 1:], axis=1)
    knn_threshold = float(np.quantile(train_knn_mean, float(knn_quantile)))

    cov_model = LedoitWolf()
    cov_model.fit(train_scaled)

    train_mahalanobis = cov_model.mahalanobis(train_scaled)
    mahalanobis_threshold = float(np.quantile(train_mahalanobis, float(mahalanobis_quantile)))

    return {
        "scaler": scaler,
        "training_columns": list(X_train_numeric.columns),
        "training_medians": X_train_numeric.median(numeric_only=True).to_dict(),
        "knn": knn,
        "knn_neighbors_used": k,
        "knn_threshold": knn_threshold,
        "knn_quantile": float(knn_quantile),
        "cov_model": cov_model,
        "mahalanobis_threshold": mahalanobis_threshold,
        "mahalanobis_quantile": float(mahalanobis_quantile),
    }


def assess_feature_space_ad(
    X_query: pd.DataFrame,
    ad_reference: dict,
) -> dict:
    """Compute kNN and Mahalanobis AD diagnostics for one query row."""
    X_query_aligned = align_query_to_training(
        X_query,
        ad_reference["training_columns"],
        ad_reference["training_medians"],
    )

    query_scaled = ad_reference["scaler"].transform(X_query_aligned)
    query_scaled = np.nan_to_num(query_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pred_distances, _ = ad_reference["knn"].kneighbors(
        query_scaled,
        n_neighbors=ad_reference["knn_neighbors_used"],
    )

    knn_mean_distance = float(np.mean(pred_distances[0]))
    knn_in_domain = bool(knn_mean_distance <= ad_reference["knn_threshold"])

    mahalanobis_distance = float(ad_reference["cov_model"].mahalanobis(query_scaled)[0])
    mahalanobis_in_domain = bool(
        mahalanobis_distance <= ad_reference["mahalanobis_threshold"]
    )

    return {
        "ad_knn_in_domain": knn_in_domain,
        "ad_knn_mean_distance": knn_mean_distance,
        "ad_knn_train_threshold": ad_reference["knn_threshold"],
        "ad_knn_neighbors_used": ad_reference["knn_neighbors_used"],
        "ad_knn_train_quantile": ad_reference["knn_quantile"],

        "ad_mahalanobis_in_domain": mahalanobis_in_domain,
        "ad_mahalanobis_distance": mahalanobis_distance,
        "ad_mahalanobis_train_threshold": ad_reference["mahalanobis_threshold"],
        "ad_mahalanobis_train_quantile": ad_reference["mahalanobis_quantile"],
    }


# -----------------------------------------------------------------------------
# MAST-ML / MADML support
# -----------------------------------------------------------------------------

class MastmlPreprocessorShim:
    """
    Minimal shim expected by MAST-ML Domain("madml").

    This mirrors the notebook approach:
        class _MastmlPreprocessorShim:
            def __init__(self):
                self.preprocessor = StandardScaler()
    """
    def __init__(self):
        self.preprocessor = StandardScaler()


class MastmlModelShim:
    """
    Minimal model shim expected by MAST-ML Domain("madml").

    This mirrors the notebook approach:
        self.model = RandomForestRegressor(...)
    """
    def __init__(self, n_estimators: int = 300, random_state: int = 42, n_jobs: int = 1):
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )


def ensure_mastml_packages(install_if_missing: bool = False):
    """
    Import MAST-ML and MADML.

    If install_if_missing=True, this attempts pip installation first.
    Source-based fallback is intentionally not included here because it can be
    brittle across Windows/Conda environments.
    """
    try:
        from mastml.domain import Domain  # noqa: F401
        import madml  # noqa: F401
        return
    except Exception as first_error:
        if not install_if_missing:
            raise RuntimeError(
                "MAST-ML/MADML are not available in this Python environment. "
                "Install them first, or rerun with --install_mastml_if_missing. "
                f"Original import error: {type(first_error).__name__}: {first_error}"
            )

    print("MAST-ML/MADML not found. Attempting pip installation...")

    for package in ["madml", "mastml"]:
        cmd = [sys.executable, "-m", "pip", "install", package]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            raise RuntimeError(f"pip install failed for {package}")

    try:
        from mastml.domain import Domain  # noqa: F401
        import madml  # noqa: F401
        return
    except Exception as second_error:
        raise RuntimeError(
            "MAST-ML/MADML still could not be imported after pip installation. "
            "This may mean the current shell and Python kernel/environment differ, "
            "or that MAST-ML/MADML dependencies are incompatible with this environment. "
            f"Import error: {type(second_error).__name__}: {second_error}"
        )


def fit_predict_mastml_madml_ad(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_query: pd.DataFrame,
    output_dir: str | Path,
    n_estimators: int = 300,
    n_repeats: int = 1,
    bins: int = 10,
    kernel: str = "epanechnikov",
    bandwidth: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit and apply MAST-ML/MADML AD workflow.

    Returns a one-row DataFrame of MAST-ML/MADML outputs, prefixed with mastml_.
    """
    from mastml.domain import Domain

    X_train_clean = clean_numeric_feature_matrix(X_train)
    X_query_aligned = align_query_to_training(
        X_query,
        list(X_train_clean.columns),
        X_train_clean.median(numeric_only=True).to_dict(),
    )

    y = pd.Series(y_train).astype(float)
    valid_y = y.replace([np.inf, -np.inf], np.nan).notna()

    if int(valid_y.sum()) < 5:
        raise ValueError(
            "MAST-ML/MADML AD requires a usable training target column. "
            "Fewer than 5 non-missing numeric target values were found."
        )

    X_train_clean = X_train_clean.loc[valid_y.values, :].reset_index(drop=True)
    y = y.loc[valid_y.values].reset_index(drop=True)

    params = {
        "n_repeats": int(n_repeats),
        "bins": int(bins),
        "kernel": str(kernel),
    }

    if bandwidth is not None:
        params["bandwidth"] = float(bandwidth)

    preprocessor = MastmlPreprocessorShim()
    model = MastmlModelShim(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=1,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting MAST-ML/MADML applicability-domain model...")
    print(f"Training rows: {X_train_clean.shape[0]}")
    print(f"Training features: {X_train_clean.shape[1]}")
    print(f"MADML params: {params}")

    ad_model = Domain(
        "madml",
        preprocessor=preprocessor,
        model=model,
        params=params,
        path=str(output_dir),
    )

    ad_model.fit(X_train_clean, y)

    print("Applying MAST-ML/MADML applicability-domain model to query molecule...")
    mastml_result = ad_model.predict(X_query_aligned)

    if not isinstance(mastml_result, pd.DataFrame):
        mastml_result = pd.DataFrame(mastml_result)

    mastml_result = mastml_result.reset_index(drop=True)

    # Prefix columns unless already prefixed.
    mastml_result = mastml_result.rename(
        columns={
            col: col if str(col).startswith("mastml_") else f"mastml_{col}"
            for col in mastml_result.columns
        }
    )

    mastml_result = rename_and_interpret_mastml_columns(mastml_result)

    return mastml_result


def rename_and_interpret_mastml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common MAST-ML/MADML outputs to readable column names and derive
    simple MADML concern labels.

    MAST-ML column names can vary slightly by version, so this uses pattern
    matching instead of requiring exact names.
    """
    out = df.copy()

    rename_map = {
        "mastml_y_pred": "mastml_internal_rf_prediction",
        "mastml_d_pred": "mastml_kde_dissimilarity",
        "mastml_y_stdu_pred": "mastml_internal_rf_uncertainty_raw",
        "mastml_y_stdc_pred": "mastml_internal_rf_uncertainty_calibrated",
    }

    for col in list(out.columns):
        col_str = str(col)

        if col_str in rename_map:
            continue

        if col_str.startswith("mastml_absres/mad_y Domain Prediction"):
            rename_map[col] = "mastml_ad_residual_rule"

        elif col_str.startswith("mastml_rmse/std_y Domain Prediction"):
            rename_map[col] = "mastml_ad_rmse_rule"

        elif col_str.startswith("mastml_cdf_area Domain Prediction"):
            rename_map[col] = "mastml_ad_calibration_area_rule"

    out = out.rename(columns=rename_map)

    domain_columns = [
        col for col in out.columns
        if "Domain Prediction" in str(col)
        or col in {
            "mastml_ad_residual_rule",
            "mastml_ad_rmse_rule",
            "mastml_ad_calibration_area_rule",
        }
    ]

    if domain_columns:
        out["mastml_in_domain_by_any_rule"] = out[domain_columns].apply(
            lambda row: any(str(value) == "ID" for value in row),
            axis=1,
        )

    ad_rule_columns = [
        col
        for col in [
            "mastml_ad_residual_rule",
            "mastml_ad_rmse_rule",
            "mastml_ad_calibration_area_rule",
        ]
        if col in out.columns
    ]

    if ad_rule_columns:
        out["mastml_id_rule_count"] = out[ad_rule_columns].apply(
            lambda row: int(sum(str(value) == "ID" for value in row)),
            axis=1,
        )

        out["mastml_overall_concern"] = out["mastml_id_rule_count"].map({
            3: "Low concern",
            2: "Moderate concern",
            1: "High concern",
            0: "High concern",
        })

        def interpret(row):
            residual_id = str(row.get("mastml_ad_residual_rule", "")) == "ID"
            rmse_id = str(row.get("mastml_ad_rmse_rule", "")) == "ID"
            calibration_id = str(row.get("mastml_ad_calibration_area_rule", "")) == "ID"
            concern = str(row.get("mastml_overall_concern", ""))

            if concern == "Low concern":
                return "All three MADML rules marked this molecule in-domain."

            if residual_id and rmse_id and not calibration_id:
                return (
                    "Prediction appears supported by similarity/error rules, "
                    "but the MADML uncertainty calibration rule flagged caution."
                )

            if residual_id or rmse_id or calibration_id:
                return (
                    "Only partial MADML support was found; use this prediction cautiously "
                    "and inspect chemistry-space proximity."
                )

            return "All MADML rules flagged this molecule as out-of-domain."

        out["mastml_interpretation"] = out.apply(interpret, axis=1)

    elif "mastml_in_domain_by_any_rule" in out.columns:
        out["mastml_overall_concern"] = np.where(
            out["mastml_in_domain_by_any_rule"].astype(bool),
            "Low concern",
            "High concern",
        )
        out["mastml_interpretation"] = np.where(
            out["mastml_in_domain_by_any_rule"].astype(bool),
            "At least one MADML rule marked this molecule in-domain.",
            "MADML rules did not mark this molecule in-domain.",
        )

    return out


# -----------------------------------------------------------------------------
# Consensus interpretation
# -----------------------------------------------------------------------------

def add_consensus_label(
    result: pd.DataFrame,
    low_concern_support_ratio: float = 0.75,
) -> pd.DataFrame:
    """
    Combine available AD method flags into a consensus concern label.
    """
    out = result.copy()

    method_flag_columns = []

    if "mastml_in_domain_by_any_rule" in out.columns:
        out["ad_mastml_in_domain"] = out["mastml_in_domain_by_any_rule"].fillna(False).astype(bool)
        method_flag_columns.append("ad_mastml_in_domain")

    if "ad_knn_in_domain" in out.columns:
        out["ad_knn_in_domain"] = out["ad_knn_in_domain"].fillna(False).astype(bool)
        method_flag_columns.append("ad_knn_in_domain")

    if "ad_mahalanobis_in_domain" in out.columns:
        out["ad_mahalanobis_in_domain"] = out["ad_mahalanobis_in_domain"].fillna(False).astype(bool)
        method_flag_columns.append("ad_mahalanobis_in_domain")

    if not method_flag_columns:
        out["ad_method_support_count"] = 0
        out["ad_method_available_count"] = 0
        out["ad_consensus_support_ratio"] = np.nan
        out["ad_consensus_concern"] = "Not assessed"
        out["ad_consensus_interpretation"] = "No AD methods were available."
        return out

    out["ad_method_support_count"] = out[method_flag_columns].astype(int).sum(axis=1)
    out["ad_method_available_count"] = len(method_flag_columns)
    out["ad_consensus_support_ratio"] = (
        out["ad_method_support_count"].astype(float)
        / np.maximum(out["ad_method_available_count"].astype(float), 1.0)
    )

    moderate_floor = 0.34
    low_floor = max(moderate_floor, min(1.0, float(low_concern_support_ratio)))

    def label(ratio):
        ratio = float(ratio)

        if ratio >= low_floor:
            return "Low concern"

        if ratio >= moderate_floor:
            return "Moderate concern"

        return "High concern"

    out["ad_consensus_concern"] = out["ad_consensus_support_ratio"].apply(label)

    def interpret(row):
        concern = str(row["ad_consensus_concern"])
        support = int(row["ad_method_support_count"])
        available = int(row["ad_method_available_count"])

        if concern == "Low concern":
            return f"{support}/{available} AD methods marked this molecule in-domain."

        if concern == "Moderate concern":
            return (
                f"{support}/{available} AD methods marked this molecule in-domain. "
                "Use caution and inspect nearest-neighbor chemistry."
            )

        return (
            f"{support}/{available} AD methods marked this molecule in-domain. "
            "Treat this as likely out-of-domain and prioritize experimental confirmation."
        )

    out["ad_consensus_interpretation"] = out.apply(interpret, axis=1)

    return out


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Applicability domain assessment for a query SMILES."
    )

    parser.add_argument(
        "--train_csv",
        required=True,
        help="CSV containing the training chemicals used to fit the conventional ML model.",
    )

    parser.add_argument(
        "--smiles_col",
        default="SMILES",
        help="Name of the SMILES column in the training CSV.",
    )

    parser.add_argument(
        "--target_col",
        default=None,
        help=(
            "Name of the training target column. Required when --use_mastml is enabled, "
            "because MADML requires y_train."
        ),
    )

    parser.add_argument(
        "--query_smiles",
        required=True,
        help="Query chemical SMILES to assess.",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius.",
    )

    parser.add_argument(
        "--n_bits",
        type=int,
        default=1024,
        help="Morgan fingerprint length.",
    )

    parser.add_argument(
        "--no_descriptors",
        action="store_true",
        help="Use only Morgan fingerprints; omit RDKit descriptors.",
    )

    parser.add_argument(
        "--no_fingerprint",
        action="store_true",
        help="Use only RDKit descriptors; omit Morgan fingerprints.",
    )

    parser.add_argument(
        "--knn_neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors for kNN AD.",
    )

    parser.add_argument(
        "--knn_quantile",
        type=float,
        default=0.95,
        help="Training quantile used as kNN in-domain threshold.",
    )

    parser.add_argument(
        "--mahalanobis_quantile",
        type=float,
        default=0.975,
        help="Training quantile used as Mahalanobis in-domain threshold.",
    )

    parser.add_argument(
        "--low_concern_support_ratio",
        type=float,
        default=0.75,
        help="Consensus support ratio required for Low concern.",
    )

    parser.add_argument(
        "--use_mastml",
        action="store_true",
        help="Enable MAST-ML/MADML applicability-domain workflow.",
    )

    parser.add_argument(
        "--install_mastml_if_missing",
        action="store_true",
        help="Attempt pip install of mastml and madml if unavailable.",
    )

    parser.add_argument(
        "--mastml_fail_soft",
        action="store_true",
        help=(
            "If MAST-ML/MADML fails, continue with kNN/Mahalanobis only instead "
            "of stopping the script."
        ),
    )

    parser.add_argument(
        "--mastml_output_dir",
        default=".cache/mastml_applicability_domain",
        help="Output/cache directory used by MAST-ML/MADML.",
    )

    parser.add_argument(
        "--mastml_random_forest_estimators",
        type=int,
        default=300,
        help="Number of trees for the internal MAST-ML random forest.",
    )

    parser.add_argument(
        "--mastml_n_repeats",
        type=int,
        default=1,
        help="MADML n_repeats parameter.",
    )

    parser.add_argument(
        "--mastml_bins",
        type=int,
        default=10,
        help="MADML bins parameter.",
    )

    parser.add_argument(
        "--mastml_kernel",
        default="epanechnikov",
        choices=["epanechnikov", "gaussian", "tophat", "exponential"],
        help="MADML kernel parameter.",
    )

    parser.add_argument(
        "--mastml_bandwidth",
        type=float,
        default=None,
        help="Optional custom MADML bandwidth.",
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save the AD result CSV.",
    )

    parser.add_argument(
        "--metadata_json",
        default=None,
        help="Optional path to save AD threshold metadata as JSON.",
    )

    args = parser.parse_args()

    include_descriptors = not args.no_descriptors
    include_fingerprint = not args.no_fingerprint

    if not include_descriptors and not include_fingerprint:
        raise ValueError("At least one feature type must be enabled.")

    train_df = pd.read_csv(args.train_csv)

    if args.smiles_col not in train_df.columns:
        raise ValueError(
            f"SMILES column '{args.smiles_col}' not found in training CSV. "
            f"Available columns: {list(train_df.columns)}"
        )

    if args.use_mastml:
        if args.target_col is None:
            raise ValueError(
                "--target_col is required when --use_mastml is enabled."
            )

        if args.target_col not in train_df.columns:
            raise ValueError(
                f"Target column '{args.target_col}' not found in training CSV. "
                f"Available columns: {list(train_df.columns)}"
            )

    train_smiles = train_df[args.smiles_col].dropna().astype(str).tolist()

    X_train_with_id = build_feature_matrix(
        train_smiles,
        radius=args.radius,
        n_bits=args.n_bits,
        include_descriptors=include_descriptors,
        include_fingerprint=include_fingerprint,
    )

    training_canonical_smiles = X_train_with_id["canonical_smiles"].copy()
    X_train = X_train_with_id.drop(columns=["canonical_smiles"])

    canonical_query = canonicalize_smiles(args.query_smiles)

    if canonical_query is None:
        result = pd.DataFrame([{
            "input_smiles": args.query_smiles,
            "canonical_smiles": None,
            "valid_smiles": False,
            "ad_consensus_concern": "Invalid SMILES",
            "ad_consensus_interpretation": "The input SMILES could not be parsed by RDKit.",
        }])
    else:
        X_query = pd.DataFrame([
            mol_to_features(
                canonical_query,
                radius=args.radius,
                n_bits=args.n_bits,
                include_descriptors=include_descriptors,
                include_fingerprint=include_fingerprint,
            )
        ])

        result = pd.DataFrame([{
            "input_smiles": args.query_smiles,
            "canonical_smiles": canonical_query,
            "valid_smiles": True,
        }])

        # Lightweight feature-space diagnostics
        feature_ad_reference = fit_feature_space_ad_reference(
            X_train,
            knn_neighbors=args.knn_neighbors,
            knn_quantile=args.knn_quantile,
            mahalanobis_quantile=args.mahalanobis_quantile,
        )

        feature_ad_result = assess_feature_space_ad(
            X_query,
            feature_ad_reference,
        )

        for key, value in feature_ad_result.items():
            result[key] = value

        # Optional MAST-ML / MADML diagnostics
        mastml_error = None

        if args.use_mastml:
            try:
                ensure_mastml_packages(
                    install_if_missing=args.install_mastml_if_missing
                )

                # Align y_train to the valid training SMILES rows retained by build_feature_matrix.
                # This assumes invalid SMILES are rare. For strict alignment, pre-clean invalid
                # rows before calling build_feature_matrix.
                train_df_valid = train_df.copy()
                train_df_valid["_canonical_smiles_for_ad"] = train_df_valid[args.smiles_col].apply(canonicalize_smiles)
                train_df_valid = train_df_valid.loc[
                    train_df_valid["_canonical_smiles_for_ad"].notna()
                ].reset_index(drop=True)

                y_train = pd.to_numeric(
                    train_df_valid[args.target_col],
                    errors="coerce",
                )

                if len(y_train) != len(X_train):
                    raise RuntimeError(
                        "Internal alignment error: y_train length does not match X_train length. "
                        "Check for invalid SMILES or duplicated preprocessing."
                    )

                mastml_result = fit_predict_mastml_madml_ad(
                    X_train=X_train,
                    y_train=y_train,
                    X_query=X_query,
                    output_dir=args.mastml_output_dir,
                    n_estimators=args.mastml_random_forest_estimators,
                    n_repeats=args.mastml_n_repeats,
                    bins=args.mastml_bins,
                    kernel=args.mastml_kernel,
                    bandwidth=args.mastml_bandwidth,
                )

                result = pd.concat(
                    [
                        result.reset_index(drop=True),
                        mastml_result.reset_index(drop=True),
                    ],
                    axis=1,
                )

            except Exception as exc:
                mastml_error = f"{type(exc).__name__}: {exc}"

                if not args.mastml_fail_soft:
                    raise

                result["mastml_error"] = mastml_error
                print(
                    "Warning: MAST-ML/MADML AD failed, but --mastml_fail_soft was set. "
                    "Continuing with kNN/Mahalanobis diagnostics only."
                )
                print(mastml_error)

        result = add_consensus_label(
            result,
            low_concern_support_ratio=args.low_concern_support_ratio,
        )

    print("\nApplicability domain result")
    print(result.to_string(index=False))

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"\nSaved AD result to: {output_path}")

    if args.metadata_json:
        metadata = {
            "n_training_chemicals": int(len(X_train)),
            "n_features": int(X_train.shape[1]),
            "feature_types": {
                "rdkit_descriptors": include_descriptors,
                "morgan_fingerprint": include_fingerprint,
            },
            "morgan_radius": args.radius,
            "morgan_n_bits": args.n_bits,
            "knn_neighbors": args.knn_neighbors,
            "knn_quantile": args.knn_quantile,
            "mahalanobis_quantile": args.mahalanobis_quantile,
            "low_concern_support_ratio": args.low_concern_support_ratio,
            "use_mastml": bool(args.use_mastml),
            "mastml_output_dir": str(args.mastml_output_dir),
            "mastml_random_forest_estimators": args.mastml_random_forest_estimators,
            "mastml_n_repeats": args.mastml_n_repeats,
            "mastml_bins": args.mastml_bins,
            "mastml_kernel": args.mastml_kernel,
            "mastml_bandwidth": args.mastml_bandwidth,
        }

        metadata_path = Path(args.metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved AD metadata to: {metadata_path}")


if __name__ == "__main__":
    main()