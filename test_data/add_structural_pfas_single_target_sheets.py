"""Add derived PFAS workbook sheets for structural and aux-fusion benchmarks.

The benchmark workbook contains structural PFAS sheets with auxiliary physiology,
species, sex, and route columns. This script derives SMILES-only target sheets
for each species/sex subset so graph and 3D workflows can run without auxiliary
variables. It writes PFAS-only, non-PFAS-only, and combined structural sheets.
The combined sheets intentionally concatenate only the PFAS structural and
non-PFAS structural source sheets; TSCA-only source sheets are not combined.

It also writes one all-data auxiliary sheet per target. These sheets retain all
numeric auxiliary columns and combine PFAS TSCA, PFAS structural, non-PFAS TSCA,
and non-PFAS structural rows for grouped Uni-Mol plus auxiliary-feature fusion.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


TARGET_CONFIGS = {
    "hle_invivo": {
        "target": "hle_invivo",
        "all_aux_sheet": "HLe_invivo_all_aux",
        "pfas_tsca_sheet": "HLe_invivo_pfas_tsca",
        "nonpfas_tsca_sheet": "HLe_invivo_nonpfas_tsca",
        "pfas_sheet": "HLe_invivo_pfas_struct",
        "nonpfas_sheet": "HLe_invivo_nonpfas_struct",
    },
    "vdss": {
        "target": "vdss",
        "all_aux_sheet": "VDss_all_aux",
        "pfas_tsca_sheet": "VDss_pfas_tsca",
        "nonpfas_tsca_sheet": "VDss_nonpfas_tsca",
        "pfas_sheet": "VDss_pfas_struct",
        "nonpfas_sheet": "VDss_nonpfas_struct",
    },
}
SUBSET_LABELS = {
    "pfas": "pfas_only",
    "nonpfas": "non_pfas_only",
    "struct": "combined",
}
REQUIRED_COLUMNS = {"target", "smiles_subset", "split", "QSAR_READY_SMILES", "TARGET_log10"}
DEFAULT_WORKBOOK = Path(__file__).resolve().parents[1] / "data" / "modeling_datasets_aux_features.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help="Workbook to update in place. Defaults to data/modeling_datasets_aux_features.xlsx.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Deterministic holdout fraction for collapsed SMILES rows.",
    )
    return parser.parse_args()


def clean_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return token or "unknown"


def stable_score(text: str) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:12], 16)


def assign_deterministic_split(smiles: pd.Series, test_fraction: float) -> pd.Series:
    values = smiles.astype(str).reset_index(drop=True)
    n_rows = len(values)
    if n_rows <= 1:
        return pd.Series(["train"] * n_rows)

    fraction = min(max(float(test_fraction), 0.05), 0.5)
    test_n = max(1, int(round(n_rows * fraction)))
    test_n = min(test_n, n_rows - 1)
    scores = values.map(stable_score)
    test_indices = set(scores.sort_values(ascending=False).head(test_n).index.tolist())
    return pd.Series(["test" if idx in test_indices else "train" for idx in range(n_rows)])


def derive_sheet(
    frame: pd.DataFrame,
    target_name: str,
    subset_name: str,
    species_column: str,
    sex_column: str,
    test_fraction: float,
) -> pd.DataFrame:
    species_mask = pd.to_numeric(frame[species_column], errors="coerce").fillna(0) > 0
    sex_mask = pd.to_numeric(frame[sex_column], errors="coerce").fillna(0) > 0
    subset = frame.loc[species_mask & sex_mask, ["QSAR_READY_SMILES", "TARGET_log10", "target"]].copy()
    subset["TARGET_log10"] = pd.to_numeric(subset["TARGET_log10"], errors="coerce")
    subset = subset.dropna(subset=["QSAR_READY_SMILES", "TARGET_log10"])
    subset["QSAR_READY_SMILES"] = subset["QSAR_READY_SMILES"].astype(str).str.strip()
    subset = subset[subset["QSAR_READY_SMILES"] != ""]

    collapsed = (
        subset.groupby("QSAR_READY_SMILES", as_index=False, sort=True)
        .agg(TARGET_log10=("TARGET_log10", "mean"))
        .reset_index(drop=True)
    )
    species = clean_token(species_column.removeprefix("species_"))
    sex = clean_token(sex_column.removeprefix("sex_"))
    collapsed.insert(0, "target", target_name)
    subset_label = SUBSET_LABELS.get(subset_name, subset_name)
    collapsed.insert(1, "smiles_subset", f"{subset_label}_structural_{species}_{sex}")
    collapsed.insert(2, "split", assign_deterministic_split(collapsed["QSAR_READY_SMILES"], test_fraction))
    return collapsed[["target", "smiles_subset", "split", "QSAR_READY_SMILES", "TARGET_log10"]]


def ensure_structural_source(frame: pd.DataFrame, target_key: str, subset_key: str) -> None:
    """Guard against accidentally building combined sheets from TSCA-only inputs."""
    if "smiles_subset" not in frame.columns:
        return
    subset_values = frame["smiles_subset"].fillna("").astype(str).str.lower()
    if subset_values.str.contains("tsca", regex=False).any():
        raise ValueError(
            f"{target_key}:{subset_key} contains TSCA-only rows. "
            "Combined non-auxiliary sheets must use only structural source sheets."
        )


def sheet_name_for(target_key: str, subset_key: str, species_column: str, sex_column: str) -> str:
    species = clean_token(str(species_column).removeprefix("species_"))
    sex = clean_token(str(sex_column).removeprefix("sex_"))
    prefix = target_key if subset_key == "pfas" else f"{target_key}_{subset_key}"
    return f"{prefix}_{species}_{sex}"[:31]


def manifest_rows(sheet_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sheet_name, frame in sheet_frames.items():
        rows.append(
            {
                "sheet": sheet_name,
                "target": str(frame["target"].dropna().iloc[0]) if not frame.empty else "",
                "smiles_subset": str(frame["smiles_subset"].dropna().iloc[0]) if not frame.empty else "",
                "rows": int(len(frame)),
                "train_rows": int(frame["split"].astype(str).str.lower().eq("train").sum()) if "split" in frame else 0,
                "test_rows": int(frame["split"].astype(str).str.lower().eq("test").sum()) if "split" in frame else 0,
                "auxiliary_feature_columns": ", ".join(auxiliary_feature_columns(frame)),
            }
        )
    return pd.DataFrame(rows)


def auxiliary_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"target", "smiles_subset", "split", "QSAR_READY_SMILES", "TARGET_log10", "source_sheet"}
    columns: list[str] = []
    for column in frame.columns:
        if str(column) in excluded:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            columns.append(str(column))
    return columns


def fill_missing_numeric_aux_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in auxiliary_feature_columns(frame):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if str(column).startswith(("species_", "sex_", "route_")):
            frame[column] = frame[column].fillna(0)
    return frame


def derive_all_aux_sheets(workbook: Path, xls: pd.ExcelFile, test_fraction: float) -> dict[str, pd.DataFrame]:
    derived: dict[str, pd.DataFrame] = {}
    for target_key, config in TARGET_CONFIGS.items():
        source_keys = ["pfas_tsca_sheet", "pfas_sheet", "nonpfas_tsca_sheet", "nonpfas_sheet"]
        missing = [str(config[key]) for key in source_keys if str(config[key]) not in xls.sheet_names]
        if missing:
            raise ValueError(f"Workbook is missing source sheet(s): {', '.join(missing)}")

        frames: list[pd.DataFrame] = []
        for key in source_keys:
            sheet_name = str(config[key])
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
            if missing_columns:
                raise ValueError(f"{target_key}:{sheet_name} is missing required column(s): {', '.join(missing_columns)}")
            frame = frame.copy()
            frame["source_sheet"] = sheet_name
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["target"] = str(config["target"])
        combined["smiles_subset"] = f"{target_key}_all_aux"
        combined["QSAR_READY_SMILES"] = combined["QSAR_READY_SMILES"].astype(str).str.strip()
        combined["TARGET_log10"] = pd.to_numeric(combined["TARGET_log10"], errors="coerce")
        combined = combined.dropna(subset=["QSAR_READY_SMILES", "TARGET_log10"])
        combined = combined[combined["QSAR_READY_SMILES"] != ""].reset_index(drop=True)
        combined = fill_missing_numeric_aux_columns(combined)
        combined["split"] = assign_deterministic_split(combined["QSAR_READY_SMILES"], test_fraction)

        ordered = ["target", "smiles_subset", "split", "QSAR_READY_SMILES", "TARGET_log10", "source_sheet"]
        remaining = [column for column in combined.columns if column not in ordered]
        combined = combined[ordered + remaining].drop_duplicates().reset_index(drop=True)
        derived[str(config["all_aux_sheet"])] = combined
    return derived


def derive_structural_sheets(workbook: Path, xls: pd.ExcelFile, test_fraction: float) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(workbook)
    required_sheets = [
        str(config[sheet_key])
        for config in TARGET_CONFIGS.values()
        for sheet_key in ["pfas_sheet", "nonpfas_sheet"]
    ]
    missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
    if missing_sheets:
        raise ValueError(f"Workbook is missing source sheet(s): {', '.join(missing_sheets)}")

    derived: dict[str, pd.DataFrame] = {}
    for target_key, config in TARGET_CONFIGS.items():
        target_name = str(config["target"])
        source_frames = {
            "pfas": pd.read_excel(workbook, sheet_name=str(config["pfas_sheet"])),
            "nonpfas": pd.read_excel(workbook, sheet_name=str(config["nonpfas_sheet"])),
        }
        for subset_key, frame in source_frames.items():
            ensure_structural_source(frame, target_key, subset_key)
        source_frames["struct"] = pd.concat([source_frames["pfas"], source_frames["nonpfas"]], ignore_index=True, sort=False)

        for subset_key, frame in source_frames.items():
            missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
            if missing_columns:
                raise ValueError(f"{target_key}:{subset_key} is missing required column(s): {', '.join(missing_columns)}")

            species_columns = [column for column in frame.columns if str(column).startswith("species_")]
            sex_columns = [column for column in frame.columns if str(column).startswith("sex_")]
            for species_column in species_columns:
                for sex_column in sex_columns:
                    sheet = derive_sheet(frame, target_name, subset_key, str(species_column), str(sex_column), test_fraction)
                    if sheet.empty:
                        continue
                    derived[sheet_name_for(target_key, subset_key, str(species_column), str(sex_column))] = sheet
    return derived


def derive_all_sheets(workbook: Path, test_fraction: float) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(workbook)
    derived = derive_structural_sheets(workbook, xls, test_fraction)
    derived.update(derive_all_aux_sheets(workbook, xls, test_fraction))
    return derived


def update_workbook(workbook: Path, derived: dict[str, pd.DataFrame]) -> None:
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    if not derived:
        raise ValueError("No derived sheets were created.")

    try:
        existing_manifest = pd.read_excel(workbook, sheet_name="manifest")
    except ValueError:
        existing_manifest = pd.DataFrame(columns=["sheet", "target", "smiles_subset", "rows", "train_rows", "test_rows", "auxiliary_feature_columns"])

    new_manifest = manifest_rows(derived)
    if "sheet" in existing_manifest.columns:
        existing_manifest = existing_manifest[~existing_manifest["sheet"].astype(str).isin(new_manifest["sheet"].astype(str))]
    updated_manifest = pd.concat([existing_manifest, new_manifest], ignore_index=True)

    with pd.ExcelWriter(workbook, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for sheet_name, frame in derived.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        updated_manifest.to_excel(writer, sheet_name="manifest", index=False)


def main() -> int:
    args = parse_args()
    workbook = args.workbook.resolve()
    derived = derive_all_sheets(workbook, args.test_fraction)
    update_workbook(workbook, derived)
    print(f"Updated workbook: {workbook}")
    print(f"Derived/updated sheets: {len(derived)}")
    for sheet_name, frame in derived.items():
        print(
            f"  - {sheet_name}: {len(frame)} rows "
            f"({frame['split'].eq('train').sum()} train, {frame['split'].eq('test').sum()} test)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
