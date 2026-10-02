"""Load the deposited benchmark artifacts the meta-analysis is built from.

Nothing here re-splits or re-standardizes molecules. Train/test partitions come from the per-model
``predictions.csv`` rows the benchmark itself wrote, and are checked against the split-signature
hashes (``split_train_hash`` / ``split_test_hash`` in ``metrics.csv``). Those hashes are computed by
``run_qsarena_benchmarks.smiles_hash`` over the ordered, stripped SMILES, and that function is
reproduced exactly by :func:`smiles_hash` below.

``predictions.csv`` files are gitignored, so :func:`build_partitions` extracts a compact copy
(dataset, split, row_index, smiles, observed) to ``data/meta_analysis/dataset_partitions.csv.gz``.
That file is committed, and a clean checkout can run the whole analysis from it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "benchmark_results" / "qsarena_benchmark_oof_ensemble"
COMPARISON_RUN_DIR = REPO_ROOT / "benchmark_results" / "benchmark_name_date"
PARTITIONS_PATH = REPO_ROOT / "data" / "meta_analysis" / "dataset_partitions.csv.gz"
MANUSCRIPT_ASSETS = REPO_ROOT / "manuscript_assets"
FAMILY_BEST_PATH = MANUSCRIPT_ASSETS / "tables" / "figure6_family_best_models.csv"
FIG6_MATRIX_PATH = MANUSCRIPT_ASSETS / "tables" / "figure6_family_gap_matrix.csv"
MANUSCRIPT_NUMBERS_PATH = MANUSCRIPT_ASSETS / "manuscript_numbers.json"

PARTITION_COLUMNS = ["dataset", "split", "row_index", "smiles", "observed"]
SUMMARY_COLUMNS = [
    "dataset",
    "n_train",
    "n_test",
    "split_strategy",
    "primary_metric",
    "target_transform",
    "split_train_hash",
    "split_test_hash",
]


def smiles_hash(smiles_values) -> str:
    """Same digest as ``run_qsarena_benchmarks.smiles_hash`` (sha256 of newline-joined, stripped SMILES)."""
    return hashlib.sha256("\n".join(str(s).strip() for s in smiles_values).encode("utf-8")).hexdigest()


def dataset_dirs(run_dir: Path | str = DEFAULT_RUN_DIR) -> list[Path]:
    return sorted(p.parent for p in Path(run_dir).glob("*/metrics.csv"))


def _first_non_empty(series: pd.Series):
    values = series.dropna()
    values = values[values.astype(str).str.strip() != ""]
    return values.iloc[0] if len(values) else np.nan


def load_dataset_summary(run_dir: Path | str = DEFAULT_RUN_DIR) -> pd.DataFrame:
    """One row per dataset with size, split protocol, primary metric and split-signature hashes."""
    rows = []
    for d in dataset_dirs(run_dir):
        frame = pd.read_csv(d / "metrics.csv", usecols=lambda c: c in set(SUMMARY_COLUMNS) - {"dataset"})
        record = {"dataset": d.name}
        for column in SUMMARY_COLUMNS[1:]:
            record[column] = _first_non_empty(frame[column]) if column in frame.columns else np.nan
        rows.append(record)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for column in ("n_train", "n_test"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce").astype("Int64")
    return summary


def build_partitions(run_dir: Path | str = DEFAULT_RUN_DIR, out_path: Path | str = PARTITIONS_PATH) -> pd.DataFrame:
    """Extract each dataset's train/test SMILES and targets from its ``predictions.csv``.

    The rows of the first model that has both train and test predictions are used, and they are
    streamed with the csv module so multi-hundred-MB prediction files never sit in memory.
    Raises if a partition does not reproduce the recorded split-signature hash.
    """
    csv.field_size_limit(2**31 - 1)
    summary = load_dataset_summary(run_dir).set_index("dataset")
    frames = []
    for d in dataset_dirs(run_dir):
        rows: dict[str, list] = {"train": [], "test": []}
        model = None
        with open(d / "predictions.csv", newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                split = record["split"]
                if split not in rows:
                    continue
                if model is None:
                    model = record["model"]
                if record["model"] != model:
                    if rows["train"] and rows["test"]:
                        break
                    continue
                rows[split].append((int(record["row_index"]), record["smiles"].strip(), float(record["observed"])))
        for split, values in rows.items():
            values.sort()
            frames.append(
                pd.DataFrame(
                    {
                        "dataset": d.name,
                        "split": split,
                        "row_index": [v[0] for v in values],
                        "smiles": [v[1] for v in values],
                        "observed": [v[2] for v in values],
                    }
                )
            )
    partitions = pd.concat(frames, ignore_index=True)[PARTITION_COLUMNS]
    mismatches = verify_partition_hashes(partitions, summary.reset_index())
    if mismatches:
        raise ValueError(f"partitions do not match the recorded split hashes: {mismatches}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partitions.to_csv(out_path, index=False, compression={"method": "gzip", "mtime": 0}, float_format="%.10g")
    return partitions


def load_partitions(path: Path | str = PARTITIONS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it once from a run with predictions.csv files: "
            "python -m qsarena.meta_analysis --build-partitions"
        )
    return pd.read_csv(path, dtype={"dataset": str, "split": str, "smiles": str})


def verify_partition_hashes(partitions: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    """Datasets whose train or test SMILES (ordered by row_index) miss the recorded split hash."""
    expected = summary.set_index("dataset")
    bad = []
    for dataset, group in partitions.groupby("dataset", sort=True):
        for split, column in (("train", "split_train_hash"), ("test", "split_test_hash")):
            smiles = group.loc[group["split"] == split].sort_values("row_index")["smiles"]
            if dataset not in expected.index or smiles_hash(smiles) != expected.at[dataset, column]:
                bad.append(f"{dataset}:{split}")
    return bad


def infer_task(observed: pd.Series) -> str:
    """Classification iff targets are strictly 0/1 (catalog task metadata is wrong for some datasets)."""
    values = pd.Series(observed).dropna().unique()
    return "classification" if len(values) and set(np.round(values, 12)) <= {0.0, 1.0} else "regression"


#: Feature-name prefix -> family, as in the notebook's ``feature_family_from_name`` (Fig 4).
FEATURE_FAMILY_PREFIXES = [
    ("morgan_bit_", "morgan"),
    ("ecfp6_bit_", "ecfp6"),
    ("fcfp6_bit_", "fcfp6"),
    ("layered_bit_", "layered"),
    ("atom_pair_bit_", "atom_pair"),
    ("topological_torsion_bit_", "topological_torsion"),
    ("rdk_path_bit_", "rdk_path"),
    ("maccs_bit_", "maccs"),
    ("rdkit_", "rdkit"),
    ("avalon_count_", "avalon"),
    ("erg_", "erg"),
    ("maplight_morgan_", "maplight"),
    ("maplight_desc_", "maplight"),
    ("gin_emb_", "gin_embedding"),
]


def feature_family_from_name(feature_name: str) -> str:
    name = str(feature_name or "")
    for prefix, family in FEATURE_FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    return "other"


def load_selected_feature_families(run_dir: Path | str = DEFAULT_RUN_DIR) -> pd.DataFrame:
    """Per-dataset count of selected features in each feature family (the Fig 4 input).

    MapLight classic is split by the runner into avalon, erg and the maplight descriptor panel;
    they are summed into ``maplight_classic`` as the manuscript does.
    """
    rows = []
    for d in dataset_dirs(run_dir):
        path = d / "selected_features.csv"
        if not path.exists():
            continue
        names = pd.read_csv(path).iloc[:, 0]
        counts = names.map(feature_family_from_name).value_counts()
        rows.append({"dataset": d.name, **{str(k): float(v) for k, v in counts.items()}})
    table = pd.DataFrame(rows).fillna(0.0)
    classic = [c for c in ("avalon", "erg", "maplight") if c in table.columns]
    if classic:
        table["maplight_classic"] = table[classic].sum(axis=1)
    return table


def load_manuscript_numbers(path: Path | str = MANUSCRIPT_NUMBERS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
