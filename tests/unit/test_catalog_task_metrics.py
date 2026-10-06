"""Binary datasets must not carry a regression metric in the dataset catalog (revision R1, Phase 6.1).

Four binary TDC datasets (CYP1A2 and CYP2C19 inhibition, hERG-Karim, PAMPA) had no metric in the catalog, so the
runner fell back to RMSE for them; models scored through that metric saved hard class labels, and their AUROC was
computed on labels until 2026-10-04. The analysis always inferred classification from the binary targets.
"""

from __future__ import annotations

import csv
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[2] / "data" / "benchmark_dataset_catalog.csv"
REGRESSION_METRICS = {"rmse", "mae", "mse", "r2", "spearman", "pearson", "mean_squared_error"}


def _rows():
    with CATALOG.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_binary_datasets_have_no_regression_metric():
    for row in _rows():
        if row.get("target_type_inferred_from_local_data") == "binary_0_1":
            assert row["recommended_metric"].strip().lower() not in REGRESSION_METRICS, row["dataset"]


def test_previously_mislabelled_binary_datasets_use_auroc():
    metric = {row["dataset"]: row["recommended_metric"] for row in _rows()}
    for dataset in ("tdc_cyp1a2_veith", "tdc_cyp2c19_veith", "tdc_herg_karim", "tdc_pampa_ncats"):
        assert metric[dataset] == "roc_auc", dataset
