"""Phase 0: every artifact documented in docs/meta_analysis/INVENTORY.md exists and has the documented columns.

Fails loudly (never skips) so downstream phases cannot silently run on invented data.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from qsarena.meta_analysis import io

INVENTORY = io.REPO_ROOT / "docs" / "meta_analysis" / "INVENTORY.md"

METRICS_COLUMNS = {
    "dataset",
    "model",
    "workflow",
    "primary_metric",
    "primary_metric_value",
    "n_train",
    "n_test",
    "split_strategy",
    "split_train_hash",
    "split_test_hash",
    "target_transform",
    "error",
}
FAMILY_BEST_COLUMNS = {
    "dataset",
    "family",
    "model",
    "task_kind",
    "analysis_metric",
    "comparison_score",
    "relative_gap_to_best",
    "rank_within_dataset",
}


def test_inventory_paths_exist():
    text = INVENTORY.read_text(encoding="utf-8")
    paths = sorted(
        set(
            re.findall(
                r"`((?:benchmark_results|data|manuscript_assets|portable_colab_qsar_bundle|qsarena|"
                r"submission|docs)/[^`*<>]+?)`",
                text,
            )
        )
    )
    assert paths, "INVENTORY.md lists no repository paths"
    missing = [p for p in paths if not (io.REPO_ROOT / p).exists()]
    assert not missing, f"INVENTORY.md paths that do not exist: {missing}"


def test_run_has_44_datasets_with_metrics_schema():
    dirs = io.dataset_dirs()
    assert len(dirs) == 44
    frame = pd.read_csv(dirs[0] / "metrics.csv", nrows=5)
    assert METRICS_COLUMNS <= set(frame.columns), METRICS_COLUMNS - set(frame.columns)


def test_dataset_summary_columns_and_split_protocols():
    summary = io.load_dataset_summary()
    assert list(summary.columns) == io.SUMMARY_COLUMNS
    assert summary["split_strategy"].value_counts().to_dict() == {
        "predefined": 27,
        "scaffold": 12,
        "target_quartiles": 4,
        "random": 1,
    }


def test_partitions_file_loads_with_documented_columns():
    partitions = io.load_partitions()
    assert list(partitions.columns) == io.PARTITION_COLUMNS
    assert partitions["dataset"].nunique() == 44
    assert len(partitions) == 156052  # manuscript total molecule count


def test_family_best_export_schema():
    frame = pd.read_csv(io.FAMILY_BEST_PATH)
    assert FAMILY_BEST_COLUMNS <= set(frame.columns)
    matrix = pd.read_csv(io.FIG6_MATRIX_PATH, index_col=0)
    assert matrix.shape[0] == 44


def test_selected_feature_families_load():
    table = io.load_selected_feature_families()
    assert len(table) == 44 and "maplight_classic" in table.columns


def test_manuscript_numbers_have_family_consistency():
    numbers = json.loads(io.MANUSCRIPT_NUMBERS_PATH.read_text(encoding="utf-8"))
    assert "family_consistency" in numbers and "run_comparison_vs_rtx" in numbers


@pytest.mark.parametrize("entry_point", ["qsarena-applicability-domain", "qsarena-benchmark"])
def test_documented_entry_points_declared(entry_point):
    pyproject = (io.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert entry_point in pyproject
