"""Phases 4 and 6: figures, end-to-end pipeline, byte-stable meta_numbers.json.

The full pipeline takes a few minutes on CPU (two fingerprint catalogs over 156k molecules plus
10,000-sample bootstraps), so these tests are marked ``slow``.
"""

from __future__ import annotations

import json
import time

import pytest

from qsarena.meta_analysis import pipeline

pytestmark = pytest.mark.slow

EXPECTED_FIGURES = [
    "figureM1_size_crossover",
    "figureM2_shift_difficulty",
    "figureM3_recommender",
    "figureS_meta_feature_selection_diversity",
]
EXPECTED_TABLES = [
    "meta_feature_catalog.csv",
    "meta_feature_catalog_sensitivity_ecfp6_4096.csv",
    "tableS9_meta_feature_catalog.csv",
    "tableS9_meta_feature_catalog.md",
    "tableS10_meta_effect_sizes.csv",
    "tableS10_meta_effect_sizes.md",
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("meta_run")
    start = time.perf_counter()
    results = pipeline.run_meta_analysis(out)
    return out, results, time.perf_counter() - start


def test_completes_under_ten_minutes_and_writes_everything(run):
    out, results, seconds = run
    assert seconds < 600, f"meta-analysis took {seconds:.0f}s"
    for stem in EXPECTED_FIGURES:
        for ext in ("pdf", "png"):
            path = out / "figures" / f"{stem}.{ext}"
            assert path.exists() and path.stat().st_size > 1000, path
    for name in EXPECTED_TABLES:
        assert (out / "tables" / name).exists(), name
    catalog = results["catalog"]
    assert len(catalog) == 44
    assert catalog[["n_train", "n_test", "task", "split_strategy"]].notna().all().all()


def test_figure_annotations_are_in_meta_numbers(run):
    out, _, _ = run
    numbers = json.loads((out / "meta_numbers.json").read_text(encoding="utf-8"))
    figures = numbers["figures"]
    assert set(figures) == {"M1", "M2", "M3", "S1"}
    assert figures["M1"]["crossover_n"] == numbers["crossover_primary"]["crossover_n"]
    assert set(figures["M1"]["slopes_pp_per_decade"]) == set(numbers["trend_families"])
    assert figures["M3"]["majority_bacc"] == numbers["recommender"]["majority_balanced_accuracy"]
    for key in ("rho_morgan_type_share", "rho_physchem_share"):
        assert key in figures["S1"]


def test_meta_numbers_are_byte_stable(run, tmp_path):
    out, _, _ = run
    pipeline.run_meta_analysis(tmp_path)
    assert (out / "meta_numbers.json").read_bytes() == (tmp_path / "meta_numbers.json").read_bytes()
