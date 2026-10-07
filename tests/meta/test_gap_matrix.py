"""Phase 2: the family gap matrix is the Fig 6 matrix, and reproduces Table 3's family statistics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qsarena.meta_analysis import gap_matrix as gm
from qsarena.meta_analysis import io


@pytest.fixture(scope="module")
def family_best():
    return gm.load_family_best()


@pytest.fixture(scope="module")
def gap(family_best):
    return gm.load_family_gap_matrix(family_best)


@pytest.fixture(scope="module")
def numbers():
    return json.loads(io.MANUSCRIPT_NUMBERS_PATH.read_text(encoding="utf-8"))


def test_matches_fig6_renderer_matrix(gap):
    fig6 = pd.read_csv(io.FIG6_MATRIX_PATH, index_col=0)
    ours = gap.loc[fig6.index, fig6.columns]
    both = fig6.notna() & ours.notna()
    assert (fig6.notna() == ours.notna()).all().all()
    assert int(both.values.sum()) >= 5
    assert np.allclose(ours.values[both.values], fig6.values[both.values], atol=1e-6, rtol=0)


def test_every_dataset_has_a_winner_at_zero_gap(gap):
    assert (gap.min(axis=1) <= 1e-9).all()
    assert len(gap) == 44


def test_chemprop_mask_matches_table3(gap, numbers):
    expected = numbers["family_consistency"][gm.CHEMPROP]["Datasets with valid results"]
    assert int(gm.chemprop_valid(gap).sum()) == expected


def test_within5_reproduces_table3(gap, numbers):
    within = gm.within5_matrix(gap)
    for family, record in numbers["family_consistency"].items():
        if family not in within.columns:
            continue
        observed = 100 * np.mean([bool(v) for v in within[family].dropna()])
        assert observed == pytest.approx(record["Within 5% of best (% of datasets)"], abs=0.05), family


def test_winner_groups_are_populated(family_best):
    winners = gm.winner_family(family_best)
    assert len(winners) == 44
    assert set(winners["winner_group"]) <= set(gm.GROUP_ORDER)
    wins = winners["winner_family"].value_counts()
    assert wins.sum() == 44 and wins.index.isin(list(gm.FAMILY_COLORS)).all()
    assert (winners["winner_group"].value_counts() >= 4).all()


def test_winner_counts_match_figure2(family_best, numbers):
    winners = gm.winner_family(family_best)["winner_family"].value_counts().to_dict()
    for family, record in numbers["wins_by_family"].items():
        assert winners.get(family, 0) == record["total"], family
