"""Shared pytest setup: make the source checkout importable and expose the tiny TDC fixture.

``tests/fixtures/tdc_tiny`` holds 36 train_val + 12 test molecules sampled from the official
TDC splits of one regression (caco2_wang) and one classification (herg) dataset, so the
integration tests exercise real chemistry in seconds with no network, GPU or ``data/`` tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "tdc_tiny"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tdc_tiny() -> Path:
    return FIXTURE_DIR
