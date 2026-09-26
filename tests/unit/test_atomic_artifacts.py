"""qsarena.artifacts: interrupt-safe writes and --fresh superseding."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qsarena import artifacts


class _ExplodingFrame:
    """Writes half a file, then fails, like a process killed mid-write."""

    def to_csv(self, path, **_kwargs):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("model,test_rmse\nElasticNet")
        raise RuntimeError("killed mid-write")


def test_atomic_write_leaves_no_partial_file_on_exception(tmp_path):
    target = tmp_path / "metrics.csv"
    pd.DataFrame({"model": ["A"], "test_rmse": [0.5]}).to_csv(target, index=False)
    before = target.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        artifacts.atomic_write_csv(target, _ExplodingFrame())
    assert target.read_text(encoding="utf-8") == before  # previous complete version survives
    assert [p.name for p in tmp_path.iterdir()] == ["metrics.csv"]  # temp file cleaned up


def test_atomic_writes_replace_and_round_trip(tmp_path):
    artifacts.atomic_write_json(tmp_path / "a.json", {"status": "running"})
    artifacts.atomic_write_json(tmp_path / "a.json", {"status": "completed"})
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {"status": "completed"}
    artifacts.atomic_write_text(tmp_path / "sub" / "b.txt", "x\n")
    assert (tmp_path / "sub" / "b.txt").read_text(encoding="utf-8") == "x\n"
    artifacts.atomic_write_pickle(tmp_path / "c.pkl", {"k": [1, 2]})
    assert (tmp_path / "c.pkl").stat().st_size > 0
    artifacts.merge_json(tmp_path / "a.json", {"n": 3})
    assert artifacts.read_json(tmp_path / "a.json") == {"status": "completed", "n": 3}


def test_direct_writes_when_atomic_disabled(tmp_path):
    artifacts.set_atomic_writes(False)
    try:
        artifacts.atomic_write_text(tmp_path / "d.txt", "direct")
    finally:
        artifacts.set_atomic_writes(True)
    assert (tmp_path / "d.txt").read_text(encoding="utf-8") == "direct"


def test_supersede_directory_renames_and_never_deletes(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.csv").write_text("x", encoding="utf-8")
    moved = artifacts.supersede_directory(run, stamp="20260101_000000")
    assert moved is not None and moved.name == "run_superseded_20260101_000000"
    assert (moved / "metrics.csv").read_text(encoding="utf-8") == "x"
    assert not run.exists()
    assert artifacts.supersede_directory(tmp_path / "missing") is None
