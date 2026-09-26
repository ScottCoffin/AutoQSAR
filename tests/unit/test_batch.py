"""qsarena.batch: manifest / directory / list sources."""

from __future__ import annotations

import pytest

from qsarena import batch


def test_manifest_entries_and_overrides(tmp_path):
    (tmp_path / "a.csv").write_text("smiles,y\nCCO,1\n", encoding="utf-8")
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "dataset_name,path,smiles_col,target_col,split,split.cv_folds,models.disable_models\n"
        "first,a.csv,smiles,y;y2,scaffold,3,Tabular CNN;SVR\n"
        ",a.csv,,,,,\n",
        encoding="utf-8",
    )
    entries = batch.resolve_batch_entries([manifest])
    assert [e.name for e in entries] == ["first", "a"]
    assert entries[0].path == (tmp_path / "a.csv").resolve()
    assert entries[0].overrides == {
        "input.smiles_col": "smiles",
        "input.target_col": ["y", "y2"],
        "split.strategy": "scaffold",
        "split.cv_folds": "3",
        "models.disable_models": ["Tabular CNN", "SVR"],
    }
    assert entries[1].overrides == {}  # empty cells mean "use the run-level value"


def test_manifest_rejects_unknown_columns(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("path,colour\na.csv,red\n", encoding="utf-8")
    with pytest.raises(batch.BatchSourceError, match="unknown manifest column"):
        batch.load_batch_manifest(manifest)


def test_directory_and_list_sources(tmp_path):
    folder = tmp_path / "dir"
    folder.mkdir()
    for name in ("b.csv", "a.csv"):
        (folder / name).write_text("smiles,target\nCCO,1\n", encoding="utf-8")
    assert [e.name for e in batch.resolve_batch_entries([folder])] == ["a", "b"]
    listing = tmp_path / "list.txt"
    listing.write_text("# my datasets\ndir/a.csv\ndir/b.csv  # second\n", encoding="utf-8")
    assert [e.name for e in batch.resolve_batch_entries([listing])] == ["a", "b"]
    assert [e.name for e in batch.resolve_batch_entries([folder / "a.csv", folder / "b.csv"])] == ["a", "b"]
    with pytest.raises(batch.BatchSourceError, match="no .csv files"):
        batch.resolve_batch_entries([tmp_path / "list.txt"], mode="directory")


def test_batch_has_no_artificial_dataset_count_limit(tmp_path):
    rows = "\n".join(f"d{i},x{i}.csv" for i in range(2500))
    manifest = tmp_path / "big.csv"
    manifest.write_text("dataset_name,path\n" + rows + "\n", encoding="utf-8")
    assert len(batch.resolve_batch_entries([manifest])) == 2500


def test_unique_names_prevent_output_collisions():
    entries = [batch.BatchEntry("x", path=None), batch.BatchEntry("X", path=None), batch.BatchEntry("y", path=None)]
    assert [e.name for e in batch.unique_names(entries)] == ["x", "X_2", "y"]
