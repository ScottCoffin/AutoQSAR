"""
qsarena.batch — batch mode over any number of user datasets.

A batch source (``--batch`` / ``batch.source``) is one of:

  manifest   a CSV with one row per dataset. ``path`` is required; ``dataset_name`` is optional
             (defaults to the file name). Any other column overrides a setting for that row only:
             the short names in :data:`MANIFEST_COLUMN_KEYS` or any per-dataset RunConfig key
             written in dotted form (``split.cv_folds``, ``models.disable_models``, ...). Empty
             cells mean "use the run-level value". Relative paths are relative to the manifest.
  directory  every ``*.csv`` in a directory (sorted by name), columns auto-detected.
  list       a ``.txt`` file with one CSV path per line (``#`` comments allowed), or several
             ``--batch`` paths.

Each dataset gets its own output subdirectory, a failure in one dataset is recorded and the batch
continues, and :func:`write_dataset_summary` writes the cross-dataset ``dataset_summary.csv``.
There is no limit on the number of datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "BatchEntry",
    "MANIFEST_COLUMN_KEYS",
    "BatchSourceError",
    "resolve_batch_entries",
    "load_batch_manifest",
    "discover_batch_directory",
    "read_list_file",
    "collect_dataset_summary",
    "write_dataset_summary",
]

#: Short manifest column -> RunConfig key.
MANIFEST_COLUMN_KEYS: dict[str, str] = {
    "smiles_col": "input.smiles_col",
    "target_col": "input.target_col",
    "id_col": "input.id_col",
    "task": "input.task",
    "classification_threshold": "input.classification_threshold",
    "target_transform": "input.target_transform",
    "minimum_rows": "input.minimum_rows",
    "split": "split.strategy",
    "test_fraction": "split.test_fraction",
    "seed": "split.seed",
    "cv_folds": "split.cv_folds",
    "predefined_split_col": "split.predefined_split_col",
    "primary_metric": "evaluation.primary_metric",
}
_RESERVED = {"dataset_name", "path"}
_LIST_KEYS = {"input.target_col", "models.disable_models", "models.only_models", "features.families",
              "deep.chemprop.variants", "ga_tuning.estimators"}


class BatchSourceError(ValueError):
    """The batch source itself is unusable (as opposed to one dataset in it)."""


@dataclass
class BatchEntry:
    name: str
    path: Path
    overrides: dict[str, Any] = field(default_factory=dict)
    origin: str = ""


def _clean_cell(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def load_batch_manifest(path: str | Path) -> list[BatchEntry]:
    import pandas as pd

    manifest = Path(path)
    try:
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise BatchSourceError(f"{manifest}: cannot read the manifest ({type(exc).__name__}: {exc}).") from None
    columns = [str(c).strip() for c in frame.columns]
    frame.columns = columns
    if "path" not in columns:
        raise BatchSourceError(f"{manifest}: a batch manifest needs a 'path' column (columns: {', '.join(columns)}).")
    unknown = [c for c in columns if c not in _RESERVED and c not in MANIFEST_COLUMN_KEYS and "." not in c]
    if unknown:
        raise BatchSourceError(
            f"{manifest}: unknown manifest column(s) {', '.join(unknown)}. Use dataset_name, path, "
            f"{', '.join(MANIFEST_COLUMN_KEYS)}, or a dotted RunConfig key such as split.cv_folds."
        )
    entries: list[BatchEntry] = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        raw_path = _clean_cell(row.get("path"))
        if raw_path is None:
            raise BatchSourceError(f"{manifest}: row {row_number} has an empty path.")
        dataset_path = Path(raw_path)
        if not dataset_path.is_absolute():
            dataset_path = (manifest.parent / dataset_path).resolve()
        overrides: dict[str, Any] = {}
        for column, value in row.items():
            if column in _RESERVED:
                continue
            cell = _clean_cell(value)
            if cell is None:
                continue
            key = MANIFEST_COLUMN_KEYS.get(column, column)
            if key in _LIST_KEYS:
                cell = [item.strip() for item in cell.split(";") if item.strip()]
            overrides[key] = cell
        name = _clean_cell(row.get("dataset_name")) or dataset_path.stem
        entries.append(BatchEntry(name=str(name), path=dataset_path, overrides=overrides,
                                  origin=f"{manifest.name} row {row_number}"))
    return entries


def discover_batch_directory(path: str | Path) -> list[BatchEntry]:
    directory = Path(path)
    files = sorted(p for p in directory.glob("*.csv") if p.is_file())
    if not files:
        raise BatchSourceError(f"{directory}: no .csv files found.")
    return [BatchEntry(name=p.stem, path=p.resolve(), origin=f"{directory.name}/{p.name}") for p in files]


def read_list_file(path: str | Path) -> list[BatchEntry]:
    list_file = Path(path)
    entries: list[BatchEntry] = []
    for line_number, line in enumerate(list_file.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        item = Path(text)
        if not item.is_absolute():
            item = (list_file.parent / item).resolve()
        entries.append(BatchEntry(name=item.stem, path=item, origin=f"{list_file.name} line {line_number}"))
    if not entries:
        raise BatchSourceError(f"{list_file}: lists no CSV paths.")
    return entries


def _looks_like_manifest(path: Path) -> bool:
    try:
        header = path.open("r", encoding="utf-8-sig").readline()
    except OSError:
        return False
    names = [h.strip().strip('"').lower() for h in header.split(",")]
    return "path" in names


def resolve_batch_entries(sources: Sequence[str | Path], mode: str = "auto") -> list[BatchEntry]:
    """Turn ``--batch`` values into entries. Raises :class:`BatchSourceError` for an unusable source."""
    paths = [Path(str(s)) for s in sources if str(s).strip()]
    if not paths:
        raise BatchSourceError("no batch source given (use --batch PATH).")
    mode = str(mode or "auto").lower()
    if mode == "list" or (mode == "auto" and len(paths) > 1):
        if len(paths) == 1 and paths[0].suffix.lower() == ".txt":
            return read_list_file(paths[0])
        return [BatchEntry(name=p.stem, path=p.resolve(), origin="--batch") for p in paths]
    source = paths[0]
    if not source.exists():
        raise BatchSourceError(f"{source}: batch source not found.")
    if mode == "directory" or (mode == "auto" and source.is_dir()):
        return discover_batch_directory(source)
    if mode == "manifest":
        return load_batch_manifest(source)
    if source.suffix.lower() == ".txt":
        return read_list_file(source)
    if _looks_like_manifest(source):
        return load_batch_manifest(source)
    return [BatchEntry(name=source.stem, path=source.resolve(), origin="--batch")]


def unique_names(entries: Iterable[BatchEntry]) -> list[BatchEntry]:
    """Make dataset names unique (``name``, ``name_2``, ...) so output directories never collide."""
    seen: dict[str, int] = {}
    out = []
    for entry in entries:
        base = entry.name
        count = seen.get(base.lower(), 0) + 1
        seen[base.lower()] = count
        if count > 1:
            entry = BatchEntry(name=f"{base}_{count}", path=entry.path, overrides=entry.overrides, origin=entry.origin)
        out.append(entry)
    return out


def _read_status(dataset_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((dataset_dir / "run_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def collect_dataset_summary(
    output_dir: str | Path,
    dataset_ids: Sequence[str],
    *,
    protocol: str = "both",
) -> "Any":
    """One row per dataset: status, size, task, the best model by test and by CV, error/remedy."""
    import pandas as pd

    from qsarena.reporting import best_model_rows, load_dataset_metrics

    output = Path(output_dir)
    rows = []
    for dataset_id in dataset_ids:
        dataset_dir = output / dataset_id
        status = _read_status(dataset_dir)
        metrics = load_dataset_metrics(dataset_dir)
        row: dict[str, Any] = {
            "dataset": dataset_id,
            "status": status.get("status", "not_run"),
            "source": status.get("source", ""),
            "task": status.get("task_type", ""),
            "n_rows": status.get("n_rows", ""),
            "primary_metric": status.get("primary_metric", ""),
            "error": status.get("error", "") or status.get("reason", ""),
            "remedy": status.get("remedy", ""),
            "elapsed_seconds": status.get("elapsed_seconds", ""),
            "ad_in_domain_fraction": (status.get("applicability_domain") or {}).get("in_domain_fraction", ""),
        }
        if metrics is not None and not metrics.empty:
            errors = metrics.get("error", pd.Series([""] * len(metrics))).fillna("").astype(str).str.strip()
            row["n_models_ok"] = int((errors == "").sum())
            row["n_models_failed"] = int((errors != "").sum())
            if not row["n_rows"] and "n_molecules" in metrics:
                values = pd.to_numeric(metrics["n_molecules"], errors="coerce").dropna()
                row["n_rows"] = int(values.iloc[0]) if len(values) else ""
            best = best_model_rows(metrics, primary_metric=row["primary_metric"] or None)
            if not row["primary_metric"]:
                row["primary_metric"] = best.get("metric", "")
            for key in ("test", "cv"):
                selected = best.get(key)
                row[f"best_by_{key}"] = selected["model"] if selected else ""
                row[f"best_by_{key}_test_value"] = selected["test_value"] if selected else ""
                row[f"best_by_{key}_cv_value"] = selected["cv_value"] if selected else ""
        else:
            row.update({"n_models_ok": 0, "n_models_failed": 0})
        rows.append(row)
    columns = [
        "dataset", "status", "task", "n_rows", "primary_metric",
        "best_by_test", "best_by_test_test_value", "best_by_cv", "best_by_cv_test_value", "best_by_cv_cv_value",
        "n_models_ok", "n_models_failed", "ad_in_domain_fraction", "elapsed_seconds", "error", "remedy", "source",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    if protocol == "test":
        columns = [c for c in columns if not c.startswith("best_by_cv")]
    elif protocol == "cv":
        columns = [c for c in columns if not c.startswith("best_by_test")]
    return frame[columns]


def write_dataset_summary(output_dir: str | Path, dataset_ids: Sequence[str], *, protocol: str = "both") -> Path:
    from qsarena.artifacts import atomic_write_csv

    frame = collect_dataset_summary(output_dir, dataset_ids, protocol=protocol)
    return atomic_write_csv(Path(output_dir) / "dataset_summary.csv", frame)
