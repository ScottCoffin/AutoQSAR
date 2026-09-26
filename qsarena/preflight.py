"""
qsarena.preflight — checks that run before any model is fitted.

``preflight_dataset`` inspects one dataset (SMILES parse rate, rows that will be dropped, duplicate
rate, inferred task, class balance, size against the guardrails) and ``backend_status`` reports
which optional backends and GPUs are usable. Every problem becomes a :class:`PreflightMessage` with
an actionable remedy; ``qsarena-benchmark`` prints them, writes them to ``preflight.json`` and the
report, and ``--dry-run`` stops right after them.

``estimate_model_seconds`` is the dry-run cost heuristic. It scales the median per-fit wall-clock of
each model family measured in the paper's A100 benchmark run (Table 6) linearly with dataset size.
It is an order-of-magnitude guide, not a prediction.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "PreflightMessage",
    "PreflightResult",
    "preflight_dataset",
    "backend_status",
    "backend_messages",
    "estimate_model_seconds",
    "estimate_pipeline_seconds",
    "format_duration",
    "SMALL_DATASET_ROWS",
    "LARGE_DATASET_ROWS",
]

#: Below this many valid rows a 20 % test split holds fewer than 20 molecules.
SMALL_DATASET_ROWS = 100
#: Above this many rows the full model library takes many GPU-hours.
LARGE_DATASET_ROWS = 50_000
#: Duplicate structures above this fraction trigger a deduplication hint.
DUPLICATE_WARN_FRACTION = 0.05
#: Minority class below this fraction triggers an imbalance hint.
MINORITY_WARN_FRACTION = 0.10
#: Preflight parses at most this many SMILES per dataset (a deterministic sample beyond it).
PARSE_SAMPLE_LIMIT = 20_000


@dataclass
class PreflightMessage:
    level: str  # info | warning | error
    code: str
    message: str
    remedy: str = ""
    dataset: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def console_text(self) -> str:
        where = f"{self.dataset}: " if self.dataset else ""
        text = f"[preflight] {self.level} {where}{self.message}"
        return text + (f" -> {self.remedy}" if self.remedy else "")


@dataclass
class PreflightResult:
    dataset: str
    n_rows: int
    n_missing: int
    n_parsed: int
    n_unparseable: int
    parse_rate: float
    n_valid: int
    duplicate_fraction: float
    task: str
    task_source: str
    class_balance: dict[str, int] = field(default_factory=dict)
    sampled: bool = False
    messages: list[PreflightMessage] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["messages"] = [m.as_dict() for m in self.messages]
        return out

    @property
    def will_skip(self) -> bool:
        return any(m.level == "error" for m in self.messages)


def _is_blank(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "na"}


def preflight_dataset(
    smiles: Sequence[Any],
    target: Sequence[Any],
    *,
    name: str,
    task: str = "auto",
    classification_threshold: float | None = None,
    minimum_rows: int = 20,
    drop_unparseable: bool = True,
    test_fraction: float = 0.2,
    sample_limit: int = PARSE_SAMPLE_LIMIT,
) -> PreflightResult:
    """Inspect one dataset without fitting anything."""
    import numpy as np
    import pandas as pd
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    frame = pd.DataFrame({"smiles": list(smiles), "target": pd.to_numeric(pd.Series(list(target)), errors="coerce")})
    n_rows = int(len(frame))
    usable = frame[~frame["smiles"].map(_is_blank) & frame["target"].notna() & np.isfinite(frame["target"])]
    n_missing = int(n_rows - len(usable))
    sampled = len(usable) > sample_limit
    probe = usable.sample(n=sample_limit, random_state=0) if sampled else usable
    canonical: list[str | None] = []
    for text in probe["smiles"].astype(str).str.strip():
        mol = Chem.MolFromSmiles(text)
        canonical.append(Chem.MolToSmiles(mol) if mol is not None else None)
    parsed_mask = np.array([c is not None for c in canonical], dtype=bool)
    parse_rate = float(parsed_mask.mean()) if len(parsed_mask) else 0.0
    n_unparseable = int(round((1.0 - parse_rate) * len(usable))) if len(usable) else 0
    n_parsed = int(len(usable) - n_unparseable)
    n_valid = n_parsed if drop_unparseable else (int(len(usable)) if n_unparseable == 0 else 0)
    parsed = [c for c in canonical if c is not None]
    duplicate_fraction = float(1.0 - len(set(parsed)) / len(parsed)) if parsed else 0.0

    values = probe["target"].to_numpy(dtype=float)[parsed_mask] if len(probe) else np.array([])
    if classification_threshold is not None:
        values = (values >= float(classification_threshold)).astype(float)
    unique = np.unique(values) if len(values) else np.array([])
    task_text = str(task or "auto").lower()
    if task_text in {"classification", "regression"}:
        inferred, source = task_text, "set by input.task"
    elif classification_threshold is not None:
        inferred, source = "classification", "input.classification_threshold"
    elif len(unique) == 2:
        inferred, source = "classification", "auto: target has exactly two values"
    else:
        inferred, source = "regression", f"auto: target has {len(unique)} distinct values"
    class_balance: dict[str, int] = {}
    if inferred == "classification" and len(values):
        for value in unique:
            label = str(int(value)) if float(value).is_integer() else f"{value:g}"
            class_balance[label] = int(round(float((values == value).sum()) * (len(usable) / max(1, len(probe)))))

    result = PreflightResult(
        dataset=name,
        n_rows=n_rows,
        n_missing=n_missing,
        n_parsed=n_parsed,
        n_unparseable=n_unparseable,
        parse_rate=parse_rate,
        n_valid=n_valid,
        duplicate_fraction=duplicate_fraction,
        task=inferred,
        task_source=source,
        class_balance=class_balance,
        sampled=sampled,
    )
    add = result.messages.append
    if n_missing:
        add(PreflightMessage("info", "missing_values", f"{n_missing} of {n_rows} rows have no SMILES or no numeric target and will be ignored.", dataset=name))
    if n_unparseable:
        if drop_unparseable:
            add(PreflightMessage(
                "warning", "unparseable_smiles",
                f"{n_unparseable} SMILES ({(1 - parse_rate):.1%}) cannot be parsed by RDKit and will be dropped.",
                "Fix them in the CSV, or pass --no-drop-unparseable to stop instead of dropping.", name,
            ))
        else:
            add(PreflightMessage(
                "error", "unparseable_smiles",
                f"{n_unparseable} SMILES cannot be parsed and --no-drop-unparseable is set; the dataset will fail.",
                "Fix the SMILES or allow dropping with --drop-unparseable.", name,
            ))
    if n_valid < int(minimum_rows):
        add(PreflightMessage(
            "error", "too_few_rows",
            f"only {n_valid} usable rows (minimum_rows={int(minimum_rows)}); the dataset will be skipped.",
            "Add data or lower --minimum-rows.", name,
        ))
    elif n_valid < SMALL_DATASET_ROWS:
        n_test = max(1, int(math.ceil(float(test_fraction) * n_valid)))
        add(PreflightMessage(
            "warning", "small_dataset",
            f"{n_valid} usable rows: the test split will hold about {n_test} molecules, so test metrics are noisy.",
            "Prefer the cross-validated scores (--selection-protocol cv or both) and treat rankings as tentative.", name,
        ))
    if n_valid > LARGE_DATASET_ROWS:
        add(PreflightMessage(
            "warning", "large_dataset",
            f"{n_valid:,} rows: the full model library can take many GPU-hours on a dataset this size.",
            "Start with --benchmark-profile quick or --dry-run to see the estimate.", name,
        ))
    if duplicate_fraction > DUPLICATE_WARN_FRACTION:
        add(PreflightMessage(
            "warning", "duplicates",
            f"{duplicate_fraction:.1%} of parsed rows repeat a structure already present.",
            "Merge them with --deduplicate canonical_smiles so one molecule cannot sit in both train and test.", name,
        ))
    if task_text == "classification" and classification_threshold is None and len(unique) != 2:
        add(PreflightMessage(
            "error", "not_binary",
            f"task is classification but the target has {len(unique)} distinct values.",
            "Binarize with --classification-threshold VALUE, or use --task regression.", name,
        ))
    if inferred == "classification" and class_balance:
        counts = sorted(class_balance.values())
        minority = counts[0] / max(1, sum(counts))
        if minority < MINORITY_WARN_FRACTION:
            add(PreflightMessage(
                "warning", "class_imbalance",
                f"minority class is {minority:.1%} of the data ({class_balance}).",
                "Judge models by AUPRC or balanced accuracy (--primary-metric auprc) rather than accuracy.", name,
            ))
    add(PreflightMessage("info", "task", f"task = {inferred} ({source}).", dataset=name))
    return result


_BACKENDS: list[tuple[str, tuple[str, ...], str]] = [
    ("xgboost", ("xgboost",), "pip install 'qsarena[boosting]'"),
    ("lightgbm", ("lightgbm",), "pip install 'qsarena[boosting]'"),
    ("catboost", ("catboost",), "pip install 'qsarena[boosting]'"),
    ("torch", ("torch",), "pip install 'qsarena[deep]'"),
    ("tensorflow", ("tensorflow",), "pip install 'qsarena[deep]' (Python < 3.13)"),
    ("chemprop", ("chemprop",), "pip install 'qsarena[graph]'"),
    ("unimol_tools", ("unimol_tools",), "pip install 'qsarena[foundation]'"),
    ("tabpfn", ("tabpfn", "tabpfn_client"), "pip install 'qsarena[foundation]'"),
    ("dgl", ("dgl", "dgllife"), "install dgl and dgllife from the DGL wheel index (see README, 'Installation via pip')"),
]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def backend_status(gpu_available: bool) -> dict[str, dict[str, Any]]:
    """Which optional backends are importable (by package metadata, without importing them)."""
    status: dict[str, dict[str, Any]] = {}
    for name, modules, remedy in _BACKENDS:
        if name == "dgl":
            available = all(_module_available(m) for m in modules)
        else:
            available = any(_module_available(m) for m in modules)
        status[name] = {"available": bool(available), "remedy": remedy}
    status["gpu"] = {
        "available": bool(gpu_available),
        "remedy": "run on a machine with an NVIDIA GPU and a CUDA build of PyTorch (pip install 'qsarena[foundation]')",
    }
    return status


def backend_messages(status: dict[str, dict[str, Any]], wants: dict[str, bool]) -> list[PreflightMessage]:
    """Actionable messages for requested model families whose backend is missing.

    ``wants`` maps: boosting, chemml_pytorch, cnn, chemprop, unimol_v1, unimol_v2, unimol_auto,
    tabpfn, maplight_gnn -> requested?
    """
    out: list[PreflightMessage] = []

    def missing(name: str) -> bool:
        return not bool(status.get(name, {}).get("available", False))

    if wants.get("boosting") and any(missing(n) for n in ("xgboost", "lightgbm", "catboost")):
        names = [n for n in ("xgboost", "lightgbm", "catboost") if missing(n)]
        out.append(PreflightMessage(
            "warning", "backend_missing",
            f"{', '.join(names)} not installed: those gradient-boosting models (and MapLight CatBoost) are skipped.",
            status["xgboost"]["remedy"],
        ))
    if wants.get("chemml_pytorch") and missing("torch"):
        out.append(PreflightMessage("warning", "backend_missing", "PyTorch not installed: ChemML MLP (PyTorch) will fail and be recorded as an error.", status["torch"]["remedy"]))
    if wants.get("cnn") and missing("tensorflow"):
        out.append(PreflightMessage("info", "backend_missing", "TensorFlow not installed: the tabular CNN is skipped.", status["tensorflow"]["remedy"]))
    if wants.get("chemprop") and missing("chemprop"):
        out.append(PreflightMessage("warning", "backend_missing", "Chemprop not installed: Chemprop variants will be recorded as errors.", status["chemprop"]["remedy"] + " or --disable-model-families graph_nn"))
    if (wants.get("unimol_v1") or wants.get("unimol_v2")) and missing("unimol_tools"):
        out.append(PreflightMessage("warning", "backend_missing", "unimol_tools not installed: Uni-Mol stages will be recorded as errors.", status["unimol_tools"]["remedy"]))
    if wants.get("tabpfn") and missing("tabpfn"):
        out.append(PreflightMessage("info", "backend_missing", "TabPFN not installed: TabPFN is switched off for this run.", status["tabpfn"]["remedy"]))
    if wants.get("maplight_gnn") and missing("dgl"):
        out.append(PreflightMessage("warning", "backend_missing", "dgl/dgllife not installed: MapLight + GNN is recorded as skipped.", status["dgl"]["remedy"]))
    if not status.get("gpu", {}).get("available", False):
        if wants.get("unimol_auto"):
            out.append(PreflightMessage(
                "info", "no_gpu",
                "GPU not detected: Uni-Mol V1/V2 are skipped (their default is GPU-only). Chemprop and the other deep "
                "models run on CPU, more slowly.",
                "Use a CUDA GPU with pip install 'qsarena[foundation]', or force Uni-Mol V1 on CPU with --run-unimol-v1.",
            ))
        if wants.get("unimol_v2"):
            out.append(PreflightMessage("warning", "no_gpu", "Uni-Mol V2 was requested but needs a GPU; it will be skipped.", status["gpu"]["remedy"]))
    return out


#: Median own wall-clock seconds per model-dataset fit in the paper's A100 run (Table 6), at the
#: median dataset size of that run (1,605 molecules).
_TABLE6_MEDIAN_SECONDS = {
    "conventional_ml": 6.8,
    "gradient_boosting": 6.8,
    "deep_tabular": 39.8,
    "maplight_gnn": 137.4,
    "graph_nn": 268.6,
    "pretrained_3d": 373.5,
    "fusion": 0.3,
    "ensemble": 0.6,
}
_REFERENCE_ROWS = 1605.0
#: Assumed slow-down of the GPU-accelerated families when no GPU is present.
_CPU_PENALTY = {"graph_nn": 8.0, "pretrained_3d": 8.0, "deep_tabular": 3.0, "maplight_gnn": 2.0}


def estimate_model_seconds(family: str, n_rows: int, *, gpu: bool) -> float:
    base = _TABLE6_MEDIAN_SECONDS.get(family, 6.8)
    scale = max(0.05, float(n_rows) / _REFERENCE_ROWS)
    seconds = base * scale
    if not gpu:
        seconds *= _CPU_PENALTY.get(family, 1.0)
    return float(seconds)


def estimate_pipeline_seconds(
    n_rows: int,
    families: Iterable[str],
    *,
    gpu: bool,
    selector_seconds: float = 0.0,
    ga_fits: int = 0,
) -> dict[str, float]:
    """Seconds per stage for one dataset: features, selection, each model family, GA."""
    out: dict[str, float] = {"features": 5.0 + 0.02 * float(n_rows), "feature_selection": float(selector_seconds)}
    for family in families:
        out[family] = out.get(family, 0.0) + estimate_model_seconds(family, n_rows, gpu=gpu)
    if ga_fits:
        out["ga_tuning"] = float(ga_fits) * estimate_model_seconds("conventional_ml", n_rows, gpu=gpu) / 5.0
    return out


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"
