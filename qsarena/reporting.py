"""
qsarena.reporting — the run / batch report, in both HTML and Markdown (always both).

``write_run_reports`` reads a finished run directory (per-dataset ``metrics.csv`` and
``run_status.json``, ``preflight.json``, the resolved config) and writes:

  report.html        self-contained: inline CSS and inline SVG plots, no external requests
  report.md          plain-text friendly; plots are the same SVGs under report_assets/
  report_data.json   every number shown in the reports (``report.machine_readable_manifest``)

Sections: run overview, warnings, best model per dataset (by test and/or by CV, per
``selection.protocol``), leaderboard / estimated-rank table, plots (won-by-family, cost-vs-gap,
applicability-domain coverage), model failures with remedies, "what to do next", and the resolved
configuration. The plots are drawn as plain SVG so no plotting library is needed.
"""

from __future__ import annotations

import html
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "metric_lower_is_better",
    "load_dataset_metrics",
    "best_model_rows",
    "collect_run_report_data",
    "render_markdown_report",
    "render_html_report",
    "write_run_reports",
    "remedy_for_error",
    "svg_bar_chart",
    "svg_scatter",
]

_LOWER_IS_BETTER = {"rmse", "mae", "mse"}
_HIGHER_IS_BETTER = {"r2", "spearman", "pearson", "roc_auc", "auprc", "balanced_accuracy", "mcc", "accuracy"}
_METRIC_LABELS = {
    "rmse": "RMSE", "mae": "MAE", "r2": "R2", "spearman": "Spearman", "pearson": "Pearson",
    "roc_auc": "AUROC", "auprc": "AUPRC", "balanced_accuracy": "Bal. acc.", "mcc": "MCC", "accuracy": "Accuracy",
}
#: Categorical palette (colour-blind safe, Okabe-Ito order).
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def metric_lower_is_better(metric: str | None) -> bool:
    return str(metric or "").lower() in _LOWER_IS_BETTER


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    if number is None:
        return "–"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def load_dataset_metrics(dataset_dir: str | Path) -> Any:
    import pandas as pd

    path = Path(dataset_dir) / "metrics.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _ok_rows(metrics: Any) -> Any:
    import pandas as pd

    frame = metrics.copy()
    errors = frame["error"].fillna("").astype(str).str.strip() if "error" in frame else pd.Series([""] * len(frame), index=frame.index)
    status = frame["status"].fillna("").astype(str).str.lower() if "status" in frame else pd.Series([""] * len(frame), index=frame.index)
    return frame[(errors == "") & ~status.str.startswith("skipped")]


def _dataset_metric(metrics: Any, primary_metric: str | None) -> str:
    if primary_metric:
        return str(primary_metric).lower()
    if "primary_metric" in metrics and metrics["primary_metric"].notna().any():
        return str(metrics["primary_metric"].dropna().astype(str).str.lower().mode().iloc[0])
    return "rmse"


def _test_value(row: Mapping[str, Any], metric: str) -> float | None:
    value = _finite(row.get(f"test_{metric}"))
    if value is None and str(row.get("primary_metric", "")).lower() == metric:
        value = _finite(row.get("primary_metric_value"))
    return value


def _cv_value(row: Mapping[str, Any], metric: str) -> float | None:
    value = _finite(row.get(f"cv_{metric}"))
    if value is None and str(row.get("primary_metric", "")).lower() == metric:
        value = _finite(row.get("cv_primary"))
    return value


def best_model_rows(metrics: Any, primary_metric: str | None = None) -> dict[str, Any]:
    """The best model of one dataset by held-out test score and by cross-validated score.

    Both are ranked on the dataset's primary metric. CV selection only considers models that
    report a CV score (conventional models, GA-tuned models, ChemML MLP, TabPFN).
    """
    metric = _dataset_metric(metrics, primary_metric)
    lower = metric_lower_is_better(metric)
    out: dict[str, Any] = {"metric": metric, "test": None, "cv": None}
    if metrics is None or len(metrics) == 0:
        return out
    ok = _ok_rows(metrics)
    candidates = []
    for row in ok.to_dict(orient="records"):
        test_value = _test_value(row, metric)
        cv_value = _cv_value(row, metric)
        candidates.append({
            "model": str(row.get("model", "")),
            "workflow": str(row.get("workflow", "")),
            "test_value": test_value,
            "cv_value": cv_value,
            "seconds": _finite(row.get("stage_duration_seconds")),
        })

    def pick(key: str) -> dict[str, Any] | None:
        pool = [c for c in candidates if c[key] is not None]
        if not pool:
            return None
        return (min if lower else max)(pool, key=lambda c: c[key])

    out["test"] = pick("test_value")
    out["cv"] = pick("cv_value")
    return out


_REMEDIES = [
    (("no module named 'chemprop'", "chemprop is not installed", "chemprop cli"), "Install Chemprop: pip install 'qsarena[graph]' (or --disable-model-families graph_nn)."),
    (("chemprop harness error",), "Chemprop ran but its output could not be read; rerun with --verbosity verbose to echo the command and check run.log."),
    (("chemprop training failed",), "Chemprop itself failed on this dataset; see run.log. Small datasets often need --chemprop-batch-size 16."),
    (("unimol", "uni-mol"), "Install Uni-Mol: pip install 'qsarena[foundation]' and run on a CUDA GPU (V2 requires one)."),
    (("catboost is unavailable",), "Install the boosting backends: pip install 'qsarena[boosting]'."),
    (("graphbolt", "dgl", "dgllife"), "Install dgl and dgllife from the DGL wheel index (README, 'Installation via pip'), or --disable-model-families maplight_gnn."),
    (("out of memory", "cuda error", "cublas"), "GPU memory exhausted: lower the batch size (--unimol-batch-size, --chemprop-batch-size) or use a larger GPU."),
    (("tensorflow",), "Install TensorFlow: pip install 'qsarena[deep]' (Python < 3.13), or --no-run-cnn."),
    (("torch",), "Install PyTorch: pip install 'qsarena[deep]', or --disable-model-families deep_tabular."),
    (("token", "tabpfn"), "TabPFN API budget or authentication problem; set PRIORLABS_API_KEY, install local tabpfn on a GPU, or --no-run-tabpfn."),
    (("binary classification metrics require exactly two classes",), "The target is not binary; use --classification-threshold or --task regression."),
]


def remedy_for_error(error: Any) -> str:
    text = str(error or "").lower()
    for needles, remedy in _REMEDIES:
        if any(needle in text for needle in needles):
            return remedy
    return "See run.log and events.jsonl for the traceback; rerun with --verbosity debug for more detail."


# ---------------------------------------------------------------------------------------------
# SVG plots (no plotting library)
# ---------------------------------------------------------------------------------------------


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def svg_bar_chart(labels: Sequence[str], values: Sequence[float], *, title: str, value_label: str = "",
                  max_value: float | None = None, percent: bool = False) -> str:
    """Horizontal bar chart. Returns an ``<svg>`` element string."""
    width, bar_h, gap = 640, 22, 8
    left, right, top = 190, 70, 34
    height = top + len(labels) * (bar_h + gap) + 30
    peak = max_value if max_value is not None else max([v for v in values if v is not None] + [1e-9])
    peak = peak if peak > 0 else 1.0
    span = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f"<title>{_esc(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="20" font-size="14" font-weight="bold" fill="#222">{_esc(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * (bar_h + gap)
        v = float(value or 0.0)
        w = max(1.0, span * min(1.0, v / peak)) if v > 0 else 1.0
        colour = _PALETTE[index % len(_PALETTE)]
        text = f"{v:.0%}" if percent else (f"{v:g}" if float(v).is_integer() else f"{v:.2f}")
        parts.append(f'<text x="{left - 8}" y="{y + bar_h * 0.7:.1f}" text-anchor="end" fill="#222">{_esc(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{colour}"><title>{_esc(label)}: {_esc(text)}</title></rect>')
        parts.append(f'<text x="{left + w + 6:.1f}" y="{y + bar_h * 0.7:.1f}" fill="#222">{_esc(text)}</text>')
    if value_label:
        parts.append(f'<text x="{left}" y="{height - 8}" fill="#555">{_esc(value_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_scatter(points: Sequence[Mapping[str, Any]], *, title: str, x_label: str, y_label: str, log_x: bool = True) -> str:
    """Labelled scatter plot of ``{"x", "y", "label"}`` points."""
    width, height = 640, 380
    left, right, top, bottom = 70, 30, 36, 50
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]

    def tx(value: float) -> float:
        return math.log10(max(value, 1e-3)) if log_x else value

    x_values = [tx(x) for x in xs] or [0.0]
    x_min, x_max = min(x_values), max(x_values)
    if x_max - x_min < 1e-9:
        x_min, x_max = x_min - 1, x_max + 1
    y_min, y_max = min(ys + [0.0]), max(ys + [0.05])
    if y_max - y_min < 1e-9:
        y_max = y_min + 1
    pad_x = 0.08 * (x_max - x_min)
    pad_y = 0.08 * (y_max - y_min)
    x_min, x_max, y_min, y_max = x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y

    def px(value: float) -> float:
        return left + (tx(value) - x_min) / (x_max - x_min) * (width - left - right)

    def py(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f"<title>{_esc(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="22" font-size="14" font-weight="bold" fill="#222">{_esc(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#888"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#888"/>',
        f'<text x="{(left + width - right) / 2:.0f}" y="{height - 12}" text-anchor="middle" fill="#444">{_esc(x_label)}</text>',
        f'<text x="16" y="{(top + height - bottom) / 2:.0f}" text-anchor="middle" fill="#444" '
        f'transform="rotate(-90 16 {(top + height - bottom) / 2:.0f})">{_esc(y_label)}</text>',
    ]
    if y_min < 0 < y_max:
        parts.append(f'<line x1="{left}" y1="{py(0):.1f}" x2="{width - right}" y2="{py(0):.1f}" stroke="#ccc" stroke-dasharray="4 3"/>')
    ticks = []
    if log_x:
        lo, hi = math.floor(x_min), math.ceil(x_max)
        ticks = [m * 10 ** k for k in range(int(lo), int(hi) + 1) for m in (1, 2, 5)]
        ticks = [t for t in ticks if x_min <= math.log10(t) <= x_max]
    for tick in ticks:
        x = px(tick)
        if left <= x <= width - right:
            label = f"{tick:g} s"
            parts.append(f'<line x1="{x:.1f}" y1="{height - bottom}" x2="{x:.1f}" y2="{height - bottom + 4}" stroke="#888"/>')
            parts.append(f'<text x="{x:.1f}" y="{height - bottom + 16}" text-anchor="middle" fill="#555">{_esc(label)}</text>')
    for tick in (y_min + pad_y, (y_min + y_max) / 2, y_max - pad_y):
        y = py(tick)
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{tick:.0%}</text>')
    for index, point in enumerate(points):
        colour = _PALETTE[index % len(_PALETTE)]
        x, y = px(float(point["x"])), py(float(point["y"]))
        tip = f'{point["label"]}: {float(point["x"]):.1f} s, gap {float(point["y"]):.1%}'
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}"><title>{_esc(tip)}</title></circle>')
        label = str(point["label"])
        if x + 9 + 7.0 * len(label) > width - 4:  # would run off the right edge: label on the left
            parts.append(f'<text x="{x - 9:.1f}" y="{y + 4:.1f}" text-anchor="end" fill="#222">{_esc(label)}</text>')
        else:
            parts.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" fill="#222">{_esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------------------------


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _median(values: Iterable[float]) -> float | None:
    data = sorted(v for v in values if v is not None and math.isfinite(v))
    if not data:
        return None
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else 0.5 * (data[mid - 1] + data[mid])


def collect_run_report_data(
    output_dir: str | Path,
    *,
    dataset_ids: Sequence[str],
    run_info: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    config_yaml: str = "",
    warnings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    import pandas as pd

    from qsarena.config import model_family

    output = Path(output_dir)
    config = dict(config or {})
    protocol = str(((config.get("selection") or {}).get("protocol")) or "both")
    eval_cfg = config.get("evaluation") or {}
    regression_metrics = list(eval_cfg.get("regression_metrics") or ["rmse", "mae", "r2", "spearman"])
    classification_metrics = list(eval_cfg.get("classification_metrics") or ["roc_auc", "auprc", "balanced_accuracy", "mcc"])

    datasets: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    ad_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    family_seconds: dict[str, list[float]] = {}
    family_gaps: dict[str, list[float]] = {}
    won: dict[str, dict[str, int]] = {"test": {}, "cv": {}}

    for dataset_id in dataset_ids:
        dataset_dir = output / dataset_id
        status = _read_status(dataset_dir)
        metrics = load_dataset_metrics(dataset_dir)
        task = str(status.get("task_type") or "")
        entry = {
            "dataset": dataset_id,
            "status": status.get("status", "not_run"),
            "task": task,
            "n_rows": status.get("n_rows"),
            "primary_metric": status.get("primary_metric"),
            "error": status.get("error") or status.get("reason") or "",
            "remedy": status.get("remedy") or "",
            "cleanup": status.get("cleanup_counts") or {},
        }
        datasets.append(entry)
        ad = status.get("applicability_domain") or {}
        if ad:
            ad_rows.append({"dataset": dataset_id, **ad})
        if metrics is None or metrics.empty:
            continue
        if not task:
            task = "classification" if str(_dataset_metric(metrics, None)) in _HIGHER_IS_BETTER - {"r2", "spearman", "pearson"} else "regression"
            entry["task"] = task
        best = best_model_rows(metrics, primary_metric=entry["primary_metric"] or None)
        metric = best["metric"]
        entry["primary_metric"] = metric
        shown = classification_metrics if task == "classification" else regression_metrics
        by_model = {str(r.get("model", "")): r for r in _ok_rows(metrics).to_dict(orient="records")}
        for key in ("test", "cv"):
            if protocol not in (key, "both"):
                continue
            selected = best.get(key)
            if not selected:
                best_rows.append({"dataset": dataset_id, "task": task, "protocol": key, "model": "", "family": "", "metric": metric,
                                  "note": "no model reports a CV score" if key == "cv" else "no successful model"})
                continue
            family = model_family(selected["model"])
            won[key][family] = won[key].get(family, 0) + 1
            row = by_model.get(selected["model"], {})
            best_rows.append({
                "dataset": dataset_id,
                "task": task,
                "protocol": key,
                "model": selected["model"],
                "family": family,
                "metric": metric,
                "test_value": selected["test_value"],
                "cv_value": selected["cv_value"],
                **{f"test_{m}": _finite(row.get(f"test_{m}")) for m in shown},
            })
        # cost vs gap
        best_test = best.get("test")
        lower = metric_lower_is_better(metric)
        if best_test and best_test["test_value"] is not None:
            reference = best_test["test_value"]
            for row in _ok_rows(metrics).to_dict(orient="records"):
                value = _test_value(row, metric)
                seconds = _finite(row.get("stage_duration_seconds"))
                if value is None:
                    continue
                gap = (value - reference) / abs(reference) if lower else (reference - value) / abs(reference)
                family = model_family(row.get("model"))
                family_gaps.setdefault(family, []).append(max(0.0, gap) if abs(reference) > 1e-12 else 0.0)
                if seconds is not None:
                    family_seconds.setdefault(family, []).append(seconds)
        # leaderboard
        lb = metrics.dropna(subset=["leaderboard_metric_name"]) if "leaderboard_metric_name" in metrics else metrics.iloc[0:0]
        lb = lb[lb["leaderboard_metric_name"].astype(str).str.strip() != ""] if len(lb) else lb
        if len(lb):
            lb_metric = str(lb["leaderboard_metric_normalized"].dropna().iloc[0]) if "leaderboard_metric_normalized" in lb and lb["leaderboard_metric_normalized"].notna().any() else metric
            matching = _ok_rows(lb)
            matching = matching[matching.get("primary_metric", pd.Series(dtype=str)).astype(str).str.lower() == lb_metric.lower()] if "primary_metric" in matching else matching.iloc[0:0]
            top10 = int(pd.to_numeric(lb["leaderboard_top10_count"], errors="coerce").fillna(0).max()) if "leaderboard_top10_count" in lb else 0
            if len(matching):
                values = pd.to_numeric(matching["primary_metric_value"], errors="coerce")
                pick = values.idxmin() if metric_lower_is_better(lb_metric) else values.idxmax()
                chosen = matching.loc[pick]
                rank = _finite(chosen.get("leaderboard_estimated_rank_vs_top10"))
                cautions = []
                if top10 < 10:
                    cautions.append(f"sparse reference ({top10} published entries)")
                cautions.append("test-selected")
                leaderboard_rows.append({
                    "dataset": dataset_id,
                    "leaderboard_metric": lb_metric,
                    "reference_best": _finite(chosen.get("leaderboard_top10_best_reference")),
                    "model": str(chosen.get("model", "")),
                    "value": _finite(chosen.get("primary_metric_value")),
                    "estimated_rank_vs_top10": int(rank) if rank is not None else None,
                    "reference_entries": top10,
                    "caution": "; ".join(cautions),
                })
        # failures and skips
        for row in metrics.to_dict(orient="records"):
            error = str(row.get("error", "") or "").strip()
            status_text = str(row.get("status", "") or "").strip().lower()
            if error.lower() in {"nan", "none"}:
                error = ""
            if status_text.startswith("skipped"):
                skipped.append({"dataset": dataset_id, "model": str(row.get("model", "")), "reason": error or status_text})
            elif error:
                failures.append({"dataset": dataset_id, "model": str(row.get("model", "")), "error": error[:300], "remedy": remedy_for_error(error)})

    cost_vs_gap = []
    for family in sorted(set(family_seconds) | set(family_gaps)):
        seconds = _median(family_seconds.get(family, []))
        gap = _median(family_gaps.get(family, []))
        if seconds is not None and gap is not None:
            cost_vs_gap.append({"family": family, "median_seconds": seconds, "median_relative_gap": gap,
                                "n_fits": len(family_gaps.get(family, []))})

    counts: dict[str, int] = {}
    for entry in datasets:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1

    data = {
        "title": "QSARena run report",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output),
        "run": dict(run_info or {}),
        "protocol": protocol,
        "counts": counts,
        "datasets": datasets,
        "best_models": best_rows,
        "leaderboard": leaderboard_rows,
        "families_won": won,
        "cost_vs_gap": cost_vs_gap,
        "ad_coverage": ad_rows,
        "model_failures": failures,
        "skipped_stages": skipped,
        "warnings": [dict(w) for w in warnings],
        "preflight": _read_json(output / "preflight.json", default={}) or {},
        "shown_metrics": {"regression": regression_metrics, "classification": classification_metrics},
        "config": config,
        "config_yaml": config_yaml,
        "files": {
            name: name for name in (
                "run_config.json", "run_config.yaml", "environment_manifest.json", "dataset_summary.csv",
                "summary_metrics.csv", "predictions.csv", "run.log", "events.jsonl", "preflight.json",
            ) if (output / name).exists()
        },
    }
    data["what_next"] = what_next_items(data)
    return data


def _read_status(dataset_dir: Path) -> dict[str, Any]:
    return _read_json(dataset_dir / "run_status.json", default={}) or {}


def what_next_items(data: Mapping[str, Any]) -> list[dict[str, str]]:
    """Actionable follow-ups derived from the run's outcome."""
    items: list[dict[str, str]] = []
    run = data.get("run") or {}
    output_dir = str(data.get("output_dir", "RUN_DIR"))
    failed = [d for d in data.get("datasets", []) if d.get("status") == "failed"]
    for entry in failed:
        items.append({
            "text": f"Fix dataset {entry['dataset']}: {entry.get('error', '')[:160]}" + (f" ({entry['remedy']})" if entry.get("remedy") else ""),
            "command": f'qsarena-benchmark --config "{output_dir}/run_config.yaml" --resume',
        })
    remedies: dict[str, list[str]] = {}
    for failure in data.get("model_failures", []):
        remedies.setdefault(failure["remedy"], []).append(failure["model"])
    for remedy, models in remedies.items():
        unique = sorted(set(models))
        items.append({"text": f"{len(models)} model fit(s) failed ({', '.join(unique[:4])}{' ...' if len(unique) > 4 else ''}). {remedy}", "command": ""})
    best = data.get("best_models", [])
    by_dataset: dict[str, dict[str, str]] = {}
    for row in best:
        by_dataset.setdefault(row["dataset"], {})[row["protocol"]] = row.get("model", "")
    disagreements = [d for d, picks in by_dataset.items() if picks.get("test") and picks.get("cv") and picks["test"] != picks["cv"]]
    if disagreements:
        items.append({
            "text": f"On {len(disagreements)} dataset(s) the test-selected winner differs from the CV-selected one "
                    f"({', '.join(disagreements[:5])}). Choosing on the test set is optimistic; report the CV-selected model "
                    "and its test score as the honest estimate.",
            "command": "",
        })
    small = [d["dataset"] for d in data.get("datasets", []) if isinstance(d.get("n_rows"), (int, float)) and d["n_rows"] and d["n_rows"] < 100]
    if small or run.get("single_seed", True):
        items.append({
            "text": "All scores come from one train/test split and one seed. Estimate split-to-split variance by repeating "
                    "the run with other seeds" + (f" (especially the small dataset(s) {', '.join(small[:5])})" if small else "") + ".",
            "command": f'qsarena-benchmark --config "{output_dir}/run_config.yaml" --random-seed 1 --output-dir "{output_dir}_seed1"',
        })
    if run.get("tdc22_datasets") and not run.get("tdc22_multiseed_done"):
        items.append({
            "text": "Some datasets are official TDC ADMET Benchmark Group tasks: get the leaderboard-style mean ± SD over five seeds.",
            "command": f'qsarena-benchmark --tdc22-multiseed --tdc22-multiseed-source-run "{output_dir}" --output-dir "{output_dir}_multiseed"',
        })
    ood = [a for a in data.get("ad_coverage", []) if _finite(a.get("in_domain_fraction")) is not None and a["in_domain_fraction"] < 0.8]
    for entry in ood:
        items.append({
            "text": f"{1 - entry['in_domain_fraction']:.0%} of the test molecules of {entry['dataset']} fall outside the applicability "
                    "domain; read their predictions with caution (see applicability_domain.csv in that dataset's folder).",
            "command": "",
        })
    if not data.get("ad_coverage"):
        items.append({
            "text": "No applicability-domain flags were computed in this run. Enable them to see which predictions are extrapolations.",
            "command": "qsarena-benchmark ... --ad-method both",
        })
    if run.get("profile") == "quick":
        items.append({
            "text": "This was a quick-profile run (scikit-learn models only). Run the full model library once the data look right.",
            "command": f'qsarena-benchmark --config "{output_dir}/run_config.yaml" --benchmark-profile cost_optimized --output-dir "{output_dir}_full"',
        })
    if not run.get("gpu_available", False) and run.get("deep_skipped_without_gpu"):
        items.append({
            "text": "Uni-Mol was skipped because no GPU was detected. On a CUDA machine with qsarena[foundation] installed it runs automatically.",
            "command": "",
        })
    items.append({
        "text": "Check new molecules against the training set's applicability domain before trusting their predictions.",
        "command": "qsarena-applicability-domain --train_csv TRAIN.csv --smiles_col smiles --target_col target --query_smiles \"CCO\"",
    })
    return items


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


def _plots(data: Mapping[str, Any]) -> dict[str, str]:
    plots: dict[str, str] = {}
    won = data.get("families_won", {})
    protocol = data.get("protocol", "both")
    key = "cv" if protocol == "cv" else "test"
    counts = won.get(key) or {}
    if counts:
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        plots["won_by_family"] = svg_bar_chart(
            [k for k, _ in items], [float(v) for _, v in items],
            title=f"Datasets won by model family ({'CV' if key == 'cv' else 'test'}-selected)",
            value_label="number of datasets on which the family produced the best model",
        )
    points = [
        {"x": max(1e-3, row["median_seconds"]), "y": row["median_relative_gap"], "label": row["family"]}
        for row in data.get("cost_vs_gap", [])
    ]
    if points:
        plots["cost_vs_gap"] = svg_scatter(
            points, title="Cost versus gap to the best model",
            x_label="median wall-clock per fit (s, log scale)", y_label="median relative gap to best test score",
        )
    ad = [a for a in data.get("ad_coverage", []) if _finite(a.get("in_domain_fraction")) is not None]
    if ad:
        plots["ad_coverage"] = svg_bar_chart(
            [a["dataset"] for a in ad], [float(a["in_domain_fraction"]) for a in ad],
            title="Applicability-domain coverage of the test set", max_value=1.0, percent=True,
            value_label="fraction of test molecules inside the applicability domain (all enabled methods agree)",
        )
    return plots


_PLOT_CAPTIONS = {
    "won_by_family": "Which model family produced each dataset's best model.",
    "cost_vs_gap": "Each point is a model family: its median fitting time against its median relative gap to the best "
                   "test score of the same dataset (0% = it was the best). Lower-left is cheap and competitive.",
    "ad_coverage": "Share of each dataset's test molecules that every enabled applicability-domain method places inside the domain.",
}


def _counts_line(data: Mapping[str, Any]) -> str:
    counts = data.get("counts") or {}
    total = sum(counts.values())
    parts = [f"{counts.get(k, 0)} {k}" for k in ("completed", "resumed", "failed", "skipped") if counts.get(k)]
    return f"{total} dataset(s): " + (", ".join(parts) if parts else "none")


def _best_tables(data: Mapping[str, Any]) -> list[tuple[str, list[str], list[list[str]]]]:
    """One (title, header, rows) table per task present in the run."""
    tables = []
    for task in ("regression", "classification"):
        rows = [row for row in data.get("best_models", []) if (row.get("task") or "regression") == task]
        if rows:
            header, body = _best_table(data, rows, task)
            tables.append((task.capitalize(), header, body))
    return tables


_CV_NOTE = (
    "CV scores are computed on the training split after train-only feature selection on that same split, so they "
    "are optimistic in absolute terms: use them to rank models, and quote the test score of the CV-selected model."
)


def _best_table(data: Mapping[str, Any], selected_rows: Sequence[Mapping[str, Any]], task: str) -> tuple[list[str], list[list[str]]]:
    shown = data.get("shown_metrics", {})
    ordered = list(shown.get(task, []))
    header = ["Dataset", "Selected by", "Best model", "Family", "Primary metric", "CV score", "Test score"]
    header += [f"Test {_METRIC_LABELS.get(m, m)}" for m in ordered]
    rows = []
    for row in selected_rows:
        cells = [
            row["dataset"], "CV (honest)" if row["protocol"] == "cv" else "test (optimistic)",
            row.get("model") or f"– ({row.get('note', '')})", row.get("family", ""),
            _METRIC_LABELS.get(row.get("metric", ""), row.get("metric", "")),
            _fmt(row.get("cv_value")), _fmt(row.get("test_value")),
        ]
        cells += [_fmt(row.get(f"test_{m}")) for m in ordered]
        rows.append(cells)
    return header, rows


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(clean(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(clean(c) for c in row) + " |" for row in rows]
    return lines


def render_markdown_report(data: Mapping[str, Any], *, asset_dir: str = "report_assets") -> str:
    run = data.get("run") or {}
    lines = [
        f"# {data.get('title', 'QSARena run report')}",
        "",
        f"- Output directory: `{data.get('output_dir')}`",
        f"- Generated: {data.get('generated_at')} with qsarena {run.get('qsarena_version', '')}",
        f"- Mode: {run.get('mode', '')}; profile: {run.get('profile', '')}; GPU: {'yes' if run.get('gpu_available') else 'no'}",
        f"- {_counts_line(data)}",
        f"- Config signature: `{run.get('config_signature', '')}`",
        "",
    ]
    warnings = data.get("warnings") or []
    lines += ["## Warnings", ""]
    if warnings:
        for w in warnings:
            where = f"**{w.get('dataset')}**: " if w.get("dataset") else ""
            lines.append(f"- {where}{w.get('message', '')}" + (f" *Remedy:* {w['remedy']}" if w.get("remedy") else ""))
    else:
        lines.append("None.")
    lines.append("")

    lines += ["## Datasets", ""]
    ds_rows = []
    for d in data.get("datasets", []):
        cleanup = d.get("cleanup") or {}
        dropped = cleanup.get("unparseable_smiles", "")
        ds_rows.append([d["dataset"], d.get("status", ""), d.get("task", ""), d.get("n_rows") or "", dropped,
                        _METRIC_LABELS.get(str(d.get("primary_metric") or ""), d.get("primary_metric") or ""),
                        (str(d.get("error", ""))[:120] + (f" – {d['remedy']}" if d.get("remedy") else "")) if d.get("error") else ""])
    lines += _md_table(["Dataset", "Status", "Task", "Rows used", "Unparseable SMILES dropped", "Primary metric", "Error / remedy"], ds_rows)
    lines.append("")

    lines += ["## Best model per dataset", ""]
    protocol = data.get("protocol", "both")
    lines.append(
        "Selection protocol: **" + protocol + "**. *Test-selected* picks the best held-out score, which is optimistic "
        "because the test set chooses. *CV-selected* picks the best cross-validated training score (only models that "
        "report one) and then shows its untouched test score: the honest estimate."
    )
    lines += ["", _CV_NOTE, ""]
    tables = _best_tables(data)
    if not tables:
        lines.append("No successful model.")
    for title, header, rows in tables:
        if len(tables) > 1:
            lines += [f"**{title}**", ""]
        lines += _md_table(header, rows)
        lines.append("")
    lines.append("")

    lines += ["## Leaderboard / rank table", ""]
    if data.get("leaderboard"):
        lb_rows = [[r["dataset"], _METRIC_LABELS.get(r["leaderboard_metric"], r["leaderboard_metric"]), r["model"], _fmt(r["value"]),
                    _fmt(r["reference_best"]), r["estimated_rank_vs_top10"] if r["estimated_rank_vs_top10"] is not None else "–",
                    r["reference_entries"], r["caution"]] for r in data["leaderboard"]]
        lines += _md_table(["Dataset", "Leaderboard metric", "Our model", "Our value", "Published best", "Est. rank vs top-10",
                            "Published entries", "Caution"], lb_rows)
        lines += ["", "Ranks are estimated by inserting our test score into the cached published top-10; they are only "
                      "leaderboard-equivalent on official splits."]
    else:
        lines.append("No dataset in this run has a leaderboard reference (ranks exist only for the curated benchmark datasets).")
    lines.append("")

    plots = _plots(data)
    lines += ["## Plots", ""]
    if plots and data.get("include_plots", True):
        for name, _svg in plots.items():
            lines += [f"![{name}]({asset_dir}/{name}.svg)", "", _PLOT_CAPTIONS[name], ""]
    else:
        lines += ["No plots (no successful model, or report.include_plots is false).", ""]
    if not data.get("ad_coverage"):
        lines += ["Applicability domain: not run (applicability_domain.method: off).", ""]

    lines += ["## Model failures and skipped stages", ""]
    if data.get("model_failures"):
        lines += _md_table(["Dataset", "Model", "Error", "Remedy"],
                           [[f["dataset"], f["model"], f["error"][:160], f["remedy"]] for f in data["model_failures"]])
    else:
        lines.append("No model failed.")
    if data.get("skipped_stages"):
        lines += ["", "Skipped by design:", ""]
        lines += [f"- {s['dataset']} / {s['model']}: {str(s['reason'])[:160]}" for s in data["skipped_stages"]]
    lines.append("")

    if data.get("what_next_enabled", True):
        lines += ["## What to do next", ""]
        for item in data.get("what_next", []):
            lines.append(f"- {item['text']}")
            if item.get("command"):
                lines += ["", "  ```bash", f"  {item['command']}", "  ```", ""]
        lines.append("")

    lines += ["## Resolved configuration", "",
              "Rerun exactly this configuration with `qsarena-benchmark --config run_config.yaml` (in this directory).", ""]
    if data.get("config_yaml"):
        lines += ["```yaml", str(data["config_yaml"]).rstrip(), "```", ""]
    lines += ["## Files", ""]
    lines += [f"- `{name}`" for name in data.get("files", {})]
    lines.append("")
    return "\n".join(lines)


_CSS = """
body{font-family:Helvetica,Arial,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1d1d1d;background:#fff;line-height:1.45}
h1{font-size:1.6em;margin-bottom:.2em}h2{font-size:1.2em;margin-top:1.6em;border-bottom:1px solid #ddd;padding-bottom:.2em}
table{border-collapse:collapse;margin:.6em 0;font-size:.9em;display:block;overflow-x:auto}
th,td{border:1px solid #d6d6d6;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f3f5f7}
code,pre{background:#f5f5f5;border-radius:3px}pre{padding:10px;overflow-x:auto;font-size:.85em}
.warn{border-left:4px solid #D55E00;background:#fff6f0;padding:6px 10px;margin:4px 0}
.note{color:#555;font-size:.92em}.plot{margin:12px 0 20px}.plot svg{max-width:100%;height:auto}
"""


def _html_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html_report(data: Mapping[str, Any]) -> str:
    run = data.get("run") or {}
    out = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{_esc(data.get('title', 'QSARena run report'))}</title><style>{_CSS}</style></head><body>",
        f"<h1>{_esc(data.get('title', 'QSARena run report'))}</h1>",
        "<ul>",
        f"<li>Output directory: <code>{_esc(data.get('output_dir'))}</code></li>",
        f"<li>Generated: {_esc(data.get('generated_at'))} with qsarena {_esc(run.get('qsarena_version', ''))}</li>",
        f"<li>Mode: {_esc(run.get('mode', ''))}; profile: {_esc(run.get('profile', ''))}; GPU: {'yes' if run.get('gpu_available') else 'no'}</li>",
        f"<li>{_esc(_counts_line(data))}</li>",
        f"<li>Config signature: <code>{_esc(run.get('config_signature', ''))}</code></li>",
        "</ul>",
        "<h2>Warnings</h2>",
    ]
    warnings = data.get("warnings") or []
    if warnings:
        for w in warnings:
            where = f"<strong>{_esc(w.get('dataset'))}</strong>: " if w.get("dataset") else ""
            remedy = f" <em>Remedy:</em> {_esc(w['remedy'])}" if w.get("remedy") else ""
            out.append(f"<div class=\"warn\">{where}{_esc(w.get('message', ''))}{remedy}</div>")
    else:
        out.append("<p>None.</p>")
    out.append("<h2>Datasets</h2>")
    ds_rows = []
    for d in data.get("datasets", []):
        cleanup = d.get("cleanup") or {}
        ds_rows.append([d["dataset"], d.get("status", ""), d.get("task", ""), d.get("n_rows") or "",
                        cleanup.get("unparseable_smiles", ""),
                        _METRIC_LABELS.get(str(d.get("primary_metric") or ""), d.get("primary_metric") or ""),
                        (str(d.get("error", ""))[:160] + (f" – {d['remedy']}" if d.get("remedy") else "")) if d.get("error") else ""])
    out.append(_html_table(["Dataset", "Status", "Task", "Rows used", "Unparseable SMILES dropped", "Primary metric", "Error / remedy"], ds_rows))
    out.append("<h2>Best model per dataset</h2>")
    out.append(
        f"<p class=\"note\">Selection protocol: <strong>{_esc(data.get('protocol', 'both'))}</strong>. <em>Test-selected</em> "
        "picks the best held-out score, which is optimistic because the test set chooses. <em>CV-selected</em> picks the "
        "best cross-validated training score (only models that report one) and then shows its untouched test score: the "
        "honest estimate.</p>"
    )
    out.append(f"<p class=\"note\">{_esc(_CV_NOTE)}</p>")
    tables = _best_tables(data)
    if not tables:
        out.append("<p>No successful model.</p>")
    for title, header, rows in tables:
        if len(tables) > 1:
            out.append(f"<h3>{_esc(title)}</h3>")
        out.append(_html_table(header, rows))
    out.append("<h2>Leaderboard / rank table</h2>")
    if data.get("leaderboard"):
        lb_rows = [[r["dataset"], _METRIC_LABELS.get(r["leaderboard_metric"], r["leaderboard_metric"]), r["model"], _fmt(r["value"]),
                    _fmt(r["reference_best"]), r["estimated_rank_vs_top10"] if r["estimated_rank_vs_top10"] is not None else "–",
                    r["reference_entries"], r["caution"]] for r in data["leaderboard"]]
        out.append(_html_table(["Dataset", "Leaderboard metric", "Our model", "Our value", "Published best",
                                "Est. rank vs top-10", "Published entries", "Caution"], lb_rows))
        out.append("<p class=\"note\">Ranks are estimated by inserting our test score into the cached published top-10; "
                   "they are only leaderboard-equivalent on official splits.</p>")
    else:
        out.append("<p>No dataset in this run has a leaderboard reference (ranks exist only for the curated benchmark datasets).</p>")
    out.append("<h2>Plots</h2>")
    plots = _plots(data)
    if plots and data.get("include_plots", True):
        for name, svg in plots.items():
            out.append(f"<div class=\"plot\" id=\"{name}\">{svg}<p class=\"note\">{_esc(_PLOT_CAPTIONS[name])}</p></div>")
    else:
        out.append("<p>No plots (no successful model, or report.include_plots is false).</p>")
    if not data.get("ad_coverage"):
        out.append("<p>Applicability domain: not run (applicability_domain.method: off).</p>")
    out.append("<h2>Model failures and skipped stages</h2>")
    if data.get("model_failures"):
        out.append(_html_table(["Dataset", "Model", "Error", "Remedy"],
                               [[f["dataset"], f["model"], f["error"][:200], f["remedy"]] for f in data["model_failures"]]))
    else:
        out.append("<p>No model failed.</p>")
    if data.get("skipped_stages"):
        out.append("<p>Skipped by design:</p><ul>" + "".join(
            f"<li>{_esc(s['dataset'])} / {_esc(s['model'])}: {_esc(str(s['reason'])[:200])}</li>" for s in data["skipped_stages"]) + "</ul>")
    if data.get("what_next_enabled", True):
        out.append("<h2>What to do next</h2><ul>")
        for item in data.get("what_next", []):
            command = f"<pre><code>{_esc(item['command'])}</code></pre>" if item.get("command") else ""
            out.append(f"<li>{_esc(item['text'])}{command}</li>")
        out.append("</ul>")
    out.append("<h2>Resolved configuration</h2>")
    out.append("<p>Rerun exactly this configuration with <code>qsarena-benchmark --config run_config.yaml</code> (in this directory).</p>")
    if data.get("config_yaml"):
        out.append(f"<pre><code>{_esc(data['config_yaml'])}</code></pre>")
    out.append("<h2>Files</h2><ul>" + "".join(f"<li><code>{_esc(n)}</code></li>" for n in data.get("files", {})) + "</ul>")
    out.append("</body></html>")
    return "\n".join(out)


def write_run_reports(
    output_dir: str | Path,
    data: Mapping[str, Any],
    *,
    include_plots: bool = True,
    what_next: bool = True,
    manifest: bool = True,
) -> tuple[Path, Path]:
    """Write report.html, report.md (always both), report_assets/*.svg and report_data.json."""
    from qsarena.artifacts import atomic_write_json, atomic_write_text

    output = Path(output_dir)
    payload = dict(data)
    payload["include_plots"] = bool(include_plots)
    payload["what_next_enabled"] = bool(what_next)
    if include_plots:
        for name, svg in _plots(payload).items():
            atomic_write_text(output / "report_assets" / f"{name}.svg", svg + "\n")
    html_path = atomic_write_text(output / "report.html", render_html_report(payload))
    md_path = atomic_write_text(output / "report.md", render_markdown_report(payload))
    if manifest:
        atomic_write_json(output / "report_data.json", payload)
    return html_path, md_path
