"""
qsarena.qmrf — QMRF-style model reports organised by the five OECD (Q)SAR principles.

This is a *QMRF-style* summary, not a filled-in official QMRF template: it arranges what a
QSARena run already records under the five OECD validation principles so a regulator or reviewer
can find it, and it refuses to emit a section it cannot populate.

    1. Defined endpoint
    2. Unambiguous algorithm
    3. Defined domain of applicability
    4. Goodness-of-fit, robustness and predictivity
    5. Mechanistic interpretation, if possible

``build_qmrf_report(run)`` takes a plain dict (see ``REQUIRED_FIELDS``) and raises
``QMRFInputError`` naming EVERY missing field at once, rather than silently writing "N/A".
``render_markdown`` and ``write_qmrf_report`` produce Markdown + JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "OECD_SECTIONS",
    "REQUIRED_FIELDS",
    "QMRFInputError",
    "build_qmrf_report",
    "render_markdown",
    "write_qmrf_report",
]

OECD_SECTIONS: tuple[str, ...] = (
    "1. Defined endpoint",
    "2. Unambiguous algorithm",
    "3. Defined domain of applicability",
    "4. Goodness-of-fit, robustness and predictivity",
    "5. Mechanistic interpretation",
)

#: Required input keys per section. Dotted names are nested dict lookups.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    OECD_SECTIONS[0]: ("dataset", "endpoint", "task_type", "units", "data_source"),
    OECD_SECTIONS[1]: (
        "model_name",
        "features",
        "software_version",
        "split.strategy",
        "split.train_hash",
        "split.test_hash",
        "random_seed",
    ),
    OECD_SECTIONS[2]: ("applicability_domain.method", "applicability_domain.test_coverage"),
    OECD_SECTIONS[3]: ("metrics.train", "metrics.test", "metrics.internal_validation"),
    OECD_SECTIONS[4]: ("interpretation.status",),
}


class QMRFInputError(ValueError):
    """Raised when a run lacks fields needed to populate an OECD section."""

    def __init__(self, missing: dict[str, list[str]]):
        self.missing = missing
        lines = [f"{section}: {', '.join(fields)}" for section, fields in missing.items()]
        super().__init__("cannot build QMRF report; missing inputs ->\n  " + "\n  ".join(lines))


def _get(run: Mapping[str, Any], dotted: str) -> Any:
    node: Any = run
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (str, list, tuple, dict)) and len(value) == 0:
        return True
    return False


def build_qmrf_report(run: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``run`` and return a report dict keyed by the five OECD section titles."""
    missing: dict[str, list[str]] = {}
    for section, fields in REQUIRED_FIELDS.items():
        absent = [f for f in fields if _is_missing(_get(run, f))]
        if absent:
            missing[section] = absent
    interp = _get(run, "interpretation") or {}
    if interp.get("status") == "feature" and _is_missing(interp.get("top_features")):
        missing.setdefault(OECD_SECTIONS[4], []).append("interpretation.top_features")
    if missing:
        raise QMRFInputError(missing)

    ad = run["applicability_domain"]
    return {
        "title": f"QMRF-style report: {run['model_name']} on {run['dataset']}",
        OECD_SECTIONS[0]: {
            "dataset": run["dataset"],
            "endpoint": run["endpoint"],
            "task_type": run["task_type"],
            "units": run["units"],
            "data_source": run["data_source"],
        },
        OECD_SECTIONS[1]: {
            "model": run["model_name"],
            "features": run["features"],
            "software": run["software_version"],
            "random_seed": run["random_seed"],
            "split": dict(run["split"]),
            "reproduction": run.get("reproduction", ""),
        },
        OECD_SECTIONS[2]: {
            "method": ad["method"],
            "test_coverage": ad["test_coverage"],
            "error_in_domain": ad.get("error_in_domain"),
            "error_out_of_domain": ad.get("error_out_of_domain"),
            "reliability": ad.get("reliability"),
        },
        OECD_SECTIONS[3]: {
            "train": run["metrics"]["train"],
            "internal_validation": run["metrics"]["internal_validation"],
            "test": run["metrics"]["test"],
            "uncertainty": run["metrics"].get("uncertainty"),
        },
        OECD_SECTIONS[4]: {
            "status": interp["status"],
            "top_features": interp.get("top_features", []),
            "note": interp.get("note", ""),
        },
        "caveats": list(run.get("caveats", [])),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, Mapping):
        return "; ".join(f"{k} = {_fmt(v)}" for k, v in value.items() if v is not None)
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# {report['title']}", ""]
    for section in OECD_SECTIONS:
        lines += [f"## {section}", ""]
        body = report[section]
        if section == OECD_SECTIONS[4] and body.get("top_features"):
            lines.append(f"- **status**: {body['status']}")
            if body.get("note"):
                lines.append(f"- **note**: {body['note']}")
            lines += ["", "| rank | feature | importance |", "|---:|---|---:|"]
            for row in body["top_features"]:
                lines.append(f"| {row['rank']} | `{row['feature']}` | {row['importance']:.4f} |")
        else:
            for key, value in body.items():
                if value is None or value == "":
                    continue
                lines.append(f"- **{key.replace('_', ' ')}**: {_fmt(value)}")
        lines.append("")
    if report.get("caveats"):
        lines += ["## Caveats", ""] + [f"- {c}" for c in report["caveats"]] + [""]
    return "\n".join(lines)


def write_qmrf_report(run: Mapping[str, Any], output_dir: str | Path, stem: str | None = None) -> tuple[Path, Path]:
    """Build, then write ``<stem>.md`` and ``<stem>.json``; returns both paths."""
    report = build_qmrf_report(run)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or f"qmrf_{run['dataset']}"
    md_path, json_path = out / f"{stem}.md", out / f"{stem}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return md_path, json_path
