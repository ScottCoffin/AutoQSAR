"""Rename guard: "autoqsar" must not reappear outside an explicit allowlist.

The tool was renamed AutoQSAR -> QSARena because Schrodinger ships a commercial product called
AutoQSAR in the same domain. Every tracked text file is scanned case-insensitively. A hit is
allowed only if

  * the file is historical / generated / third-party (``ALLOWED_PATHS``), or
  * the matching line is one of the deliberate uses in ``ALLOWED_LINE_PATTERNS``: citations of
    Schrodinger's AutoQSAR / DeepAutoQSAR as prior art, the canonical run's directory name
    (renaming it would break every provenance path in the paper), and pre-rename clone
    locations kept as fallbacks.

If this test fails, rename the new occurrence -- do not widen the allowlist unless the use is
genuinely about Schrodinger's product or a frozen historical artifact.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

ALLOWED_PATHS = (
    # Frozen artifacts, caches and logs.
    "benchmark_results/",
    "logs/",
    "data/",
    "model_cache",  # external volume symlink created before the rename
    "node_modules/",
    "package-lock.json",
    "environment/requirements-lock.txt",
    "manuscript_assets/",
    "manuscript.html",
    # Stale pre-rename mirrors (see AGENTS.md: never read these as source).
    "portable_colab_qsar_bundle/run_autoqsar_ga_benchmarks.txt",
    "portable_colab_qsar_bundle/qsar_workflow_core.txt",
    # Executed notebooks carry historical outputs; the Colab notebook is checked via its builder.
    "portable_colab_qsar_bundle/benchmark_results_summary.ipynb",
    "portable_colab_qsar_bundle/pfas_aux_qsar_results_summary.ipynb",
    # History and review documents that describe the rename itself.
    "AGENTS.md",
    "TODO.md",
    "CHANGELOG.md",
    "BUILD_PROVENANCE.md",
    "CHEMPROP_FIX_PLAN.md",
    "docs/AGENT_WORK_ORDER_shared_ensemble.md",
    "LEADERBOARD_PROVENANCE_FINDINGS.md",
    "Manuscript Outline.md",
    "publication_recommendations.md",
    "audit_report.md",
    "exec_summary.md",
    "test_data/",
    "submission/cover_letter.md",
    # External review documents received under the old name.
    "submission/Competitive_Positioning_Landscape_Review_AutoQSAR_Manuscript.docx",
    "submission/Peer_Review_AutoQSAR_Manuscript_Journal_of_Cheminformatics.docx",
    ".claude/",
    ".gitignore",  # keeps ignoring the pre-rename temp dir
    "tests/unit/test_rename_residue.py",
)

ALLOWED_LINE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"DeepAutoQSAR",
        r"Schr(?:\\\"o|ö|o)dinger(?:'s)?\s+AutoQSAR",
        r"dixon2016autoqsar",
        r"\{AutoQSAR\}: an automated",
        r"AutoQSAR: an automated machine learning tool",
        r"name \*AutoQSAR\* is used by an established commercial product",
        r"AutoQSAR automates model building",
        r"AUTOQSAR_[A-Z0-9_]+",
        r"autoqsar(?:\\?_)benchmark(?:\\?_)20260623(?:\\?_)153839",
        r"https://raw\.githubusercontent\.com/ScottCoffin/AutoQSAR/main/portable_colab_qsar_bundle/qsar_workflow_core\.py",
        r"^.*# clone made before the rename.*$",
    )
]


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.split("\0") if f]


def _allowed_path(path: str) -> bool:
    return any(path == a or path.startswith(a) for a in ALLOWED_PATHS)


def test_no_autoqsar_outside_allowlist():
    offenders = []
    for rel in _tracked_files():
        if _allowed_path(rel):
            continue
        path = REPO / rel
        if "autoqsar" in rel.lower():
            offenders.append(f"{rel}: file name")
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:4096]:  # binary
            continue
        text = raw.decode("utf-8", errors="replace")
        if "autoqsar" not in text.lower():
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "autoqsar" not in line.lower():
                continue
            # Strip each allowed phrase, then see if any "autoqsar" survives on the line.
            remainder = line
            for pat in ALLOWED_LINE_PATTERNS:
                remainder = pat.sub("", remainder)
            if "autoqsar" in remainder.lower():
                offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    assert not offenders, "AutoQSAR residue outside the allowlist:\n" + "\n".join(offenders)
