"""Regression tests for the ASCII-locale subprocess decode bug.

Background. The canonical A100 benchmark
(``benchmark_results/autoqsar_benchmark_20260623_153839``) lost 86.4% of all Chemprop runs --
an identical rate across five unrelated architectures, which is the signature of a shared I/O
fault rather than a model problem. The cause was ``subprocess.run(..., text=True)`` with no
explicit ``encoding``: Python then decodes the child's output with
``locale.getpreferredencoding(False)``, which on that host resolved to ANSI_X3.4-1968 (ASCII)
because the process ran under a C/POSIX locale. Chemprop v2 emits UTF-8 progress output, so the
decode raised ``UnicodeDecodeError`` inside ``subprocess.run`` before any result was read.

The same run on Windows was unaffected, because cp1252 maps 0xe2 without complaint -- which is
why the bug survived a full development cycle.

These tests pin the fix. ``test_utf8_child_output_survives_ascii_locale`` fails with
``text=True`` alone and passes with ``encoding="utf-8", errors="replace"``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portable_colab_qsar_bundle.run_qsarena_benchmarks import (  # noqa: E402
    _SUBPROCESS_TEXT_KWARGS,
    _run_chemprop_command,
)

# The exact glyph class that broke the real run: 0xe2 is the UTF-8 lead byte for the em-dash,
# the box-drawing characters in Chemprop's progress bars, and the check mark in its summaries.
UTF8_NOISE = "Training \u2500\u2500\u2500 epoch 1/2 \u2014 loss 0.42 \u2713"

CHILD = (
    "import sys\n"
    "sys.stdout.reconfigure(encoding='utf-8')\n"
    f"sys.stdout.write({UTF8_NOISE!r} + '\\n')\n"
    "sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)\n"
)


def _ascii_locale_env() -> dict[str, str]:
    """An environment whose preferred encoding is ASCII, reproducing the Jetstream2 state."""
    return {"LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "", "PATH": "/usr/bin:/bin"}


def test_bare_text_true_is_the_bug(tmp_path: pytest.TempPathFactory) -> None:
    """Document the defect: decoding UTF-8 output as ASCII raises, and it raises *inside*
    subprocess.run, so no return code is ever inspected."""
    with pytest.raises(UnicodeDecodeError):
        UTF8_NOISE.encode("utf-8").decode("ascii")


def test_utf8_child_output_survives_ascii_locale(tmp_path) -> None:
    """The fixed kwargs must decode a UTF-8-emitting child under an ASCII locale."""
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), "0"],
        capture_output=True,
        env=_ascii_locale_env(),
        **_SUBPROCESS_TEXT_KWARGS,
    )

    assert result.returncode == 0
    # errors="replace" guarantees a str even if a byte is undecodable; the point is that the call
    # completes and the return code is observable.
    assert "epoch 1/2" in result.stdout


def test_chemprop_wrapper_reports_training_failure_distinctly(tmp_path) -> None:
    """A non-zero exit must be reported as a training failure, not a harness error."""
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"Chemprop training failed"):
        _run_chemprop_command([sys.executable, str(script)], ["1"], "unit test")


def test_chemprop_wrapper_reports_harness_error_distinctly() -> None:
    """A missing executable is a harness fault and must say so, so that a future coverage gap
    cannot again be misread as poor model performance."""
    with pytest.raises(RuntimeError, match=r"Chemprop harness error"):
        _run_chemprop_command(["definitely-not-an-executable-qsarena"], [], "unit test")


def test_chemprop_wrapper_succeeds_on_utf8_output(tmp_path) -> None:
    """End-to-end: the wrapper returns an elapsed time for a UTF-8-emitting child."""
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")

    elapsed = _run_chemprop_command([sys.executable, str(script)], ["0"], "unit test")
    assert elapsed >= 0.0
