"""Shared pytest setup: make the source checkout importable and expose the test fixtures.

``tests/fixtures/tdc_tiny`` holds 36 train_val + 12 test molecules sampled from the official
TDC splits of one regression (caco2_wang) and one classification (herg) dataset, so the
integration tests exercise real chemistry in seconds with no network, GPU or ``data/`` tree.

The tutorial fixtures are the synthetic example datasets shipped in ``qsarena/examples/data``
(<= 50 molecules each; regenerate with ``tests/fixtures/tutorial/make_tutorial_data.py``).
``run_cli`` runs ``qsarena-benchmark`` exactly as a user would: the installed console script when
it is on PATH (CI installs the package), otherwise ``python -m`` on the runner module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "tdc_tiny"
EXAMPLES_DIR = REPO_ROOT / "qsarena" / "examples" / "data"

#: Console scripts and the module each one runs when the script is not installed.
_ENTRY_POINTS = {
    "qsarena-benchmark": "portable_colab_qsar_bundle.run_qsarena_benchmarks",
    "qsarena-applicability-domain": "portable_colab_qsar_bundle.simple_applicability_domain",
    "qsarena-examples": "qsarena.examples",
}


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tdc_tiny() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def qsarena_home(tmp_path_factory) -> Path:
    """One QSARENA_HOME per test session so feature caches are shared between CLI runs."""
    return tmp_path_factory.mktemp("qsarena_home")


def command_prefix(name: str) -> list[str]:
    """Argument prefix that invokes a console script (installed script if available)."""
    script_dir = Path(sys.executable).parent
    for candidate in (script_dir / name, script_dir / f"{name}.exe", script_dir / "Scripts" / f"{name}.exe"):
        if candidate.exists():
            return [str(candidate)]
    found = shutil.which(name)
    if found:
        return [found]
    return [sys.executable, "-m", _ENTRY_POINTS[name]]


def cli_env(qsarena_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["QSARENA_HOME"] = str(qsarena_home)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


@pytest.fixture
def run_cli(qsarena_home):
    """``run_cli(["qsarena-benchmark", ...], cwd=...)`` -> CompletedProcess (text, merged output)."""

    def _run(argv: list[str], *, cwd: Path, timeout: float = 900, check: bool = True) -> subprocess.CompletedProcess:
        command = command_prefix(argv[0]) + [str(a) for a in argv[1:]]
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=cli_env(qsarena_home),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if check and completed.returncode != 0:
            raise AssertionError(f"{' '.join(command)} exited {completed.returncode}:\n{completed.stdout[-6000:]}")
        return completed

    return _run


@pytest.fixture
def examples(tmp_path) -> Path:
    """A fresh copy of the tutorial example data (what ``qsarena-examples DIR`` writes)."""
    from qsarena.examples import copy_examples

    target = tmp_path / "ex"
    copy_examples(target)
    return target


#: Fast model subset for integration tests that are about the harness, not the models.
FAST_MODELS = ["--benchmark-profile", "quick", "--selector-method", "rf_importance", "--n-jobs", "1", "--use-gpu", "false",
               "--only-model-names", "ElasticNetCV", "--only-model-names", "Random forest",
               "--only-model-names", "SVR", "--only-model-names", "LogisticRegression",
               "--only-model-names", "SVC", "--only-model-names", "Ensemble"]
