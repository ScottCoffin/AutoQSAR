"""D2: executable-docs guarantee for docs/tutorial.md (Additional file 2).

Every ``doctest: run`` command block of the tutorial is executed, in order, on the shipped example
data; every ``expect`` excerpt must match that command's real output; every flag shown for a
``qsarena-*`` command must exist in its ``--help``. See tests/docs/tutorial_command_runner.py for the
conventions.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qsarena import config as qc
from tests.docs import tutorial_command_runner as runner

pytestmark = pytest.mark.docs

REPO = Path(__file__).resolve().parents[2]
TUTORIAL = runner.parse_tutorial()

REQUIRED_SECTIONS = [
    "## 1. Overview and choosing an entry point",
    "## 2. Installation",
    "## 3. Quickstart: one dataset in five minutes",
    "## 4. The decision reference",
    "## 5. Batch mode on any number of datasets",
    "## 6. Resume",
    "## 7. Understanding the reports",
    "## 8. Applicability domain and reliability",
    "## 9. Reproducibility",
    "## 10. Troubleshooting and FAQ",
]


@pytest.fixture(scope="module")
def tutorial_run(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("tutorial")
    results = runner.run_tutorial(TUTORIAL, workdir, runner.default_env(workdir / "home"))
    return workdir, results


def test_tutorial_sections_present():
    headings = [h.strip() for h in TUTORIAL.headings]
    for section in REQUIRED_SECTIONS:
        assert section in headings, section
    for number, name in qc.GROUPS.items():
        assert f"### 4.{number} {name}" in headings, name


def test_every_bash_block_is_marked():
    assert not TUTORIAL.unmarked_bash, f"bash blocks without a doctest marker at lines {TUTORIAL.unmarked_bash}"
    for block in TUTORIAL.commands:
        if block.action == "skip":
            assert block.options.get("reason"), f"line {block.line}: a skipped block needs a reason"
    ids = [block.id for block in TUTORIAL.commands if block.action == "run"]
    assert len(ids) == len(set(ids))
    assert {e.target for e in TUTORIAL.expects} <= set(ids)


def test_all_tutorial_commands_run_on_fixture(tutorial_run):
    _workdir, results = tutorial_run
    expected = [block.id for block in TUTORIAL.commands if block.action == "run"]
    for block_id in expected:
        assert block_id in results, f"{block_id} did not run (an earlier block failed)"
        result = results[block_id]
        assert result.ok, (
            f"tutorial block {block_id} (line {result.block.line}) exited {result.returncode}, "
            f"expected {result.expected_exit}:\n{result.output[-4000:]}"
        )


def test_tutorial_expected_output_excerpts_match_fixture_run(tutorial_run):
    _workdir, results = tutorial_run
    failures = []
    for expect in TUTORIAL.expects:
        result = results.get(expect.target)
        assert result is not None, expect.target
        missing = runner.match_excerpt(expect.lines, result.output)
        if missing:
            failures.append(f"excerpt at line {expect.line} ({expect.target}): no match for {missing[0]!r}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("command", ["qsarena-benchmark", "qsarena-applicability-domain", "qsarena-examples"])
def test_documented_flags_exist_in_help(command):
    documented = runner.documented_flags(TUTORIAL)[command]
    available = runner.help_flags(command, env=runner.default_env(REPO))
    missing = sorted(flag for flag in documented if flag not in available)
    assert not missing, f"{command}: documented but not in --help: {missing}"


def test_flags_mentioned_in_prose_exist_in_help():
    """Inline `--flag` mentions in the text must also be real qsarena-benchmark flags."""
    text = runner.TUTORIAL.read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    prose = re.sub(r"<!-- BEGIN GENERATED.*?END GENERATED -->", "", prose, flags=re.DOTALL)
    mentioned = set(re.findall(r"`[^`]*?(--[a-z][a-z0-9_-]*[a-z0-9])", prose))
    available = (
        runner.help_flags("qsarena-benchmark", env=runner.default_env(REPO))
        | runner.help_flags("qsarena-applicability-domain", env=runner.default_env(REPO))
    )
    assert not sorted(mentioned - available)


def test_tutorial_yaml_blocks_are_valid_run_configs():
    for line, text in TUTORIAL.yaml_blocks:
        try:
            qc.RunConfig.from_dict(yaml.safe_load(text), source=f"docs/tutorial.md line {line}")
        except qc.ConfigError as exc:  # pragma: no cover - failure message
            pytest.fail(str(exc))


def test_generated_blocks_are_up_to_date():
    text = runner.TUTORIAL.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert qc.refresh_generated_blocks(text) == text, "run: python -m qsarena.config --write-docs"


def test_tutorial_plot_assets_match_fixture_run(tutorial_run):
    """The plots embedded in Section 7 come from the Section 5 batch run: same plots, same labels."""
    workdir, _results = tutorial_run
    generated = workdir / "qsarena_tutorial" / "runs" / "batch" / "report_assets"
    text_labels = re.compile(r">([^<>]+)</text>")
    for name in ("won_by_family.svg", "cost_vs_gap.svg", "ad_coverage.svg"):
        committed = (runner.ASSET_DIR / name).read_text(encoding="utf-8")
        fresh = (generated / name).read_text(encoding="utf-8")
        labels = {t for t in text_labels.findall(fresh) if not re.search(r"\d", t)}
        committed_labels = {t for t in text_labels.findall(committed) if not re.search(r"\d", t)}
        assert labels == committed_labels, f"{name}: refresh with tests/docs/tutorial_command_runner.py --refresh-assets"


def test_example_data_regenerates_identically(tmp_path):
    """qsarena/examples/data is exactly what make_tutorial_data.py writes."""
    script = REPO / "tests" / "fixtures" / "tutorial" / "make_tutorial_data.py"
    subprocess.run([sys.executable, str(script), "--out", str(tmp_path)], check=True, cwd=REPO, capture_output=True)
    shipped = REPO / "qsarena" / "examples" / "data"
    generated_files = sorted(tmp_path.rglob("*.csv"))
    assert len(generated_files) == 6
    for generated in generated_files:
        relative = generated.relative_to(tmp_path)
        committed = (shipped / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert generated.read_text(encoding="utf-8") == committed, relative
