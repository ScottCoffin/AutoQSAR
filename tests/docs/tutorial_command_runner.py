"""Executable-docs runner for docs/tutorial.md (Additional file 2).

Conventions in docs/tutorial.md
-------------------------------
* Every fenced ``bash`` block is preceded by exactly one marker comment:
    ``<!-- doctest: run id=NAME [exit=N] [timeout=SECONDS] -->``  executed on the fixture, in order
    ``<!-- doctest: skip reason="..." -->``                         shown only (install steps, GPU, ...)
  Flags of ``qsarena-*`` commands in *every* block (run or skip) must exist in that command's --help.
* A fenced ``text`` block preceded by ``<!-- expect: NAME -->`` is a trimmed excerpt of the output of
  block NAME: each of its lines must appear, in order, in that output. ``...`` inside a line matches
  any text; a line that is exactly ``...`` marks omitted lines.
* Commands run without a shell, so they work the same on Windows and Linux. ``cd DIR`` changes the
  working directory for the following commands; ``ls PATH`` and ``cat PATH`` are portable built-ins;
  ``qsarena-*`` commands run the installed console scripts (or ``python -m`` in a bare checkout).
* ``<!-- BEGIN GENERATED: ... -->`` blocks are rewritten by ``python -m qsarena.config --write-docs``.

Refresh the plot images embedded in the tutorial from a fresh fixture run:

    python tests/docs/tutorial_command_runner.py --refresh-assets
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TUTORIAL = REPO / "docs" / "tutorial.md"
ASSET_DIR = REPO / "docs" / "tutorial_assets"

_MARKER = re.compile(r"<!--\s*doctest:\s*(?P<action>run|skip)\b(?P<opts>.*?)-->")
_EXPECT = re.compile(r"<!--\s*expect:\s*(?P<id>[\w-]+)\s*-->")
_FENCE = re.compile(r"^```(?P<lang>[\w-]*)\s*$")
_OPT = re.compile(r'(\w+)=("([^"]*)"|\S+)')

ENTRY_POINTS = {
    "qsarena-benchmark": "portable_colab_qsar_bundle.run_qsarena_benchmarks",
    "qsarena-applicability-domain": "portable_colab_qsar_bundle.simple_applicability_domain",
    "qsarena-examples": "qsarena.examples",
}


@dataclass
class CommandBlock:
    action: str
    options: dict[str, str]
    text: str
    line: int

    @property
    def id(self) -> str:
        return self.options.get("id", f"line{self.line}")


@dataclass
class ExpectBlock:
    target: str
    lines: list[str]
    line: int


@dataclass
class Tutorial:
    commands: list[CommandBlock] = field(default_factory=list)
    expects: list[ExpectBlock] = field(default_factory=list)
    yaml_blocks: list[tuple[int, str]] = field(default_factory=list)
    unmarked_bash: list[int] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)


def parse_tutorial(path: Path = TUTORIAL) -> Tutorial:
    lines = path.read_text(encoding="utf-8").splitlines()
    tutorial = Tutorial()
    pending_marker: re.Match | None = None
    pending_expect: str | None = None
    index = 0
    in_generated = False
    while index < len(lines):
        line = lines[index]
        if line.startswith("<!-- BEGIN GENERATED"):
            in_generated = True
        if line.startswith("<!-- END GENERATED"):
            in_generated = False
        if line.startswith("#") and not in_generated:
            tutorial.headings.append(line)
        marker = _MARKER.search(line)
        expect = _EXPECT.search(line)
        if marker:
            pending_marker = marker
        elif expect:
            pending_expect = expect.group("id")
        fence = _FENCE.match(line)
        if fence and fence.group("lang"):
            lang = fence.group("lang")
            start = index + 1
            end = start
            while end < len(lines) and not lines[end].startswith("```"):
                end += 1
            body = "\n".join(lines[start:end])
            if lang in {"bash", "sh", "shell"}:
                if pending_marker is None:
                    tutorial.unmarked_bash.append(index + 1)
                else:
                    options = {m.group(1): (m.group(3) if m.group(3) is not None else m.group(2))
                               for m in _OPT.finditer(pending_marker.group("opts"))}
                    tutorial.commands.append(CommandBlock(pending_marker.group("action"), options, body, index + 1))
            elif lang == "text" and pending_expect is not None:
                tutorial.expects.append(ExpectBlock(pending_expect, lines[start:end], index + 1))
            elif lang == "yaml" and not in_generated:
                tutorial.yaml_blocks.append((index + 1, body))
            pending_marker = None
            pending_expect = None
            index = end + 1
            continue
        index += 1
    return tutorial


def split_commands(text: str) -> list[list[str]]:
    """Commands of a block: comments dropped, backslash continuations joined, POSIX quoting."""
    commands: list[list[str]] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        buffer += stripped
        commands.append(shlex.split(buffer, comments=True))
        buffer = ""
    if buffer.strip():
        commands.append(shlex.split(buffer, comments=True))
    return [c for c in commands if c]


def command_prefix(name: str) -> list[str]:
    script_dir = Path(sys.executable).parent
    for candidate in (script_dir / name, script_dir / f"{name}.exe", script_dir / "Scripts" / f"{name}.exe"):
        if candidate.exists():
            return [str(candidate)]
    found = shutil.which(name)
    if found:
        return [found]
    return [sys.executable, "-m", ENTRY_POINTS[name]]


def documented_flags(tutorial: Tutorial) -> dict[str, set[str]]:
    flags: dict[str, set[str]] = {name: set() for name in ENTRY_POINTS}
    for block in tutorial.commands:
        for argv in split_commands(block.text):
            if argv[0] in flags:
                flags[argv[0]].update(token.split("=", 1)[0] for token in argv[1:] if token.startswith("--"))
    return flags


def help_flags(command: str, env: dict[str, str] | None = None) -> set[str]:
    output = subprocess.run(command_prefix(command) + ["--help"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env, check=True).stdout
    return set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9_-]*", output))


@dataclass
class BlockResult:
    block: CommandBlock
    output: str
    returncode: int
    expected_exit: int

    @property
    def ok(self) -> bool:
        return self.returncode == self.expected_exit


def _builtin(argv: list[str], cwd: Path) -> str:
    target = cwd / argv[1] if len(argv) > 1 else cwd
    if argv[0] == "ls":
        return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())) + "\n"
    return target.read_text(encoding="utf-8")


def run_tutorial(tutorial: Tutorial, workdir: Path, env: dict[str, str]) -> dict[str, BlockResult]:
    """Run every ``doctest: run`` block in order. Stops at the first unexpected exit code."""
    results: dict[str, BlockResult] = {}
    cwd = workdir
    for block in tutorial.commands:
        if block.action != "run":
            continue
        expected_exit = int(block.options.get("exit", "0"))
        timeout = float(block.options.get("timeout", "900"))
        outputs: list[str] = []
        returncode = 0
        for argv in split_commands(block.text):
            if argv[0] == "cd":
                cwd = (cwd / argv[1]).resolve()
                continue
            if argv[0] in {"ls", "cat"}:
                outputs.append(_builtin(argv, cwd))
                continue
            if argv[0] in ENTRY_POINTS:
                command = command_prefix(argv[0]) + argv[1:]
            elif argv[0] == "python":
                command = [sys.executable] + argv[1:]
            else:
                raise ValueError(f"line {block.line}: unsupported command {argv[0]!r} in an executed block")
            completed = subprocess.run(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
            outputs.append(completed.stdout)
            returncode = completed.returncode
            if returncode != 0:
                break
        result = BlockResult(block, "".join(outputs), returncode, expected_exit)
        results[block.id] = result
        if not result.ok:
            break
    return results


def _line_pattern(expected: str) -> re.Pattern:
    parts = [re.escape(part) for part in expected.strip().split("...")]
    return re.compile(".*".join(parts))


def match_excerpt(expected_lines: list[str], output: str) -> list[str]:
    """Problems (empty = match): every non-``...`` line must match, in order, some output line.
    Paths are compared with forward slashes."""
    actual = [line.strip().replace("\\", "/") for line in output.splitlines()]
    problems: list[str] = []
    position = 0
    for expected in expected_lines:
        text = expected.strip().replace("\\", "/")
        if not text or text == "...":
            continue
        pattern = _line_pattern(text)
        for offset in range(position, len(actual)):
            if pattern.fullmatch(actual[offset]):
                position = offset + 1
                break
        else:
            problems.append(text)
    return problems


def default_env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["QSARENA_HOME"] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def refresh_assets() -> int:  # pragma: no cover - maintenance entry point
    """Run the tutorial and copy the report plots it embeds into docs/tutorial_assets/."""
    tutorial = parse_tutorial()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results = run_tutorial(tutorial, tmp_path, default_env(tmp_path / "home"))
        failed = [r for r in results.values() if not r.ok]
        if failed:
            print(failed[0].output[-4000:])
            return 1
        source = tmp_path / "qsarena_tutorial" / "runs" / "batch" / "report_assets"
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        for svg in sorted(source.glob("*.svg")):
            shutil.copyfile(svg, ASSET_DIR / svg.name)
            print(f"refreshed docs/tutorial_assets/{svg.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh-assets", action="store_true")
    ns = parser.parse_args()
    raise SystemExit(refresh_assets() if ns.refresh_assets else 0)
