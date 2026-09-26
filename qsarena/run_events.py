"""
qsarena.run_events — structured run log, JSON event stream and actionable warnings.

A run writes three complementary records into its output directory:

  run.log        the complete console transcript (everything printed, whatever the verbosity)
  events.jsonl   one JSON object per line: run/preflight/dataset/stage/model/warning events
  (report.*)     warnings collected here are also listed in report.html / report.md

``run.verbosity`` selects what reaches the console and events.jsonl:

  quiet    warnings, errors and the final summary lines only
  normal   + dataset- and model-level progress (default)
  verbose  + stage-level events and echoed backend commands
  debug    + debug events and full tracebacks
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "LEVELS",
    "VERBOSITY_LEVELS",
    "RunEventLog",
    "EVENTS",
    "ConsoleTee",
    "quiet_console_line",
    "tqdm_progress_enabled",
]

LEVELS = {"debug": 10, "detail": 15, "info": 20, "warning": 30, "error": 40}
VERBOSITY_LEVELS = {"quiet": 30, "normal": 20, "verbose": 15, "debug": 10}

#: Console lines kept in quiet mode (prefix match after stripping leading whitespace).
_QUIET_PREFIXES = (
    "[warn]",
    "[error]",
    "[fail",
    "[preflight] warning",
    "error",
    "traceback",
    "wrote benchmark outputs",
    "report:",
    "reports written",
    "batch summary",
    "dataset summary",
    "dry run complete",
)


def quiet_console_line(line: str) -> bool:
    """True if a console line should still be shown with ``--verbosity quiet``."""
    text = line.strip().lower()
    return bool(text) and text.startswith(_QUIET_PREFIXES)


def tqdm_progress_enabled(stream: Any = None) -> bool:
    """A tqdm dataset bar is drawn only on an interactive terminal with tqdm installed."""
    stream = stream if stream is not None else sys.__stdout__
    try:
        import tqdm  # noqa: F401
    except ImportError:
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


@dataclass
class RunWarning:
    code: str
    message: str
    remedy: str = ""
    dataset: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "remedy": self.remedy, "dataset": self.dataset}


@dataclass
class RunEventLog:
    """Process-wide event sink. Unconfigured (e.g. inside a dataset worker process) it is a no-op
    apart from collecting warnings."""

    path: Path | None = None
    threshold: int = LEVELS["info"]
    started: float = field(default_factory=time.time)
    warnings: list[RunWarning] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def configure(self, output_dir: str | Path, verbosity: str = "normal") -> None:
        self.path = Path(output_dir) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = VERBOSITY_LEVELS.get(str(verbosity), LEVELS["info"])
        self.started = time.time()
        self.warnings = []

    def reset(self) -> None:
        self.path = None
        self.threshold = LEVELS["info"]
        self.warnings = []

    def emit(self, event: str, level: str = "info", **fields: Any) -> None:
        if self.path is None or LEVELS.get(level, 20) < self.threshold:
            return
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "level": level,
            "event": event,
        }
        record.update({key: _jsonable(value) for key, value in fields.items()})
        line = json.dumps(record, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def warn(self, code: str, message: str, *, remedy: str = "", dataset: str = "", echo: bool = True) -> RunWarning:
        """Record an actionable warning: printed as ``[warn] ...``, written to events.jsonl and
        listed in the report."""
        warning = RunWarning(code=code, message=message, remedy=remedy, dataset=dataset)
        base_code = code.removeprefix("preflight_")
        for existing in self.warnings:
            # One warning per (topic, dataset): the in-run check repeats what preflight already said.
            if existing.dataset == dataset and existing.code.removeprefix("preflight_") == base_code:
                return existing
        self.warnings.append(warning)
        if echo:
            prefix = f"[warn] {dataset}: " if dataset else "[warn] "
            print(prefix + message + (f" Remedy: {remedy}" if remedy else ""), flush=True)
        self.emit("warning", level="warning", **warning.as_dict())
        return warning


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(v) for v in value]
        return str(value)


EVENTS = RunEventLog()


class ConsoleTee(io.TextIOBase):
    """Replace ``sys.stdout``/``sys.stderr``: every line goes to run.log; the console receives
    everything, or only :func:`quiet_console_line` lines in quiet mode."""

    def __init__(self, console: TextIO, log_handle: TextIO, *, quiet: bool = False) -> None:
        super().__init__()
        self._console = console
        self._log = log_handle
        self._quiet = bool(quiet)
        self._pending = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:  # pragma: no cover - some libraries query it
        return getattr(self._console, "encoding", "utf-8") or "utf-8"

    def isatty(self) -> bool:
        try:
            return bool(self._console.isatty())
        except Exception:
            return False

    def fileno(self) -> int:  # pragma: no cover - needed by faulthandler / subprocess helpers
        return self._console.fileno()

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            try:
                self._log.write(text)
            except Exception:  # pragma: no cover - never let logging break a run
                pass
            if not self._quiet:
                self._console.write(text)
                return len(text)
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                if quiet_console_line(line):
                    self._console.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        with self._lock:
            try:
                self._log.flush()
            except Exception:  # pragma: no cover
                pass
            try:
                self._console.flush()
            except Exception:  # pragma: no cover
                pass


class RunLogSession:
    """Context manager that tees stdout/stderr into ``<output_dir>/run.log``."""

    def __init__(self, output_dir: str | Path, verbosity: str = "normal") -> None:
        self.path = Path(output_dir) / "run.log"
        self.verbosity = str(verbosity)
        self._handle: TextIO | None = None
        self._saved: tuple[TextIO, TextIO] | None = None

    def __enter__(self) -> "RunLogSession":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", errors="replace")
        self._handle.write(f"===== run.log session started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        quiet = self.verbosity == "quiet"
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = ConsoleTee(self._saved[0], self._handle, quiet=quiet)
        sys.stderr = ConsoleTee(self._saved[1], self._handle, quiet=quiet)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            if self._saved is not None:
                sys.stdout, sys.stderr = self._saved
            if self._handle is not None:
                self._handle.close()
