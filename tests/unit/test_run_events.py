"""qsarena.run_events: JSON event stream, verbosity, warnings and the run.log tee."""

from __future__ import annotations

import io
import json
import sys

from qsarena import run_events


def test_json_events_are_valid_json_lines_and_respect_verbosity(tmp_path):
    log = run_events.RunEventLog()
    log.configure(tmp_path, "normal")
    log.emit("run_started", mode="single", extra={"a": 1})
    log.emit("stage_started", level="detail", stage_index=2)  # below normal: dropped
    log.emit("debug_thing", level="debug")
    log.warn("small_dataset", "tiny", remedy="get more data", dataset="d1", echo=False)
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in records] == ["run_started", "warning"]
    assert records[0]["mode"] == "single" and records[0]["extra"] == {"a": 1}
    assert records[1]["remedy"] == "get more data" and records[1]["level"] == "warning"
    verbose = run_events.RunEventLog()
    verbose.configure(tmp_path / "v", "verbose")
    verbose.emit("stage_started", level="detail")
    assert (tmp_path / "v" / "events.jsonl").exists()


def test_warnings_are_deduplicated_per_topic_and_dataset(tmp_path):
    log = run_events.RunEventLog()
    log.configure(tmp_path)
    log.warn("preflight_small_dataset", "from preflight", dataset="d1", echo=False)
    log.warn("small_dataset", "from the run", dataset="d1", echo=False)
    log.warn("small_dataset", "other dataset", dataset="d2", echo=False)
    assert [w.message for w in log.warnings] == ["from preflight", "other dataset"]


def test_quiet_console_keeps_only_warnings_and_summary(tmp_path):
    console = io.StringIO()
    handle = (tmp_path / "run.log").open("w", encoding="utf-8")
    tee = run_events.ConsoleTee(console, handle, quiet=True)
    tee.write("[1/1] d | stage 4/9: conventional model SVR\n[warn] d: something\nWrote benchmark outputs to x\n")
    tee.flush()
    handle.close()
    assert console.getvalue() == "[warn] d: something\nWrote benchmark outputs to x\n"
    assert "stage 4/9" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_run_log_session_tees_stdout(tmp_path):
    with run_events.RunLogSession(tmp_path, "normal"):
        print("hello from the run")
    assert "hello from the run" in (tmp_path / "run.log").read_text(encoding="utf-8")
    assert not isinstance(sys.stdout, run_events.ConsoleTee)


def test_tqdm_bar_only_on_a_terminal():
    assert run_events.tqdm_progress_enabled(io.StringIO()) is False

    class _Tty(io.StringIO):
        def isatty(self):
            return True

    try:
        import tqdm  # noqa: F401
    except ImportError:
        assert run_events.tqdm_progress_enabled(_Tty()) is False
    else:
        assert run_events.tqdm_progress_enabled(_Tty()) is True
