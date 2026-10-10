"""Structured logs, per-job counts and OpenTelemetry spans (#161).

The done-when: the logs parse as JSON, and the counts the job logs are the
counts its report prints. Every model call is answered from recorded
responses.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from typer.testing import CliRunner

from openodke.cli.main import app
from openodke.llm import ModelSpec
from openodke.llm.base import Completion, Message, ProviderError
from openodke.observe import KEYS, JobCounts, Observer, configure_logs

runner = CliRunner()


@pytest.fixture
def logs() -> Iterator[io.StringIO]:
    """The package's logs as JSON, into a buffer, and put back as they were afterwards."""
    stream = io.StringIO()
    configure_logs("json", stream=stream)
    yield stream
    configure_logs("text")


def _events(text: str) -> list[dict[str, Any]]:
    """Every line a JSON object, with the keys every event has: the first half of done."""
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    for found in events:
        assert set(KEYS) <= set(found), found
        assert found["ts"].endswith("Z") and found["level"] in ("info", "warning")
    return events


def _one(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    found = [e for e in events if e["event"] == name]
    assert len(found) == 1, (name, [e["event"] for e in events])
    return found[0]


# --------------------------------------------------------------------------- #
# The counts the job logs are the counts its report prints
# --------------------------------------------------------------------------- #


def test_odke_run_logs_json_and_its_job_counts_are_its_reports(example: Path) -> None:
    result = runner.invoke(app, ["run", str(example / "odke.yaml"), "--log-format", "json"])
    assert result.exit_code == 0, result.output
    events = _events(result.stderr)
    assert not any(line.startswith("{") for line in result.stdout.splitlines())

    start, end = _one(events, "job.start"), _one(events, "job.end")
    assert start["command"] == end["command"] == "run"
    assert {e["run"] for e in events} == {start["run"]}
    stats = json.loads((example / "out" / "manifest.json").read_text(encoding="utf-8"))["stats"]
    counts, graph = end["counts"], stats["graph"]
    assert counts == {
        "facts_in": stats["candidates"],
        "facts_out": graph["facts"],
        "refused": stats["refused"],
        "merged": stats["candidates"] + stats["derived"] - stats["refused"] - graph["facts"],
        "linked": sum(graph["links"].values()),
        "review": 0,
    }
    assert counts == {
        "facts_in": 36, "facts_out": 25, "refused": 1, "merged": 10, "linked": 4, "review": 0
    }  # fmt: skip
    # The numbers the report prints are the same ones.
    assert f"refused       {counts['refused']}" in result.stdout
    assert f"graph         {counts['facts_out']} facts" in result.stdout
    assert f"{counts['linked']} links" in result.stdout
    assert end["documents"] == stats["documents"] and end["failed"] == 0

    stages = [e["stage"] for e in events if e["event"] == "stage"]
    assert stages == [
        "chunk", "extract", "ground", "normalize", "resolve", "corroborate", "gate", "write"
    ]  # fmt: skip
    by_stage = {e["stage"]: e for e in events if e["event"] == "stage"}
    assert by_stage["extract"]["counts"]["facts"] == stats["candidates"]
    assert by_stage["gate"]["counts"] == {"accepted": 25, "refused": 1}
    assert all(e["latency_s"] >= 0 for e in by_stage.values())
    # Each stage's calls are its own, and they add up to the job's.
    calls = [e for e in events if e["event"] == "model.call"]
    assert by_stage["extract"]["cost"]["calls"] == sum(c["stage"] == "extract" for c in calls)
    assert by_stage["ground"]["cost"]["calls"] == sum(c["stage"] == "ground" for c in calls)
    assert end["cost"]["calls"] == len(calls) == stats["spent"]["calls"]
    ground = next(c for c in calls if c["stage"] == "ground")
    assert ground["model"] == "anthropic/claude-haiku-4-5-20251001"
    assert set(ground["cost"]) == {"input_tokens", "output_tokens", "usd", "cached"}
    documents = [e for e in events if e["event"] == "document"]
    assert len(documents) == stats["documents"]
    assert sum(e["counts"]["facts"] for e in documents) == by_stage["ground"]["counts"]["facts"]


def test_a_bad_log_format_is_a_usage_error(example: Path) -> None:
    config = str(example / "odke.yaml")
    assert runner.invoke(app, ["run", config, "--log-format", "xml"]).exit_code == 2
    assert runner.invoke(app, ["run", config, "--log-text"]).exit_code == 2


class _Failing:
    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: Any = None
    ) -> Completion:
        raise ProviderError("upstream said: Halden Robotics was founded in Trondheim")


def test_a_failed_call_logs_its_error_by_type_alone(logs: io.StringIO) -> None:
    observer = Observer("validate")
    client = observer.client(_Failing(), "ground")
    with pytest.raises(ProviderError):
        client.complete([Message(content="Claim: …")], spec=ModelSpec(model="openai/gpt-x"))
    call = _one(_events(logs.getvalue()), "model.call")
    assert (call["level"], call["error"], call["cost"]) == ("warning", "ProviderError", None)
    assert "Trondheim" not in logs.getvalue()


def test_without_a_handler_an_event_goes_nowhere(capsys: pytest.CaptureFixture[str]) -> None:
    observer = Observer()
    observer.start()
    with observer.stage("extract") as counts:
        counts["facts"] = 1
    observer.finish(JobCounts())
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert not logging.getLogger("openodke").handlers
    with pytest.raises(ValueError, match="json or text"):
        configure_logs("xml")  # type: ignore[arg-type]


def test_without_a_provider_or_the_extra_there_are_no_spans_and_the_events_still_flow(
    logs: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opentelemetry import trace

    # No provider configured: the API's own, which records nothing.
    assert not isinstance(trace.get_tracer_provider(), TracerProvider)
    observer = Observer()
    observer.start()
    assert not observer._job.is_recording()

    # No extra installed: no tracer at all, and the same events.
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    observer = Observer()
    assert observer._tracer is None
    observer.start()
    with observer.stage("ground"):
        pass
    observer.finish(JobCounts(facts_in=1))
    names = [e["event"] for e in _events(logs.getvalue())]
    assert names == ["job.start", "job.start", "stage", "job.end"]
