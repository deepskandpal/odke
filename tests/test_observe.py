"""Structured logs, per-job counts and OpenTelemetry spans (#161).

The events and spans of one job, on an `Observer` of the test's own.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider

from openodke.llm import ModelSpec
from openodke.llm.base import Completion, Message, ProviderError
from openodke.observe import KEYS, JobCounts, Observer, configure_logs


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
