"""Structured logs, per-job counts and OpenTelemetry spans (#161).

The done-when: the logs parse as JSON, and the counts the job logs are the
counts its report prints. Every model call is answered from recorded
responses; the spans go to the OpenTelemetry SDK's in-memory exporter.
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
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typer.testing import CliRunner

from openodke import Document, Ontology, observe
from openodke.cli.main import app
from openodke.corroborate import PairJudge
from openodke.llm import ModelRoles, ModelSpec
from openodke.llm.base import Completion, Message, ProviderError
from openodke.llm.testing import RecordedClient
from openodke.loaders import DirectoryLoader
from openodke.observe import KEYS, JobCounts, Observer, configure_logs
from openodke.sinks.jsonl import JsonlSink
from openodke.validator import ValidationReport, Validator
from test_pair_judge import DOC, GROUNDED, ROWS, _client
from test_pair_judge import PEOPLE as JUDGED

runner = CliRunner()
REPO = Path(__file__).parent.parent
TRIPLES = REPO / "examples" / "triples"
COUNTED = ("facts_in", "facts_out", "refused", "merged", "linked", "review")


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


def _texts() -> list[Document]:
    return list(DirectoryLoader().load(TRIPLES / "texts"))


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


def test_the_validators_job_counts_are_its_reports_and_the_queue_is_review(
    logs: io.StringIO, tmp_path: Path
) -> None:
    judge = PairJudge(client=_client("same", "different"), queue=tmp_path / "pairs.jsonl")
    validator = Validator(JUDGED, client=RecordedClient(GROUNDED), judge=judge)
    _, report = validator.validate(ROWS, [DOC])
    end = _one(_events(logs.getvalue()), "job.end")

    assert end["command"] == "validate"
    assert end["counts"] == report.job.model_dump()
    assert end["counts"] == {
        "facts_in": report.facts_in,
        "facts_out": report.facts_out,
        "refused": report.refused,
        "merged": report.merged,
        "linked": report.linked,
        "review": report.judge["queued"] if report.judge else -1,
    }
    assert end["counts"]["review"] == 1
    assert end["cost"]["calls"] == report.calls == 4
    assert JobCounts().model_dump() == dict.fromkeys(COUNTED, 0)
    assert ValidationReport().job.review == 0


def test_odke_validate_and_odke_ground_log_the_counts_they_print(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "recorded.json").write_text(
        json.dumps([{"match": "Claim:", "response": {"verdict": "supported"}}])
    )
    (tmp_path / "models.yaml").write_text("models:\n  replay:\n    ground: recorded.json\n")
    facts = ["--facts", str(TRIPLES / "triples.jsonl"), "--texts", str(TRIPLES / "texts")]
    common = [*facts, "--config", "models.yaml", "--log-format", "json"]

    result = runner.invoke(app, ["validate", *common, "-o", "v"])
    assert result.exit_code == 0, result.output
    validation = json.loads((tmp_path / "v" / "manifest.json").read_text())["stats"]["validation"]
    end = _one(_events(result.stderr), "job.end")
    assert end["command"] == "validate"
    assert {k: end["counts"][k] for k in COUNTED[:-1]} == {k: validation[k] for k in COUNTED[:-1]}
    assert f"in            {validation['facts_in']} facts" in result.stdout

    result = runner.invoke(app, ["ground", *common, "-o", "g"])
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "g" / "summary.json").read_text())
    end = _one(_events(result.stderr), "job.end")
    assert end["command"] == "ground"
    assert end["counts"] == {
        "facts_in": summary["rows"],
        "facts_out": summary["facts"],
        "refused": sum(summary["refused"].values()),
        "merged": 0,
        "linked": 0,
        "review": 0,
    }
    assert end["cost"]["calls"] == summary["calls"]
    assert f"free checks   {end['counts']['refused']} refused" in result.stdout
    # The handler is gone once the command is.
    assert observe.ROOT.handlers == [] and observe.ROOT.propagate


def test_a_bad_log_format_is_a_usage_error(example: Path) -> None:
    config = str(example / "odke.yaml")
    assert runner.invoke(app, ["run", config, "--log-format", "xml"]).exit_code == 2
    assert runner.invoke(app, ["run", config, "--log-text"]).exit_code == 2


# --------------------------------------------------------------------------- #
# No text, no secret
# --------------------------------------------------------------------------- #

PASSAGE = "Halden Robotics was founded in Trondheim in 2014 by Ingrid Moe."
REPLY = "I would say the passage mentions Trondheim, so perhaps."


class _Rambling:
    """A model that answers in prose, which the grounder cannot read and logs."""

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: Any = None
    ) -> Completion:
        return Completion(text=REPLY, model="ground-large-20260501", prompt_tokens=40)


def test_no_passage_reply_or_secret_reaches_the_logs_unless_asked(tmp_path: Path) -> None:
    rows = [{"doc": "d", "subject": "Halden Robotics", "predicate": "founded_in",
             "object": "Trondheim", "quote": "founded in Trondheim"}]  # fmt: skip
    doc = Document(id="d", text=PASSAGE)
    secret = ModelSpec(model="acme/ground-large", extra={"api_key": "hunter2"})
    roles = ModelRoles(extract=secret, ground=secret)

    for text in (False, True):
        stream = io.StringIO()
        configure_logs("json", stream=stream, text=text)
        try:
            Validator(roles=roles, client=_Rambling(), sinks=[JsonlSink(tmp_path)]).validate(
                rows, [doc]
            )
        finally:
            configure_logs("text")
        logged = stream.getvalue()
        events = _events(logged)
        warning = next(e for e in events if e["event"] == "log")
        assert warning["template"] == "unreadable grounding answer for fact %s: %r"
        assert "hunter2" not in logged
        call = _one(events, "model.call")
        assert (call["model"], call["served"]) == (
            "acme/ground-large",
            "acme/ground-large-20260501",
        )
        if text:
            assert REPLY in warning["message"]
        else:
            assert "message" not in warning
            for words in (PASSAGE, "Trondheim", "Halden", REPLY):
                assert words not in logged


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


# --------------------------------------------------------------------------- #
# OpenTelemetry
# --------------------------------------------------------------------------- #


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """A tracer provider of the test's own, with the in-memory exporter as its stand-in."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observe, "TRACER_PROVIDER", provider)
    return exporter


def _named(found: Sequence[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [span for span in found if span.name == name]


def test_a_job_is_a_span_each_stage_a_child_and_each_call_a_child_of_its_stage(
    spans: InMemorySpanExporter,
) -> None:
    client = RecordedClient.from_fixture(TRIPLES / "recorded" / "ground.json")
    ontology = Ontology.from_json(TRIPLES / "ontology.json")
    _, report = Validator(ontology, client=client).validate(TRIPLES / "triples.jsonl", _texts())

    found = spans.get_finished_spans()
    (job,) = _named(found, "odke validate")
    stages = [span for span in found if span.name.startswith("stage ")]
    assert {span.name for span in stages} >= {"stage extract", "stage ground", "stage gate"}
    assert all(span.parent is not None for span in stages)
    assert {span.parent.span_id for span in stages if span.parent} == {job.context.span_id}
    (ground,) = _named(found, "stage ground")
    calls = _named(found, "model ground")
    assert len(calls) == report.calls == 4
    assert {span.parent.span_id for span in calls if span.parent} == {ground.context.span_id}
    assert {span.context.trace_id for span in found} == {job.context.trace_id}

    assert job.attributes is not None and ground.attributes is not None
    assert job.attributes["odke.counts.facts_in"] == report.facts_in
    assert job.attributes["odke.counts.facts_out"] == report.facts_out
    assert job.attributes["odke.cost.calls"] == report.calls
    assert ground.attributes["odke.counts.facts"] == 5
    assert calls[0].attributes is not None
    assert calls[0].attributes["odke.model"] == "anthropic/claude-haiku-4-5-20251001"


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
