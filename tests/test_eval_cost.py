"""Cost and latency: collected through the client, with no stage Protocol changed.

No model is called. Clients here are scripted or fake, and the clock ticks a
quarter-second per read, so every call takes exactly 0.25 s.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from openodke import Chunk, Document, Entity, Fact, GroundingVerdict, Ontology, Pipeline
from openodke.eval.cost import CallRecord, CostMeter, CostReport, compare_costs
from openodke.llm import Completion, LLMClient, Message, ModelSpec, ScriptedClient

SPEC = ModelSpec(model="test/model")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


class _PricedClient:
    """Reports tokens and a price, as litellm does for hosted providers."""

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        return Completion(
            text="yes", model=spec.model, prompt_tokens=100, completion_tokens=2, cost_usd=0.0001
        )


class _ModelExtractor:
    """A model-backed extractor: its client is a constructor argument, which is the hook."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        out = self.client.complete([Message(content=f"Extract: {chunk.text}")], spec=SPEC)
        assert out.parsed is not None
        return [
            Fact(
                subject=Entity(key=chunk.doc_id, type="Person"),
                predicate="name",
                object_value=out.parsed["name"],
            )
        ]


class _ModelGrounder:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def ground(self, fact: Fact, doc: Document) -> Fact:
        out = self.client.complete(
            [Message(content=f"Does '{doc.text}' support {fact.object_value}?")],
            spec=ModelSpec(model="test/small"),
        )
        verdict = GroundingVerdict.SUPPORTED if out.text == "yes" else GroundingVerdict.NOT_FOUND
        return fact.model_copy(update={"verdict": verdict})


def test_a_metered_client_is_still_an_llm_client() -> None:
    meter = CostMeter()
    assert isinstance(meter.client(ScriptedClient(["x"]), stage="extract"), LLMClient)


def test_a_pipeline_run_is_metered_without_changing_a_protocol() -> None:
    """Wrap the clients, build the stages as usual, run the pipeline, read the meter."""
    meter = CostMeter(clock=_Clock())
    extractor = _ModelExtractor(
        meter.client(ScriptedClient([{"name": "Ada"}, {"name": "Charles"}]), "extract")
    )
    grounder = _ModelGrounder(meter.client(_PricedClient(), "ground"))
    docs = [Document(id="d1", text="Ada."), Document(id="d2", text="Charles.")]
    kg = Pipeline(Ontology(), extractor, grounder=grounder).run(docs)
    assert all(f.verdict is GroundingVerdict.SUPPORTED for f in kg.facts)

    report = meter.report(documents=kg.stats["documents"])
    extract, ground = report.stages["extract"], report.stages["ground"]
    assert (extract.calls, ground.calls) == (2, 2)
    assert (ground.prompt_tokens, ground.completion_tokens) == (200, 4)
    assert ground.cost_usd == pytest.approx(0.0002)
    assert ground.latency_s == pytest.approx(0.5)
    assert ground.models == ("test/small",)
    # ScriptedClient reports no cost, and unknown is not zero.
    assert extract.cost_usd is None
    assert extract.unpriced_calls == 2
    assert extract.prompt_tokens > 0


def test_one_unpriced_call_makes_the_total_unknown_not_understated() -> None:
    report = CostReport(
        documents=1,
        records=(
            CallRecord(stage="ground", model="m", cost_usd=0.5),
            CallRecord(stage="ground", model="m", cost_usd=None),
        ),
    )
    assert report.total.cost_usd is None
    assert report.total.priced_usd == 0.5
    stage_report = report.as_stage_report()
    assert stage_report.metrics["cost_usd"] is None
    assert stage_report.metrics["usd_per_1k_documents"] is None
    assert "1 of 2 call(s) reported no cost" in stage_report.notes[0]
    cost_line = next(line for line in report.render().splitlines() if "  cost_usd " in line)
    assert cost_line.rstrip().endswith("—")


def test_per_thousand_documents_is_a_straight_line_extrapolation() -> None:
    records = tuple(
        CallRecord(
            stage="extract",
            model="big",
            prompt_tokens=900,
            completion_tokens=100,
            cost_usd=0.002,
            latency_s=1.5,
        )
        for _ in range(4)
    )
    report = CostReport(documents=4, records=records)
    scaled = report.per_1k_documents()
    # Four calls over four documents: 1000 calls, 1M tokens, $2.00 and 1500 s per 1k documents.
    assert scaled["extract"]["calls"] == 1000
    assert scaled["extract"]["tokens"] == 1_000_000
    assert scaled["extract"]["cost_usd"] == pytest.approx(2.0)
    assert scaled["total"]["latency_s"] == pytest.approx(1500.0)
    assert report.as_stage_report().breakdown["extract"]["usd_per_1k"] == pytest.approx(2.0)
    assert CostReport.model_validate_json(report.model_dump_json()) == report


def test_zero_documents_scales_to_unknown() -> None:
    report = CostReport(documents=0, records=(CallRecord(stage="x", model="m", cost_usd=1.0),))
    assert report.per_1k_documents()["total"]["cost_usd"] is None


def _run(extract_calls: int, documents: int = 4) -> CostReport:
    meter = CostMeter()
    for _ in range(extract_calls):
        meter.record(CallRecord(stage="extract", model="big", cost_usd=0.01))
    for _ in range(documents):
        meter.record(CallRecord(stage="ground", model="small", cost_usd=0.001))
    return meter.report(documents=documents)


def test_hybrid_against_model_only() -> None:
    """A pattern extractor took three of four documents: one extraction call instead of four."""
    report = compare_costs({"model-only": _run(4), "hybrid": _run(1)})
    b = report.breakdown
    assert b["hybrid"]["cost_usd"] == pytest.approx(3.5)
    assert b["model-only"]["cost_usd"] == pytest.approx(11.0)
    assert b["hybrid / extract"]["calls"] == 250
    assert b["hybrid / ground"]["cost_usd"] == pytest.approx(b["model-only / ground"]["cost_usd"])
    assert report.notes == (
        "USD per 1k documents, cheapest first: hybrid $3.50, model-only $11.00",
    )


def test_a_comparison_names_the_run_whose_cost_is_unknown() -> None:
    unknown = CostReport(documents=1, records=(CallRecord(stage="ground", model="local"),))
    report = compare_costs({"hosted": _run(1), "local": unknown})
    assert "cost unknown for: local" in report.notes
