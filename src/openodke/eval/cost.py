"""Cost and latency per stage, collected without touching a Protocol.

Accuracy is the first question anyone evaluating a pipeline asks; cost is the
second. Every model call in this library goes through `LLMClient`, and a
model-backed stage takes its client as a constructor argument — so the hook is
the client, not the stage. `CostMeter.client(inner, stage="ground")` returns a
`MeteredClient` that satisfies `LLMClient`, forwards every call, and records
its tokens, `Completion.cost_usd` and wall-clock latency under that stage
name. Hand it to the grounder in place of the real client, run the pipeline as
usual, and `meter.report(documents=...)` says what the run cost per stage and
per thousand documents. The stage Protocols, `Pipeline` and the stages
themselves never learn they were measured.

Unknown cost stays unknown. `Completion.cost_usd` is `None` when a provider
does not report it, and a stage with any unpriced call has `cost_usd` `None`
rather than the sum of the calls that happened to be priced: a partial sum
would understate the bill while looking like a measurement. That partial sum
is still there, as `priced_usd`, under a name that says what it is.

A pattern extractor makes no calls and costs nothing. So the comparisons
worth making — hybrid against model-only extraction, a cheap grounder against
the extraction model grounding its own output — are two metered runs over the
same documents, set side by side with `compare_costs`. Failed calls are not
recorded; latency is the sum of per-call wall-clock time, which a concurrent
run beats.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openodke.eval.report import Metric, StageReport
from openodke.llm.base import Completion, LLMClient, Message, ModelSpec
from openodke.types import Frozen


class CallRecord(Frozen):
    """One model call, as the meter saw it."""

    stage: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # None when the provider did not say. Never 0.0 standing in for that.
    cost_usd: float | None = None
    latency_s: float = 0.0


class StageCost(Frozen):
    """Every call one stage made, added up."""

    stage: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    # None when any call in the stage was unpriced.
    cost_usd: float | None
    # The sum over priced calls only, whatever the rest cost.
    priced_usd: float
    unpriced_calls: int
    latency_s: float
    models: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def of(cls, stage: str, records: Sequence[CallRecord]) -> StageCost:
        unpriced = sum(1 for r in records if r.cost_usd is None)
        priced = sum(r.cost_usd for r in records if r.cost_usd is not None)
        return cls(
            stage=stage,
            calls=len(records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            cost_usd=None if unpriced else priced,
            priced_usd=priced,
            unpriced_calls=unpriced,
            latency_s=sum(r.latency_s for r in records),
            models=tuple(dict.fromkeys(r.model for r in records)),
        )


class CostReport(Frozen):
    """What a run cost: the raw call records, and the document count to scale them by.

    Holding the records rather than only the totals is what makes the number
    reproducible — the JSON of a report is the measurement, not a summary of it.
    """

    documents: int
    records: tuple[CallRecord, ...] = ()

    @property
    def stages(self) -> dict[str, StageCost]:
        names = dict.fromkeys(r.stage for r in self.records)
        return {n: StageCost.of(n, [r for r in self.records if r.stage == n]) for n in names}

    @property
    def total(self) -> StageCost:
        return StageCost.of("total", self.records)

    def per_1k_documents(self) -> dict[str, dict[str, Metric]]:
        """Calls, tokens, USD and latency per thousand documents, per stage and in total.

        A straight-line extrapolation from this run: it assumes the next
        thousand documents look like these, which is the caller's to judge.
        """
        rows = {**self.stages, "total": self.total}
        return {name: _scaled(cost, self.documents) for name, cost in rows.items()}

    def as_stage_report(self) -> StageReport:
        total, scaled = self.total, self.per_1k_documents()
        breakdown: dict[str, dict[str, Metric]] = {
            name: {
                "calls": cost.calls,
                "prompt_tokens": cost.prompt_tokens,
                "completion_tokens": cost.completion_tokens,
                "cost_usd": cost.cost_usd,
                "latency_s": cost.latency_s,
                "usd_per_1k": scaled[name]["cost_usd"],
                "tokens_per_1k": scaled[name]["tokens"],
                "latency_s_per_1k": scaled[name]["latency_s"],
            }
            for name, cost in self.stages.items()
        }
        notes = []
        if total.unpriced_calls:
            notes.append(
                f"{total.unpriced_calls} of {total.calls} call(s) reported no cost, so USD is "
                f"unknown wherever they occur; the priced calls came to ${total.priced_usd:.4f}"
            )
        metrics: dict[str, Metric] = {
            "documents": self.documents,
            "calls": total.calls,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
            "cost_usd": total.cost_usd,
            "priced_usd": total.priced_usd,
            "unpriced_calls": total.unpriced_calls,
            "latency_s": total.latency_s,
            "usd_per_1k_documents": scaled["total"]["cost_usd"],
            "tokens_per_1k_documents": scaled["total"]["tokens"],
            "latency_s_per_1k_documents": scaled["total"]["latency_s"],
        }
        return StageReport(
            stage="cost", n=self.documents, metrics=metrics, breakdown=breakdown, notes=tuple(notes)
        )

    def render(self) -> str:
        return self.as_stage_report().render()


class CostMeter:
    """Collects `CallRecord`s from every client it hands out.

    One meter per run. `clock` is injectable so a test can make latency exact.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.records: list[CallRecord] = []

    def client(self, inner: LLMClient, stage: str) -> MeteredClient:
        """`inner`, with every call it makes recorded under `stage`."""
        return MeteredClient(inner, self, stage)

    def record(self, record: CallRecord) -> None:
        # list.append is atomic, so a grounder calling from threads is fine.
        self.records.append(record)

    def report(self, documents: int) -> CostReport:
        """The run so far, to be scaled by `documents` — `kg.stats["documents"]` after a run."""
        return CostReport(documents=documents, records=tuple(self.records))


class MeteredClient:
    """An `LLMClient` that forwards to another and tells a meter what each call cost."""

    def __init__(self, inner: LLMClient, meter: CostMeter, stage: str) -> None:
        self.inner = inner
        self.meter = meter
        self.stage = stage

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        start = self.meter.clock()
        completion = self.inner.complete(messages, spec=spec, schema=schema)
        self.meter.record(
            CallRecord(
                stage=self.stage,
                model=completion.model or spec.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost_usd=completion.cost_usd,
                latency_s=self.meter.clock() - start,
            )
        )
        return completion


def compare_costs(reports: Mapping[str, CostReport]) -> StageReport:
    """Several metered runs side by side, per thousand documents.

    Keyed by whatever the caller calls each configuration — "hybrid" and
    "model-only", "haiku grounder" and "same-model grounder". Each gets a
    total row and a row per stage, so the stage that moved is visible.
    """
    breakdown: dict[str, dict[str, Metric]] = {}
    for name, report in reports.items():
        scaled = report.per_1k_documents()
        breakdown[name] = {"documents": report.documents, **scaled["total"]}
        for stage in report.stages:
            breakdown[f"{name} / {stage}"] = {"documents": report.documents, **scaled[stage]}
    priced = {
        name: usd for name in reports if isinstance(usd := breakdown[name]["cost_usd"], float)
    }
    notes = []
    if priced:
        ranked = ", ".join(
            f"{name} ${usd:.2f}" for name, usd in sorted(priced.items(), key=lambda kv: kv[1])
        )
        notes.append(f"USD per 1k documents, cheapest first: {ranked}")
    if unknown := [name for name in reports if name not in priced]:
        notes.append(f"cost unknown for: {', '.join(unknown)}")
    return StageReport(stage="cost", n=len(reports), breakdown=breakdown, notes=tuple(notes))


def _scaled(cost: StageCost, documents: int) -> dict[str, Metric]:
    factor = 1000 / documents if documents else None

    def scale(value: float | None) -> Metric:
        return None if value is None or factor is None else value * factor

    return {
        "calls": scale(cost.calls),
        "tokens": scale(cost.total_tokens),
        "cost_usd": scale(cost.cost_usd),
        "latency_s": scale(cost.latency_s),
    }


__all__ = [
    "CallRecord",
    "CostMeter",
    "CostReport",
    "MeteredClient",
    "StageCost",
    "compare_costs",
]
