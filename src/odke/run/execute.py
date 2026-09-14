"""Running a built config: bootstrap, load, run, collect every stage's counts, write.

The pipeline writes the moment it has a graph, and a stage's own counts — the
grounder's verdicts, the extractor's rejections, the corroborator's conflicts —
are not in that graph. So the pipeline here is handed stand-ins that carry each
real sink's `PlatformProfile` and write nothing: the double-stage warning still
fires at construction (DECISIONS #21), and the real sinks write once the stats
are in `KnowledgeGraph.stats`, where a `JsonlSink` manifest and a report can
read them. `pipeline.py` is unchanged.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

from odke.corroborate.provenance import CONFLICT
from odke.run.build import Built, build
from odke.run.config import STAGES, RunConfig
from odke.stages import PlatformProfile, Sink
from odke.types import Document, Fact, KnowledgeGraph

# How many facts a dry run prints before saying how many more there were.
SAMPLE = 20


class _StandIn:
    """What the pipeline sees in place of a sink: its profile, and a write that does nothing."""

    def __init__(self, profile: PlatformProfile | None) -> None:
        self.profile = profile

    def write(self, kg: KnowledgeGraph) -> None:
        return None


@dataclass
class RunResult:
    """What a run produced, wrote or would have written, and what it warned about."""

    graph: KnowledgeGraph
    dry_run: bool
    written: list[str] = field(default_factory=list)
    bootstrap: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, Any]:
        return self.graph.stats

    def render(self) -> str:
        return render(self)


def execute(config: RunConfig, *, dry_run: bool = False) -> RunResult:
    """Run the config. A dry run loads, extracts and grounds, and writes nothing."""
    return run_built(build(config), dry_run=dry_run)


def run_built(built: Built, *, dry_run: bool = False) -> RunResult:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline = built.pipeline(sinks=[_StandIn(p.profile) for p in built.sinks])
    result_warnings = [str(w.message) for w in caught]

    opened: list[Sink] = []
    applied: list[str] = []
    try:
        if built.config.bootstrap:
            constrainer = built.stages["constrainer"]
            if dry_run:
                applied = [
                    s
                    for p in built.sinks
                    if p.can_bootstrap
                    for s in p.ddl(built.ontology, constrainer)
                ]
        if not dry_run:
            opened = [p.open() for p in built.sinks]
            # Before any model is called: a database that refuses the DDL fails
            # the run while it is still free.
            if built.config.bootstrap:
                applied = _bootstrap(built, opened)

        docs = built.documents()
        register_documents(built.stages["extractor"], docs)
        kg = pipeline.run(docs)
        kg = kg.model_copy(update={"stats": collect_stats(built, kg)})

        if dry_run:
            written = [line for p in built.sinks for line in p.describe(kg)]
        else:
            written = []
            for plan, sink in zip(built.sinks, opened, strict=True):
                sink.write(kg)
                written.extend(plan.describe(kg))
    finally:
        for sink in opened:
            close = getattr(sink, "close", None)
            if callable(close):
                close()
    return RunResult(
        graph=kg, dry_run=dry_run, written=written, bootstrap=applied, warnings=result_warnings
    )


def _bootstrap(built: Built, opened: list[Sink]) -> list[str]:
    applied: list[str] = []
    constrainer = built.stages["constrainer"]
    for plan, sink in zip(built.sinks, opened, strict=True):
        if not plan.can_bootstrap:
            continue
        bootstrap = sink.bootstrap  # type: ignore[attr-defined]
        if plan.name == "neo4j":
            applied.extend(bootstrap(built.ontology, constrainer=constrainer))
        else:
            applied.extend(bootstrap(built.ontology) or ())
    return applied


def register_documents(extractor: Any, docs: list[Document]) -> None:
    """Hand the loaded documents to an extractor that looks them up (DECISIONS #19)."""
    lookup = getattr(extractor, "documents", None)
    if isinstance(lookup, dict):
        lookup.update({doc.id: doc for doc in docs})


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def collect_stats(built: Built, kg: KnowledgeGraph) -> dict[str, Any]:
    """The pipeline's counts, the graph's shape, and every stage's own counts.

    A stage reports by carrying a `stats` mapping, as the grounders and the
    verdict validator do; a user's stage that does the same is reported the
    same way. The extractor's rejections and routing report, and the
    corroborator's conflicts, are read from where those stages keep them.
    """
    stats: dict[str, Any] = dict(kg.stats)
    stats["graph"] = {
        "facts": len(kg.facts),
        "edges": len(kg.edges),
        "properties": len(kg.properties),
        "entities": len(kg.entities),
        "links": dict(sorted(Counter(link.kind.value for link in kg.links).items())),
    }
    stages: dict[str, Any] = {}
    for name in STAGES:
        stage = built.stages.get(name)
        if stage is None:
            continue
        own = getattr(stage, "stats", None)
        report = jsonable(own) if isinstance(own, Mapping) else {}
        if name == "extractor":
            report.update(_extractor_stats(stage))
        if name == "corroborator":
            report["conflicts"] = _conflicts(kg.facts)
        if report:
            stages[name] = report
    stats["stages"] = stages
    meter = built.context.meter
    if meter is not None:
        cost = meter.report(documents=int(kg.stats.get("documents", 0))).as_stage_report()
        stats["cost"] = jsonable({"metrics": cost.metrics, "stages": cost.breakdown})
    return stats


def _extractor_stats(extractor: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    totals = getattr(extractor, "totals", None)
    if callable(totals):
        out["paths"] = jsonable(totals())
    rejections: Counter[str] = Counter()
    calls = 0
    for source in (extractor, getattr(extractor, "llm", None)):
        found = getattr(source, "rejections", None)
        if isinstance(found, list):
            rejections.update(str(getattr(r, "reason", r)) for r in found)
        made = getattr(source, "calls", None)
        if isinstance(made, list):
            calls += len(made)
    if rejections:
        out["rejections"] = dict(sorted(rejections.items()))
    if calls and "paths" not in out:
        out["model_calls"] = calls
    return out


def _conflicts(facts: tuple[Fact, ...]) -> dict[str, int]:
    found = Counter(
        str(stamp.get("status"))
        for fact in facts
        if isinstance(stamp := fact.qualifiers.get(CONFLICT), Mapping)
    )
    return dict(sorted(found.items()))


def jsonable(value: Any) -> Any:
    """Plain JSON types, so stats survive a manifest and a `--json` reader."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((jsonable(v) for v in value), key=repr)
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def render(result: RunResult) -> str:
    stats = result.stats
    graph = stats.get("graph", {})
    lines = ["odke run — dry run, nothing written" if result.dry_run else "odke run"]
    lines.append(
        _row(
            "documents",
            f"{stats.get('documents', 0)} ({stats.get('chunks', 0)} chunks; "
            f"{stats.get('skipped', 0)} skipped, {stats.get('deferred', 0)} deferred)",
        )
    )
    for name, report in stats.get("stages", {}).items():
        lines.append(_row(name, _summary(report)))
    lines.append(_row("refused", str(stats.get("refused", 0))))
    links = graph.get("links", {})
    lines.append(
        _row(
            "graph",
            f"{graph.get('facts', 0)} facts ({graph.get('edges', 0)} edges, "
            f"{graph.get('properties', 0)} properties), {graph.get('entities', 0)} entities, "
            f"{sum(links.values())} links" + (f" ({_summary(links)})" if links else ""),
        )
    )
    cost = stats.get("cost", {}).get("metrics")
    if cost:
        usd = cost.get("cost_usd")
        lines.append(
            _row(
                "cost",
                f"{cost.get('calls', 0)} model calls, "
                f"{cost.get('prompt_tokens', 0) + cost.get('completion_tokens', 0)} tokens, "
                + (f"${usd:.4f}" if isinstance(usd, int | float) else "USD unknown"),
            )
        )
    if result.bootstrap:
        verb = "would apply" if result.dry_run else "applied"
        lines.append(_row("bootstrap", f"{verb} {len(result.bootstrap)} statements"))
        lines.extend(f"  {statement}" for statement in result.bootstrap)
    verb = "would write" if result.dry_run else "wrote"
    if result.written:
        lines.append(_row(verb, result.written[0]))
        lines.extend(f"  {line}" for line in result.written[1:])
    else:
        lines.append(_row(verb, "nothing — no sink configured"))
    if result.dry_run and result.graph.facts:
        lines.append("")
        lines.append("facts that would be written:")
        for fact in result.graph.facts[:SAMPLE]:
            lines.append(f"  {_fact_line(fact)}")
        if len(result.graph.facts) > SAMPLE:
            lines.append(f"  … and {len(result.graph.facts) - SAMPLE} more")
    return "\n".join(lines)


def _row(label: str, text: str) -> str:
    return f"{label:<13} {text}"


def _summary(report: Mapping[str, Any]) -> str:
    parts = []
    for key, value in report.items():
        if isinstance(value, Mapping):
            inner = _summary(value)
            if inner:
                parts.append(f"{key} ({inner})")
        elif isinstance(value, bool) or value in (None, 0, 0.0, [], ""):
            continue
        elif isinstance(value, float):
            parts.append(f"{key} {value:.4f}")
        elif isinstance(value, list):
            parts.append(f"{key} {'+'.join(str(v) for v in value)}")
        else:
            parts.append(f"{key} {value}")
    return ", ".join(parts)


def _fact_line(fact: Fact) -> str:
    subject = fact.subject.label or fact.subject.key
    obj = (
        fact.object_entity.label or fact.object_entity.key
        if fact.object_entity is not None
        else repr(fact.object_value)
    )
    negated = "" if fact.polarity.value == "asserted" else f" ({fact.polarity.value})"
    source = fact.evidence[0].doc_id if fact.evidence else "no evidence"
    return (
        f"{subject} —{fact.predicate}→ {obj}{negated}  "
        f"[{fact.verdict.value}, {fact.confidence:.2f}, {source}]"
    )


__all__ = [
    "RunResult",
    "collect_stats",
    "execute",
    "jsonable",
    "register_documents",
    "render",
    "run_built",
]
