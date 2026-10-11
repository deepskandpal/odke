"""The write report: what each sink wrote, merged into what it held, and skipped (#159).

A run ends with one entry per sink. A sink that keeps `writes`, a
`WriteReport`, says per kind (entities, facts, links):

- `written`: new to the store, a node, relationship or line it did not hold;
- `merged`: written into one it held, which a rerun of the same graph does to
  every row, so a rerun's report says it made nothing new;
- `skipped`: rows that wrote nothing, a link whose end was not there to match;

and, for a database, the transactions committed. `Neo4jSink` counts from the
database's own counters, `JsonlSink` from the keys its files hold. A sink that
keeps no report is reported with what it was `handed`. The counts run on
across a sink's writes, so a report for one job is the difference `since`
what the sink had counted before it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openodke.types import KnowledgeGraph

KINDS = ("entities", "facts", "links")
OUTCOMES = ("written", "merged", "skipped")


class WriteReport:
    """A sink's counts, by kind and outcome, run on across its writes."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = {k: dict.fromkeys(OUTCOMES, 0) for k in KINDS}
        self.transactions = 0

    def add(self, kind: str, *, written: int = 0, merged: int = 0, skipped: int = 0) -> None:
        held = self.counts[kind]
        held["written"] += written
        held["merged"] += merged
        held["skipped"] += skipped

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {kind: dict(self.counts[kind]) for kind in KINDS}
        out["transactions"] = self.transactions
        return out


def report_of(sink: Any) -> dict[str, Any] | None:
    """A sink's write report so far, or None for a sink that keeps none."""
    writes = getattr(sink, "writes", None)
    return writes.as_dict() if isinstance(writes, WriteReport) else None


def since(now: Mapping[str, Any], before: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a report counted after `before`, key by key."""
    if before is None:
        return {k: dict(v) if isinstance(v, Mapping) else v for k, v in now.items()}
    out: dict[str, Any] = {}
    for key, value in now.items():
        earlier = before.get(key)
        if isinstance(value, Mapping):
            out[key] = since(value, earlier if isinstance(earlier, Mapping) else None)
        else:
            out[key] = int(value) - int(earlier or 0)
    return out


def handed(kg: KnowledgeGraph, shape: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What a sink that keeps no report was given to write.

    `shape` is the run's counts (`openodke.stream.shape`), which a streamed
    run's graph, holding no facts, cannot give.
    """
    if shape is not None:
        links = sum(int(v) for v in (shape.get("links") or {}).values())
        counts = (int(shape.get("entities", 0)), int(shape.get("facts", 0)), links)
    else:
        counts = (len(kg.entities), len(kg.facts), len(kg.links))
    return {kind: {"handed": count} for kind, count in zip(KINDS, counts, strict=True)}


def sink_name(at: int, sink: Any) -> str:
    """A sink's key in a write report: its position and class, as a reconcile report keys it."""
    return f"{at}:{type(sink).__name__}"


class Writes:
    """The write report of one job over its sinks: snapshot before, `report(kg)` after.

    A streamed run (#158) takes the snapshot before its first micro-batch and
    the report after its last, so the report is the run's.
    """

    def __init__(self, sinks: Sequence[Any]) -> None:
        self.sinks = list(sinks)
        self.before = [report_of(sink) for sink in self.sinks]

    def report(
        self, kg: KnowledgeGraph, *, shape: Mapping[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Each sink's writes since the snapshot; `shape` is the run's counts, for the rest."""
        out: dict[str, dict[str, Any]] = {}
        for at, (sink, before) in enumerate(zip(self.sinks, self.before, strict=True)):
            now = report_of(sink)
            out[sink_name(at, sink)] = handed(kg, shape) if now is None else since(now, before)
        return out


def summary(report: Mapping[str, Any]) -> str:
    """One sink's report on one line: `facts 5 written, 2 merged; …; 1 transaction`."""
    parts = []
    for kind in KINDS:
        counts = report.get(kind) or {}
        said = [f"{counts[k]} {k}" for k in (*OUTCOMES, "handed") if counts.get(k)]
        parts.append(f"{kind} " + (", ".join(said) if said else "none"))
    count = int(report.get("transactions") or 0)
    if count:
        parts.append(f"{count} transaction{'' if count == 1 else 's'}")
    return "; ".join(parts)


__all__ = [
    "KINDS",
    "OUTCOMES",
    "WriteReport",
    "Writes",
    "handed",
    "report_of",
    "since",
    "sink_name",
    "summary",
]
