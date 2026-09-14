"""Sink idempotency: write a graph twice, count once.

No labels needed. A sink whose second write of the same graph changes what
the store holds will double every edge on the first re-run after a crash, and
"our `MERGE` is idempotent" is a claim until the store's own counts have been
compared across two writes (#22). So the check is generic: write, count,
write again, count again, and every count must be where it was.

What "count" means is the store's business — nodes and relationships for a
graph database, triples for RDF, lines for a file — so it is a callable the
caller passes. `jsonl_counts` is that callable for `JsonlSink`; a Neo4j sink's
test passes one that runs `MATCH (n) RETURN count(n)` and friends, and calls
the same `assert_idempotent`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from openodke.eval.report import Metric, StageReport
from openodke.stages import Sink
from openodke.types import KnowledgeGraph

Counter = Callable[[], Mapping[str, int]]

_MANIFEST_COUNTS = ("entities", "facts", "edges", "properties", "links")
_STREAMS = ("entities.jsonl", "facts.jsonl", "links.jsonl")


def jsonl_counts(directory: str | Path) -> Counter:
    """What a `JsonlSink` at `directory` holds: its manifest's counts and its line counts.

    Both, because a sink that rewrote its manifest correctly while appending
    to its streams would pass on the manifest alone.
    """
    root = Path(directory)

    def count() -> dict[str, int]:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        counts = {key: int(manifest[key]) for key in _MANIFEST_COUNTS}
        for stream in _STREAMS:
            with (root / stream).open(encoding="utf-8") as fh:
                counts[stream] = sum(1 for line in fh if line.strip())
        return counts

    return count


def check_idempotency(
    sink: Sink, kg: KnowledgeGraph, counts: Counter, *, writes: int = 2
) -> StageReport:
    """Write `kg` through `sink` `writes` times and compare the counts after the first and last."""
    if writes < 2:
        raise ValueError("idempotency needs at least two writes to compare")
    sink.write(kg)
    first = dict(counts())
    for _ in range(writes - 1):
        sink.write(kg)
    last = dict(counts())

    keys = sorted(first.keys() | last.keys())
    changed = [k for k in keys if first.get(k) != last.get(k)]
    breakdown: dict[str, dict[str, Metric]] = {
        k: {"after_first": first.get(k), "after_last": last.get(k)} for k in keys
    }
    if changed:
        notes = tuple(
            f"not idempotent: {k} went from {first.get(k)} to {last.get(k)} over {writes} writes"
            for k in changed
        )
    else:
        notes = (f"idempotent: every count unchanged over {writes} writes",)
    return StageReport(
        stage="sink",
        n=writes,
        metrics={"writes": writes, "counts": len(keys), "changed": len(changed)},
        breakdown=breakdown,
        notes=notes,
    )


def assert_idempotent(
    sink: Sink, kg: KnowledgeGraph, counts: Counter, *, writes: int = 2
) -> StageReport:
    """`check_idempotency` for a test: `AssertionError`, carrying the report, if a count moved."""
    report = check_idempotency(sink, kg, counts, writes=writes)
    if report.metrics["changed"]:
        raise AssertionError(report.render())
    return report


__all__ = ["Counter", "assert_idempotent", "check_idempotency", "jsonl_counts"]
