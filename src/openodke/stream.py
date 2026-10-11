"""Micro-batches: a run of any size, one slice at a time (#158).

The pipeline runs each phase over the whole batch: every chunk extracted, then
every document grounded, then resolution, corroboration, the score and the
gate over all of it. That is what lets the model calls run concurrently, and it
is what resolution and corroboration need, because they compare the batch with
itself. It is also why a batch is held in memory whole.

So a large run is cut into micro-batches (DECISIONS #45). `batch_size`
documents, or triples rows, are read, run through every stage and written; then
the next. The inputs are read as they are needed and the graph is never held
whole, so memory follows the micro-batch, not the run. One ledger counts every
call of every micro-batch, the response cache answers across them, and one
document's failure is still its own.

The price is what a micro-batch cannot see. A fact two micro-batches both state
is merged by the store, when the second is written: through a sink that says
what it holds, in the corroborator, before the score (DECISIONS #35). Without
such a sink the two stay two, and a JSONL file that only appends holds a line
for each. Entities are resolved within a micro-batch, and against the store
with a store lookup (DECISIONS #31). Near-duplicate texts are compared within a
micro-batch, and the coverage report knows the names its micro-batch knows.

`Totals` sums each micro-batch's counts into the run's, so the report and the
JSONL manifest describe the run, not its last slice.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeVar

from openodke.stages import Sink
from openodke.types import KnowledgeGraph

T = TypeVar("T")

# The pipeline's own counts, each summed over the micro-batches. `candidates`
# is the facts the batch stages took in: a job's "in" (`openodke.observe`).
COUNTED = (
    "documents",
    "chunks",
    "skipped",
    "deferred",
    "empty_extractions",
    "derived",
    "refused",
    "candidates",
)
# How many documents' coverage records a streamed run keeps: those with a gap,
# the first this many. The totals count every document.
KEPT = 100
# What the resolver counts for one call, which a streamed run sums: the store
# lookup's counts and batch normalisation's.
PER_CALL = ("store", "batch")


def micro_batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """`items`, `size` at a time, each read only when its micro-batch is."""
    if size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def by_text(items: Iterable[T], size: int, text: Callable[[T], Any]) -> Iterator[list[T]]:
    """`size` items at a time, each micro-batch closed where the text the items cite changes.

    A text's rows that sit together in the input stay in one micro-batch, so
    the text is grounded, measured and written once. A text with more than
    twice `size` rows in a row is split there, so no micro-batch outgrows it.
    """
    if size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list[T] = []
    last: Any = None
    for item in items:
        here = text(item)
        if batch and (len(batch) >= 2 * size or (len(batch) >= size and here != last)):
            yield batch
            batch = []
        batch.append(item)
        last = here
    if batch:
        yield batch


def shape(kg: KnowledgeGraph) -> dict[str, Any]:
    """A graph's counts: facts, edges, properties, entities, and links by kind."""
    edges = sum(fact.is_edge for fact in kg.facts)
    return {
        "facts": len(kg.facts),
        "edges": edges,
        "properties": len(kg.facts) - edges,
        "entities": len(kg.entities),
        "links": dict(sorted(Counter(link.kind.value for link in kg.links).items())),
    }


def streams(sink: Any) -> bool:
    """Whether a sink can take a run a micro-batch at a time.

    One that writes a file of the whole graph on every `write` (RDF, a Cypher
    script, neo4j-admin CSV) says `streams = False`: written once per
    micro-batch, it would end holding the last one.
    """
    return bool(getattr(sink, "streams", True))


def write(sinks: Sequence[Sink], kg: KnowledgeGraph, *, first: bool) -> None:
    """One micro-batch to every sink.

    The first is written, so a sink replaces what an earlier run left, as an
    unbatched run does. Each after it is appended where a sink can append
    (`JsonlSink.append`), and written where it cannot: a Neo4j write merges.
    """
    for sink in sinks:
        append = getattr(sink, "append", None)
        if first or not callable(append):
            sink.write(kg)
        else:
            append(kg)


class Totals:
    """A run's counts, as its micro-batches add up.

    `add(kg)` takes one micro-batch's graph: the pipeline's counts in its
    `stats`, and its shape. `stats()` is what an unbatched run's
    `KnowledgeGraph.stats` would hold of them; `graph()` the run's shape.
    Counts are summed, so an entity two micro-batches both mention counts in
    each. `failed` gathers every micro-batch's; `stopped` is the first stop,
    with the chunks left unextracted and the facts left unchecked summed from
    there on. The coverage totals are summed, a relation is unused when no
    micro-batch used it, and the records of the first `KEPT` documents with a
    gap are kept.
    """

    def __init__(self) -> None:
        self.batches = 0
        self._counts: Counter[str] = Counter()
        self._graph: Counter[str] = Counter()
        self._links: Counter[str] = Counter()
        self._failed: dict[str, str] = {}
        self._stopped: dict[str, Any] | None = None
        self._coverage: dict[str, Any] | None = None
        self._reextract: dict[str, Any] | None = None
        self._resolver: dict[str, dict[str, Any]] = {}

    def add(self, kg: KnowledgeGraph, *, resolver: Any = None) -> None:
        """One micro-batch's graph, and the resolver that resolved it.

        The resolver's `stats["store"]` and `stats["batch"]` count one call, so
        they are summed here.
        """
        stats = kg.stats
        self.batches += 1
        for key in COUNTED:
            self._counts[key] += int(stats.get(key) or 0)
        found = shape(kg)
        for key in ("facts", "edges", "properties", "entities"):
            self._graph[key] += found[key]
        self._links.update(found["links"])
        if isinstance(failed := stats.get("failed"), Mapping):
            self._failed.update({str(k): str(v) for k, v in failed.items()})
        if isinstance(stopped := stats.get("stopped"), Mapping):
            if self._stopped is None:
                self._stopped = dict(stopped)
            else:
                for key in ("unextracted", "unchecked"):
                    self._stopped[key] = int(self._stopped.get(key) or 0) + int(
                        stopped.get(key) or 0
                    )
        if isinstance(coverage := stats.get("coverage"), Mapping):
            self._coverage = _coverage(self._coverage, coverage)
        if isinstance(reextract := stats.get("reextract"), Mapping):
            self._reextract = _summed(self._reextract or {}, reextract)
        own = getattr(resolver, "stats", None)
        for key in PER_CALL:
            if isinstance(own, Mapping) and isinstance(counts := own.get(key), Mapping):
                self._resolver[key] = _summed(self._resolver.get(key, {}), counts)

    def stats(self) -> dict[str, Any]:
        """The pipeline's counts for the run so far, keyed as `Pipeline.run` keys them."""
        out: dict[str, Any] = {key: self._counts[key] for key in COUNTED}
        if self._failed:
            out["failed"] = dict(self._failed)
        if self._stopped is not None:
            out["stopped"] = dict(self._stopped)
        if self._coverage is not None:
            out["coverage"] = {**self._coverage, "documents": list(self._coverage["documents"])}
        if self._reextract is not None:
            out["reextract"] = dict(self._reextract)
        return out

    def graph(self) -> dict[str, Any]:
        """The run's shape so far, as `shape` gives one graph's."""
        return {
            **{key: self._graph[key] for key in ("facts", "edges", "properties", "entities")},
            "links": dict(sorted(self._links.items())),
        }

    def resolver(self, key: str) -> dict[str, Any] | None:
        """The resolver's per-call counts under `key` (`PER_CALL`), summed; None if none."""
        found = self._resolver.get(key)
        return None if found is None else dict(found)

    @property
    def store(self) -> dict[str, Any] | None:
        """The resolver's store counts, summed; None when no micro-batch had a lookup."""
        return self.resolver("store")


def _coverage(held: dict[str, Any] | None, new: Mapping[str, Any]) -> dict[str, Any]:
    unused = list(new.get("unused") or ())
    if held is None:
        held = {
            "sentences": 0,
            "uncovered": 0,
            "missed_entities": 0,
            "not_offered": new.get("not_offered"),
            "unused": unused,
            "documents": [],
        }
    else:
        held["unused"] = [p for p in held["unused"] if p in unused]
    for key in ("sentences", "uncovered", "missed_entities"):
        held[key] += int(new.get(key) or 0)
    for record in new.get("documents") or ():
        if len(held["documents"]) >= KEPT:
            break
        if isinstance(record, Mapping) and (record.get("uncovered") or record.get("missed")):
            held["documents"].append(dict(record))
    return held


def _summed(held: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Two mappings of counts added key by key, nested mappings too; anything else is the new."""
    out = dict(held)
    for key, value in new.items():
        before = out.get(key)
        if isinstance(value, Mapping):
            out[key] = _summed(before if isinstance(before, Mapping) else {}, value)
        elif isinstance(value, bool) or not isinstance(value, int | float):
            out[key] = value
        else:
            out[key] = (before if isinstance(before, int | float) else 0) + value
    return out


__all__ = [
    "COUNTED",
    "KEPT",
    "PER_CALL",
    "Totals",
    "by_text",
    "micro_batches",
    "shape",
    "streams",
    "write",
]
