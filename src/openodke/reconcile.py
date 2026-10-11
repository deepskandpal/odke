"""The reconciler: retract a source that changed or went, and retire what it leaves bare (#116).

Sources change. A page is edited, a document withdrawn, a record corrected.
A layer that only ever adds keeps facts whose text no longer exists, so the
support lists (DECISIONS #33) make the change mechanical:

- **delete:** every trace of the document goes from every fact it backed:
  its evidence, and its place in the support list. An entry left with no
  document goes, and `support` is the length of what is left;
- **update:** the same, then the new version is validated
  (`Validator.validate(..., update=True)`), so the facts it still states merge
  with the store again and regain their support (DECISIONS #35);
- a fact left with no source is **retired**: `Fact.retired_at` is the
  transaction time it lost its last one, and it is kept, because it was not
  proven false. The valid clock is left alone. `hard_delete=True` removes it
  instead;
- a derived fact cites its parent's evidence, so the same retraction reaches
  both; a derived fact whose parent is retired is retired with it (DECISIONS
  #28).

Reconciling the same change twice changes nothing the second time: a fact
that no longer cites the document is not touched, and a retired fact keeps
the time it was first retired. The stores do the work, each through its own
`retract()`: `JsonlSink` on its files and `Neo4jSink` in one write
transaction through its evidence indexes. `Reconciler` runs one change over
all of them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from openodke.corroborate.inverses import derived_from
from openodke.corroborate.merge import _aware, support_of
from openodke.corroborate.provenance import NEAR_DUPLICATES, near_duplicates
from openodke.stages import Retractable, Sink
from openodke.types import Fact, Frozen, Support

# What a store's `retract()` counts: facts that cited the documents, those that
# kept a source, those retired, and those removed by a hard delete.
COUNTS = ("cited", "lost", "retired", "deleted")


def cites(fact: Fact, doc_ids: Collection[str]) -> bool:
    """True when any of the fact's evidence or support names one of these documents."""
    return any(e.doc_id in doc_ids for e in fact.evidence) or any(
        doc in doc_ids for entry in fact.supported_by for doc in entry.doc_ids
    )


def retract(fact: Fact, doc_ids: Collection[str], at: datetime) -> Fact:
    """`fact` with these documents taken out of its evidence and support, retired if none is left.

    A support entry keeps its other documents, with its tier and clock taken
    again from their evidence; an entry with none left goes. A near-duplicate
    group loses the documents too, and goes when one is left. A fact with no
    support list, written before there were any, is counted afresh from the
    evidence left, by `source_of`. A fact that cites none of the documents, or
    is retired already, comes back as it was, the same object.
    """
    if fact.retired_at is not None or not cites(fact, doc_ids):
        return fact
    evidence = tuple(e for e in fact.evidence if e.doc_id not in doc_ids)
    groups = tuple(
        kept
        for group in near_duplicates(fact.qualifiers.get(NEAR_DUPLICATES))
        if len(kept := tuple(d for d in group if d not in doc_ids)) > 1
    )
    qualifiers = {k: v for k, v in fact.qualifiers.items() if k != NEAR_DUPLICATES}
    if groups:
        qualifiers[NEAR_DUPLICATES] = groups
    listed: list[Support] = []
    for entry in fact.supported_by:
        docs = tuple(d for d in entry.doc_ids if d not in doc_ids)
        if docs:
            listed.append(_refreshed(entry, docs, evidence))
    if not listed and evidence:
        listed = list(support_of(evidence, groups))
    if not listed:
        return retired(fact.model_copy(update={"qualifiers": qualifiers}), at)
    update: dict[str, Any] = {
        "evidence": evidence,
        "supported_by": tuple(listed),
        "support": len(listed),
        "qualifiers": qualifiers,
    }
    return fact.model_copy(update=update)


def retired(fact: Fact, at: datetime) -> Fact:
    """`fact` retired at `at`: no evidence, no support, kept. Its valid clock is not touched."""
    update = {"evidence": (), "supported_by": (), "support": 0, "retired_at": at}
    return fact.model_copy(update=update)


def with_parents(facts: Sequence[Fact], at: datetime) -> list[Fact]:
    """The facts, with each derived one retired whose parent among them is retired."""
    gone = {f.signature for f in facts if f.retired_at is not None}
    return [
        retired(f, at) if f.retired_at is None and derived_from(f) in gone else f for f in facts
    ]


def reconciled(facts: Iterable[Fact], doc_ids: Collection[str], at: datetime) -> list[Fact]:
    """Every fact with the documents retracted, and derived facts retired with their parents."""
    return with_parents([retract(f, doc_ids, at) for f in facts], at)


def tally(before: Sequence[Fact], after: Sequence[Fact], *, hard: bool) -> dict[str, int]:
    """What a retraction did, fact by fact: the counts every store's `retract()` returns."""
    counts = dict.fromkeys(COUNTS, 0)
    for old, new in zip(before, after, strict=True):
        if new is old:
            continue
        counts["cited"] += 1
        if new.retired_at is None:
            counts["lost"] += 1
        else:
            counts["deleted" if hard else "retired"] += 1
    return counts


def _refreshed(entry: Support, docs: tuple[str, ...], evidence: Sequence[Any]) -> Support:
    """The entry with only `docs`, its tier and clock read again from their evidence."""
    cited = [e for e in evidence if e.doc_id in docs]
    if not cited:
        return entry.model_copy(update={"doc_ids": docs})
    return entry.model_copy(
        update={
            "doc_ids": docs,
            "tier": max((e.tier for e in cited), key=lambda tier: tier.weight),
            "retrieved_at": max((e.retrieved_at for e in cited), key=_aware),
        }
    )


class ReconcileReport(Frozen):
    """What one reconciliation did, summed over the stores it ran on."""

    documents: tuple[str, ...] = ()
    hard_delete: bool = False
    # Facts that cited a retracted document.
    cited: int = 0
    # Of those, kept with fewer sources.
    lost: int = 0
    # Left with none: retired and kept, or deleted with `hard_delete`.
    retired: int = 0
    deleted: int = 0
    # Each store's own counts, by its position and class name.
    stores: dict[str, dict[str, int]] = Field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        """The totals, by the names in `COUNTS`."""
        return {key: getattr(self, key) for key in COUNTS}

    def render(self) -> str:
        none = "nothing: no fact cited them" if not self.cited else ""
        lines = [
            "odke reconcile",
            _row("retracted", ", ".join(self.documents) or "no document"),
            _row("cited", none or _n(self.cited, "fact")),
        ]
        if self.cited:
            lines.append(_row("lost support", f"{_n(self.lost, 'fact')}, still backed"))
            if self.hard_delete:
                lines.append(_row("deleted", f"{_n(self.deleted, 'fact')} left with no source"))
            else:
                lines.append(
                    _row("retired", f"{_n(self.retired, 'fact')} left with no source, kept")
                )
        return "\n".join(lines)


class Reconciler:
    """Retracts sources from every store given that can retract one.

    ```python
    report = Reconciler([JsonlSink("out", merge=True)]).delete(["doc-a"])
    ```

    `delete(doc_ids)` takes every trace of those documents out of the facts
    they backed, in every store, and retires what is left with no source.
    An update is a delete and then the new version validated, which the
    Validator does with `validate(..., update=True)`. `hard_delete=True`
    removes a fact left with no source instead of retiring it. `at`, the
    transaction time a fact is retired at, defaults to now.
    """

    def __init__(self, stores: Sink | Sequence[Sink], *, hard_delete: bool = False) -> None:
        given = list(stores) if isinstance(stores, Sequence) else [stores]
        self.stores = [store for store in given if isinstance(store, Retractable)]
        if not self.stores:
            raise ValueError(
                "nothing to reconcile: give a store that can retract a source, such as "
                "JsonlSink or Neo4jSink"
            )
        self.hard_delete = hard_delete

    def delete(
        self, doc_ids: str | Iterable[str], *, at: datetime | None = None
    ) -> ReconcileReport:
        docs = tuple(sorted({doc_ids} if isinstance(doc_ids, str) else set(doc_ids)))
        when = at if at is not None else datetime.now(UTC)
        totals = dict.fromkeys(COUNTS, 0)
        stores: dict[str, dict[str, int]] = {}
        for i, store in enumerate(self.stores):
            counts = {
                k: int(v) for k, v in store.retract(docs, at=when, hard=self.hard_delete).items()
            }
            stores[f"{i}:{type(store).__name__}"] = counts
            for key in COUNTS:
                totals[key] += counts.get(key, 0)
        return ReconcileReport(
            documents=docs, hard_delete=self.hard_delete, stores=stores, **totals
        )


def _row(label: str, text: str) -> str:
    return f"{label:<13} {text}"


def _n(count: int, noun: str) -> str:
    return f"{count} {noun if count == 1 else noun + 's'}"


__all__ = [
    "COUNTS",
    "ReconcileReport",
    "Reconciler",
    "cites",
    "reconciled",
    "retired",
    "retract",
    "tally",
    "with_parents",
]
