"""The re-extract hook: hand a coverage gap back to the extractor, ground what returns (#102).

A validator cannot invent facts (DECISIONS #24), but the coverage report
(`openodke.coverage`) says where the extractor left some behind: a sentence
naming two known entities that no fact cites, or a known entity no fact names.
Each such sentence is a *window*. The window goes back to whichever extractor
produced the batch, with the relations that could hold between the entities it
names and the facts already taken from it, and what comes back is grounded like
any other fact.

The idea is GraphRAG's "gleaning" pass (Edge et al. 2024, arXiv 2404.16130),
which asks the model again whether it missed anything. Here it is scoped to one
window rather than a whole chunk, aimed by a report that needs no model, and
checked by the grounder afterwards, so a fact invented on the second ask is
counted as refused rather than written as found.

The hook is one method, so any extractor can take part (DECISIONS #5):

    reextract(window: Chunk, relations: list[str], already: list[Fact], ontology) -> list[Fact]

`LLMExtractor` implements it with the registered `reextract` prompt. Bounded:
each window is asked once per run, and at most `Reextract.windows` per
document. Off by default: `Pipeline(reextract=Reextract())`, or `reextract:` in
an `odke run` config.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from openodke._batch import is_config_error, told
from openodke.chunking import sentences
from openodke.coverage import CoverageReport, NameMatcher, identity, known_entities
from openodke.llm.budget import BudgetExceeded
from openodke.ontology import Ontology
from openodke.types import Chunk, Document, Entity, Fact, GroundingVerdict, Span

log = logging.getLogger("openodke.reextract")

# On Fact.qualifiers: {"window": [start, end]}, the gap a fact came back from.
REEXTRACT = "odke.reextract"

Ground = Callable[[list[tuple[list[Fact], Document]]], list[list[Fact]]]


@runtime_checkable
class Reextractor(Protocol):
    """Asked again about one window: facts from it that the first pass missed.

    `relations` are the predicates the window could hold; `already` are the
    facts already taken from it, which must not come back. Return only facts
    the window states, with spans into its document: they are grounded next.
    """

    def reextract(
        self, window: Chunk, relations: list[str], already: list[Fact], ontology: Ontology
    ) -> list[Fact]: ...


@dataclass(frozen=True)
class Reextract:
    """How many gaps to hand back, and to whom.

    `windows` caps the windows per document, uncovered sentences first and then
    the sentences holding a missed entity, each in document order. `hook` is
    the extractor to ask; left out, the pipeline's own extractor, which must
    have a `reextract` method.
    """

    windows: int = 3
    hook: Reextractor | None = None

    def __post_init__(self) -> None:
        if self.windows < 1:
            raise ValueError(f"windows must be at least 1, got {self.windows}")


def hook_for(policy: Reextract, extractor: object) -> Reextractor:
    """The hook `policy` names, or `extractor` when it can re-extract; TypeError otherwise."""
    if policy.hook is not None:
        return policy.hook
    if callable(getattr(extractor, "reextract", None)):
        return extractor  # type: ignore[return-value]
    raise TypeError(
        f"{type(extractor).__name__} has no reextract(window, relations, already, ontology); "
        "pass Reextract(hook=...) with one that does"
    )


def relations_for(ontology: Ontology, types: set[str]) -> list[str]:
    """The predicates that could hold in a window naming entities of `types`.

    One whose domain fits one of the types, and whose range is another of them
    (or an ancestor of one) or a literal. Most important first.
    """
    found = []
    for p in ontology.predicates.values():
        if p.domain and not any(set(p.domain) & ontology.lineage(t) for t in types):
            continue
        if p.is_edge_in(ontology) and not any(p.range in ontology.lineage(t) for t in types):
            continue
        found.append(p)
    return [p.name for p in sorted(found, key=lambda p: (-p.importance, p.name))]


def reextract(
    policy: Reextract,
    hook: Reextractor,
    ground: Ground,
    ontology: Ontology,
    routed: Sequence[tuple[Document, Sequence[Chunk]]],
    grounded: list[list[Fact]],
    report: CoverageReport,
) -> tuple[list[list[Fact]], dict[str, Any]]:
    """Each document's grounded facts with what its gap windows gave back, and the counts.

    `routed` and `grounded` are the pipeline's, one row per document; `report`
    is the coverage measured on them. Returned facts that repeat one already
    held (by `Fact.signature`) are dropped and counted, and the rest are
    grounded with `ground` and stamped `odke.reextract`. A window whose call
    fails, or a document whose returned facts fail to ground, is counted under
    `failed` and skipped; the first pass stands for it.
    """
    every = [fact for row in grounded for fact in row]
    known = known_entities(every)
    matcher = NameMatcher(known)
    types: dict[str, set[str]] = {}
    for entity in known:
        if (key := identity(entity)) is not None:
            types.setdefault(key, set()).add(entity.type)
    by_doc = {record.doc_id: record for record in report.documents}

    counts = {"windows": 0, "returned": 0, "duplicates": 0, "kept": 0, "refused": 0}
    verdicts: dict[str, int] = {}
    out: list[list[Fact]] = []
    asked: list[tuple[int, list[Fact], Document]] = []
    for at, ((doc, chunks), facts) in enumerate(zip(routed, grounded, strict=True)):
        out.append(list(facts))
        record = by_doc.get(doc.id)
        if record is None:
            continue
        held = {fact.signature for fact in facts}
        returned: list[Fact] = []
        for window in _windows(doc, chunks, record.uncovered, record.missed, policy.windows):
            keys = {key for named, _, _ in matcher.find(window.text) for key in named}
            relations = relations_for(ontology, set().union(*(types.get(k, set()) for k in keys)))
            if not relations:
                continue
            counts["windows"] += 1
            already = [f for f in facts if _in_window(f, window, keys)]
            try:
                found = list(hook.reextract(window, relations, already, ontology))
            except Exception as exc:
                # One window's failure costs that window: the first pass stands (#162).
                if is_config_error(exc) or isinstance(exc, BudgetExceeded):
                    raise
                counts["failed"] = counts.get("failed", 0) + 1
                log.warning("re-extraction failed for a window of %s, skipped: %s", doc.id, exc)
                continue
            for fact in found:
                counts["returned"] += 1
                if fact.signature in held:
                    counts["duplicates"] += 1
                    continue
                held.add(fact.signature)
                stamp = {**fact.qualifiers, REEXTRACT: {"window": [window.start, window.end]}}
                returned.append(fact.model_copy(update={"qualifiers": stamp}))
        if returned:
            asked.append((at, returned, doc))
    if asked:
        failures: dict[int, Exception] = {}
        try:
            grounded_rows = ground([(f, d) for _, f, d in asked])
        except Exception as exc:
            if is_config_error(exc) or isinstance(exc, BudgetExceeded):
                raise
            if (got := told(exc, len(asked))) is None:
                raise
            grounded_rows, failures = got
        for i, ((at, _, doc), checked) in enumerate(zip(asked, grounded_rows, strict=True)):
            if i in failures:
                # What came back for a document its grounding failed on is not kept.
                counts["failed"] = counts.get("failed", 0) + 1
                log.warning("grounding re-extracted facts failed for %s: %s", doc.id, failures[i])
                continue
            for fact in checked:
                verdicts[fact.verdict.value] = verdicts.get(fact.verdict.value, 0) + 1
                refused = fact.verdict in (
                    GroundingVerdict.CONTRADICTED,
                    GroundingVerdict.NOT_FOUND,
                )
                counts["refused" if refused else "kept"] += 1
            out[at].extend(checked)
    return out, {**counts, "verdicts": dict(sorted(verdicts.items()))}


def summary(stats: dict[str, Any]) -> str:
    """One line from the counts `reextract` returns: what `odke run` prints."""
    line = (
        f"{stats.get('windows', 0)} windows asked, {stats.get('returned', 0)} facts returned "
        f"({stats.get('duplicates', 0)} already held), {stats.get('kept', 0)} kept and "
        f"{stats.get('refused', 0)} refused by grounding"
    )
    # Counted only when one did: a window or a document whose gap pass raised.
    return line + (f", {stats['failed']} failed" if stats.get("failed") else "")


def _windows(
    doc: Document,
    chunks: Sequence[Chunk],
    uncovered: Sequence[Span],
    missed: Sequence[Span],
    limit: int,
) -> list[Chunk]:
    """Up to `limit` distinct sentences: the uncovered ones, then those holding a missed name."""
    bounds = sentences(doc.text)
    picked: dict[tuple[int, int], None] = {}
    for span in uncovered:
        picked.setdefault((span.start, span.end))
    for span in missed:
        around = next(((s, e) for s, e in bounds if s <= span.start < e), None)
        if around is not None:
            picked.setdefault(around)
    windows = []
    for start, end in list(picked)[:limit]:
        index = next((c.index for c in chunks if c.start <= start < c.end), 0)
        windows.append(
            Chunk(doc_id=doc.id, start=start, end=end, text=doc.text[start:end], index=index)
        )
    return windows


def _in_window(fact: Fact, window: Chunk, keys: set[str]) -> bool:
    """A fact cited inside the window, or one naming an entity the window names."""
    for evidence in fact.evidence:
        span = evidence.span
        if span is not None and span.start < window.end and window.start < span.end:
            return True
    return any(identity(entity) in keys for entity in _entities(fact))


def _entities(fact: Fact) -> tuple[Entity, ...]:
    if fact.object_entity is None:
        return (fact.subject,)
    return (fact.subject, fact.object_entity)


__all__ = [
    "REEXTRACT",
    "Reextract",
    "Reextractor",
    "hook_for",
    "reextract",
    "relations_for",
    "summary",
]
