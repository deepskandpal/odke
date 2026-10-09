"""What the extractor left behind, counted per document with no model (#130).

A validator cannot invent facts (DECISIONS #24), but it can count where the
facts are not, and that count is what the re-extract hook (#102) hands back.
Three counts, deterministic and free:

- **Entities in no fact.** A *known name* is a label or alias of any subject
  or edge object in the batch. One that a document's text mentions and none of
  that document's facts names is a missed entity.
- **Sentences no fact covers.** A sentence that names two or more known
  entities is one a fact could have come from. It is covered when any fact's
  evidence span overlaps it, and the rest are gaps. A sentence naming fewer is
  not counted: most are narrative, and counting them would bury the gaps.
- **Relations never offered.** The ontology's predicates the extractor was not
  shown, when it can say what it showed (`offered(ontology)`), and the
  predicates it was shown that no fact in the batch used.

A span nobody chose (`SpanOrigin.CONTEXT`, DECISIONS #25) overlaps every
sentence and so says nothing about which one holds the fact. A fact cited that
way covers the window the span locator (#112) finds for it, the sentence or two
naming both its subject and its object, and nothing when there is none.

Names are found by the span locator's rules (`openodke.ground.locate`): a
label or alias as written or as its name key, whole words, case and accents
ignored, except that a capitalised name is found only capitalised and never
inside a longer capitalised name (`Africa` is not in `South Africa`).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from openodke.chunking import sentences
from openodke.corroborate.normalize import name_key
from openodke.ground.locate import _entity_names, _extends, _sentences, locate_span
from openodke.ground.span import SpanStatus, check_span
from openodke.ontology import Ontology
from openodke.types import Document, Entity, Fact, Frozen, Span, SpanOrigin

Region = tuple[int, int]
# Which known entities one mention names: usually one, more when two share a name.
Mention = tuple[frozenset[str], int, int]


def identity(entity: Entity) -> str | None:
    """What two mentions of one known entity share: its label's name key, or None."""
    names = _names(entity)
    key = name_key(names[0]) if names else ""
    return key or None


class NameMatcher:
    """Finds known entities in a text, by the span locator's rules.

    Each entity is found by the forms `openodke.ground.locate` finds it by, and
    held back by the same two checks. The locator asks where one fact's two
    names are; this asks which of every known entity a text names, so the forms
    are indexed once and each sentence is read once, longest form first.
    """

    def __init__(self, entities: Iterable[Entity]) -> None:
        self._forms: dict[tuple[str, ...], list[tuple[str, bool]]] = {}
        for entity in entities:
            key = identity(entity)
            if key is None:
                continue
            for name in _entity_names(entity):
                entry = (key, name.capital)
                if entry not in self._forms.setdefault(name.keys, []):
                    self._forms[name.keys].append(entry)
        self._lengths = sorted({len(k) for k in self._forms}, reverse=True)

    def __len__(self) -> int:
        """How many known entities can be found: those with a usable name."""
        return len({key for entries in self._forms.values() for key, _ in entries})

    def find(self, text: str) -> list[Mention]:
        """Every mention of a known entity in `text`, in order and never overlapping."""
        found: list[Mention] = []
        for sentence in _sentences(text) if self._forms else ():
            words = sentence.words
            keys = [w.key for w in words]
            i = 0
            while i < len(words):
                step = 1
                for n in self._lengths:
                    if i + n > len(words):
                        continue
                    named = frozenset(
                        key
                        for key, capital in self._forms.get(tuple(keys[i : i + n]), ())
                        if words[i].capital or not capital
                    )
                    if named and not (words[i].capital and _extends(text, words, i, i + n)):
                        found.append((named, words[i].start, words[i + n - 1].end))
                        step = n
                        break
                i += step
        return found


def _entities(fact: Fact) -> tuple[Entity, ...]:
    if fact.object_entity is None:
        return (fact.subject,)
    return (fact.subject, fact.object_entity)


def _names(entity: Entity) -> list[str]:
    return [name for name in (entity.label, *entity.aliases) if name]


def known_entities(facts: Iterable[Fact]) -> list[Entity]:
    """Every subject and edge object in the batch, once each: the entities a batch knows."""
    found: dict[str, Entity] = {}
    for fact in facts:
        for entity in _entities(fact):
            found.setdefault(entity.key, entity)
    return list(found.values())


class Coverage(Frozen):
    """One document's gaps: where a fact could have come from and none did."""

    doc_id: str
    # Sentences naming two or more known entities: the ones a fact could come from.
    sentences: int = 0
    # Those of them that no fact's evidence covers, each with its text.
    uncovered: tuple[Span, ...] = ()
    # The first mention of each known entity that no fact of this document names.
    missed: tuple[Span, ...] = ()


class CoverageReport(Frozen):
    """Every document's `Coverage`, and the relations the extractor never reached."""

    documents: tuple[Coverage, ...] = ()
    # Predicates the extractor was not shown; None when it cannot say what it showed.
    not_offered: tuple[str, ...] | None = None
    # Predicates it was shown (all of them, when that is unknown) that no fact used.
    unused: tuple[str, ...] = ()

    @property
    def sentences(self) -> int:
        return sum(d.sentences for d in self.documents)

    @property
    def uncovered(self) -> int:
        return sum(len(d.uncovered) for d in self.documents)

    @property
    def missed(self) -> int:
        return sum(len(d.missed) for d in self.documents)

    def stats(self) -> dict[str, Any]:
        """The report as `KnowledgeGraph.stats["coverage"]` holds it: totals, then each document."""
        return {
            "sentences": self.sentences,
            "uncovered": self.uncovered,
            "missed_entities": self.missed,
            "not_offered": None if self.not_offered is None else list(self.not_offered),
            "unused": list(self.unused),
            "documents": [d.model_dump(mode="json") for d in self.documents],
        }


def summary(stats: Mapping[str, Any]) -> str:
    """One line from `CoverageReport.stats()`: what `odke run` prints and the ablation notes."""
    not_offered = stats.get("not_offered")
    offered = (
        "relations offered: unknown"
        if not_offered is None
        else f"{len(not_offered)} relations never offered"
    )
    return (
        f"{stats.get('uncovered', 0)} of {stats.get('sentences', 0)} sentences naming two "
        f"known entities uncovered, {stats.get('missed_entities', 0)} entities in no fact, "
        f"{offered}, {len(stats.get('unused', ()))} unused"
    )


def offered_by(extractor: object, ontology: Ontology) -> list[str] | None:
    """The predicates `extractor` shows its model, from its `offered(ontology)`; None if unknown."""
    offered = getattr(extractor, "offered", None)
    if not callable(offered):
        return None
    shown = offered(ontology)
    return None if shown is None else list(shown)


def measure(
    documents: Sequence[Document],
    facts: Iterable[Fact],
    ontology: Ontology,
    *,
    offered: Collection[str] | None = None,
    regions: Mapping[str, Sequence[Region]] | None = None,
) -> CoverageReport:
    """The coverage report for `documents`, given every fact extracted from the batch.

    A fact counts for each document its evidence cites. `offered` is what the
    extractor was shown (`offered_by`); left out, nothing is known to have been
    withheld. `regions` limits a document to the `(start, end)` ranges that were
    extracted from, so a chunk the router skipped is not counted as a gap; left
    out, the whole text counts.
    """
    every = list(facts)
    matcher = NameMatcher(known_entities(every))
    by_doc: dict[str, list[Fact]] = {}
    for fact in every:
        for doc_id in dict.fromkeys(e.doc_id for e in fact.evidence):
            by_doc.setdefault(doc_id, []).append(fact)
    records = tuple(
        _document(
            doc,
            by_doc.get(doc.id, []),
            matcher,
            None if regions is None else regions.get(doc.id, ()),
        )
        for doc in documents
    )
    used = {fact.predicate for fact in every}
    shown = set(ontology.predicates) if offered is None else set(offered)
    return CoverageReport(
        documents=records,
        not_offered=None if offered is None else tuple(sorted(set(ontology.predicates) - shown)),
        unused=tuple(sorted(p for p in ontology.predicates if p in shown and p not in used)),
    )


def _document(
    doc: Document, facts: Sequence[Fact], matcher: NameMatcher, regions: Sequence[Region] | None
) -> Coverage:
    def inside(start: int, end: int) -> bool:
        return regions is None or any(start < e and s < end for s, e in regions)

    hits = [hit for hit in matcher.find(doc.text) if inside(hit[1], hit[2])]
    named = {identity(entity) for fact in facts for entity in _entities(fact)}
    missed: dict[frozenset[str], Span] = {}
    for keys, start, end in hits:
        if not keys & named and keys not in missed:
            missed[keys] = _span(doc, start, end)

    # Where the facts are: each cited or located span, and for a fact whose span
    # is only the whole text, the window the span locator finds for it.
    cited: list[Region] = []
    for fact in facts:
        for evidence in fact.evidence:
            if evidence.span is None or check_span(evidence, doc) is not SpanStatus.LOCATED:
                continue
            span = (
                locate_span(fact, doc)
                if evidence.span_origin is SpanOrigin.CONTEXT
                else evidence.span
            )
            if span is not None:
                cited.append((span.start, span.end))

    starts = [start for _, start, _ in hits]
    counted = 0
    uncovered: list[Span] = []
    for start, end in sentences(doc.text):
        if not inside(start, end):
            continue
        at = bisect_left(starts, start)
        if len({keys for keys, _, _ in hits[at : bisect_left(starts, end)]}) < 2:
            continue
        counted += 1
        if not any(s < end and start < e for s, e in cited):
            uncovered.append(_span(doc, start, end))
    return Coverage(
        doc_id=doc.id, sentences=counted, uncovered=tuple(uncovered), missed=tuple(missed.values())
    )


def _span(doc: Document, start: int, end: int) -> Span:
    return Span(doc_id=doc.id, start=start, end=end, quote=doc.text[start:end])


__all__ = [
    "Coverage",
    "CoverageReport",
    "NameMatcher",
    "identity",
    "known_entities",
    "measure",
    "offered_by",
    "summary",
]
