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
way covers the sentences naming both its subject and its object instead, which
is where the span locator (#112) will put it.

Names are compared as `name_key` token runs: casefolded, accents dropped, a
dotted initialism closed up, a leading "the" and a trailing legal form ignored.
The matcher here is a small one until the span locator's lands; then the two
should be one.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from openodke.chunking import sentences
from openodke.corroborate.normalize import name_key
from openodke.ground.span import SpanStatus, check_span
from openodke.ontology import Ontology
from openodke.types import Document, Entity, Fact, Frozen, Span, SpanOrigin

# A dotted initialism is one token, as `name_key` closes "U.S." up into "us";
# "&" is a token too, because `name_key` reads it as "and".
_TOKEN = re.compile(r"(?:[^\W_]\.){2,}|[^\W_]+|&")
_WORD = re.compile(r"[^\W_]+")

Region = tuple[int, int]


class NameMatcher:
    """Finds known names in a text: `name_key` token runs, the longest first.

    A name whose key is a single character is left out, because it matches
    nearly everything. Deliberately small; see the module's last paragraph.
    """

    def __init__(self, names: Iterable[str]) -> None:
        self._keys: dict[tuple[str, ...], str] = {}
        for name in names:
            key = name_key(name)
            if len(key.replace(" ", "")) > 1:
                self._keys.setdefault(tuple(key.split()), key)
        self._lengths = sorted({len(k) for k in self._keys}, reverse=True)

    def __len__(self) -> int:
        return len(self._keys)

    def find(self, text: str) -> list[tuple[str, int, int]]:
        """`(key, start, end)` of every known name in `text`, in order and never overlapping."""
        if not self._keys:
            return []
        tokens = _tokens(text)
        found: list[tuple[str, int, int]] = []
        i = 0
        while i < len(tokens):
            for n in self._lengths:
                if i + n > len(tokens):
                    continue
                key = self._keys.get(tuple(t for t, _, _ in tokens[i : i + n]))
                if key is not None:
                    found.append((key, tokens[i][1], tokens[i + n - 1][2]))
                    i += n
                    break
            else:
                i += 1
        return found


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """Each word of `text` folded as `name_key` folds it, with the offsets it came from."""
    out: list[tuple[str, int, int]] = []
    for match in _TOKEN.finditer(text):
        raw = match.group()
        if raw == "&":
            out.append(("and", match.start(), match.end()))
            continue
        folded = unicodedata.normalize("NFKD", raw.replace(".", ""))
        folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).casefold()
        out.extend((part, match.start(), match.end()) for part in _WORD.findall(folded))
    return out


def _entities(fact: Fact) -> tuple[Entity, ...]:
    if fact.object_entity is None:
        return (fact.subject,)
    return (fact.subject, fact.object_entity)


def _names(entity: Entity) -> list[str]:
    return [name for name in (entity.label, *entity.aliases) if name]


def known_names(facts: Iterable[Fact]) -> list[str]:
    """Every subject's and edge object's label and aliases: the names a batch knows."""
    found: dict[str, None] = {}
    for fact in facts:
        for entity in _entities(fact):
            found.update(dict.fromkeys(_names(entity)))
    return list(found)


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
    matcher = NameMatcher(known_names(every))
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
    named = {name_key(n) for fact in facts for entity in _entities(fact) for n in _names(entity)}
    missed: dict[str, Span] = {}
    for key, start, end in hits:
        if key not in named and key not in missed:
            missed[key] = _span(doc, start, end)

    cited: list[Region] = []
    uncited: list[Fact] = []
    for fact in facts:
        for evidence in fact.evidence:
            if evidence.span is None or check_span(evidence, doc) is not SpanStatus.LOCATED:
                continue
            if evidence.span_origin is SpanOrigin.CONTEXT:
                uncited.append(fact)
            else:
                cited.append((evidence.span.start, evidence.span.end))

    starts = [start for _, start, _ in hits]
    counted = 0
    uncovered: list[Span] = []
    for start, end in sentences(doc.text):
        if not inside(start, end):
            continue
        at = bisect_left(starts, start)
        keys = {key for key, s, _ in hits[at : bisect_left(starts, end)]}
        if len(keys) < 2:
            continue
        counted += 1
        if any(s < end and start < e for s, e in cited):
            continue
        text = doc.text[start:end]
        if any(_names_its_claim(fact, keys, text) for fact in uncited):
            continue
        uncovered.append(_span(doc, start, end))
    return Coverage(
        doc_id=doc.id, sentences=counted, uncovered=tuple(uncovered), missed=tuple(missed.values())
    )


def _names_its_claim(fact: Fact, keys: set[str], text: str) -> bool:
    """Whether a sentence names a context-cited fact's subject, and its object or value."""
    if not {name_key(n) for n in _names(fact.subject)} & keys:
        return False
    if fact.object_entity is not None:
        return bool({name_key(n) for n in _names(fact.object_entity)} & keys)
    value = str(fact.object_value).strip().casefold() if fact.object_value is not None else ""
    return bool(value) and value in text.casefold()


def _span(doc: Document, start: int, end: int) -> Span:
    return Span(doc_id=doc.id, start=start, end=end, quote=doc.text[start:end])


__all__ = [
    "Coverage",
    "CoverageReport",
    "NameMatcher",
    "known_names",
    "measure",
    "offered_by",
    "summary",
]
