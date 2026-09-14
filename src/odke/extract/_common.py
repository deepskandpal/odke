"""What every extractor shares, so a pattern fact and a model fact are one shape.

The hybrid extractor merges the two by `Fact.signature`, which keys on
`subject.key`. If the pattern path and the model path spelled one entity's key
differently, nothing would ever merge — so both build entities, spans and
evidence here, and nowhere else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias

from odke.ontology import Ontology, Predicate
from odke.types import Chunk, Document, Entity, Evidence, Fact, Polarity, Span

# What an extractor accepts as its document lookup: by id, or just the documents.
Documents: TypeAlias = Mapping[str, Document] | Iterable[Document] | None

_EDGE_PUNCTUATION = " \t\r\n.,;:!?\"'()[]{}"
_FOLD = re.compile(r"[\W_]+")


def index_documents(documents: Documents) -> dict[str, Document]:
    """Documents by id.

    DECISIONS #19 keeps `Chunk` to offsets and text, so an extractor that needs a
    document's modality, tier or URI looks it up here. `Pipeline.run` is handed
    the documents; the caller hands the same ones to the extractor, and can add
    more later with `extractor.documents[doc.id] = doc`.
    """
    if documents is None:
        return {}
    items = documents.values() if isinstance(documents, Mapping) else documents
    return {doc.id: doc for doc in items}


def entity_key(type_name: str, name: str) -> str:
    """The key both extractors give an entity: its type and its folded name.

    Deliberately naive — case, spacing and surrounding punctuation only. Deciding
    that "A. Lovelace" is "Ada Lovelace" is the resolver's job (M3), and doing it
    here would hide the decision where no `Resolution` records it. The type is in
    the key so that Paris the person and Paris the city stay two nodes.
    """
    folded = " ".join(name.split()).strip(_EDGE_PUNCTUATION).casefold()
    return f"{type_name}:{folded}"


def fold(label: str) -> str:
    """Case- and punctuation-blind form for matching a header or key to a predicate."""
    return _FOLD.sub("", label.casefold())


class PredicateNames:
    """Ontology predicates by name, label and alias, matched on `fold`.

    A name beats a label beats an alias, so one predicate's alias can never
    shadow another predicate's own name.
    """

    def __init__(self, ontology: Ontology) -> None:
        self._by: dict[str, Predicate] = {}
        predicates = list(ontology.predicates.values())
        for p in predicates:
            self._by.setdefault(fold(p.name), p)
        for p in predicates:
            if p.label:
                self._by.setdefault(fold(p.label), p)
        for p in predicates:
            for alias in p.aliases:
                self._by.setdefault(fold(alias), p)
        self._by.pop("", None)

    def match(self, label: str) -> Predicate | None:
        return self._by.get(fold(label))


def applies(ontology: Ontology, predicate: Predicate, type_name: str) -> bool:
    lineage = ontology.lineage(type_name)
    return not predicate.domain or any(d in lineage for d in predicate.domain)


def best_type(
    ontology: Ontology, predicates: Iterable[Predicate], preferred: str | None = None
) -> str | None:
    """The entity type a group of predicates most plausibly describes.

    Only predicates with a domain vote: an open predicate fits every type and
    says nothing. Most votes wins; a tie goes to the type carrying the fewest
    predicates overall — the tightest fit, so `Person` beats `Scientist` for a
    row with no scientist-only column — and then to the name, so the choice is
    stable. None when nothing votes, rather than a guess.
    """
    if preferred is not None:
        return preferred
    voting = [p for p in predicates if p.domain]
    best: tuple[int, int, str] | None = None
    for type_name in ontology.types:
        votes = sum(applies(ontology, p, type_name) for p in voting)
        if votes:
            rank = (-votes, len(ontology.predicates_for(type_name)), type_name)
            best = rank if best is None or rank < best else best
    return None if best is None else best[2]


def key_predicates(ontology: Ontology, type_name: str) -> list[str]:
    """The identity predicates (`EntityType.keys`) of a type, then of its ancestors."""
    order: list[str] = []
    seen: set[str] = set()
    queue = [type_name]
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        node = ontology.types.get(current)
        if node is not None:
            order.extend(k for k in node.keys if k not in order)
            queue.extend(node.parents)
    return order


def subject_entity(type_name: str, name: str) -> Entity:
    return Entity(key=entity_key(type_name, name), type=type_name, label=name)


class ChunkContext:
    """A chunk and, when known, its document: where spans are made and checked."""

    def __init__(self, chunk: Chunk, doc: Document | None) -> None:
        self.chunk = chunk
        self.doc = doc
        # The chunk as a document of its own, so a chunk whose document the
        # caller never registered is still checked by `Span.is_faithful`.
        self._view = Document(id=chunk.doc_id, text=chunk.text)

    def span(self, start: int, quote: str) -> Span | None:
        """`quote` claimed at chunk-local `start`, as a document span — or None.

        Checked twice, both times with `Span.is_faithful`: against the chunk,
        then against the document at the shifted offsets when the document is
        known, which also catches a chunker whose text drifted from the
        document it claims to slice.
        """
        if not quote or start < 0:
            return None
        local = Span(doc_id=self.chunk.doc_id, start=start, end=start + len(quote), quote=quote)
        if not local.is_faithful(self._view):
            return None
        at = self.chunk.start + start
        span = Span(doc_id=self.chunk.doc_id, start=at, end=at + len(quote), quote=quote)
        if self.doc is not None and not span.is_faithful(self.doc):
            return None
        return span

    def evidence(self, span: Span) -> Evidence:
        if self.doc is None:
            return Evidence(doc_id=span.doc_id, span=span)
        return Evidence(
            doc_id=self.doc.id,
            span=span,
            uri=self.doc.uri,
            tier=self.doc.tier,
            retrieved_at=self.doc.retrieved_at,
        )

    def fact(
        self,
        ontology: Ontology,
        *,
        subject: Entity,
        predicate: Predicate,
        value: Any,
        span: Span,
        extractor: str,
        confidence: float,
        polarity: Polarity = Polarity.ASSERTED,
        qualifiers: Mapping[str, Any] | None = None,
    ) -> Fact:
        """One fact, with `identity_keys` stamped from the ontology (DECISIONS #15).

        A predicate whose range is an entity type makes an edge to an entity
        keyed exactly as a subject of that name would be.
        """
        object_entity: Entity | None = None
        object_value = value
        if predicate.is_edge_in(ontology):
            object_entity, object_value = subject_entity(predicate.range, str(value)), None
        return Fact(
            subject=subject,
            predicate=predicate.name,
            object_entity=object_entity,
            object_value=object_value,
            polarity=polarity,
            qualifiers=dict(qualifiers or {}),
            identity_keys=ontology.identity_keys(predicate.name),
            evidence=(self.evidence(span),),
            extractor=extractor,
            confidence=confidence,
        )
