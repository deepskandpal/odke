"""Triples from any extractor: one documented row format, and a stage that replays it.

openodke checks facts whoever extracted them (DECISIONS #24). That needs one
place where any extractor's output can land: a JSON Lines file with one triple
per row, next to the texts the triples came from. The adapters map onto this
format, and a pattern extractor or a homemade script can write it directly.

A row is the lowest common denominator of what other tools emit, not a second
`Fact`. A text id, a subject, a predicate and an object are required; the rest
is optional. What a row carries decides how its evidence is made:

- **offsets** (`start`, `end`, and optionally the `quote` found at them) are a
  citation, checked for free by `SpanGrounder` like any extractor's;
- **a quote and no offsets** is a citation too, found by exact match in the
  text. A quote that is not in the text gets no span, and the span check
  refuses the fact without a model call;
- **neither** grounds the fact against the whole text it came from, marked
  `SpanOrigin.CONTEXT` so it never passes for a citation (DECISIONS #25).

The ontology, when there is one, decides what the extractor's labels cannot.
A predicate whose range is an entity type makes an edge, and a subject with no
type takes the predicate's domain. Without one, a string object is an entity
unless `object_type` names a literal type, and anything untyped is a `Thing`.

The format is versioned (`schema_version`, "1.0"; DECISIONS #48), and
`triples.schema.json`, beside this file, is its JSON Schema for writers in any
language. A row may name the version it was written to, and one that names
another major version is refused by name. Left out, a row is read as this one.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import Field, ValidationError, field_validator, model_validator

from openodke.extract._common import entity_key
from openodke.ground.span import Counts
from openodke.ontology import Ontology, Predicate
from openodke.types import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    Frozen,
    Polarity,
    Span,
    SpanOrigin,
)

# The format's version: a minor adds optional fields, a major changes them.
SCHEMA_VERSION = "1.0"
SCHEMA_PATH = Path(__file__).with_name("triples.schema.json")

# What an entity is called when nothing says what it is.
THING = "Thing"
# `object_type` values that name a literal rather than a node: the ontology's
# literal ranges, and "text", which graph extractors use for a string node.
LITERAL_TYPES = frozenset(
    {"string", "text", "integer", "number", "float", "boolean", "date", "datetime", "quantity"}
)


class TripleRow(Frozen):
    """One triple, as any extractor can write it. The format is this model.

    `doc` names the text the triple came from: a `Document.id`, or the file
    name of a loaded document without its suffix (`notes/acme.txt` is `acme`).
    A chunk an extractor read on its own is a text too; give it an id and
    load its text like any other.
    """

    # The format version the row was written to; left out, it is this one.
    schema_version: str = SCHEMA_VERSION
    doc: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str | int | float | bool
    subject_type: str | None = None
    object_type: str | None = None
    # Character offsets into the text, half-open, and the text found there.
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    quote: str | None = None
    polarity: Polarity = Polarity.ASSERTED
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    # Unset, these take the stage's own: the extractor's name, and a prior the
    # scorer calibrates.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    extractor: str | None = None
    # Give one to join the row's fact to labels or to another run.
    id: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _one_major_version(cls, value: str) -> str:
        major, _, minor = value.partition(".")
        if not (major.isdigit() and minor.isdigit()):
            raise ValueError(
                f"schema_version {value!r} is not major.minor, such as {SCHEMA_VERSION!r}"
            )
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"schema_version {value!r}: this openodke reads {SCHEMA_VERSION}")
        return value

    @model_validator(mode="after")
    def _offsets_come_in_pairs(self) -> TripleRow:
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end come together: give both offsets or neither")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        if self.quote == "":
            raise ValueError("quote is empty: leave it out instead")
        return self


def read_triples(
    source: str | Path | Iterable[TripleRow | Mapping[str, Any]],
) -> Iterator[TripleRow]:
    """Rows from a JSON Lines file, or from rows already in hand, one at a time (#158).

    An iterator: a file is read a line at a time as the rows are asked for, so
    a file of any size can be streamed. `list(read_triples(path))` reads it all.

    A bad row is an error naming its line, never a row skipped in silence: a
    triple that cannot be read is a fact that would vanish from the count. It
    is raised when the iterator reaches it.
    """
    if not isinstance(source, str | Path):
        for row in source:
            yield row if isinstance(row, TripleRow) else TripleRow.model_validate(row)
        return
    path = Path(source)
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = TripleRow.model_validate(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not JSON: {exc.msg}") from None
            except ValidationError as exc:
                error = exc.errors()[0]
                where = ".".join(str(part) for part in error["loc"])
                raise ValueError(
                    f"{path}:{number}: {where + ': ' if where else ''}{error['msg']}"
                ) from None
            yield row


def to_fact(
    row: TripleRow,
    doc: Document,
    ontology: Ontology | None = None,
    *,
    extractor: str = "triples",
    confidence: float = 0.5,
) -> Fact:
    """The `Fact` a row states, with its evidence made from `doc`."""
    predicate = _predicate(ontology, row.predicate)
    domain = predicate.domain[0] if predicate is not None and len(predicate.domain) == 1 else None
    subject_type = _type(ontology, row.subject_type) or domain or row.subject_type or THING
    subject = Entity(
        key=entity_key(subject_type, row.subject), type=subject_type, label=row.subject
    )
    object_entity: Entity | None = None
    object_value: Any = row.object
    object_type = _object_type(row, predicate, ontology)
    if object_type is not None:
        label = str(row.object)
        object_entity = Entity(key=entity_key(object_type, label), type=object_type, label=label)
        object_value = None
    fact = Fact(
        subject=subject,
        predicate=predicate.name if predicate is not None else row.predicate,
        object_entity=object_entity,
        object_value=object_value,
        polarity=row.polarity,
        qualifiers=dict(row.qualifiers),
        identity_keys=(
            ontology.identity_keys(predicate.name)
            if ontology is not None and predicate is not None
            else ()
        ),
        evidence=(_evidence(row, doc),),
        extractor=row.extractor or extractor,
        confidence=confidence if row.confidence is None else row.confidence,
    )
    return fact.model_copy(update={"id": row.id}) if row.id else fact


class TriplesExtractor:
    """Another extractor's triples, as the extract stage: everything after it runs unchanged.

    `odke run` names it `triples`, with the rows' `path`. The documents are
    the run's inputs, and each row finds its text by `Document.id` or by file
    name. In Python, pass `documents=` the same documents the pipeline runs on,
    or the stage reads each text from the chunk it is handed. A row has one
    text: when two documents share the file name it gives, the first to arrive
    gets its rows and the other gets none, with a warning naming both.

    Triples are not chunked. A document's triples all arrive with its first
    chunk, so leave the chunker out; a router that skips a first chunk skips
    that document's triples. `stats` counts how each row's evidence was made,
    the rows whose text never came, and the rows whose file name was ambiguous.

    A file is read through once when the stage is made, so a bad row is an
    error before anything runs, and its rows are read again, and kept, the
    first time they are asked for. `feed(rows)` swaps in the next
    micro-batch's rows, which is how the Validator streams a file (#158).
    """

    name = "triples"

    def __init__(
        self,
        triples: str | Path | Iterable[TripleRow | Mapping[str, Any]],
        *,
        extractor: str = "triples",
        confidence: float = 0.5,
        documents: Iterable[Document] = (),
    ) -> None:
        self.extractor = extractor
        self.confidence = confidence
        # The file, or the rows, the stage was made from.
        self.source = triples
        self._rows: dict[str, list[TripleRow]] | None = None
        if isinstance(triples, str | Path):
            for _ in read_triples(triples):
                pass
        else:
            self._rows = _by_doc(read_triples(triples))
        # Filled by `odke run` with the loaded inputs (DECISIONS #19).
        self.documents: dict[str, Document] = {doc.id: doc for doc in documents}
        # Each row `doc` and the document its rows went to. A file name is not
        # unique (2023/report.txt and 2024/report.txt are both `report`), and
        # one row must not become a fact about each text that shares its name.
        self._served: dict[str, str] = {}
        self._ambiguous: set[str] = set()
        self._counts = Counts("rows", "cited", "quoted", "quote_not_found", "context")
        # What rows fed before the current ones left unmatched or ambiguous.
        self._unmatched = 0
        self._unmatched_docs: list[str] = []
        self._ambiguous_rows = 0

    @property
    def rows(self) -> dict[str, list[TripleRow]]:
        """The rows by the text they name, read from the file the first time they are asked for."""
        if self._rows is None:
            self._rows = _by_doc(read_triples(self.source))
        return self._rows

    @property
    def served(self) -> Mapping[str, str]:
        """Each row `doc` that found its text, and the id of the document its rows went to."""
        return self._served

    def feed(self, rows: Iterable[TripleRow | Mapping[str, Any]]) -> None:
        """The next micro-batch's rows, in place of the ones held (#158).

        Which document each name went to is kept, so a name split across
        micro-batches goes to one text, and the counts go on: the rows replaced
        stay counted as unmatched or ambiguous when they were.
        """
        if self._rows is not None:
            unmatched = [doc for doc in self._rows if doc not in self._served]
            self._unmatched += sum(len(self._rows[doc]) for doc in unmatched)
            self._unmatched_docs = sorted({*self._unmatched_docs, *unmatched})[:20]
            self._ambiguous_rows += sum(
                len(held) for doc, held in self._rows.items() if doc in self._ambiguous
            )
        self._rows = _by_doc(read_triples(rows))

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        if chunk.index != 0:
            return []
        doc = self.documents.get(chunk.doc_id) or Document(id=chunk.doc_id, text=chunk.text)
        key = doc.id if doc.id in self.rows else _file_name(doc)
        if key is None or key not in self.rows:
            return []
        owner = self._served.setdefault(key, doc.id)
        if owner != doc.id:
            # Which of the two the rows meant is not written anywhere, so the
            # first keeps them and the second is told, rather than given copies.
            self._ambiguous.add(key)
            warnings.warn(
                f"the triples for {key!r} went to document {owner!r}, and document "
                f"{doc.id!r} answers to {key!r} too, so it gets none; name each text "
                "by its id to tell them apart",
                stacklevel=2,
            )
            return []
        facts = []
        for row in self.rows[key]:
            fact = to_fact(row, doc, ontology, extractor=self.extractor, confidence=self.confidence)
            self._counts.bump("rows")
            self._counts.bump(_how(row, fact))
            facts.append(fact)
        return facts

    @property
    def stats(self) -> dict[str, Any]:
        """Rows replayed, by how their evidence was made; rows no document or two matched."""
        counts: dict[str, Any] = dict(self._counts.snapshot())
        rows = self.rows
        unmatched = [doc for doc in rows if doc not in self._served]
        counts["unmatched_rows"] = self._unmatched + sum(len(rows[doc]) for doc in unmatched)
        named = sorted({*self._unmatched_docs, *unmatched})[:20]
        if named:
            counts["unmatched_docs"] = named
        ambiguous = sorted(self._ambiguous)
        counts["ambiguous_rows"] = self._ambiguous_rows + sum(
            len(rows[doc]) for doc in ambiguous if doc in rows
        )
        if ambiguous:
            counts["ambiguous_docs"] = ambiguous[:20]
        return counts


def _by_doc(rows: Iterable[TripleRow]) -> dict[str, list[TripleRow]]:
    grouped: dict[str, list[TripleRow]] = defaultdict(list)
    for row in rows:
        grouped[row.doc].append(row)
    return grouped


def _evidence(row: TripleRow, doc: Document) -> Evidence:
    def cite(span: Span | None, origin: SpanOrigin = SpanOrigin.CITED) -> Evidence:
        return Evidence(
            doc_id=doc.id,
            span=span,
            span_origin=origin,
            uri=doc.uri,
            tier=doc.tier,
            retrieved_at=doc.retrieved_at,
        )

    if row.start is not None and row.end is not None:
        # Not checked here: `SpanGrounder` checks every span, and counts what it refuses.
        return cite(Span(doc_id=doc.id, start=row.start, end=row.end, quote=row.quote))
    if row.quote is not None:
        at = doc.text.find(row.quote)
        if at < 0:
            return cite(None)
        return cite(Span(doc_id=doc.id, start=at, end=at + len(row.quote), quote=row.quote))
    return cite(Span(doc_id=doc.id, start=0, end=len(doc.text)), SpanOrigin.CONTEXT)


def _how(row: TripleRow, fact: Fact) -> str:
    if row.start is not None:
        return "cited"
    if row.quote is not None:
        return "quoted" if fact.evidence[0].span is not None else "quote_not_found"
    return "context"


def _predicate(ontology: Ontology | None, name: str) -> Predicate | None:
    if ontology is None:
        return None
    found = ontology.predicates.get(name)
    if found is not None:
        return found
    folded = name.casefold()
    return next((p for key, p in ontology.predicates.items() if key.casefold() == folded), None)


def _type(ontology: Ontology | None, name: str | None) -> str | None:
    """`name` as the ontology spells it, or None when it is not one of its types."""
    if ontology is None or name is None:
        return None
    if name in ontology.types:
        return name
    folded = name.casefold()
    return next((t for t in ontology.types if t.casefold() == folded), None)


def _object_type(
    row: TripleRow, predicate: Predicate | None, ontology: Ontology | None
) -> str | None:
    """The object's entity type, or None when the object is a literal."""
    if predicate is not None and ontology is not None:
        if not predicate.is_edge_in(ontology):
            return None
        return _type(ontology, row.object_type) or predicate.range
    if not isinstance(row.object, str):
        return None
    if row.object_type is None:
        return THING
    if row.object_type.casefold() in LITERAL_TYPES:
        return None
    return _type(ontology, row.object_type) or row.object_type


def _file_name(doc: Document) -> str | None:
    """A loaded file's name without its suffix: what a row's `doc` usually says."""
    if not doc.uri:
        return None
    return Path(unquote(urlparse(doc.uri).path)).stem or None


__all__ = [
    "LITERAL_TYPES",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "THING",
    "TripleRow",
    "TriplesExtractor",
    "read_triples",
    "to_fact",
]
