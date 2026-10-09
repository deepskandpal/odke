"""LangExtract's annotated documents as triples rows (#96).

LangExtract (google/langextract) locates every extraction in its text and
records where as `char_interval`, which is a citation: the offsets become the
row's `start` and `end`, and `extraction_text` its `quote` unless the alignment
was fuzzy or partial, which says the text found there is not the extraction's.
An extraction LangExtract could not locate has `char_interval = None` and is
grounded against its whole text, marked `SpanOrigin.CONTEXT` (DECISIONS #25).

An extraction is an entity with attributes, not a triple, so reading one as
triples is a convention. The default: each attribute is a triple about the
extraction, `(extraction_text, key, value)`, with the class as the subject's
type. An extraction with no attributes is an entity and states nothing to
ground, so it gives no row. Anything else is the caller's mapping, passed as
`triples=`: it is handed each extraction in LangExtract's JSON form and returns
the rows' contents, and the extraction's evidence is added to every one.

The file is read, and the objects by attribute; langextract is never imported.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openodke.interop.triples import TripleRow
from openodke.types import Document

# One extraction in LangExtract's JSON form, to the contents of its rows: any
# `TripleRow` field but `doc`. `start`, `end` and `quote` default to the
# extraction's own.
Triples = Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]]

# The fields of `langextract.data.Extraction` its JSON form keeps.
_FIELDS = (
    "extraction_class",
    "extraction_text",
    "char_interval",
    "alignment_status",
    "extraction_index",
    "group_index",
    "description",
    "attributes",
)


def attribute_triples(extraction: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The default reading: each attribute is a triple about the extraction.

    The extraction's text is the subject and its class the subject's type; an
    attribute's key is the predicate and its value the object, one row for each
    value of a list. The object is a literal unless the ontology's range for the
    predicate is a type. No attributes, no rows.
    """
    rows = []
    for key, value in (extraction.get("attributes") or {}).items():
        for item in value if isinstance(value, list) else [value]:
            if item is None or item == "":
                continue
            rows.append(
                {
                    "subject": extraction["extraction_text"],
                    "subject_type": extraction.get("extraction_class") or None,
                    "predicate": str(key),
                    "object": item if isinstance(item, str | int | float | bool) else str(item),
                    "object_type": "string",
                }
            )
    return rows


def from_langextract(
    annotated: str | Path | Any | Iterable[Any],
    *,
    triples: Triples | None = None,
) -> tuple[list[TripleRow], list[Document]]:
    """Rows for LangExtract's extractions, and the texts they were found in.

    `annotated` is the JSON Lines file `lx.io.save_annotated_documents` writes,
    the `AnnotatedDocument` (or list of them) `lx.extract` returns, or their
    JSON form as dicts. Each document's `document_id` and `text` become a
    `Document`. Under the default reading, extractions that give no row are
    counted in a warning, so a corpus of bare entities does not vanish quietly.
    """
    mapping = triples or attribute_triples
    rows: list[TripleRow] = []
    texts: dict[str, Document] = {}
    entities = total = 0
    for item in _read(annotated):
        doc = _document(item)
        if texts.setdefault(doc.id, doc).text != doc.text:
            raise ValueError(f"two annotated documents have the id {doc.id!r} and different texts")
        for index, raw in enumerate(_get(item, "extractions") or ()):
            extraction = _as_json(raw)
            total += 1
            evidence = _evidence(extraction)
            made = list(mapping(extraction))
            entities += not made
            for content in made:
                try:
                    rows.append(TripleRow.model_validate({**evidence, **content, "doc": doc.id}))
                except ValidationError as exc:
                    error = exc.errors()[0]
                    where = ".".join(str(part) for part in error["loc"])
                    raise ValueError(
                        f"document {doc.id!r}, extraction {index}: "
                        f"{where + ': ' if where else ''}{error['msg']}"
                    ) from None
    if entities and triples is None:
        warnings.warn(
            f"{entities} of {total} LangExtract extractions have no attributes, so they are "
            "entities and state no triple; they were left out. Pass triples= to read them",
            stacklevel=2,
        )
    return rows, list(texts.values())


def _read(source: str | Path | Any | Iterable[Any]) -> Iterable[Any]:
    if isinstance(source, str | Path):
        path = Path(source)
        out = []
        with path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: not JSON: {exc.msg}") from None
        return out
    if _get(source, "extractions") is not None or _get(source, "text") is not None:
        return [source]
    return list(source)


def _get(obj: Any, name: str) -> Any:
    """A field of a LangExtract object, or of its JSON form."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_json(extraction: Any) -> dict[str, Any]:
    """An `Extraction`, or its JSON form, as the JSON form."""
    out: dict[str, Any] = {name: _get(extraction, name) for name in _FIELDS}
    interval = out["char_interval"]
    if interval is not None:
        out["char_interval"] = {
            "start_pos": _get(interval, "start_pos"),
            "end_pos": _get(interval, "end_pos"),
        }
    if isinstance(out["alignment_status"], Enum):
        out["alignment_status"] = out["alignment_status"].value
    return out


def _document(annotated: Any) -> Document:
    doc_id, text = _get(annotated, "document_id"), _get(annotated, "text")
    if not doc_id:
        raise ValueError("an annotated document has no document_id")
    if text is None:
        raise ValueError(f"annotated document {doc_id!r} has no text to ground on")
    return Document(id=str(doc_id), text=text)


def _evidence(extraction: Mapping[str, Any]) -> dict[str, Any]:
    """The extraction's offsets and quote, or nothing when LangExtract could not place it."""
    interval = extraction.get("char_interval") or {}
    start, end = interval.get("start_pos"), interval.get("end_pos")
    if start is None or end is None:
        return {}
    # A fuzzy or partial alignment found text other than the extraction's: the
    # offsets are still LangExtract's citation, and the quote would not match
    # them. An exact one keeps it, so offsets that do not hold it are refused.
    inexact = extraction.get("alignment_status") not in (None, "match_exact")
    quote = None if inexact else extraction.get("extraction_text") or None
    return {"start": start, "end": end, "quote": quote}


__all__ = ["Triples", "attribute_triples", "from_langextract"]
