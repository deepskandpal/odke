"""A Neo4j graph read back as triples rows, to ground it where it stands (#98).

Two kinds of graph are read. One openodke wrote: `Neo4jSink` puts every fact's
provenance on its relationship, so each piece of evidence listed there (a
document id, its offsets, who chose them) becomes a row citing exactly what the
run cited. The texts are not in the graph. The caller passes the documents, and
each evidence finds its text by document id, then by URI. Any other graph names
the relationship property that holds the text each relationship was extracted
from: that text is its evidence, grounded whole and marked
`SpanOrigin.CONTEXT` (DECISIONS #25). A relationship with neither is left out,
and a warning counts them rather than guessing at a text.

Reading never writes. `write_verdicts` writes verdicts back onto the
relationships they were read from, by element id, and only when called. The
caller's `neo4j` driver does the talking, so nothing here imports it.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

from openodke.corroborate.provenance import CONFLICT, SCORE, SOURCE_FORM
from openodke.interop.triples import TripleRow
from openodke.sinks.neo4j import _PROVENANCE, CLAIM_LABEL, ENTITY_LABEL
from openodke.types import Document, Fact, GroundingVerdict, LinkKind, SpanOrigin

# The stamps a sink stores as JSON text, read back as the mappings they were.
_STAMPS = frozenset({SOURCE_FORM, CONFLICT, SCORE})
# Relationships openodke writes that are not facts: the resolver's links.
_LINKS = sorted(kind.value.upper() for kind in LinkKind)
# One verdict for a relationship read as several rows: the corroborator's order.
_PRECEDENCE = (
    GroundingVerdict.SUPPORTED,
    GroundingVerdict.CONTRADICTED,
    GroundingVerdict.NOT_FOUND,
    GroundingVerdict.UNCHECKED,
)


def _queries(text: bool, limited: bool) -> tuple[str, str]:
    """What to read, and how many relationships it leaves out, under the caller's options."""
    has_text = " OR r[$text] IS NOT NULL" if text else ""
    lacks_text = " AND r[$text] IS NULL" if text else ""
    of_type = "\n  AND type(r) IN $predicates" if limited else ""
    read = (
        "MATCH (s)-[r]->(o)\n"
        f"WHERE (r.evidence_doc_ids IS NOT NULL{has_text}){of_type}\n"
        "RETURN elementId(r) AS id, type(r) AS type, properties(r) AS properties,\n"
        "       labels(s) AS subject_labels, s.label AS subject_label,\n"
        "       s[$name] AS subject_name, s.key AS subject_key, elementId(s) AS subject_id,\n"
        "       labels(o) AS object_labels, o.label AS object_label,\n"
        "       o[$name] AS object_name, o.key AS object_key, elementId(o) AS object_id,\n"
        "       o.value AS value\n"
        "ORDER BY id"
    )
    unread = (
        "MATCH ()-[r]->()\n"
        f"WHERE r.evidence_doc_ids IS NULL{lacks_text} AND NOT type(r) IN $links{of_type}\n"
        "RETURN count(r) AS unread"
    )
    return read, unread


_WRITE = """UNWIND $rows AS row
MATCH ()-[r]->() WHERE elementId(r) = row.id
SET r += row.props
RETURN count(r) AS written"""


def read_neo4j(
    driver: Any,
    *,
    documents: Iterable[Document] = (),
    text_property: str | None = None,
    name_property: str = "name",
    predicates: Iterable[str] | None = None,
    database: str | None = None,
) -> tuple[list[TripleRow], list[Document]]:
    """Rows for the facts a Neo4j graph holds, and the texts they cite.

    `documents` are the texts a graph openodke wrote was built from. A row whose
    text is not among them keeps the document id it names, and
    `TriplesExtractor.stats` counts it as unmatched. `text_property` names the
    relationship property holding each relationship's source text, for a graph
    openodke did not write. `predicates` limits the read to those relationship
    types. Every row's `id` is its relationship's element id, which
    `write_verdicts` writes back by.
    """
    by_id = {doc.id: doc for doc in documents}
    by_uri = {doc.uri: doc for doc in by_id.values() if doc.uri}
    parameters = {
        "text": text_property,
        "name": name_property,
        "predicates": None if predicates is None else sorted(set(predicates)),
        "links": _LINKS,
    }

    queries = _queries(text_property is not None, predicates is not None)

    def read(tx: Any) -> tuple[list[dict[str, Any]], int]:
        found = list(tx.run(queries[0], **parameters).data())
        unread = tx.run(queries[1], **parameters).data()
        return found, int(unread[0]["unread"]) if unread else 0

    with driver.session(**({"database": database} if database else {})) as session:
        found, unread = session.execute_read(read)
    rows: list[TripleRow] = []
    texts: dict[str, Document] = {}
    for record in found:
        properties = dict(record["properties"] or {})
        claim: dict[str, Any] = {
            "subject": _name(record, "subject"),
            "subject_type": _type(record["subject_labels"]),
            "predicate": record["type"],
            "id": record["id"],
        }
        if CLAIM_LABEL in (record["object_labels"] or ()):
            value = _plain(record["value"])
            if not isinstance(value, str | int | float | bool):
                value = json.dumps(value, default=str, ensure_ascii=False)
            claim["object"] = value
            claim["object_type"] = "string" if isinstance(value, str) else None
        else:
            claim["object"] = _name(record, "object")
            claim["object_type"] = _type(record["object_labels"])
        if properties.get("evidence_doc_ids") is not None:
            claim.update(_receipts(properties))
            for cited in _evidence(properties):
                uri = cited.pop("uri")
                doc = by_id.get(cited["doc"]) or (by_uri.get(uri) if uri else None)
                if doc is not None:
                    cited["doc"] = texts.setdefault(doc.id, doc).id
                rows.append(TripleRow.model_validate({**claim, **cited}))
        else:
            text = str(properties[text_property]) if text_property else ""
            digest = hashlib.sha256(text.encode()).hexdigest()[:12]
            doc = texts.setdefault(f"neo4j-{digest}", Document(id=f"neo4j-{digest}", text=text))
            qualifiers = {
                k: _plain(v) for k, v in properties.items() if k != text_property and not _own(k)
            }
            rows.append(
                TripleRow.model_validate({**claim, "doc": doc.id, "qualifiers": qualifiers})
            )
    if unread:
        named = f"a {text_property!r} property" if text_property else "a text_property="
        warnings.warn(
            f"{unread} relationships carry neither openodke's provenance nor {named}, so "
            "nothing says what text to ground them against; they were left out",
            stacklevel=2,
        )
    return rows, list(texts.values())


def write_verdicts(
    driver: Any,
    facts: Iterable[Fact],
    *,
    property: str = "odke_verdict",
    database: str | None = None,
) -> int:
    """Write each fact's verdict onto the relationship it was read from; the count written.

    A fact's `id` is its relationship's element id, as `read_neo4j` gives it.
    Several facts read from one relationship (one per piece of evidence) give
    it one verdict, the corroborator's way: one support is enough, otherwise a
    contradiction outweighs silence. Facts merged by a corroborator carry the
    first relationship's id. Only `property` is written; `"verdict"` updates the
    sink's own receipt on a graph openodke wrote.
    """
    verdicts: dict[str, set[GroundingVerdict]] = {}
    for fact in facts:
        verdicts.setdefault(fact.id, set()).add(fact.verdict)
    rows = [
        {"id": rel, "props": {property: next(v for v in _PRECEDENCE if v in found).value}}
        for rel, found in sorted(verdicts.items())
    ]
    if not rows:
        return 0

    def write(tx: Any) -> int:
        result = tx.run(_WRITE, rows=rows).data()
        return int(result[0]["written"]) if result else 0

    with driver.session(**({"database": database} if database else {})) as session:
        written: int = session.execute_write(write)
    return written


def _name(record: Mapping[str, Any], end: str) -> str:
    """A node's name: openodke's label on a node the sink wrote, else the named property.

    The sink may project a literal predicate called `name` onto a node, so on
    its own nodes the label wins; on anyone else's, the property the caller named.
    """
    order = (
        ("label", "name") if ENTITY_LABEL in (record[f"{end}_labels"] or ()) else ("name", "label")
    )
    for field in (*order, "key", "id"):
        value = record.get(f"{end}_{field}")
        if value not in (None, ""):
            return str(value)
    return ""


def _own(key: str) -> bool:
    """A property openodke writes on a relationship that is not part of the claim."""
    return key in _PROVENANCE or key.startswith("odke_")


def _type(labels: Iterable[str] | None) -> str | None:
    """A node's type: its label, other than the one the sink adds to every entity."""
    found = sorted(set(labels or ()) - {ENTITY_LABEL, CLAIM_LABEL})
    return found[0] if found else None


def _receipts(properties: Mapping[str, Any]) -> dict[str, Any]:
    """What the sink kept of the claim itself: polarity, qualifiers, extractor, confidence."""
    qualifiers: dict[str, Any] = {}
    for key, value in properties.items():
        if _own(key):
            continue
        original = key.removeprefix("qualifier_")
        if original != key and original not in _PROVENANCE:
            original = key
        qualifiers[original] = _stamp(original, _plain(value))
    out: dict[str, Any] = {"qualifiers": qualifiers}
    for field in ("polarity", "extractor", "confidence"):
        if properties.get(field) is not None:
            out[field] = properties[field]
    return out


def _evidence(properties: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One citation per piece of evidence the sink listed: a document, offsets if cited."""
    doc_ids = list(properties["evidence_doc_ids"])
    count = len(doc_ids)

    def column(name: str, default: Any) -> list[Any]:
        values = list(properties.get(name) or ())
        return (values + [default] * count)[:count]

    uris, starts, ends = (
        column("evidence_uris", ""),
        column("evidence_starts", -1),
        column("evidence_ends", -1),
    )
    # A graph written before the sink recorded who chose a span has only citations.
    origins = column("evidence_span_origins", SpanOrigin.CITED.value)
    out = []
    for doc_id, uri, start, end, origin in zip(doc_ids, uris, starts, ends, origins, strict=True):
        cited: dict[str, Any] = {"doc": str(doc_id), "uri": uri}
        # A context span is the whole text again on the way back in, and a
        # row cannot say openodke located a span, so only a citation keeps offsets.
        offsets = start is not None and end is not None and 0 <= start <= end
        if origin == SpanOrigin.CITED.value and offsets:
            cited.update(start=int(start), end=int(end))
        out.append(cited)
    return out


def _stamp(key: str, value: Any) -> Any:
    """An `odke.` stamp the sink stored as JSON text, as the mapping it was."""
    if key in _STAMPS and isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return loaded if isinstance(loaded, dict) else value
    return value


def _plain(value: Any) -> Any:
    """A driver value as a row can hold it: temporal values as ISO text."""
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        value = to_native()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


__all__ = ["read_neo4j", "write_verdicts"]
