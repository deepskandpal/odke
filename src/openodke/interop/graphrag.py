"""neo4j-graphrag's graphs as triples rows, with its chunks as evidence (#111).

Provisional (DECISIONS #48): it reads neo4j-graphrag's objects, whose shape
changes with that library. It may change in a minor release, with a CHANGELOG
line.

neo4j-graphrag's extractor reads one chunk at a time, and `SimpleKGPipeline`
records which. Beside the entities it writes a lexical graph: a `Chunk` node
holding each chunk's `text`, `FROM_DOCUMENT` from chunk to document, and
`FROM_CHUNK` from every entity to the chunk it came from. So a relationship
whose two ends share one chunk is grounded against that chunk, which is what
the extractor read, rather than against the whole document. When the ends share
no single chunk of a document (resolution merged an entity found in several),
the row falls back to that document: the text the caller gives, or its chunks
in order. Either way the span is the whole text, marked `SpanOrigin.CONTEXT`,
because neo4j-graphrag cites nothing finer (DECISIONS #25).

There are two sources. One is the `Neo4jGraph` the extractor returns, or its
JSON form. The other is a store it has already written, read through the
caller's driver with the names of its `LexicalGraphConfig`. Neither imports
neo4j-graphrag, and the store is read, never written.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openodke.interop.triples import TripleRow
from openodke.sinks.neo4j import _ident
from openodke.types import Document

# `LexicalGraphConfig`'s defaults, by its field names. A config object or a
# mapping with any of these names overrides them.
LEXICAL_DEFAULTS: Mapping[str, str] = {
    "document_node_label": "Document",
    "chunk_node_label": "Chunk",
    "chunk_to_document_relationship_type": "FROM_DOCUMENT",
    "next_chunk_relationship_type": "NEXT_CHUNK",
    "node_to_chunk_relationship_type": "FROM_CHUNK",
    "chunk_index_property": "index",
    "chunk_text_property": "text",
}
# The label neo4j-graphrag's writer adds to every extracted node in a store.
ENTITY_LABEL = "__Entity__"
# Labels its writer adds that are not the node's type.
_WRITER_LABELS = frozenset({"__Entity__", "__KGBuilder__"})


@dataclass
class _Chunk:
    index: int
    text: str
    document: str | None


@dataclass
class _Lexical:
    """Either source, read into one shape: entities, relationships, and where each came from."""

    types: dict[str, str] = field(default_factory=dict)
    properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    relationships: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)
    chunks: dict[str, _Chunk] = field(default_factory=dict)
    paths: dict[str, str | None] = field(default_factory=dict)
    sources: dict[str, set[str]] = field(default_factory=dict)


def from_graphrag(
    graph: str | Path | Any,
    *,
    document: Document | None = None,
    config: Any = None,
    name_property: str = "name",
) -> tuple[list[TripleRow], list[Document]]:
    """Rows for a `Neo4jGraph` from neo4j-graphrag's extractor, and the texts they cite.

    `graph` is the `Neo4jGraph`, its `model_dump()` dict, or a JSON file of it.
    Its lexical graph, when built (`create_lexical_graph=True`, the default),
    supplies the chunks. `document` is the text the graph was extracted from,
    for a row its chunks cannot place and for a graph built without them.
    """
    if isinstance(graph, str | Path):
        graph = json.loads(Path(graph).read_text(encoding="utf-8"))
    names = _names(config)
    lexical = _Lexical()
    for node in _get(graph, "nodes") or ():
        node_id, label = str(_get(node, "id")), str(_get(node, "label"))
        properties = dict(_get(node, "properties") or {})
        if label == names["chunk_node_label"]:
            index = properties.get(names["chunk_index_property"], -1)
            text = properties.get(names["chunk_text_property"]) or ""
            lexical.chunks[node_id] = _Chunk(int(index), str(text), None)
        elif label == names["document_node_label"]:
            lexical.paths[node_id] = properties.get("path")
        else:
            lexical.types[node_id] = label
            lexical.properties[node_id] = properties
    from_chunk = names["node_to_chunk_relationship_type"]
    for rel in _get(graph, "relationships") or ():
        start, end = str(_get(rel, "start_node_id")), str(_get(rel, "end_node_id"))
        kind = str(_get(rel, "type"))
        if kind == names["chunk_to_document_relationship_type"] and start in lexical.chunks:
            lexical.chunks[start].document = end
        elif kind == from_chunk and end in lexical.chunks:
            lexical.sources.setdefault(start, set()).add(end)
        elif kind not in {from_chunk, names["next_chunk_relationship_type"]} and not (
            {start, end} & set(lexical.chunks) or {start, end} & set(lexical.paths)
        ):
            lexical.relationships.append((start, end, kind, dict(_get(rel, "properties") or {})))
    if not lexical.chunks and document is None and lexical.relationships:
        raise ValueError(
            f"the graph has no {names['chunk_node_label']} nodes to ground against: build it "
            "with create_lexical_graph=True, or pass document= the text it was extracted from"
        )
    return _rows(lexical, document=document, name_property=name_property)


def read_graphrag(
    driver: Any,
    *,
    database: str | None = None,
    config: Any = None,
    name_property: str = "name",
    entity_label: str = ENTITY_LABEL,
) -> tuple[list[TripleRow], list[Document]]:
    """Rows for the graph neo4j-graphrag wrote to a store, and the chunks they cite.

    Reads, in one read transaction, every node labelled `entity_label`, the
    relationships between them, and the chunks they link to by `FROM_CHUNK`,
    under the names in `config` (a `LexicalGraphConfig`, or a mapping of its
    field names). `driver` is a `neo4j` driver; nothing is written.
    """
    names = _names(config)
    chunk, document = _ident(names["chunk_node_label"]), _ident(names["document_node_label"])
    entity = _ident(entity_label)
    queries = {
        "chunks": (
            f"MATCH (c:{chunk})\n"
            f"OPTIONAL MATCH (c)-[:{_ident(names['chunk_to_document_relationship_type'])}]"
            f"->(d:{document})\n"
            "RETURN elementId(c) AS id, c[$index] AS index, c[$text] AS text,\n"
            "       elementId(d) AS document, d.path AS path"
        ),
        "entities": (
            f"MATCH (n:{entity})\n"
            f"OPTIONAL MATCH (n)-[:{_ident(names['node_to_chunk_relationship_type'])}]"
            f"->(c:{chunk})\n"
            "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties,\n"
            "       collect(elementId(c)) AS chunks"
        ),
        "relationships": (
            f"MATCH (s:{entity})-[r]->(o:{entity})\n"
            "RETURN elementId(s) AS start, elementId(o) AS end, type(r) AS type,\n"
            "       properties(r) AS properties\n"
            "ORDER BY start, end, type"
        ),
    }
    parameters = {"index": names["chunk_index_property"], "text": names["chunk_text_property"]}

    def read(tx: Any) -> dict[str, list[dict[str, Any]]]:
        return {name: list(tx.run(cypher, **parameters).data()) for name, cypher in queries.items()}

    with driver.session(**({"database": database} if database else {})) as session:
        found = session.execute_read(read)
    lexical = _Lexical()
    for row in found["chunks"]:
        index, text = row.get("index"), str(row.get("text") or "")
        position = -1 if index is None else int(index)
        lexical.chunks.setdefault(row["id"], _Chunk(position, text, row["document"]))
        if row["document"] is not None:
            lexical.paths.setdefault(row["document"], row.get("path"))
    for row in found["entities"]:
        types = sorted(set(row["labels"]) - _WRITER_LABELS - {entity_label})
        lexical.types[row["id"]] = types[0] if types else entity_label
        lexical.properties[row["id"]] = {
            k: v for k, v in (row["properties"] or {}).items() if not k.startswith("__")
        }
        lexical.sources[row["id"]] = {c for c in row["chunks"] if c in lexical.chunks}
    for row in found["relationships"]:
        properties = {
            k: _native(v) for k, v in (row["properties"] or {}).items() if not k.startswith("__")
        }
        lexical.relationships.append((row["start"], row["end"], row["type"], properties))
    return _rows(lexical, document=None, name_property=name_property)


def _rows(
    lexical: _Lexical, *, document: Document | None, name_property: str
) -> tuple[list[TripleRow], list[Document]]:
    texts: dict[str, Document] = {}
    rows: list[TripleRow] = []
    unplaced = dangling = 0

    def name(node: str) -> str:
        return str(lexical.properties[node].get(name_property) or node)

    def grounds(*ends: str) -> list[str]:
        """The texts a claim about `ends` is grounded against: one per document they share."""
        held = [lexical.sources.get(end, set()) for end in ends]
        documents = set.intersection(*({lexical.chunks[c].document for c in h} for h in held))
        shared = set.intersection(*held)
        out = []
        for key in sorted(documents, key=str):
            here = sorted(c for c in shared if lexical.chunks[c].document == key)
            text = one_chunk(here[0]) if len(here) == 1 else whole(key)
            out.append(texts.setdefault(text.id, text).id)
        if not out and document is not None:
            out.append(texts.setdefault(document.id, document).id)
        return out

    def one_chunk(chunk_id: str) -> Document:
        chunk = lexical.chunks[chunk_id]
        path = lexical.paths.get(chunk.document) if chunk.document else None
        meta = {"chunk_index": chunk.index, "document": chunk.document}
        return Document(id=chunk_id, text=chunk.text, uri=path, metadata=meta)

    def whole(key: str | None) -> Document:
        if document is not None:
            return document
        ordered = sorted(
            (c for c in lexical.chunks.values() if c.document == key), key=lambda c: c.index
        )
        doc_id = key if key is not None else "graphrag-document"
        uri = lexical.paths.get(key) if key is not None else None
        return Document(id=doc_id, text="\n".join(c.text for c in ordered), uri=uri)

    for start, end, kind, properties in lexical.relationships:
        if start not in lexical.types or end not in lexical.types:
            dangling += 1
            continue
        placed = grounds(start, end)
        unplaced += not placed
        for doc_id in placed:
            rows.append(
                TripleRow(
                    doc=doc_id,
                    subject=name(start),
                    subject_type=lexical.types[start],
                    predicate=kind,
                    object=name(end),
                    object_type=lexical.types[end],
                    qualifiers={k: _native(v) for k, v in properties.items()},
                )
            )
    for node, properties in lexical.properties.items():
        literals = [
            (key, item)
            for key, value in properties.items()
            if key != name_property
            for item in (value if isinstance(value, list) else [value])
            if item is not None and item != ""
        ]
        placed = grounds(node) if literals else []
        unplaced += bool(literals) and not placed
        for doc_id in placed:
            for key, item in literals:
                literal, literal_type = _literal(item)
                rows.append(
                    TripleRow(
                        doc=doc_id,
                        subject=name(node),
                        subject_type=lexical.types[node],
                        predicate=key,
                        object=literal,
                        object_type=literal_type,
                    )
                )
    if unplaced or dangling:
        warnings.warn(
            f"{unplaced} relationships or entities have no chunk or document to ground "
            f"against, and {dangling} relationships end at a node the graph does not have; "
            "both were left out",
            stacklevel=3,
        )
    return rows, list(texts.values())


def _names(config: Any) -> dict[str, str]:
    names = dict(LEXICAL_DEFAULTS)
    for key in names:
        value = _get(config, key) if config is not None else None
        if value:
            names[key] = str(value)
    return names


def _get(obj: Any, name: str) -> Any:
    """A field of a neo4j-graphrag object, or of its JSON form."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _native(value: Any) -> Any:
    """A driver's temporal value as Python's; anything else as it is."""
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


def _literal(value: Any) -> tuple[str | int | float | bool, str | None]:
    """A property value as a row's object, and the literal type that keeps it a value."""
    value = _native(value)
    if isinstance(value, bool | int | float):
        return value, None
    if isinstance(value, datetime):
        return value.isoformat(), "datetime"
    if isinstance(value, date):
        return value.isoformat(), "date"
    return (value if isinstance(value, str) else str(value)), "string"


__all__ = ["ENTITY_LABEL", "LEXICAL_DEFAULTS", "from_graphrag", "read_graphrag"]
