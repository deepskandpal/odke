"""LangChain's `GraphDocument`s as triples rows (#97).

Provisional (DECISIONS #48): it reads LangChain's objects, whose shape changes
with that library. It may change in a minor release, with a CHANGELOG line.

`LLMGraphTransformer` reads a whole `Document` and returns nodes and
relationships with no offsets and no quote: its finest provenance is the
document. So every row it gives is grounded against that document's whole text,
marked `SpanOrigin.CONTEXT` (DECISIONS #25), and `odke eval spans` counts it as
having no span of its own.

Objects are read by attribute and their JSON form (`GraphDocument.model_dump()`)
by key, so neither needs langchain installed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from openodke.interop.triples import TripleRow
from openodke.types import Document

# The type LLMGraphTransformer gives a node the model left untyped. It names no
# type, so the row leaves the type out and the ontology decides.
UNTYPED = "Node"


def from_graph_documents(
    graph_documents: str | Path | Iterable[Any],
) -> tuple[list[TripleRow], list[Document]]:
    """One row per relationship and per node property, and the texts they came from.

    `graph_documents` is `GraphDocument`s, their `model_dump()` dicts, or a JSON
    file of either form's dumps: an array, or one per line. Each text is the
    source `Document`'s `page_content`, with its `id` when it has one and a hash
    of the text when it has not, so the same text is one document.
    """
    rows: list[TripleRow] = []
    texts: dict[str, Document] = {}
    for graph in _read(graph_documents):
        doc = _document(_get(graph, "source"))
        known = texts.setdefault(doc.id, doc)
        if known.text != doc.text:
            raise ValueError(f"two source documents have the id {doc.id!r} and different texts")
        for rel in _get(graph, "relationships") or ():
            source, target = _get(rel, "source"), _get(rel, "target")
            rows.append(
                TripleRow(
                    doc=doc.id,
                    subject=str(_get(source, "id")),
                    subject_type=_type(source),
                    predicate=str(_get(rel, "type")),
                    object=str(_get(target, "id")),
                    object_type=_type(target),
                    qualifiers=dict(_get(rel, "properties") or {}),
                )
            )
        for node in _get(graph, "nodes") or ():
            for key, value in (_get(node, "properties") or {}).items():
                for item in value if isinstance(value, list) else [value]:
                    if item is None or item == "":
                        continue
                    literal = item if isinstance(item, str | int | float | bool) else str(item)
                    rows.append(
                        TripleRow(
                            doc=doc.id,
                            subject=str(_get(node, "id")),
                            subject_type=_type(node),
                            predicate=str(key),
                            object=literal,
                            # A string property is a value, not a node named by it.
                            object_type="string" if isinstance(literal, str) else None,
                        )
                    )
    return rows, list(texts.values())


def _read(source: str | Path | Iterable[Any]) -> Iterable[Any]:
    if not isinstance(source, str | Path):
        return source
    text = Path(source).read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [loaded] if isinstance(loaded, Mapping) else loaded


def _get(obj: Any, name: str) -> Any:
    """A field of a LangChain object, or of its JSON form."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _type(node: Any) -> str | None:
    label = _get(node, "type")
    return None if not label or label == UNTYPED else str(label)


def _document(source: Any) -> Document:
    if source is None:
        raise ValueError("a graph document has no source document, so no text to ground on")
    text = _get(source, "page_content") or ""
    metadata = dict(_get(source, "metadata") or {})
    doc_id = _get(source, "id") or "langchain-" + hashlib.sha256(text.encode()).hexdigest()[:12]
    where = metadata.get("source")
    title = metadata.get("title")
    return Document(
        id=str(doc_id),
        text=text,
        uri=str(where) if where else None,
        title=title if isinstance(title, str) else None,
        metadata=metadata,
    )


__all__ = ["UNTYPED", "from_graph_documents"]
