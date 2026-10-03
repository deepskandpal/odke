"""`replay:Replayed` — a competitor's triples, as an openodke extractor stage.

openodke's grounder and corroborator then run on them unchanged. A competitor
cites no clause, so each fact's evidence is its whole document: with the
grounder's `context: document` that is the paper's own setting, the whole page
against one triple.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from openodke import Chunk, Entity, Evidence, Fact, Span
from openodke.ontology import Ontology
from openodke.types import Document


class Replayed:
    name = "replayed"

    def __init__(self, path: str, system: str = "competitor") -> None:
        self.system = system
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in Path(path).read_text().splitlines():
            if line:
                row = json.loads(line)
                self.rows[row["doc"]].append(row)
        # Filled by `register_documents`, so a chunk's doc id finds its file name.
        self.documents: dict[str, Document] = {}

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        doc = self.documents.get(chunk.doc_id)
        name = Path(unquote(urlparse(doc.uri).path)).stem if doc and doc.uri else None
        text = doc.text if doc is not None else chunk.text
        evidence = (
            Evidence(
                doc_id=chunk.doc_id,
                span=Span(doc_id=chunk.doc_id, start=0, end=len(text), quote=text),
            ),
        )
        canonical = {t.lower(): t for t in ontology.types}
        facts = []
        for row in self.rows.get(name or "", []):
            predicate = ontology.predicates.get(row["predicate"].lower())
            if predicate is None:
                continue
            head_type = canonical.get(row["subject_type"].lower(), row["subject_type"])
            subject = Entity(key=row["subject"].casefold(), type=head_type, label=row["subject"])
            if predicate.range in ontology.types:
                obj = {
                    "object_entity": Entity(
                        key=row["object"].casefold(), type=predicate.range, label=row["object"]
                    )
                }
            else:
                obj = {"object_value": row["object"]}
            facts.append(
                Fact(
                    subject=subject,
                    predicate=predicate.name,
                    evidence=evidence,
                    extractor=self.system,
                    confidence=0.5,
                    **obj,
                )
            )
        return facts
