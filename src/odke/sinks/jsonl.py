"""The dependency-free sink: newline-delimited JSON.

Ships in the base install so that `odke` produces something usable before any
driver is present, and so the test suite has a sink to exercise the protocol
against without a database.
"""

from __future__ import annotations

import json
from pathlib import Path

from odke.types import KnowledgeGraph


class JsonlSink:
    """Writes entities and facts as two JSONL streams under one directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def write(self, kg: KnowledgeGraph) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "entities.jsonl").open("w", encoding="utf-8") as fh:
            for entity in kg.entities:
                fh.write(entity.model_dump_json() + "\n")
        with (self.directory / "facts.jsonl").open("w", encoding="utf-8") as fh:
            for fact in kg.facts:
                fh.write(fact.model_dump_json() + "\n")
        (self.directory / "manifest.json").write_text(
            json.dumps(
                {
                    "ontology": kg.ontology_name,
                    "created_at": kg.created_at.isoformat(),
                    "entities": len(kg.entities),
                    "facts": len(kg.facts),
                    "edges": len(kg.edges),
                    "properties": len(kg.properties),
                    "stats": kg.stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
