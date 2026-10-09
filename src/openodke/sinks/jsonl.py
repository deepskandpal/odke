"""The dependency-free sink: newline-delimited JSON.

Ships in the base install so that `openodke` produces something usable before any
driver is present, and so the test suite has a sink to exercise the protocol
against without a database.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from openodke.types import Entity, EntityLink, Fact, KnowledgeGraph

M = TypeVar("M", bound=BaseModel)


class JsonlSink:
    """Writes entities, facts and links as three JSONL streams under one directory.

    By default each `write` replaces the files. With `merge=True` the files are
    a store, and a write merges into them: what they hold stays, an entity, a
    fact or a link the graph restates replaces its line (by type and key, by
    signature, by kind and ends), and the rest is added. `stored()` then says
    which of a batch's facts the file already holds, so the corroborator can
    merge their support lists before the write; the Validator hands it the
    sink to do that (#153). Without `merge`, there is nothing to merge with.
    """

    def __init__(self, directory: str | Path, *, merge: bool = False) -> None:
        self.directory = Path(directory)
        self.merge = merge

    def stored(self, facts: Sequence[Fact]) -> dict[tuple[Any, ...], Fact]:
        """The facts `facts.jsonl` holds under these facts' signatures; none unless merging."""
        if not self.merge:
            return {}
        held = {fact.signature: fact for fact in self._read("facts.jsonl", Fact)}
        return {f.signature: held[f.signature] for f in facts if f.signature in held}

    def write(self, kg: KnowledgeGraph) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entities, facts, links = list(kg.entities), list(kg.facts), list(kg.links)
        if self.merge:
            entities = _upsert(self._read("entities.jsonl", Entity), entities, _entity_key)
            facts = _upsert(self._read("facts.jsonl", Fact), facts, _fact_key)
            links = _upsert(self._read("links.jsonl", EntityLink), links, _link_key)
        self._write("entities.jsonl", entities)
        self._write("facts.jsonl", facts)
        self._write("links.jsonl", links)
        edges = sum(fact.is_edge for fact in facts)
        (self.directory / "manifest.json").write_text(
            json.dumps(
                {
                    "ontology": kg.ontology_name,
                    "created_at": kg.created_at.isoformat(),
                    "entities": len(entities),
                    "facts": len(facts),
                    "edges": edges,
                    "properties": len(facts) - edges,
                    "links": len(links),
                    "stats": kg.stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _read(self, name: str, model: type[M]) -> list[M]:
        path = self.directory / name
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as fh:
            return [model.model_validate_json(line) for line in fh if line.strip()]

    def _write(self, name: str, rows: Iterable[BaseModel]) -> None:
        with (self.directory / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(row.model_dump_json() + "\n")


def _entity_key(entity: Entity) -> Hashable:
    return (entity.type, entity.key)


def _fact_key(fact: Fact) -> Hashable:
    return fact.signature


def _link_key(link: EntityLink) -> Hashable:
    return (link.kind, link.source_key, link.target_key)


def _upsert(held: list[M], incoming: list[M], key: Callable[[M], Hashable]) -> list[M]:
    """`held` with each item `incoming` restates replaced where it stood, and the rest after."""
    merged: dict[Hashable, M] = {key(item): item for item in held}
    for item in incoming:
        merged[key(item)] = item
    return list(merged.values())
