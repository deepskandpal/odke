"""The dependency-free sink: newline-delimited JSON.

Ships in the base install so that `openodke` produces something usable before any
driver is present, and so the test suite has a sink to exercise the protocol
against without a database.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from openodke.reconcile import reconciled, tally
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

    `retract(doc_ids, at=...)` reconciles the files with a source that changed
    or went: `facts.jsonl` is read, every fact citing the documents is
    retracted and retired when left with none, and the file is written back
    (#116).

    `append(kg)` adds a micro-batch to what the files hold (#158): a streamed
    run writes its first micro-batch and appends the rest.
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
        self._manifest(manifest_of(kg, entities, facts, links))

    def append(self, kg: KnowledgeGraph) -> None:
        """One micro-batch more: its lines after the files' own, and its counts on the manifest's.

        Nothing is read back, so an entity, fact or link two micro-batches both
        state is a line in each, and a reader keeps the last, as `merge` does.
        The manifest's counts are the files' lines, and its `stats` are the
        graph's: a streamed run hands each micro-batch the run's so far. A
        sink that merges merges, as `write` does.
        """
        if self.merge:
            self.write(kg)
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write("entities.jsonl", kg.entities, mode="a")
        self._write("facts.jsonl", kg.facts, mode="a")
        self._write("links.jsonl", kg.links, mode="a")
        path = self.directory / "manifest.json"
        held: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )
        added = manifest_of(kg, kg.entities, kg.facts, kg.links)
        for key in ("entities", "facts", "edges", "properties", "links"):
            added[key] += int(held.get(key, 0))
        added["ontology"] = held.get("ontology", added["ontology"])
        added["created_at"] = held.get("created_at", added["created_at"])
        self._manifest(added)

    def _manifest(self, manifest: dict[str, Any]) -> None:
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def retract(
        self, doc_ids: Collection[str], *, at: datetime, hard: bool = False
    ) -> dict[str, int]:
        """Take these documents out of every fact in `facts.jsonl` that cites them.

        As `openodke.reconcile.retract` does it: a fact left with no source is
        retired at `at` and kept, or dropped with `hard`, and a derived fact is
        retired with its parent. The manifest's counts follow. Retracting the
        same documents again leaves the files as they are.
        """
        before = self._read("facts.jsonl", Fact)
        after = reconciled(before, set(doc_ids), at)
        counts = tally(before, after, hard=hard)
        if not counts["cited"]:
            return counts
        gone = [
            hard and new is not old and new.retired_at is not None
            for old, new in zip(before, after, strict=True)
        ]
        kept = [fact for fact, dropped in zip(after, gone, strict=True) if not dropped]
        self._write("facts.jsonl", kept)
        manifest = self.directory / "manifest.json"
        if manifest.is_file():
            found = json.loads(manifest.read_text(encoding="utf-8"))
            edges = sum(fact.is_edge for fact in kept)
            found.update(facts=len(kept), edges=edges, properties=len(kept) - edges)
            manifest.write_text(json.dumps(found, indent=2), encoding="utf-8")
        return counts

    def _read(self, name: str, model: type[M]) -> list[M]:
        path = self.directory / name
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as fh:
            return [model.model_validate_json(line) for line in fh if line.strip()]

    def _write(self, name: str, rows: Iterable[BaseModel], *, mode: str = "w") -> None:
        with (self.directory / name).open(mode, encoding="utf-8") as fh:
            for row in rows:
                fh.write(row.model_dump_json() + "\n")


def manifest_of(
    kg: KnowledgeGraph,
    entities: Sequence[Entity],
    facts: Sequence[Fact],
    links: Sequence[EntityLink],
) -> dict[str, Any]:
    """What `manifest.json` says of a graph: its ontology, when it was made, its counts, its stats.

    The counts are of what the files hold, which with `merge=True` is more
    than this graph. A run adds its own manifest beside these keys
    (`openodke.manifest`) and never replaces them.
    """
    edges = sum(fact.is_edge for fact in facts)
    return {
        "ontology": kg.ontology_name,
        "created_at": kg.created_at.isoformat(),
        "entities": len(entities),
        "facts": len(facts),
        "edges": edges,
        "properties": len(facts) - edges,
        "links": len(links),
        "stats": kg.stats,
    }


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
