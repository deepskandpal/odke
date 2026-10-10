"""Looking the store up instead of loading it: the in-memory `StoreLookup`.

`NativeResolver(lookup=...)` resolves a batch against what a store already
holds by asking it for candidates, entity by entity, on the keys blocking
already uses (DECISIONS #31). `MemoryLookup` answers from entities already in
memory, today's `EntityIndex` mapping: the default for a caller with no
database, and the stand-in the benchmark and the tests use. `Neo4jLookup`, in
`openodke.sinks.neo4j`, answers the same question with index queries.

The optional vector lookup is a slot, not a dependency. Nothing here embeds
anything: a caller who passes `embed`, a function from texts to vectors, gets
the store's nearest entities of the same type as candidates too. A candidate
found that way is judged by the same rules as any other; an embedding widens
which entities are compared, never what counts as a match.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from openodke.corroborate.resolve import Embed, _profile, _unit, block_keys, name_similarity
from openodke.stages import EntityIndex
from openodke.types import Entity


def embedding_text(entity: Entity) -> str:
    """What a vector lookup embeds for an entity: its label, or its key when it has none."""
    return entity.label or entity.key


def in_tenant(entity: Entity, tenant: str | None, prop: str) -> bool:
    """Whether a lookup scoped to `tenant` may return `entity`: always, when unscoped."""
    return tenant is None or entity.attributes.get(prop) == tenant


class MemoryLookup:
    """A `StoreLookup` over entities already in memory, by the resolver's own block keys.

    `index` is the store: a mapping of key to entity, as `EntityIndex` has
    always been. For each entity asked about, the candidates are the stored
    entity with its key, and every stored entity of its type sharing an
    external id, a domain, or the first or last token of a name with it.

    `tenant` scopes the store to the entities whose `attributes[tenant_property]`
    is `tenant`. Tenant keys arrive with #159; until then the tenant is a
    property like any other, and an entity without it is in no tenant.

    `limit` caps the entities one name token returns. A token shared by more
    than that, a first word like "bank", returns the `limit` whose names are
    nearest the one asked about. With `embed`, the `vector_k` stored entities
    of the same type whose label embeds nearest are candidates as well; every
    stored label is embedded once, on the first call.
    """

    def __init__(
        self,
        index: EntityIndex,
        *,
        tenant: str | None = None,
        tenant_property: str = "tenant",
        limit: int = 100,
        embed: Embed | None = None,
        vector_k: int = 5,
    ) -> None:
        if limit < 1 or vector_k < 1:
            raise ValueError("limit and vector_k must be at least 1")
        self.tenant = tenant
        self.tenant_property = tenant_property
        self.limit = limit
        self.embed = embed
        self.vector_k = vector_k
        self._entities = {e.key: e for e in index.values() if in_tenant(e, tenant, tenant_property)}
        self._blocks: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for key in sorted(self._entities):
            entity = self._entities[key]
            keys = block_keys(entity)
            for token in keys.tokens:
                self._blocks[("name", entity.type, token)].append(key)
            for found in keys.domains:
                self._blocks[("domain", entity.type, found)].append(key)
            for scheme, value in keys.ids:
                self._blocks[("id", entity.type, scheme, value)].append(key)
        self._vectors: list[tuple[Entity, list[float]]] | None = None

    def candidates(self, entities: Sequence[Entity]) -> Mapping[str, Sequence[Entity]]:
        entities = list(entities)
        nearest = self._nearest_vectors(entities) if self.embed is not None else {}
        out: dict[str, list[Entity]] = {}
        for entity in entities:
            found: dict[str, Entity] = {}
            same = self._entities.get(entity.key)
            if same is not None and same.type == entity.type:
                found[same.key] = same
            keys = block_keys(entity)
            for scheme, value in sorted(keys.ids):
                self._add(found, self._blocks.get(("id", entity.type, scheme, value), ()))
            for domain in sorted(keys.domains):
                self._add(found, self._blocks.get(("domain", entity.type, domain), ()))
            for token in sorted(keys.tokens):
                members = self._blocks.get(("name", entity.type, token), [])
                if len(members) > self.limit:
                    members = self._nearest_names(entity, members)
                self._add(found, members)
            self._add(found, nearest.get(entity.key, ()))
            out[entity.key] = list(found.values())
        return out

    def _add(self, found: dict[str, Entity], keys: Sequence[str]) -> None:
        for key in keys:
            found.setdefault(key, self._entities[key])

    def _nearest_names(self, entity: Entity, members: Sequence[str]) -> list[str]:
        names = _profile(entity).names

        def similarity(key: str) -> float:
            stored = _profile(self._entities[key]).names
            return max((name_similarity(a, b) for a in names for b in stored), default=0.0)

        return sorted(members, key=lambda k: (-similarity(k), k))[: self.limit]

    def _nearest_vectors(self, entities: Sequence[Entity]) -> dict[str, list[str]]:
        assert self.embed is not None
        if self._vectors is None:
            stored = [self._entities[k] for k in sorted(self._entities)]
            vectors = self.embed([embedding_text(e) for e in stored]) if stored else []
            self._vectors = [(e, _unit(v)) for e, v in zip(stored, vectors, strict=True)]
        if not entities or not self._vectors:
            return {}
        probes = self.embed([embedding_text(e) for e in entities])
        out: dict[str, list[str]] = {}
        for entity, vector in zip(entities, probes, strict=True):
            probe = _unit(vector)
            scored = [
                (-sum(a * b for a, b in zip(probe, other, strict=True)), stored.key)
                for stored, other in self._vectors
                if stored.type == entity.type
            ]
            out[entity.key] = [key for _, key in sorted(scored)[: self.vector_k]]
        return out


__all__ = ["Embed", "MemoryLookup", "embedding_text", "in_tenant"]
