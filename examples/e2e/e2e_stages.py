"""The example's own stage: registration numbers become external ids, then odke resolves.

odke's extractors key an entity by its type and name and never guess an
identifier (DECISIONS #18). This corpus has one — every company in the register
carries its company number — and a stage of your own is how a fact about one
corpus enters a run without entering the package (DECISIONS #20). The config
names it as `e2e_stages:RegistryResolver`, with this directory on `pythonpath`.
"""

from __future__ import annotations

from collections.abc import Iterable

from odke import Entity, EntityLink, Fact, NativeResolver
from odke.stages import EntityIndex


class RegistryResolver:
    """Stamps each company's registration number on as its `external_id`, then resolves.

    With the numbers in place, two companies that share a name and not a
    number become a `DIFFERENT` link naming both numbers, rather than a
    `SIMILAR` one somebody would have to look into. Everything else is
    `NativeResolver`'s, unchanged.
    """

    def __init__(self, predicate: str = "registration_number", threshold: float = 0.9) -> None:
        self.predicate = predicate
        self.inner = NativeResolver(threshold=threshold)

    def resolve(
        self, facts: Iterable[Fact], index: EntityIndex
    ) -> tuple[list[Fact], list[EntityLink]]:
        facts = list(facts)
        numbers = {
            f.subject.key: str(f.object_value)
            for f in facts
            if f.predicate == self.predicate and f.object_value is not None
        }

        def stamp(entity: Entity) -> Entity:
            number = numbers.get(entity.key)
            if number is None or entity.external_id is not None:
                return entity
            return entity.model_copy(update={"external_id": number})

        stamped = [
            f.model_copy(
                update={
                    "subject": stamp(f.subject),
                    "object_entity": stamp(f.object_entity) if f.object_entity else None,
                }
            )
            for f in facts
        ]
        return self.inner.resolve(stamped, {key: stamp(e) for key, e in index.items()})
