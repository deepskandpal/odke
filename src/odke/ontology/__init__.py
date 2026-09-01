"""Ontologies, and the snippets the extractor is actually prompted with.

The ODKE+ paper's central move is that the model is never shown the whole
schema. For each entity type it is shown a small, ranked, textualised fragment —
an *ontology snippet* — listing only the predicates that matter for that type,
each with its label, description, expected range and qualifiers. That is what
keeps extraction schema-aligned across 195 predicates without the prompt growing
without bound, and it is why this module exists separately from the extractor.

Everything here is deterministic. Inferring an ontology from a corpus (M5) is a
separate, model-backed path that produces one of these objects and then hands
over to exactly the same code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Cardinality = Literal["single", "multi"]


class Predicate(BaseModel):
    """One relation or attribute, with the metadata a prompt needs to use it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str | None = None
    description: str | None = None
    # A predicate is an edge when its range names an entity type, and a property
    # when its range is a literal type. One field, because the distinction is
    # exactly "does this string appear in Ontology.types".
    domain: tuple[str, ...] = ()
    range: str = "string"
    cardinality: Cardinality = "single"
    qualifiers: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    # Drives snippet ranking. In the paper this comes from frequency in the
    # existing KG; when there is no KG yet, the ontology author sets it, and an
    # inferred ontology sets it from corpus support.
    importance: float = 0.5
    examples: tuple[str, ...] = ()

    def is_edge_in(self, ontology: Ontology) -> bool:
        return self.range in ontology.types


class EntityType(BaseModel):
    """A node label, and which predicates are worth asking about for it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parents: tuple[str, ...] = ()
    # Identity: which predicates, together, name this thing. Used by the
    # corroborator to decide two mentions are one entity.
    keys: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


class Ontology(BaseModel):
    """A schema the extractor is held to.

    Supplied by the caller (JSON/YAML, OWL, SHACL, a Neo4j schema, Pydantic
    models) or inferred from the corpus. Either way, downstream stages see this
    one shape.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "untitled"
    version: str = "0"
    types: dict[str, EntityType] = Field(default_factory=dict)
    predicates: dict[str, Predicate] = Field(default_factory=dict)
    # Set when the ontology came out of the inference path rather than from the
    # caller, so a sink can refuse to write an unreviewed schema into production.
    inferred: bool = False

    def predicates_for(self, type_name: str) -> list[Predicate]:
        """Predicates whose domain covers this type, including inherited ones.

        A predicate with an empty domain is open — it applies everywhere. That is
        the useful default for an inferred ontology, where domains are the least
        reliable thing a model produces.
        """
        lineage = self.lineage(type_name)
        return [
            p
            for p in self.predicates.values()
            if not p.domain or any(d in lineage for d in p.domain)
        ]

    def lineage(self, type_name: str) -> set[str]:
        """A type and all of its ancestors, cycle-safe."""
        seen: set[str] = set()
        stack = [type_name]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            node = self.types.get(current)
            if node:
                stack.extend(node.parents)
        return seen

    def snippet(self, type_name: str, *, limit: int = 25) -> OntologySnippet:
        """The textualised schema fragment for one entity type.

        `limit` is the knob the paper's scaling story rests on: the prompt stays
        a fixed size as the ontology grows, and the predicates that fall off the
        end are the ones the ranking says matter least for this type.
        """
        ranked = sorted(
            self.predicates_for(type_name),
            key=lambda p: (-p.importance, p.name),
        )[:limit]
        return OntologySnippet(
            type_name=type_name,
            type_description=(self.types.get(type_name) or EntityType(name=type_name)).description,
            predicates=tuple(ranked),
            ontology_name=self.name,
            ontology_version=self.version,
            truncated=len(self.predicates_for(type_name)) > limit,
        )


class OntologySnippet(BaseModel):
    """What actually reaches the model, and how it renders.

    Held as data rather than as a formatted string so the same snippet can be
    rendered as prose for a chat prompt or as a JSON Schema for a structured
    -output call, without the two drifting apart.
    """

    model_config = ConfigDict(extra="forbid")

    type_name: str
    type_description: str | None = None
    predicates: tuple[Predicate, ...] = ()
    ontology_name: str = "untitled"
    ontology_version: str = "0"
    truncated: bool = False

    def render(self) -> str:
        """A compact, stable textual schema. Stable order matters: an unstable
        prompt defeats provider-side prompt caching and makes runs unrepeatable.
        """
        lines = [f"Entity type: {self.type_name}"]
        if self.type_description:
            lines.append(f"Description: {self.type_description}")
        lines.append("Properties you may extract:")
        for p in self.predicates:
            bits = [f"- {p.name} ({p.range}, {p.cardinality})"]
            if p.description:
                bits.append(f": {p.description}")
            if p.qualifiers:
                bits.append(f" [qualifiers: {', '.join(p.qualifiers)}]")
            if p.examples:
                bits.append(f" e.g. {'; '.join(p.examples)}")
            lines.append("".join(bits))
        return "\n".join(lines)

    def json_schema(self) -> dict[str, Any]:
        """The same snippet as a JSON Schema, for providers with structured output."""
        props: dict[str, Any] = {}
        for p in self.predicates:
            base: dict[str, Any] = {"type": _JSON_TYPES.get(p.range, "string")}
            if p.description:
                base["description"] = p.description
            props[p.name] = {"type": "array", "items": base} if p.cardinality == "multi" else base
        return {
            "type": "object",
            "title": self.type_name,
            "properties": props,
            "additionalProperties": False,
        }


_JSON_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "float": "number",
    "boolean": "boolean",
    "date": "string",
    "datetime": "string",
}

__all__ = ["Cardinality", "EntityType", "Ontology", "OntologySnippet", "Predicate"]
