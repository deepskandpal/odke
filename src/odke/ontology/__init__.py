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

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from odke.ontology.from_models import ontology_from_models
from odke.ontology.load import OntologyLoadError, Source, load_dict, load_json, load_yaml

Cardinality = Literal["single", "multi"]


class Qualifier(BaseModel):
    """What one qualifier key means for a fact's identity.

    Two kinds share the `qualifiers` bucket. A *reconcilable* qualifier
    (`start_time`, `end_time`, `rank`) is one claim seen imprecisely — "CEO since
    2019" and "CEO 2019-2024" — and the corroborator merges them. An
    *identity-bearing* qualifier (`percentile`, `tier`, `region`) makes two facts
    different claims — uptime at p50 is not uptime at p95 — and merging them is
    data loss. Only the ontology author knows which is which, so it is declared
    here and stamped onto `Fact.identity_keys` at extraction.
    """

    model_config = ConfigDict(extra="forbid")

    identity: bool = False
    description: str | None = None


class Predicate(BaseModel):
    """One relation or attribute, with the metadata a prompt needs to use it.

    `cardinality` says how many values a subject may hold; `cardinality_scope`
    says *within what*. "One price per subject" and "one price per subject per
    tier" are both `single`, and a flat count treats the second as a stream of
    contradictions — two in five facts in a real corpus are qualifier-scoped, so
    that queue would be mostly noise.

    The scope is a list of qualifier keys, and the only keys that make sense in
    it are the identity-bearing ones: a fact that differs on `tier` is a
    different claim (DECISIONS #15), so uniqueness has to be counted per tier or
    the store sees a conflict where there is none. That makes the default
    obvious — `None` means "every identity-bearing qualifier", and it is right
    for every predicate without a second declaration. An explicit list exists
    for the two other cases: `()` to count per subject regardless of qualifiers,
    and a spelled-out key list so the constraint is visible in the schema. A
    list that names a reconcilable qualifier is a mistake `validate()` reports
    rather than a scope that quietly does nothing. `scope_keys` resolves all
    three to the tuple M4 compiles into a composite uniqueness constraint.
    """

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
    cardinality_scope: tuple[str, ...] | None = None
    # Every entity in the domain is expected to hold a value. False by default:
    # a hand-written ontology says nothing about it, and a pydantic model says
    # it with a non-Optional field that has no default.
    required: bool = False
    qualifiers: dict[str, Qualifier] = Field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    # Drives snippet ranking. In the paper this comes from frequency in the
    # existing KG; when there is no KG yet, the ontology author sets it, and an
    # inferred ontology sets it from corpus support.
    importance: float = 0.5
    examples: tuple[str, ...] = ()

    @field_validator("qualifiers", mode="before")
    @classmethod
    def _names_are_reconcilable(cls, value: Any) -> Any:
        # A bare list of names is the older, shorter spelling; every name in it
        # is reconcilable, which keeps DECISIONS #11 the default.
        if isinstance(value, list | tuple):
            return dict.fromkeys(value, {})
        return value

    @property
    def identity_keys(self) -> tuple[str, ...]:
        """The qualifier keys that make two facts on this predicate different claims."""
        return tuple(sorted(k for k, q in self.qualifiers.items() if q.identity))

    @property
    def scope_keys(self) -> tuple[str, ...]:
        """The qualifier keys a `single` value is unique within, next to the subject.

        Resolved rather than declared, so the constraint compiler reads one
        thing: `(subject, *scope_keys)` is the composite uniqueness key.
        """
        if self.cardinality_scope is None:
            return self.identity_keys
        return tuple(sorted(self.cardinality_scope))

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

    @model_validator(mode="before")
    @classmethod
    def _names_default_to_keys(cls, value: Any) -> Any:
        # `Person: {name: Person, ...}` says the name twice, and the second copy
        # is where key/name mismatches come from. The key wins when the entry
        # has no name of its own.
        if not isinstance(value, Mapping):
            return value
        out = dict(value)
        for section in ("types", "predicates"):
            entries = out.get(section)
            if isinstance(entries, Mapping):
                out[section] = {
                    key: {"name": key, **entry}
                    if isinstance(entry, Mapping) and "name" not in entry
                    else entry
                    for key, entry in entries.items()
                }
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Ontology:
        """An ontology from an already-parsed mapping, with load-time errors explained.

        Each problem is one line naming the offending key by its dotted path
        (`predicates.employer.range`) and what was found there; a misspelt key
        gets a suggestion. That message, not a pydantic traceback, is what a
        user has to act on.
        """
        return load_dict(cls, data)

    @classmethod
    def from_json(cls, source: Source) -> Ontology:
        """An ontology from a JSON file path or a JSON string.

        A `str` that looks like a document (a newline, or an opening brace) is
        parsed as one; any other string is a path. Syntax errors carry a line
        and column; everything else is explained as `from_dict` explains it.
        """
        return load_json(cls, source)

    @classmethod
    def from_yaml(cls, source: Source) -> Ontology:
        """An ontology from a YAML file path or a YAML string.

        PyYAML is imported here and only here, behind the `yaml` extra, so the
        base install stays pydantic and typer (DECISIONS #1).
        """
        return load_yaml(cls, source)

    @classmethod
    def from_pydantic(
        cls, *models: type[BaseModel], name: str = "untitled", version: str = "0"
    ) -> Ontology:
        """An ontology from the pydantic models a Python-first user already has.

        Each model is an entity type and each field a predicate; a field typed as
        another model passed here is an edge to it. A field whose annotation has
        no ontology range is reported by `Model.field` and the whole conversion
        raises `OntologyLoadError` — dropping it would leave a predicate the
        extractor is never asked about, and nobody would notice.
        """
        return ontology_from_models(cls, models, name=name, version=version)

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

    def identity_keys(self, predicate: str) -> tuple[str, ...]:
        """What to stamp onto `Fact.identity_keys` for a fact on this predicate.

        The fact carries the keys rather than a reference to the ontology so that
        `Fact.signature` stays a pure property and a serialised fact still means
        the same thing after the schema it came from has moved on.
        """
        found = self.predicates.get(predicate)
        return found.identity_keys if found else ()

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

__all__ = [
    "Cardinality",
    "EntityType",
    "Ontology",
    "OntologyLoadError",
    "OntologySnippet",
    "Predicate",
    "Qualifier",
]
