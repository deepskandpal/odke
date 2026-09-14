"""The Neo4j platform sink.

Writes with `MERGE` on entity keys and fact signatures, so a pipeline run twice
updates the graph it wrote rather than duplicating it.

The module never imports the driver at the top. `import odke.sinks.neo4j` works
on the base install (DECISIONS #1); the driver is imported the first time a
connection is needed, behind the `[neo4j]` extra.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from datetime import date, datetime, time
from enum import Enum
from typing import Any, NamedTuple

from odke.ontology import Ontology
from odke.stages import PlatformProfile
from odke.types import Entity, EntityLink, Fact, KnowledgeGraph, Polarity

EXTRA_HINT = "the neo4j driver is not installed; run: pip install 'odke[neo4j]'"

# The two labels the sink owns. `Entity` goes on every node next to its type,
# so a link — which knows only keys — can find both ends without their types.
# `Claim` is the object of a literal-valued fact.
ENTITY_LABEL = "Entity"
CLAIM_LABEL = "Claim"

# Relationship property names provenance owns. A qualifier with one of these
# names lands under a prefix rather than overwriting the receipt.
_PROVENANCE = frozenset(
    {
        "fact_id",
        "signature",
        "polarity",
        "identity_keys",
        "extractor",
        "verdict",
        "confidence",
        "support",
        "valid_from",
        "valid_to",
        "retrieved_at",
        "extracted_at",
        "evidence_doc_ids",
        "evidence_uris",
        "evidence_starts",
        "evidence_ends",
        "evidence_tiers",
        "evidence_retrieved_at",
    }
)
# Node property names the sink writes from `Entity` fields; an attribute or a
# projected predicate with one of these names is prefixed, never merged into.
_ENTITY_FIELDS = frozenset(
    {
        "key",
        "label",
        "aliases",
        "external_id",
        "resolution_method",
        "resolution_score",
        "resolution_linker",
    }
)
_SCALARS = (str, int, float, bool, datetime, date, time)


# --------------------------------------------------------------------------- #
# Names and values
# --------------------------------------------------------------------------- #


def _ident(name: str) -> str:
    """A label, relationship type or property name, quoted so any string is legal.

    Types and predicates come from an ontology, which may be inferred; a name is
    never spliced into Cypher unquoted.
    """
    return "`" + name.replace("`", "``") + "`"


def qualifier_property(key: str) -> str:
    """Where a qualifier lands on a relationship: its own name, unless provenance owns it."""
    return f"qualifier_{key}" if key in _PROVENANCE else key


def attribute_property(key: str) -> str:
    return f"attribute_{key}" if key in _ENTITY_FIELDS else key


def projection_property(predicate: str) -> str:
    """Where a literal fact's value is projected on its subject node."""
    return f"property_{predicate}" if predicate in _ENTITY_FIELDS else predicate


def signature_of(fact: Fact) -> str:
    """`Fact.signature` as one fixed-width string: the MERGE key of every fact.

    Hashed rather than serialised so the key is index-safe whatever the object
    value's repr is; the components are on the relationship as plain properties.
    """
    canonical = json.dumps(fact.signature, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def storable(value: Any) -> Any:
    """A Neo4j property value: a scalar, a homogeneous list of scalars, or JSON text.

    Neo4j stores nothing nested and no mixed lists. Rather than drop what does
    not fit, it goes in as text and is still there to read back.
    """
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, _SCALARS):
        return value
    if isinstance(value, list | tuple | set | frozenset):
        items = [storable(v) for v in value]
        if isinstance(value, set | frozenset):
            items.sort(key=repr)
        kinds = {type(v) for v in items}
        if kinds == {int, float}:
            return [float(v) for v in items]
        if not items or (len(kinds) == 1 and isinstance(items[0], _SCALARS)):
            return items
        return json.dumps(items, default=str, ensure_ascii=False)
    return json.dumps(value, default=str, ensure_ascii=False)


def provenance_of(fact: Fact, extracted_at: datetime) -> dict[str, Any]:
    """Everything a relationship carries to answer "why is this here?".

    `retrieved_at` is the freshest evidence — the last time a source confirmed
    it — with every source's clock kept in `evidence_retrieved_at`. Evidence is
    written as parallel lists because a Neo4j list holds no nulls and no maps:
    a missing uri is `""`, a missing span is `-1`, and position i in every list
    is the same piece of evidence.
    """
    evidence = fact.evidence
    props: dict[str, Any] = {
        "fact_id": fact.id,
        "signature": signature_of(fact),
        "polarity": fact.polarity.value,
        "identity_keys": list(fact.identity_keys),
        "extractor": fact.extractor,
        "verdict": fact.verdict.value,
        "confidence": fact.confidence,
        "support": fact.support,
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "retrieved_at": max((e.retrieved_at for e in evidence), default=None),
        "extracted_at": extracted_at,
        "evidence_doc_ids": [e.doc_id for e in evidence],
        "evidence_uris": [e.uri or "" for e in evidence],
        "evidence_starts": [e.span.start if e.span else -1 for e in evidence],
        "evidence_ends": [e.span.end if e.span else -1 for e in evidence],
        "evidence_tiers": [e.tier.value for e in evidence],
        "evidence_retrieved_at": [e.retrieved_at for e in evidence],
    }
    for key, value in fact.qualifiers.items():
        props[qualifier_property(key)] = storable(value)
    return props


def _entity_row(entity: Entity) -> dict[str, Any]:
    resolution = entity.resolution
    return {
        "key": entity.key,
        "label": entity.label,
        "aliases": list(entity.aliases),
        "external_id": entity.external_id,
        "resolution_method": resolution.method if resolution else None,
        "resolution_score": resolution.score if resolution else None,
        "resolution_linker": resolution.linker if resolution else None,
        "attributes": {attribute_property(k): storable(v) for k, v in entity.attributes.items()},
    }


def _is_scoped(fact: Fact) -> bool:
    return any(k in fact.qualifiers for k in fact.identity_keys)


# --------------------------------------------------------------------------- #
# The write plan
# --------------------------------------------------------------------------- #


class Statement(NamedTuple):
    """One parameterised Cypher statement and the rows it `UNWIND`s."""

    cypher: str
    rows: list[dict[str, Any]]


def plan(kg: KnowledgeGraph, *, ontology: Ontology | None = None) -> list[Statement]:
    """The graph as `UNWIND … MERGE` statements, in write order. Pure, so it is testable.

    Order matters: nodes before the relationships that `MATCH` them, links last.
    Within each kind the groups are sorted, so one graph always compiles to the
    same statements.
    """
    statements: list[Statement] = []

    by_label: dict[str, list[dict[str, Any]]] = {}
    for (label, _), entity in sorted(_entities_of(kg).items()):
        by_label.setdefault(label, []).append(_entity_row(entity))
    for label, rows in sorted(by_label.items()):
        statements.append(Statement(_entity_cypher(label), rows))

    edges: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    claims: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for fact in kg.facts:
        props = provenance_of(fact, kg.created_at)
        row = {"subject_key": fact.subject.key, "signature": props["signature"], "props": props}
        if fact.object_entity is not None:
            group = (fact.subject.type, fact.predicate, fact.object_entity.type)
            edges.setdefault(group, []).append({**row, "object_key": fact.object_entity.key})
        else:
            claim = {
                "predicate": fact.predicate,
                "subject_type": fact.subject.type,
                "value": storable(fact.object_value),
            }
            claims.setdefault((fact.subject.type, fact.predicate), []).append({**row, **claim})
    for (subject_type, predicate, object_type), rows in sorted(edges.items()):
        statements.append(Statement(_edge_cypher(subject_type, predicate, object_type), rows))
    for (subject_type, predicate), rows in sorted(claims.items()):
        statements.append(Statement(_claim_cypher(subject_type, predicate), rows))
    for (subject_type, predicate), rows in sorted(_projections(kg, ontology).items()):
        statements.append(Statement(_projection_cypher(subject_type, predicate), rows))

    links: dict[str, list[dict[str, Any]]] = {}
    for link in kg.links:
        links.setdefault(link.kind.value.upper(), []).append(_link_row(link))
    for kind, rows in sorted(links.items()):
        statements.append(Statement(_link_cypher(kind), rows))
    return statements


def _entities_of(kg: KnowledgeGraph) -> dict[tuple[str, str], Entity]:
    """Every node to write, by (type, key).

    A fact's own subject and object are written too, so an edge whose entity
    the caller left out of `kg.entities` still finds both ends. The graph's
    list wins where both name the same node.
    """
    seen: dict[tuple[str, str], Entity] = {}
    for fact in kg.facts:
        seen.setdefault((fact.subject.type, fact.subject.key), fact.subject)
        if fact.object_entity is not None:
            obj = fact.object_entity
            seen.setdefault((obj.type, obj.key), obj)
    for entity in kg.entities:
        seen[(entity.type, entity.key)] = entity
    return seen


def _projections(
    kg: KnowledgeGraph, ontology: Ontology | None
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """The literal values that go onto subject nodes as plain properties.

    Only asserted, unscoped claims with a value project: a denial is not a
    value, and a value with an identity-bearing qualifier means nothing without
    its scope. A single-valued predicate projects its best-supported claim; a
    multi-valued one, when the ontology says so, the list in that order.
    """
    grouped: dict[tuple[str, str, str], list[Fact]] = {}
    for fact in kg.facts:
        if (
            fact.object_entity is not None
            or fact.object_value is None
            or fact.polarity is not Polarity.ASSERTED
            or _is_scoped(fact)
        ):
            continue
        grouped.setdefault((fact.subject.type, fact.predicate, fact.subject.key), []).append(fact)

    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (subject_type, predicate, subject_key), facts in sorted(grouped.items()):
        ranked = sorted(facts, key=lambda f: (-f.support, -f.confidence, repr(f.object_value)))
        declared = ontology.predicates.get(predicate) if ontology else None
        if declared is not None and declared.cardinality == "multi":
            values: list[Any] = []
            for fact in ranked:
                value = storable(fact.object_value)
                if value not in values:
                    values.append(value)
            projected = storable(values)
        else:
            projected = storable(ranked[0].object_value)
        out.setdefault((subject_type, predicate), []).append(
            {"subject_key": subject_key, "value": projected}
        )
    return out


def _link_row(link: EntityLink) -> dict[str, Any]:
    return {
        "source_key": link.source_key,
        "target_key": link.target_key,
        "props": {
            "score": link.score,
            "reason": link.reason,
            "created_at": link.created_at,
            "evidence_doc_ids": [e.doc_id for e in link.evidence],
            "evidence_uris": [e.uri or "" for e in link.evidence],
        },
    }


def _entity_cypher(label: str) -> str:
    return (
        "UNWIND $rows AS row\n"
        f"MERGE (n:{_ident(label)} {{key: row.key}})\n"
        f"SET n:{_ident(ENTITY_LABEL)}, n += row.attributes,\n"
        "    n.label = row.label, n.aliases = row.aliases, n.external_id = row.external_id,\n"
        "    n.resolution_method = row.resolution_method,\n"
        "    n.resolution_score = row.resolution_score,\n"
        "    n.resolution_linker = row.resolution_linker"
    )


def _edge_cypher(subject_type: str, predicate: str, object_type: str) -> str:
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (s:{_ident(subject_type)} {{key: row.subject_key}})\n"
        f"MATCH (o:{_ident(object_type)} {{key: row.object_key}})\n"
        f"MERGE (s)-[r:{_ident(predicate)} {{signature: row.signature}}]->(o)\n"
        "SET r += row.props"
    )


def _claim_cypher(subject_type: str, predicate: str) -> str:
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (s:{_ident(subject_type)} {{key: row.subject_key}})\n"
        f"MERGE (c:{_ident(CLAIM_LABEL)} {{signature: row.signature}})\n"
        "SET c.predicate = row.predicate, c.subject_key = row.subject_key,\n"
        "    c.subject_type = row.subject_type, c.value = row.value\n"
        f"MERGE (s)-[r:{_ident(predicate)} {{signature: row.signature}}]->(c)\n"
        "SET r += row.props"
    )


def _projection_cypher(subject_type: str, predicate: str) -> str:
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (s:{_ident(subject_type)} {{key: row.subject_key}})\n"
        f"SET s.{_ident(projection_property(predicate))} = row.value"
    )


def _link_cypher(kind: str) -> str:
    # MATCH, not MERGE, on the ends: a link never creates the entities it is
    # about, and it can point at nodes an earlier run wrote.
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (a:{_ident(ENTITY_LABEL)} {{key: row.source_key}})\n"
        f"MATCH (b:{_ident(ENTITY_LABEL)} {{key: row.target_key}})\n"
        f"MERGE (a)-[l:{_ident(kind)}]->(b)\n"
        "SET l += row.props"
    )


def _batches(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


# --------------------------------------------------------------------------- #
# The sink
# --------------------------------------------------------------------------- #


def _connect(uri: str, auth: Any) -> Any:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    return GraphDatabase.driver(uri, auth=auth)


def _unwind(tx: Any, cypher: str, rows: list[dict[str, Any]]) -> None:
    tx.run(cypher, rows=rows).consume()


class Neo4jSink:
    """Writes a `KnowledgeGraph` to Neo4j with batched, idempotent `UNWIND … MERGE`.

    The shape, and why:

    - An entity is a node `(:Type:Entity {key})`, MERGEd per type on `key`,
      with `label`, `aliases`, `external_id`, `resolution_*` and its
      attributes as properties.
    - An edge fact is `(s)-[:predicate {signature, …}]->(o)`, MERGEd on the
      fact's signature — subject, predicate, object, polarity and the
      identity-bearing qualifiers (DECISIONS #14, #15) — so a second run finds
      the edge it wrote and updates it rather than adding another. Two facts
      that differ only in a reconcilable qualifier are one edge; uptime at p50
      and at p95 are two; a denial is its own edge with `polarity: 'denied'`.
    - A literal fact has the same shape with a `:Claim` node as its object:
      `(s)-[:predicate {signature, …}]->(:Claim {signature, value})`. A
      property cannot carry provenance, nor two contested values, nor a
      denial, so the Claim is the record and provenance lives on its
      relationship exactly as it does on an edge — one query shape for both
      kinds of fact. The value is *also* projected onto the subject as
      `s.predicate`, because that is what a Cypher query reaches for first.
      The projection is derived and lossy on purpose: only asserted, unscoped
      claims project; a single-valued predicate carries its best-supported
      claim; a multi-valued one (when the ontology is known) the list.
    - An `EntityLink` is `(a)-[:SAME_AS|SIMILAR|DIFFERENT {score, reason}]->(b)`.
      Nodes are never merged (DECISIONS #16).

    Every fact relationship carries its provenance: evidence document ids,
    uris and span offsets, tiers, extractor, verdict, confidence, support,
    both clocks, and the reconcilable qualifiers as properties. "Why is this
    edge here?" is a read of the edge; "forget this source" is
    `MATCH ()-[r]->() WHERE $doc IN r.evidence_doc_ids DELETE r`.

    A single-valued predicate with two objects is written as two edges, not
    replaced: the store holds the conflict and a check query reports it
    (09-quality §4) — picking a winner at write time would destroy the
    evidence for the loser.

    Rows go in batches of `batch_size`, one managed transaction each, so a
    failed batch rolls back whole and never half-writes. Earlier batches stay
    committed; because every statement is a MERGE, recovering is running the
    write again.

    `profile` says the store constrains: the uniqueness constraints are what
    make these MERGEs correct under concurrent writers. It neither resolves
    nor prunes.
    """

    profile = PlatformProfile(name="neo4j", resolves=False, constrains=True, prunes=False)

    def __init__(
        self,
        uri: str | None = None,
        auth: Any = None,
        *,
        database: str | None = None,
        batch_size: int = 500,
        driver: Any = None,
        ontology: Ontology | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if driver is None:
            if uri is None:
                raise ValueError("Neo4jSink needs a uri (with auth) or a driver")
            driver = _connect(uri, auth)
        self._driver = driver
        self.database = database
        self.batch_size = batch_size
        # Decides whether a projected property is one value or a list.
        self.ontology = ontology

    def statements(self, kg: KnowledgeGraph) -> list[Statement]:
        """What `write()` would run, without running it."""
        return plan(kg, ontology=self.ontology)

    def write(self, kg: KnowledgeGraph) -> None:
        with self._driver.session(**self._session_config()) as session:
            for statement in self.statements(kg):
                for batch in _batches(statement.rows, self.batch_size):
                    session.execute_write(_unwind, statement.cypher, batch)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _session_config(self) -> dict[str, Any]:
        return {"database": self.database} if self.database else {}


__all__ = [
    "CLAIM_LABEL",
    "ENTITY_LABEL",
    "EXTRA_HINT",
    "Neo4jSink",
    "Statement",
    "plan",
    "provenance_of",
    "signature_of",
    "storable",
]
