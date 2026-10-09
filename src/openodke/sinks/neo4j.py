"""The Neo4j platform: the sink, the constrainer that is its other half, and the lookup.

One module because they are one design. The sink writes with `MERGE` on entity
keys and fact signatures, so a pipeline run twice updates the graph it wrote
rather than duplicating it. The constrainer compiles the ontology into the
uniqueness constraints that make those MERGEs correct under concurrency, and
into check queries for the rule Neo4j cannot enforce. That is the dotted arrow
of the design: the ontology that shaped the prompt shapes the store, and
nobody writes the rules twice. The lookup reads the store back through those
indexes, so a batch is resolved against the graph without loading it.

The module never imports the driver at the top. `import openodke.sinks.neo4j` works
on the base install (DECISIONS #1), and so does printing the DDL; the driver is
imported the first time a connection is needed, behind the `[neo4j]` extra.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, time
from enum import Enum
from typing import Any, NamedTuple

from openodke.corroborate.lookup import Embed, embedding_text
from openodke.corroborate.merge import _aware
from openodke.corroborate.provenance import CONFLICT
from openodke.corroborate.resolve import block_keys
from openodke.ontology import Ontology, Predicate
from openodke.stages import DDL, Constrainer, PlatformProfile
from openodke.types import (
    Entity,
    EntityLink,
    Fact,
    KnowledgeGraph,
    Polarity,
    Resolution,
    SourceTier,
    Support,
)

EXTRA_HINT = "the neo4j driver is not installed; run: pip install 'openodke[neo4j]'"

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
        "support_sources",
        "support_doc_ids",
        "support_doc_sources",
        "support_tiers",
        "support_retrieved_at",
        "valid_from",
        "valid_to",
        "retrieved_at",
        "extracted_at",
        "evidence_doc_ids",
        "evidence_uris",
        "evidence_starts",
        "evidence_ends",
        "evidence_span_origins",
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
    is the same piece of evidence. `evidence_span_origins` says who chose each
    span, so a whole-text stand-in never reads as a citation (DECISIONS #25).

    The support list is parallel lists too, one position per independent
    source (`support_sources`, `support_tiers`, `support_retrieved_at`), and
    its documents are `support_doc_ids`, each beside the source it belongs to
    in `support_doc_sources`, because a list holds no lists. `support` is the
    length of `support_sources` whenever the list is filled (#115).

    A naive clock beside an aware one is read as UTC, as the corroborator reads
    it: Python cannot compare the two, and to Neo4j they are two types, which no
    list may mix.
    """
    evidence, sources = fact.evidence, fact.supported_by
    clocks = _one_kind([e.retrieved_at for e in evidence])
    props: dict[str, Any] = {
        "fact_id": fact.id,
        "signature": signature_of(fact),
        "polarity": fact.polarity.value,
        "identity_keys": list(fact.identity_keys),
        "extractor": fact.extractor,
        "verdict": fact.verdict.value,
        "confidence": fact.confidence,
        "support": fact.support,
        "support_sources": [entry.source for entry in sources],
        "support_doc_ids": [doc for entry in sources for doc in entry.doc_ids],
        "support_doc_sources": [entry.source for entry in sources for _ in entry.doc_ids],
        "support_tiers": [entry.tier.value for entry in sources],
        "support_retrieved_at": _one_kind([entry.retrieved_at for entry in sources]),
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "retrieved_at": max(clocks, default=None),
        "extracted_at": extracted_at,
        "evidence_doc_ids": [e.doc_id for e in evidence],
        "evidence_uris": [e.uri or "" for e in evidence],
        "evidence_starts": [e.span.start if e.span else -1 for e in evidence],
        "evidence_ends": [e.span.end if e.span else -1 for e in evidence],
        "evidence_span_origins": [e.span_origin.value for e in evidence],
        "evidence_tiers": [e.tier.value for e in evidence],
        "evidence_retrieved_at": clocks,
    }
    for key, value in fact.qualifiers.items():
        props[qualifier_property(key)] = storable(value)
    return props


def _one_kind(clocks: list[datetime]) -> list[datetime]:
    """Clocks Neo4j can hold in one list: all aware when any is, a naive one read as UTC."""
    if len({c.tzinfo is None for c in clocks}) > 1:
        return [_aware(c) for c in clocks]
    return clocks


def support_from(props: Mapping[str, Any]) -> tuple[Support, ...]:
    """A relationship's support lists read back as the `Fact.supported_by` that wrote them.

    The inverse of `provenance_of`'s `support_*` properties, from Neo4j, a
    NetworkX edge or a CSV row once its arrays are split. Empty when the fact
    had no list, as a relationship written before support lists existed has not.
    """
    documents: dict[str, list[str]] = defaultdict(list)
    owners = props.get("support_doc_sources") or ()
    for doc, owner in zip(props.get("support_doc_ids") or (), owners, strict=True):
        documents[str(owner)].append(str(doc))
    sources = [str(s) for s in props.get("support_sources") or ()]
    tiers = list(props.get("support_tiers") or ())
    clocks = [_native(c) for c in props.get("support_retrieved_at") or ()]
    return tuple(
        Support(
            source=source,
            doc_ids=tuple(documents.get(source, ())),
            tier=SourceTier(tier),
            retrieved_at=clock,
        )
        for source, tier, clock in zip(sources, tiers, clocks, strict=True)
    )


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


def is_scoped(fact: Fact) -> bool:
    """True when the fact carries a value for one of its identity-bearing qualifiers."""
    return any(k in fact.qualifiers for k in fact.identity_keys)


def is_outvoted(fact: Fact) -> bool:
    """True when the corroborator weighed this fact against a rival and it lost.

    The loser stays a record — its edge, its claim, its statement node — but it
    is not the value. More sources can back the loser than the winner, so
    anything that picks "the" value has to ask this rather than rank on support.
    """
    conflict = fact.qualifiers.get(CONFLICT)
    return isinstance(conflict, Mapping) and conflict.get("status") == "lost"


# --------------------------------------------------------------------------- #
# The write plan
# --------------------------------------------------------------------------- #


class Statement(NamedTuple):
    """One parameterised Cypher statement and the rows it `UNWIND`s.

    `kind` and `names` say what the rows are without parsing the Cypher, which
    is how the bulk sinks lay the same plan out as a script or as CSV files.
    """

    cypher: str
    rows: list[dict[str, Any]]
    # entity | edge | claim | projection | link
    kind: str = ""
    # (label,) · (subject type, predicate, object type) · (subject type,
    # predicate) for a claim or a projection · (link kind,)
    names: tuple[str, ...] = ()


def plan(kg: KnowledgeGraph, *, ontology: Ontology | None = None) -> list[Statement]:
    """The graph as `UNWIND … MERGE` statements, in write order. Pure, so it is testable.

    Order matters: nodes before the relationships that `MATCH` them, links last.
    Within each kind the groups are sorted, so one graph always compiles to the
    same statements.
    """
    statements: list[Statement] = []

    by_label: dict[str, list[dict[str, Any]]] = {}
    for (label, _), entity in sorted(entities_of(kg).items()):
        by_label.setdefault(label, []).append(_entity_row(entity))
    for label, rows in sorted(by_label.items()):
        statements.append(Statement(_entity_cypher(label), rows, "entity", (label,)))

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
        cypher = _edge_cypher(subject_type, predicate, object_type)
        statements.append(Statement(cypher, rows, "edge", (subject_type, predicate, object_type)))
    for (subject_type, predicate), rows in sorted(claims.items()):
        cypher = _claim_cypher(subject_type, predicate)
        statements.append(Statement(cypher, rows, "claim", (subject_type, predicate)))
    for (subject_type, predicate), rows in sorted(projections(kg, ontology).items()):
        cypher = _projection_cypher(subject_type, predicate)
        statements.append(Statement(cypher, rows, "projection", (subject_type, predicate)))

    links: dict[str, list[dict[str, Any]]] = {}
    for link in kg.links:
        links.setdefault(link.kind.value.upper(), []).append(link_row(link))
    for kind, rows in sorted(links.items()):
        statements.append(Statement(_link_cypher(kind), rows, "link", (kind,)))
    return statements


def entities_of(kg: KnowledgeGraph) -> dict[tuple[str, str], Entity]:
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


def projections(
    kg: KnowledgeGraph, ontology: Ontology | None
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """The literal values that go onto subject nodes as plain properties.

    Only asserted, unscoped claims with a value project: a denial is not a
    value, and a value with an identity-bearing qualifier means nothing without
    its scope. Nor does a claim the corroborator voted down, however many
    sources back it. A single-valued predicate projects its best-supported
    claim; a multi-valued one, when the ontology says so, the list in that order.
    """
    grouped: dict[tuple[str, str, str], list[Fact]] = {}
    for fact in kg.facts:
        if (
            fact.object_entity is not None
            or fact.object_value is None
            or fact.polarity is not Polarity.ASSERTED
            or is_scoped(fact)
            or is_outvoted(fact)
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


def link_row(link: EntityLink) -> dict[str, Any]:
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
    # `=`, not `+=`: a rewritten fact replaces what its relationship says, so a
    # conflict stamp from a run that had a rival does not outlive the rival. The
    # props carry the signature, so the key the MERGE matched on survives.
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (s:{_ident(subject_type)} {{key: row.subject_key}})\n"
        f"MATCH (o:{_ident(object_type)} {{key: row.object_key}})\n"
        f"MERGE (s)-[r:{_ident(predicate)} {{signature: row.signature}}]->(o)\n"
        "SET r = row.props"
    )


def _claim_cypher(subject_type: str, predicate: str) -> str:
    return (
        "UNWIND $rows AS row\n"
        f"MATCH (s:{_ident(subject_type)} {{key: row.subject_key}})\n"
        f"MERGE (c:{_ident(CLAIM_LABEL)} {{signature: row.signature}})\n"
        "SET c.predicate = row.predicate, c.subject_key = row.subject_key,\n"
        "    c.subject_type = row.subject_type, c.value = row.value\n"
        f"MERGE (s)-[r:{_ident(predicate)} {{signature: row.signature}}]->(c)\n"
        "SET r = row.props"
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
      the edge it wrote and replaces its properties rather than adding another
      edge; what the fact no longer carries goes with the rewrite. Two facts
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
      claims project, and never one the corroborator voted down; a
      single-valued predicate carries its best-supported claim; a multi-valued
      one (when the ontology is known) the list.
    - An `EntityLink` is `(a)-[:SAME_AS|SIMILAR|DIFFERENT {score, reason}]->(b)`.
      Nodes are never merged (DECISIONS #16).

    Every fact relationship carries its provenance: evidence document ids,
    uris, span offsets and who chose them, tiers, extractor, verdict,
    confidence, support and the sources it counts, both clocks, and the
    reconcilable qualifiers as properties. "Why is this edge here?" is a read
    of the edge. Forgetting a source is not a delete: other sources may back
    the same fact, and its `:Claim` and projected value would stay. That is
    the reconciler's job (#116).

    A single-valued predicate with two objects is written as two edges, not
    replaced: the store holds the conflict and a check query reports it
    (09-quality §4) — picking a winner at write time would destroy the
    evidence for the loser.

    Rows go in batches of `batch_size`, one managed transaction each, so a
    failed batch rolls back whole and never half-writes. Earlier batches stay
    committed; because every statement is a MERGE, recovering is running the
    write again.

    `profile` says the store constrains: once `bootstrap()` has applied a
    `Neo4jConstrainer`'s DDL, the uniqueness constraints are what make these
    MERGEs correct under concurrent writers. It neither resolves nor prunes.
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
        if ontology is not None:
            ontology.warn_if_unreviewed("Neo4jSink will shape its writes")

    def statements(self, kg: KnowledgeGraph) -> list[Statement]:
        """What `write()` would run, without running it."""
        return plan(kg, ontology=self.ontology)

    def write(self, kg: KnowledgeGraph) -> None:
        with self._driver.session(**self._session_config()) as session:
            for statement in self.statements(kg):
                for batch in _batches(statement.rows, self.batch_size):
                    session.execute_write(_unwind, statement.cypher, batch)

    def bootstrap(
        self,
        ontology: Ontology,
        *,
        constrainer: Constrainer | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Apply the ontology's constraints and indexes; safe to run on every start.

        Every statement is `IF NOT EXISTS`, so a database already constrained is
        left as it is. Returns the DDL it ran; with `dry_run` it runs nothing,
        so a DBA can apply the same statements by hand. Check queries are not
        run here — they return violators, and `check()` is where to ask.

        An ontology still marked `inferred` warns (`UnreviewedOntologyWarning`):
        constraints compiled from a schema nobody reviewed are hard to take back.
        """
        ontology.warn_if_unreviewed("Neo4jSink.bootstrap is compiling store constraints")
        self.ontology = ontology
        compiled = (constrainer or Neo4jConstrainer()).constrain(ontology)
        ddl = [s for s in compiled if not is_check(s)]
        if not dry_run:
            with self._driver.session(**self._session_config()) as session:
                for statement in ddl:
                    session.run(statement).consume()
        return ddl

    def check(
        self, ontology: Ontology, *, constrainer: Constrainer | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Run the cardinality checks; the violators, by predicate.

        A check and never a repair (09-quality §4): nothing here deletes, merges
        or picks a winner. What to do with a Person holding two employers is a
        person's call, and the rows say which ones to look at.
        """
        compiled = (constrainer or Neo4jConstrainer()).constrain(ontology)
        found: dict[str, list[dict[str, Any]]] = {}
        with self._driver.session(**self._session_config()) as session:
            for statement in (s for s in compiled if is_check(s)):
                rows = session.execute_read(_rows, statement)
                if rows:
                    found[check_target(statement)] = rows
        return found

    def lookup(self, **options: Any) -> Neo4jLookup:
        """A `Neo4jLookup` on this sink's connection, database and ontology.

        It shares the driver, so closing the sink closes it. `options` are
        `Neo4jLookup`'s: `tenant`, `tenant_property`, `limit`, `embed`,
        `vector_index`, `vector_k`.
        """
        options.setdefault("ontology", self.ontology)
        return Neo4jLookup(driver=self._driver, database=self.database, **options)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _session_config(self) -> dict[str, Any]:
        return {"database": self.database} if self.database else {}


# --------------------------------------------------------------------------- #
# The lookup
# --------------------------------------------------------------------------- #

# Read once per lookup: which indexes the store has, so that no query is sent
# that would scan for want of one.
SHOW_INDEXES = "SHOW INDEXES YIELD name, type, labelsOrTypes, properties"


class StoreIndexes(NamedTuple):
    """The indexes a lookup can read through, by entity type."""

    # Types with a range index on `key`: the per-type uniqueness constraint's.
    keys: frozenset[str]
    # The sink's `:Entity(key)` index, which finds a key without its type's.
    entity_key: bool
    # Types with a range index on `external_id`.
    ids: frozenset[str]
    # Type -> a full-text index over `label` and `aliases`.
    names: Mapping[str, str]


def store_indexes(rows: Iterable[Mapping[str, Any]]) -> StoreIndexes:
    """`SHOW INDEXES` rows read as what a lookup can use. Bootstrap's own names win ties."""
    keys: set[str] = set()
    ids: set[str] = set()
    names: dict[str, str] = {}
    entity_key = False
    for row in sorted(rows, key=lambda r: str(r.get("name"))):
        labels = list(row.get("labelsOrTypes") or ())
        props = list(row.get("properties") or ())
        kind = row.get("type")
        if kind == "RANGE" and len(labels) == 1 and props == ["key"]:
            if labels[0] == ENTITY_LABEL:
                entity_key = True
            keys.add(labels[0])
        elif kind == "RANGE" and len(labels) == 1 and props == ["external_id"]:
            ids.add(labels[0])
        elif kind == "FULLTEXT" and {"label", "aliases"} & set(props):
            for label in labels:
                own = _schema_name("names", label)
                if label not in names or row.get("name") == own:
                    names[label] = str(row["name"])
    return StoreIndexes(frozenset(keys), entity_key, frozenset(ids), names)


class LookupQuery(NamedTuple):
    """One read statement of a lookup: its Cypher, its parameters, and what it reads."""

    cypher: str
    params: dict[str, Any]
    # The entity type it is scoped to, and how it finds them: key | id | names | vector.
    type: str
    kind: str


def _lookup_cypher(find: str, conditions: Sequence[str]) -> str:
    """`UNWIND` the batch's block keys, find each through an index, keep what passes."""
    lines = ["UNWIND $rows AS row", find]
    if conditions:
        lines.append("WHERE " + " AND ".join(conditions))
    lines.append("RETURN row.block AS block, properties(n) AS node")
    return "\n".join(lines)


def _accented(tokens: Iterable[str], entity: Entity) -> set[str]:
    """The name tokens again as written, where the writing has accents a name key drops.

    The full-text index's analyzer lowercases and keeps accents, and a name
    key drops them, so "José" is asked for as `jose` and as `josé`.
    """
    wanted = set(tokens)
    found: set[str] = set()
    for name in (entity.label, *entity.aliases):
        for word in re.findall(r"[^\W_]+", name.casefold()) if name else ():
            plain = "".join(
                c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c)
            )
            if plain != word and plain in wanted:
                found.add(word)
    return found


def _phrase(text: str) -> str:
    """One term or phrase for Lucene's query parser, quoted so no word is an operator."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def id_forms(raw: str) -> list[str]:
    """The spellings an external id is looked up under: as written, and in each case.

    The resolver compares ids compacted and casefolded within a scheme
    (`wikidata:Q95` is `wikidata:q95`); a range index matches exactly, so the
    lookup asks for the forms a writer is likely to have used. Any other
    spelling is still found through the name, if the names share a token.
    """
    text = raw.strip()
    compact = "".join(text.split())
    forms = [text, compact, compact.lower(), compact.upper()]
    scheme, sep, value = compact.partition(":")
    if sep and value and not value.startswith("//"):
        forms += [f"{scheme.lower()}:{value.upper()}", f"{scheme.lower()}:{value.lower()}"]
    return list(dict.fromkeys(forms))


def _native(value: Any) -> Any:
    """A driver value as Python's own: a Neo4j date or time is not equal to a `date`."""
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        return to_native()
    if isinstance(value, list):
        return [_native(v) for v in value]
    return value


def stored_entity(
    props: Mapping[str, Any], entity_type: str, *, ontology: Ontology | None = None
) -> Entity:
    """A node the sink wrote, read back as the `Entity` that wrote it.

    The inverse of the sink's entity row: `label`, `aliases`, `external_id` and
    `resolution_*` are fields, an attribute the sink prefixed gets its name
    back, and every other property is an attribute, except a value the ontology
    says was projected from a fact. Written again, it sets every property to
    what it already holds.
    """
    rest = {name: _native(value) for name, value in props.items()}
    method = rest.pop("resolution_method", None)
    score = rest.pop("resolution_score", None)
    linker = rest.pop("resolution_linker", None)
    resolution = (
        Resolution(method=method, score=score, linker=linker)
        if method in ("caller", "external_id", "linker")
        else None
    )
    key = str(rest.pop("key"))
    label = rest.pop("label", None)
    aliases = tuple(str(a) for a in rest.pop("aliases", None) or ())
    external_id = rest.pop("external_id", None)
    projected = (
        {projection_property(p) for p in ontology.predicates} if ontology is not None else set()
    )
    attributes: dict[str, Any] = {}
    for name, value in sorted(rest.items()):
        if name in projected:
            continue
        own = name.removeprefix("attribute_")
        attributes[own if own != name and own in _ENTITY_FIELDS else name] = value
    return Entity(
        key=key,
        type=entity_type,
        label=label,
        aliases=aliases,
        external_id=external_id,
        resolution=resolution,
        attributes=attributes,
    )


class _Blocks:
    """Block keys numbered as they are first asked for, each with the entities that asked."""

    def __init__(self) -> None:
        self.probes: list[list[str]] = []
        self._ids: dict[tuple[str, ...], int] = {}

    def ask(self, key: tuple[str, ...], probe: str) -> tuple[int, bool]:
        """The block's id, and whether this is the first time it was asked for."""
        new = key not in self._ids
        if new:
            self._ids[key] = len(self.probes)
            self.probes.append([])
        at = self._ids[key]
        if probe not in self.probes[at]:
            self.probes[at].append(probe)
        return at, new


def _block_rows(
    members: Sequence[Entity], blocks: _Blocks, vectors: Mapping[str, list[float]] | None
) -> dict[str, list[dict[str, Any]]]:
    """One type's distinct block keys as `UNWIND` rows, by how they are looked up.

    A block key two entities share is one row, asked once; the rows carry the
    block's id, and the reply is handed to every entity that asked.
    """
    rows: dict[str, list[dict[str, Any]]] = {"key": [], "id": [], "names": [], "vector": []}
    for entity in members:
        kind, probe = entity.type, entity.key
        at, new = blocks.ask(("key", kind, entity.key), probe)
        if new:
            rows["key"].append({"block": at, "value": entity.key})
        keys = block_keys(entity)
        for scheme, value in sorted(keys.ids):
            at, new = blocks.ask(("id", kind, scheme, value), probe)
            if new and entity.external_id:
                rows["id"] += [{"block": at, "value": f} for f in id_forms(entity.external_id)]
        for domain in sorted(keys.domains):
            at, new = blocks.ask(("domain", kind, domain), probe)
            if new:
                query = f"{_phrase(domain)} OR {_phrase('www.' + domain)}"
                rows["names"].append({"block": at, "query": query})
        for token in sorted(keys.tokens | _accented(keys.tokens, entity)):
            at, new = blocks.ask(("name", kind, token), probe)
            if new:
                rows["names"].append({"block": at, "query": _phrase(token)})
        if vectors is not None:
            text = embedding_text(entity)
            at, new = blocks.ask(("vector", kind, text), probe)
            if new:
                rows["vector"].append({"block": at, "vector": vectors[text]})
    return rows


def _read_all(tx: Any, queries: Sequence[LookupQuery]) -> list[list[dict[str, Any]]]:
    return [tx.run(q.cypher, q.params).data() for q in queries]


class Neo4jLookup:
    """A `StoreLookup` over a graph `Neo4jSink` wrote: index queries, one read transaction a batch.

    Resolving a batch against the store asks it, per entity, what it already
    holds of the same type under the same key, an external id, a domain, or a
    name token (DECISIONS #31). Each of those is an index query, and each query
    `UNWIND`s the batch's distinct block keys, so a key two entities share is
    asked once:

    - the key, through the per-type uniqueness constraint, or the sink's
      `:Entity(key)` index;
    - the external id, through the per-type `external_id` index, under the
      spellings `id_forms` gives;
    - a name token or a domain, through the per-type full-text index on
      `label` and `aliases`: `limit` hits a token, the best first;
    - with `embed` and `vector_index`, the `vector_k` nearest of a vector index
      the caller keeps. openodke writes no embeddings; the index and its
      property are yours, and `embed` must be the function that filled it.

    `Neo4jSink.bootstrap(ontology)` creates every index above but the vector
    one. The lookup reads which exist once, with `SHOW INDEXES`, and never
    queries without one: a type with no full-text index has no names looked
    up, and it warns once saying so, rather than scan. The whole batch is read
    in one read transaction, and nothing is written.

    `tenant` keeps only nodes whose `tenant_property` equals it. Tenant keys
    arrive with #159; until then it is a filter on a property, applied after
    the index, so a full-text `limit` counts nodes of every tenant.
    """

    def __init__(
        self,
        uri: str | None = None,
        auth: Any = None,
        *,
        database: str | None = None,
        driver: Any = None,
        tenant: str | None = None,
        tenant_property: str = "tenant",
        limit: int = 100,
        embed: Embed | None = None,
        vector_index: str | None = None,
        vector_k: int = 5,
        ontology: Ontology | None = None,
    ) -> None:
        if limit < 1 or vector_k < 1:
            raise ValueError("limit and vector_k must be at least 1")
        if (embed is None) != (vector_index is None):
            raise ValueError("a vector lookup needs both embed and vector_index")
        self._owns_driver = driver is None
        if driver is None:
            if uri is None:
                raise ValueError("Neo4jLookup needs a uri (with auth) or a driver")
            driver = _connect(uri, auth)
        self._driver = driver
        self.database = database
        self.tenant = tenant
        self.tenant_property = tenant_property
        self.limit = limit
        self.embed = embed
        self.vector_index = vector_index
        self.vector_k = vector_k
        # Tells a projected fact value from an attribute when a node is read back.
        self.ontology = ontology
        self._indexes: StoreIndexes | None = None
        self._warned: set[str] = set()
        self.stats: dict[str, Any] = {"transactions": 0, "statements": 0, "unindexed": []}

    def indexes(self) -> StoreIndexes:
        """The store's indexes, read once with `SHOW INDEXES`."""
        if self._indexes is None:
            with self._driver.session(**self._session_config()) as session:
                self._indexes = store_indexes(session.run(SHOW_INDEXES).data())
        return self._indexes

    def candidates(self, entities: Sequence[Entity]) -> Mapping[str, Sequence[Entity]]:
        entities = list(entities)
        if not entities:
            return {}
        queries, probes = self._plan(entities)
        found: dict[str, dict[str, Entity]] = {e.key: {} for e in entities}
        if queries:
            with self._driver.session(**self._session_config()) as session:
                results = session.execute_read(_read_all, queries)
            self.stats["transactions"] += 1
            self.stats["statements"] += len(queries)
            for query, rows in zip(queries, results, strict=True):
                for row in rows:
                    entity = stored_entity(row["node"], query.type, ontology=self.ontology)
                    for probe in probes[row["block"]]:
                        found[probe].setdefault(entity.key, entity)
        return {key: list(stored.values()) for key, stored in found.items()}

    def statements(self, entities: Sequence[Entity]) -> list[LookupQuery]:
        """What `candidates()` would read, without reading it (the indexes are read once)."""
        return self._plan(list(entities))[0]

    def _plan(self, entities: list[Entity]) -> tuple[list[LookupQuery], list[list[str]]]:
        indexes = self.indexes()
        blocks = _Blocks()
        vectors = self._vectors(entities) if self.vector_index is not None else None
        by_type: dict[str, list[Entity]] = {}
        for entity in entities:
            by_type.setdefault(entity.type, []).append(entity)
        queries: list[LookupQuery] = []
        for entity_type in sorted(by_type):
            rows = _block_rows(by_type[entity_type], blocks, vectors)
            queries += self._typed(entity_type, rows, indexes)
        return [q for q in queries if q.params["rows"]], blocks.probes

    def _typed(
        self, entity_type: str, rows: Mapping[str, list[dict[str, Any]]], indexes: StoreIndexes
    ) -> list[LookupQuery]:
        """One type's statements, each through an index the store has."""
        label = _ident(entity_type)
        scoped = self._tenant_conditions()
        out: list[LookupQuery] = []
        if entity_type in indexes.keys:
            cypher = _lookup_cypher(f"MATCH (n:{label} {{key: row.value}})", scoped)
            out.append(self._query(cypher, rows["key"], entity_type, "key"))
        elif indexes.entity_key:
            # The sink's `:Entity(key)` index, where the type has no constraint of its own.
            find = f"MATCH (n:{_ident(ENTITY_LABEL)} {{key: row.value}})"
            out.append(
                self._query(
                    _lookup_cypher(find, [f"n:{label}", *scoped]), rows["key"], entity_type, "key"
                )
            )
        else:
            self._unindexed(entity_type, "key")
        if rows["id"]:
            if entity_type in indexes.ids:
                find = f"MATCH (n:{label} {{external_id: row.value}})"
                out.append(self._query(_lookup_cypher(find, scoped), rows["id"], entity_type, "id"))
            else:
                self._unindexed(entity_type, "external_id")
        if rows["names"]:
            index = indexes.names.get(entity_type)
            if index is not None:
                find = (
                    "CALL db.index.fulltext.queryNodes($index, row.query, {limit: $limit}) "
                    "YIELD node AS n"
                )
                cypher = _lookup_cypher(find, [f"n:{label}", *scoped])
                out.append(
                    self._query(
                        cypher, rows["names"], entity_type, "names", index=index, limit=self.limit
                    )
                )
            else:
                self._unindexed(entity_type, "names")
        if rows["vector"]:
            find = "CALL db.index.vector.queryNodes($index, $k, row.vector) YIELD node AS n"
            cypher = _lookup_cypher(find, [f"n:{label}", *scoped])
            out.append(
                self._query(
                    cypher,
                    rows["vector"],
                    entity_type,
                    "vector",
                    index=self.vector_index,
                    k=self.vector_k,
                )
            )
        return out

    def _query(
        self, cypher: str, rows: list[dict[str, Any]], entity_type: str, kind: str, **params: Any
    ) -> LookupQuery:
        if self.tenant is not None:
            params["tenant"] = self.tenant
        return LookupQuery(cypher, {"rows": rows, **params}, entity_type, kind)

    def _tenant_conditions(self) -> list[str]:
        if self.tenant is None:
            return []
        return [f"n.{_ident(self.tenant_property)} = $tenant"]

    def _vectors(self, entities: Sequence[Entity]) -> dict[str, list[float]]:
        if self.embed is None or not entities:
            return {}
        texts = list(dict.fromkeys(embedding_text(e) for e in entities))
        return {
            text: [float(x) for x in vector]
            for text, vector in zip(texts, self.embed(texts), strict=True)
        }

    def _unindexed(self, entity_type: str, what: str) -> None:
        gap = f"{entity_type}.{what}"
        if gap in self._warned:
            return
        self._warned.add(gap)
        self.stats["unindexed"] = sorted(self._warned)
        warnings.warn(
            f"the store has no index for {gap}, so the lookup does not read it rather than "
            "scan; Neo4jSink.bootstrap(ontology) creates the index for each type in the "
            "ontology",
            stacklevel=2,
        )

    def close(self) -> None:
        """Close the driver, when this lookup opened it."""
        if self._owns_driver:
            self._driver.close()

    def __enter__(self) -> Neo4jLookup:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _session_config(self) -> dict[str, Any]:
        return {"database": self.database} if self.database else {}


# --------------------------------------------------------------------------- #
# The constrainer
# --------------------------------------------------------------------------- #

# A check query returns violators and changes nothing. The marker is a Cypher
# comment, so the DDL stays runnable by hand, and it lets `bootstrap()` run the
# DDL and `check()` run the checks without either running the other.
CHECK_MARKER = "// odke:check "


def is_check(statement: str) -> bool:
    return statement.startswith(CHECK_MARKER)


def check_target(statement: str) -> str:
    """The predicate a check query is about."""
    return statement.splitlines()[0].removeprefix(CHECK_MARKER)


def cardinality_scope(predicate: Predicate) -> tuple[str, ...]:
    """Deprecated: `predicate.scope_keys`, which the constrainer reads itself."""
    warnings.warn(
        "cardinality_scope(predicate) is deprecated; use predicate.scope_keys",
        DeprecationWarning,
        stacklevel=2,
    )
    return predicate.scope_keys


def _schema_name(kind: str, name: str) -> str:
    # Constraint and index names must be plain identifiers. A squashed name
    # carries a hash of the original so `A-B` and `A_B` stay distinct: an
    # existing name turns `IF NOT EXISTS` into a silent no-op.
    slug = re.sub(r"\W", "_", name, flags=re.ASCII)
    if slug != name:
        slug = f"{slug}_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]}"
    return f"odke_{kind}_{slug}"


def _rows(tx: Any, cypher: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = tx.run(cypher).data()
    return rows


class Neo4jConstrainer:
    """Compiles the ontology into what Neo4j enforces, and checks for what it cannot.

    `Neo4jSink.profile` says the store constrains, and the pipeline's
    double-stage warning takes that at its word. So this says exactly how much
    that covers.

    Neo4j enforces, in every edition, once `Neo4jSink.bootstrap()` has run:

    - one node per (type, key): a uniqueness constraint per entity type. It is
      what makes the sink's `MERGE` correct under concurrent writers — without
      it two writers can both miss and both create — and it is the index the
      MERGE looks up;
    - one relationship per (predicate, signature) and one `:Claim` per
      signature: the same guarantee for facts. Relationship uniqueness
      constraints need Neo4j 5.7 or later.

    Indexes, which enforce nothing: `external_id` per type; a full-text index
    over `label` and `aliases` per type, full-text being the index kind that
    covers a `LIST<STRING>` property; and `key` on the sink's `:Entity` label,
    which is how a link finds a node without knowing its type.

    Neo4j cannot enforce:

    - relationship cardinality. No constraint says "a Person has at most one
      employer". Every single-cardinality predicate gets a *check query*
      instead, marked `// odke:check <predicate>`, which returns the violators
      and changes nothing — the rule used as a check, not as inference: a
      reasoner told "at most one" concludes the two employers are one company
      (09-quality §4). Only asserted, open-ended relationships count: a denial
      is not a value, and one with `valid_to` set expired correctly (DECISIONS
      #17). The check groups by the predicate's `scope_keys`, its
      identity-bearing qualifiers and any declared `cardinality_scope`: uptime
      is single per percentile. Nothing runs these on write;
      `Neo4jSink.check()` does.
    - existence, property-type and node-key constraints. Neo4j has them in
      Enterprise Edition only, so none is emitted and `EntityType.keys` is not
      compiled — the corroborator, not the store, is where those keys are used.
    - domain and range — that an employer is a Company. The gate is where
      those are checked.

    Only what the ontology names is compiled. A type or predicate that turns up
    in a graph but not in the schema gets no constraint: the store enforces
    nothing it was not told about.
    """

    # The platform this DDL is for. A sink whose profile has the same name is
    # this compiler's other half, so configuring both is not a stage run twice.
    platform = "neo4j"

    def constrain(self, ontology: Ontology) -> DDL:
        return [*self.schema(ontology), *self.checks(ontology)]

    def schema(self, ontology: Ontology) -> list[str]:
        """Constraints and indexes, every one `IF NOT EXISTS`, in a stable order."""
        out = [
            f"CREATE INDEX odke_entity_key IF NOT EXISTS FOR (n:{_ident(ENTITY_LABEL)}) ON (n.key)",
            f"CREATE CONSTRAINT odke_claim_signature IF NOT EXISTS "
            f"FOR (c:{_ident(CLAIM_LABEL)}) REQUIRE c.signature IS UNIQUE",
        ]
        for name in sorted(ontology.types):
            label = _ident(name)
            out += [
                f"CREATE CONSTRAINT {_schema_name('key', name)} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.key IS UNIQUE",
                f"CREATE INDEX {_schema_name('external_id', name)} IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.external_id)",
                f"CREATE FULLTEXT INDEX {_schema_name('names', name)} IF NOT EXISTS "
                f"FOR (n:{label}) ON EACH [n.label, n.aliases]",
            ]
        for name in sorted(ontology.predicates):
            out.append(
                f"CREATE CONSTRAINT {_schema_name('signature', name)} IF NOT EXISTS "
                f"FOR ()-[r:{_ident(name)}]-() REQUIRE r.signature IS UNIQUE"
            )
        return out

    def checks(self, ontology: Ontology) -> list[str]:
        """One violator-returning query per single-cardinality predicate."""
        return [
            _cardinality_check(name, predicate.scope_keys)
            for name, predicate in sorted(ontology.predicates.items())
            if predicate.cardinality == "single"
        ]


def _cardinality_check(predicate: str, scope: Iterable[str]) -> str:
    scoped = ", ".join(f"r.{_ident(qualifier_property(k))}" for k in scope)
    objects = "collect(DISTINCT coalesce(o.key, o.value)) AS objects"
    lines = [
        f"{CHECK_MARKER}{predicate}",
        f"MATCH (s)-[r:{_ident(predicate)}]->(o)",
        "WHERE r.polarity = 'asserted' AND r.valid_to IS NULL",
    ]
    if scoped:
        lines += [
            f"WITH s, [{scoped}] AS scope, {objects}",
            "WHERE size(objects) > 1",
            "RETURN labels(s) AS labels, s.key AS subject, scope, objects",
        ]
    else:
        lines += [
            f"WITH s, {objects}",
            "WHERE size(objects) > 1",
            "RETURN labels(s) AS labels, s.key AS subject, objects",
        ]
    return "\n".join(lines)


__all__ = [
    "CHECK_MARKER",
    "CLAIM_LABEL",
    "ENTITY_LABEL",
    "EXTRA_HINT",
    "SHOW_INDEXES",
    "LookupQuery",
    "Neo4jConstrainer",
    "Neo4jLookup",
    "Neo4jSink",
    "Statement",
    "StoreIndexes",
    "cardinality_scope",
    "check_target",
    "entities_of",
    "id_forms",
    "is_check",
    "is_outvoted",
    "is_scoped",
    "link_row",
    "plan",
    "projections",
    "provenance_of",
    "signature_of",
    "storable",
    "store_indexes",
    "stored_entity",
    "support_from",
]
