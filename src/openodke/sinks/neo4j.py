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

import contextlib
import hashlib
import json
import re
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, time
from enum import Enum
from typing import Any, NamedTuple

from openodke._renamed import deprecated
from openodke.corroborate.lookup import Embed, embedding_text
from openodke.corroborate.merge import _aware
from openodke.corroborate.provenance import (
    CHECK,
    CONFLICT,
    DERIVED,
    NEAR_DUPLICATES,
    ONTOLOGY,
    SCORE,
    SOURCE_FORM,
    WIDEN,
)
from openodke.corroborate.resolve import block_keys
from openodke.ontology import Ontology, Predicate
from openodke.reconcile import COUNTS, reconciled, tally
from openodke.sinks.report import WriteReport
from openodke.stages import DDL, Constrainer, PlatformProfile
from openodke.tenants import TENANT, scoped, tenant_name, unscoped
from openodke.types import (
    Entity,
    EntityLink,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    Polarity,
    Resolution,
    SourceTier,
    Span,
    SpanOrigin,
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
        "retired_at",
        "evidence_doc_ids",
        "evidence_uris",
        "evidence_starts",
        "evidence_ends",
        "evidence_span_origins",
        "evidence_tiers",
        "evidence_retrieved_at",
        # The tenant a store keys the fact under (#159).
        TENANT,
    }
)
# Node property names the sink writes from `Entity` fields; an attribute or a
# projected predicate with one of these names is prefixed, never merged into.
_ENTITY_FIELDS = frozenset(
    {
        TENANT,
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


def signature_of(fact: Fact, tenant: str | None = None) -> str:
    """`Fact.signature` as one fixed-width string: the MERGE key of every fact.

    Hashed rather than serialised so the key is index-safe whatever the object
    value's repr is; the components are on the relationship as plain properties.
    With a `tenant`, the tenant is hashed with it, so two tenants' identical
    facts have two keys and never merge (#159); without one, the key is what
    it always was.
    """
    identity: Any = fact.signature if tenant is None else [tenant, *fact.signature]
    canonical = json.dumps(identity, separators=(",", ":"), ensure_ascii=False)
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


def provenance_of(fact: Fact, extracted_at: datetime, tenant: str | None = None) -> dict[str, Any]:
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

    With a `tenant`, the signature is the tenant's and `tenant` names it (#159).
    """
    evidence, sources = fact.evidence, fact.supported_by
    clocks = _one_kind([e.retrieved_at for e in evidence])
    props: dict[str, Any] = {
        "fact_id": fact.id,
        "signature": signature_of(fact, tenant),
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
        "retired_at": fact.retired_at,
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
    if tenant is not None:
        props[TENANT] = tenant
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


# The odke.* stamps a relationship keeps as JSON text, read back as the values they were.
_JSON_STAMPS = frozenset({SOURCE_FORM, CONFLICT, SCORE, DERIVED, NEAR_DUPLICATES, WIDEN, CHECK})


def stored_qualifiers(props: Mapping[str, Any]) -> dict[str, Any]:
    """A relationship's qualifiers, named as the fact named them.

    Every property provenance does not own is a qualifier; one that provenance
    does own was written as `qualifier_<name>`. An `odke.*` stamp the sink kept
    as JSON text is read back as the mapping or list it was.
    """
    out: dict[str, Any] = {}
    for name, value in props.items():
        if name in _PROVENANCE:
            continue
        own = name.removeprefix("qualifier_")
        key = own if own != name and own in _PROVENANCE else name
        value = _native(value)
        if key in _JSON_STAMPS and isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        out[key] = value
    return out


def stored_evidence(props: Mapping[str, Any]) -> tuple[Evidence, ...]:
    """A relationship's evidence lists read back, one `Evidence` per position.

    A missing uri was written `""` and a missing span `-1`. Quotes and mentions
    are not stored, so neither comes back.
    """
    docs = [str(d) for d in props.get("evidence_doc_ids") or ()]

    def column(name: str, default: Any) -> list[Any]:
        values = list(props.get(name) or ())
        return (values + [default] * len(docs))[: len(docs)]

    rows = zip(
        docs,
        column("evidence_uris", ""),
        column("evidence_starts", -1),
        column("evidence_ends", -1),
        column("evidence_span_origins", SpanOrigin.CITED.value),
        column("evidence_tiers", SourceTier.UNVERIFIED.value),
        column("evidence_retrieved_at", None),
        strict=True,
    )
    return tuple(
        Evidence(
            doc_id=doc,
            span=Span(doc_id=doc, start=int(start), end=int(end)) if min(start, end) >= 0 else None,
            span_origin=SpanOrigin(origin),
            uri=uri or None,
            tier=SourceTier(tier),
            **({"retrieved_at": _native(at)} if at is not None else {}),
        )
        for doc, uri, start, end, origin, tier, at in rows
    )


def stored_fact(fact: Fact, props: Mapping[str, Any]) -> Fact:
    """The fact stored under `fact`'s signature: the claim as `fact` states it, receipts as stored.

    Evidence, support list and count, qualifiers, extractor, verdict,
    confidence, both clocks and the id are the relationship's. The subject,
    the object and the identity-bearing qualifiers are `fact`'s, which share
    the signature: a value or a qualifier stored as JSON text would otherwise
    read back as a different claim.
    """
    qualifiers = stored_qualifiers(props)
    for key in fact.identity_keys:
        if key in fact.qualifiers:
            qualifiers[key] = fact.qualifiers[key]
    update: dict[str, Any] = {
        "id": str(props.get("fact_id") or fact.id),
        "evidence": stored_evidence(props),
        "supported_by": support_from(props),
        "support": int(props.get("support") or 0),
        "qualifiers": qualifiers,
        "extractor": str(props.get("extractor") or fact.extractor),
        "verdict": GroundingVerdict(props.get("verdict") or GroundingVerdict.UNCHECKED.value),
        "confidence": float(props.get("confidence") or 0.0),
        "valid_from": _native(props.get("valid_from")),
        "valid_to": _native(props.get("valid_to")),
        "retired_at": _native(props.get("retired_at")),
    }
    return fact.model_copy(update=update)


def relationship_fact(row: Mapping[str, Any]) -> Fact:
    """A fact relationship read back with its two ends, as the reconciler reads it.

    `row` holds the relationship's `predicate` and `props`, and each end's
    `key` and `labels`; a `:Claim` object also its `value`. The type of an end
    is its label other than `Entity` and `Claim`. The relationship's own
    receipts are `stored_fact`'s.
    """
    subject = Entity(key=str(row["subject_key"]), type=_end_type(row["subject_labels"]))
    props = row["props"]
    claim = CLAIM_LABEL in (row.get("object_labels") or ())
    obj = (
        None if claim else Entity(key=str(row["object_key"]), type=_end_type(row["object_labels"]))
    )
    stated = Fact(
        subject=subject,
        predicate=str(row["predicate"]),
        object_entity=obj,
        object_value=_native(row.get("value")) if claim else None,
        polarity=Polarity(props.get("polarity") or Polarity.ASSERTED.value),
        identity_keys=tuple(str(k) for k in props.get("identity_keys") or ()),
    )
    return stored_fact(stated, props)


def _end_type(labels: Iterable[str] | None) -> str:
    found = sorted(set(labels or ()) - {ENTITY_LABEL, CLAIM_LABEL})
    return found[0] if found else ENTITY_LABEL


def _entity_row(entity: Entity, tenant: str | None = None) -> dict[str, Any]:
    resolution = entity.resolution
    attributes = {attribute_property(k): storable(v) for k, v in entity.attributes.items()}
    if tenant is not None:
        attributes[TENANT] = tenant
    return {
        "key": scoped(entity.key, tenant),
        "label": entity.label,
        "aliases": list(entity.aliases),
        "external_id": entity.external_id,
        "resolution_method": resolution.method if resolution else None,
        "resolution_score": resolution.score if resolution else None,
        "resolution_linker": resolution.linker if resolution else None,
        "attributes": attributes,
    }


def is_scoped(fact: Fact) -> bool:
    """True when the fact carries a value for one of its identity-bearing qualifiers."""
    return any(k in fact.qualifiers for k in fact.identity_keys)


def is_retired(fact: Fact) -> bool:
    """True when the reconciler took the fact's last source away (#116).

    A retired fact stays a record, as a voted-down one does, and is never
    the value: it is not projected, and no plain RDF triple asserts it.
    """
    return fact.retired_at is not None


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


def plan(
    kg: KnowledgeGraph, *, ontology: Ontology | None = None, tenant: str | None = None
) -> list[Statement]:
    """The graph as `UNWIND … MERGE` statements, in write order. Pure, so it is testable.

    Order matters: nodes before the relationships that `MATCH` them, links last.
    Within each kind the groups are sorted, so one graph always compiles to the
    same statements. With a `tenant`, every key and signature is the tenant's
    (`openodke.tenants`) and every node and fact names it; the Cypher is the
    same.
    """
    statements: list[Statement] = []

    def key(entity_key: str) -> str:
        return scoped(entity_key, tenant)

    by_label: dict[str, list[dict[str, Any]]] = {}
    for (label, _), entity in sorted(entities_of(kg).items()):
        by_label.setdefault(label, []).append(_entity_row(entity, tenant))
    for label, rows in sorted(by_label.items()):
        statements.append(Statement(_entity_cypher(label), rows, "entity", (label,)))

    edges: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    claims: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for fact in kg.facts:
        props = provenance_of(fact, kg.created_at, tenant)
        subject = key(fact.subject.key)
        row = {"subject_key": subject, "signature": props["signature"], "props": props}
        if fact.object_entity is not None:
            group = (fact.subject.type, fact.predicate, fact.object_entity.type)
            edges.setdefault(group, []).append({**row, "object_key": key(fact.object_entity.key)})
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
        rows = [{**row, "subject_key": key(row["subject_key"])} for row in rows]
        statements.append(Statement(cypher, rows, "projection", (subject_type, predicate)))

    links: dict[str, list[dict[str, Any]]] = {}
    for link in kg.links:
        row = link_row(link)
        row.update(source_key=key(link.source_key), target_key=key(link.target_key))
        links.setdefault(link.kind.value.upper(), []).append(row)
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
    sources back it, or one the reconciler retired. A single-valued predicate
    projects its best-supported claim; a multi-valued one, when the ontology
    says so, the list in that order.
    """
    grouped: dict[tuple[str, str, str], list[Fact]] = {}
    for fact in kg.facts:
        if (
            fact.object_entity is not None
            or fact.object_value is None
            or fact.polarity is not Polarity.ASSERTED
            or is_scoped(fact)
            or is_outvoted(fact)
            or is_retired(fact)
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


# The reconciler's statements (#116). Facts citing a document are found through
# the per-predicate evidence indexes, and rewritten or deleted by signature.
_TYPES = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS name"
_LINK_TYPES = ("SAME_AS", "SIMILAR", "DIFFERENT")


def _citing(tenant: str | None) -> str:
    """The facts citing a document, through the evidence indexes: the tenant's alone (#159)."""
    within = "r.tenant = $tenant" if tenant is not None else "r.tenant IS NULL"
    return (
        "UNWIND $indexes AS idx\n"
        "CALL db.index.fulltext.queryRelationships(idx, $terms) YIELD relationship AS r\n"
        f"WITH r, startNode(r) AS s, endNode(r) AS o WHERE {within}\n"
        "RETURN elementId(r) AS id, type(r) AS predicate, properties(r) AS props,\n"
        "       s.key AS subject_key, labels(s) AS subject_labels,\n"
        "       o.key AS object_key, labels(o) AS object_labels, o.value AS value"
    )


# What a retraction changes on a relationship; everything else stays as written.
_RETRACTED = (
    "support",
    "support_sources",
    "support_doc_ids",
    "support_doc_sources",
    "support_tiers",
    "support_retrieved_at",
    "retrieved_at",
    "retired_at",
    "evidence_doc_ids",
    "evidence_uris",
    "evidence_starts",
    "evidence_ends",
    "evidence_span_origins",
    "evidence_tiers",
    "evidence_retrieved_at",
)


def checked_cypher(predicate: str) -> str:
    """The facts on `predicate` one ontology last checked: `$fingerprint`, through its index.

    `odke.ontology` is the fact's qualifier (#163), a property of the
    relationship under its own name, so it is quoted. `bootstrap()` gives each
    predicate a range index on it, and the query names that index, so it is
    read through it or not at all: never a scan.
    """
    rel, prop = _ident(predicate), _ident(ONTOLOGY)
    return (
        f"MATCH (s)-[r:{rel}]->(o) USING INDEX r:{rel}({prop})\n"
        f"WHERE r.{prop} = $fingerprint\n"
        "RETURN type(r) AS predicate, properties(r) AS props,\n"
        "       s.key AS subject_key, labels(s) AS subject_labels,\n"
        "       o.key AS object_key, labels(o) AS object_labels, o.value AS value"
    )


def _retract_cypher(predicate: str) -> str:
    # Found again by its MERGE key, through the uniqueness constraint's index.
    return (
        "UNWIND $rows AS row\n"
        f"MATCH ()-[r:{_ident(predicate)} {{signature: row.signature}}]->()\n"
        "SET r += row.props"
    )


def _delete_cypher(predicate: str) -> str:
    return (
        "UNWIND $rows AS row\n"
        f"MATCH ()-[r:{_ident(predicate)} {{signature: row.signature}}]->(o)\n"
        "DELETE r\n"
        f"WITH o WHERE o:{_ident(CLAIM_LABEL)}\n"
        "DETACH DELETE o"
    )


def _batches(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def transactions(
    statements: Iterable[Statement], size: int
) -> Iterator[list[tuple[Statement, list[dict[str, Any]]]]]:
    """The statements' rows packed into transactions of at most `size` rows, in write order.

    A transaction holds the end of one statement and the start of the next, so
    a graph of many small groups is a few transactions, not one per group.
    """
    held: list[tuple[Statement, list[dict[str, Any]]]] = []
    count = 0
    for statement in statements:
        start = 0
        while start < len(statement.rows):
            take = min(size - count, len(statement.rows) - start)
            held.append((statement, list(statement.rows[start : start + take])))
            count += take
            start += take
            if count >= size:
                yield held
                held, count = [], 0
    if held:
        yield held


# --------------------------------------------------------------------------- #
# The sink
# --------------------------------------------------------------------------- #


def _connect(uri: str, auth: Any) -> Any:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    return GraphDatabase.driver(uri, auth=auth)


# What `write()` adds to each statement it runs: how many rows reached the MERGE.
COUNTED = "\nRETURN count(*) AS reached"
# Which statements' rows the write report counts, and as what.
_COUNTED_AS = {"entity": "entities", "edge": "facts", "claim": "facts", "link": "links"}


def _unwind(
    tx: Any, pieces: Sequence[tuple[str, list[dict[str, Any]]]]
) -> list[tuple[int, int, int]]:
    """Each piece in one transaction: rows that reached its MERGE, nodes and relationships made."""
    out: list[tuple[int, int, int]] = []
    for cypher, rows in pieces:
        result = tx.run(cypher + COUNTED, rows=rows)
        found = result.data()
        counters = getattr(result.consume(), "counters", None)
        out.append(
            (
                int(found[0]["reached"]) if found else 0,
                int(getattr(counters, "nodes_created", 0)),
                int(getattr(counters, "relationships_created", 0)),
            )
        )
    return out


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
    the same fact, and its `:Claim` and projected value would stay.
    `retract()` is the reconciler's way to do it (#116).

    A single-valued predicate with two objects is written as two edges, not
    replaced: the store holds the conflict and a check query reports it
    (09-quality §4) — picking a winner at write time would destroy the
    evidence for the loser.

    `stored(facts)` reads what the store already holds under a batch's
    signatures, so the corroborator can merge each fact with its stored twin
    before the write and its support list grows rather than being replaced
    (#153). The Validator does this by default.

    Rows go in transactions of at most `batch_size` rows, packed across
    statements in write order (`transactions`), one managed transaction each,
    so a failed one rolls back whole and never half-writes. Earlier ones stay
    committed; because every statement is a MERGE, recovering is running the
    write again, and a rerun of the same graph makes nothing new.

    `writes` is the sink's write report, run on across its writes (#159): per
    kind (entities, facts, links), the rows `written` new, `merged` into a
    node or relationship the store held, and `skipped` because an end was not
    there to `MATCH`, and the transactions committed.

    `tenant` keys the sink's writes and reads apart from every other tenant's
    in the same store (`openodke.tenants`, DECISIONS #44): its entities are
    keyed `<tenant>/<key>`, its facts' signatures are hashed with the tenant,
    and `stored()`, `lookup()`, `retract()` and `check()` see that tenant's
    alone. A sink with no tenant sees only what no tenant wrote.
    `scoped(tenant)` is the same connection scoped to one.

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
        tenant: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.tenant = tenant_name(tenant)
        if driver is None:
            if uri is None:
                raise ValueError("Neo4jSink needs a uri (with auth) or a driver")
            driver = _connect(uri, auth)
        self._driver = driver
        self.database = database
        self.batch_size = batch_size
        self.writes = WriteReport()
        # Decides whether a projected property is one value or a list.
        self.ontology = ontology
        # What `stored()` reads through; made on first use, and again after a bootstrap.
        self._reader: Neo4jLookup | None = None
        if ontology is not None:
            ontology.warn_if_unreviewed("Neo4jSink will shape its writes")

    def statements(self, kg: KnowledgeGraph) -> list[Statement]:
        """What `write()` would run, without running it: each statement then counts its rows."""
        return plan(kg, ontology=self.ontology, tenant=self.tenant)

    def write(self, kg: KnowledgeGraph) -> None:
        with self._driver.session(**self._session_config()) as session:
            for pieces in transactions(self.statements(kg), self.batch_size):
                counted = session.execute_write(
                    _unwind, [(statement.cypher, rows) for statement, rows in pieces]
                )
                # Counted once the transaction has committed.
                self.writes.transactions += 1
                for (statement, rows), (reached, nodes, relationships) in zip(
                    pieces, counted, strict=True
                ):
                    kind = _COUNTED_AS.get(statement.kind)
                    if kind is None:
                        continue
                    made = nodes if statement.kind == "entity" else relationships
                    self.writes.add(
                        kind,
                        written=made,
                        merged=max(0, reached - made),
                        skipped=max(0, len(rows) - reached),
                    )

    def scoped(self, tenant: str | None) -> Neo4jSink:
        """This sink's connection, reading and writing `tenant` alone (#159).

        The sink itself when it is that tenant's already. A sink scoped to
        another tenant is refused: one sink never writes for two. The new sink
        shares the driver, so closing either closes both.
        """
        tenant = tenant_name(tenant)
        if tenant == self.tenant:
            return self
        if self.tenant is not None:
            raise ValueError(f"this sink writes tenant {self.tenant!r}, not {tenant!r}")
        return Neo4jSink(
            driver=self._driver,
            database=self.database,
            batch_size=self.batch_size,
            ontology=self.ontology,
            tenant=tenant,
        )

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
            self._reader = None
        return ddl

    def check(
        self, ontology: Ontology, *, constrainer: Constrainer | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Run the cardinality checks; the violators, by predicate.

        A check and never a repair (09-quality §4): nothing here deletes, merges
        or picks a winner. What to do with a Person holding two employers is a
        person's call, and the rows say which ones to look at.

        A subject node is one tenant's, so a check counts within a tenant. A
        sink with a tenant returns that tenant's violators, keyed as it keys
        them; one without returns the whole store's, keyed as stored.
        """
        compiled = (constrainer or Neo4jConstrainer()).constrain(ontology)
        found: dict[str, list[dict[str, Any]]] = {}
        with self._driver.session(**self._session_config()) as session:
            for statement in (s for s in compiled if is_check(s)):
                rows = self._own(session.execute_read(_rows, statement))
                if rows:
                    found[check_target(statement)] = rows
        return found

    def _own(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A check's rows whose subject is this sink's tenant's, with the tenant's keys."""
        if self.tenant is None:
            return rows
        prefix = scoped("", self.tenant)
        out = []
        for row in rows:
            if not str(row.get("subject", "")).startswith(prefix):
                continue
            objects = [
                unscoped(o, self.tenant) if isinstance(o, str) else o for o in row["objects"]
            ]
            subject = unscoped(row["subject"], self.tenant)
            out.append({**row, "subject": subject, "objects": objects})
        return out

    def stored(self, facts: Sequence[Fact]) -> dict[tuple[Any, ...], Fact]:
        """What the store holds under these facts' signatures, as `Neo4jLookup.stored` reads it.

        This makes the sink a `FactLookup`: the Validator hands it to the
        corroborator, which merges each incoming fact with the one stored under
        its signature before anything is written (#153). One read transaction,
        through the signature indexes `bootstrap()` creates.
        """
        return self._lookup().stored(facts)

    def retract(
        self, doc_ids: Collection[str], *, at: datetime, hard: bool = False
    ) -> dict[str, int]:
        """Take these documents out of every fact they back, in one write transaction (#116).

        The facts are found through the per-predicate full-text index over
        `evidence_doc_ids` that `bootstrap()` creates, never by a scan, and a
        relationship type with no such index is not read and warns. Each is
        read back, retracted by `openodke.reconcile.retract`, and its evidence,
        support list, `support` and `retired_at` set again, found by its
        signature through the uniqueness constraint's index; a derived fact
        whose parent is retired is retired with it. With `hard`, a fact left
        with no source is deleted, with its `:Claim`. A value projected from a
        claim that lost support is projected again from the claims left, and
        removed when none is. Returns the counts `openodke.reconcile.COUNTS`
        names; the same documents retracted twice change nothing. Only the
        sink's tenant's facts are touched: another tenant's text with the same
        id is another tenant's (#159).
        """
        docs = sorted(set(doc_ids))
        reader = self._lookup()
        indexes = reader.indexes()
        with self._driver.session(**self._session_config()) as session:
            types = [row["name"] for row in session.run(_TYPES).data()]
            for name in sorted(set(types) - set(indexes.evidence) - set(_LINK_TYPES)):
                reader._unindexed(name, "evidence_doc_ids")
            names = sorted(indexes.evidence.values())
            if not docs or not names:
                return dict.fromkeys(COUNTS, 0)
            counts: dict[str, int] = session.execute_write(self._retract, docs, names, at, hard)
        return counts

    def checked_under(self, ontology: Ontology | str) -> list[Fact]:
        """The facts the store holds that `ontology`, or the fingerprint given, last checked.

        What a reader filters by (#163): after a breaking change, the facts
        the old ontology checked are the ones to validate again. One read
        transaction, per relationship type through the `odke.ontology` index
        `bootstrap()` creates, never by a scan: a type with no such index is
        not read, and warns.
        """
        wanted = ontology if isinstance(ontology, str) else ontology.fingerprint
        reader = self._lookup()
        indexed = reader.indexes().ontologies
        with self._driver.session(**self._session_config()) as session:
            types = [row["name"] for row in session.run(_TYPES).data()]
            for name in sorted(set(types) - indexed - set(_LINK_TYPES)):
                reader._unindexed(name, ONTOLOGY)
            rows = session.execute_read(_checked, sorted(indexed & set(types)), wanted)
        return [relationship_fact(row) for row in rows]

    def _retract(
        self, tx: Any, docs: list[str], names: list[str], at: datetime, hard: bool
    ) -> dict[str, int]:
        terms = " OR ".join(_phrase(doc) for doc in docs)
        asked = {"indexes": names, "terms": terms, "tenant": self.tenant}
        rows = tx.run(_citing(self.tenant), asked).data()
        before = [relationship_fact(row) for row in rows]
        after = reconciled(before, set(docs), at)
        updates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        deletes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        claims: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row, old, new in zip(rows, before, after, strict=True):
            if new is old:
                continue
            # The stored key, not one computed again: a value read back may not repr the same.
            signature = row["props"]["signature"]
            if hard and new.retired_at is not None:
                deletes[new.predicate].append({"signature": signature})
            else:
                props = provenance_of(new, at)
                changed = {key: props[key] for key in _RETRACTED}
                near = qualifier_property(NEAR_DUPLICATES)
                changed[near] = props.get(near)
                updates[new.predicate].append({"signature": signature, "props": changed})
            if new.object_entity is None:
                claims[(new.subject.type, new.predicate)].add(new.subject.key)
        for predicate, batch in sorted(updates.items()):
            tx.run(_retract_cypher(predicate), rows=batch).consume()
        for predicate, batch in sorted(deletes.items()):
            tx.run(_delete_cypher(predicate), rows=batch).consume()
        self._project_again(tx, claims)
        return tally(before, after, hard=hard)

    def _project_again(self, tx: Any, claims: Mapping[tuple[str, str], set[str]]) -> None:
        """The projected value of each subject and predicate, from the claims it has left."""
        for (subject_type, predicate), keys in sorted(claims.items()):
            cypher = (
                "UNWIND $rows AS row\n"
                f"MATCH (s:{_ident(subject_type)} {{key: row.key}})"
                f"-[r:{_ident(predicate)}]->(c:{_ident(CLAIM_LABEL)})\n"
                "RETURN row.key AS subject_key, labels(s) AS subject_labels, "
                "properties(r) AS props, labels(c) AS object_labels, c.value AS value"
            )
            found = tx.run(cypher, rows=[{"key": key} for key in sorted(keys)]).data()
            left = [
                relationship_fact({**row, "predicate": predicate, "object_key": None})
                for row in found
            ]
            graph = KnowledgeGraph(facts=tuple(left))
            values = {
                row["subject_key"]: row["value"]
                for row in projections(graph, self.ontology).get((subject_type, predicate), ())
            }
            rows = [{"subject_key": key, "value": values.get(key)} for key in sorted(keys)]
            tx.run(_projection_cypher(subject_type, predicate), rows=rows).consume()

    def _lookup(self) -> Neo4jLookup:
        if self._reader is None:
            self._reader = self.lookup()
        return self._reader

    def lookup(self, **options: Any) -> Neo4jLookup:
        """A `Neo4jLookup` on this sink's connection, database and ontology.

        It shares the driver, so closing the sink closes it. `options` are
        `Neo4jLookup`'s: `tenant` (the sink's, unless given), `limit`, `embed`,
        `vector_index`, `vector_k`.
        """
        options.setdefault("ontology", self.ontology)
        options.setdefault("tenant", self.tenant)
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
SHOW_INDEXES = "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties"


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
    # Relationship types with a range index on `signature`: the uniqueness
    # constraint's, through which a stored fact is found by its MERGE key.
    signatures: frozenset[str] = frozenset()
    # Relationship type -> a full-text index over `evidence_doc_ids`, through
    # which the reconciler finds the facts a document backs.
    evidence: Mapping[str, str] = {}
    # Relationship types with a range index on `odke.ontology`, through which a
    # reader finds the facts one ontology checked (#163).
    ontologies: frozenset[str] = frozenset()


def store_indexes(rows: Iterable[Mapping[str, Any]]) -> StoreIndexes:
    """`SHOW INDEXES` rows read as what a lookup can use. Bootstrap's own names win ties."""
    keys: set[str] = set()
    ids: set[str] = set()
    names: dict[str, str] = {}
    signatures: set[str] = set()
    evidence: dict[str, str] = {}
    ontologies: set[str] = set()
    entity_key = False
    for row in sorted(rows, key=lambda r: str(r.get("name"))):
        labels = list(row.get("labelsOrTypes") or ())
        props = list(row.get("properties") or ())
        kind = row.get("type")
        if row.get("entityType") == "RELATIONSHIP":
            if kind == "RANGE" and len(labels) == 1 and props == ["signature"]:
                signatures.add(labels[0])
            elif kind == "RANGE" and len(labels) == 1 and props == [ONTOLOGY]:
                ontologies.add(labels[0])
            elif kind == "FULLTEXT" and props == ["evidence_doc_ids"]:
                for label in labels:
                    own = _schema_name("evidence", label)
                    if label not in evidence or row.get("name") == own:
                        evidence[label] = str(row["name"])
        elif kind == "RANGE" and len(labels) == 1 and props == ["key"]:
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
    return StoreIndexes(
        frozenset(keys),
        entity_key,
        frozenset(ids),
        names,
        frozenset(signatures),
        evidence,
        frozenset(ontologies),
    )


class LookupQuery(NamedTuple):
    """One read statement of a lookup: its Cypher, its parameters, and what it reads."""

    cypher: str
    params: dict[str, Any]
    # The entity type it is scoped to, and how it finds them: key | id | names |
    # vector; or the predicate, found by its signature.
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
    props: Mapping[str, Any],
    entity_type: str,
    *,
    ontology: Ontology | None = None,
    tenant: str | None = None,
) -> Entity:
    """A node the sink wrote, read back as the `Entity` that wrote it.

    The inverse of the sink's entity row: `label`, `aliases`, `external_id` and
    `resolution_*` are fields, an attribute the sink prefixed gets its name
    back, and every other property is an attribute, except a value the ontology
    says was projected from a fact. Written again, it sets every property to
    what it already holds. A tenant's node comes back under the key its
    tenant gave it, and `tenant` is the sink's, never an attribute.
    """
    rest = {name: _native(value) for name, value in props.items()}
    rest.pop(TENANT, None)
    method = rest.pop("resolution_method", None)
    score = rest.pop("resolution_score", None)
    linker = rest.pop("resolution_linker", None)
    resolution = (
        Resolution(method=method, score=score, linker=linker)
        if method in ("caller", "external_id", "linker")
        else None
    )
    key = unscoped(str(rest.pop("key")), tenant)
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
    members: Sequence[Entity],
    blocks: _Blocks,
    vectors: Mapping[str, list[float]] | None,
    tenant: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """One type's distinct block keys as `UNWIND` rows, by how they are looked up.

    A block key two entities share is one row, asked once; the rows carry the
    block's id, and the reply is handed to every entity that asked. A key is
    asked for as the tenant's (`openodke.tenants`), so the index finds it.
    """
    rows: dict[str, list[dict[str, Any]]] = {"key": [], "id": [], "names": [], "vector": []}
    for entity in members:
        kind, probe = entity.type, entity.key
        at, new = blocks.ask(("key", kind, entity.key), probe)
        if new:
            rows["key"].append({"block": at, "value": scoped(entity.key, tenant)})
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

    `tenant` scopes every read to one tenant's nodes and facts (#159,
    DECISIONS #44). A key is asked for as the tenant's key, so the uniqueness
    index finds the tenant's node and no other, and a fact by the tenant's
    signature. An external id, a name or a vector is found through its index
    and kept when `tenant` is the node's, so a full-text `limit` still counts
    every tenant's hits. A lookup with no tenant reads only what no tenant
    wrote.
    """

    def __init__(
        self,
        uri: str | None = None,
        auth: Any = None,
        *,
        database: str | None = None,
        driver: Any = None,
        tenant: str | None = None,
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
        self.tenant = tenant_name(tenant)
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
                    entity = stored_entity(
                        row["node"], query.type, ontology=self.ontology, tenant=self.tenant
                    )
                    for probe in probes[row["block"]]:
                        found[probe].setdefault(entity.key, entity)
        return {key: list(stored.values()) for key, stored in found.items()}

    def statements(self, entities: Sequence[Entity]) -> list[LookupQuery]:
        """What `candidates()` would read, without reading it (the indexes are read once)."""
        return self._plan(list(entities))[0]

    def stored(self, facts: Sequence[Fact]) -> dict[tuple[Any, ...], Fact]:
        """The facts the store holds under these facts' signatures, keyed by `Fact.signature`.

        Per predicate, one `UNWIND` of the batch's distinct signature keys
        through the relationship index the predicate's uniqueness constraint
        brings, and every predicate in one read transaction. Each relationship
        found is read back by `stored_fact`: the claim as the batch states it,
        the receipts as the store holds them. A predicate with no index is not
        read, and warns once, as a type with no index does.
        """
        by_key: dict[str, list[Fact]] = defaultdict(list)
        for fact in facts:
            by_key[signature_of(fact, self.tenant)].append(fact)
        queries = self.fact_statements(facts)
        if not queries:
            return {}
        with self._driver.session(**self._session_config()) as session:
            results = session.execute_read(_read_all, queries)
        self.stats["transactions"] += 1
        self.stats["statements"] += len(queries)
        out: dict[tuple[Any, ...], Fact] = {}
        for rows in results:
            for row in rows:
                for fact in by_key.get(row["signature"], ()):
                    out.setdefault(fact.signature, stored_fact(fact, row["props"]))
        return out

    def fact_statements(self, facts: Sequence[Fact]) -> list[LookupQuery]:
        """What `stored()` would read, without reading it (the indexes are read once)."""
        indexes = self.indexes()
        by_predicate: dict[str, set[str]] = defaultdict(set)
        for fact in facts:
            by_predicate[fact.predicate].add(signature_of(fact, self.tenant))
        queries: list[LookupQuery] = []
        for predicate in sorted(by_predicate):
            if predicate not in indexes.signatures:
                self._unindexed(predicate, "signature")
                continue
            cypher = (
                "UNWIND $rows AS row\n"
                f"MATCH ()-[r:{_ident(predicate)} {{signature: row.signature}}]->()\n"
                "RETURN row.signature AS signature, properties(r) AS props"
            )
            rows = [{"signature": key} for key in sorted(by_predicate[predicate])]
            queries.append(LookupQuery(cypher, {"rows": rows}, predicate, "signature"))
        return queries

    def _plan(self, entities: list[Entity]) -> tuple[list[LookupQuery], list[list[str]]]:
        indexes = self.indexes()
        blocks = _Blocks()
        vectors = self._vectors(entities) if self.vector_index is not None else None
        by_type: dict[str, list[Entity]] = {}
        for entity in entities:
            by_type.setdefault(entity.type, []).append(entity)
        queries: list[LookupQuery] = []
        for entity_type in sorted(by_type):
            rows = _block_rows(by_type[entity_type], blocks, vectors, self.tenant)
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
            return [f"n.{_ident(TENANT)} IS NULL"]
        return [f"n.{_ident(TENANT)} = $tenant"]

    def scoped(self, tenant: str | None) -> Neo4jLookup:
        """This lookup's connection, reading `tenant` alone; itself when it does already.

        A lookup scoped to another tenant is refused. The new one shares the
        driver and never closes it.
        """
        tenant = tenant_name(tenant)
        if tenant == self.tenant:
            return self
        if self.tenant is not None:
            raise ValueError(f"this lookup reads tenant {self.tenant!r}, not {tenant!r}")
        return Neo4jLookup(
            driver=self._driver,
            database=self.database,
            tenant=tenant,
            limit=self.limit,
            embed=self.embed,
            vector_index=self.vector_index,
            vector_k=self.vector_k,
            ontology=self.ontology,
        )

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
            "scan; Neo4jSink.bootstrap(ontology) creates the index for each type and "
            "predicate in the ontology",
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
    deprecated("openodke.sinks.neo4j.cardinality_scope(predicate)", "predicate.scope_keys")
    return predicate.scope_keys


def _schema_name(kind: str, name: str) -> str:
    # Constraint and index names must be plain identifiers. A squashed name
    # carries a hash of the original so `A-B` and `A_B` stay distinct: an
    # existing name turns `IF NOT EXISTS` into a silent no-op.
    slug = re.sub(r"\W", "_", name, flags=re.ASCII)
    if slug != name:
        slug = f"{slug}_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]}"
    return f"odke_{kind}_{slug}"


def _checked(tx: Any, predicates: list[str], fingerprint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predicate in predicates:
        rows.extend(tx.run(checked_cypher(predicate), {"fingerprint": fingerprint}).data())
    return rows


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
    covers a `LIST<STRING>` property; `key` on the sink's `:Entity` label,
    which is how a link finds a node without knowing its type; and a
    full-text index over `evidence_doc_ids` per predicate, with the keyword
    analyzer so a document id is one exact term, which is how the reconciler
    finds every fact a document backs without a scan (#116); and `odke.ontology`
    per predicate, which is how a reader finds the facts one ontology checked
    (#163).

    Neo4j cannot enforce:

    - relationship cardinality. No constraint says "a Person has at most one
      employer". Every single-cardinality predicate gets a *check query*
      instead, marked `// odke:check <predicate>`, which returns the violators
      and changes nothing — the rule used as a check, not as inference: a
      reasoner told "at most one" concludes the two employers are one company
      (09-quality §4). Only asserted, open-ended relationships count: a denial
      is not a value, one with `valid_to` set expired correctly (DECISIONS
      #17), and a retired one lost its evidence (#116). The check groups by
      the predicate's `scope_keys`, its identity-bearing qualifiers and any
      declared `cardinality_scope`: uptime is single per percentile. Nothing runs these on write;
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
            out += [
                f"CREATE CONSTRAINT {_schema_name('signature', name)} IF NOT EXISTS "
                f"FOR ()-[r:{_ident(name)}]-() REQUIRE r.signature IS UNIQUE",
                f"CREATE FULLTEXT INDEX {_schema_name('evidence', name)} IF NOT EXISTS "
                f"FOR ()-[r:{_ident(name)}]-() ON EACH [r.evidence_doc_ids] "
                "OPTIONS {indexConfig: {`fulltext.analyzer`: 'keyword'}}",
                f"CREATE INDEX {_schema_name('ontology', name)} IF NOT EXISTS "
                f"FOR ()-[r:{_ident(name)}]-() ON (r.{_ident(ONTOLOGY)})",
            ]
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
        "WHERE r.polarity = 'asserted' AND r.valid_to IS NULL AND r.retired_at IS NULL",
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
    "COUNTED",
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
    "checked_cypher",
    "entities_of",
    "id_forms",
    "is_check",
    "is_outvoted",
    "is_retired",
    "is_scoped",
    "link_row",
    "plan",
    "projections",
    "provenance_of",
    "relationship_fact",
    "signature_of",
    "storable",
    "store_indexes",
    "stored_entity",
    "transactions",
    "stored_evidence",
    "stored_fact",
    "stored_qualifiers",
    "support_from",
]
