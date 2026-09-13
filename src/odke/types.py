"""The vocabulary every stage of the pipeline speaks.

These types are the seam between the five stages. An extractor emits `Fact`s, a
grounder accepts and returns `Fact`s, a sink consumes a `KnowledgeGraph`. Nothing
downstream of extraction knows whether a fact came from a regex over a CSV column
or from a language model reading prose, which is what makes the hybrid extractor
of the ODKE+ paper expressible without a second code path.

Everything here is deterministic and provider-free on purpose: `import odke` must
work with no model provider, no database driver and no network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Frozen(BaseModel):
    """Immutable, extra-rejecting base.

    Facts get passed through four stages and merged across sources; a stage that
    mutated one in place would make provenance a lie. Stages return new objects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Sources and evidence
# --------------------------------------------------------------------------- #


class SourceTier(StrEnum):
    """How much a source is trusted when two of them disagree.

    The corroborator resolves conflicts on freshness *and* trust; without a tier
    a stale hand-curated record loses to a fresh scrape, which is backwards.
    """

    CURATED = "curated"
    AUTHORITATIVE = "authoritative"
    COMMUNITY = "community"
    UNVERIFIED = "unverified"

    @property
    def weight(self) -> float:
        return {"curated": 1.0, "authoritative": 0.8, "community": 0.5, "unverified": 0.2}[
            self.value
        ]


class Document(Frozen):
    """One unit of input text plus everything needed to cite it later."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    text: str
    uri: str | None = None
    title: str | None = None
    # "structured" and "semi_structured" inputs are eligible for the pattern
    # extractor, which is cheaper and exact; "unstructured" needs a model.
    modality: Literal["unstructured", "semi_structured", "structured"] = "unstructured"
    tier: SourceTier = SourceTier.UNVERIFIED
    retrieved_at: datetime = Field(default_factory=_utcnow)
    # Survives loading and chunking so a sink can write back whatever the caller
    # keyed their own records on.
    metadata: dict[str, Any] = Field(default_factory=dict)


class Span(Frozen):
    """A half-open character range into `Document.text`.

    Character offsets rather than a quoted string: a quote can be paraphrased by
    a model and still look plausible, while an offset either resolves to the
    claimed text or does not. `resolve()` is what the grounder checks.
    """

    doc_id: str
    start: int
    end: int
    quote: str | None = None

    def resolve(self, doc: Document) -> str:
        return doc.text[self.start : self.end]

    def is_faithful(self, doc: Document) -> bool:
        """True when the recorded quote is really what sits at those offsets."""
        if self.quote is None:
            return True
        return self.resolve(doc) == self.quote


class Evidence(Frozen):
    """Why we believe a fact: a document, and where in it."""

    doc_id: str
    span: Span | None = None
    uri: str | None = None
    tier: SourceTier = SourceTier.UNVERIFIED
    retrieved_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


class GroundingVerdict(StrEnum):
    """What the grounder concluded about a candidate fact."""

    UNCHECKED = "unchecked"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_FOUND = "not_found"


class Polarity(StrEnum):
    """Whether the source asserts the claim, denies it, or qualifies it.

    A graph holding only positive assertions cannot answer "does X sell customer
    data?" when the source says it does not. Six percent of a hand-annotated
    corpus was negative or partial, so this is a field rather than a qualifier —
    and it is part of `Fact.signature`, because a denial and its assertion are
    opposite claims, not one claim told twice.
    """

    ASSERTED = "asserted"
    DENIED = "denied"
    # "Only for EU customers", "in most cases": true with a scope the source
    # states and the triple cannot carry.
    PARTIAL = "partial"


class Entity(Frozen):
    """A node. `key` is the identity the corroborator merges on."""

    key: str
    type: str
    label: str | None = None
    aliases: tuple[str, ...] = ()
    # A resolved identifier in some external authority (Wikidata Q-id, an
    # internal customer id) when entity linking found one.
    external_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Fact(Frozen):
    """One (subject, predicate, object) assertion with its receipts.

    `object_entity` and `object_value` are exclusive: a predicate whose range is
    another type produces an edge, a predicate whose range is a literal produces
    a property. Both live in one class because both need identical provenance,
    grounding and corroboration, and splitting them duplicated every stage.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    subject: Entity
    predicate: str
    object_entity: Entity | None = None
    object_value: Any = None
    polarity: Polarity = Polarity.ASSERTED
    # Predicate-scoped modifiers from the ontology: start_time, end_time, rank.
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    # Which qualifier keys make this a different claim rather than the same one
    # told imprecisely — `percentile`, `tier`, not `start_time`. Stamped by the
    # extractor from `Ontology.identity_keys()`, so the fact never needs the
    # ontology in hand to know its own identity.
    identity_keys: tuple[str, ...] = ()

    evidence: tuple[Evidence, ...] = ()
    extractor: str = "unknown"
    confidence: float = 0.0
    verdict: GroundingVerdict = GroundingVerdict.UNCHECKED
    # Filled by the corroborator: how many independent sources agreed.
    support: int = 1

    @property
    def is_edge(self) -> bool:
        return self.object_entity is not None

    @property
    def signature(self) -> tuple[str, str, str, str, str, tuple[tuple[str, str], ...]]:
        """The identity two extractions must share to count as the same claim.

        Polarity is included: if it were not, "X sells data" and "X does not
        sell data" would merge and each would raise the other's `support`.

        Reconcilable qualifiers are deliberately excluded: "CEO of X (2019-2024)"
        and "CEO of X (since 2019)" are the same claim told two ways, and the
        corroborator's job is to reconcile them rather than to emit both. The
        keys named in `identity_keys` are included, sorted: uptime at p50 and
        uptime at p95 are two measurements, and merging them inflates support.
        """
        obj = self.object_entity.key if self.object_entity else repr(self.object_value)
        present = (k for k in self.identity_keys if k in self.qualifiers)
        scoped = tuple(sorted((k, repr(self.qualifiers[k])) for k in present))
        return (
            self.subject.key,
            self.subject.type,
            self.predicate,
            obj,
            self.polarity.value,
            scoped,
        )


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


class LinkKind(StrEnum):
    """What a resolver concluded about two entity keys."""

    SAME_AS = "same_as"
    SIMILAR = "similar"
    # The one nobody else records. A strong identifier that disagrees kills a
    # match no matter how alike the names are, and writing that down is what
    # makes a threshold retunable rather than a one-way door.
    DIFFERENT = "different"


class EntityLink(Frozen):
    """A resolver's opinion about two entities, kept next to the graph.

    Resolution here proposes links; it never replaces nodes. A merge that
    destroys one of two entities cannot be undone when the threshold turns out
    to be wrong — and it is wrong on the first try, every time. A link can be
    lowered, raised or ignored, and the evidence it was made on is still there.
    """

    source_key: str
    target_key: str
    kind: LinkKind
    score: float | None = None
    evidence: tuple[Evidence, ...] = ()
    # For DIFFERENT: name the disagreeing identifier, so the rejection is
    # auditable — "external_id mismatch: DE-114322 vs GB-889401".
    reason: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# The output
# --------------------------------------------------------------------------- #


class KnowledgeGraph(Frozen):
    """The pipeline's output, and the only thing a sink has to understand.

    Kept storage-neutral: entities, facts, the links between entities, and the
    ontology they were extracted against. A Neo4j sink, a Turtle serialiser and
    a NetworkX exporter all read the same object, which is what "or any graph
    DB" has to mean in practice. A sink that has no use for links ignores them.
    """

    entities: tuple[Entity, ...] = ()
    facts: tuple[Fact, ...] = ()
    links: tuple[EntityLink, ...] = ()
    ontology_name: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def edges(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.is_edge)

    @property
    def properties(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if not f.is_edge)

    def __len__(self) -> int:
        return len(self.facts)
