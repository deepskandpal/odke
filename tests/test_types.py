"""The data model's load-bearing behaviours, not its field list."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from odke import (
    Chunk,
    Document,
    Entity,
    EntityLink,
    Evidence,
    Fact,
    KnowledgeGraph,
    LinkKind,
    Polarity,
    Resolution,
    RouteVerdict,
    SourceTier,
    Span,
)


def test_facts_are_immutable() -> None:
    """Stages return new facts rather than editing them, or provenance is a lie."""
    fact = Fact(subject=Entity(key="a", type="Person"), predicate="name", object_value="Ada")
    with pytest.raises(ValidationError):
        fact.predicate = "other"  # type: ignore[misc]


def test_span_faithfulness_catches_a_paraphrased_quote() -> None:
    """The grounder's cheapest check: do the offsets really say what was claimed."""
    doc = Document(id="d1", text="Ada Lovelace wrote the first algorithm.")
    honest = Span(doc_id="d1", start=0, end=12, quote="Ada Lovelace")
    invented = Span(doc_id="d1", start=0, end=12, quote="Alan Turing")
    assert honest.is_faithful(doc)
    assert not invented.is_faithful(doc)
    assert honest.resolve(doc) == "Ada Lovelace"


def test_a_chunk_is_a_materialised_span_so_local_offsets_map_back() -> None:
    """Provenance has to survive chunking, or the grounder has nothing to check."""
    doc = Document(id="d1", text="Ada Lovelace wrote the first algorithm. It ran on paper.")
    chunk = Chunk(doc_id="d1", start=40, end=56, text=doc.text[40:56], index=1)
    local = chunk.text.index("paper")
    span = Span(doc_id="d1", start=chunk.start + local, end=chunk.start + local + 5)
    assert span.resolve(doc) == "paper"


def test_a_route_verdict_defaults_to_chunk_scope_and_carries_no_taxonomy() -> None:
    """`label` is the caller's word; the package ships no fact/policy/narrative enum."""
    verdict = RouteVerdict(action="skip", label="marketing", reason="brand voice")
    assert verdict.scope == "chunk"
    assert RouteVerdict(action="extract").label is None
    with pytest.raises(ValidationError):
        RouteVerdict(action="maybe")  # type: ignore[arg-type]


def test_signature_ignores_qualifiers_so_the_same_claim_merges() -> None:
    """'CEO since 2019' and 'CEO 2019-2024' are one claim told two ways."""
    subject = Entity(key="p1", type="Person")
    obj = Entity(key="c1", type="Company")
    a = Fact(subject=subject, predicate="ceo_of", object_entity=obj, qualifiers={"start": "2019"})
    b = Fact(subject=subject, predicate="ceo_of", object_entity=obj, qualifiers={"end": "2024"})
    assert a.signature == b.signature
    assert a.id != b.id


def test_identity_qualifiers_split_the_signature_and_reconcilable_ones_do_not() -> None:
    """Uptime at p50 and at p95 are two claims; 'since 2019' and '2019-2024' are one."""
    subject = Entity(key="c1", type="Company")
    p50 = Fact(
        subject=subject,
        predicate="has_uptime",
        object_value="99.9%",
        qualifiers={"percentile": "p50", "start_time": "2024"},
        identity_keys=("percentile",),
    )
    p95 = p50.model_copy(update={"qualifiers": {"percentile": "p95", "start_time": "2024"}})
    later = p50.model_copy(update={"qualifiers": {"percentile": "p50", "start_time": "2025"}})
    assert p50.signature != p95.signature
    assert p50.signature == later.signature


def test_identity_keys_absent_from_the_qualifiers_are_ignored() -> None:
    """A declared key the extractor did not fill must not split the claim on a blank."""
    subject = Entity(key="c1", type="Company")
    with_key = Fact(subject=subject, predicate="p", object_value=1, identity_keys=("tier",))
    without = Fact(subject=subject, predicate="p", object_value=1)
    assert with_key.signature == without.signature


def test_a_denial_does_not_corroborate_its_own_contradiction() -> None:
    """'X sells data' and 'X does not sell data' must never merge into one claim."""
    subject = Entity(key="c1", type="Company")
    asserted = Fact(subject=subject, predicate="sells", object_value="customer_data")
    denied = asserted.model_copy(update={"polarity": Polarity.DENIED})
    partial = asserted.model_copy(update={"polarity": Polarity.PARTIAL})
    assert asserted.polarity is Polarity.ASSERTED
    assert len({asserted.signature, denied.signature, partial.signature}) == 3


def test_polarity_survives_serialisation() -> None:
    fact = Fact(subject=Entity(key="c1", type="Company"), predicate="p", polarity=Polarity.DENIED)
    assert Fact.model_validate_json(fact.model_dump_json()).signature == fact.signature


def test_edges_and_properties_split_on_the_object_kind() -> None:
    subject = Entity(key="p1", type="Person")
    edge = Fact(
        subject=subject, predicate="works_at", object_entity=Entity(key="c1", type="Company")
    )
    prop = Fact(subject=subject, predicate="name", object_value="Ada")
    kg = KnowledgeGraph(facts=(edge, prop))
    assert kg.edges == (edge,)
    assert kg.properties == (prop,)
    assert len(kg) == 2


def test_valid_time_is_the_second_clock_and_stays_out_of_the_signature() -> None:
    """When a claim was true is not what the claim is; the interval is reconciled, not split on."""
    subject = Entity(key="p1", type="Person")
    obj = Entity(key="c1", type="Company")
    open_ended = Fact(
        subject=subject,
        predicate="ceo_of",
        object_entity=obj,
        valid_from=datetime(2019, 1, 1, tzinfo=UTC),
    )
    closed = open_ended.model_copy(update={"valid_to": datetime(2024, 1, 1, tzinfo=UTC)})
    assert open_ended.valid_to is None
    assert open_ended.signature == closed.signature
    # The transaction clock lives on the evidence, and defaults to now.
    assert Evidence(doc_id="d1").retrieved_at.tzinfo is not None


def test_links_record_a_rejected_merge_with_its_reason() -> None:
    """DIFFERENT is the disagreement rule made auditable: it says which identifier disagreed."""
    same = EntityLink(source_key="a", target_key="b", kind=LinkKind.SAME_AS, score=1.0)
    similar = EntityLink(source_key="a", target_key="c", kind=LinkKind.SIMILAR, score=0.94)
    different = EntityLink(
        source_key="a",
        target_key="d",
        kind=LinkKind.DIFFERENT,
        score=0.94,
        reason="external_id mismatch: DE-114322 vs GB-889401",
        evidence=(Evidence(doc_id="d1"),),
    )
    assert {same.kind, similar.kind, different.kind} == set(LinkKind)
    assert different.reason is not None and "DE-114322" in different.reason
    assert same.reason is None


def test_links_ride_on_the_graph_and_default_to_none() -> None:
    """One output object is what 'any sink' means; a sink that wants no links ignores them."""
    assert KnowledgeGraph().links == ()
    link = EntityLink(source_key="a", target_key="b", kind=LinkKind.SAME_AS)
    kg = KnowledgeGraph(links=(link,))
    round_tripped = KnowledgeGraph.model_validate_json(kg.model_dump_json())
    assert round_tripped.links[0].kind is LinkKind.SAME_AS


def test_an_entity_records_how_its_key_was_decided() -> None:
    """A wrong link is invisible without this; with it, it is a query."""
    unresolved = Entity(key="acme", type="Company")
    linked = Entity(
        key="acme",
        type="Company",
        resolution=Resolution(method="linker", score=0.87, linker="splink"),
    )
    by_id = Entity(key="Q95", type="Company", resolution=Resolution(method="external_id"))
    assert unresolved.resolution is None
    assert linked.resolution is not None and linked.resolution.score == 0.87
    assert by_id.resolution is not None and by_id.resolution.score is None
    assert Entity.model_validate_json(linked.model_dump_json()).resolution == linked.resolution
    with pytest.raises(ValidationError):
        Resolution(method="guess")  # type: ignore[arg-type]


def test_trust_tiers_are_ordered_so_conflicts_can_be_resolved() -> None:
    """A curated record must outrank an unverified scrape when they disagree."""
    assert SourceTier.CURATED.weight > SourceTier.AUTHORITATIVE.weight
    assert SourceTier.AUTHORITATIVE.weight > SourceTier.COMMUNITY.weight
    assert SourceTier.COMMUNITY.weight > SourceTier.UNVERIFIED.weight
