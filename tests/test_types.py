"""The data model's load-bearing behaviours, not its field list."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from odke import Document, Entity, Fact, KnowledgeGraph, SourceTier, Span


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


def test_signature_ignores_qualifiers_so_the_same_claim_merges() -> None:
    """'CEO since 2019' and 'CEO 2019-2024' are one claim told two ways."""
    subject = Entity(key="p1", type="Person")
    obj = Entity(key="c1", type="Company")
    a = Fact(subject=subject, predicate="ceo_of", object_entity=obj, qualifiers={"start": "2019"})
    b = Fact(subject=subject, predicate="ceo_of", object_entity=obj, qualifiers={"end": "2024"})
    assert a.signature == b.signature
    assert a.id != b.id


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


def test_trust_tiers_are_ordered_so_conflicts_can_be_resolved() -> None:
    """A curated record must outrank an unverified scrape when they disagree."""
    assert SourceTier.CURATED.weight > SourceTier.AUTHORITATIVE.weight
    assert SourceTier.AUTHORITATIVE.weight > SourceTier.COMMUNITY.weight
    assert SourceTier.COMMUNITY.weight > SourceTier.UNVERIFIED.weight
