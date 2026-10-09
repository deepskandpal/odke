"""Inverse and symmetric partners: the other direction of a stated fact, with no model."""

from __future__ import annotations

from openodke import Entity, Evidence, Fact, GroundingVerdict, Ontology, Span
from openodke.corroborate import DERIVED, SignatureCorroborator, derived_from, partners

ONTOLOGY = Ontology.from_dict(
    {
        "types": {"Place": {}, "Person": {}},
        "predicates": {
            "located_in": {"domain": ["Place"], "range": "Place", "inverse_of": "contains"},
            "contains": {
                "domain": ["Place"],
                "range": "Place",
                "qualifiers": {"level": {"identity": True}},
            },
            "spouse": {"domain": ["Person"], "range": "Person", "symmetric": True},
            "born_in": {"domain": ["Person"], "range": "Place"},
            "population": {"domain": ["Place"], "range": "integer"},
        },
    }
)
BRITTANY = Entity(key="brittany", type="Place", label="Brittany")
FRANCE = Entity(key="france", type="Place", label="France")
ADA = Entity(key="ada", type="Person", label="Ada")
WILLIAM = Entity(key="william", type="Person", label="William")


def _evidence(doc: str) -> Evidence:
    return Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=24, quote=None))


def _fact(subject: Entity, predicate: str, obj: Entity, doc: str = "d1", **extra: object) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object_entity=obj,
        evidence=(_evidence(doc),),
        extractor="llm",
        confidence=0.8,
        verdict=GroundingVerdict.SUPPORTED,
        **extra,  # type: ignore[arg-type]
    )


def test_an_inverse_partner_is_the_same_claim_the_other_way_round() -> None:
    stated = _fact(BRITTANY, "located_in", FRANCE, qualifiers={"level": "region"})
    (partner,) = partners([stated], ONTOLOGY)
    assert (partner.subject, partner.predicate, partner.object_entity) == (
        FRANCE,
        "contains",
        BRITTANY,
    )
    # Same receipts, same verdict: never grounded again, and the gate treats both alike.
    assert partner.evidence == stated.evidence
    assert (partner.verdict, partner.confidence, partner.extractor) == (
        GroundingVerdict.SUPPORTED,
        0.8,
        "llm",
    )
    assert partner.id != stated.id
    # The inverse's own identity keys, so its signature is the inverse's.
    assert partner.identity_keys == ("level",)
    assert partner.signature[-1] == (("level", "'region'"),)
    assert partner.qualifiers[DERIVED] == {"rule": "inverse", "of": stated.signature}
    assert derived_from(partner) == stated.signature
    assert derived_from(stated) is None


def test_a_symmetric_partner_swaps_the_ends_and_keeps_the_predicate() -> None:
    stated = _fact(ADA, "spouse", WILLIAM)
    (partner,) = partners([stated], ONTOLOGY)
    assert (partner.subject.key, partner.predicate, partner.object_entity) == (
        "william",
        "spouse",
        ADA,
    )
    assert partner.qualifiers[DERIVED]["rule"] == "symmetric"


def test_nothing_is_derived_without_a_declared_pair_or_from_a_value() -> None:
    facts = [
        _fact(ADA, "born_in", FRANCE),
        Fact(subject=FRANCE, predicate="population", object_value=68_000_000),
        # A literal on an edge predicate is malformed, and still has no inverse.
        Fact(subject=BRITTANY, predicate="located_in", object_value="France"),
    ]
    assert partners(facts, ONTOLOGY) == []
    assert partners([_fact(BRITTANY, "located_in", FRANCE)], Ontology()) == []


def test_no_partner_when_the_batch_already_states_it() -> None:
    """The stated fact stands on its own evidence: no duplicate, with or without a corroborator."""
    both = [_fact(BRITTANY, "located_in", FRANCE), _fact(FRANCE, "contains", BRITTANY, doc="d2")]
    assert partners(both, ONTOLOGY) == []
    spouses = [_fact(ADA, "spouse", WILLIAM), _fact(WILLIAM, "spouse", ADA)]
    assert partners(spouses, ONTOLOGY) == []


def test_a_derived_fact_derives_nothing() -> None:
    """Run again over its own output, the step adds nothing: not even the fact it came from."""
    (partner,) = partners([_fact(BRITTANY, "located_in", FRANCE)], ONTOLOGY)
    assert partners([partner], ONTOLOGY) == []


def test_a_contradicted_fact_hands_its_verdict_to_its_partner() -> None:
    stated = _fact(BRITTANY, "located_in", FRANCE).model_copy(
        update={"verdict": GroundingVerdict.CONTRADICTED}
    )
    (partner,) = partners([stated], ONTOLOGY)
    assert partner.verdict is GroundingVerdict.CONTRADICTED


def test_a_partner_shares_its_source_support_and_never_merges_with_it() -> None:
    """Two documents state the claim: each partner cites one, and they merge as their sources do."""
    stated = [_fact(BRITTANY, "located_in", FRANCE, doc=doc) for doc in ("d1", "d2")]
    merged = SignatureCorroborator(ONTOLOGY).corroborate([*stated, *partners(stated, ONTOLOGY)])
    by_predicate = {fact.predicate: fact for fact in merged}
    assert len(merged) == 2
    source, partner = by_predicate["located_in"], by_predicate["contains"]
    assert source.support == partner.support == 2
    assert {e.doc_id for e in partner.evidence} == {"d1", "d2"}
    assert DERIVED not in source.qualifiers
    assert derived_from(partner) == source.signature


def test_the_link_survives_json() -> None:
    stated = _fact(
        FRANCE, "contains", BRITTANY, qualifiers={"level": "region"}, identity_keys=("level",)
    )
    (partner,) = partners([stated], ONTOLOGY)
    assert partner.predicate == "located_in" and partner.identity_keys == ()
    read_back = Fact.model_validate_json(partner.model_dump_json())
    assert read_back.qualifiers[DERIVED]["of"][-1] == [["level", "'region'"]]
    assert derived_from(read_back) == stated.signature
