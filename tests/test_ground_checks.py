"""The free checks: the ontology's room for a fact, decided before any model is asked."""

from __future__ import annotations

import pytest

from openodke import Document, Entity, Evidence, Fact, GroundingVerdict, Ontology, Span
from openodke.corroborate import CHECK
from openodke.ground import CheckedGrounder, LLMGrounder, schema_problem
from openodke.llm.testing import RecordedClient

TEXT = "Halden Robotics opened an office in Lyon in 2019."
DOC = Document(id="d", text=TEXT)
HALDEN = Entity(key="Company:halden robotics", type="Company", label="Halden Robotics")


@pytest.fixture
def companies() -> Ontology:
    return Ontology.from_dict(
        {
            "name": "companies",
            "types": {"Company": {}, "Startup": {"parents": ["Company"]}, "City": {}, "Person": {}},
            "predicates": {
                "office_in": {"domain": ["Company"], "range": "City"},
                "founded": {"domain": ["Company"], "range": "integer"},
            },
        }
    )


def _fact(predicate: str = "office_in", subject: Entity = HALDEN, **obj: object) -> Fact:
    whole = Evidence(doc_id="d", span=Span(doc_id="d", start=0, end=len(TEXT)))
    if not obj:
        obj = {"object_entity": Entity(key="City:lyon", type="City", label="Lyon")}
    return Fact(subject=subject, predicate=predicate, evidence=(whole,), **obj)  # type: ignore[arg-type]


def test_a_fact_the_ontology_has_room_for_has_no_problem(companies: Ontology) -> None:
    assert schema_problem(_fact(), companies) is None
    assert schema_problem(_fact("founded", object_value=2014), companies) is None
    # A subtype fits its parent's domain.
    startup = HALDEN.model_copy(update={"type": "Startup"})
    assert schema_problem(_fact(subject=startup), companies) is None


def test_each_check_names_what_does_not_fit(companies: Ontology) -> None:
    check, reason = schema_problem(_fact("headquartered_in"), companies) or ("", "")
    assert check == "predicate" and "'headquartered_in' is not a predicate" in reason
    person = Entity(key="Person:mara", type="Person", label="Mara Quist")
    check, reason = schema_problem(_fact(subject=person), companies) or ("", "")
    assert check == "domain" and "'Person' is not in the domain of 'office_in'" in reason
    leeds = Entity(key="Person:leeds", type="Person", label="Leeds")
    assert schema_problem(_fact(object_entity=leeds), companies) == (
        "range",
        "object type 'Person' is not the range of 'office_in' (City)",
    )
    assert schema_problem(_fact(object_value="Lyon"), companies) == (
        "range",
        "'office_in' is an edge to City, not a value",
    )
    city = Entity(key="City:2014", type="City", label="2014")
    assert schema_problem(_fact("founded", object_entity=city), companies) == (
        "range",
        "'founded' holds a value (integer), not an entity",
    )


def test_an_ontology_with_no_predicates_finds_nothing_wrong() -> None:
    assert schema_problem(_fact("anything at all"), Ontology()) is None


def test_a_refused_fact_is_stamped_and_never_asked_about(companies: Ontology) -> None:
    client = RecordedClient([{"match": "Claim:", "response": {"verdict": "supported"}}])
    grounder = CheckedGrounder(LLMGrounder(client=client), ontology=companies)
    stated, invented = _fact(), _fact("headquartered_in")
    grounded = grounder.ground_many([invented, stated], DOC)

    assert [f.verdict for f in grounded] == [GroundingVerdict.UNCHECKED, GroundingVerdict.SUPPORTED]
    assert grounded[0].qualifiers[CHECK]["check"] == "predicate"
    assert CHECK not in grounded[1].qualifiers
    assert len(client.calls) == 1
    stats = grounder.stats
    assert stats["checks"] == {"facts": 2, "refused": 1, "predicate": 1, "domain": 0, "range": 0}
    assert stats["calls"] == 1 and stats["span"]["facts"] == 1


def test_with_no_grounder_it_is_the_free_checks_alone(companies: Ontology) -> None:
    free = CheckedGrounder(ontology=companies, locate=True)
    whole = Evidence(
        doc_id="d", span=Span(doc_id="d", start=0, end=len(TEXT)), span_origin="context"
    )
    uncited = _fact().model_copy(update={"evidence": (whole,)})
    invented = _fact().model_copy(
        update={
            "evidence": (Evidence(doc_id="d", span=Span(doc_id="d", start=0, end=4, quote="Lyon")),)
        }
    )
    located, refused = free.ground_many([uncited, invented], DOC)
    assert located.verdict is GroundingVerdict.UNCHECKED
    assert located.evidence[0].span_origin == "located"
    assert refused.verdict is GroundingVerdict.NOT_FOUND
    assert free.stats["span"]["not_found"] == 1
    assert free.stats["locate"] == {"facts": 1, "located": 1, "not_located": 0}


def test_a_grounder_does_its_own_locating(companies: Ontology) -> None:
    with pytest.raises(ValueError, match="LLMGrounder\\(locate=True\\)"):
        CheckedGrounder(LLMGrounder(client=RecordedClient([])), ontology=companies, locate=True)
