"""Normalisation: one spelling per value, and the source's spelling still there."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from openodke import Entity, Fact, KnowledgeGraph, Normalizer, Ontology, Predicate
from openodke.corroborate import (
    NAME_KEY,
    SOURCE_FORM,
    ValueNormalizer,
    name_key,
    normalize_date,
    normalize_quantity,
    normalize_value,
)

ADA = Entity(key="p1", type="Person", label="Ada Lovelace")


def _fact(value: object, **kw: object) -> Fact:
    return Fact(subject=ADA, predicate="born", object_value=value, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    ["10 December 1815", "1815-12-10", "Dec 10, 1815", "December 10th, 1815", "10th of Dec. 1815",
     "1815/12/10", "10/12/1815 ", "  dec 10 1815"],
)  # fmt: skip
def test_common_date_spellings_become_one_iso_date(text: str) -> None:
    assert normalize_date(text, day_first=True) == "1815-12-10"


def test_the_issue_s_three_spellings_share_one_signature() -> None:
    """The #18 case: three textual forms, one claim for the corroborator to count."""
    normalizer = ValueNormalizer()
    facts = [
        normalizer.normalize(_fact(t)) for t in ("10 December 1815", "1815-12-10", "Dec 10, 1815")
    ]
    assert {f.object_value for f in facts} == {"1815-12-10"}
    assert len({f.signature for f in facts}) == 1


def test_an_ambiguous_numeric_date_is_left_alone_unless_the_caller_says() -> None:
    """03/04/2020 is two different days on two sides of the Atlantic; do not guess."""
    assert normalize_date("03/04/2020") is None
    assert normalize_date("03/04/2020", day_first=True) == "2020-04-03"
    assert normalize_date("03/04/2020", day_first=False) == "2020-03-04"
    # Unambiguous either way.
    assert normalize_date("13/04/2020") == "2020-04-13"
    assert normalize_date("04/13/2020") == "2020-04-13"


def test_partial_dates_and_datetimes_keep_their_precision() -> None:
    assert normalize_date("December 1815") == "1815-12"
    assert normalize_date("1815-12") == "1815-12"
    assert normalize_date("2024-01-02T10:00:00Z") == "2024-01-02T10:00:00+00:00"
    assert normalize_value(date(1815, 12, 10)) == "1815-12-10"
    assert normalize_value(datetime(2024, 1, 2, tzinfo=UTC)) == "2024-01-02T00:00:00+00:00"


def test_impossible_dates_are_not_dates() -> None:
    assert normalize_date("31 February 2020") is None
    assert normalize_date("10 Smarch 1815") is None
    assert normalize_date("2019-20") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234", 1234),
        ("1,234.50", 1234.5),
        ("-3", -3),
        ("3 million", 3_000_000),
        ("12.5%", "12.5 %"),
        ("12.5 percent", "12.5 %"),
        ("$1,200", "1200 USD"),
        ("USD 1,200", "1200 USD"),
        ("1200 USD", "1200 USD"),
        ("€5m", "5000000 EUR"),
        ("1.2bn EUR", "1200000000 EUR"),
        ("£3.2 million", "3200000 GBP"),
        ("1.5 GB", "1500000000 B"),
        ("1.5 GiB", "1610612736 B"),
    ],
)
def test_numbers_and_units_have_one_canonical_form(text: str, expected: object) -> None:
    result = normalize_quantity(text)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize("text", ["0123", "1,2", "5 m", "10 Mb", "12 apples", "1.234,5"])
def test_what_is_not_surely_a_quantity_is_refused(text: str) -> None:
    """A leading zero is an identifier, 'm' alone is metres, 'Mb' is megabits."""
    assert normalize_quantity(text) is None


def test_the_ontology_range_decides_what_a_value_may_become() -> None:
    """A registration number with a string range must not turn into an integer."""
    ontology = Ontology(
        predicates={
            "registration": Predicate(name="registration", range="string"),
            "founded": Predicate(name="founded", range="date"),
            "employees": Predicate(name="employees", range="integer"),
        }
    )
    normalizer = ValueNormalizer(ontology)
    acme = Entity(key="c1", type="Company", label="Acme")

    def run(pred: str, value: str) -> object:
        fact = Fact(subject=acme, predicate=pred, object_value=value)
        return normalizer.normalize(fact).object_value

    assert run("registration", " 114322 ") == "114322"
    assert run("founded", "Dec 10, 1815") == "1815-12-10"
    assert run("founded", "1,234") == "1,234"
    assert run("employees", "1,234") == 1234
    assert run("employees", "1815-12-10") == "1815-12-10"


def test_the_source_spelling_is_recorded_not_destroyed() -> None:
    fact = _fact("Dec 10, 1815", qualifiers={"start_time": "Jan 5, 2019", "rank": "preferred"})
    normalized = ValueNormalizer().normalize(fact)
    assert normalized.object_value == "1815-12-10"
    assert normalized.qualifiers["start_time"] == "2019-01-05"
    assert normalized.qualifiers["rank"] == "preferred"
    assert normalized.qualifiers[SOURCE_FORM] == {
        "object_value": ("Dec 10, 1815",),
        "start_time": ("Jan 5, 2019",),
    }
    # Provenance is not identity: the record never splits a signature.
    assert normalized.signature == ValueNormalizer().normalize(_fact("1815-12-10")).signature


def test_normalisation_is_idempotent() -> None:
    normalizer = ValueNormalizer()
    for value in ("Dec 10, 1815", "$1.2bn", "12.5 percent", "1.5 GiB", "1,234", "Ada", 7):
        once = normalizer.normalize(_fact(value))
        twice = normalizer.normalize(once)
        assert twice is once


def test_a_fact_with_nothing_to_normalise_is_the_same_object() -> None:
    normalizer = ValueNormalizer()
    fact = normalizer.normalize(_fact("Ada"))
    assert normalizer.normalize(fact) is fact
    assert SOURCE_FORM not in fact.qualifiers


@pytest.mark.parametrize(
    "spelling", ["Acme Inc.", "Acme, Inc", "ACME Inc", "Acme Corporation", "The Acme Company",
                 "Acmé GmbH", "Acme Pvt. Ltd.", "Acme S.A."],
)  # fmt: skip
def test_organisation_forms_share_a_comparison_key(spelling: str) -> None:
    assert name_key(spelling) == "acme"


def test_a_name_is_never_reduced_to_nothing() -> None:
    assert name_key("Limited") == "limited"
    assert name_key("The Company") == "company"


def test_person_forms_reorder_and_drop_titles_but_keep_initials() -> None:
    assert name_key("Lovelace, Ada", person=True) == "ada lovelace"
    assert name_key("Dr. Ada Lovelace", person=True) == "ada lovelace"
    assert name_key("Martin Luther King Jr.", person=True) == "martin luther king"
    # Initials are evidence for the resolver to weigh, not a match to assume.
    assert name_key("J. Smith", person=True) != name_key("John Smith", person=True)


def test_entity_labels_stay_as_the_source_said_and_gain_a_comparison_key() -> None:
    normalizer = ValueNormalizer(person_types=("Person",))
    acme = Entity(key="c1", type="Company", label="Acme, Inc.")
    fact = Fact(
        subject=Entity(key="p1", type="Person", label="Lovelace, Ada"),
        predicate="works_at",
        object_entity=acme,
    )
    normalized = normalizer.normalize(fact)
    assert normalized.subject.label == "Lovelace, Ada"
    assert normalized.subject.attributes[NAME_KEY] == "ada lovelace"
    assert normalized.object_entity is not None
    assert normalized.object_entity.label == "Acme, Inc."
    assert normalized.object_entity.attributes[NAME_KEY] == "acme"


def test_the_normaliser_satisfies_its_protocol_and_survives_json() -> None:
    normalizer = ValueNormalizer()
    assert isinstance(normalizer, Normalizer)
    fact = normalizer.normalize(_fact("Dec 10, 1815", qualifiers={"start_time": "Jan 2019"}))
    back = KnowledgeGraph.model_validate_json(KnowledgeGraph(facts=(fact,)).model_dump_json())
    # Lists after the round trip; normalising again still changes nothing.
    again = normalizer.normalize(back.facts[0])
    assert again.qualifiers[SOURCE_FORM] == {
        "object_value": ("Dec 10, 1815",),
        "start_time": ("Jan 2019",),
    }
    assert again.object_value == "1815-12-10"
