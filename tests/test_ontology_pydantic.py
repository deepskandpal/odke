"""Ontology.from_pydantic: the models a user already has, and nothing silently dropped."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from openodke.ontology import Ontology, OntologyLoadError


class Company(BaseModel):
    """An incorporated organisation."""

    legal_name: str = Field(description="Name on the certificate of incorporation.")
    founded: date | None = None


class Person(BaseModel):
    """An individual human being."""

    full_name: str = Field(title="Full name", description="As written in the source.")
    employer: Company | None = Field(default=None, description="Where they work.")
    past_employers: list[Company] = []
    nicknames: tuple[str, ...] = ()
    height_cm: float | None = None


class Scientist(Person):
    field: str = Field(description="Primary field of research.", examples=["chemistry"])


class Tier(Enum):
    FREE = "free"
    PRO = "pro"


class Plan(BaseModel):
    tier: Tier
    seats: Literal[1, 5, 50]
    active: bool = True
    renewed_at: datetime | None = None


def test_two_related_models_produce_one_edge_between_them() -> None:
    ontology = Ontology.from_pydantic(Person, Company, name="people")
    edges = [p for p in ontology.predicates.values() if p.is_edge_in(ontology)]
    single = [p for p in edges if p.cardinality == "single"]
    assert [(p.name, p.domain, p.range) for p in single] == [("employer", ("Person",), "Company")]
    assert set(ontology.types) == {"Person", "Company"}
    assert ontology.name == "people"


def test_a_list_of_a_model_is_a_multi_edge_and_a_tuple_of_str_is_a_multi_property() -> None:
    ontology = Ontology.from_pydantic(Person, Company)
    past = ontology.predicates["past_employers"]
    assert (past.range, past.cardinality) == ("Company", "multi")
    nicknames = ontology.predicates["nicknames"]
    assert (nicknames.range, nicknames.cardinality) == ("string", "multi")


def test_optional_or_defaulted_is_not_required() -> None:
    """Required means the source must state it: no None, and no default to fall back on."""
    predicates = Ontology.from_pydantic(Person, Company).predicates
    assert predicates["full_name"].required
    assert predicates["legal_name"].required
    assert not predicates["employer"].required
    assert not predicates["height_cm"].required
    assert not predicates["past_employers"].required
    assert not predicates["founded"].required


def test_descriptions_already_written_become_the_schema_text() -> None:
    ontology = Ontology.from_pydantic(Person, Company)
    full_name = ontology.predicates["full_name"]
    assert full_name.description == "As written in the source."
    assert full_name.label == "Full name"
    assert ontology.types["Company"].description == "An incorporated organisation."
    assert ontology.predicates["founded"].range == "date"
    assert "Where they work." in ontology.snippet("Person").render()


def test_a_subclass_is_a_child_type_and_does_not_redeclare_inherited_fields() -> None:
    ontology = Ontology.from_pydantic(Person, Scientist, Company)
    assert ontology.types["Scientist"].parents == ("Person",)
    assert ontology.types["Scientist"].description is None
    assert ontology.predicates["full_name"].domain == ("Person",)
    assert ontology.predicates["field"].domain == ("Scientist",)
    assert ontology.predicates["field"].examples == ("chemistry",)
    assert "full_name" in {p.name for p in ontology.predicates_for("Scientist")}


def test_enums_literals_and_scalars_map_to_literal_ranges() -> None:
    predicates = Ontology.from_pydantic(Plan).predicates
    assert {n: p.range for n, p in predicates.items()} == {
        "tier": "string",
        "seats": "integer",
        "active": "boolean",
        "renewed_at": "datetime",
    }


def test_the_result_is_a_valid_serialisable_ontology() -> None:
    ontology = Ontology.from_pydantic(Person, Scientist, Company)
    assert Ontology.from_json(ontology.model_dump_json()) == ontology


class _Vendor(BaseModel):
    name: str
    headquarters: str | None = None


class _Buyer(BaseModel):
    name: str
    headquarters: str


def test_a_field_shared_by_two_models_is_one_predicate_with_both_domains() -> None:
    predicate = Ontology.from_pydantic(_Vendor, _Buyer).predicates["headquarters"]
    assert predicate.domain == ("_Vendor", "_Buyer")
    # Per predicate, not per domain: required only where every declaring model agrees.
    assert not predicate.required
    assert Ontology.from_pydantic(_Vendor, _Buyer).predicates["name"].required


class _Unmappable(BaseModel):
    ok: str
    attributes: dict[str, int]
    anything: Any
    either: int | str
    owner: Company
    pair: tuple[int, str]


def test_every_unsupported_annotation_is_reported_by_field_name() -> None:
    """Nothing is silently dropped: a dropped field is a predicate nobody asks about."""
    with pytest.raises(OntologyLoadError) as info:
        Ontology.from_pydantic(_Unmappable)
    problems = info.value.problems
    assert [p.split(":", 1)[0] for p in problems] == [
        "_Unmappable.attributes",
        "_Unmappable.anything",
        "_Unmappable.either",
        "_Unmappable.owner",
        "_Unmappable.pair",
    ]
    assert "dict[str, int] has no ontology range" in problems[0]
    assert "union" in problems[2]
    assert "not passed to from_pydantic" in problems[3]
    assert str(info.value).splitlines()[0] == "5 problems"


class _Rival(BaseModel):
    legal_name: int


def test_the_same_field_name_with_a_different_range_is_a_conflict() -> None:
    with pytest.raises(OntologyLoadError, match=r"_Rival\.legal_name: integer/single conflicts"):
        Ontology.from_pydantic(Company, _Rival)


def test_something_that_is_not_a_model_is_refused() -> None:
    with pytest.raises(OntologyLoadError, match="not a pydantic model class"):
        Ontology.from_pydantic(Company, dict)  # type: ignore[arg-type]
