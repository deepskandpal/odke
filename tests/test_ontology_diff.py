"""Ontology.diff: schema drift, each change marked by whether it breaks a live graph."""

from __future__ import annotations

from pathlib import Path

from openodke.ontology import EntityType, Ontology, Predicate

EXAMPLE = Path(__file__).parent.parent / "examples" / "people.ontology.json"


def _changes(old: Ontology, new: Ontology) -> list[tuple[str, str, bool]]:
    return [(c.kind, c.path, c.breaking) for c in old.diff(new)]


def test_an_unchanged_schema_has_no_changes() -> None:
    assert Ontology.from_json(EXAMPLE).diff(Ontology.from_json(EXAMPLE)) == []


def test_additions_are_compatible_and_removals_break() -> None:
    old = Ontology.from_json(EXAMPLE)
    types = {k: v for k, v in old.types.items() if k != "Scientist"}
    types["Robot"] = EntityType(name="Robot")
    predicates = {k: v for k, v in old.predicates.items() if k != "headquarters"}
    predicates["ceo"] = Predicate(name="ceo", domain=("Company",), range="Person")
    new = old.model_copy(update={"types": types, "predicates": predicates})
    assert _changes(old, new) == [
        ("removed", "types.Scientist", True),
        ("added", "types.Robot", False),
        ("removed", "predicates.headquarters", True),
        ("added", "predicates.ceo", False),
    ]
    assert str(old.diff(new)[0]).split()[:3] == ["breaking", "removed", "types.Scientist:"]


def _ranges(**ranges: str) -> Ontology:
    return Ontology(
        types={
            "Company": EntityType(name="Company"),
            "PublicCompany": EntityType(name="PublicCompany", parents=("Company",)),
        },
        predicates={
            name: Predicate(name=name, range=ranges.get(name, default))
            for name, default in (
                ("employer", "PublicCompany"),
                ("headcount", "integer"),
                ("hq", "Company"),
            )
        },
    )


def test_a_widened_range_is_compatible_and_a_narrowed_one_breaks() -> None:
    widened = _ranges().diff(_ranges(employer="Company", headcount="number"))
    assert [(c.path, c.breaking) for c in widened] == [
        ("predicates.employer.range", False),
        ("predicates.headcount.range", False),
    ]
    (narrowed,) = _ranges().diff(_ranges(hq="PublicCompany"))
    assert narrowed.breaking and "narrowed" in narrowed.detail
    (flipped,) = _ranges().diff(_ranges(hq="string"))
    assert flipped.breaking
    assert flipped.detail.startswith("'Company' → 'string'")


def test_a_changed_cardinality_breaks_in_either_direction() -> None:
    single = Ontology(predicates={"email": Predicate(name="email")})
    multi = Ontology(predicates={"email": Predicate(name="email", cardinality="multi")})
    assert _changes(single, multi) == [("changed", "predicates.email.cardinality", True)]
    assert _changes(multi, single) == [("changed", "predicates.email.cardinality", True)]


def test_qualifier_changes_break_when_they_move_fact_identity() -> None:
    old = Ontology(
        predicates={
            "price": Predicate(
                name="price", qualifiers={"tier": {"identity": True}, "as_of": {}, "note": {}}
            )
        }
    )
    new = Ontology(
        predicates={
            "price": Predicate(
                name="price",
                qualifiers={
                    "tier": {"identity": False},
                    "as_of": {},
                    "currency": {},
                    "region": {"identity": True},
                },
            )
        }
    )
    assert _changes(old, new) == [
        # The scope follows the identity qualifiers, so it lost `tier`.
        ("changed", "predicates.price.cardinality_scope", True),
        ("changed", "predicates.price.qualifiers.tier.identity", True),
        ("removed", "predicates.price.qualifiers.note", True),
        ("added", "predicates.price.qualifiers.currency", False),
        ("added", "predicates.price.qualifiers.region", True),
    ]
    assert old.diff(new)[0].detail.startswith("(tier) → (region)")


def test_a_scope_change_breaks_only_when_the_compiled_scope_loses_a_key() -> None:
    """Identity keys are always in scope, so spelling them out changes no constraint."""
    identity = {"tier": {"identity": True}, "region": {"identity": True}}

    def scoped(scope: tuple[str, ...], qualifiers: dict = identity) -> Ontology:
        return Ontology(
            predicates={
                "price": Predicate(name="price", qualifiers=qualifiers, cardinality_scope=scope)
            }
        )

    path = "predicates.price.cardinality_scope"
    assert _changes(scoped(()), scoped(("region", "tier"))) == [("changed", path, False)]
    assert _changes(scoped(("region", "tier")), scoped(("tier",))) == [("changed", path, False)]
    # A key only the declaration carried is a real narrowing of what the store groups by.
    declared_only = {"tier": {"identity": True}, "region": {}}
    before, after = scoped(("region",), declared_only), scoped((), declared_only)
    assert _changes(before, after) == [("changed", path, True)]


def test_a_narrowed_domain_or_a_new_requirement_breaks_and_the_reverse_does_not() -> None:
    types = {"Person": EntityType(name="Person"), "Robot": EntityType(name="Robot")}
    old = Ontology(
        types=types,
        predicates={
            "name": Predicate(name="name"),
            "age": Predicate(name="age", domain=("Person", "Robot")),
            "ssn": Predicate(name="ssn"),
        },
    )
    new = Ontology(
        types=types,
        predicates={
            "name": Predicate(name="name", domain=("Person",)),
            "age": Predicate(name="age", domain=("Person",)),
            "ssn": Predicate(name="ssn", required=True),
        },
    )
    assert _changes(old, new) == [
        ("changed", "predicates.name.domain", True),
        ("changed", "predicates.age.domain", True),
        ("changed", "predicates.ssn.required", True),
    ]
    assert all(not c.breaking for c in new.diff(old))
    assert "no longer extracted for Robot" in old.diff(new)[1].detail


def test_identity_and_inheritance_changes_break_and_documentation_does_not() -> None:
    old = Ontology(
        version="1",
        types={
            "Agent": EntityType(name="Agent"),
            "Person": EntityType(name="Person", parents=("Agent",), keys=("full_name",)),
        },
    )
    new = Ontology(
        version="2",
        types={
            "Agent": EntityType(name="Agent", description="Anything that acts."),
            "Person": EntityType(name="Person", keys=("full_name", "birth_date")),
        },
    )
    assert _changes(old, new) == [
        ("changed", "version", False),
        ("changed", "types.Agent.description", False),
        ("changed", "types.Person.parents", True),
        ("changed", "types.Person.keys", True),
    ]
