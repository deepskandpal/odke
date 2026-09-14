"""Ontology.validate(): a subtly wrong schema is caught at load, not blamed on the model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from odke import Diagnostic
from odke.ontology import EntityType, Ontology, OntologyLoadError, Predicate

EXAMPLE = Path(__file__).parent.parent / "examples" / "people.ontology.json"


def _found(ontology: Ontology) -> list[tuple[str, str, str]]:
    return [(d.code, d.path, d.severity) for d in ontology.validate()]


def test_the_example_is_clean() -> None:
    assert Ontology.from_json(EXAMPLE).validate() == []


def test_a_misspelt_range_is_an_error_that_names_the_key_and_suggests_the_fix() -> None:
    """`Compnay` would otherwise turn an edge into a string property, silently."""
    ontology = Ontology(
        types={"Company": EntityType(name="Company")},
        predicates={
            "employer": Predicate(name="employer", range="Compnay"),
            "founded": Predicate(name="founded", range="date"),
        },
    )
    (diagnostic,) = ontology.validate()
    assert (diagnostic.code, diagnostic.path, diagnostic.severity) == (
        "unknown-range",
        "predicates.employer.range",
        "error",
    )
    assert "did you mean 'Company'?" in diagnostic.message
    assert str(diagnostic).startswith("error: predicates.employer.range: 'Compnay' is neither")
    assert str(diagnostic).endswith("[unknown-range]")


def test_a_parent_that_does_not_exist_is_named_by_index() -> None:
    ontology = Ontology(
        types={
            "Person": EntityType(name="Person"),
            "Scientist": EntityType(name="Scientist", parents=("Person", "Persn")),
        }
    )
    assert _found(ontology) == [("unknown-parent", "types.Scientist.parents[1]", "error")]


def test_an_inheritance_cycle_is_one_warning_per_cycle_not_an_exception() -> None:
    """lineage() already survives it; still worth saying, and not worth refusing."""
    ontology = Ontology(
        types={
            "A": EntityType(name="A", parents=("B",)),
            "B": EntityType(name="B", parents=("A",)),
            "C": EntityType(name="C", parents=("C",)),
        }
    )
    assert _found(ontology) == [
        ("inheritance-cycle", "types.A.parents", "warning"),
        ("inheritance-cycle", "types.C.parents", "warning"),
    ]
    assert "A, B inherit from each other" in ontology.validate()[0].message
    assert "C inherits from itself" in ontology.validate()[1].message


def test_a_domain_naming_no_type_makes_a_predicate_unextractable() -> None:
    ontology = Ontology(
        types={"Person": EntityType(name="Person")},
        predicates={
            "orphan": Predicate(name="orphan", domain=("Persn", "Human")),
            "half": Predicate(name="half", domain=("Person", "Robot")),
        },
    )
    assert _found(ontology) == [
        ("unreachable-predicate", "predicates.orphan.domain", "error"),
        ("unknown-domain", "predicates.half.domain[1]", "warning"),
    ]
    # The error is true: no snippet will ever carry it.
    assert "orphan" not in {p.name for p in ontology.predicates_for("Person")}
    assert "half" in {p.name for p in ontology.predicates_for("Person")}


def test_an_identity_key_must_be_a_predicate_the_type_can_hold() -> None:
    ontology = Ontology(
        types={
            "Person": EntityType(name="Person", keys=("full_name", "legal_name", "ssn")),
            "Company": EntityType(name="Company"),
            # Inherited predicates count: a Scientist is keyed by a Person's name.
            "Scientist": EntityType(name="Scientist", parents=("Person",), keys=("full_name",)),
        },
        predicates={
            "full_name": Predicate(name="full_name", domain=("Person",)),
            "legal_name": Predicate(name="legal_name", domain=("Company",)),
        },
    )
    assert _found(ontology) == [
        ("key-outside-domain", "types.Person.keys[1]", "error"),
        ("unknown-key", "types.Person.keys[2]", "error"),
    ]


def test_an_alias_pointing_at_two_entries_is_ambiguous_within_its_namespace() -> None:
    ontology = Ontology(
        types={
            "Company": EntityType(name="Company", aliases=("Firm",)),
            "Employer": EntityType(name="Employer", aliases=("firm",)),
        },
        predicates={
            # Names claim their surface form too, and case and underscores do not matter.
            "employer": Predicate(name="employer", aliases=("works_for",)),
            "works_for": Predicate(name="works_for"),
            "job": Predicate(name="job", aliases=("Works For",)),
        },
    )
    # A type called Employer and a predicate called employer are not ambiguous.
    assert _found(ontology) == [
        ("duplicate-alias", "types.Employer.aliases[0]", "error"),
        ("duplicate-alias", "predicates.employer.aliases[0]", "error"),
        ("duplicate-alias", "predicates.job.aliases[0]", "error"),
    ]
    assert "already refers to the predicate 'works_for'" in ontology.validate()[1].message


def test_an_entry_whose_name_disagrees_with_its_key_is_an_error() -> None:
    """Facts carry the name, lookups use the key; identity keys would never be stamped."""
    ontology = Ontology(
        types={"Person": EntityType(name="Human")},
        predicates={"age": Predicate(name="years", range="integer")},
    )
    assert _found(ontology) == [
        ("name-mismatch", "types.Person.name", "error"),
        ("name-mismatch", "predicates.age.name", "error"),
    ]


# --------------------------------------------------------------------------- #
# R4: cardinality scope
# --------------------------------------------------------------------------- #

_QUALIFIERS = {"tier": {"identity": True}, "region": {"identity": True}, "as_of": {}}


def test_a_scope_may_only_name_identity_bearing_qualifiers() -> None:
    ontology = Ontology(
        predicates={
            "implied": Predicate(name="implied", qualifiers=_QUALIFIERS),
            "explicit": Predicate(
                name="explicit", qualifiers=_QUALIFIERS, cardinality_scope=("region", "tier")
            ),
            "reconcilable": Predicate(
                name="reconcilable",
                qualifiers=_QUALIFIERS,
                cardinality_scope=("tier", "region", "as_of"),
            ),
            "typo": Predicate(
                name="typo", qualifiers=_QUALIFIERS, cardinality_scope=("teir", "region")
            ),
        }
    )
    assert _found(ontology) == [
        ("scope-not-identity", "predicates.reconcilable.cardinality_scope[2]", "error"),
        ("scope-undeclared-qualifier", "predicates.typo.cardinality_scope[0]", "error"),
    ]
    assert "did you mean 'tier'?" in ontology.validate()[1].message


def test_a_scope_naming_only_some_identity_keys_is_clean() -> None:
    """The identity keys are always in scope, so leaving one out cannot cause a false conflict."""
    partial = Predicate(name="price", qualifiers=_QUALIFIERS, cardinality_scope=("tier",))
    assert Ontology(predicates={"price": partial}).validate() == []
    assert partial.scope_keys == ("region", "tier")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_BROKEN = {
    "types": {"Person": {}},
    "predicates": {"employer": {"domain": ["Person"], "range": "Compnay"}},
}


def test_errors_block_loading_and_say_why(tmp_path: Path) -> None:
    with pytest.raises(OntologyLoadError) as info:
        Ontology.from_dict(_BROKEN)
    assert str(info.value).startswith("predicates.employer.range: 'Compnay' is neither")
    assert str(info.value).endswith("[unknown-range]")
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(_BROKEN), encoding="utf-8")
    with pytest.raises(OntologyLoadError, match=rf"^{path}: predicates\.employer\.range"):
        Ontology.from_json(path)


def test_warnings_do_not_block_loading() -> None:
    loaded = Ontology.from_dict({"types": {"A": {"parents": ["A"]}}})
    assert [d.severity for d in loaded.validate()] == ["warning"]


def test_strict_false_loads_a_broken_schema_so_it_can_be_diagnosed() -> None:
    loaded = Ontology.from_json(json.dumps(_BROKEN), strict=False)
    assert [d.code for d in loaded.validate()] == ["unknown-range"]


def test_a_diagnostic_is_frozen_structured_data() -> None:
    diagnostic = Diagnostic(code="x", path="types.A", message="m", severity="warning")
    with pytest.raises(ValidationError):
        diagnostic.code = "y"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Diagnostic(code="x", path="types.A", message="m", severity="fatal")  # type: ignore[arg-type]
