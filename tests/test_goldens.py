"""Golden ontologies, and the round-trip that every sink and config file rests on.

The goldens live in `tests/fixtures/ontologies/`:

- `minimal.json` — one type, one predicate, names left to default to their keys;
- `deep.json` — eight types, four levels, and a diamond (`Bank` is both a
  `Company` and a `FinancialInstitution`);
- `wide.json` — 240 predicates over twelve types, generated once and checked in.
  It doubles as the snippet-truncation fixture: it is the case the paper's
  scaling story is about;
- `scoped.yaml` — R4's qualifier-scoped cardinality, and the YAML path;
- `pathological.json` — every diagnostic code at least once. It must produce
  diagnostics, never an exception.

The property test generates arbitrary ontologies with `hypothesis` — dangling
ranges, cycles and undeclared scope keys included — because a round-trip that
only holds for well-formed schemas is not the identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from odke.ontology import EntityType, Ontology, OntologyLoadError, Predicate, Qualifier

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
EXAMPLE = Path(__file__).parent.parent / "examples" / "people.ontology.json"
GOLDENS = [*sorted(FIXTURES.glob("*.json")), *sorted(FIXTURES.glob("*.yaml")), EXAMPLE]


def _load(path: Path, *, strict: bool = False) -> Ontology:
    if path.suffix == ".yaml":
        return Ontology.from_yaml(path, strict=strict)
    return Ontology.from_json(path, strict=strict)


def test_every_golden_is_collected() -> None:
    assert {p.name for p in GOLDENS} == {
        "deep.json",
        "minimal.json",
        "pathological.json",
        "people.ontology.json",
        "scoped.yaml",
        "wide.json",
    }


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.name)
def test_every_golden_round_trips_through_json_dict_and_yaml(path: Path) -> None:
    ontology = _load(path)
    assert Ontology.from_json(ontology.model_dump_json(), strict=False) == ontology
    assert Ontology.from_dict(ontology.model_dump(), strict=False) == ontology
    as_yaml = yaml.safe_dump(ontology.model_dump(mode="json"))
    assert Ontology.from_yaml(as_yaml, strict=False) == ontology
    assert ontology.diff(Ontology.from_json(ontology.model_dump_json(), strict=False)) == []


@pytest.mark.parametrize("name", ["minimal.json", "deep.json", "wide.json", "scoped.yaml"])
def test_the_well_formed_goldens_are_clean_and_load_strictly(name: str) -> None:
    path = FIXTURES / name
    assert _load(path).validate() == []
    assert _load(path, strict=True) == _load(path)


def test_deep_inheritance_carries_predicates_down_a_diamond() -> None:
    ontology = _load(FIXTURES / "deep.json")
    assert ontology.lineage("CentralBank") == {
        "CentralBank",
        "Bank",
        "Company",
        "FinancialInstitution",
        "Organization",
        "Agent",
        "Entity",
    }
    assert {p.name for p in ontology.predicates_for("CentralBank")} == {
        "label",
        "founded",
        "legal_name",
        "regulator",
        "swift_code",
        "governor",
    }


def test_the_wide_golden_is_the_truncation_case_the_paper_is_about() -> None:
    """The prompt stays a fixed size as the ontology grows; the tail is what drops."""
    ontology = _load(FIXTURES / "wide.json")
    assert len(ontology.predicates) >= 200
    candidates = ontology.predicates_for("Type01")
    assert len(candidates) > 25
    snippet = ontology.snippet("Type01", limit=25)
    assert snippet.truncated
    assert (
        list(snippet.predicates) == sorted(candidates, key=lambda p: (-p.importance, p.name))[:25]
    )
    # A three-line header and one line per predicate, however wide the schema.
    assert len(snippet.render().splitlines()) == 3 + 25
    assert len(snippet.json_schema()["properties"]) == 25


def test_the_scoped_golden_resolves_cardinality_scope_all_three_ways() -> None:
    predicates = _load(FIXTURES / "scoped.yaml").predicates
    assert predicates["price"].cardinality_scope is None
    assert predicates["price"].scope_keys == ("tier",)
    assert predicates["uptime"].scope_keys == ("percentile", "region")
    assert predicates["hq_country"].scope_keys == ()


ALL_CODES = {
    "duplicate-alias",
    "inheritance-cycle",
    "key-outside-domain",
    "name-mismatch",
    "scope-not-identity",
    "scope-omits-identity",
    "scope-undeclared-qualifier",
    "unknown-domain",
    "unknown-key",
    "unknown-parent",
    "unknown-range",
    "unreachable-predicate",
}

PATHOLOGICAL = [
    ("unknown-key", "types.Company.keys[1]", "error"),
    ("key-outside-domain", "types.Employer.keys[0]", "error"),
    ("unknown-parent", "types.Scientist.parents[0]", "error"),
    ("name-mismatch", "types.Person.name", "error"),
    ("inheritance-cycle", "types.A.parents", "warning"),
    ("inheritance-cycle", "types.Loop.parents", "warning"),
    ("unknown-range", "predicates.employer.range", "error"),
    ("unreachable-predicate", "predicates.orphan.domain", "error"),
    ("unknown-domain", "predicates.half.domain[1]", "warning"),
    ("scope-not-identity", "predicates.price.cardinality_scope[0]", "error"),
    ("scope-undeclared-qualifier", "predicates.price.cardinality_scope[1]", "error"),
    ("scope-omits-identity", "predicates.price.cardinality_scope", "warning"),
    ("duplicate-alias", "types.Employer.aliases[0]", "error"),
    ("duplicate-alias", "predicates.works_for.aliases[0]", "error"),
]


def test_the_pathological_golden_produces_diagnostics_not_exceptions() -> None:
    path = FIXTURES / "pathological.json"
    ontology = _load(path)
    assert [(d.code, d.path, d.severity) for d in ontology.validate()] == PATHOLOGICAL
    assert {code for code, _, _ in PATHOLOGICAL} == ALL_CODES
    # The rest of the library keeps working on it.
    assert ontology.lineage("A") == {"A", "B"}
    assert ontology.snippet("Loop").type_name == "Loop"
    assert ontology.diff(ontology) == []
    # Strict loading refuses it, naming every error and none of the warnings.
    with pytest.raises(OntologyLoadError) as info:
        _load(path, strict=True)
    assert len(info.value.problems) == sum(s == "error" for _, _, s in PATHOLOGICAL)


# --------------------------------------------------------------------------- #
# The property: ontology -> JSON -> ontology is the identity, for any ontology
# --------------------------------------------------------------------------- #

_names = st.text(alphabet="abcXYZ_", min_size=1, max_size=5)
_text = st.none() | st.text(max_size=20)


@st.composite
def _predicates(draw: st.DrawFn, name: str, type_names: list[str]) -> Predicate:
    qualifiers = draw(
        st.dictionaries(
            _names,
            st.builds(Qualifier, identity=st.booleans(), description=_text),
            max_size=3,
        )
    )
    # Ranges and domains draw from real types, literals and a name that exists
    # nowhere, so dangling references are generated as often as valid ones.
    pool = [*type_names, "string", "integer", "date", "Nowhere"]
    scope_pool = [*sorted(qualifiers), "undeclared"]
    return Predicate(
        name=name,
        label=draw(_text),
        description=draw(_text),
        domain=tuple(draw(st.lists(st.sampled_from(pool), max_size=3))),
        range=draw(st.sampled_from(pool)),
        cardinality=draw(st.sampled_from(["single", "multi"])),
        cardinality_scope=draw(
            st.none() | st.lists(st.sampled_from(scope_pool), unique=True, max_size=3).map(tuple)
        ),
        required=draw(st.booleans()),
        qualifiers=qualifiers,
        aliases=tuple(draw(st.lists(_names, max_size=2))),
        importance=draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        examples=tuple(draw(st.lists(st.text(max_size=10), max_size=2))),
    )


@st.composite
def ontologies(draw: st.DrawFn) -> Ontology:
    type_names = draw(st.lists(_names, unique=True, max_size=5))
    parent_pool = [*type_names, "Missing"]
    types = {
        name: EntityType(
            name=name,
            description=draw(_text),
            parents=tuple(draw(st.lists(st.sampled_from(parent_pool), max_size=2))),
            keys=tuple(draw(st.lists(_names, max_size=2))),
            aliases=tuple(draw(st.lists(_names, max_size=2))),
        )
        for name in type_names
    }
    predicate_names = draw(st.lists(_names, unique=True, max_size=6))
    return Ontology(
        name=draw(st.text(max_size=10)),
        version=draw(st.text(max_size=5)),
        types=types,
        predicates={name: draw(_predicates(name, type_names)) for name in predicate_names},
        inferred=draw(st.booleans()),
    )


@settings(max_examples=200, deadline=None)
@given(ontologies())
def test_json_round_trip_is_the_identity_for_any_ontology(ontology: Ontology) -> None:
    assert Ontology.from_json(ontology.model_dump_json(), strict=False) == ontology
    assert Ontology.from_dict(ontology.model_dump(), strict=False) == ontology


@settings(deadline=None)
@given(ontologies())
def test_validate_and_diff_never_raise_on_any_ontology(ontology: Ontology) -> None:
    """A pathological schema produces diagnostics, never an exception."""
    assert all(d.severity in {"error", "warning"} for d in ontology.validate())
    assert ontology.diff(ontology) == []
