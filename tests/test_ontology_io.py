"""Getting a schema in: every path loads the example, and every failure names its key."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from odke.ontology import Ontology, OntologyLoadError, Predicate

EXAMPLE = Path(__file__).parent.parent / "examples" / "people.ontology.json"


def _scoped() -> Ontology:
    """An ontology that uses every field, so a round-trip that drops one fails."""
    return Ontology(
        name="scoped",
        version="3",
        types={"Vendor": {"name": "Vendor", "keys": ("legal_name",), "aliases": ("Supplier",)}},
        predicates={
            "price": Predicate(
                name="price",
                label="Price",
                description="List price.",
                domain=("Vendor",),
                range="number",
                qualifiers={"tier": {"identity": True}, "as_of": {"description": "Quote date."}},
                cardinality_scope=("tier",),
                aliases=("cost",),
                importance=0.9,
                examples=("120.00",),
            ),
            "legal_name": Predicate(name="legal_name", domain=("Vendor",), required=True),
        },
    )


# --------------------------------------------------------------------------- #
# The happy paths
# --------------------------------------------------------------------------- #


def test_the_example_loads_the_same_through_every_path(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    as_yaml = tmp_path / "people.ontology.yaml"
    as_yaml.write_text(yaml.safe_dump(json.loads(text)), encoding="utf-8")

    loaded = [
        Ontology.from_json(EXAMPLE),
        Ontology.from_json(str(EXAMPLE)),
        Ontology.from_json(text),
        Ontology.from_dict(json.loads(text)),
        Ontology.from_yaml(as_yaml),
        Ontology.from_yaml(as_yaml.read_text(encoding="utf-8")),
    ]
    assert all(o == loaded[0] for o in loaded)
    assert loaded[0].name == "people-and-companies"
    assert loaded[0].predicates["employer"].identity_keys == ()


@pytest.mark.parametrize("ontology", [_scoped(), Ontology()], ids=["scoped", "empty"])
def test_json_is_the_identity(ontology: Ontology) -> None:
    """The contract every sink and config file rests on."""
    assert Ontology.from_json(ontology.model_dump_json()) == ontology
    assert Ontology.from_dict(ontology.model_dump()) == ontology
    assert Ontology.from_yaml(yaml.safe_dump(ontology.model_dump())) == ontology


def test_a_name_defaults_to_its_key() -> None:
    """`Person: {name: Person}` says it twice; the second copy is where mismatches come from."""
    loaded = Ontology.from_dict(
        {"types": {"Person": {}}, "predicates": {"age": {"range": "integer"}}}
    )
    assert loaded.types["Person"].name == "Person"
    assert loaded.predicates["age"].name == "age"
    # An explicit name is kept as written; validate() is the judge of a mismatch.
    mismatched = Ontology.from_dict({"types": {"Person": {"name": "Human"}}}, strict=False)
    assert mismatched.types["Person"].name == "Human"
    assert [d.code for d in mismatched.validate()] == ["name-mismatch"]


def test_a_yaml_file_and_a_yaml_string_are_told_apart(tmp_path: Path) -> None:
    path = tmp_path / "o.yml"
    path.write_text("name: from-file\n", encoding="utf-8")
    assert Ontology.from_yaml(path).name == "from-file"
    assert Ontology.from_yaml(str(path)).name == "from-file"
    assert Ontology.from_yaml("name: from-string\ntypes: {}\n").name == "from-string"


# --------------------------------------------------------------------------- #
# The failures, which have to be actionable without reading our source
# --------------------------------------------------------------------------- #


def test_an_unknown_key_is_named_with_a_suggestion() -> None:
    with pytest.raises(OntologyLoadError) as info:
        Ontology.from_dict({"predicates": {"employer": {"rnage": "Company"}}})
    assert str(info.value) == "predicates.employer.rnage: unknown key — did you mean 'range'?"
    # The pydantic exception is replaced, not chained: no forty-line context.
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_an_unknown_key_with_nothing_close_lists_what_is_accepted() -> None:
    with pytest.raises(OntologyLoadError, match=r"types\.Person\.colour: unknown key — EntityType"):
        Ontology.from_dict({"types": {"Person": {"colour": "blue"}}})


def test_a_bad_value_names_the_path_and_what_was_found() -> None:
    with pytest.raises(OntologyLoadError) as info:
        Ontology.from_dict({"predicates": {"employer": {"cardinality": "many"}}})
    message = str(info.value)
    assert message.startswith("predicates.employer.cardinality: ")
    assert "'single'" in message and "'multi'" in message
    assert message.endswith("(got 'many')")


def test_every_problem_is_listed_and_the_file_is_blamed(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text(
        json.dumps(
            {
                "types": {"Person": {"parents": "Agent"}},
                # Not "yes": pydantic reads that as True, and there would be one problem.
                "predicates": {"employer": {"qualifiers": {"role": {"identity": "sometimes"}}}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(OntologyLoadError) as info:
        Ontology.from_json(path)
    lines = str(info.value).splitlines()
    assert lines[0] == f"{path}: 2 problems"
    assert lines[1].startswith("  types.Person.parents: ")
    assert lines[2].startswith("  predicates.employer.qualifiers.role.identity: ")
    assert len(info.value.problems) == 2


def test_a_json_syntax_error_carries_line_and_column(tmp_path: Path) -> None:
    path = tmp_path / "o.json"
    path.write_text('{\n  "name": "x",\n  "types": {\n}', encoding="utf-8")
    with pytest.raises(OntologyLoadError, match=rf"^{path}: line 4 column 2: "):
        Ontology.from_json(path)


def test_a_yaml_syntax_error_carries_line_and_column() -> None:
    with pytest.raises(OntologyLoadError, match=r"^line 2 column \d+: mapping values"):
        Ontology.from_yaml("name: x\ntypes: Person: {}\n")


def test_a_document_that_is_not_a_mapping_says_so() -> None:
    with pytest.raises(OntologyLoadError, match="top level must be a mapping.*not a list"):
        Ontology.from_json("[]")
    with pytest.raises(OntologyLoadError, match="not an empty document"):
        Ontology.from_yaml("# nothing here\n")


def test_a_missing_file_is_a_missing_file_not_a_parse_error(tmp_path: Path) -> None:
    with pytest.raises(OntologyLoadError, match=r"nowhere\.json: no such file"):
        Ontology.from_json(tmp_path / "nowhere.json")
    with pytest.raises(OntologyLoadError, match="no such file"):
        Ontology.from_yaml("nowhere.yaml")


def test_yaml_without_pyyaml_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """DECISIONS #1: the base install must fail with the fix in the message."""
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match=r'pip install "odke\[yaml\]"'):
        Ontology.from_yaml("name: x\n")
