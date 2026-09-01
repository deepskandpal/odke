"""Snippet generation — the paper's scaling story, tested as a contract."""

from __future__ import annotations

from odke.ontology import EntityType, Ontology, Predicate


def _ontology() -> Ontology:
    return Ontology(
        name="demo",
        types={
            "Person": EntityType(name="Person", description="A human being."),
            "Scientist": EntityType(name="Scientist", parents=("Person",)),
            "Company": EntityType(name="Company"),
        },
        predicates={
            "birth_date": Predicate(
                name="birth_date", domain=("Person",), range="date", importance=0.9
            ),
            "employer": Predicate(
                name="employer", domain=("Person",), range="Company", importance=0.7
            ),
            "field": Predicate(name="field", domain=("Scientist",), range="string", importance=0.4),
            "founded": Predicate(name="founded", domain=("Company",), range="date"),
            "label": Predicate(name="label", range="string", importance=1.0),
        },
    )


def test_predicates_are_inherited_from_parent_types() -> None:
    """A Scientist is a Person, so it gets Person's predicates too."""
    names = {p.name for p in _ontology().predicates_for("Scientist")}
    assert {"birth_date", "employer", "field"} <= names
    assert "founded" not in names


def test_a_domainless_predicate_applies_everywhere() -> None:
    """The useful default for an inferred ontology, where domains are least reliable."""
    for type_name in ("Person", "Company", "Scientist"):
        assert "label" in {p.name for p in _ontology().predicates_for(type_name)}


def test_snippet_ranks_by_importance_and_truncates() -> None:
    """The prompt stays a fixed size as the ontology grows; the tail is what drops."""
    snippet = _ontology().snippet("Scientist", limit=2)
    assert [p.name for p in snippet.predicates] == ["label", "birth_date"]
    assert snippet.truncated


def test_snippet_is_not_marked_truncated_when_everything_fits() -> None:
    assert not _ontology().snippet("Company", limit=25).truncated


def test_render_is_stable_across_calls() -> None:
    """An unstable prompt defeats prompt caching and makes runs unrepeatable."""
    ontology = _ontology()
    assert ontology.snippet("Person").render() == ontology.snippet("Person").render()


def test_render_mentions_type_description_and_ranges() -> None:
    text = _ontology().snippet("Person").render()
    assert "Entity type: Person" in text
    assert "A human being." in text
    assert "birth_date (date, single)" in text


def test_json_schema_mirrors_the_rendered_snippet() -> None:
    """Both renderings come off one object, so they cannot drift apart."""
    snippet = _ontology().snippet("Person")
    schema = snippet.json_schema()
    assert set(schema["properties"]) == {p.name for p in snippet.predicates}
    assert schema["properties"]["birth_date"]["type"] == "string"
    assert schema["additionalProperties"] is False


def test_lineage_survives_a_cycle() -> None:
    """Hand-written ontologies contain cycles; that must not hang the extractor."""
    ontology = Ontology(
        types={
            "A": EntityType(name="A", parents=("B",)),
            "B": EntityType(name="B", parents=("A",)),
        }
    )
    assert ontology.lineage("A") == {"A", "B"}


def test_multi_cardinality_becomes_an_array_in_the_schema() -> None:
    ontology = Ontology(
        predicates={"alias": Predicate(name="alias", cardinality="multi", range="string")}
    )
    schema = ontology.snippet("Anything").json_schema()
    assert schema["properties"]["alias"]["type"] == "array"
