"""Snippet generation — the paper's scaling story, tested as a contract."""

from __future__ import annotations

from odke.ontology import EntityType, Ontology, Predicate, Qualifier


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


def test_the_ontology_declares_which_qualifiers_bear_identity() -> None:
    """Identity keys come out sorted so they can be stamped straight onto a fact."""
    ontology = Ontology(
        predicates={
            "has_uptime": Predicate(
                name="has_uptime",
                qualifiers={
                    "tier": Qualifier(identity=True),
                    "percentile": {"identity": True},
                    "start_time": {},
                },
            )
        }
    )
    assert ontology.identity_keys("has_uptime") == ("percentile", "tier")
    assert ontology.identity_keys("not_declared") == ()


def test_a_bare_list_of_qualifier_names_is_still_accepted_and_reconcilable() -> None:
    """The shorter spelling keeps working, and every name in it keeps DECISIONS #11."""
    predicate = Predicate(name="employer", qualifiers=["start_date", "end_date", "role"])
    assert set(predicate.qualifiers) == {"start_date", "end_date", "role"}
    assert predicate.identity_keys == ()
    assert not any(q.identity for q in predicate.qualifiers.values())


def test_qualifier_names_still_render_into_the_snippet() -> None:
    ontology = Ontology(
        predicates={"employer": Predicate(name="employer", qualifiers=["start_date", "role"])}
    )
    assert "[qualifiers: start_date, role]" in ontology.snippet("Person").render()


def test_multi_cardinality_becomes_an_array_in_the_schema() -> None:
    ontology = Ontology(
        predicates={"alias": Predicate(name="alias", cardinality="multi", range="string")}
    )
    schema = ontology.snippet("Anything").json_schema()
    assert schema["properties"]["alias"]["type"] == "array"


# --------------------------------------------------------------------------- #
# Cardinality scope (R4)
# --------------------------------------------------------------------------- #

_SCOPED_QUALIFIERS = {"tier": {"identity": True}, "region": {"identity": True}, "as_of": {}}


def test_the_scope_always_includes_the_identity_bearing_qualifiers() -> None:
    """'One price per subject per tier' needs no second declaration: identity implies it."""
    price = Predicate(name="price", qualifiers=_SCOPED_QUALIFIERS)
    assert price.cardinality_scope == ()
    assert price.scope_keys == ("region", "tier")
    assert Predicate(name="flat").scope_keys == ()


def test_a_declared_scope_adds_keys_and_can_never_remove_an_identity_key() -> None:
    """A partial list cannot make two different claims look like a conflict."""
    partial = Predicate(name="price", qualifiers=_SCOPED_QUALIFIERS, cardinality_scope=("tier",))
    assert partial.scope_keys == ("region", "tier")
    # Judging a declared key is validate()'s job; the union takes it at its word.
    assert Predicate(name="p", cardinality_scope=("b", "a")).scope_keys == ("a", "b")
    assert Predicate.model_validate_json(partial.model_dump_json()) == partial


def test_scope_keys_is_exactly_what_the_neo4j_check_groups_by() -> None:
    """R4 is declared here and compiled in M4; the two must not drift apart."""
    from odke.sinks.neo4j import cardinality_scope

    for predicate in (
        Predicate(name="flat"),
        Predicate(name="implied", qualifiers=_SCOPED_QUALIFIERS),
        Predicate(name="partial", qualifiers=_SCOPED_QUALIFIERS, cardinality_scope=("tier",)),
        Predicate(name="extra", qualifiers=_SCOPED_QUALIFIERS, cardinality_scope=("as_of",)),
    ):
        assert cardinality_scope(predicate) == predicate.scope_keys, predicate.name
