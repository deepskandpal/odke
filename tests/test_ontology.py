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


def test_cardinality_scope_defaults_to_the_identity_bearing_qualifiers() -> None:
    """'One price per subject per tier' needs no second declaration: identity implies it."""
    price = Predicate(name="price", qualifiers=_SCOPED_QUALIFIERS)
    assert price.cardinality_scope is None
    assert price.scope_keys == ("region", "tier")
    assert Predicate(name="flat").scope_keys == ()


def test_an_empty_scope_counts_per_subject_regardless_of_qualifiers() -> None:
    """`()` and `None` are different answers, and both must survive JSON."""
    flat = Predicate(name="price", qualifiers=_SCOPED_QUALIFIERS, cardinality_scope=())
    assert flat.scope_keys == ()
    assert Predicate.model_validate_json(flat.model_dump_json()).cardinality_scope == ()
    implied = Predicate(name="price", qualifiers=_SCOPED_QUALIFIERS)
    assert Predicate.model_validate_json(implied.model_dump_json()).cardinality_scope is None


def test_an_explicit_scope_is_what_the_constraint_compiler_reads() -> None:
    """Spelled out, sorted, and taken at its word — judging it is validate()'s job."""
    price = Predicate(name="price", qualifiers=_SCOPED_QUALIFIERS, cardinality_scope=("tier",))
    assert price.scope_keys == ("tier",)
    assert Predicate(name="p", cardinality_scope=("b", "a")).scope_keys == ("a", "b")
