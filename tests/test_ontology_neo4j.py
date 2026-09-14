"""Ontology.from_neo4j: a live graph's schema, ranked by how much the graph uses it.

No database runs here: a recording driver replays the schema procedures'
recorded output. `test_reflecting_a_live_neo4j` reads a real server when
`NEO4J_URI` is set.
"""

from __future__ import annotations

import json
import math
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

import pytest

from odke import (
    Entity,
    Fact,
    KnowledgeGraph,
    Ontology,
    OntologyImportWarning,
    OntologyLoadError,
    Qualifier,
)
from odke.ontology.from_neo4j import (
    NODE_TYPE_PROPERTIES,
    REL_TYPE_PROPERTIES,
    SCHEMA_VISUALIZATION,
    rel_type_name,
    value_type,
)
from odke.sinks.neo4j import Neo4jSink
from test_neo4j_sink import FakeDriver

RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "neo4j" / "schema.json").read_text(encoding="utf-8")
)


def _driver(recorded: dict[str, Any] = RECORDED, **extra: list[dict[str, Any]]) -> FakeDriver:
    """A driver answering the procedures and the count queries from a recording."""
    answers: dict[str, list[dict[str, Any]]] = {
        "db.schema.nodeTypeProperties": recorded["nodeTypeProperties"],
        "db.schema.relTypeProperties": recorded["relTypeProperties"],
        "db.schema.visualization": [recorded["visualization"]],
        **extra,
    }
    for rel, n in recorded.get("relationshipCounts", {}).items():
        answers[f"[r:`{rel}`]->()"] = [{"n": n}]
    for label, row in recorded.get("propertyCounts", {}).items():
        answers[f"MATCH (n:`{label}`)"] = [row]
    return FakeDriver(answers)


# --------------------------------------------------------------------------- #
# A graph Neo4jSink wrote
# --------------------------------------------------------------------------- #


def test_a_populated_graph_ranks_its_most_used_predicates_first() -> None:
    ontology = Ontology.from_neo4j(_driver())
    assert [p.name for p in ontology.snippet("Company").predicates] == ["uptime", "founded"]
    assert [p.name for p in ontology.snippet("Person").predicates] == ["employer", "name", "born"]


def test_importance_is_each_count_log_scaled_against_the_most_used() -> None:
    predicates = Ontology.from_neo4j(_driver()).predicates
    top = math.log1p(5)
    assert predicates["uptime"].importance == 1.0
    assert predicates["employer"].importance == pytest.approx(math.log1p(3) / top)
    assert predicates["founded"].importance == pytest.approx(math.log1p(2) / top)
    assert predicates["born"].importance == pytest.approx(math.log1p(1) / top)


def test_the_shape_neo4j_sink_writes_reads_back_as_that_shape() -> None:
    ontology = Ontology.from_neo4j(_driver(), name="hr", version="3")
    assert (ontology.name, ontology.version) == ("hr", "3")
    # The sink's own labels are not types, nor its fields predicates, nor its links.
    assert set(ontology.types) == {"Person", "Company"}
    assert set(ontology.predicates) == {"employer", "founded", "name", "uptime", "born"}
    shapes = {
        n: (p.domain, p.range, p.cardinality, p.required, dict(p.qualifiers))
        for n, p in ontology.predicates.items()
    }
    assert shapes == {
        # An edge; its non-provenance properties are reconcilable qualifiers.
        "employer": (
            ("Person",),
            "Company",
            "multi",
            False,
            {"signature": Qualifier(), "start_time": Qualifier()},
        ),
        # Claims plus their projection: one predicate, typed by the projected value.
        "founded": (("Company",), "date", "single", True, {}),
        "name": (("Person",), "string", "single", True, {}),
        # Claims never projected, because they are scoped.
        "uptime": (("Company",), "string", "multi", False, {"percentile": Qualifier()}),
        # A plain node property.
        "born": (("Person",), "integer", "single", False, {}),
    }
    assert ontology.validate() == []


def test_only_the_stable_schema_procedures_and_counts_are_run_and_nothing_writes() -> None:
    driver = _driver()
    Ontology.from_neo4j(driver, database="kg")
    cyphers = [cypher for _, cypher, _ in driver.calls]
    assert cyphers[:3] == [NODE_TYPE_PROPERTIES, REL_TYPE_PROPERTIES, SCHEMA_VISUALIZATION]
    assert all(mode == "read" for mode, _, _ in driver.calls)
    assert all(c.startswith(("CALL db.schema.", "MATCH ")) for c in cyphers)
    assert not any(word in c for c in cyphers for word in ("CREATE", "MERGE", "SET ", "DELETE"))
    assert "MATCH ()-[r:`uptime`]->() RETURN count(r) AS n" in cyphers
    assert "MATCH (n:`Person`) RETURN count(n.`born`) AS `born`, count(n.`name`) AS `name`" in (
        cyphers
    )
    assert driver.sessions == [{"database": "kg"}]
    assert Ontology.from_neo4j(_driver(), database="kg").name == "kg"


# --------------------------------------------------------------------------- #
# Any graph
# --------------------------------------------------------------------------- #

MOVIES: dict[str, Any] = {
    "nodeTypeProperties": [
        {
            "nodeLabels": ["Person"],
            "propertyName": "name",
            "propertyTypes": ["String"],
            "mandatory": True,
        },
        {
            "nodeLabels": ["Person"],
            "propertyName": "born",
            "propertyTypes": ["Long"],
            "mandatory": False,
        },
        {
            "nodeLabels": ["Movie"],
            "propertyName": "title",
            "propertyTypes": ["STRING"],
            "mandatory": True,
        },
        {
            "nodeLabels": ["Movie"],
            "propertyName": "rating",
            "propertyTypes": ["Long", "Double"],
            "mandatory": False,
        },
        {
            "nodeLabels": ["Movie"],
            "propertyName": "tags",
            "propertyTypes": ["LIST<STRING NOT NULL>"],
            "mandatory": False,
        },
        {"nodeLabels": ["Genre"], "propertyName": None, "propertyTypes": None, "mandatory": False},
    ],
    "relTypeProperties": [
        {
            "relType": ":`ACTED_IN`",
            "propertyName": "roles",
            "propertyTypes": ["StringArray"],
            "mandatory": False,
        },
        {
            "relType": ":`SIMILAR`",
            "propertyName": "score",
            "propertyTypes": ["Double"],
            "mandatory": True,
        },
    ],
    "visualization": {
        "nodes": [{"name": "Person"}, {"name": "Movie"}, {"name": "Genre"}],
        "relationships": [
            [{"name": "Person"}, "ACTED_IN", {"name": "Movie"}],
            [{"name": "Movie"}, "SIMILAR", {"name": "Movie"}],
        ],
    },
    "relationshipCounts": {"ACTED_IN": 172, "SIMILAR": 4},
    "propertyCounts": {"Person": {"name": 133, "born": 128}, "Movie": {"title": 38, "rating": 3}},
}


def test_a_graph_odke_never_wrote_maps_labels_relationships_and_properties() -> None:
    ontology = Ontology.from_neo4j(_driver(MOVIES))
    assert set(ontology.types) == {"Person", "Movie", "Genre"}
    shapes = {n: (p.domain, p.range, p.cardinality) for n, p in ontology.predicates.items()}
    assert shapes == {
        "ACTED_IN": (("Person",), "Movie", "multi"),
        # Not odke's link: it has no created_at, so it is somebody's predicate.
        "SIMILAR": (("Movie",), "Movie", "multi"),
        "name": (("Person",), "string", "single"),
        "born": (("Person",), "integer", "single"),
        "title": (("Movie",), "string", "single"),
        "rating": (("Movie",), "number", "single"),
        "tags": (("Movie",), "string", "multi"),
    }
    assert "roles" in ontology.predicates["ACTED_IN"].qualifiers
    assert ontology.predicates["name"].required and not ontology.predicates["born"].required
    assert [p.name for p in ontology.snippet("Movie").predicates][:2] == ["title", "SIMILAR"]


def test_value_types_are_read_in_both_spellings() -> None:
    assert value_type("String") == ("string", False)
    assert value_type("StringArray") == ("string", True)
    assert value_type("LIST<STRING NOT NULL>") == ("string", True)
    assert value_type("Long") == ("integer", False)
    assert value_type("INTEGER") == ("integer", False)
    assert value_type("DoubleArray") == ("number", True)
    assert value_type("DateTime") == ("datetime", False)
    assert value_type("ZONED DATETIME") == ("datetime", False)
    assert value_type("Date") == ("date", False)
    assert value_type("Point") == (None, False)
    assert rel_type_name(":`WORKS AT`") == "WORKS AT"
    assert rel_type_name(":`a``b`") == "a`b"
    assert rel_type_name("KNOWS") == "KNOWS"


def test_an_empty_graph_is_an_empty_ontology() -> None:
    empty = {"nodeTypeProperties": [], "relTypeProperties": [], "visualization": {}}
    ontology = Ontology.from_neo4j(_driver(empty))
    assert (ontology.types, ontology.predicates, ontology.name) == ({}, {}, "neo4j")


# --------------------------------------------------------------------------- #
# What cannot be mapped
# --------------------------------------------------------------------------- #

AWKWARD: dict[str, Any] = {
    "nodeTypeProperties": [
        {
            "nodeLabels": ["Place"],
            "propertyName": "location",
            "propertyTypes": ["Point"],
            "mandatory": True,
        },
        {
            "nodeLabels": ["Place"],
            "propertyName": "code",
            "propertyTypes": ["String", "Boolean"],
            "mandatory": True,
        },
        {
            "nodeLabels": ["Person"],
            "propertyName": "LIKES",
            "propertyTypes": ["String"],
            "mandatory": False,
        },
    ],
    "relTypeProperties": [{"relType": ":`LIKES`", "propertyName": None}],
    "visualization": {
        "nodes": [{"name": "Place"}, {"name": "Person"}, {"name": "Company"}],
        "relationships": [
            [{"name": "Person"}, "LIKES", {"name": "Place"}],
            [{"name": "Person"}, "LIKES", {"name": "Company"}],
        ],
    },
    "relationshipCounts": {"LIKES": 5},
    "propertyCounts": {"Place": {"location": 2, "code": 2}},
}


def _awkward() -> FakeDriver:
    return _driver(
        AWKWARD,
        **{
            "[r:`LIKES`]->(:`Place`)": [{"n": 1}],
            "[r:`LIKES`]->(:`Company`)": [{"n": 4}],
        },
    )


def test_what_cannot_be_mapped_is_reported_and_strict_refuses_it() -> None:
    with pytest.raises(OntologyLoadError) as info:
        Ontology.from_neo4j(_awkward())
    problems = info.value.problems
    assert len(problems) == 4, problems
    assert any(
        "'LIKES': ends at Company, Place" in p and "'Company', the most used" in p for p in problems
    )
    assert any("'location': holds Point" in p for p in problems)
    assert any("'code': holds values of several types (boolean, string)" in p for p in problems)
    assert any("property 'LIKES': is also a relationship type" in p for p in problems)


def test_not_strict_loads_the_rest_and_warns_once() -> None:
    with pytest.warns(OntologyImportWarning) as caught:
        ontology = Ontology.from_neo4j(_awkward(), strict=False)
    assert len([w for w in caught if isinstance(w.message, OntologyImportWarning)]) == 1
    assert ontology.predicates["LIKES"].range == "Company"
    assert ontology.predicates["location"].range == "string"


def test_a_uri_connects_through_the_neo4j_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "neo4j", None)
    with pytest.raises(ImportError, match=r"odke\[neo4j\]"):
        Ontology.from_neo4j("bolt://localhost:7687", auth=("neo4j", "unused"))


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_reflecting_a_live_neo4j() -> None:
    """Write a graph whose use is known, reflect it, and read the ranking back."""
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    person, company = f"Person_{suffix}", f"Company_{suffix}"
    employer, uptime, motto = f"employer_{suffix}", f"uptime_{suffix}", f"motto_{suffix}"
    ada = Entity(key=f"p:ada:{suffix}", type=person)
    grace = Entity(key=f"p:grace:{suffix}", type=person)
    acme = Entity(key=f"c:acme:{suffix}", type=company)
    facts = (
        Fact(subject=ada, predicate=employer, object_entity=acme),
        Fact(subject=grace, predicate=employer, object_entity=acme),
        *(
            Fact(
                subject=acme,
                predicate=uptime,
                object_value=value,
                qualifiers={"percentile": p},
                identity_keys=("percentile",),
            )
            for p, value in (("p50", "99.9%"), ("p95", "99.5%"), ("p99", "99.0%"))
        ),
        Fact(subject=acme, predicate=motto, object_value="Anvils"),
    )
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth) as sink:
        driver = sink._driver
        try:
            sink.write(KnowledgeGraph(facts=facts))
            with warnings.catch_warnings():
                # A shared server holds other graphs; their problems are not this test's.
                warnings.simplefilter("ignore", OntologyImportWarning)
                ontology = Ontology.from_neo4j(driver, strict=False)
            assert {person, company} <= set(ontology.types)
            edge = ontology.predicates[employer]
            assert (edge.domain, edge.range) == ((person,), company)
            assert "percentile" in ontology.predicates[uptime].qualifiers
            assert ontology.predicates[motto].cardinality == "single"
            mine = [p.name for p in ontology.snippet(company).predicates if p.name.endswith(suffix)]
            assert mine == [uptime, motto]
        finally:
            driver.execute_query(
                f"MATCH (n) WHERE n:`{person}` OR n:`{company}` OR "
                f"(n:Claim AND n.subject_type IN ['{person}', '{company}']) DETACH DELETE n"
            )
