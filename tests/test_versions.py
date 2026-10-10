"""The ontology that checked each fact, and the schema slice its extractor saw (#163).

The done-when: a sink writes the version, and a reader can filter by it. Every
sink here writes the graph the pipeline stamped and reads `odke.ontology` back;
the filter runs on facts, through a recording driver, and against a live Neo4j
when `NEO4J_URI` is set. Model calls answer from recorded responses.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import (
    Document,
    Entity,
    Evidence,
    Fact,
    KnowledgeGraph,
    Ontology,
    Pipeline,
    Span,
    Validator,
)
from openodke.cli.main import app
from openodke.corroborate import ONTOLOGY, SCHEMA_SLICE, checked_by, checked_under
from openodke.eval.spans import load_facts
from openodke.sinks.bulk import CypherFileSink, Neo4jAdminCsvSink
from openodke.sinks.jsonl import JsonlSink
from openodke.sinks.neo4j import (
    SHOW_INDEXES,
    Neo4jConstrainer,
    Neo4jSink,
    checked_cypher,
    stored_qualifiers,
)
from openodke.sinks.networkx import NetworkXSink
from openodke.sinks.rdf import DEFAULT_BASE, RdfSink
from test_neo4j_sink import FakeDriver
from test_reconcile import _operators

runner = CliRunner()
TEXT = "Ada Lovelace worked for Acme from 1833. Acme is based in London."
DOC = Document(id="d1", text=TEXT)
OLD = Ontology.from_dict(
    {
        "name": "people",
        "version": "1",
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company"},
            "hq": {"domain": ["Company"]},
        },
    }
)
# The same predicates, one of them narrowed: a breaking change, and a new fingerprint.
NEW = Ontology.from_dict(
    {
        **OLD.model_dump(mode="json", include={"name", "types"}),
        "version": "2",
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company", "cardinality": "multi"},
            "hq": {"domain": ["Company"]},
        },
    }
)
ADA = Entity(key="p:ada", type="Person", label="Ada Lovelace")
ACME = Entity(key="c:acme", type="Company", label="Acme")


def _cited(quote: str) -> Evidence:
    start = TEXT.index(quote)
    return Evidence(doc_id="d1", span=Span(doc_id="d1", start=start, end=start + len(quote)))


FACTS = [
    Fact(
        subject=ADA,
        predicate="employer",
        object_entity=ACME,
        evidence=(_cited("Ada Lovelace worked for Acme"),),
    ),
    Fact(
        subject=ACME,
        predicate="hq",
        object_value="London",
        evidence=(_cited("Acme is based in London"),),
    ),
]


class _Replay:
    """The facts above, handed back as an extractor's."""

    def extract(self, chunk: Any, ontology: Ontology) -> list[Fact]:
        return list(FACTS) if chunk.index == 0 else []


def _graph(ontology: Ontology = OLD) -> KnowledgeGraph:
    return Pipeline(ontology, _Replay()).run([DOC])


# --------------------------------------------------------------------------- #
# Every fact, stamped
# --------------------------------------------------------------------------- #


def test_every_fact_the_gate_lets_through_names_the_ontology_that_checked_it() -> None:
    kg = _graph()
    assert len(kg.facts) == 2
    assert {checked_by(fact) for fact in kg.facts} == {OLD.fingerprint}
    assert {fact.qualifiers[ONTOLOGY] for fact in kg.facts} == {OLD.fingerprint}
    # Not part of a fact's identity: the same claim, whatever checked it.
    assert {f.signature for f in kg.facts} == {f.signature for f in FACTS}
    # A schema with nothing in it checks nothing, and stamps nothing.
    unchecked = Pipeline(Ontology(), _Replay()).run([DOC]).facts
    assert unchecked and all(ONTOLOGY not in f.qualifiers for f in unchecked)


def test_a_fact_checked_again_names_the_ontology_that_checked_it_last() -> None:
    first = _graph(OLD)
    kg, report = Validator(NEW, grounder=_Supported()).validate(list(first.facts), [DOC])
    assert report.facts_out == 2
    assert {checked_by(fact) for fact in kg.facts} == {NEW.fingerprint}


class _Supported:
    """A grounder that finds every fact supported, so the job needs no model."""

    def ground(self, fact: Fact, doc: Document) -> Fact:
        from openodke import GroundingVerdict

        return fact.model_copy(update={"verdict": GroundingVerdict.SUPPORTED})


def test_odke_run_stamps_every_fact_and_the_model_path_the_slice_it_saw(example: Path) -> None:
    result = runner.invoke(app, ["run", str(example / "odke.yaml")])
    assert result.exit_code == 0, result.output
    facts = load_facts(example / "out")
    ontology = Ontology.from_json(example / "ontology.json")
    assert {checked_by(fact) for fact in facts} == {ontology.fingerprint}
    sliced = [fact for fact in facts if SCHEMA_SLICE in fact.qualifiers]
    # The facts the model extracted carry the slice it was shown for their subject's type.
    assert sliced and all(
        fact.qualifiers[SCHEMA_SLICE] == ontology.snippet(fact.subject.type).fingerprint
        for fact in sliced
    )


# --------------------------------------------------------------------------- #
# Every sink writes it, and reads it back where it reads
# --------------------------------------------------------------------------- #


def test_every_sink_writes_the_ontology_that_checked_each_fact(tmp_path: Path) -> None:
    kg = _graph()
    version = OLD.fingerprint

    JsonlSink(tmp_path / "jsonl").write(kg)
    assert {checked_by(f) for f in load_facts(tmp_path / "jsonl")} == {version}

    driver = FakeDriver()
    Neo4jSink(driver=driver, ontology=OLD).write(kg)
    rows = [row for _, params in driver.writes for row in params.get("rows", ())]
    props = [row["props"] for row in rows if "props" in row and "signature" in row["props"]]
    assert len(props) == 2 and {p[ONTOLOGY] for p in props} == {version}
    assert {stored_qualifiers(p)[ONTOLOGY] for p in props} == {version}

    graph = NetworkXSink(ontology=OLD).to_graph(kg)
    assert {data[ONTOLOGY] for _, _, data in graph.edges(data=True)} == {version}

    from rdflib import Literal, URIRef

    rdf = RdfSink(tmp_path / "g.ttl", ontology=OLD).graph(kg)
    stamped = URIRef(f"{DEFAULT_BASE}schema/qualifier/{ONTOLOGY}")
    assert set(rdf.objects(None, stamped)) == {Literal(version)}

    CypherFileSink(tmp_path / "graph.cypher", ontology=OLD).write(kg)
    assert (tmp_path / "graph.cypher").read_text(encoding="utf-8").count(version) == 2

    Neo4jAdminCsvSink(tmp_path / "csv", ontology=OLD).write(kg)
    found = [
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "csv").glob("*.csv")
        if version in path.read_text(encoding="utf-8")
    ]
    assert found and all(ONTOLOGY in text.splitlines()[0] for text in found)


# --------------------------------------------------------------------------- #
# A reader filters by it
# --------------------------------------------------------------------------- #


def test_the_facts_one_ontology_checked_are_a_filter_away() -> None:
    old, new = _graph(OLD).facts, _graph(NEW).facts
    mixed = [*old, new[0]]
    assert checked_under(mixed, OLD) == list(old)
    assert checked_under(mixed, NEW.fingerprint) == [new[0]]
    assert checked_under(FACTS, OLD) == []


def _index(predicate: str) -> dict[str, Any]:
    return {
        "name": f"odke_ontology_{predicate}",
        "type": "RANGE",
        "entityType": "RELATIONSHIP",
        "labelsOrTypes": [predicate],
        "properties": [ONTOLOGY],
    }


def test_neo4j_finds_them_through_its_index_and_reads_no_type_without_one() -> None:
    stored = {
        "predicate": "employer",
        "props": {"signature": "x", "polarity": "asserted", ONTOLOGY: OLD.fingerprint},
        "subject_key": "p:ada",
        "subject_labels": ["Person", "Entity"],
        "object_key": "c:acme",
        "object_labels": ["Company", "Entity"],
        "value": None,
    }
    driver = FakeDriver(
        {
            SHOW_INDEXES: [_index("employer")],
            "db.relationshipTypes": [{"name": "employer"}, {"name": "hq"}, {"name": "SAME_AS"}],
            "= $fingerprint": [stored],
        }
    )
    with pytest.warns(UserWarning, match=r"no index for hq\.odke\.ontology"):
        found = Neo4jSink(driver=driver).checked_under(OLD)
    assert [(f.subject.key, f.predicate, checked_by(f)) for f in found] == [
        ("p:ada", "employer", OLD.fingerprint)
    ]
    reads = [(cypher, params) for mode, cypher, params in driver.calls if mode == "read"]
    assert reads == [(checked_cypher("employer"), {"fingerprint": OLD.fingerprint})]
    assert reads[0][0].startswith(
        "MATCH (s)-[r:`employer`]->(o) USING INDEX r:`employer`(`odke.ontology`)\n"
        "WHERE r.`odke.ontology` = $fingerprint\n"
    )
    # The index bootstrap creates is the one read through.
    ddl = Neo4jConstrainer().schema(OLD)
    assert (
        "CREATE INDEX odke_ontology_employer IF NOT EXISTS "
        "FOR ()-[r:`employer`]-() ON (r.`odke.ontology`)"
    ) in ddl


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_the_filter_against_a_live_neo4j() -> None:
    """Bootstrapped, written under two ontologies, and read back by each, through the index."""
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    names = {name: f"{name}_{suffix}" for name in ("Person", "Company", "employer", "hq")}

    def named(ontology: Ontology) -> Ontology:
        data = json.loads(json.dumps(ontology.model_dump(mode="json")))
        text = json.dumps(data)
        for old, new in names.items():
            text = re.sub(rf'"{old}"', f'"{new}"', text)
        return Ontology.from_dict(json.loads(text))

    def renamed(fact: Fact) -> Fact:
        subject = fact.subject.model_copy(
            update={"type": names[fact.subject.type], "key": f"{fact.subject.key}:{suffix}"}
        )
        obj = fact.object_entity
        if obj is not None:
            obj = obj.model_copy(update={"type": names[obj.type], "key": f"{obj.key}:{suffix}"})
        return fact.model_copy(
            update={"subject": subject, "object_entity": obj, "predicate": names[fact.predicate]}
        )

    old, new = named(OLD), named(NEW)
    first = [renamed(f) for f in FACTS]
    kg_old = Pipeline(old, _Listed(first[:1])).run([DOC])
    kg_new = Pipeline(new, _Listed(first[1:])).run([DOC])
    mine = (
        f"MATCH (n) WHERE n:`{names['Person']}` OR n:`{names['Company']}` OR "
        f"(n:Claim AND n.subject_type = '{names['Company']}') "
    )
    created = [
        re.search(r"CREATE (?:FULLTEXT )?(CONSTRAINT|INDEX) (\S+) IF NOT EXISTS", s)
        for s in Neo4jConstrainer().schema(new)
    ]
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth, ontology=new) as sink:
        driver = sink._driver
        try:
            sink.bootstrap(new)
            driver.execute_query("CALL db.awaitIndexes(300)")
            sink.write(kg_old)
            sink.write(kg_new)
            under_old = sink.checked_under(old)
            under_new = sink.checked_under(new.fingerprint)
            assert [(f.predicate, checked_by(f)) for f in under_old] == [
                (names["employer"], old.fingerprint)
            ]
            assert [(f.predicate, f.object_value) for f in under_new] == [(names["hq"], "London")]
            with driver.session() as session:
                plan = (
                    session.run(
                        "EXPLAIN " + checked_cypher(names["employer"]),
                        {"fingerprint": old.fingerprint},
                    )
                    .consume()
                    .plan
                )
            operators = _operators(plan)
            assert not any("Scan" in op for op in operators), operators
        finally:
            driver.execute_query(mine + "DETACH DELETE n")
            for match in created:
                if match and suffix in match.group(2):
                    driver.execute_query(f"DROP {match.group(1)} {match.group(2)} IF EXISTS")


class _Listed:
    def __init__(self, facts: list[Fact]) -> None:
        self.facts = facts

    def extract(self, chunk: Any, ontology: Ontology) -> list[Fact]:
        return list(self.facts) if chunk.index == 0 else []
