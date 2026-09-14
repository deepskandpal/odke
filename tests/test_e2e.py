"""The end-to-end example, run the way a newcomer runs it.

`examples/e2e/` is a small invented corpus — a register extract, a staff list, a
factsheet and two trade-press notes about fictional companies — with an
ontology, hand-authored model responses, one stage of its own and two configs.
The tests here run it into JSON Lines on those recorded responses, always.

`test_the_example_end_to_end_into_a_live_neo4j` runs the Neo4j config into a
real server when `NEO4J_URI` is set (with `NEO4J_USER` and `NEO4J_PASSWORD`,
as the other live tests read them), then runs the README's queries against
what it wrote. It refuses to write into a database that already holds
`Company` or `Person` nodes, and leaves the database as it found it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Ontology
from openodke.cli.main import app
from openodke.eval.sinks import assert_idempotent
from openodke.run import execute, load_config
from openodke.sinks import neo4j as neo4j_module
from openodke.sinks.neo4j import CHECK_MARKER, Neo4jConstrainer

# `example`, a copy of examples/e2e/ to run in, is in conftest.py.
EXAMPLE = Path(__file__).parent.parent / "examples" / "e2e"
HALDEN_NOTE = "corpus/notes/halden-robotics.md"
runner = CliRunner()


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def queries() -> list[str]:
    """The numbered queries in `queries.cypher`, comments stripped, without the `;`."""
    text = (EXAMPLE / "queries.cypher").read_text(encoding="utf-8")
    found = []
    for block in text.split("\n\n"):
        body = [line for line in block.splitlines() if not line.startswith("//")]
        if body:
            found.append("\n".join(body).rstrip(";"))
    return found


# --------------------------------------------------------------------------- #
# Recorded responses, into JSON Lines
# --------------------------------------------------------------------------- #


def test_the_example_runs_end_to_end_on_recorded_responses(example: Path) -> None:
    result = runner.invoke(app, ["run", str(example / "odke.yaml")])
    assert result.exit_code == 0, result.output
    out = example / "out"
    stats = json.loads((out / "manifest.json").read_text())["stats"]

    # Eight documents: three register rows, two staff rows, a factsheet, two notes.
    assert (stats["documents"], stats["chunks"], stats["refused"]) == (8, 8, 1)
    stages = stats["stages"]
    assert stages["extractor"]["rejections"] == {"quote not in the passage": 1}
    paths = stages["extractor"]["paths"]
    assert (paths["pattern_facts"], paths["llm_facts"], paths["merged"]) == (20, 18, 2)
    assert paths["model_calls"] == 3
    grounder = stages["grounder"]
    assert (grounder["calls"], grounder["failed"], grounder["unparseable"]) == (36, 0, 0)
    assert (grounder["supported"], grounder["not_found"], grounder["contradicted"]) == (20, 15, 1)
    assert stages["validator"]["refused"] == {"contradicted": 1}
    assert stages["corroborator"]["conflicts"] == {"lost": 1, "won": 2}
    assert stats["graph"] == {
        "facts": 25,
        "edges": 8,
        "properties": 17,
        "entities": 7,
        "links": {"different": 1, "similar": 3},
    }

    facts = _lines(out / "facts.jsonl")
    assert all(f["evidence"] for f in facts), "every written fact cites a document"
    by_claim = {(f["subject"]["key"], f["predicate"], f["object_value"]): f for f in facts}
    # The model's invented head office for the GmbH was contradicted by its own span.
    assert ("Company:halden robotics gmbh", "headquarters", "Leeds") not in by_claim
    # "10 March 2014" in the note and "2014-03-10" in the register are one claim.
    founded = by_claim[("Company:halden robotics ltd", "founded", "2014-03-10")]
    assert founded["support"] == 2
    assert {e["doc_id"] for e in founded["evidence"]} == {"corpus/register.csv#L2", HALDEN_NOTE}
    # Two head offices for one company: both kept, the note's lost to the register.
    sheffield = by_claim[("Company:halden robotics ltd", "headquarters", "Sheffield")]
    leeds = by_claim[("Company:halden robotics ltd", "headquarters", "Leeds")]
    assert sheffield["qualifiers"]["odke.conflict"]["status"] == "lost"
    assert leeds["confidence"] > sheffield["confidence"]

    (different,) = [link for link in _lines(out / "links.jsonl") if link["kind"] == "different"]
    assert different["reason"] == "external_id mismatch: DE-551902 vs HR-104233"


def test_the_neo4j_config_is_the_jsonl_one_but_for_where_it_writes() -> None:
    jsonl, neo4j = (load_config(EXAMPLE / name) for name in ("odke.yaml", "odke.neo4j.yaml"))
    ignore = {"sink", "constrainer"}
    assert jsonl.model_dump(exclude={"stages": ignore, "bootstrap": True}) == neo4j.model_dump(
        exclude={"stages": ignore, "bootstrap": True}
    )
    assert (neo4j.bootstrap, neo4j.stages.sink[0].use) == (True, "neo4j")


def test_a_dry_run_into_neo4j_needs_no_database_and_no_password(
    example: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(uri: str, auth: Any) -> Any:
        raise AssertionError("a dry run must not connect")

    monkeypatch.setattr(neo4j_module, "_connect", refuse)
    for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    result = runner.invoke(app, ["run", str(example / "odke.neo4j.yaml"), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would apply" in result.output
    assert "CREATE CONSTRAINT odke_key_Company IF NOT EXISTS" in result.output
    assert "neo4j → $NEO4J_URI:" in result.output


def test_the_reference_config_runs_the_same_example(example: Path) -> None:
    """examples/run.yaml is the commented form of odke.yaml, and builds the same graph."""
    reference = execute(load_config(example.parent / "run.yaml"), dry_run=True)
    assert reference.stats["graph"]["facts"] == 25
    assert reference.stats["stages"]["grounder"]["failed"] == 0


def test_the_readme_quotes_every_query_and_query_4_is_the_compiled_check() -> None:
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    found = queries()
    assert len(found) == 5
    for query in found:
        assert query in readme, f"README is missing:\n{query}"
    ontology = Ontology.from_json(EXAMPLE / "ontology.json")
    (check,) = [
        c
        for c in Neo4jConstrainer().checks(ontology)
        if c.startswith(f"{CHECK_MARKER}headquarters")
    ]
    assert found[3] == check.split("\n", 1)[1]


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #

MINE = (
    "MATCH (n) WHERE n:Company OR n:Person "
    "OR (n:Claim AND n.subject_type IN ['Company', 'Person']) "
)


def _schema_names(driver: Any) -> set[tuple[str, str]]:
    constraints = driver.execute_query("SHOW CONSTRAINTS YIELD name").records
    indexes = driver.execute_query("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'").records
    return {("CONSTRAINT", r["name"]) for r in constraints} | {
        ("INDEX", r["name"]) for r in indexes
    }


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_the_example_end_to_end_into_a_live_neo4j(example: Path) -> None:
    """`odke run` into Neo4j, then the README's queries, then a second run that changes nothing."""
    neo4j = pytest.importorskip("neo4j")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with neo4j.GraphDatabase.driver(os.environ["NEO4J_URI"], auth=auth) as driver:
        held = driver.execute_query(MINE + "RETURN count(n) AS n").records[0]["n"]
        if held:
            pytest.skip("the database already holds Company or Person nodes; not writing into it")
        before = _schema_names(driver)
        try:
            _run_and_query(example, driver)
        finally:
            driver.execute_query(MINE + "DETACH DELETE n")
            for kind, name in sorted(_schema_names(driver) - before):
                driver.execute_query(f"DROP {kind} {name} IF EXISTS")


def _run_and_query(example: Path, driver: Any) -> None:
    result = runner.invoke(app, ["run", str(example / "odke.neo4j.yaml")])
    assert result.exit_code == 0, result.output
    assert "applied" in result.output
    names = {name for _, name in _schema_names(driver)}
    assert {"odke_key_Company", "odke_signature_headquarters"} <= names

    def rows(query: str) -> list[dict[str, Any]]:
        return [record.data() for record in driver.execute_query(query).records]

    provenance, why, different, cardinality, recent = (rows(q) for q in queries())

    assert {(r["subject"], r["predicate"], r["object"]) for r in provenance} == {
        ("Halden Robotics Ltd", "legal_name", "Halden Robotics Ltd"),
        ("Halden Robotics Ltd", "founded", "2014-03-10"),
        ("Halden Robotics Ltd", "headquarters", "Sheffield"),
        ("Halden Robotics Ltd", "chief_executive", "Maya Okafor"),
        ("Maya Okafor", "full_name", "Maya Okafor"),
        ("Maya Okafor", "employer", "Corvid Analytics"),
        ("Maya Okafor", "employer", "Halden Robotics GmbH"),
        ("Halden Robotics GmbH", "legal_name", "Halden Robotics GmbH"),
    }
    assert all(len(r["starts"]) == len(r["ends"]) >= 1 for r in provenance)

    assert [r["headquarters"] for r in why] == ["Leeds", "Sheffield"]
    assert json.loads(why[1]["conflict"])["status"] == "lost"
    assert why[0]["tiers"] == ["curated"]

    assert different == [
        {
            "company": "Halden Robotics GmbH",
            "other": "Halden Robotics Ltd",
            "name_similarity": 1.0,
            "reason": "external_id mismatch: DE-551902 vs HR-104233",
        }
    ]

    assert [(r["subject"], sorted(r["objects"])) for r in cardinality] == [
        ("Company:halden robotics ltd", ["Leeds", "Sheffield"])
    ]
    assert sum(r["facts"] for r in recent) == 25

    # Run the pipeline again and write its graph twice: new fact ids and new
    # clocks, the same claims, so no count moves from what the first run wrote.
    count = MINE + (
        "OPTIONAL MATCH (n)-[r]-() "
        "RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS relationships"
    )
    first = driver.execute_query(count).records[0].data()
    assert first == {"nodes": 24, "relationships": 29}
    again = execute(load_config(example / "odke.neo4j.yaml"), dry_run=True).graph
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with neo4j_module.Neo4jSink(os.environ["NEO4J_URI"], auth) as sink:
        report = assert_idempotent(
            sink, again, lambda: driver.execute_query(count).records[0].data()
        )
    assert {k: v["after_last"] for k, v in report.breakdown.items()} == first
