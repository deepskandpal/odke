"""`openodke.Validator` (#129): the whole layer over another extractor's facts.

Every model call is answered from recorded responses. The Neo4j sink writes
through the recording driver from `test_neo4j_sink`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openodke import Document, GroundingVerdict, Ontology
from openodke.corroborate import CHECK
from openodke.gate import VerdictGate
from openodke.ground import LLMGrounder
from openodke.interop import to_fact
from openodke.interop.triples import TripleRow
from openodke.llm.testing import RecordedClient, ScriptedClient
from openodke.loaders import DirectoryLoader
from openodke.sinks.jsonl import JsonlSink
from openodke.sinks.neo4j import Neo4jSink
from openodke.stages import PassThroughGate, PassThroughResolver
from openodke.validator import ValidationReport, Validator
from test_neo4j_sink import FakeDriver

EXAMPLE = Path(__file__).parent.parent / "examples" / "triples"
PLACES = Ontology.from_dict(
    {
        "name": "places",
        "types": {"Region": {}, "Country": {}, "Company": {}},
        "predicates": {
            "located_in": {"domain": ["Region"], "range": "Country", "inverse_of": "contains"},
            "contains": {"domain": ["Country"], "range": "Region"},
            "founded": {"domain": ["Company"], "range": "integer"},
        },
    }
)
A = Document(
    id="a", text="Brittany is a region located in France. Halden Robotics was founded in 2014."
)
B = Document(
    id="b", text="Brittany lies in France, as does Normandy. Halden Robotic opened in Lyon."
)
FOUNDED = "Halden Robotics was founded in 2014."
ROWS: list[dict[str, Any]] = [
    # Stated twice, in two texts: one fact, supported by both.
    {"doc": "a", "subject": "Brittany", "predicate": "located_in", "object": "France",
     "quote": "Brittany is a region located in France."},
    {"doc": "b", "subject": "Brittany", "predicate": "located_in", "object": "France",
     "quote": "Brittany lies in France"},
    # The text says otherwise.
    {"doc": "a", "subject": "Halden Robotics", "predicate": "founded", "object": 2012,
     "quote": FOUNDED},
    {"doc": "a", "subject": "Halden Robotics", "predicate": "founded", "object": 2014,
     "quote": FOUNDED},
    # Not in the ontology, under a misspelt name the resolver links.
    {"doc": "b", "subject": "Halden Robotic", "subject_type": "Company", "predicate": "opened_in",
     "object": "Lyon"},
]  # fmt: skip
RECORDED = [
    {"match": "— 2012.", "response": {"verdict": "contradicted"}},
    {"match": "Claim:", "response": {"verdict": "supported"}},
]


def _texts() -> list[Document]:
    return list(DirectoryLoader().load(EXAMPLE / "texts"))


def test_the_triples_example_runs_end_to_end_into_jsonl(tmp_path: Path) -> None:
    client = RecordedClient.from_fixture(EXAMPLE / "recorded" / "ground.json")
    ontology = Ontology.from_json(EXAMPLE / "ontology.json")
    validator = Validator(ontology, client=client, sinks=[JsonlSink(tmp_path / "out")])
    kg, report = validator.validate(EXAMPLE / "triples.jsonl", _texts(), extractor="hand-written")

    assert (report.facts_in, report.documents, report.facts_out) == (5, 1, 5)
    assert report.verdicts == {"supported": 3, "contradicted": 0, "not_found": 2, "unchecked": 0}
    assert report.checked["not_in_text"] == 1
    assert (report.refused, report.merged, report.linked, report.derived) == (0, 0, 0, 0)
    assert (report.calls, report.prompts) == (4, ("ground.span@1",))
    assert report.coverage is not None and report.coverage["uncovered"] == 0
    assert {f.extractor for f in kg.facts} == {"hand-written"}

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["facts"] == 5
    assert manifest["stats"]["validation"]["facts_in"] == 5
    assert {"extractor", "grounder", "validator"} <= set(manifest["stats"]["stages"])
    lines = (tmp_path / "out" / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5


def test_the_report_counts_what_each_stage_did() -> None:
    client = RecordedClient(RECORDED)
    kg, report = Validator(PLACES, client=client).validate(ROWS, [A, B])

    assert report.facts_in == 5
    assert report.verdicts == {"supported": 3, "contradicted": 1, "not_found": 0, "unchecked": 1}
    # The off-schema row cost no call: four asked, of five.
    assert report.calls == len(client.calls) == 4
    assert report.checked["predicate"] == 1
    assert (report.refused, report.refused_by) == (2, {"contradicted": 1, "predicate": 1})
    # France contains Brittany, once from each text; each pair then merges.
    assert report.derived == 2 and report.merged == 2
    assert (report.linked, report.links) == (1, {"similar": 1})
    assert report.facts_out == len(kg.facts) == 3
    assert report.facts_in + report.derived - report.refused - report.merged == report.facts_out
    (brittany,) = [f for f in kg.facts if f.predicate == "located_in"]
    assert brittany.support == 2
    text = report.render()
    for line in ("refused       2 by the gate", "merged        2", "derived       2"):
        assert line in text


def test_a_dry_run_calls_no_model_and_writes_nothing() -> None:
    client = ScriptedClient([])  # strict: any call would fail the test
    driver = FakeDriver()
    sink = Neo4jSink(driver=driver, ontology=PLACES)
    kg, report = Validator(PLACES, client=client, sinks=[sink]).validate(ROWS, [A, B], dry_run=True)
    assert client.calls == [] and driver.calls == []
    assert report.dry_run and (report.calls, report.cost_usd) == (0, None)
    assert report.verdicts["unchecked"] == 5
    # Everything deterministic still ran: the schema check, resolution, partners, merging.
    assert (report.refused_by, report.derived, report.linked) == ({"predicate": 1}, 2, 1)
    assert "dry run, no model called" in report.render()
    assert kg.stats["validation"]["dry_run"] is True


def test_a_neo4j_sink_writes_through_its_driver() -> None:
    driver = FakeDriver()
    sink = Neo4jSink(driver=driver, ontology=PLACES)
    validator = Validator(PLACES, client=RecordedClient(RECORDED), sinks=[sink])
    # A store never bootstrapped has no signature index to merge through, and says so.
    with pytest.warns(UserWarning, match=r"no index for \w+\.signature"):
        kg, _ = validator.validate(ROWS, [A, B])
    assert driver.writes, "nothing was written"
    rows = [row for _, params in driver.writes for row in params["rows"]]
    assert {row.get("key") for row in rows} >= {"Region:brittany", "Country:france"}
    # The refused founding year never reaches the store; the supported one does.
    assert {row["value"] for row in rows if "value" in row} == {2014}


def test_facts_are_taken_as_they_are() -> None:
    texts = {"a": A, "b": B}
    facts = [to_fact(TripleRow.model_validate(row), texts[row["doc"]], PLACES) for row in ROWS]
    kg, report = Validator(PLACES, client=RecordedClient(RECORDED)).validate(facts, [A, B])
    assert report.facts_in == 5 and report.facts_out == 3
    lost = to_fact(
        TripleRow.model_validate({**ROWS[0], "doc": "gone"}), Document(id="gone", text="x")
    )
    _, report = Validator(PLACES, client=RecordedClient(RECORDED)).validate([lost], [A])
    assert (report.facts_in, report.unmatched) == (0, 1)
    with pytest.raises(ValueError, match="rows or Facts, not both"):
        Validator(PLACES).validate([facts[0], ROWS[0]], [A], dry_run=True)


def test_a_stage_given_replaces_its_default_and_a_grounder_still_follows_the_checks() -> None:
    client = RecordedClient(RECORDED)
    validator = Validator(
        PLACES,
        grounder=LLMGrounder(client=client),
        resolver=PassThroughResolver(),
        gate=PassThroughGate(),
    )
    kg, report = validator.validate(ROWS, [A, B])
    assert report.linked == 0 and report.refused == 0
    # The free checks ran ahead of the grounder given, and the pass-through gate kept what
    # they refused, stamped with why.
    (opened,) = [f for f in kg.facts if f.predicate == "opened_in"]
    assert opened.verdict is GroundingVerdict.UNCHECKED and opened.qualifiers[CHECK]["check"]
    assert len(client.calls) == 4


def test_a_validator_kept_between_calls_reports_each_call_alone() -> None:
    gate = VerdictGate(schema=True)
    validator = Validator(PLACES, client=RecordedClient(RECORDED), gate=gate)
    first = validator.validate(ROWS, [A, B]).report
    second = validator.validate(ROWS, [A, B]).report
    assert first == second
    assert (second.calls, second.refused_by) == (4, {"contradicted": 1, "predicate": 1})
    assert gate.stats["refused"] == {"contradicted": 2, "predicate": 2}


def test_with_no_ontology_the_layer_runs_on_what_the_rows_say() -> None:
    client = RecordedClient(RECORDED)
    _, report = Validator(client=client).validate(ROWS, [A, B])
    assert report.checked == {"not_in_text": 0, "predicate": 0, "domain": 0, "range": 0}
    assert report.derived == 0 and report.calls == 5


def test_the_locator_is_the_grounders_when_one_is_given() -> None:
    with pytest.raises(ValueError, match="LLMGrounder\\(locate=True\\)"):
        Validator(grounder=LLMGrounder(client=RecordedClient(RECORDED)), locate=True)
    uncited = {"doc": "a", "subject": "Brittany", "predicate": "located_in", "object": "France"}
    kg, report = Validator(locate=True).validate([uncited], [A], dry_run=True)
    assert kg.facts[0].evidence[0].span_origin == "located"
    assert isinstance(report, ValidationReport) and report.facts_out == 1
