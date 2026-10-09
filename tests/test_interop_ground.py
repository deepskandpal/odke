"""`ground_graph` (#99): any adapter's facts grounded in one call, and what it adds up to."""

from __future__ import annotations

import json
from pathlib import Path

from openodke import Document, GroundingVerdict, Ontology
from openodke.corroborate import CHECK
from openodke.eval.spans import load_facts
from openodke.ground import LLMGrounder
from openodke.interop import from_graph_documents, ground_graph
from openodke.llm.testing import RecordedClient

FIXTURE = Path(__file__).parent / "fixtures" / "interop" / "langchain.graph_documents.jsonl"
TEXT = "Halden Robotics was founded in Leeds in 2014. It opened a second office in Lyon in 2019."
DOC = Document(id="halden", text=TEXT)
CLAIM = {"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in"}
COMPANIES = Ontology.from_dict(
    {
        "name": "companies",
        "types": {"Company": {}, "City": {}, "Person": {}},
        "predicates": {
            "office_in": {"domain": ["Company"], "range": "City"},
            "founded_in": {"domain": ["Company"], "range": "City"},
            "founded": {"domain": ["Person"], "range": "Company"},
            "foundingYear": {"domain": ["Company"], "range": "integer"},
        },
    }
)


def _quoting(quote: str, city: str = "Lyon") -> dict[str, str]:
    return {**CLAIM, "object": city, "quote": quote}


def test_a_fact_the_ontology_has_no_room_for_costs_no_call() -> None:
    rows, docs = from_graph_documents(FIXTURE)
    client = RecordedClient([{"match": "Claim:", "response": {"verdict": "supported"}}])
    grounded = ground_graph(rows, docs, ontology=COMPANIES, grounder=LLMGrounder(client=client))
    (studied,) = [f for f in grounded.facts if f.predicate == "STUDIED_IN"]
    assert studied.verdict is GroundingVerdict.UNCHECKED
    assert studied.qualifiers[CHECK] == {
        "check": "predicate",
        "reason": "'STUDIED_IN' is not a predicate of ontology 'companies'",
    }
    assert grounded.summary.refused == {"not_in_text": 0, "predicate": 1, "domain": 0, "range": 0}
    assert len(client.calls) == grounded.summary.calls == len(grounded.facts) - 1


def test_a_dry_run_is_the_free_checks_and_the_locator_and_nothing_else() -> None:
    rows = [{**CLAIM, "object": "Lyon"}, _quoting("an office in Oslo", "Oslo")]
    grounded = ground_graph(rows, [DOC], locate=True)
    lyon, oslo = grounded.facts
    assert lyon.verdict is GroundingVerdict.UNCHECKED and lyon.evidence[0].span_origin == "located"
    assert oslo.verdict is GroundingVerdict.NOT_FOUND
    summary = grounded.summary
    assert summary.dry_run and (summary.calls, summary.prompts) == (0, ())
    assert (summary.located, summary.refused["not_in_text"]) == (1, 1)
    assert "unknown without the model" in summary.render()


def test_the_two_failure_shapes_are_told_apart() -> None:
    rows = [
        _quoting("Lyon"),  # names the object alone: too narrow to state the claim
        _quoting("It opened a second office in Lyon in 2019."),  # never names the subject
        _quoting("Halden Robotics was founded in Leeds in 2014. It opened a second office in Lyon"),
        {**CLAIM, "object": "Berlin"},  # cites nothing, and the text never says it
        _quoting("Halden Robotics was founded in Leeds", "Leeds"),  # contradicted, read whole
    ]
    client = RecordedClient(
        [
            {"match": "— Leeds (City)", "response": {"verdict": "contradicted"}},
            {"match": "Claim:", "response": {"verdict": "not_found"}},
        ]
    )
    summary = ground_graph(rows, [DOC], grounder=LLMGrounder(client=client)).summary
    assert summary.too_narrow.count == 2
    assert summary.unsupported.count == 3
    assert all("Lyon" in line for line in summary.too_narrow.examples)
    assert {line.split("→ ")[1].split()[0] for line in summary.unsupported.examples} == {
        "Lyon",
        "Berlin",
        "Leeds",
    }

    # Read whole, a citation is never what the model was shown.
    paper = LLMGrounder(client=client, context="document")
    whole = ground_graph(rows, [DOC], grounder=paper).summary
    assert (whole.too_narrow.count, whole.unsupported.count) == (0, 5)


def test_one_fact_comes_back_per_row_whose_text_was_given() -> None:
    rows = [{**CLAIM, "object": "Lyon"}, {**CLAIM, "doc": "gone", "object": "Oslo"}]
    grounded = ground_graph(rows, [DOC])
    assert len(grounded.facts) == 1
    assert (grounded.summary.rows, grounded.summary.unmatched_rows) == (2, 1)
    assert "1 name a text that was not given" in grounded.summary.render()


def test_write_puts_the_facts_where_odke_eval_spans_reads_them(tmp_path: Path) -> None:
    grounded = ground_graph([_quoting("Lyon")], [DOC])
    written = grounded.write(tmp_path / "out")
    assert [p.name for p in written] == ["facts.jsonl", "summary.json"]
    assert [f.id for f in load_facts(tmp_path / "out")] == [grounded.facts[0].id]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["dry_run"] is True and summary["spans"]["n"] == 1
