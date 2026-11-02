"""LangChain `GraphDocument`s (#97): read without langchain, grounded against their document.

The stand-ins below carry exactly the attribute names of
`langchain_community.graphs.graph_document` (`Node`, `Relationship`,
`GraphDocument`) and `langchain_core.documents.Document`. The fixture is
`GraphDocument.model_dump_json()` from langchain-community 0.4.2, one per line.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from openodke import Document, GroundingVerdict, Ontology, Pipeline, SpanOrigin
from openodke.eval.spans import evaluate_spans
from openodke.ground import LLMGrounder
from openodke.interop import TriplesExtractor, from_graph_documents
from openodke.llm.testing import RecordedClient

FIXTURE = Path(__file__).parent / "fixtures" / "interop" / "langchain.graph_documents.jsonl"
TEXT = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. The company builds "
    "warehouse robots, and it opened a second office in Lyon in 2019."
)


@dataclass
class LCDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class Node:
    id: str | int
    type: str = "Node"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    source: Node
    target: Node
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDocument:
    nodes: list[Node]
    relationships: list[Relationship]
    source: LCDocument


def _halden() -> GraphDocument:
    company = Node("Halden Robotics", "Company", {"foundingYear": "2014"})
    lyon, berlin = Node("Lyon", "City"), Node("Berlin", "City")
    return GraphDocument(
        nodes=[company, lyon, berlin],
        relationships=[
            Relationship(
                Node("Halden Robotics", "Company"), lyon, "OFFICE_IN", {"startDate": "2019"}
            ),
            Relationship(Node("Halden Robotics", "Company"), berlin, "OFFICE_IN"),
        ],
        source=LCDocument(TEXT, {"source": "notes/halden.txt"}),
    )


@pytest.fixture
def companies() -> Ontology:
    return Ontology.from_dict(
        {
            "name": "companies",
            "types": {"Company": {}, "City": {}, "Person": {}},
            "predicates": {
                "office_in": {"domain": ["Company"], "range": "City"},
                "founded_in": {"domain": ["Company"], "range": "City"},
                "founded": {"domain": ["Person"], "range": "Company"},
                "studied_in": {"domain": ["Person"], "range": "City"},
                "foundingYear": {"domain": ["Company"], "range": "integer"},
            },
        }
    )


def test_a_relationship_is_a_row_and_its_properties_are_qualifiers() -> None:
    rows, (doc,) = from_graph_documents([_halden()])
    lyon = rows[0]
    assert (lyon.doc, lyon.subject, lyon.subject_type) == (doc.id, "Halden Robotics", "Company")
    assert (lyon.predicate, lyon.object, lyon.object_type) == ("OFFICE_IN", "Lyon", "City")
    assert lyon.qualifiers == {"startDate": "2019"}
    # LLMGraphTransformer cites nothing, and the row says nothing it does not have.
    assert (lyon.start, lyon.end, lyon.quote) == (None, None, None)


def test_a_node_property_is_a_literal_row_about_the_node() -> None:
    rows, _ = from_graph_documents([_halden()])
    (founded,) = [r for r in rows if r.predicate == "foundingYear"]
    assert (founded.subject, founded.object, founded.object_type) == (
        "Halden Robotics",
        "2014",
        "string",
    )


def test_the_text_is_the_source_documents_page_content() -> None:
    _, (doc,) = from_graph_documents([_halden()])
    assert doc.text == TEXT
    assert doc.uri == "notes/halden.txt"
    # No id on the LangChain document: a hash of the text, so the same text is one document.
    again = _halden()
    _, docs = from_graph_documents([_halden(), again])
    assert [d.id for d in docs] == [doc.id]
    named = _halden()
    named.source.id = "halden"
    assert from_graph_documents([named])[1][0].id == "halden"


def test_one_id_with_two_texts_is_refused() -> None:
    first, second = _halden(), _halden()
    first.source.id = second.source.id = "same"
    second.source.page_content = "Another text."
    with pytest.raises(ValueError, match="'same'"):
        from_graph_documents([first, second])


def test_the_untyped_node_type_leaves_the_type_to_the_ontology(companies: Ontology) -> None:
    rows, docs = from_graph_documents(FIXTURE)
    (studied,) = [r for r in rows if r.predicate == "STUDIED_IN"]
    assert studied.object_type is None
    stage = TriplesExtractor(rows, documents=docs)
    kg = Pipeline(companies, stage).run(docs)
    (fact,) = [f for f in kg.facts if f.predicate == "studied_in"]
    assert fact.object_entity is not None and fact.object_entity.type == "City"


def _read(source: Any) -> tuple[list[Any], list[tuple[str, str, str | None]]]:
    rows, docs = from_graph_documents(source)
    return rows, [(d.id, d.text, d.uri) for d in docs]


def test_the_json_form_reads_as_the_objects_do(tmp_path: Path) -> None:
    lines = FIXTURE.read_text().splitlines()
    array = tmp_path / "graph_documents.json"
    array.write_text(json.dumps([json.loads(line) for line in lines], indent=2))
    assert _read(FIXTURE) == _read([json.loads(line) for line in lines]) == _read(array)
    halden = json.loads(lines[0])
    halden["nodes"] = [
        n for n in halden["nodes"] if n["id"] in {"Halden Robotics", "Lyon", "Berlin"}
    ]
    halden["relationships"] = [r for r in halden["relationships"] if r["type"] == "OFFICE_IN"]
    assert _read([halden]) == _read([_halden()])


def test_reading_needs_no_langchain() -> None:
    from_graph_documents(FIXTURE)
    assert not any(name.split(".")[0].startswith("langchain") for name in sys.modules)


def test_every_fact_is_grounded_against_its_whole_document(companies: Ontology) -> None:
    rows, docs = from_graph_documents(FIXTURE)
    client = RecordedClient(
        [
            {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
            {"match": "Mara Quist (Person) — studied in", "response": {"verdict": "supported"}},
            {"match": "Halden Robotics", "response": {"verdict": "supported"}},
        ]
    )
    stage = TriplesExtractor(rows, extractor="langchain", documents=docs)
    kg = Pipeline(companies, stage, grounder=LLMGrounder(client=client)).run(docs)

    assert stage.stats["context"] == stage.stats["rows"] == len(rows) == 6
    texts = {d.id: d.text for d in docs}
    for fact in kg.facts:
        (evidence,) = fact.evidence
        assert evidence.span_origin is SpanOrigin.CONTEXT
        assert evidence.span is not None
        assert (evidence.span.start, evidence.span.end) == (0, len(texts[evidence.doc_id]))
    verdicts = {
        (f.predicate, f.object_entity.label if f.object_entity else f.object_value): f.verdict
        for f in kg.facts
    }
    assert verdicts[("office_in", "Berlin")] is GroundingVerdict.NOT_FOUND
    assert verdicts[("office_in", "Lyon")] is GroundingVerdict.SUPPORTED
    assert verdicts[("foundingYear", "2014")] is GroundingVerdict.SUPPORTED
    # Each was asked about against the whole text it came from.
    (berlin,) = [m for m, _, _ in client.calls if "Berlin" in str(m)]
    assert TEXT in berlin[-1].content
    # A whole document is not a citation, and the span report says so.
    report = evaluate_spans(kg)
    assert (report.metrics["with_span"], report.metrics["no_span"]) == (0, len(kg.facts))


def test_a_document_with_no_source_is_refused() -> None:
    with pytest.raises(ValueError, match="no source document"):
        from_graph_documents([{"nodes": [], "relationships": []}])


def test_rows_ground_against_a_loaded_document_too(companies: Ontology) -> None:
    # The texts the adapter returns are ordinary documents: any pipeline input will do.
    rows, docs = from_graph_documents([_halden()])
    doc = Document(id=docs[0].id, text=TEXT, uri="file:///notes/halden.txt")
    kg = Pipeline(companies, TriplesExtractor(rows, documents=[doc])).run([doc])
    assert {e.uri for f in kg.facts for e in f.evidence} == {"file:///notes/halden.txt"}
