"""neo4j-graphrag's graphs (#111): each relationship grounded against the chunk it came from.

The fixture is a `Neo4jGraph` built by neo4j-graphrag 1.22.0's own
`LLMEntityRelationExtractor` over two chunks, with an LLM stand-in that
answered each chunk with fixed JSON. The chunk-id prefixes, the lexical graph
and `FROM_CHUNK` are the library's. The stand-ins below carry exactly the
attribute names of `neo4j_graphrag.components.types` (`Neo4jNode`,
`Neo4jRelationship`, `Neo4jGraph`, `LexicalGraphConfig`). A store is read
through the recording driver, and through a real server when `NEO4J_URI` is set.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from openodke import Document, GroundingVerdict, Ontology, Pipeline, SpanOrigin
from openodke.ground import LLMGrounder
from openodke.interop import TriplesExtractor, from_graphrag, read_graphrag
from openodke.llm.testing import RecordedClient
from test_neo4j_sink import FakeDriver

FIXTURE = Path(__file__).parent / "fixtures" / "interop" / "graphrag.graph.json"
CHUNKS = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist.",
    "It builds warehouse robots, and it opened a second office in Lyon in 2019.",
)


@dataclass
class Neo4jNode:
    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding_properties: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class Neo4jRelationship:
    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding_properties: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class Neo4jGraph:
    nodes: list[Neo4jNode] = field(default_factory=list)
    relationships: list[Neo4jRelationship] = field(default_factory=list)


@dataclass
class LexicalGraphConfig:
    document_node_label: str = "Document"
    chunk_node_label: str = "Chunk"
    chunk_to_document_relationship_type: str = "FROM_DOCUMENT"
    next_chunk_relationship_type: str = "NEXT_CHUNK"
    node_to_chunk_relationship_type: str = "FROM_CHUNK"
    chunk_id_property: str = "id"
    chunk_index_property: str = "index"
    chunk_text_property: str = "text"
    chunk_embedding_property: str = "embedding"


def _objects() -> Neo4jGraph:
    """The fixture, as the objects the extractor returns."""
    data = json.loads(FIXTURE.read_text())
    return Neo4jGraph(
        nodes=[Neo4jNode(**n) for n in data["nodes"]],
        relationships=[Neo4jRelationship(**r) for r in data["relationships"]],
    )


def _shape(found: tuple[list[Any], list[Document]]) -> tuple[list[dict[str, Any]], list[Any]]:
    rows, docs = found
    return (
        [r.model_dump(exclude_defaults=True) for r in rows],
        [(d.id, d.text, d.uri) for d in docs],
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
                "founded_by": {"domain": ["Company"], "range": "Person"},
                "founded": {"domain": ["Company"], "range": "integer"},
            },
        }
    )


# --------------------------------------------------------------------------- #
# The graph the extractor returns
# --------------------------------------------------------------------------- #


def test_each_relationship_is_grounded_against_the_chunk_both_ends_came_from() -> None:
    rows, docs = _shape(from_graphrag(FIXTURE))
    assert [(r["doc"], r["subject"], r["predicate"], r["object"]) for r in rows[:4]] == [
        ("chunk-0", "Halden Robotics", "FOUNDED_IN", "Leeds"),
        ("chunk-0", "Halden Robotics", "FOUNDED_BY", "Mara Quist"),
        ("chunk-1", "Halden Robotics", "OFFICE_IN", "Lyon"),
        ("chunk-1", "Halden Robotics", "OFFICE_IN", "Berlin"),
    ]
    assert (rows[2]["subject_type"], rows[2]["object_type"]) == ("Company", "City")
    assert rows[2]["qualifiers"] == {"since": 2019}
    # The chunks are the texts, carrying the document's path; the document is not needed.
    assert docs == [
        ("chunk-0", CHUNKS[0], "notes/halden.txt"),
        ("chunk-1", CHUNKS[1], "notes/halden.txt"),
    ]


def test_an_entity_property_is_a_literal_row_on_its_chunk() -> None:
    rows, _ = _shape(from_graphrag(FIXTURE))
    assert rows[4:] == [
        {
            "doc": "chunk-0",
            "subject": "Halden Robotics",
            "subject_type": "Company",
            "predicate": "founded",
            "object": 2014,
        }
    ]


def test_the_objects_and_the_json_read_alike() -> None:
    assert _shape(from_graphrag(_objects())) == _shape(from_graphrag(FIXTURE))
    assert _shape(from_graphrag(json.loads(FIXTURE.read_text()))) == _shape(from_graphrag(FIXTURE))


def test_the_lexical_graph_is_read_by_the_configs_names() -> None:
    renamed = {"Chunk": "Passage", "FROM_CHUNK": "MENTIONED_IN", "text": "body"}
    graph = _objects()
    for node in graph.nodes:
        node.label = renamed.get(node.label, node.label)
        if "text" in node.properties:
            node.properties["body"] = node.properties.pop("text")
    for rel in graph.relationships:
        rel.type = renamed.get(rel.type, rel.type)
    config = LexicalGraphConfig(
        chunk_node_label="Passage",
        node_to_chunk_relationship_type="MENTIONED_IN",
        chunk_text_property="body",
    )
    assert _shape(from_graphrag(graph, config=config)) == _shape(from_graphrag(FIXTURE))
    # A mapping of the config's field names works the same way.
    assert _shape(from_graphrag(graph, config=vars(config))) == _shape(from_graphrag(FIXTURE))


def test_a_graph_built_without_chunks_needs_the_text_it_came_from() -> None:
    graph = _objects()
    graph.nodes = [n for n in graph.nodes if n.label not in {"Chunk", "Document"}]
    graph.relationships = [
        r
        for r in graph.relationships
        if r.type not in {"FROM_CHUNK", "FROM_DOCUMENT", "NEXT_CHUNK"}
    ]
    with pytest.raises(ValueError, match="no Chunk nodes .* pass document="):
        from_graphrag(graph)
    doc = Document(id="halden", text=" ".join(CHUNKS))
    rows, docs = from_graphrag(graph, document=doc)
    assert {r.doc for r in rows} == {"halden"} and docs == [doc]
    assert len(rows) == 5


def test_a_relationship_to_a_node_the_graph_lacks_is_left_out_and_said_so() -> None:
    graph = _objects()
    graph.relationships.append(Neo4jRelationship("chunk-1:0", "chunk-9:9", "OFFICE_IN"))
    with pytest.warns(UserWarning, match="1 relationships end at a node the graph does not have"):
        rows, _ = from_graphrag(graph)
    assert len(rows) == 5


def test_the_name_property_is_the_label_and_the_id_stands_in_without_one() -> None:
    graph = _objects()
    for node in graph.nodes:
        if node.properties.get("name") == "Berlin":
            node.properties = {"title": "Berlin"}
    rows, _ = from_graphrag(graph)
    assert ("chunk-1:2", "title", "Berlin") in {(r.subject, r.predicate, r.object) for r in rows}
    by_title, _ = from_graphrag(graph, name_property="title")
    assert "Berlin" in {r.object for r in by_title}


def test_reading_needs_no_neo4j_graphrag() -> None:
    from_graphrag(FIXTURE)
    assert not any(name.startswith("neo4j_graphrag") for name in sys.modules)


def test_the_graph_grounds_against_its_chunks_end_to_end(companies: Ontology) -> None:
    rows, texts = from_graphrag(FIXTURE)
    client = RecordedClient(
        [
            {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
            {"match": "Halden Robotics", "response": {"verdict": "supported"}},
        ]
    )
    stage = TriplesExtractor(rows, extractor="neo4j-graphrag", documents=texts)
    kg = Pipeline(companies, stage, grounder=LLMGrounder(client=client)).run(texts)
    assert stage.stats["context"] == len(rows) == len(kg.facts) == 5
    for fact in kg.facts:
        (evidence,) = fact.evidence
        assert evidence.span_origin is SpanOrigin.CONTEXT
        assert evidence.uri == "notes/halden.txt"
    berlin = next(f for f in kg.facts if f.object_entity and f.object_entity.label == "Berlin")
    assert berlin.verdict is GroundingVerdict.NOT_FOUND
    assert berlin.evidence[0].doc_id == "chunk-1"
    # The model read the chunk the relationship came from, not the whole document.
    (asked,) = [m for m, _, _ in client.calls if "Berlin" in str(m)]
    assert CHUNKS[1] in asked[-1].content and CHUNKS[0] not in asked[-1].content


# --------------------------------------------------------------------------- #
# A store neo4j-graphrag wrote
# --------------------------------------------------------------------------- #


class Moment:
    """What the driver returns for a Neo4j date: a value with `to_native()`."""

    def __init__(self, value: date) -> None:
        self.value = value

    def to_native(self) -> date:
        return self.value


def _store() -> FakeDriver:
    """After resolution: Halden Robotics and Leeds were each found in both chunks."""
    return FakeDriver(
        {
            "AS text": [
                {"id": "c0", "index": 0, "text": CHUNKS[0], "document": "d", "path": "halden.txt"},
                {"id": "c1", "index": 1, "text": CHUNKS[1], "document": "d", "path": "halden.txt"},
            ],
            "collect(": [
                {
                    "id": "h",
                    "labels": ["__KGBuilder__", "Company", "__Entity__"],
                    "properties": {"name": "Halden Robotics", "__tmp_internal_id": None},
                    "chunks": ["c0", "c1"],
                },
                {
                    "id": "l",
                    "labels": ["City", "__Entity__"],
                    "properties": {"name": "Leeds"},
                    "chunks": ["c0", "c1"],
                },
                {
                    "id": "m",
                    "labels": ["Person", "__Entity__"],
                    "properties": {"name": "Mara Quist", "born": Moment(date(1971, 3, 1))},
                    "chunks": ["c0"],
                },
                {
                    "id": "y",
                    "labels": ["City", "__Entity__"],
                    "properties": {"name": "Lyon"},
                    "chunks": ["c1"],
                },
                {
                    "id": "b",
                    "labels": ["City", "__Entity__"],
                    "properties": {"name": "Berlin"},
                    "chunks": [],
                },
            ],
            "type(r)": [
                {"start": "h", "end": "l", "type": "FOUNDED_IN", "properties": {}},
                {"start": "h", "end": "m", "type": "FOUNDED_BY", "properties": {}},
                {"start": "h", "end": "y", "type": "OFFICE_IN", "properties": {"since": 2019}},
                {"start": "h", "end": "b", "type": "OFFICE_IN", "properties": {}},
            ],
        }
    )


def test_a_store_is_read_in_one_read_transaction_by_the_configs_names() -> None:
    driver = _store()
    config = LexicalGraphConfig(chunk_node_label="Passage", chunk_text_property="body")
    with pytest.warns(UserWarning):
        read_graphrag(driver, database="graphs", config=config)
    assert driver.sessions == [{"database": "graphs"}]
    assert {mode for mode, _, _ in driver.calls} == {"read"}
    cyphers = [cypher for _, cypher, _ in driver.calls]
    assert all("`Passage`" in c for c in cyphers[:2]) and "`__Entity__`" in cyphers[2]
    assert driver.calls[0][2] == {"index": "index", "text": "body"}


def test_a_store_grounds_on_a_shared_chunk_and_falls_back_to_the_document() -> None:
    with pytest.warns(UserWarning, match="1 relationships or entities have no chunk"):
        rows, docs = read_graphrag(_store())
    placed = {(r.predicate, str(r.object)): r.doc for r in rows}
    # Both ends in both chunks: no one chunk is the source, so the whole document.
    assert placed[("FOUNDED_IN", "Leeds")] == "d"
    # One chunk shared: that chunk.
    assert placed[("FOUNDED_BY", "Mara Quist")] == "c0"
    assert placed[("OFFICE_IN", "Lyon")] == "c1"
    # Berlin was linked to no chunk: left out, and the warning says so.
    assert ("OFFICE_IN", "Berlin") not in placed
    # A Neo4j date comes back as a date, and a writer's internal property is not a fact.
    assert placed[("born", "1971-03-01")] == "c0"
    assert "__tmp_internal_id" not in {r.predicate for r in rows}
    texts = {d.id: d for d in docs}
    assert texts["d"].text == "\n".join(CHUNKS) and texts["d"].uri == "halden.txt"
    assert texts["c0"].text == CHUNKS[0]
    (company,) = {r.subject_type for r in rows if r.subject == "Halden Robotics"}
    assert company == "Company"


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_a_graphrag_store_reads_back_from_a_live_neo4j() -> None:
    """The lexical graph as neo4j-graphrag's writer leaves it, under labels of this test's own."""
    neo4j = pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    chunk, document, entity = f"Chunk_{suffix}", f"Document_{suffix}", f"Entity_{suffix}"
    company, city = f"Company_{suffix}", f"City_{suffix}"
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    write = (
        f"CREATE (d:`{document}`:__KGBuilder__ {{path: 'notes/halden.txt'}})\n"
        f"CREATE (c0:`{chunk}`:__KGBuilder__ {{text: $t0, index: 0}})-[:FROM_DOCUMENT]->(d)\n"
        f"CREATE (c1:`{chunk}`:__KGBuilder__ {{text: $t1, index: 1}})-[:FROM_DOCUMENT]->(d)\n"
        "CREATE (c0)-[:NEXT_CHUNK]->(c1)\n"
        f"CREATE (h:`{company}`:`{entity}`:__KGBuilder__ "
        "{name: 'Halden Robotics', founded: date('2014-03-01')})\n"
        f"CREATE (l:`{city}`:`{entity}`:__KGBuilder__ {{name: 'Leeds'}})\n"
        f"CREATE (y:`{city}`:`{entity}`:__KGBuilder__ {{name: 'Lyon'}})\n"
        "CREATE (h)-[:FROM_CHUNK]->(c0), (h)-[:FROM_CHUNK]->(c1), (l)-[:FROM_CHUNK]->(c0),\n"
        "       (l)-[:FROM_CHUNK]->(c1), (y)-[:FROM_CHUNK]->(c1)\n"
        "CREATE (h)-[:FOUNDED_IN]->(l), (h)-[:OFFICE_IN {since: 2019}]->(y)"
    )
    with neo4j.GraphDatabase.driver(os.environ["NEO4J_URI"], auth=auth) as driver:
        try:
            driver.execute_query(write, t0=CHUNKS[0], t1=CHUNKS[1])
            config = {"chunk_node_label": chunk, "document_node_label": document}
            rows, docs = read_graphrag(driver, config=config, entity_label=entity)
        finally:
            driver.execute_query(
                f"MATCH (n) WHERE n:`{chunk}` OR n:`{document}` OR n:`{entity}` DETACH DELETE n"
            )
    found = {(r.subject_type, r.predicate, str(r.object), r.doc) for r in rows}
    texts = {d.id: d for d in docs}
    by_text = {d.text: d.id for d in docs}
    assert found == {
        (company, "FOUNDED_IN", "Leeds", by_text["\n".join(CHUNKS)]),
        (company, "OFFICE_IN", "Lyon", by_text[CHUNKS[1]]),
        (company, "founded", "2014-03-01", by_text["\n".join(CHUNKS)]),
    }
    (lyon,) = [r for r in rows if r.object == "Lyon"]
    assert lyon.qualifiers == {"since": 2019}
    assert {d.uri for d in texts.values()} == {"notes/halden.txt"}
