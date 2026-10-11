"""A Neo4j graph read back and grounded (#98), and its verdicts written back on request.

The recording driver is answered with what the sink really writes:
`provenance_of` builds each relationship's properties, so a change to the
sink's receipts shows up here. `test_the_sinks_graph_reads_back_from_a_live_neo4j`
writes with `Neo4jSink`, reads back and grounds against a real server when
`NEO4J_URI` is set.
"""

from __future__ import annotations

import os
import uuid
import warnings
from datetime import UTC, date, datetime
from typing import Any

import pytest

from openodke import (
    Document,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    Ontology,
    Pipeline,
    SpanOrigin,
)
from openodke.corroborate import CONFLICT
from openodke.corroborate.provenance import SCORE
from openodke.ground import LLMGrounder
from openodke.interop import (
    TripleRow,
    TriplesExtractor,
    read_neo4j,
    to_fact,
    write_verdicts,
)
from openodke.llm.testing import RecordedClient
from openodke.sinks.neo4j import Neo4jSink, provenance_of
from test_neo4j_sink import FakeDriver

TEXT = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. The company builds "
    "warehouse robots, and it opened a second office in Lyon in 2019."
)
WHEN = datetime(2026, 10, 1, tzinfo=UTC)


def _halden(**overrides: Any) -> Document:
    return Document(
        **{"id": "halden", "text": TEXT, "uri": "file:///notes/halden.txt", **overrides}
    )


def _rows(company: str = "Company", city: str = "City") -> list[TripleRow]:
    lyon = TEXT.index("a second office in Lyon")
    return [
        TripleRow(
            doc="halden",
            subject="Halden Robotics",
            subject_type=company,
            predicate="office_in",
            object="Lyon",
            object_type=city,
            start=lyon,
            end=lyon + len("a second office in Lyon"),
            confidence=0.8,
            extractor="llm",
        ),
        TripleRow(
            doc="halden",
            subject="Halden Robotics",
            subject_type=company,
            predicate="office_in",
            object="Berlin",
            object_type=city,
        ),
        TripleRow(
            doc="halden",
            subject="Halden Robotics",
            subject_type=company,
            predicate="founded",
            object=2014,
            qualifiers={"place": "Leeds", "signature": "a qualifier the sink must prefix"},
        ),
        TripleRow(
            doc="halden",
            subject="Halden Robotics",
            subject_type=company,
            predicate="sells",
            object="customer data",
            object_type="string",
            polarity="denied",
        ),
    ]


def _facts(doc: Document, company: str = "Company", city: str = "City") -> list[Fact]:
    return [to_fact(row, doc) for row in _rows(company, city)]


def _record(fact: Fact, rel: str) -> dict[str, Any]:
    """What the read query returns for a relationship the sink wrote for `fact`."""
    subject = fact.subject
    record = {
        "id": rel,
        "type": fact.predicate,
        "properties": provenance_of(fact, WHEN),
        "subject_labels": [subject.type, "Entity"],
        "subject_label": subject.label,
        "subject_name": None,
        "subject_key": subject.key,
        "subject_id": f"n-{subject.key}",
    }
    obj = fact.object_entity
    if obj is not None:
        record |= {
            "object_labels": [obj.type, "Entity"],
            "object_label": obj.label,
            "object_name": None,
            "object_key": obj.key,
            "object_id": f"n-{obj.key}",
            "value": None,
        }
    else:
        record |= {
            "object_labels": ["Claim"],
            "object_label": None,
            "object_name": None,
            "object_key": None,
            "object_id": f"c-{rel}",
            "value": fact.object_value,
        }
    return record


def _driver(records: list[dict[str, Any]], unread: int = 0) -> FakeDriver:
    return FakeDriver(
        {"ORDER BY id": records, "AS unread": [{"unread": unread}], "AS written": [{"written": 2}]}
    )


def _same(fact: Fact) -> tuple[Any, ...]:
    """What a fact read back must keep of the one written."""
    return (
        fact.signature,
        fact.polarity,
        sorted(fact.qualifiers.items()),
        fact.confidence,
        fact.extractor,
        tuple(
            (e.doc_id, e.uri, e.span.start if e.span else None, e.span.end if e.span else None)
            + (e.span_origin,)
            for e in fact.evidence
        ),
    )


# --------------------------------------------------------------------------- #
# A graph openodke wrote
# --------------------------------------------------------------------------- #


def test_the_sinks_receipts_read_back_as_the_same_facts() -> None:
    doc = _halden()
    facts = _facts(doc)
    driver = _driver([_record(f, f"r{i}") for i, f in enumerate(facts)])
    rows, texts = read_neo4j(driver, documents=[doc])
    assert texts == [doc]
    again = [to_fact(row, doc) for row in rows]
    assert [_same(f) for f in again] == [_same(f) for f in facts]
    # The id is the relationship's, to write a verdict back to.
    assert [f.id for f in again] == ["r0", "r1", "r2", "r3"]


def test_a_cited_span_keeps_its_offsets_and_a_context_span_is_the_whole_text_again() -> None:
    doc = _halden()
    cited, context, *_ = _facts(doc)
    rows, _ = read_neo4j(_driver([_record(cited, "r0"), _record(context, "r1")]), documents=[doc])
    assert (rows[0].start, rows[0].end) == (
        cited.evidence[0].span.start,
        cited.evidence[0].span.end,
    )
    assert (rows[1].start, rows[1].end) == (None, None)
    (evidence,) = to_fact(rows[1], doc).evidence
    assert evidence.span_origin is SpanOrigin.CONTEXT


def test_each_piece_of_evidence_is_a_row_and_a_located_span_is_not_a_citation() -> None:
    doc, other = _halden(), _halden(id="other", uri="file:///notes/other.txt")
    cited = _facts(doc)[0]
    second = cited.evidence[0].model_copy(update={"doc_id": "other", "uri": other.uri})
    located = cited.evidence[0].model_copy(
        update={"doc_id": "third", "uri": None, "span_origin": SpanOrigin.LOCATED}
    )
    fact = cited.model_copy(update={"evidence": (*cited.evidence, second, located)})
    rows, texts = read_neo4j(_driver([_record(fact, "r0")]), documents=[doc, other])
    assert [(r.doc, r.start is not None, r.id) for r in rows] == [
        ("halden", True, "r0"),
        ("other", True, "r0"),
        # Not among the documents: it keeps its id, for TriplesExtractor to count unmatched.
        ("third", False, "r0"),
    ]
    assert [d.id for d in texts] == ["halden", "other"]


def test_a_graph_written_before_span_origins_reads_its_offsets_as_citations() -> None:
    doc = _halden()
    record = _record(_facts(doc)[0], "r0")
    del record["properties"]["evidence_span_origins"]
    (row,), _ = read_neo4j(_driver([record]), documents=[doc])
    assert row.start is not None


def test_a_document_is_found_by_id_and_then_by_uri() -> None:
    written = _facts(_halden(id="run-1-uuid"))
    reloaded = _halden(id="run-2-uuid")
    rows, texts = read_neo4j(_driver([_record(written[0], "r0")]), documents=[reloaded])
    assert rows[0].doc == "run-2-uuid" and texts == [reloaded]


def test_qualifiers_come_back_unprefixed_and_the_stamps_as_mappings() -> None:
    doc = _halden()
    founded = _facts(doc)[2]
    stamped = founded.model_copy(
        update={
            "qualifiers": {
                **founded.qualifiers,
                CONFLICT: {"status": "lost", "to": "2015"},
                SCORE: {"extractor": 0.5, "evidence": 0.9},
            }
        }
    )
    record = _record(stamped, "r0")
    record["properties"]["odke_verdict"] = "supported"  # an earlier audit's write-back
    (row,), _ = read_neo4j(_driver([record]), documents=[doc])
    assert row.qualifiers == {
        "place": "Leeds",
        "signature": "a qualifier the sink must prefix",
        CONFLICT: {"status": "lost", "to": "2015"},
        SCORE: {"extractor": 0.5, "evidence": 0.9},
    }
    assert row.object == 2014 and row.object_type is None


def test_reading_runs_two_read_queries_and_writes_nothing() -> None:
    driver = _driver([_record(_facts(_halden())[0], "r0")])
    read_neo4j(driver, database="graphs", predicates=["office_in"])
    assert driver.sessions == [{"database": "graphs"}]
    assert [mode for mode, _, _ in driver.calls] == ["read", "read"]
    read, unread = (cypher for _, cypher, _ in driver.calls)
    assert "type(r) IN $predicates" in read and "type(r) IN $predicates" in unread
    assert "$text" not in read
    assert driver.calls[0][2]["predicates"] == ["office_in"]
    assert driver.calls[1][2]["links"] == ["DIFFERENT", "SAME_AS", "SIMILAR"]


# --------------------------------------------------------------------------- #
# A graph openodke did not write
# --------------------------------------------------------------------------- #


def _foreign(rel: str, sentence: str, city: str, **properties: Any) -> dict[str, Any]:
    return {
        "id": rel,
        "type": "OFFICE_IN",
        "properties": {"sentence": sentence, **properties},
        "subject_labels": ["Company"],
        "subject_label": None,
        "subject_name": "Halden Robotics",
        "subject_key": None,
        "subject_id": "n1",
        "object_labels": ["City", "Place"],
        "object_label": None,
        "object_name": city,
        "object_key": None,
        "object_id": f"n-{city}",
        "value": None,
    }


def test_a_named_text_property_is_the_evidence_grounded_whole() -> None:
    sentence = "It opened a second office in Lyon in 2019."
    records = [
        _foreign("r1", sentence, "Lyon", since=2019, odke_verdict="not_found"),
        _foreign("r2", sentence, "Berlin", opened=date(2019, 3, 1)),
    ]
    driver = _driver(records, unread=3)
    with pytest.warns(UserWarning, match=r"3 relationships carry neither .* 'sentence' property"):
        rows, texts = read_neo4j(driver, text_property="sentence")
    assert "r[$text]" in driver.calls[0][1] and driver.calls[0][2]["text"] == "sentence"
    (text,) = texts
    assert text.text == sentence
    lyon, berlin = rows
    assert (lyon.subject, lyon.subject_type, lyon.object, lyon.object_type) == (
        "Halden Robotics",
        "Company",
        "Lyon",
        "City",
    )
    assert lyon.doc == berlin.doc == text.id
    assert lyon.qualifiers == {"since": 2019}
    assert berlin.qualifiers == {"opened": "2019-03-01"}
    assert (lyon.start, lyon.quote) == (None, None)


def test_with_no_text_named_the_warning_says_what_to_name() -> None:
    with pytest.warns(UserWarning, match="text_property="):
        rows, texts = read_neo4j(_driver([], unread=5))
    assert rows == [] and texts == []


# --------------------------------------------------------------------------- #
# Writing verdicts back, when asked
# --------------------------------------------------------------------------- #


def test_verdicts_are_written_back_one_per_relationship_only_when_asked() -> None:
    doc = _halden()
    fact = _facts(doc)[0]
    facts = [
        fact.model_copy(update={"id": "r0", "verdict": GroundingVerdict.NOT_FOUND}),
        fact.model_copy(update={"id": "r0", "verdict": GroundingVerdict.SUPPORTED}),
        fact.model_copy(update={"id": "r1", "verdict": GroundingVerdict.CONTRADICTED}),
        fact.model_copy(update={"id": "r1", "verdict": GroundingVerdict.NOT_FOUND}),
    ]
    driver = _driver([])
    assert write_verdicts(driver, facts, database="graphs") == 2
    ((cypher, params),) = driver.writes
    assert "elementId(r) = row.id" in cypher
    assert params["rows"] == [
        {"id": "r0", "props": {"odke_verdict": "supported"}},
        {"id": "r1", "props": {"odke_verdict": "contradicted"}},
    ]
    assert driver.sessions == [{"database": "graphs"}]
    write_verdicts(driver, facts[:1], property="verdict")
    assert driver.writes[-1][1]["rows"] == [{"id": "r0", "props": {"verdict": "not_found"}}]
    assert write_verdicts(_driver([]), []) == 0


def test_a_graph_read_back_grounds_and_its_verdicts_go_back_to_its_relationships() -> None:
    doc = _halden()
    driver = _driver([_record(f, f"r{i}") for i, f in enumerate(_facts(doc))])
    rows, texts = read_neo4j(driver, documents=[doc])
    client = RecordedClient(
        [
            {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
            {"match": "Halden Robotics", "response": {"verdict": "supported"}},
        ]
    )
    stage = TriplesExtractor(rows, documents=texts)
    kg = Pipeline(Ontology(name="audit"), stage, grounder=LLMGrounder(client=client)).run(texts)
    assert (stage.stats["cited"], stage.stats["context"]) == (1, 3)
    write_verdicts(driver, kg.facts)
    (_, params), *_ = driver.writes
    written = {row["id"]: row["props"]["odke_verdict"] for row in params["rows"]}
    assert written == {"r0": "supported", "r1": "not_found", "r2": "supported", "r3": "supported"}


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_the_sinks_graph_reads_back_from_a_live_neo4j() -> None:
    """Write with the sink, read back the same facts, ground them, and write the verdicts.

    Types are this test's own, and everything it made is deleted afterwards.
    """
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    company, city, place = f"Company_{suffix}", f"City_{suffix}", f"Place_{suffix}"
    doc = _halden()
    facts = _facts(doc, company, city)
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    mine = (
        f"MATCH (n) WHERE n:`{company}` OR n:`{city}` OR n:`{place}` OR "
        f"(n:Claim AND n.subject_type = '{company}') "
    )
    with Neo4jSink(os.environ["NEO4J_URI"], auth) as sink:
        driver = sink._driver
        try:
            sink.write(KnowledgeGraph(facts=tuple(facts)))
            # A relationship of somebody else's graph, with its source sentence on it.
            driver.execute_query(
                f"CREATE (:`{place}` {{name: 'Halden Robotics'}})"
                f"-[:HQ_IN {{sentence: $sentence}}]->(:`{place}` {{name: 'Leeds'}})",
                sentence="Halden Robotics was founded in Leeds.",
            )
            predicates = ["office_in", "founded", "sells", "HQ_IN"]
            # Whatever else a shared database holds may go unread; not this test's count.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows, texts = read_neo4j(
                    driver, documents=[doc], text_property="sentence", predicates=predicates
                )
            rows = [r for r in rows if r.subject_type in {company, place}]
            ours = [r for r in rows if r.subject_type == company]
            again = [to_fact(row, doc) for row in ours]
            assert sorted(map(_same, again), key=repr) == sorted(map(_same, facts), key=repr)
            (foreign,) = [r for r in rows if r.subject_type == place]
            assert (foreign.subject, foreign.predicate, foreign.object) == (
                "Halden Robotics",
                "HQ_IN",
                "Leeds",
            )
            assert doc in texts

            # Reading wrote nothing; writing the verdicts writes one property per relationship.
            ids = sorted({r.id for r in rows})
            check = (
                "MATCH ()-[r]->() WHERE elementId(r) IN $ids "
                "RETURN elementId(r) AS id, r.odke_verdict AS verdict, r.verdict AS receipt"
            )
            before = driver.execute_query(check, ids=ids).records
            assert {record["verdict"] for record in before} == {None}
            client = RecordedClient(
                [
                    {"match": "— Berlin", "response": {"verdict": "not_found"}},
                    {"match": "Halden Robotics", "response": {"verdict": "supported"}},
                ]
            )
            stage = TriplesExtractor(rows, documents=texts)
            kg = Pipeline(Ontology(name="audit"), stage, grounder=LLMGrounder(client=client))
            grounded = kg.run(texts)
            assert write_verdicts(driver, grounded.facts) == len(ids) == 5
            after = {r["id"]: r for r in driver.execute_query(check, ids=ids).records}
            verdicts = {row.object: after[row.id]["verdict"] for row in rows if row.object != 2014}
            assert verdicts["Berlin"] == "not_found"
            assert verdicts["Lyon"] == verdicts["Leeds"] == "supported"
            # The sink's own receipt is left as the first run wrote it.
            assert {after[i]["receipt"] for i in ids} <= {"unchecked", None}
        finally:
            driver.execute_query(mine + "DETACH DELETE n")
