"""Streaming (#158): a run of any size, a micro-batch at a time, in bounded memory.

The micro-batches and the totals that join them; the JSONL sink that appends;
the Validator and `odke run` with `batch_size`; and the done-when: a large
synthetic input whose peak memory grows sublinearly with its size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openodke import Entity, Evidence, Fact, KnowledgeGraph, Span
from openodke.sinks import JsonlSink
from openodke.stream import KEPT, Totals, by_text, micro_batches, shape, streams, write
from openodke.types import EntityLink, LinkKind

ADA = Entity(key="p:ada", type="Person", label="Ada")
ACME = Entity(key="c:acme", type="Company", label="Acme")


def _fact(doc: str, obj: Entity = ACME) -> Fact:
    cited = Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=3))
    return Fact(subject=ADA, predicate="employer", object_entity=obj, evidence=(cited,))


# --------------------------------------------------------------------------- #
# Micro-batches
# --------------------------------------------------------------------------- #


def test_micro_batches_are_read_as_they_are_needed() -> None:
    read: list[int] = []

    def numbers() -> Any:
        for n in range(7):
            read.append(n)
            yield n

    batches = micro_batches(numbers(), 3)
    assert next(batches) == [0, 1, 2] and read == [0, 1, 2]
    assert list(batches) == [[3, 4, 5], [6]]
    with pytest.raises(ValueError, match="at least 1"):
        list(micro_batches([], 0))


def test_a_micro_batch_of_rows_closes_where_the_text_changes_and_never_past_twice_its_size() -> (
    None
):
    rows = ["a", "a", "a", "b", "b", "c"]
    assert list(by_text(rows, 2, str)) == [["a", "a", "a"], ["b", "b"], ["c"]]
    assert list(by_text(["a"] * 5, 2, str)) == [["a"] * 4, ["a"]]


def test_the_totals_sum_each_micro_batch_and_keep_the_first_stop() -> None:
    first = KnowledgeGraph(
        facts=(_fact("d1"),),
        entities=(ADA, ACME),
        links=(EntityLink(source_key="c:a", target_key="c:acme", kind=LinkKind.SIMILAR),),
        stats={
            "documents": 2,
            "chunks": 3,
            "refused": 1,
            "failed": {"d2": "extract: ProviderError: down"},
            "coverage": {
                "sentences": 4,
                "uncovered": 1,
                "missed_entities": 0,
                "not_offered": None,
                "unused": ["born", "hq"],
                "documents": [
                    {"doc_id": "d1", "sentences": 4, "uncovered": [{"start": 0}], "missed": []},
                    {"doc_id": "d2", "sentences": 0, "uncovered": [], "missed": []},
                ],
            },
        },
    )
    second = KnowledgeGraph(
        facts=(_fact("d3"), _fact("d3", ADA)),
        stats={
            "documents": 1,
            "chunks": 1,
            "stopped": {"limit": "calls", "stage": "ground", "unextracted": 0, "unchecked": 2},
            "coverage": {"sentences": 2, "uncovered": 0, "missed_entities": 1, "unused": ["hq"]},
        },
    )
    third = KnowledgeGraph(stats={"documents": 1, "stopped": {"limit": "calls", "unchecked": 3}})
    totals = Totals()
    for kg in (first, second, third):
        totals.add(kg)
    stats = totals.stats()
    assert (stats["documents"], stats["chunks"], stats["refused"]) == (4, 4, 1)
    assert stats["failed"] == {"d2": "extract: ProviderError: down"}
    # The first stop, with what every later micro-batch left unchecked added on.
    assert stats["stopped"] == {
        "limit": "calls",
        "stage": "ground",
        "unextracted": 0,
        "unchecked": 5,
    }
    coverage = stats["coverage"]
    assert (coverage["sentences"], coverage["uncovered"], coverage["missed_entities"]) == (6, 1, 1)
    # Unused only when no micro-batch used it; and only documents with a gap are kept.
    assert coverage["unused"] == ["hq"]
    assert [d["doc_id"] for d in coverage["documents"]] == ["d1"]
    assert totals.graph() == {
        "facts": 3,
        "edges": 3,
        "properties": 0,
        "entities": 2,
        "links": {"similar": 1},
    }
    assert totals.batches == 3 and shape(first)["links"] == {"similar": 1}


def test_a_streamed_coverage_report_keeps_the_first_documents_with_a_gap() -> None:
    record = {"sentences": 1, "uncovered": [{"start": 0}], "missed": []}
    totals = Totals()
    for i in range(3):
        coverage = {
            "sentences": KEPT,
            "uncovered": KEPT,
            "missed_entities": 0,
            "unused": [],
            "documents": [{**record, "doc_id": f"{i}-{n}"} for n in range(KEPT)],
        }
        totals.add(KnowledgeGraph(stats={"coverage": coverage}))
    kept = totals.stats()["coverage"]
    assert kept["uncovered"] == 3 * KEPT and len(kept["documents"]) == KEPT


# --------------------------------------------------------------------------- #
# The JSONL sink appends
# --------------------------------------------------------------------------- #


def test_a_jsonl_sink_writes_the_first_micro_batch_and_appends_the_rest(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out).mkdir()
    (out / "facts.jsonl").write_text("left by an earlier run\n", encoding="utf-8")
    sink = JsonlSink(out)
    one = KnowledgeGraph(facts=(_fact("d1"),), entities=(ADA, ACME), stats={"batch": 1})
    two = KnowledgeGraph(facts=(_fact("d2", ADA),), entities=(ADA,), stats={"batch": 2})
    write([sink], one, first=True)
    write([sink], two, first=False)
    facts = (out / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert [Fact.model_validate_json(line).evidence[0].doc_id for line in facts] == ["d1", "d2"]
    # Ada is in both micro-batches, so she is a line in each.
    assert len((out / "entities.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["facts"], manifest["edges"], manifest["entities"]) == (2, 2, 3)
    assert manifest["stats"] == {"batch": 2}
    assert manifest["created_at"] == one.created_at.isoformat()


def test_a_merging_jsonl_sink_folds_micro_batches_into_one_line_each(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path, merge=True)
    write([sink], KnowledgeGraph(facts=(_fact("d1"),)), first=True)
    write([sink], KnowledgeGraph(facts=(_fact("d1"),)), first=False)
    assert len((tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_sink_that_writes_the_whole_graph_says_it_does_not_stream(tmp_path: Path) -> None:
    from openodke.sinks.bulk import CypherFileSink, Neo4jAdminCsvSink

    assert not streams(CypherFileSink(tmp_path / "g.cypher"))
    assert not streams(Neo4jAdminCsvSink(tmp_path / "import"))
    assert streams(JsonlSink(tmp_path)) and streams(object())
