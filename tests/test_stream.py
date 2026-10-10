"""Streaming (#158): a run of any size, a micro-batch at a time, in bounded memory.

The micro-batches and the totals that join them; the JSONL sink that appends;
the Validator and `odke run` with `batch_size`; and the done-when: a large
synthetic input whose peak memory grows sublinearly with its size.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from openodke import Document, Entity, Evidence, Fact, KnowledgeGraph, Span, Validator
from openodke.interop import to_fact
from openodke.interop.triples import TripleRow
from openodke.llm.budget import Budget, Ledger
from openodke.llm.testing import RecordedClient
from openodke.sinks import JsonlSink
from openodke.stream import KEPT, Totals, by_text, micro_batches, shape, streams, write
from openodke.types import EntityLink, LinkKind
from test_validator import PLACES, RECORDED, ROWS, A, B

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


# --------------------------------------------------------------------------- #
# The Validator, a micro-batch at a time
# --------------------------------------------------------------------------- #


def _held(directory: Path) -> dict[Any, int]:
    """What a JSONL store holds: each fact's signature and its support."""
    lines = (directory / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    return {(f := Fact.model_validate_json(line)).signature: f.support for line in lines}


def test_micro_batches_merge_through_the_store_into_the_facts_one_batch_makes(
    tmp_path: Path,
) -> None:
    """Two micro-batches stating one claim make one fact with both sources, as one batch does."""
    whole, streamed = (
        JsonlSink(tmp_path / "whole", merge=True),
        JsonlSink(tmp_path / "s", merge=True),
    )
    Validator(PLACES, client=RecordedClient(RECORDED), sinks=[whole]).validate(ROWS, [A, B])
    kg, report = Validator(PLACES, client=RecordedClient(RECORDED), sinks=[streamed]).validate(
        ROWS, [A, B], batch_size=1
    )
    assert _held(tmp_path / "s") == _held(tmp_path / "whole")
    assert max(_held(tmp_path / "s").values()) == 2
    # The rows of one text that sit together stay together: a, b, a a, b.
    assert report.batches == 4 and kg.facts == ()
    # The second text's claim met the first in the store, not in memory.
    assert report.restated == 2 and report.merged == 0
    assert (report.facts_in, report.documents, report.refused) == (5, 2, 2)
    assert report.verdicts == {"supported": 3, "contradicted": 1, "not_found": 0, "unchecked": 1}
    assert "5 facts from 2 texts, in 4 micro-batches" in report.render()
    assert kg.stats["validation"]["batches"] == 4 and kg.stats["graph"]["facts"] == 5


def test_a_streamed_job_appends_each_micro_batch_and_the_manifest_holds_the_run(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    c = Document(id="c", text="A text no row cites.")
    kg, report = Validator(
        PLACES, client=RecordedClient(RECORDED), sinks=[JsonlSink(out)]
    ).validate(rows, [A, B, c], batch_size=1)
    # Without a store to merge with, the claim both texts state is a line in each.
    lines = (out / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == report.facts_out == 5
    assert len({Fact.model_validate_json(line).signature for line in lines}) == 3
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["facts"] == 5 and manifest["stats"]["validation"]["batches"] == 5
    # The text no row cites was run too, as one batch runs every text given.
    assert report.documents == manifest["stats"]["documents"] == 3


def test_a_streamed_job_replays_facts_and_refuses_a_sink_that_writes_the_whole_graph(
    tmp_path: Path,
) -> None:
    from openodke.sinks.bulk import CypherFileSink

    facts = [to_fact(TripleRow.model_validate(row), A if row["doc"] == "a" else B) for row in ROWS]
    _, report = Validator(PLACES, client=RecordedClient(RECORDED)).validate(
        facts, [A, B], batch_size=2
    )
    assert report.facts_in == 5 and report.batches == 3
    with pytest.raises(ValueError, match="CypherFileSink writes the whole graph"):
        Validator(PLACES, sinks=[CypherFileSink(tmp_path / "g.cypher")]).validate(
            ROWS, [A, B], batch_size=2
        )
    with pytest.raises(ValueError, match="rows or Facts, not both"):
        Validator(PLACES, client=RecordedClient(RECORDED)).validate(
            [facts[0], ROWS[1]], [A, B], batch_size=2
        )


def test_a_budget_stop_in_one_micro_batch_stops_the_rest_and_keeps_what_each_has(
    tmp_path: Path,
) -> None:
    """One ledger for the run (#157): the micro-batches after the stop ask nothing."""
    client = RecordedClient(RECORDED)
    held = Ledger(Budget(calls=1)).client(client)
    kg, report = Validator(PLACES, client=held, sinks=[JsonlSink(tmp_path)]).validate(
        ROWS, [A, B], batch_size=1
    )
    assert len(client.calls) == 1
    assert report.stopped is not None and report.stopped["unchecked"] >= 1
    assert report.verdicts["supported"] == 1
    # Every micro-batch still wrote what it kept.
    assert len((tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()) >= 3
    assert kg.stats["stopped"] == report.stopped


# --------------------------------------------------------------------------- #
# Done when: a large synthetic input runs in bounded memory
# --------------------------------------------------------------------------- #

# Run in a fresh interpreter, so the peak is this job's and not the test session's.
MEASURE = r"""
import json, resource, sys, tempfile
from pathlib import Path

from openodke import Document, Ontology, Validator
from openodke.llm.base import Completion
from openodke.sinks import JsonlSink

TEXTS, PER_TEXT = 5000, 10
ANSWER = '{"verdict": "supported"}'


class Supported:
    # A scripted client that keeps nothing: every claim is supported.
    def complete(self, messages, *, spec, schema=None):
        return Completion(text=ANSWER, parsed={"verdict": "supported"}, model=spec.model)


def sentence(i, k):
    return f"Firm{i} has an office in Town{k}."


def peak():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def job(rows, texts, out, batch_size):
    path = out / f"rows-{rows}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for n in range(rows):
            i, k = divmod(n, PER_TEXT)
            row = {"doc": f"t{i}", "subject": f"Firm{i}", "predicate": "office_in",
                   "object": f"Town{k}", "quote": sentence(i, k)}
            fh.write(json.dumps(row) + "\n")
    sink = JsonlSink(out / f"graph-{rows}-{batch_size}")
    _, report = Validator(ONTOLOGY, client=Supported(), sinks=[sink]).validate(
        path, texts, batch_size=batch_size
    )
    assert report.facts_out == rows, report.render()
    return peak()


ONTOLOGY = Ontology.from_dict({
    "name": "offices",
    "types": {"Firm": {}, "Town": {}},
    "predicates": {"office_in": {"domain": ["Firm"], "range": "Town"}},
})
out = Path(tempfile.mkdtemp(dir=sys.argv[1]))
small, large, size = (int(arg) for arg in sys.argv[2:5])
# Everything imported and every stage warmed, before anything is measured.
first = " ".join(sentence(0, k) for k in range(PER_TEXT))
job(PER_TEXT, [Document(id="t0", text=first)], out, size)
base = peak()
# The texts are held, because a row may cite any of them; the same texts for both sizes.
texts = [
    Document(id=f"t{i}", text=" ".join(sentence(i, k) for k in range(PER_TEXT)))
    for i in range(TEXTS)
]
streamed = [job(small, texts, out, size), job(large, texts, out, size)]
# The control, last since a peak only rises: the same measurement sees an
# unbatched job, on a tenth of the rows and the texts they cite, grow with them.
whole = [job(n, texts[: n // PER_TEXT], out, None) for n in (small // 10, large // 10)]
print(json.dumps({"base": base, "streamed": streamed, "whole": whole}))
"""


def test_a_large_synthetic_input_runs_in_bounded_memory(tmp_path: Path) -> None:
    """Twice the rows, the same peak: 25,000 and 50,000 triples rows, a scripted client.

    Peak RSS in a fresh interpreter, after the same 5,000 texts are loaded,
    which a streamed job holds whatever its size. Every stage is the
    Validator's default but the client, and the graph goes to JSONL a
    micro-batch at a time. Linear growth would double what the run adds to the
    interpreter; the bound is half that. The control runs the same job
    unbatched on a tenth of the rows, and grows with them.
    """
    done = subprocess.run(
        [sys.executable, "-c", MEASURE, str(tmp_path), "25000", "50000", "1000"],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    measured = json.loads(done.stdout.splitlines()[-1])
    base, (small, large), (few, more) = (
        measured["base"],
        measured["streamed"],
        measured["whole"],
    )
    assert large - base < 1.5 * (small - base), measured
    assert more - base > 1.5 * (few - base), measured
