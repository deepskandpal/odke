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
import yaml
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact, KnowledgeGraph, Span, Validator
from openodke.cli.main import app
from openodke.corroborate.provenance import ONTOLOGY
from openodke.interop import to_fact
from openodke.interop.triples import TripleRow
from openodke.llm.budget import Budget, Ledger
from openodke.llm.testing import RecordedClient
from openodke.manifest import read_manifest
from openodke.observe import configure_logs
from openodke.run import ConfigError, StageSpec, build, execute, load_config
from openodke.sinks import JsonlSink
from openodke.stream import KEPT, Totals, by_text, micro_batches, shape, streams, write
from openodke.types import EntityLink, LinkKind
from test_manifest import _facts, _timeless
from test_observe import _events, _one
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


def test_the_resolvers_per_call_counts_are_summed_over_micro_batches() -> None:
    """Its store lookup's and batch normalisation's counts describe one call each."""

    class Resolver:
        stats = {"store": {"looked_up": 2, "rekeyed": 1}, "batch": {"alike": 3, "merged": 2}}

    totals = Totals()
    for _ in range(3):
        totals.add(KnowledgeGraph(), resolver=Resolver())
    assert totals.store == {"looked_up": 6, "rekeyed": 3}
    assert totals.resolver("batch") == {"alike": 9, "merged": 6}
    assert Totals().resolver("batch") is None


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
# odke run and odke validate, a micro-batch at a time
# --------------------------------------------------------------------------- #


def test_odke_run_streams_the_example_with_the_calls_and_counts_of_one_batch(
    example: Path,
) -> None:
    whole = execute(load_config(example / "odke.yaml"))
    streamed = execute(load_config(example / "odke.yaml"), batch_size=2)
    for key in ("documents", "chunks", "empty_extractions", "refused"):
        assert streamed.stats[key] == whole.stats[key], key
    # The same documents asked the same questions: one ledger, the same bill.
    assert streamed.stats["spent"]["calls"] == whole.stats["spent"]["calls"] > 0
    assert streamed.stats["batches"] == -(-whole.stats["documents"] // 2)
    assert streamed.graph.facts == () and streamed.stats["graph"]["facts"] >= len(whole.graph.facts)
    out = example / "out"
    lines = (out / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == streamed.stats["graph"]["facts"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["facts"] == len(lines)
        and manifest["stats"]["batches"] == streamed.stats["batches"]
    )
    text = streamed.render()
    assert f"in {streamed.stats['batches']} micro-batches" in text
    assert "appended in" in streamed.written[0]

    # A dry run streams too, and prints the first facts it would have written.
    dry = execute(load_config(example / "odke.yaml").with_batch_size(3), dry_run=True)
    assert dry.sample and "facts that would be written:" in dry.render()


def test_a_streamed_run_reads_its_documents_one_at_a_time_under_the_ids_one_batch_gives(
    example: Path,
) -> None:
    config = load_config(example / "odke.yaml")
    twice = config.model_copy(update={"inputs": (*config.inputs, config.inputs[0])})
    built = build(twice)
    streamed = [doc.id for doc in built.iter_documents()]
    assert streamed == [doc.id for doc in built.documents()]
    assert any(name.endswith("~2") for name in streamed)


def test_a_streamed_run_refuses_a_sink_that_writes_the_whole_graph(example: Path) -> None:
    config = load_config(example / "odke.yaml")
    sink = StageSpec.model_validate({"use": "cypher_file", "path": "out/graph.cypher"})
    whole = config.model_copy(update={"stages": config.stages.model_copy(update={"sink": (sink,)})})
    with pytest.raises(ConfigError, match="cypher_file writes a file of the whole graph"):
        build(whole.with_batch_size(2))
    with pytest.raises(ConfigError, match="batch_size: cypher_file"):
        execute(whole, batch_size=2)


def test_the_commands_take_batch_size_and_a_bad_row_is_named_mid_stream(
    example: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = CliRunner().invoke(app, ["run", str(example / "odke.yaml"), "--batch-size", "3"])
    assert result.exit_code == 0, result.output
    assert "micro-batches" in result.output and "appended in" in result.output

    monkeypatch.chdir(tmp_path)
    triples = Path(__file__).parent.parent / "examples" / "triples"
    args = ["validate", "--facts", str(triples / "triples.jsonl"), "--texts"]
    args += [str(triples / "texts"), "-o", "out", "--dry-run", "--batch-size", "2"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    # One text: its rows stay together up to twice the size, then split.
    assert "5 facts from 1 text, in 2 micro-batches" in result.output

    rows = (triples / "triples.jsonl").read_text(encoding="utf-8")
    (tmp_path / "bad.jsonl").write_text(rows + "{not json\n", encoding="utf-8")
    args[2] = str(tmp_path / "bad.jsonl")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "bad.jsonl:6: not JSON" in result.output


# --------------------------------------------------------------------------- #
# The run manifest, the job's events, and a replay
# --------------------------------------------------------------------------- #


def _streamed_config(example: Path, size: int = 2) -> Path:
    """The example's config with `batch_size`, beside it."""
    data = yaml.safe_load((example / "odke.yaml").read_text(encoding="utf-8"))
    path = example / "streamed.yaml"
    path.write_text(yaml.safe_dump({**data, "batch_size": size}), encoding="utf-8")
    return path


def test_a_streamed_manifest_holds_the_summed_run_and_the_inputs_one_batch_hashes(
    example: Path,
) -> None:
    whole = execute(load_config(example / "odke.yaml"))
    streamed = execute(load_config(_streamed_config(example)))
    one, many = whole.manifest, streamed.manifest
    assert one is not None and many is not None
    # Every micro-batch's documents, hashed into the digest one batch takes.
    assert many.inputs == one.inputs
    # batch_size is part of the run, and of its hash.
    assert (many.config["batch_size"], one.config["batch_size"]) == (2, None)
    assert many.config_hash != one.config_hash
    # The counts are the micro-batches' summed.
    assert many.counts["batches"] == streamed.stats["batches"] > 1 and "batches" not in one.counts
    for key in ("documents", "chunks", "refused", "empty_extractions"):
        assert many.counts[key] == one.counts[key], key
    assert many.counts["facts"] == streamed.stats["graph"]["facts"] >= one.counts["facts"]
    assert many.spent == one.spent
    # And so are the job's: every candidate the micro-batches took in.
    assert many.job == streamed.job.model_dump() and streamed.job.facts_in == whole.job.facts_in
    written = read_manifest(example / "out")
    assert written.counts == many.counts and written.inputs == many.inputs
    on_disk = json.loads((example / "out" / "manifest.json").read_text(encoding="utf-8"))
    lines = (example / "out" / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert on_disk["facts"] == len(lines) == many.counts["facts"]
    # Every micro-batch's gate stamps the ontology it checked under (#163).
    stamps = {json.loads(line)["qualifiers"].get(ONTOLOGY) for line in lines}
    assert stamps == {many.ontology_hash} and many.ontology_hash is not None


def test_a_streamed_validation_hashes_its_rows_as_one_batch_does(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    # Out of order on purpose: a, b, a, a, b.
    rows.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    _, whole = Validator(PLACES, client=RecordedClient(RECORDED)).validate(rows, [A, B])
    _, streamed = Validator(PLACES, client=RecordedClient(RECORDED)).validate(
        rows, [A, B], batch_size=1
    )
    assert whole.manifest is not None and streamed.manifest is not None
    assert streamed.manifest.inputs == whole.manifest.inputs
    assert streamed.manifest.inputs.facts is not None
    assert streamed.manifest.inputs.facts.rows == 5
    assert streamed.manifest.counts["batches"] == streamed.batches
    assert streamed.manifest.job == streamed.job.model_dump()
    assert streamed.manifest.config["batch_size"] == 1


def test_a_streamed_run_is_replayed_streamed_from_its_manifest(example: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["run", str(_streamed_config(example))]).exit_code == 0
    first = json.loads((example / "out" / "manifest.json").read_text(encoding="utf-8"))
    facts = _facts(example / "out")

    result = runner.invoke(app, ["run", "--from-manifest", str(example / "out")])
    assert result.exit_code == 0, result.output
    assert f"config {first['config_hash'][:12]}" in result.output
    assert f"in {first['counts']['batches']} micro-batches" in result.output
    second = json.loads((example / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert _timeless(second) == _timeless(first) and _facts(example / "out") == facts

    # The micro-batch size is part of the run, so a replay takes no other.
    result = runner.invoke(
        app, ["run", "--from-manifest", str(example / "out"), "--batch-size", "3"]
    )
    assert result.exit_code == 2 and "--batch-size" in result.output
    # A changed text is refused before the first micro-batch is written.
    note = example / "corpus" / "notes" / "corvid-analytics.md"
    note.write_text(note.read_text(encoding="utf-8") + "\nCorvid moved.\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--from-manifest", str(example / "out")])
    assert result.exit_code == 2
    assert "documents changed: corpus/notes/corvid-analytics.md" in result.output
    assert json.loads((example / "out" / "manifest.json").read_text(encoding="utf-8")) == second


def test_a_streamed_jobs_manifest_and_its_log_events_say_the_same(
    example: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One streamed run and one streamed validation: job.end is the summed run, as the manifest."""
    runner = CliRunner()
    args = ["run", str(example / "odke.yaml"), "--batch-size", "2", "--log-format", "json"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    manifests, streams = [read_manifest(example / "out")], [_events(result.stderr)]

    monkeypatch.chdir(tmp_path)
    (tmp_path / "recorded.json").write_text(
        json.dumps([{"match": "Claim:", "response": {"verdict": "supported"}}])
    )
    (tmp_path / "models.yaml").write_text("models:\n  replay:\n    ground: recorded.json\n")
    triples = Path(__file__).parent.parent / "examples" / "triples"
    args = ["validate", "--facts", str(triples / "triples.jsonl"), "--texts"]
    args += [str(triples / "texts"), "--config", "models.yaml", "-o", "v"]
    result = runner.invoke(app, [*args, "--batch-size", "2", "--log-format", "json"])
    assert result.exit_code == 0, result.output
    manifests.append(read_manifest(tmp_path / "v"))
    streams.append(_events(result.stderr))
    configure_logs("text")

    for manifest, events in zip(manifests, streams, strict=True):
        end = _one(events, "job.end")
        _one(events, "job.start")
        assert manifest.run == end["run"] and {e["run"] for e in events} == {manifest.run}
        assert manifest.job == end["counts"] and manifest.spent == end["cost"]
        assert manifest.counts["documents"] == end["documents"]
        # One write a micro-batch, each its own event.
        writes = [e for e in events if e["event"] == "stage" and e["stage"] == "write"]
        assert len(writes) == manifest.counts["batches"] > 1
        # And one write report for the run: the JSONL sink's every appended line.
        assert manifest.writes == end["writes"]
        (jsonl,) = manifest.writes.values()
        facts = manifest.counts["facts" if manifest.command == "run" else "facts_out"]
        assert jsonl["facts"] == {"written": facts, "merged": 0, "skipped": 0}


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
    # Linux keeps the forking process's peak in ru_maxrss across exec; VmHWM is this one's.
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**10
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

    Peak RSS (VmHWM) in a fresh interpreter, after the same 5,000 texts are loaded,
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
