"""`odke eval pipeline` (#138): your pipeline run three ways, scored to one report.

The pipelines here are stand-ins that look their triples up rather than
extract them, so every number is fixed by hand and no model is called; the
Validator's grounder answers from recorded responses. Not a benchmark.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact
from openodke.cli.main import app
from openodke.eval import read_report
from openodke.eval.datasets import text2kgbench
from openodke.eval.eval_report import EvalReport
from openodke.eval.harness import (
    PipelineError,
    evaluate_pipeline,
    labelled,
    load_callable,
    run_command,
)
from openodke.interop.triples import TripleRow
from test_datasets import _t2k_raw

TRIPLES = Path(__file__).resolve().parents[1] / "examples" / "triples"
PYTHON = shlex.quote(sys.executable)
runner = CliRunner()

# A stand-in pipeline: the rows in the JSON file named by its third argument,
# for the texts it finds in {in}.
FAKE = """\
import json, pathlib, sys
inbox, out, table = (pathlib.Path(a) for a in sys.argv[1:4])
names = {p.stem for p in inbox.glob("*.txt")}
rows = [r for r in json.loads(table.read_text()) if r["doc"] in names]
out.write_text("".join(json.dumps(r) + "\\n" for r in rows))
"""

BENCH_ROWS = [
    # Sentence 1: its director, right; sentence 2: a director it does not state.
    {"doc": "ont_1_movie_test_1", "subject": "Bleach: Hell Verse", "predicate": "director",
     "object": "Noriyuki Abe"},
    {"doc": "ont_1_movie_test_2", "subject": "Keyboard Cat", "predicate": "director",
     "object": "Somebody Else"},
]  # fmt: skip


def labels_of_the_example() -> dict[str, Any]:
    return {
        "labels": TRIPLES / "gold.jsonl",
        "documents": TRIPLES / "texts",
        "ontology": TRIPLES / "ontology.json",
    }


@pytest.fixture
def fake(tmp_path: Path) -> Path:
    script = tmp_path / "fake_pipeline.py"
    script.write_text(FAKE)
    return script


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """A tiny prepared Text2KGBench set: two sentences, three gold triples."""
    return text2kgbench.prepare(_t2k_raw(tmp_path), "ont_1_movie", tmp_path / "set")


def _same(*reports: EvalReport) -> None:
    first = reports[0]
    for other in reports[1:]:
        assert other.rows == first.rows
        assert other.stages == first.stages
        assert other.bootstrap == first.bootstrap
        assert other.run == first.run


# --------------------------------------------------------------------------- #
# Three modes, one report
# --------------------------------------------------------------------------- #


def test_three_modes_score_the_triples_example_to_the_same_report() -> None:
    pipeline = TRIPLES / "pipeline.py"
    by_command = evaluate_pipeline(
        command=f"{PYTHON} {shlex.quote(str(pipeline))} {{in}} {{out}}", **labels_of_the_example()
    )
    by_callable = evaluate_pipeline(function=f"{pipeline}:extract", **labels_of_the_example())
    by_files = evaluate_pipeline(predictions=TRIPLES / "triples.jsonl", **labels_of_the_example())
    _same(by_command, by_callable, by_files)
    (row,) = by_files.rows
    # Founder, Lyon and the chief executive are right; Berlin is spurious, and
    # 2012 is a wrong value: one written wrong, the gold 2014 not written.
    c = row.counts
    assert (c.hits, c.over_extraction, c.under_extraction) == (3, 2, 1)
    assert row.performance.precision.value == pytest.approx(3 / 5)
    assert row.performance.recall.value == pytest.approx(3 / 4)
    assert row.conformance is not None and row.conformance.rate == 1.0
    assert by_files.title == "pipeline" and by_files.stages[0].stage == "pipeline"
    assert by_command.notes[0].startswith("command ")
    assert by_callable.notes[0].startswith("callable ")
    assert by_files.notes[0] == "triples file triples.jsonl: 5 row(s), 5 fact(s)"


def test_three_modes_score_a_prepared_bench_set_to_the_same_report(
    tmp_path: Path, fake: Path, bench: Path
) -> None:
    table = tmp_path / "rows.json"
    table.write_text(json.dumps(BENCH_ROWS))
    written = tmp_path / "written.jsonl"
    written.write_text("".join(json.dumps(r) + "\n" for r in BENCH_ROWS))

    def pipeline(documents: list[Document]) -> list[dict[str, Any]]:
        names = {doc.id for doc in documents}
        return [row for row in BENCH_ROWS if row["doc"] in names]

    by_command = evaluate_pipeline(
        command=f"{PYTHON} {shlex.quote(str(fake))} {{in}} {{out}} {table}", bench=bench
    )
    by_callable = evaluate_pipeline(function=pipeline, bench=bench)
    by_files = evaluate_pipeline(predictions=written, bench=bench)
    _same(by_command, by_callable, by_files)

    # The benchmark's own arithmetic over the same triples, sentence by sentence.
    gold = [json.loads(line) for line in (bench / "gold.jsonl").read_text().splitlines()]
    meta = json.loads((bench / "dataset.json").read_text())
    predicted = {r["doc"]: [(r["subject"], r["predicate"], r["object"])] for r in BENCH_ROWS}
    expected = text2kgbench.score(gold, predicted, meta["ontology"])
    (row,) = by_files.rows
    assert row.performance.precision.value == pytest.approx(expected["precision"]) == 0.5
    assert row.performance.recall.value == pytest.approx(expected["recall"]) == 0.25
    assert row.performance.average == "macro"
    assert (row.counts.hits, row.counts.over_extraction, row.counts.under_extraction) == (1, 1, 2)
    assert by_files.run.dataset is not None and by_files.run.dataset.name == "text2kgbench"
    assert by_files.stages[0].stage == "text2kgbench:ont_1_movie"


# --------------------------------------------------------------------------- #
# The command mode
# --------------------------------------------------------------------------- #


def _corpus() -> Any:
    return labelled(**labels_of_the_example())


def test_the_command_runs_without_a_shell(tmp_path: Path, fake: Path) -> None:
    """A `;` in the template is a word handed to the program, never a second command."""
    table = tmp_path / "rows.json"
    table.write_text("[]")
    planted = tmp_path / "planted"
    template = f"{PYTHON} {fake} {{in}} {{out}} {table} ; touch {planted}"
    output = run_command(template, _corpus())
    assert output.items == [] and not planted.exists()


def test_the_command_reads_one_text_per_document_named_by_its_id(tmp_path: Path) -> None:
    script = tmp_path / "list.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "inbox, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
        "rows = [{'doc': p.stem, 'subject': 'S', 'predicate': 'p', 'object': p.read_text()[:7]}"
        " for p in sorted(inbox.iterdir())]\n"
        "out.write_text(''.join(json.dumps(r) + '\\n' for r in rows))\n"
    )
    output = run_command(f"{PYTHON} {script} {{in}} {{out}}", _corpus())
    assert [(r.doc, r.object) for r in output.items] == [("halden", "Halden ")]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("import sys; sys.stderr.write('boom: no key\\n'); sys.exit(3)", "exited 3:\nboom: no key"),
        ("pass", "wrote nothing at {out}"),
        ("import time; time.sleep(5)", "ran past 0.5 s"),
    ],
)
def test_a_failing_command_says_how_it_failed(tmp_path: Path, body: str, message: str) -> None:
    script = tmp_path / "broken.py"
    script.write_text(body + "\n")
    with pytest.raises(PipelineError) as caught:
        run_command(f"{PYTHON} {script} {{in}} {{out}}", _corpus(), timeout=0.5)
    assert message in str(caught.value)


def test_a_command_must_name_both_folders_and_exist() -> None:
    with pytest.raises(ValueError, match=r"needs \{in\} and \{out\}"):
        run_command(f"{PYTHON} extract.py {{in}}", _corpus())
    with pytest.raises(PipelineError, match="no such program"):
        run_command("odke-no-such-program {in} {out}", _corpus())


# --------------------------------------------------------------------------- #
# The callable and files modes
# --------------------------------------------------------------------------- #


def _halden(predicate: str, value: Any, doc: str = "halden.txt") -> Fact:
    subject = Entity(key="Company:halden robotics", type="Company", label="Halden Robotics")
    return Fact(
        subject=subject, predicate=predicate, object_value=value, evidence=(Evidence(doc_id=doc),)
    )


def test_a_callable_may_return_facts_rows_or_both() -> None:
    def pipeline(documents: list[Document]) -> list[Any]:
        assert [doc.id for doc in documents] == ["halden.txt"]
        return [
            _halden("founded", 2014, doc="halden"),  # named by its file's stem: renamed
            TripleRow(
                doc="halden", subject="Halden Robotics", predicate="office_in", object="Lyon"
            ),
            {"doc": "nowhere", "subject": "X", "predicate": "founder", "object": "Y"},
        ]

    report = evaluate_pipeline(function=pipeline, **labels_of_the_example())
    (row,) = report.rows
    assert (row.counts.hits, row.counts.over_extraction, row.counts.under_extraction) == (2, 0, 2)
    assert "1 row(s) named no document the pipeline was given and were left out: nowhere" in (
        report.notes
    )


def test_a_callable_that_returns_nothing_usable_is_an_error() -> None:
    with pytest.raises(PipelineError, match="returned str"):
        evaluate_pipeline(function=lambda documents: "facts", **labels_of_the_example())


def test_a_callable_loads_from_a_module_or_a_file() -> None:
    assert callable(load_callable(f"{TRIPLES / 'pipeline.py'}:extract"))
    assert load_callable("json:dumps")([1]) == "[1]"
    with pytest.raises(ValueError, match="cannot load"):
        load_callable("json:nothing_here")
    with pytest.raises(ValueError, match="module:function"):
        load_callable("json")


def test_files_read_a_sink_s_facts_and_an_adapter_s_output(tmp_path: Path) -> None:
    facts = tmp_path / "facts.jsonl"
    facts.write_text(_halden("founded", 2014).model_dump_json() + "\n")
    report = evaluate_pipeline(predictions=facts, **labels_of_the_example())
    assert report.rows[0].counts.hits == 1
    assert report.notes[0] == "facts file facts.jsonl: 1 row(s), 1 fact(s)"

    # A LangChain GraphDocument dump, its source matched to the document by its text.
    text = (TRIPLES / "texts" / "halden.txt").read_text()
    company = {"id": "Halden Robotics", "type": "Company"}
    graph = {
        "nodes": [company, {"id": "Mara Quist", "type": "Person"}],
        "relationships": [
            {"source": company, "target": {"id": "Mara Quist", "type": "Person"}, "type": "founder"}
        ],
        "source": {"page_content": text, "metadata": {}},
    }
    dump = tmp_path / "graph.json"
    dump.write_text(json.dumps([graph]))
    report = evaluate_pipeline(predictions=dump, adapter="langchain", **labels_of_the_example())
    assert report.rows[0].counts.hits == 1


def test_the_inputs_are_checked_before_anything_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one of --cmd, --run and --predictions"):
        evaluate_pipeline(**labels_of_the_example())
    with pytest.raises(ValueError, match="--labels .* or --bench"):
        evaluate_pipeline(predictions=TRIPLES / "triples.jsonl")
    with pytest.raises(ValueError, match="--labels needs --documents"):
        evaluate_pipeline(predictions=TRIPLES / "triples.jsonl", labels=TRIPLES / "gold.jsonl")
    with pytest.raises(ValueError, match="unknown adapter"):
        evaluate_pipeline(
            predictions=TRIPLES / "triples.jsonl", adapter="spacy", **labels_of_the_example()
        )
    gold = tmp_path / "gold.jsonl"
    gold.write_text((TRIPLES / "gold.jsonl").read_text().replace("halden.txt", "elsewhere.txt"))
    with pytest.raises(ValueError, match="not among the documents: elsewhere.txt"):
        labelled(gold, TRIPLES / "texts")


# --------------------------------------------------------------------------- #
# --validator
# --------------------------------------------------------------------------- #


@pytest.fixture
def strict(tmp_path: Path) -> Iterator[Path]:
    """The example's config with a gate that refuses `not_found` too."""
    copy = tmp_path / "triples"
    shutil.copytree(TRIPLES, copy, ignore=shutil.ignore_patterns("out"))
    config = yaml.safe_load((copy / "odke.yaml").read_text())
    config["stages"]["gate"] = {"use": "verdict", "refuse_not_found": True}
    path = copy / "strict.yaml"
    path.write_text(yaml.safe_dump(config))
    yield path


def test_the_validator_checks_the_same_output_and_both_rows_are_reported(strict: Path) -> None:
    report = evaluate_pipeline(
        predictions=TRIPLES / "triples.jsonl",
        validator=True,
        config=strict,
        **labels_of_the_example(),
    )
    pipeline, checked = report.rows
    assert (pipeline.name, checked.name) == ("pipeline", "+ validator")
    assert pipeline.cost is None
    # Berlin comes back not_found from the model, and 2012's quote is nowhere in
    # the text: the strict gate refuses both, and only the gold 2014 is missed.
    c = checked.counts
    assert (c.hits, c.over_extraction, c.under_extraction) == (3, 0, 1)
    assert checked.performance.precision.value == 1.0
    assert checked.cost is not None and checked.cost.calls == 4
    assert checked.cost.usd is None  # recorded responses carry no price
    assert report.run.prompts == ("ground.span@1",)
    assert report.run.models == {"ground": "anthropic/claude-haiku-4-5-20251001"}
    assert [s.stage for s in report.stages] == ["pipeline", "+ validator"]
    assert (
        "+ validator: openodke.Validator over the same facts, with strict.yaml: grounded "
        "supported 3, contradicted 0, not_found 2, unchecked 0; 2 refused (2 not_found), "
        "0 merged, 0 linked, 0 derived"
    ) in report.notes


def test_the_validator_takes_the_example_s_own_config_as_it_is(strict: Path) -> None:
    """Its gate keeps not_found, so here the Validator's row scores as the pipeline's."""
    report = evaluate_pipeline(
        predictions=TRIPLES / "triples.jsonl",
        validator=True,
        config=strict.with_name("odke.yaml"),
        **labels_of_the_example(),
    )
    pipeline, checked = report.rows
    assert checked.performance == pipeline.performance
    with pytest.raises(ValueError, match="--config is for --validator"):
        evaluate_pipeline(
            predictions=TRIPLES / "triples.jsonl", config=strict, **labels_of_the_example()
        )


# --------------------------------------------------------------------------- #
# From the shell
# --------------------------------------------------------------------------- #


def _shell(*extra: str) -> list[str]:
    return [
        "eval", "pipeline",
        "--labels", str(TRIPLES / "gold.jsonl"),
        "--documents", str(TRIPLES / "texts"),
        "--ontology", str(TRIPLES / "ontology.json"),
        *extra,
    ]  # fmt: skip


def test_odke_eval_pipeline_from_the_shell(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    template = f"{sys.executable} {TRIPLES / 'pipeline.py'} {{in}} {{out}}"
    result = runner.invoke(app, _shell("--cmd", template, "--report", str(out)))
    assert result.exit_code == 0, result.output
    assert result.output.startswith("pipeline  (n=4)\n")
    assert "0.600 [0.600, 0.600]" in result.output
    assert read_report(out).rows[0].counts.hits == 3

    as_json = runner.invoke(app, _shell("--run", f"{TRIPLES / 'pipeline.py'}:extract", "--json"))
    assert as_json.exit_code == 0, as_json.output
    assert EvalReport.model_validate_json(as_json.output).rows == read_report(out).rows


def test_odke_eval_pipeline_runs_the_bench_set_s_own_checks(tmp_path: Path, bench: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in BENCH_ROWS))
    items = tmp_path / "items.jsonl"
    args = ["eval", "pipeline", "--bench", str(bench), "--predictions", str(rows)]
    result = runner.invoke(app, [*args, "--items", str(items)])
    assert result.exit_code == 0, result.output
    assert "pipeline  0.500 [" in result.output
    # Text2KGBench's scorer has no view for the diagnosis yet, and says so.
    assert "no diagnosis: text2kgbench:ont_1_movie's scorer has no view for one yet" in (
        result.output
    )
    assert "where it loses facts" not in result.output
    written = [json.loads(line) for line in items.read_text().splitlines()]
    assert [(r["tp"], r["fp"], r["fn"]) for r in written] == [(1, 0, 1), (0, 1, 1)]


@pytest.mark.parametrize(
    ("extra", "message", "code"),
    [
        ([], "exactly one of --cmd, --run and --predictions", 2),
        (["--predictions", "p.jsonl", "--cmd", "x {in} {out}"], "exactly one of", 2),
        (["--predictions", str(TRIPLES / "triples.jsonl"), "--config", "x.yaml"], "--config", 2),
        (["--predictions", str(TRIPLES / "triples.jsonl"), "--applied", "inverses"], "compare", 2),
        (["--cmd", f"{sys.executable} -c pass {{in}} {{out}}"], "wrote nothing at {out}", 1),
    ],
)
def test_pipeline_mistakes_and_failures_exit_with_a_message(
    extra: list[str], message: str, code: int
) -> None:
    result = runner.invoke(app, _shell(*extra))
    assert result.exit_code == code, result.output
    assert message in result.output


def test_the_pipeline_flags_belong_to_pipeline() -> None:
    labels = str(TRIPLES / "gold.jsonl")
    result = runner.invoke(app, ["eval", "extract", "--labels", labels, "--cmd", "x {in} {out}"])
    assert result.exit_code == 2
    assert "are for pipeline" in result.output
    described = runner.invoke(app, ["eval", "pipeline", "--describe"])
    assert described.exit_code == 0 and "{in}" in described.output


def test_odke_eval_pipeline_validator_from_the_shell(strict: Path) -> None:
    rows = str(TRIPLES / "triples.jsonl")
    result = runner.invoke(
        app, _shell("--predictions", rows, "--validator", "--config", str(strict), "--json")
    )
    assert result.exit_code == 0, result.output
    report = EvalReport.model_validate_json(result.output)
    assert [row.name for row in report.rows] == ["pipeline", "+ validator"]
    assert report.rows[1].performance.precision.value == 1.0
