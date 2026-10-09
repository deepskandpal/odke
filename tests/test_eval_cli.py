"""`odke eval <stage>`: a user with labels scores a stage without reading our source."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openodke import Chunk, Document, Entity, Fact, Ontology
from openodke.cli.main import app
from openodke.eval import StageReport, dump_jsonl, read_report
from openodke.eval.runner import evaluate_files, load_stage, report_files
from openodke.stages import PassThroughRouter

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
SUPPORT = "odke_eval_cli_support"
runner = CliRunner()


def _files(stage: str) -> list[str]:
    args = ["eval", stage, "--labels", str(FIXTURES / f"{stage}.labels.jsonl")]
    predictions = FIXTURES / f"{stage}.predictions.jsonl"
    return [*args, "--predictions", str(predictions)] if predictions.exists() else args


def _flat(text: str) -> str:
    return " ".join(text.split())


class _YearExtractor:
    """Finds 'born in YYYY' and nothing else."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        words = chunk.text.rstrip(".").split()
        if "born" in words:
            yield Fact(
                subject=Entity(key=words[0].lower(), type="Person"),
                predicate="born",
                object_value=words[-1],
            )


@pytest.fixture
def support(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """An importable module for `--run`, independent of how pytest imports tests."""
    module = types.ModuleType(SUPPORT)
    module.YearExtractor = _YearExtractor  # type: ignore[attr-defined]
    module.ROUTER = PassThroughRouter()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, SUPPORT, module)
    return module


# --------------------------------------------------------------------------- #
# Scoring files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", ["route", "extract", "ground", "resolve", "score", "validate"])
def test_every_stage_scores_its_fixture_from_the_shell(stage: str) -> None:
    result = runner.invoke(app, _files(stage))
    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"{stage}  (n=")


def test_the_route_report_leads_with_what_matters() -> None:
    result = runner.invoke(app, _files("route"))
    assert "facts_skipped" in result.output
    assert "1 of 3 chunks labelled extract were skipped" in _flat(result.output)
    assert "confusion (rows: labelled, columns: predicted)" in result.output


def test_the_score_report_prints_the_sentence() -> None:
    result = runner.invoke(app, _files("score"))
    assert "of facts scored ≥ 0.9, 67% were true (2 of 3)" in result.output


def test_json_output_is_a_stage_report() -> None:
    result = runner.invoke(app, [*_files("extract"), "--json"])
    assert result.exit_code == 0
    report = StageReport.model_validate_json(result.output)
    assert report.metrics["f1"] == pytest.approx(0.4)


def test_help_says_the_fixtures_are_not_a_benchmark() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "not a benchmark" in _flat(result.output)


def test_describe_prints_the_formats_without_labels() -> None:
    result = runner.invoke(app, ["eval", "resolve", "--describe"])
    assert result.exit_code == 0
    assert "PairLabel" in result.output and "LinkRow" in result.output
    assert "B-cubed" in result.output


def test_ground_and_score_score_the_label_rows_as_they_stand() -> None:
    """The label rows' facts carry a verdict and a confidence, so predictions are optional."""
    report = evaluate_files("ground", FIXTURES / "ground.labels.jsonl")
    assert report.metrics["unchecked"] == 6
    score = evaluate_files("score", FIXTURES / "score.labels.jsonl")
    assert score.metrics["brier"] == pytest.approx(0.2515625)


# --------------------------------------------------------------------------- #
# Mistakes
# --------------------------------------------------------------------------- #


ROUTE_LABELS = str(FIXTURES / "route.labels.jsonl")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["eval", "corroborate", "--labels", ROUTE_LABELS], "unknown stage"),
        (["eval", "route"], "--labels is required"),
        (
            ["eval", "extract", "--labels", str(FIXTURES / "extract.labels.jsonl")],
            "needs --predictions",
        ),
        ([*_files("route"), "--run", "openodke.stages:PassThroughRouter"], "not both"),
        (
            [
                "eval",
                "resolve",
                "--labels",
                ROUTE_LABELS,
                "--run",
                "openodke.stages:PassThroughResolver",
            ],
            "resolve cannot run from labels",
        ),
        (
            ["eval", "route", "--labels", ROUTE_LABELS, "--run", "nowhere.at_all:Nothing"],
            "cannot load",
        ),
        (
            ["eval", "route", "--labels", ROUTE_LABELS, "--run", "openodke.stages:Delegated"],
            "cannot instantiate",
        ),
        (
            ["eval", "route", "--labels", "/nonexistent/labels.jsonl", "--predictions", "x"],
            "labels.jsonl",
        ),
        (
            [
                "eval",
                "route",
                "--labels",
                str(FIXTURES / "score.labels.jsonl"),
                "--predictions",
                "x",
            ],
            "not a RouteLabel row",
        ),
    ],
)
def test_mistakes_exit_2_with_a_message(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert message in _flat(result.output)


# --------------------------------------------------------------------------- #
# Running a stage over the labels
# --------------------------------------------------------------------------- #


def test_run_scores_an_importable_router_over_the_labels() -> None:
    """The pass-through router extracts everything: no fact chunk skipped, three wasted."""
    args = [
        "eval",
        "route",
        "--labels",
        ROUTE_LABELS,
        "--run",
        "openodke.stages:PassThroughRouter",
        "--json",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    report = StageReport.model_validate_json(result.output)
    assert report.metrics["action_accuracy"] == 0.5
    assert report.metrics["facts_skipped"] == 0
    assert report.metrics["wasted_extractions"] == 3


def test_run_validate_needs_an_ontology_and_uses_it(tmp_path: Path) -> None:
    labels = FIXTURES / "validate.labels.jsonl"
    with pytest.raises(ValueError, match="needs --ontology"):
        evaluate_files("validate", labels, run="openodke.stages:PassThroughGate")
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({"name": "demo"}))
    report = evaluate_files(
        "validate", labels, run="openodke.stages:PassThroughGate", ontology=ontology
    )
    # Accepting everything agrees on the three accept rows only.
    assert report.metrics["agreement"] == 0.5


def test_run_extract_reads_documents_and_an_ontology(
    tmp_path: Path, support: types.ModuleType
) -> None:
    gold = FIXTURES / "extract.labels.jsonl"
    with pytest.raises(ValueError, match="needs --documents and --ontology"):
        evaluate_files("extract", gold, run=f"{SUPPORT}:YearExtractor")
    documents = tmp_path / "documents.jsonl"
    dump_jsonl(
        documents,
        [
            Document(id="d1", text="Ada was born in 1815."),
            Document(id="d2", text="Babbage was born in 1791."),
        ],
    )
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({"name": "demo"}))
    args = [
        "eval", "extract", "--labels", str(gold), "--run", f"{SUPPORT}:YearExtractor",
        "--documents", str(documents), "--ontology", str(ontology), "--json",
    ]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    report = StageReport.model_validate_json(result.output)
    # Both birth years; the name and both employers are missing.
    assert (report.metrics["tp"], report.metrics["missing"], report.metrics["spurious"]) == (
        2,
        3,
        0,
    )


def test_load_stage_instantiates_a_class_and_takes_an_instance_as_is(
    support: types.ModuleType,
) -> None:
    assert isinstance(load_stage("openodke.stages:PassThroughRouter"), PassThroughRouter)
    assert load_stage(f"{SUPPORT}:ROUTER") is support.ROUTER
    with pytest.raises(ValueError, match="package.module:Name"):
        load_stage("openodke.stages")


# --------------------------------------------------------------------------- #
# The eval report (#139)
# --------------------------------------------------------------------------- #


def test_report_files_scores_extract_as_evaluate_files_does_with_ranges() -> None:
    labels, predictions = FIXTURES / "extract.labels.jsonl", FIXTURES / "extract.predictions.jsonl"
    report = report_files("extract", labels, predictions)
    assert report.stages == (evaluate_files("extract", labels, predictions),)
    (row,) = report.rows
    assert row.performance.f1.value == pytest.approx(0.4)
    assert report.run.dataset is not None
    assert (report.run.dataset.documents, report.run.dataset.labels) == (2, 5)


@pytest.mark.parametrize("stage", ["route", "ground", "resolve", "score", "validate"])
def test_report_files_carries_any_other_stage_with_no_rows(stage: str) -> None:
    labels = FIXTURES / f"{stage}.labels.jsonl"
    predictions = FIXTURES / f"{stage}.predictions.jsonl"
    found = predictions if predictions.exists() else None
    report = report_files(stage, labels, found)
    assert report.stages == (evaluate_files(stage, labels, found),)
    assert report.rows == () and report.bootstrap is None


def test_extract_takes_an_ontology_for_conformance(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({"name": "demo", "predicates": {"born": {"range": "integer"}}}))
    report = report_files(
        "extract",
        FIXTURES / "extract.labels.jsonl",
        FIXTURES / "extract.predictions.jsonl",
        ontology=ontology,
    )
    found = report.rows[0].conformance
    # Only the two `born` facts have a predicate this ontology declares.
    assert found is not None and (found.conformant, found.facts) == (2, 5)


def test_odke_eval_writes_the_report_and_json_stays_a_stage_report(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    args = _files("extract")
    result = runner.invoke(app, [*args, "--report", str(out)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("extract  (n=5)\n")
    assert "0.400 [" in result.output and "95% ranges: 2 documents" in result.output
    assert result.output.rstrip().endswith(f"wrote {out}")
    assert read_report(out).rows[0].performance.f1.value == pytest.approx(0.4)

    again = tmp_path / "again.json"
    as_json = runner.invoke(app, [*args, "--json", "--report", str(again)])
    assert as_json.exit_code == 0, as_json.output
    assert StageReport.model_validate_json(as_json.output).metrics["f1"] == pytest.approx(0.4)
    assert read_report(again) == read_report(out)


@pytest.mark.parametrize("stage", ["route", "ground", "resolve", "score", "validate"])
def test_every_stage_writes_a_report(tmp_path: Path, stage: str) -> None:
    out = tmp_path / f"{stage}.json"
    result = runner.invoke(app, [*_files(stage), "--report", str(out)])
    assert result.exit_code == 0, result.output
    report = read_report(out)
    assert report.stages[0].stage == stage and report.rows == ()


def test_odke_eval_spans_and_ablation_write_a_report(tmp_path: Path, example: Path) -> None:
    out = tmp_path / "spans.json"
    facts = FIXTURES / "spans.facts.jsonl"
    result = runner.invoke(app, ["eval", "spans", "--facts", str(facts), "--report", str(out)])
    assert result.exit_code == 0, result.output
    assert read_report(out).stages[0].stage == "spans"

    out = tmp_path / "ablation.json"
    config, gold = str(example / "odke.yaml"), str(example / "gold.jsonl")
    args = ["eval", "ablation", "--config", config, "--labels", gold, "--report", str(out)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(read_report(out).rows) == 3


def test_odke_eval_compare_writes_its_block_into_a_report(tmp_path: Path) -> None:
    for name, extra in (("a", 0), ("b", 1)):
        rows = [{"id": f"d{i}", "tp": 3 + (extra and i % 2), "fp": 1, "fn": 1} for i in range(10)]
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "report.json"
    args = ["eval", "compare", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
    result = runner.invoke(app, [*args, "--report", str(out)])
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith(f"wrote {out}")
    report = read_report(out)
    assert report.comparison is not None and report.comparison.metric == "f1"
    assert (report.title, report.n, report.rows) == ("compare", 10, ())
