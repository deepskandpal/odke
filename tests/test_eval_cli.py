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
from openodke.eval import StageReport, dump_jsonl
from openodke.eval.runner import evaluate_files, load_stage
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
        evaluate_files("validate", labels, run="openodke.stages:PassThroughValidator")
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({"name": "demo"}))
    report = evaluate_files(
        "validate", labels, run="openodke.stages:PassThroughValidator", ontology=ontology
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
