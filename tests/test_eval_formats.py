"""The BYOLD contract: every stage has a row format, and the arithmetic keeps "undefined" honest."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import odke.eval
from odke import EntityLink, LinkKind
from odke.eval import (
    LABEL_FORMATS,
    PREDICTION_FORMATS,
    GroundingLabel,
    LinkRow,
    RouteLabel,
    StageReport,
    describe,
    dump_jsonl,
    load_jsonl,
)
from odke.eval.formats import cite
from odke.eval.report import accuracy, cohen_kappa, macro_f1, per_class, prf

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
STAGES = ("route", "extract", "ground", "resolve", "score", "validate")


# --------------------------------------------------------------------------- #
# Formats and fixtures
# --------------------------------------------------------------------------- #


def test_every_stage_has_a_label_format_and_a_prediction_format() -> None:
    assert tuple(LABEL_FORMATS) == STAGES
    assert tuple(PREDICTION_FORMATS) == STAGES


@pytest.mark.parametrize("stage", STAGES)
def test_every_fixture_loads_and_is_too_small_to_be_a_benchmark(stage: str) -> None:
    labels = load_jsonl(FIXTURES / f"{stage}.labels.jsonl", LABEL_FORMATS[stage])
    assert 0 < len(labels) <= 10
    predictions = FIXTURES / f"{stage}.predictions.jsonl"
    if predictions.exists():
        assert load_jsonl(predictions, PREDICTION_FORMATS[stage])


def test_the_package_says_the_fixtures_are_not_a_benchmark() -> None:
    assert odke.eval.__doc__ is not None
    assert "not a benchmark" in odke.eval.__doc__


@pytest.mark.parametrize("stage", STAGES)
def test_describe_prints_what_a_row_is_without_reading_source(stage: str) -> None:
    text = describe(stage)
    assert "--labels" in text and "--predictions" in text
    assert LABEL_FORMATS[stage].__name__ in text


def test_describe_rejects_an_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        describe("corroborate")


def test_a_bad_row_is_reported_with_its_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "c1", "text": "x", "action": "extract"}\n\n{"id": "c2"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:3: not a RouteLabel row"):
        load_jsonl(path, RouteLabel)


def test_rows_round_trip_through_dump(tmp_path) -> None:
    rows = load_jsonl(FIXTURES / "route.labels.jsonl", RouteLabel)
    dump_jsonl(tmp_path / "out.jsonl", rows)
    assert load_jsonl(tmp_path / "out.jsonl", RouteLabel) == rows


def test_a_link_row_reads_a_serialised_entity_link_as_a_triple() -> None:
    """A sink's links.jsonl is already a predictions file."""
    link = EntityLink(source_key="a", target_key="b", kind=LinkKind.SAME_AS, score=0.9)
    assert LinkRow.model_validate_json(link.model_dump_json()) == LinkRow(
        a="a", b="b", kind=LinkKind.SAME_AS
    )
    with pytest.raises(ValidationError):
        LinkRow(a="a", b="b", kind="maybe")  # type: ignore[arg-type]


def test_a_route_label_becomes_the_chunk_a_router_sees() -> None:
    chunk = RouteLabel(id="c1", text="Ada.", action="extract").as_chunk()
    assert (chunk.doc_id, chunk.text, chunk.start, chunk.end) == ("c1", "Ada.", 0, 4)


def test_a_grounding_label_names_the_document_its_span_indexes() -> None:
    (row, *_) = load_jsonl(FIXTURES / "ground.labels.jsonl", GroundingLabel)
    doc = row.as_document()
    assert doc.id == "d1"
    span = row.fact.evidence[0].span
    assert span is not None and span.resolve(doc) == row.text[span.start : span.end]


def test_cite_stamps_evidence_only_on_a_fact_that_has_none() -> None:
    (row, *_) = load_jsonl(FIXTURES / "ground.labels.jsonl", GroundingLabel)
    assert cite(row.fact, "elsewhere") is row.fact
    bare = row.fact.model_copy(update={"evidence": ()})
    assert cite(bare, "d9").evidence[0].doc_id == "d9"


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def test_prf_keeps_undefined_apart_from_zero() -> None:
    """Nothing predicted is an undefined precision, not a precision of zero."""
    assert prf(0, 0, 0) == {
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    missed = prf(0, 0, 2)
    assert (missed["precision"], missed["recall"], missed["f1"]) == (None, 0.0, 0.0)
    spurious = prf(0, 3, 0)
    assert (spurious["precision"], spurious["recall"], spurious["f1"]) == (0.0, None, 0.0)
    # tp=2 fp=1 fn=1: P = R = 2/3, F1 = 2/3.
    assert prf(2, 1, 1)["f1"] == pytest.approx(2 / 3)


def test_per_class_gives_a_never_predicted_class_its_row() -> None:
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("c", "b")]
    breakdown, confusion = per_class(pairs, classes=("a", "b", "c", "d"))
    # a: tp1 fn1 fp0. b: tp1 fp2 fn0. c: fn1. d: nothing either side.
    assert breakdown["a"]["precision"] == 1.0 and breakdown["a"]["recall"] == 0.5
    assert breakdown["b"]["precision"] == pytest.approx(1 / 3) and breakdown["b"]["recall"] == 1.0
    assert (breakdown["c"]["precision"], breakdown["c"]["recall"], breakdown["c"]["f1"]) == (
        None,
        0.0,
        0.0,
    )
    assert breakdown["d"]["f1"] is None and breakdown["d"]["support"] == 0
    assert confusion["c"] == {"a": 0, "b": 1, "c": 0, "d": 0}
    assert accuracy(confusion) == 0.5
    # Mean over a, b, c — d has no F1 to average.
    assert macro_f1(breakdown) == pytest.approx((2 / 3 + 0.5 + 0.0) / 3)


def test_kappa_is_zero_for_a_validator_that_always_says_the_majority() -> None:
    confusion = {"accept": {"accept": 9, "refuse": 0}, "refuse": {"accept": 1, "refuse": 0}}
    assert accuracy(confusion) == 0.9
    assert cohen_kappa(confusion) == 0.0


def test_a_report_is_frozen_serialisable_and_renders_undefined_as_a_dash() -> None:
    report = StageReport(
        stage="demo",
        n=3,
        metrics={"precision": None, "recall": 0.5, "tp": 1},
        breakdown={"x": {"f1": 0.25}},
        confusion={"x": {"x": 1, "y": 2}},
        notes=("fixture only",),
    )
    with pytest.raises(ValidationError):
        report.n = 4  # type: ignore[misc]
    assert StageReport.model_validate_json(report.model_dump_json()) == report
    text = report.render()
    assert "demo  (n=3)" in text
    assert "precision  —" in text
    assert "0.500" in text and "0.250" in text
    assert "- fixture only" in text
    assert isinstance(StageReport.model_validate_json(report.model_dump_json()).metrics["tp"], int)
