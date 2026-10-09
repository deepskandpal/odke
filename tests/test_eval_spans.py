"""Span width by verdict: the evaluator that scores citations with no labels at all.

Expected numbers are hand-computed from tests/fixtures/eval/spans.facts.jsonl —
thirteen facts — nine modelled on the narrow-citation failure, plus four edge
cases — and not a benchmark:

    not_found     s1..s7   widths 6, 7, 7, 8, 9, 12, 13
    supported     s8, s9   widths 46, 81
    contradicted  s10      width 30
    unchecked     s11      width 20
    supported     s12      cites a document and no offsets
    unchecked     s13      cites nothing

So, by hand: eleven widths, sorted 6 7 7 8 9 12 13 20 30 46 81, median the sixth,
12. not_found median 8, its quartiles the interpolated 7.0 and 10.5. supported
median (46 + 81) / 2 = 63.5, quartiles 46 + 0.25 * 35 = 54.75 and 46 + 0.75 * 35
= 72.25. 10.5 < 54.75, so the distributions separate. Seven of the eleven facts a
grounder ruled on came back not_found.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openodke import Entity, Evidence, Fact, GroundingVerdict, KnowledgeGraph, Span, SpanOrigin
from openodke.cli.main import app
from openodke.eval import StageReport
from openodke.eval.spans import evaluate_spans, load_facts, span_width
from openodke.sinks import JsonlSink

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "spans.facts.jsonl"
FACTS = load_facts(FIXTURE)
ACME = Entity(key="acme-cloud", type="Company")
runner = CliRunner()


def cited(width: int, verdict: str, *, start: int = 0, doc_id: str = "d1") -> Fact:
    """One fact whose citation is `width` characters wide."""
    span = Span(doc_id=doc_id, start=start, end=start + width)
    return Fact(
        subject=ACME,
        predicate="operates_in",
        object_value=f"{verdict}-{width}",
        evidence=(Evidence(doc_id=doc_id, span=span),),
        verdict=GroundingVerdict(verdict),
    )


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# The distribution
# --------------------------------------------------------------------------- #


def test_the_widths_are_split_by_verdict() -> None:
    report = evaluate_spans(FACTS)
    assert (report.stage, report.n) == ("spans", 13)
    rows = report.breakdown
    assert list(rows) == ["supported", "contradicted", "not_found", "unchecked"]
    assert rows["not_found"] == {
        "n": 7,
        "no_span": 0,
        "min": 6,
        "p25": 7.0,
        "median": 8,
        "p75": 10.5,
        "max": 13,
    }
    assert rows["supported"] == {
        "n": 3,
        "no_span": 1,
        "min": 46,
        "p25": 54.75,
        "median": 63.5,
        "p75": 72.25,
        "max": 81,
    }
    # One fact each: its width is every quartile, and `quantiles` is not asked.
    assert rows["contradicted"] == {
        "n": 1,
        "no_span": 0,
        "min": 30,
        "p25": 30.0,
        "median": 30,
        "p75": 30.0,
        "max": 30,
    }
    assert rows["unchecked"]["n"] == 2 and rows["unchecked"]["median"] == 20


def test_the_headline_numbers_are_the_proxy_and_the_gap() -> None:
    report = evaluate_spans(FACTS)
    m = report.metrics
    assert (m["with_span"], m["no_span"]) == (11, 2)
    assert m["no_span_rate"] == pytest.approx(2 / 13)
    assert m["median_width"] == 12
    # Seven not_found of the eleven facts that got a verdict; unchecked is not one.
    assert m["not_found_rate"] == pytest.approx(7 / 11)
    assert (m["not_found_median"], m["supported_median"]) == (8, 63.5)
    assert m["median_gap"] == 63.5 - 8


def test_a_verdict_nothing_carries_a_width_for_keeps_its_row() -> None:
    """An empty row is `—`, not zero: no width is not a width of nothing."""
    rows = evaluate_spans([cited(40, "supported")]).breakdown
    assert rows["not_found"] == dict.fromkeys(("min", "p25", "median", "p75", "max")) | {
        "n": 0,
        "no_span": 0,
    }
    assert "—" in evaluate_spans([cited(40, "supported")]).render()


# --------------------------------------------------------------------------- #
# The one-line summary
# --------------------------------------------------------------------------- #


def test_the_summary_names_the_gap_when_the_widths_separate() -> None:
    report = evaluate_spans(FACTS)
    assert report.notes[0] == (
        "not_found median 8 chars vs supported 63.5 — citations are too narrow"
    )
    assert report.notes[1] == "2 of 13 fact(s) cite no span of their own and have no width"
    assert "no labels were used" in report.notes[-1]


def test_overlapping_widths_say_nothing_alarming() -> None:
    """Narrow and wide on both sides: the medians differ, the distributions do not."""
    facts = [cited(w, "not_found") for w in (10, 40, 70)]
    facts += [cited(w, "supported") for w in (20, 50, 80)]
    report = evaluate_spans(facts)
    assert report.metrics["median_gap"] == 10
    assert report.notes[0] == (
        "not_found median 40 chars vs supported 50: the widths overlap, "
        "so width does not explain the verdicts"
    )
    assert "too narrow" not in " ".join(report.notes)


def test_one_side_missing_leaves_nothing_to_compare() -> None:
    report = evaluate_spans([cited(8, "not_found"), cited(9, "contradicted")])
    assert report.metrics["median_gap"] is None
    assert report.notes[0].startswith("nothing to compare")
    assert report.metrics["not_found_rate"] == 0.5


def test_an_ungrounded_run_says_to_ground_it_first() -> None:
    report = evaluate_spans([cited(8, "unchecked"), cited(64, "unchecked")])
    assert report.notes[0] == (
        "every fact is unchecked, so there is no verdict to split the widths by"
    )
    assert report.metrics["not_found_rate"] is None
    assert report.metrics["median_width"] == 36


def test_no_facts_at_all() -> None:
    """An empty run is its own finding, and not "every fact is unchecked"."""
    report = evaluate_spans([])
    assert report.n == 0 and report.metrics["median_width"] is None
    assert report.metrics["no_span_rate"] is None
    assert report.notes[0] == "no facts: the run wrote nothing to measure"


# --------------------------------------------------------------------------- #
# What a width is, and where the facts come from
# --------------------------------------------------------------------------- #


def test_the_width_is_the_span_the_grounder_would_read() -> None:
    """The first evidence carrying offsets — a document-level citation has no width."""
    span = Span(doc_id="d1", start=10, end=18)
    fact = Fact(subject=ACME, predicate="operates_in", object_value="Ireland")
    assert span_width(fact) is None
    assert span_width(fact.model_copy(update={"evidence": (Evidence(doc_id="d1"),)})) is None
    citations = (Evidence(doc_id="d1"), Evidence(doc_id="d1", span=span))
    assert span_width(fact.model_copy(update={"evidence": citations})) == 8


def test_a_knowledge_graph_is_read_as_its_facts() -> None:
    assert (
        evaluate_spans(KnowledgeGraph(facts=tuple(FACTS))).metrics == evaluate_spans(FACTS).metrics
    )


def test_the_directory_a_sink_wrote_is_the_input(tmp_path: Path) -> None:
    """No reshaping: point it at the run's output and it finds facts.jsonl."""
    JsonlSink(tmp_path).write(KnowledgeGraph(facts=tuple(FACTS)))
    assert [f.id for f in load_facts(tmp_path)] == [f.id for f in FACTS]
    assert evaluate_spans(load_facts(tmp_path / "facts.jsonl")).n == 13


def test_the_sink_keeps_who_chose_a_span_and_an_older_file_reads_as_cited(tmp_path: Path) -> None:
    """The fixture predates `span_origin`, so every span in it loads as cited (DECISIONS #25)."""
    assert {e.span_origin for f in FACTS for e in f.evidence} == {SpanOrigin.CITED}
    whole = Evidence(
        doc_id="d1", span=Span(doc_id="d1", start=0, end=200), span_origin=SpanOrigin.CONTEXT
    )
    bare = cited(200, "supported").model_copy(update={"evidence": (whole,)})
    JsonlSink(tmp_path).write(KnowledgeGraph(facts=(cited(8, "supported"), bare)))
    assert [f.evidence[0].span_origin for f in load_facts(tmp_path)] == [
        SpanOrigin.CITED,
        SpanOrigin.CONTEXT,
    ]


# --------------------------------------------------------------------------- #
# From the shell
# --------------------------------------------------------------------------- #


def test_the_command_reports_the_fixture() -> None:
    result = runner.invoke(app, ["eval", "spans", "--facts", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("spans  (n=13)")
    assert "citations are too narrow" in _flat(result.output)
    assert "not_found 7 0 6 7.000 8 10.500 13" in _flat(result.output)


def test_json_output_is_a_stage_report() -> None:
    result = runner.invoke(app, ["eval", "spans", "--facts", str(FIXTURE), "--json"])
    assert result.exit_code == 0, result.output
    report = StageReport.model_validate_json(result.output)
    assert report.metrics["not_found_rate"] == pytest.approx(7 / 11)


def test_a_facts_file_passed_as_predictions_is_taken_as_the_facts() -> None:
    """`facts.jsonl` is already the predictions file for the fact stages."""
    result = runner.invoke(app, ["eval", "spans", "--predictions", str(FIXTURE), "--json"])
    assert result.exit_code == 0, result.output
    assert StageReport.model_validate_json(result.output).n == 13


def test_describe_says_it_needs_no_labelled_data() -> None:
    result = runner.invoke(app, ["eval", "spans", "--describe"])
    assert result.exit_code == 0
    assert "Needs no labelled data" in result.output


def test_the_help_says_spans_takes_no_labels() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "no labels at all" in _flat(result.output)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--labels", str(FIXTURE)], "spans needs no labelled data"),
        ([], "spans needs --facts"),
        (["--facts", str(FIXTURE), "--run", "openodke.stages:PassThroughRouter"], "nothing else"),
        (["--facts", "/nonexistent/facts.jsonl"], "facts.jsonl"),
    ],
)
def test_mistakes_exit_2_with_a_message(args: list[str], message: str) -> None:
    result = runner.invoke(app, ["eval", "spans", *args])
    assert result.exit_code == 2
    assert message in _flat(result.output)


def test_facts_is_refused_by_the_labelled_stages(tmp_path: Path) -> None:
    labels = tmp_path / "route.labels.jsonl"
    labels.write_text(json.dumps({"id": "c1", "text": "x", "action": "skip"}) + "\n")
    result = runner.invoke(app, ["eval", "route", "--labels", str(labels), "--facts", str(FIXTURE)])
    assert result.exit_code == 2
    assert "--facts is for spans" in _flat(result.output)
