"""Grounding: verdict accuracy and confusion, and the ablation scorer #38 calls.

Expected numbers are hand-computed from tests/fixtures/eval/ground.*.jsonl — six
facts, not a benchmark:

    g1 supported     -> supported     right
    g2 supported     -> not_found     a true fact lost
    g3 supported     -> supported     right
    g4 contradicted  -> contradicted  right
    g5 contradicted  -> supported     a false fact let through
    g6 not_found     -> not_found     right
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openodke import Document, Fact, Grounder, GroundingVerdict
from openodke.eval import GroundingLabel, load_jsonl
from openodke.eval.grounding import evaluate_grounding, grounding_ablation, kept, run_ground

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
LABELS = load_jsonl(FIXTURES / "ground.labels.jsonl", GroundingLabel)
PREDICTIONS = load_jsonl(FIXTURES / "ground.predictions.jsonl", Fact)


def test_accuracy_per_verdict_and_the_confusion() -> None:
    report = evaluate_grounding(LABELS, PREDICTIONS)
    b = report.breakdown
    assert report.stage == "ground" and report.n == 6
    assert report.metrics["accuracy"] == pytest.approx(4 / 6)
    # supported: tp g1 g3, fp g5, fn g2.
    assert (b["supported"]["precision"], b["supported"]["recall"]) == (
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    )
    # contradicted: tp g4, fn g5. not_found: tp g6, fp g2.
    assert (b["contradicted"]["precision"], b["contradicted"]["recall"]) == (1.0, 0.5)
    assert (b["not_found"]["precision"], b["not_found"]["recall"]) == (0.5, 1.0)
    assert report.metrics["macro_f1"] == pytest.approx(2 / 3)
    assert report.confusion["supported"] == {"supported": 2, "contradicted": 0, "not_found": 1}
    assert report.confusion["contradicted"] == {"supported": 1, "contradicted": 1, "not_found": 0}


def test_the_two_errors_are_counted_apart() -> None:
    """A false fact let through and a true fact thrown away are different costs."""
    report = evaluate_grounding(LABELS, PREDICTIONS)
    assert report.metrics["false_support"] == 1
    assert report.metrics["lost_support"] == 1
    assert report.metrics["unchecked"] == 0


def test_without_predictions_the_label_rows_own_verdicts_are_scored() -> None:
    """The fixture's label facts were never grounded, and unchecked is never right."""
    report = evaluate_grounding(LABELS)
    assert report.metrics["accuracy"] == 0.0
    assert report.metrics["unchecked"] == 6
    assert report.confusion["supported"]["unchecked"] == 3
    assert any("6 fact(s) came back unchecked" in n for n in report.notes)


def test_unjoined_rows_are_dropped_and_said_so() -> None:
    report = evaluate_grounding(LABELS, PREDICTIONS[:5])
    assert report.n == 5
    assert any("1 labelled fact(s) had no prediction" in n for n in report.notes)


class _SubstringGrounder:
    """Supported when the cited span contains the value; otherwise not found.

    Cannot tell a contradiction from an absence, which is exactly what the
    confusion should show.
    """

    def ground(self, fact: Fact, doc: Document) -> Fact:
        span = fact.evidence[0].span
        assert span is not None
        found = str(fact.object_value) in span.resolve(doc)
        verdict = GroundingVerdict.SUPPORTED if found else GroundingVerdict.NOT_FOUND
        return fact.model_copy(update={"verdict": verdict})


def test_run_ground_scores_a_grounder_against_each_rows_own_passage() -> None:
    grounder = _SubstringGrounder()
    assert isinstance(grounder, Grounder)
    grounded = run_ground(grounder, LABELS)
    report = evaluate_grounding(LABELS, grounded)
    # g1 g2 g3 found; g4 (1792) and g5 (1870) not in their spans; g6 has no value.
    assert [f.verdict.value for f in grounded] == [
        "supported",
        "supported",
        "supported",
        "not_found",
        "not_found",
        "not_found",
    ]
    assert report.metrics["accuracy"] == pytest.approx(4 / 6)
    assert report.breakdown["contradicted"]["recall"] == 0.0
    assert report.metrics["false_support"] == 0


def test_the_ablation_reports_precision_with_grounding_off_and_on() -> None:
    """Off keeps all six (three true); on drops g2 g4 g6 and keeps g1 g3 g5 (two true)."""
    report = grounding_ablation(LABELS, PREDICTIONS)
    m = report.metrics
    assert report.stage == "ground:ablation"
    assert (m["kept_off"], m["kept_on"]) == (6, 3)
    assert m["precision_off"] == pytest.approx(0.5)
    assert m["precision_on"] == pytest.approx(2 / 3)
    assert m["precision_gain"] == pytest.approx(1 / 6)
    assert m["recall_on"] == pytest.approx(2 / 3)
    assert report.breakdown["off"]["recall"] == 1.0
    assert (
        report.notes[0]
        == "grounding moved precision from 0.500 to 0.667, keeping 2 of 3 true facts"
    )


def test_the_ablation_gate_is_configurable() -> None:
    """A gate that drops only contradictions keeps g2 and g6 too."""
    report = grounding_ablation(LABELS, PREDICTIONS, drop=(GroundingVerdict.CONTRADICTED,))
    # Kept: g1 g2 g3 g5 g6 — true g1 g2 g3.
    assert report.metrics["kept_on"] == 5
    assert report.metrics["precision_on"] == pytest.approx(3 / 5)
    assert report.metrics["recall_on"] == 1.0


def test_kept_treats_unchecked_as_the_pass_through_does() -> None:
    (fact, *_) = PREDICTIONS
    assert kept(fact)
    assert kept(fact.model_copy(update={"verdict": GroundingVerdict.UNCHECKED}))
    assert not kept(fact.model_copy(update={"verdict": GroundingVerdict.NOT_FOUND}))
