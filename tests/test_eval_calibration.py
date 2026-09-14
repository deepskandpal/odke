"""Calibration: a Brier, a reliability curve and an ECE you can check by hand.

tests/fixtures/eval/score.labels.jsonl — eight facts, not a benchmark:

    fact  confidence  true   (c - o)^2   bin
    s1    0.95        yes    0.0025      0.9–1.0
    s2    0.90        yes    0.0100      0.9–1.0
    s3    0.90        no     0.8100      0.9–1.0
    s4    0.70        yes    0.0900      0.7–0.8
    s5    0.60        no     0.3600      0.6–0.7
    s6    0.30        no     0.0900      0.3–0.4
    s7    0.20        yes    0.6400      0.2–0.3
    s8    0.10        no     0.0100      0.1–0.2

Brier = 2.0125 / 8 = 0.2515625. Base rate 4/8, so the constant baseline is 0.25.
Bin 0.9–1.0 holds three facts, mean confidence 2.75/3, two true: gap 0.25.
The single-fact bins have gaps 0.3, 0.6, 0.3, 0.8 and 0.1.
ECE = (3 × 0.25 + 0.3 + 0.6 + 0.3 + 0.8 + 0.1) / 8 = 2.85 / 8 = 0.35625.
Of the three facts scored ≥ 0.9, two were true: 67%.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openodke import Entity, Fact, Scorer
from openodke.eval import CalibrationLabel, load_jsonl
from openodke.eval.calibration import bin_of, evaluate_calibration, run_score

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
LABELS = load_jsonl(FIXTURES / "score.labels.jsonl", CalibrationLabel)


def test_brier_by_hand() -> None:
    report = evaluate_calibration(LABELS)
    assert report.stage == "score" and report.n == 8
    assert report.metrics["brier"] == pytest.approx(0.2515625)
    assert report.metrics["brier_baseline"] == pytest.approx(0.25)
    assert report.metrics["base_rate"] == 0.5
    assert report.metrics["mean_confidence"] == pytest.approx(4.65 / 8)


def test_the_sentence_that_matters_comes_first() -> None:
    report = evaluate_calibration(LABELS)
    assert report.notes[0] == "of facts scored ≥ 0.9, 67% were true (2 of 3)"
    assert report.metrics["scored_at_threshold"] == 3
    assert report.metrics["true_rate_at_threshold"] == pytest.approx(2 / 3)
    assert "of facts scored ≥ 0.9" in report.render()


def test_a_scorer_worse_than_a_constant_is_told_so() -> None:
    report = evaluate_calibration(LABELS)
    assert "Brier 0.252 is no better than always answering the base rate (0.250)" in report.notes


def test_the_reliability_curve_has_ten_bins_including_empty_ones() -> None:
    curve = evaluate_calibration(LABELS).breakdown
    assert list(curve) == [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)]
    top = curve["0.9–1.0"]
    assert top["n"] == 3
    assert top["mean_confidence"] == pytest.approx(2.75 / 3)
    assert top["fraction_true"] == pytest.approx(2 / 3)
    assert top["gap"] == pytest.approx(0.25)
    assert curve["0.2–0.3"]["gap"] == pytest.approx(0.8)
    assert curve["0.4–0.5"] == {"n": 0, "mean_confidence": None, "fraction_true": None, "gap": None}


def test_ece_by_hand() -> None:
    report = evaluate_calibration(LABELS)
    assert report.metrics["ece"] == pytest.approx(0.35625)
    assert report.metrics["max_gap"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.0, 0), (0.1, 1), (0.3, 3), (0.7, 7), (0.9, 9), (0.95, 9), (1.0, 9), (0.29, 2)],
)
def test_bins_are_where_a_reader_expects_despite_float_rounding(
    confidence: float, expected: int
) -> None:
    assert bin_of(confidence) == expected


def test_predictions_override_the_label_rows_confidence() -> None:
    """One labelled slice, scored under a second scorer that says 1.0 to everything."""
    certain = [row.fact.model_copy(update={"confidence": 1.0}) for row in LABELS]
    report = evaluate_calibration(LABELS, certain)
    # (1 - o)^2 is 1 for each of the four false facts: 4 / 8.
    assert report.metrics["brier"] == 0.5
    assert report.notes[0] == "of facts scored ≥ 0.9, 50% were true (4 of 8)"


class _ConstantScorer:
    """Scores everything at the base rate."""

    def score(self, fact: Fact) -> Fact:
        return fact.model_copy(update={"confidence": 0.5})


def test_a_useless_scorer_can_have_zero_ece_which_is_why_brier_is_beside_it() -> None:
    scorer = _ConstantScorer()
    assert isinstance(scorer, Scorer)
    report = evaluate_calibration(LABELS, run_score(scorer, LABELS))
    assert report.metrics["ece"] == 0.0
    assert report.metrics["brier"] == 0.25
    assert report.notes[0] == "no fact was scored ≥ 0.9"


def test_a_confidence_outside_zero_and_one_is_refused() -> None:
    fact = Fact(subject=Entity(key="a", type="T"), predicate="p", confidence=1.5)
    with pytest.raises(ValueError, match=r"confidence 1\.5"):
        evaluate_calibration([CalibrationLabel(fact=fact, true=True)])
