"""Routing: per-label P/R/F1, and a skipped fact chunk is named, not averaged away.

Expected numbers are hand-computed from tests/fixtures/eval/route.*.jsonl — six
chunks, not a benchmark:

    c1 extract/fact      -> extract/fact       right
    c2 extract/fact      -> skip/marketing     a fact chunk skipped
    c3 skip/marketing    -> skip/marketing     right
    c4 skip/marketing    -> extract/fact       wasted extraction
    c5 extract/fact      -> extract/fact       right
    c6 skip/policy       -> skip/policy        right
"""

from __future__ import annotations

from pathlib import Path

import pytest

from odke import Chunk, Router, RouteVerdict
from odke.eval import RouteLabel, RoutePrediction, load_jsonl
from odke.eval.routing import evaluate_routing, run_route

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
LABELS = load_jsonl(FIXTURES / "route.labels.jsonl", RouteLabel)
PREDICTIONS = load_jsonl(FIXTURES / "route.predictions.jsonl", RoutePrediction)


def test_action_metrics_and_the_skip_extract_confusion() -> None:
    report = evaluate_routing(LABELS, PREDICTIONS)
    m = report.metrics
    assert report.stage == "route" and report.n == 6
    assert m["action_accuracy"] == pytest.approx(4 / 6)
    # extract: tp c1 c5, fp c4, fn c2. skip: tp c3 c6, fp c2, fn c4.
    assert m["extract_precision"] == pytest.approx(2 / 3)
    assert m["extract_recall"] == pytest.approx(2 / 3)
    assert m["extract_f1"] == pytest.approx(2 / 3)
    assert m["skip_precision"] == pytest.approx(2 / 3)
    assert m["skip_recall"] == pytest.approx(2 / 3)
    assert report.confusion["extract"] == {"extract": 2, "skip": 1, "defer": 0}
    assert report.confusion["skip"] == {"extract": 1, "skip": 2, "defer": 0}


def test_a_skipped_fact_chunk_is_its_own_line() -> None:
    """The expensive failure: nothing downstream can recover a chunk that was never extracted."""
    report = evaluate_routing(LABELS, PREDICTIONS)
    assert report.metrics["facts_skipped"] == 1
    assert report.metrics["facts_skipped_rate"] == pytest.approx(1 / 3)
    assert report.metrics["facts_deferred"] == 0
    assert report.metrics["wasted_extractions"] == 1
    assert report.notes[0].startswith("1 of 3 chunks labelled extract were skipped")


def test_per_label_precision_recall_f1() -> None:
    report = evaluate_routing(LABELS, PREDICTIONS)
    fact, marketing, policy = (report.breakdown[k] for k in ("fact", "marketing", "policy"))
    # fact: tp c1 c5, fp c4, fn c2. marketing: tp c3, fp c2, fn c4. policy: tp c6.
    assert (fact["precision"], fact["recall"]) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert (marketing["precision"], marketing["recall"], marketing["f1"]) == (0.5, 0.5, 0.5)
    assert (policy["precision"], policy["recall"], policy["f1"]) == (1.0, 1.0, 1.0)
    assert report.metrics["label_accuracy"] == pytest.approx(4 / 6)
    assert report.metrics["label_macro_f1"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
    assert "fact" in report.render()


def test_unjoined_rows_are_dropped_and_said_so() -> None:
    predictions = [p for p in PREDICTIONS if p.id != "c6"]
    predictions.append(RoutePrediction(id="stray", action="skip"))
    report = evaluate_routing(LABELS, predictions)
    assert report.n == 5
    assert any("1 labelled chunk(s) had no prediction" in n for n in report.notes)
    assert any("1 prediction(s) matched no labelled chunk" in n for n in report.notes)


def test_rows_without_a_label_score_actions_only() -> None:
    labels = [RouteLabel(id="x", text="Ada.", action="extract")]
    report = evaluate_routing(labels, [RoutePrediction(id="x", action="extract", label="fact")])
    assert report.metrics["action_accuracy"] == 1.0
    assert report.breakdown == {}
    assert report.metrics["label_accuracy"] is None
    assert any("only the action metrics" in n for n in report.notes)


class _ExclamationRouter:
    """Skips what sounds like an advert. Wrong about the policy chunk, on purpose."""

    def route(self, chunk: Chunk) -> RouteVerdict:
        if "!" in chunk.text or "offer" in chunk.text:
            return RouteVerdict(action="skip", label="marketing", scope="document")
        return RouteVerdict(action="extract", label="fact")


def test_run_route_scores_a_router_in_process() -> None:
    router = _ExclamationRouter()
    assert isinstance(router, Router)
    predictions = run_route(router, LABELS)
    assert [p.action for p in predictions] == [
        "extract",
        "extract",
        "skip",
        "skip",
        "extract",
        "extract",
    ]
    report = evaluate_routing(LABELS, predictions)
    # Only c6 is wrong: policy sent to extraction. No fact chunk was skipped.
    assert report.metrics["action_accuracy"] == pytest.approx(5 / 6)
    assert report.metrics["facts_skipped"] == 0
    assert report.metrics["wasted_extractions"] == 1
    assert report.breakdown["policy"]["recall"] == 0.0
