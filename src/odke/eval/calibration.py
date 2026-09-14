"""Scoring a Scorer: Brier, a ten-bin reliability curve, and ECE.

`Fact.confidence` is the number every downstream threshold is set against,
and it means nothing until it has been checked against outcomes somebody
labelled. Calibration is often implemented as normalisation — squeezing
scores into [0, 1] — which makes them look like probabilities without making
them behave like ones. The test is whether the facts scored 0.9 were true
about 90% of the time, and the report prints exactly that sentence first.

- **Brier** = mean((confidence − outcome)²). Zero is perfect. Always
  answering the slice's base rate p scores p(1 − p), printed beside it as
  `brier_baseline`: a scorer that does not beat it is worse than a constant.
- **Reliability curve**: ten equal-width bins, [0.0, 0.1) up to [0.9, 1.0],
  each with its mean confidence beside the fraction of its facts that were
  true. Empty bins stay in the breakdown with `—`, so the curve has ten rows.
- **ECE**: the gap between those two per bin, weighted by the bin's share of
  the facts. It can be zero for a useless scorer — every fact at the base
  rate — which is why Brier is reported next to it.

The package ships this measurement and never the labels: a calibration number
computed against the pipeline's own output measures nothing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from odke.eval.formats import CalibrationLabel
from odke.eval.report import Metric, StageReport
from odke.stages import Scorer
from odke.types import Fact

BINS = 10


def run_score(scorer: Scorer, labels: Iterable[CalibrationLabel]) -> list[Fact]:
    """Score every labelled fact in-process."""
    return [scorer.score(row.fact) for row in labels]


def bin_of(confidence: float, bins: int = BINS) -> int:
    """Which equal-width bin a confidence falls in; 1.0 belongs to the last.

    The epsilon keeps 0.7, whose float times ten is 7.000000000000001, and
    0.3, whose float times ten is 3.0000000000000004, in the bins a reader
    expects rather than wherever rounding sends them.
    """
    return min(math.floor(confidence * bins + 1e-9), bins - 1)


def evaluate_calibration(
    labels: Sequence[CalibrationLabel],
    predictions: Iterable[Fact] | None = None,
    *,
    threshold: float = 0.9,
) -> StageReport:
    """Brier, the reliability curve and ECE, over `Fact.confidence` against outcomes."""
    pairs, notes = _join(labels, predictions)
    for _, fact in pairs:
        if not 0.0 <= fact.confidence <= 1.0:
            raise ValueError(
                f"fact {fact.id} has confidence {fact.confidence}; calibration needs a "
                "probability in [0, 1]"
            )
    scored = [(fact.confidence, 1.0 if row.true else 0.0) for row, fact in pairs]
    n = len(scored)

    buckets: list[list[tuple[float, float]]] = [[] for _ in range(BINS)]
    for confidence, outcome in scored:
        buckets[bin_of(confidence)].append((confidence, outcome))
    breakdown: dict[str, dict[str, Metric]] = {}
    ece = 0.0
    for index, bucket in enumerate(buckets):
        name = f"{index / BINS:.1f}–{(index + 1) / BINS:.1f}"
        if not bucket:
            breakdown[name] = {"n": 0, "mean_confidence": None, "fraction_true": None, "gap": None}
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        fraction_true = sum(o for _, o in bucket) / len(bucket)
        gap = abs(mean_confidence - fraction_true)
        ece += len(bucket) / n * gap
        breakdown[name] = {
            "n": len(bucket),
            "mean_confidence": mean_confidence,
            "fraction_true": fraction_true,
            "gap": gap,
        }

    brier = sum((c - o) ** 2 for c, o in scored) / n if n else None
    base_rate = sum(o for _, o in scored) / n if n else None
    baseline = base_rate * (1 - base_rate) if base_rate is not None else None
    high = [o for c, o in scored if c >= threshold]
    true_high = sum(high) / len(high) if high else None

    if true_high is None:
        notes.insert(0, f"no fact was scored ≥ {threshold:g}")
    else:
        notes.insert(
            0,
            f"of facts scored ≥ {threshold:g}, {true_high:.0%} were true "
            f"({int(sum(high))} of {len(high)})",
        )
    if brier is not None and baseline is not None and brier >= baseline:
        notes.append(
            f"Brier {brier:.3f} is no better than always answering the base rate ({baseline:.3f})"
        )

    gaps = [g for row in breakdown.values() if (g := row["gap"]) is not None]
    metrics: dict[str, Metric] = {
        "brier": brier,
        "brier_baseline": baseline,
        "ece": ece if n else None,
        "max_gap": max(gaps) if gaps else None,
        "base_rate": base_rate,
        "mean_confidence": sum(c for c, _ in scored) / n if n else None,
        "threshold": threshold,
        "scored_at_threshold": len(high),
        "true_rate_at_threshold": true_high,
    }
    return StageReport(stage="score", n=n, metrics=metrics, breakdown=breakdown, notes=tuple(notes))


def _join(
    labels: Sequence[CalibrationLabel], predictions: Iterable[Fact] | None
) -> tuple[list[tuple[CalibrationLabel, Fact]], list[str]]:
    if predictions is None:
        return [(row, row.fact) for row in labels], []
    by_id = {f.id: f for f in predictions}
    pairs = [(row, by_id[row.fact.id]) for row in labels if row.fact.id in by_id]
    notes = []
    if len(pairs) < len(labels):
        notes.append(
            f"{len(labels) - len(pairs)} labelled fact(s) had no prediction and were not scored"
        )
    known = {row.fact.id for row in labels}
    if stray := sum(1 for key in by_id if key not in known):
        notes.append(f"{stray} prediction(s) matched no labelled fact and were ignored")
    return pairs, notes


__all__ = ["BINS", "bin_of", "evaluate_calibration", "run_score"]
