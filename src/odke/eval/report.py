"""One result type for every evaluator, and the arithmetic they share.

Every evaluator in `odke.eval` returns a `StageReport`: the stage it scored,
how many rows it saw, a flat dict of headline numbers, a per-key breakdown
(per label, per predicate, per verdict, per confidence bin — whatever the
stage partitions on), a labelled-by-predicted confusion where the stage has
classes, and notes for what a number cannot say. One type, so `odke eval`
prints any of them the same way and eight of them fit in one table.

Precision, recall and F1 live here rather than in each evaluator because the
edge cases are where evaluators quietly disagree: a class nothing was
predicted for has an *undefined* precision, not a precision of zero, and the
report keeps that as `None`. Zero would read as "every prediction was
wrong", which is a different and worse finding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias

from pydantic import Field

from odke.types import Frozen

# A headline number: `int` for counts, `float` for rates, `None` for
# undefined — never 0.0 standing in for "could not be computed".
Metric: TypeAlias = int | float | None


class StageReport(Frozen):
    """What one evaluator concluded about one stage, against the caller's labels."""

    stage: str
    # Rows scored, after any that could not be joined were dropped — and
    # `notes` says how many were.
    n: int
    metrics: dict[str, Metric] = Field(default_factory=dict)
    # Keyed by what the stage partitions on: a label, a predicate, a verdict,
    # a confidence bin.
    breakdown: dict[str, dict[str, Metric]] = Field(default_factory=dict)
    # Labelled class -> predicted class -> count. Empty for stages without classes.
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        """The plain-text form `odke eval` prints: three decimals, `—` for undefined."""
        lines = [f"{self.stage}  (n={self.n})"]
        width = max((len(k) for k in self.metrics), default=0)
        lines.extend(f"  {k:<{width}}  {_fmt(v)}" for k, v in self.metrics.items())
        if self.breakdown:
            columns = list(dict.fromkeys(c for row in self.breakdown.values() for c in row))
            body = [
                [key, *(_fmt(row.get(c)) for c in columns)] for key, row in self.breakdown.items()
            ]
            lines += ["", *_table(["", *columns], body)]
        if self.confusion:
            predicted = list(dict.fromkeys(p for row in self.confusion.values() for p in row))
            body = [
                [g, *(str(row.get(p, 0)) for p in predicted)] for g, row in self.confusion.items()
            ]
            lines += ["", "  confusion (rows: labelled, columns: predicted)"]
            lines += _table(["", *predicted], body)
        if self.notes:
            lines += ["", *(f"  - {note}" for note in self.notes)]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def prf(tp: int, fp: int, fn: int) -> dict[str, Metric]:
    """Precision, recall and F1 from counts, keeping "undefined" distinct from zero.

    Precision is undefined when nothing was predicted for the class, recall
    when nothing was labelled with it. F1 is `None` only when both are, and
    0.0 when the class was wholly missed or wholly spurious — a real score
    for a real failure.
    """
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1: Metric
    if precision is None and recall is None:
        f1 = None
    elif not precision or not recall:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def per_class(
    pairs: Iterable[tuple[str, str]], classes: Iterable[str] = ()
) -> tuple[dict[str, dict[str, Metric]], dict[str, dict[str, int]]]:
    """Per-class P/R/F1 and a square confusion matrix from (labelled, predicted) pairs.

    Classes are the ones named, then any others seen on either side, so a
    class that was never predicted still gets a row — and its recall of zero
    is the row worth reading.
    """
    order: dict[str, None] = dict.fromkeys(classes)
    counts: dict[tuple[str, str], int] = {}
    for gold, predicted in pairs:
        order.setdefault(gold)
        order.setdefault(predicted)
        counts[gold, predicted] = counts.get((gold, predicted), 0) + 1
    confusion = {g: {p: counts.get((g, p), 0) for p in order} for g in order}
    breakdown: dict[str, dict[str, Metric]] = {}
    for c in order:
        tp = confusion[c][c]
        fn = sum(confusion[c].values()) - tp
        fp = sum(confusion[g][c] for g in order) - tp
        breakdown[c] = {**prf(tp, fp, fn), "support": tp + fn}
    return breakdown, confusion


def accuracy(confusion: Mapping[str, Mapping[str, int]]) -> Metric:
    """The diagonal over the total. `None` on an empty matrix."""
    total = sum(sum(row.values()) for row in confusion.values())
    return sum(row.get(c, 0) for c, row in confusion.items()) / total if total else None


def cohen_kappa(confusion: Mapping[str, Mapping[str, int]]) -> Metric:
    """Agreement beyond what the two sides' class frequencies give by chance.

    Accuracy alone flatters a validator that accepts everything on a slice
    that is mostly acceptable; kappa is zero for it.
    """
    total = sum(sum(row.values()) for row in confusion.values())
    if not total:
        return None
    classes = list(confusion)
    observed = sum(confusion[c].get(c, 0) for c in classes) / total
    expected = sum(
        (sum(confusion[c].values()) / total)
        * (sum(confusion[g].get(c, 0) for g in classes) / total)
        for c in classes
    )
    return (observed - expected) / (1 - expected) if expected < 1 else None


def macro_f1(breakdown: Mapping[str, Mapping[str, Metric]]) -> Metric:
    """Mean F1 over the classes that have one. `None` when none does."""
    scores = [s for row in breakdown.values() if (s := row.get("f1")) is not None]
    return sum(scores) / len(scores) if scores else None


def ratio(numerator: float, denominator: float) -> Metric:
    return numerator / denominator if denominator else None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(value: Metric) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    everything = [headers, *rows]
    widths = [max(len(r[i]) for r in everything) for i in range(len(headers))]
    return [
        ("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True))).rstrip()
        for r in everything
    ]


__all__ = [
    "Metric",
    "StageReport",
    "accuracy",
    "cohen_kappa",
    "macro_f1",
    "per_class",
    "prf",
    "ratio",
]
