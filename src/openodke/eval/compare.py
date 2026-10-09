"""`odke eval compare`: two runs on the same items, and whether the change was real.

A comparison reads each run's per-item outcomes, which `odke eval <stage>
--items` writes beside the report: one row per labelled document for
extraction, carrying its counts, and one row per labelled fact or chunk, right
or wrong, for grounding, validation and routing. Both runs must cover the same
items; a comparison over different items measures the difference between the
sets, so it is refused, naming the items only one side has.

One metric is primary and decides the verdict: F1 where the rows carry counts,
accuracy where they carry right or wrong. The others are guardrails, printed
with their own intervals and never deciding, so that three metrics are not
three chances of a false alarm on every run.

`Comparison` is one JSON block, `--json`, so a versioned eval report (#139) can
carry it as its comparison section unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from openodke.eval.extraction import document_counts
from openodke.eval.formats import load_jsonl
from openodke.eval.report import _table, join_by_id
from openodke.eval.stats import (
    RESAMPLES,
    Item,
    McNemar,
    MetricFn,
    Paired,
    f1,
    mcnemar,
    paired_bootstraps,
    precision,
    recall,
    share,
)
from openodke.types import Frozen

COUNTS = ("tp", "fp", "fn")
# Each metric, the fields of an item row it reads, and how it is computed from their totals.
METRICS: dict[str, tuple[tuple[str, ...], MetricFn]] = {
    "f1": (COUNTS, f1),
    "precision": (COUNTS, precision),
    "recall": (COUNTS, recall),
    "accuracy": (("correct",), share),
}
ITEM_STAGES = ("extract", "ground", "validate", "route")

DESCRIPTION = """\
odke eval compare RUN_A_ITEMS RUN_B_ITEMS

Two runs on the same items: better, worse or inconclusive, with the detection
limit printed. Each file is one run's per-item outcomes, as JSONL, written by
`odke eval <stage> --items PATH` for extract, ground, validate or route:

  {"id": "d1", "tp": 3, "fp": 1, "fn": 0}   a document: its counts against the gold facts
  {"id": "g7", "correct": true}             a labelled fact or chunk, right or wrong

Both files must hold the same ids. Items are resampled with replacement and the
metric recomputed for both runs on each resample (a paired bootstrap; for
corpus precision, recall and F1 the item is the document). The primary metric
(F1 for counts, accuracy otherwise; --metric to choose) decides the verdict:
worse or better when its whole 95% interval is one side of zero, inconclusive
otherwise. The other metrics are guardrails: reported, never deciding.

Exit 1 when the verdict is worse. --fail-under X also exits 1 when the low end
of B's 95% range for the primary metric is under X; --fail-on-inconclusive
also exits 1 on inconclusive, which otherwise passes. Exit 2 is files that
cannot be compared."""


class ItemRow(Frozen):
    """One item's outcome in one run: what `odke eval compare` reads.

    A document carries its counts against the gold facts, so corpus precision,
    recall and F1 are recomputed from summed counts on every resample. A
    labelled fact or chunk carries `correct`.
    """

    id: str
    tp: int | None = Field(default=None, ge=0)
    fp: int | None = Field(default=None, ge=0)
    fn: int | None = Field(default=None, ge=0)
    correct: bool | None = None

    @model_validator(mode="after")
    def _an_outcome(self) -> ItemRow:
        counts = [getattr(self, name) for name in COUNTS]
        if any(c is not None for c in counts) and any(c is None for c in counts):
            raise ValueError("a document row carries all three of tp, fp and fn")
        if counts[0] is None and self.correct is None:
            raise ValueError("an item row carries tp, fp and fn, or correct")
        return self


class Comparison(Frozen):
    """Run A against run B on the same items, by one primary metric and its guardrails."""

    a: str
    b: str
    unit: Literal["document", "item"]
    metric: str
    primary: Paired
    # The other metrics the rows carry: reported with their intervals, never deciding.
    guardrails: dict[str, Paired] = Field(default_factory=dict)
    # The exact test on the flips, for a right-or-wrong primary metric.
    mcnemar: McNemar | None = None
    notes: tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        return self.primary.verdict

    def render(self) -> str:
        """The plain text `odke eval compare` prints."""
        p, unit = self.primary, f"{self.unit}s"
        flips = f"{p.flips} of {p.items} {unit} changed ({p.flip_share:.1%})"
        if self.mcnemar:
            flips += (
                f": {self.mcnemar.gained} gained, {self.mcnemar.lost} lost, "
                f"McNemar exact p = {self.mcnemar.p_value:.3g}"
            )
        rows = [
            ("verdict", _sentence(p, unit)),
            (
                self.metric,
                f"A {p.a:.3f}  B {p.b:.3f}  difference {_points(p.difference)} points, "
                f"{p.level:.0%} interval {_points(p.interval[0])} to {_points(p.interval[1])}",
            ),
            ("limit", _limit(p, unit)),
            ("flips", flips),
            ("B range", f"{p.b_interval[0]:.3f} to {p.b_interval[1]:.3f} ({p.level:.0%})"),
        ]
        width = max(len(label) for label, _ in rows)
        lines = [f"compare  {self.metric} over {p.items} {unit}  (A: {self.a}, B: {self.b})"]
        lines += [f"  {label:<{width}}  {text}" for label, text in rows]
        if self.guardrails:
            body = [
                [
                    name,
                    f"{g.a:.3f}",
                    f"{g.b:.3f}",
                    _points(g.difference),
                    f"{_points(g.interval[0])} to {_points(g.interval[1])}",
                    g.verdict,
                ]
                for name, g in self.guardrails.items()
            ]
            lines += ["", "  guardrails: reported, not deciding the verdict"]
            lines += _table(["", "A", "B", "points", "interval", "verdict"], body)
        if self.notes:
            lines += ["", *(f"  - {note}" for note in self.notes)]
        return "\n".join(lines)


def compare_items(
    a: Sequence[ItemRow],
    b: Sequence[ItemRow],
    *,
    metric: str | None = None,
    resamples: int = RESAMPLES,
    seed: int = 0,
    a_name: str = "A",
    b_name: str = "B",
) -> Comparison:
    """Compare two runs' item rows: the primary metric's verdict and the guardrails.

    Raises `ValueError` when the runs cover different items, when an id is
    listed twice, or when the metric needs fields the rows do not carry.
    """
    a_by, b_by = _index(a, a_name), _index(b, b_name)
    _same_items(a_by, b_by, a_name, b_name)
    rows_a = list(a_by.values())
    rows_b = [b_by[key] for key in a_by]
    carried = [
        name
        for name, (fields, _) in METRICS.items()
        if _carries(rows_a, fields) and _carries(rows_b, fields)
    ]
    if metric is not None and metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; one of: {', '.join(METRICS)}")
    primary = metric or next(iter(carried), "f1")
    if primary not in carried:
        needed = ", ".join(METRICS[primary][0])
        raise ValueError(f"{primary} needs {needed} on every row of both runs")

    notes: list[str] = []
    # Metrics that read the same fields share one set of resamples.
    by_fields: dict[tuple[str, ...], dict[str, MetricFn]] = {}
    for name in [primary, *(m for m in carried if m != primary)]:
        fields, compute = METRICS[name]
        if name != primary and not _defined(compute, rows_a, rows_b, fields):
            notes.append(f"guardrail {name} not compared: it is undefined on a run as a whole")
            continue
        by_fields.setdefault(fields, {})[name] = compute
    results: dict[str, Paired] = {}
    for fields, metrics in by_fields.items():
        results |= paired_bootstraps(
            [_value(row, fields) for row in rows_a],
            [_value(row, fields) for row in rows_b],
            metrics,
            n=resamples,
            seed=seed,
        )
    main = results.pop(primary)
    results = {name: results[name] for name in carried if name in results}

    exact = None
    if METRICS[primary][0] == ("correct",):
        exact = mcnemar([bool(r.correct) for r in rows_a], [bool(r.correct) for r in rows_b])
        if (exact.p_value < 1 - main.level) != (main.verdict != "inconclusive"):
            notes.append(
                f"McNemar's exact test and the bootstrap disagree (p = {exact.p_value:.3g}): "
                "the change is at the edge of what these items can show"
            )
    return Comparison(
        a=a_name,
        b=b_name,
        unit="document" if METRICS[primary][0] == COUNTS else "item",
        metric=primary,
        primary=main,
        guardrails=results,
        mcnemar=exact,
        notes=tuple(notes),
    )


def compare_files(
    a: str | Path,
    b: str | Path,
    *,
    metric: str | None = None,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> Comparison:
    """`compare_items` over two `--items` files, each run named by its path."""
    return compare_items(
        load_jsonl(a, ItemRow),
        load_jsonl(b, ItemRow),
        metric=metric,
        resamples=resamples,
        seed=seed,
        a_name=str(a),
        b_name=str(b),
    )


def gate(
    comparison: Comparison,
    *,
    fail_under: float | None = None,
    fail_on_inconclusive: bool = False,
) -> list[str]:
    """Why a CI step should fail on this comparison; empty when it passes.

    A *worse* verdict always fails. `fail_under` also fails a B whose own 95%
    range for the primary metric reaches below the threshold: B has to clear
    the floor with its uncertainty, not with a lucky point estimate. An
    *inconclusive* verdict passes unless `fail_on_inconclusive`: a change too
    small for this set to see is not a regression it saw, and the detection
    limit printed beside it says how small that is.
    """
    p, name, reasons = comparison.primary, comparison.metric, []
    if p.verdict == "worse":
        reasons.append(
            f"{name} is worse: {_points(p.difference)} points, {p.level:.0%} interval "
            f"{_points(p.interval[0])} to {_points(p.interval[1])}"
        )
    if fail_under is not None and p.b_interval[0] < fail_under:
        reasons.append(
            f"B's {name} could be as low as {p.b_interval[0]:.3f} ({p.level:.0%} range), "
            f"under --fail-under {fail_under:g}"
        )
    if fail_on_inconclusive and p.verdict == "inconclusive":
        reasons.append(f"--fail-on-inconclusive: {_sentence(p, f'{comparison.unit}s')}")
    return reasons


# --------------------------------------------------------------------------- #
# Writing a run's items
# --------------------------------------------------------------------------- #


def item_rows(stage: str, labels: Sequence[Any], predicted: Any) -> tuple[list[ItemRow], list[str]]:
    """Each labelled item's outcome under one run, and what could not be put in a row.

    Extraction is per document, over every document the gold facts name, so two
    runs scored against the same labels cover the same documents. A wrong value
    or wrong entity counts as a false positive and a false negative, as in
    `evaluate_extraction`, and the rows sum to its totals. Grounding,
    validation and routing are one row per labelled item that has a prediction.
    """
    if stage == "extract":
        # The same counts an eval report resamples for its ranges.
        found = document_counts(labels, predicted)
        rows = [ItemRow(id=doc, tp=tp, fp=fp, fn=fn) for doc, (tp, fp, fn) in found.by_doc.items()]
        warnings = (
            [
                f"{found.uncited} spurious prediction(s) cite no document and are in no "
                "document's row"
            ]
            if found.uncited
            else []
        )
        return rows, warnings
    if stage == "ground":
        pairs = (
            [(row, row.fact) for row in labels]
            if predicted is None
            else join_by_id(labels, predicted, lambda row: row.fact.id)[0]
        )
        return [
            ItemRow(id=row.fact.id, correct=row.verdict == f.verdict.value) for row, f in pairs
        ], []
    if stage == "validate":
        joined, _ = join_by_id(labels, predicted, lambda row: row.fact.id)
        return [ItemRow(id=row.fact.id, correct=row.action == p.action) for row, p in joined], []
    if stage == "route":
        joined, _ = join_by_id(labels, predicted, lambda row: row.id, noun="chunk")
        return [ItemRow(id=row.id, correct=row.action == p.action) for row, p in joined], []
    raise ValueError(f"--items is written for {', '.join(ITEM_STAGES)}; not for {stage}")


def write_items(path: str | Path, rows: Sequence[ItemRow]) -> None:
    """One row per line, leaving out the fields the row does not carry."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json(exclude_none=True) + "\n")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _index(rows: Sequence[ItemRow], name: str) -> dict[str, ItemRow]:
    by_id: dict[str, ItemRow] = {}
    for row in rows:
        if row.id in by_id:
            raise ValueError(f"{name}: item {row.id!r} is listed twice; each item has one row")
        by_id[row.id] = row
    if not by_id:
        raise ValueError(f"{name}: no items")
    return by_id


def _same_items(a: dict[str, ItemRow], b: dict[str, ItemRow], a_name: str, b_name: str) -> None:
    only_a = [key for key in a if key not in b]
    only_b = [key for key in b if key not in a]
    if not only_a and not only_b:
        return
    parts = [
        f"{len(ids)} only in {name} ({_some(ids)})"
        for name, ids in ((a_name, only_a), (b_name, only_b))
        if ids
    ]
    message = f"the runs score different items: {'; '.join(parts)}"
    if len(only_a) == len(a) and len(only_b) == len(b):
        message += (
            ". No id is in both: were both written from the same labels? A labelled fact "
            'with no "id" gets a fresh one on every load'
        )
    raise ValueError(message)


def _some(ids: Sequence[str], shown: int = 5) -> str:
    listed = ", ".join(ids[:shown])
    return listed if len(ids) <= shown else f"{listed}, and {len(ids) - shown} more"


def _carries(rows: Sequence[ItemRow], fields: Sequence[str]) -> bool:
    return all(getattr(row, f) is not None for row in rows for f in fields)


def _defined(
    compute: MetricFn, a: Sequence[ItemRow], b: Sequence[ItemRow], fields: tuple[str, ...]
) -> bool:
    return all(compute(_totals(rows, fields), len(rows)) is not None for rows in (a, b))


def _totals(rows: Sequence[ItemRow], fields: Sequence[str]) -> tuple[int, ...]:
    return tuple(sum(int(getattr(row, f)) for row in rows) for f in fields)


def _value(row: ItemRow, fields: Sequence[str]) -> Item:
    if fields == ("correct",):
        return bool(row.correct)
    return tuple(int(getattr(row, f)) for f in fields)


def _points(value: float) -> str:
    return f"{value * 100:+.1f}"


def _sentence(p: Paired, unit: str) -> str:
    if p.verdict == "worse":
        return f"worse: B is below A, and the whole {p.level:.0%} interval is below zero"
    if p.verdict == "better":
        return f"better: B is above A, and the whole {p.level:.0%} interval is above zero"
    if p.detection_limit is None:
        return f"inconclusive: no {unit[:-1]} changed between the runs"
    return (
        "inconclusive: with this set the smallest change it can detect is "
        f"{p.detection_limit * 100:.1f} points"
    )


def _limit(p: Paired, unit: str) -> str:
    if p.detection_limit is None:
        return f"unknown: no {unit[:-1]} changed, so the set has shown no spread to measure"
    return (
        f"{p.detection_limit * 100:.1f} points: the smallest change these {p.items} {unit} "
        f"detect ({p.level:.0%} confidence, 80% power)"
    )


__all__ = [
    "COUNTS",
    "DESCRIPTION",
    "ITEM_STAGES",
    "METRICS",
    "Comparison",
    "ItemRow",
    "compare_files",
    "compare_items",
    "gate",
    "item_rows",
    "write_items",
]
