"""Was the change real? Two runs compared on the same items, and one run's range.

An eval is an instrument with three ways to lie: what it sampled, how its
judges are calibrated, and its statistics. This module is the third. It never
compares two runs as two independent samples. Every item was scored by both,
most pass or fail in both, and only the items that flip carry information, so
the comparison is paired: a **paired bootstrap** over items (Koehn 2004)
resamples items with replacement, recomputes the metric for both runs on the
same resample, and takes the difference's percentile interval. `mcnemar` is the
exact test on the flips, a cross-check for a pass/fail outcome.

There are three verdicts, never two. *better* and *worse* need the whole
interval on one side of zero. Everything else is *inconclusive*, and is
reported with the **detection limit**: the smallest change this set could have
seen, at 95% confidence and 80% power. An inconclusive result is not "no
regression"; it is "a change smaller than this would not show here".

Every metric here is a ratio of sums: the share of items that pass, or
precision, recall and F1 from summed counts. So a resample is its column
totals, and the bootstrap never builds the resampled list. For a set-level
metric an item is a document, and its counts are what it adds to the corpus
totals. Standard library only (DECISIONS #1), and the same seed gives the same
interval on every interpreter.

`bootstrap_interval` is the same resampler over one run, for a metric's own 95%
range. The draws and the percentiles are `openodke.eval.bootstrap`'s, the ones
an eval report's ranges come from, so a report's ranges and its comparisons
come from one bootstrap.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import lru_cache
from operator import mul
from statistics import NormalDist, stdev
from typing import Literal, TypeAlias

from openodke.eval.bootstrap import RESAMPLES, draws, percentile_range
from openodke.eval.report import prf
from openodke.types import Frozen

# One item's outcome: pass/fail, a number, or a document's counts (tp, fp, fn).
Item: TypeAlias = bool | int | float | tuple[int | float, ...]
Totals: TypeAlias = tuple[int | float, ...]
# A run's metric from its items' column totals and how many items there were.
MetricFn: TypeAlias = Callable[[Totals, int], float | None]
Verdict: TypeAlias = Literal["better", "worse", "inconclusive"]

ALPHA = 0.05
POWER = 0.8


# --------------------------------------------------------------------------- #
# Metrics, as functions of column totals
# --------------------------------------------------------------------------- #


def share(totals: Totals, n: int) -> float | None:
    """The share of items that pass: the mean of a pass/fail outcome."""
    return totals[0] / n if n else None


def _prf(totals: Totals, key: str) -> float | None:
    tp, fp, fn = (int(t) for t in totals)
    value = prf(tp, fp, fn)[key]
    return None if value is None else float(value)


def precision(totals: Totals, n: int) -> float | None:
    """Corpus precision from summed (tp, fp, fn). Undefined when nothing was predicted."""
    return _prf(totals, "precision")


def recall(totals: Totals, n: int) -> float | None:
    """Corpus recall from summed (tp, fp, fn). Undefined when nothing was labelled."""
    return _prf(totals, "recall")


def f1(totals: Totals, n: int) -> float | None:
    """Corpus F1 from summed (tp, fp, fn), with `prf`'s rules for the undefined cases."""
    return _prf(totals, "f1")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class Paired(Frozen):
    """Run A against run B on the same items: the difference and what it means.

    `difference` is B minus A, so a positive one means B scored higher; every
    metric here is better higher. `interval` is the difference's percentile
    interval at `level`, and decides the verdict. `a_interval` and `b_interval`
    are each run's own range from the same resamples. `flips` counts the items
    whose outcome differs between the runs: a pass that became a fail or back,
    or a document whose counts changed.
    """

    items: int
    a: float
    b: float
    difference: float
    interval: tuple[float, float]
    a_interval: tuple[float, float]
    b_interval: tuple[float, float]
    flips: int
    flip_share: float
    # The smallest change this set detects at `level` with 80% power. None when
    # no item changed: the runs then disagree nowhere, and the set has shown no
    # spread to measure a change against.
    detection_limit: float | None
    verdict: Verdict
    level: float
    # Resamples with a defined difference. Fewer than asked for means some
    # resample left the metric undefined (no prediction at all, say), and was dropped.
    resamples: int
    seed: int


class McNemar(Frozen):
    """The exact McNemar test: given an item flipped, was each direction equally likely?"""

    items: int
    # Failed in A and passed in B; passed in A and failed in B.
    gained: int
    lost: int
    p_value: float


# --------------------------------------------------------------------------- #
# The tests
# --------------------------------------------------------------------------- #


def detection_limit(n: int, flip_share: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """The smallest paired difference `n` items detect: Z·sqrt(d/n), as a fraction.

    `d` is the share of items that flip, and Z is the two-sided critical value
    plus the power's: 1.959964 + 0.841621 at 95% and 80%. Each flip moves a
    pass rate by 1/n, so the paired difference has a standard error of about
    sqrt(d/n). 300 items with 10% flips detect about 5.1 points; 1,500 about 2.3.
    """
    if n <= 0:
        raise ValueError(f"the detection limit needs at least one item, got n={n}")
    if not 0.0 <= flip_share <= 1.0:
        raise ValueError(f"flip_share is a share of items, between 0 and 1; got {flip_share}")
    return _z(alpha, power) * math.sqrt(flip_share / n)


def paired_bootstrap(
    a_items: Sequence[Item],
    b_items: Sequence[Item],
    metric: MetricFn = share,
    n: int = RESAMPLES,
    seed: int = 0,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
) -> Paired:
    """Resample items with replacement, recompute both runs, and read the difference.

    Element *i* of each list is the same item under each run. The verdict is
    *worse* when the whole interval is below zero, *better* when it is above,
    and *inconclusive* otherwise.

    The detection limit is Z times the standard error of the difference. For
    pass/fail items scored by `share`, each flip moves the metric by exactly
    1/n, so it is `detection_limit(n, flip_share)`. For a corpus metric a
    document moves it by its own amount, not 1/n, so it is Z times the
    bootstrap's own standard error: the same quantity, measured instead of
    assumed.
    """
    metrics = {"the metric": metric}
    return paired_bootstraps(a_items, b_items, metrics, n, seed, alpha=alpha, power=power)[
        "the metric"
    ]


def paired_bootstraps(
    a_items: Sequence[Item],
    b_items: Sequence[Item],
    metrics: Mapping[str, MetricFn],
    n: int = RESAMPLES,
    seed: int = 0,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
) -> dict[str, Paired]:
    """`paired_bootstrap` for several metrics of the same items, from one set of resamples.

    One pass instead of one per metric, and a primary metric and its guardrails
    are read off the same draws.
    """
    a, b = _vectors(a_items), _vectors(b_items)
    if len(a) != len(b):
        raise ValueError(f"the runs must score the same items: {len(a)} against {len(b)}")
    if not a:
        raise ValueError("no items to compare")
    whole: dict[str, tuple[float, float]] = {}
    for name, metric in metrics.items():
        a_value, b_value = metric(_totals(a), len(a)), metric(_totals(b), len(b))
        if a_value is None or b_value is None:
            raise ValueError(f"{name} is undefined on a run as a whole; nothing to compare")
        whole[name] = (a_value, b_value)

    drawn: dict[str, tuple[list[float], list[float]]] = {name: ([], []) for name in metrics}
    for ta, tb in _resample([a, b], n, seed):
        for name, metric in metrics.items():
            ma, mb = metric(ta, len(a)), metric(tb, len(a))
            if ma is not None and mb is not None:
                drawn[name][0].append(ma)
                drawn[name][1].append(mb)

    flips = sum(1 for x, y in zip(a, b, strict=True) if x != y)
    pass_fail = all(v in ((0,), (1,)) for v in (*a, *b))
    results = {}
    for name, metric in metrics.items():
        a_draws, b_draws = drawn[name]
        if len(a_draws) < 2:
            raise ValueError(f"{name} is undefined on almost every resample; too few items")
        diffs = sorted(mb - ma for ma, mb in zip(a_draws, b_draws, strict=True))
        interval = _percentile(diffs, alpha)
        limit: float | None
        if not flips:
            limit = None
        elif metric is share and pass_fail:
            limit = detection_limit(len(a), flips / len(a), alpha, power)
        else:
            limit = _z(alpha, power) * stdev(diffs)
        verdict: Verdict = (
            "worse" if interval[1] < 0 else "better" if interval[0] > 0 else "inconclusive"
        )
        a_value, b_value = whole[name]
        results[name] = Paired(
            items=len(a),
            a=a_value,
            b=b_value,
            difference=b_value - a_value,
            interval=interval,
            a_interval=_percentile(sorted(a_draws), alpha),
            b_interval=_percentile(sorted(b_draws), alpha),
            flips=flips,
            flip_share=flips / len(a),
            detection_limit=limit,
            verdict=verdict,
            level=1 - alpha,
            resamples=len(diffs),
            seed=seed,
        )
    return results


def bootstrap_interval(
    items: Sequence[Item],
    metric: MetricFn = share,
    n: int = RESAMPLES,
    seed: int = 0,
    *,
    alpha: float = ALPHA,
) -> tuple[float, float] | None:
    """One run's percentile interval for `metric`: its 95% range, by default.

    None when the metric is undefined on all but one resample.
    """
    vectors = _vectors(items)
    if not vectors:
        raise ValueError("no items to resample")
    values = sorted(
        value
        for (totals,) in _resample([vectors], n, seed)
        if (value := metric(totals, len(vectors))) is not None
    )
    return _percentile(values, alpha) if len(values) > 1 else None


def mcnemar(a_items: Sequence[bool], b_items: Sequence[bool]) -> McNemar:
    """The exact two-sided McNemar test on pass/fail outcomes of the same items.

    Exact binomial rather than the chi-square approximation, because the flip
    counts that matter are small enough that the approximation is wrong
    exactly when the decision is close.
    """
    if len(a_items) != len(b_items):
        raise ValueError(
            f"the runs must score the same items: {len(a_items)} against {len(b_items)}"
        )
    gained = sum(1 for x, y in zip(a_items, b_items, strict=True) if not x and y)
    lost = sum(1 for x, y in zip(a_items, b_items, strict=True) if x and not y)
    flips = gained + lost
    # P(a split at least as uneven as this one | each flip is a fair coin).
    term, tail = 1, 0
    for k in range(min(gained, lost) + 1):
        tail += term
        term = term * (flips - k) // (k + 1)
    p_value = min(1.0, 2 * tail / (1 << flips)) if flips else 1.0
    return McNemar(items=len(a_items), gained=gained, lost=lost, p_value=p_value)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def _z(alpha: float, power: float) -> float:
    normal = NormalDist()
    return normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)


def _vectors(items: Sequence[Item]) -> list[Totals]:
    return [tuple(item) if isinstance(item, tuple) else (int(item),) for item in items]


def _totals(vectors: Sequence[Totals]) -> Totals:
    return tuple(sum(column) for column in zip(*vectors, strict=True))


def _resample(
    runs: Sequence[Sequence[Totals]], resamples: int, seed: int
) -> Iterator[list[Totals]]:
    """Each resample's column totals for every run, items drawn together across runs.

    Drawing the same items for every run is what makes the bootstrap paired.
    When the items fall into few distinct outcomes (pass/fail has at most four
    pairings), a resample is how many of each outcome it drew, a multinomial
    draw costing the number of outcomes rather than the number of items. Both
    ways draw from the same distribution; which one runs depends only on the
    data, so the same items and seed always give the same interval. Both use
    only `random()`, whose sequence for a seed Python keeps across versions.
    """
    rng = random.Random(seed)
    n = len(runs[0])
    rows = list(zip(*runs, strict=True))
    groups: dict[tuple[Totals, ...], int] = {}
    for row in rows:
        groups[row] = groups.get(row, 0) + 1
    if 4 * len(groups) <= n:
        outcomes = sorted(groups, key=groups.__getitem__, reverse=True)
        sizes = [groups[o] for o in outcomes]
        # Per run, per column: each outcome's value, so a total is one dot product.
        weights = [list(zip(*(o[run] for o in outcomes), strict=True)) for run in range(len(runs))]
        for _ in range(resamples):
            counts = _multinomial(rng, n, sizes)
            yield [tuple(sum(map(mul, counts, col)) for col in cols) for cols in weights]
        return
    # Item by item: the eval report's own draws (`openodke.eval.bootstrap.draws`).
    columns = [list(zip(*run, strict=True)) for run in runs]
    for picked in draws(n, resamples=resamples, seed=seed):
        yield [tuple(sum(map(col.__getitem__, picked)) for col in cols) for cols in columns]


def _multinomial(rng: random.Random, trials: int, sizes: Sequence[int]) -> list[int]:
    """How many of `trials` draws land in each group, each group drawn by its size."""
    counts, left, rest = [], trials, sum(sizes)
    for size in sizes[:-1]:
        drawn = _binomial(rng, left, size / rest) if left else 0
        counts.append(drawn)
        left -= drawn
        rest -= size
    counts.append(left)
    return counts


def _binomial(rng: random.Random, trials: int, p: float) -> int:
    """One binomial draw, by inversion searched outward from the mode.

    Any fixed order of visiting outcomes is an exact inversion, and starting at
    the mode visits about sqrt(n·p·q) of them rather than n.
    `random.binomialvariate` would do, but only from Python 3.12, and its
    stream would differ from this one's on 3.11.
    """
    if trials <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return trials
    mode, pmf, odds = _peak(trials, p)
    u = rng.random() - pmf
    if u < 0:
        return mode
    up = down = mode
    p_up = p_down = pmf
    while up < trials or down > 0:
        if up < trials:
            p_up *= (trials - up) / (up + 1) * odds
            up += 1
            u -= p_up
            if u < 0:
                return up
        if down > 0:
            p_down *= down / (trials - down + 1) / odds
            down -= 1
            u -= p_down
            if u < 0:
                return down
    return mode  # the probabilities summed a rounding error short of 1


@lru_cache(maxsize=4096)
def _peak(trials: int, p: float) -> tuple[int, float, float]:
    """A binomial's mode, the mode's probability, and p/q: where each draw starts.

    The probability comes from logs, so a large trial count cannot underflow
    it. Cached, because a resample's later groups are drawn from the few trial
    counts the earlier ones leave.
    """
    q = 1.0 - p
    mode = min(trials, int((trials + 1) * p))
    log_pmf = (
        math.lgamma(trials + 1)
        - math.lgamma(mode + 1)
        - math.lgamma(trials - mode + 1)
        + mode * math.log(p)
        + (trials - mode) * math.log(q)
    )
    return mode, math.exp(log_pmf), p / q


def _percentile(ordered: Sequence[float], alpha: float) -> tuple[float, float]:
    return percentile_range(ordered, 1 - alpha)


__all__ = [
    "ALPHA",
    "POWER",
    "RESAMPLES",
    "Item",
    "McNemar",
    "MetricFn",
    "Paired",
    "Verdict",
    "bootstrap_interval",
    "detection_limit",
    "f1",
    "mcnemar",
    "paired_bootstrap",
    "paired_bootstraps",
    "precision",
    "recall",
    "share",
]
