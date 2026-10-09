"""The statistics of a comparison: numbers checked by hand, and the resampler checked exact.

detection limit = Z·sqrt(d/n), Z = 1.959964 + 0.841621 = 2.801585
  300 items, 10% flips:   2.801585 × sqrt(0.1/300)  = 0.05115  (5.1 points)
  1,500 items:            2.801585 × sqrt(0.1/1500) = 0.02288  (2.3 points)
  5,000 items:            2.801585 × sqrt(0.1/5000) = 0.01253  (1.3 points)

McNemar, 1 gained and 9 lost: 10 flips, P(a split at least that uneven)
  = 2 × (C(10,0) + C(10,1)) / 2^10 = 22 / 1024 = 0.021484375
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from openodke.eval.report import prf
from openodke.eval.stats import (
    _binomial,
    bootstrap_interval,
    detection_limit,
    f1,
    mcnemar,
    paired_bootstrap,
    precision,
    share,
)

Z = 1.959964 + 0.841621


def _pass_fail(both: int, gained: int, lost: int, neither: int) -> tuple[list[bool], list[bool]]:
    """Two runs over the same items: pass in both, fail then pass, pass then fail, fail in both."""
    a = [True] * both + [False] * gained + [True] * lost + [False] * neither
    b = [True] * both + [True] * gained + [False] * lost + [False] * neither
    return a, b


# --------------------------------------------------------------------------- #
# The detection limit and McNemar, by hand
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("n", "points"), [(300, 0.05115), (1_500, 0.02288), (5_000, 0.01253)])
def test_detection_limit_by_hand(n: int, points: float) -> None:
    assert detection_limit(n, 0.10) == pytest.approx(points, abs=5e-5)
    assert detection_limit(n, 0.10) == pytest.approx(Z * math.sqrt(0.10 / n), rel=1e-6)


def test_detection_limit_shrinks_with_items_and_grows_with_flips() -> None:
    assert detection_limit(1_200, 0.1) == pytest.approx(detection_limit(300, 0.1) / 2)
    assert detection_limit(300, 0.4) == pytest.approx(detection_limit(300, 0.1) * 2)
    # Less power or more confidence asks for a bigger change.
    assert detection_limit(300, 0.1, power=0.9) > detection_limit(300, 0.1)
    assert detection_limit(300, 0.1, alpha=0.01) > detection_limit(300, 0.1)


@pytest.mark.parametrize(("n", "share_"), [(0, 0.1), (300, -0.1), (300, 1.5)])
def test_detection_limit_refuses_what_is_not_a_set(n: int, share_: float) -> None:
    with pytest.raises(ValueError):
        detection_limit(n, share_)


def test_mcnemar_by_hand() -> None:
    a, b = _pass_fail(both=80, gained=1, lost=9, neither=10)
    result = mcnemar(a, b)
    assert (result.items, result.gained, result.lost) == (100, 1, 9)
    assert result.p_value == pytest.approx(22 / 1024)


def test_mcnemar_with_no_flips_or_an_even_split_finds_nothing() -> None:
    assert mcnemar(*_pass_fail(50, 0, 0, 50)).p_value == 1.0
    assert mcnemar(*_pass_fail(50, 6, 6, 50)).p_value == 1.0


# --------------------------------------------------------------------------- #
# The paired bootstrap
# --------------------------------------------------------------------------- #


def test_identical_runs_are_inconclusive_with_no_limit_to_claim() -> None:
    a = [True] * 70 + [False] * 30
    result = paired_bootstrap(a, list(a))
    assert result.difference == 0.0 and result.interval == (0.0, 0.0)
    assert (result.flips, result.verdict) == (0, "inconclusive")
    # Nothing flipped, so the set has shown no spread to measure a change against.
    assert result.detection_limit is None


def test_a_large_paired_drop_is_worse_and_the_limit_is_the_formula() -> None:
    a, b = _pass_fail(both=240, gained=3, lost=27, neither=30)
    result = paired_bootstrap(a, b)
    assert result.items == 300 and result.flips == 30 and result.flip_share == 0.1
    assert result.a == pytest.approx(267 / 300) and result.b == pytest.approx(243 / 300)
    assert result.difference == pytest.approx(-0.08)
    assert result.interval[1] < 0 and result.verdict == "worse"
    assert result.detection_limit == pytest.approx(detection_limit(300, 0.1))
    assert result.a_interval[0] < result.a < result.a_interval[1]
    assert result.b_interval[0] < result.b < result.b_interval[1]


def test_a_gain_is_better() -> None:
    b, a = _pass_fail(both=240, gained=3, lost=27, neither=30)
    assert paired_bootstrap(a, b).verdict == "better"


def test_the_same_seed_gives_the_same_interval() -> None:
    a, b = _pass_fail(both=240, gained=12, lost=18, neither=30)
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)
    assert paired_bootstrap(a, b, seed=7).interval != paired_bootstrap(a, b, seed=8).interval


def test_the_resampler_is_exact_on_pass_fail_items() -> None:
    """The bootstrap variance of a paired pass/fail difference is (d − δ²)/n, exactly.

    A metric that is not `share` itself reports Z times the bootstrap's own
    standard deviation, so the limit reads the resampler's spread back out. Few
    distinct outcomes take the multinomial draw; a tag column that makes every
    item distinct forces the draw item by item. Both must match the formula.
    """
    a, b = _pass_fail(both=240, gained=12, lost=18, neither=30)
    n, d, delta = 300, 30 / 300, -6 / 300
    expected = math.sqrt((d - delta**2) / n)

    def first_share(totals: tuple[float, ...], items: int) -> float:
        return totals[0] / items

    grouped = paired_bootstrap(a, b, first_share, n=4000)
    tagged = paired_bootstrap(
        [(int(x), i) for i, x in enumerate(a)],
        [(int(x), i) for i, x in enumerate(b)],
        first_share,
        n=4000,
    )
    for result in (grouped, tagged):
        assert result.detection_limit is not None
        # The standard deviation of 4,000 resamples is good to about 1.1%.
        assert result.detection_limit / Z == pytest.approx(expected, rel=0.05)
    assert grouped.interval == pytest.approx(tagged.interval, abs=0.006)


def test_a_binomial_draw_follows_the_binomial() -> None:
    rng = random.Random(3)
    trials, p, draws = 20, 0.3, 40_000
    seen = Counter(_binomial(rng, trials, p) for _ in range(draws))
    chi2 = 0.0
    for k in range(trials + 1):
        expected = draws * math.comb(trials, k) * p**k * (1 - p) ** (trials - k)
        if expected >= 5:
            chi2 += (seen[k] - expected) ** 2 / expected
    # 13 cells with an expectation of 5 or more; the 99.9th percentile of chi²(12) is 32.9.
    assert chi2 < 32.9
    # A trial count whose terms underflow a float still draws around its mean.
    big = [_binomial(rng, 100_000, 0.37) for _ in range(500)]
    assert sum(big) / len(big) == pytest.approx(37_000, abs=4 * math.sqrt(23_310 / 500))
    assert _binomial(rng, 10, 0.0) == 0 and _binomial(rng, 10, 1.0) == 10


def test_a_corpus_metric_is_resampled_by_document() -> None:
    """Per document: its counts, and corpus F1 recomputed from the summed counts."""
    rng = random.Random(11)
    a = [(rng.randint(0, 5), rng.randint(0, 2), rng.randint(0, 2)) for _ in range(200)]
    # B loses one true fact in every fifth document.
    b = [
        (tp - 1, fp, fn + 1) if i % 5 == 0 and tp else (tp, fp, fn)
        for i, (tp, fp, fn) in enumerate(a)
    ]
    result = paired_bootstrap(a, b, f1)
    sums_a = [sum(col) for col in zip(*a, strict=True)]
    sums_b = [sum(col) for col in zip(*b, strict=True)]
    assert result.a == pytest.approx(prf(*sums_a)["f1"])
    assert result.b == pytest.approx(prf(*sums_b)["f1"])
    assert result.flips == sum(1 for x, y in zip(a, b, strict=True) if x != y)
    assert result.verdict == "worse"
    # A document moves corpus F1 by its own amount, so the limit is measured, not 1/n per flip.
    assert result.detection_limit is not None
    assert result.detection_limit != pytest.approx(detection_limit(200, result.flip_share))


def test_runs_over_different_numbers_of_items_are_refused() -> None:
    with pytest.raises(ValueError, match="same items"):
        paired_bootstrap([True, False], [True])
    with pytest.raises(ValueError, match="no items"):
        paired_bootstrap([], [])
    with pytest.raises(ValueError, match="same items"):
        mcnemar([True], [True, False])


def test_a_metric_undefined_on_a_whole_run_is_refused() -> None:
    nothing_predicted = [(0, 0, 3)] * 10
    with pytest.raises(ValueError, match="undefined"):
        paired_bootstrap(nothing_predicted, [(1, 0, 2)] * 10, precision)


def test_one_runs_range() -> None:
    items = [True] * 240 + [False] * 60
    low, high = bootstrap_interval(items) or (0.0, 0.0)
    assert low < 0.8 < high
    assert bootstrap_interval(items, seed=5) == bootstrap_interval(items, seed=5)
    # Four times the items, about half the width.
    low4, high4 = bootstrap_interval(items * 4) or (0.0, 0.0)
    assert (high4 - low4) == pytest.approx((high - low) / 2, rel=0.2)
    assert share((240,), 300) == 0.8
