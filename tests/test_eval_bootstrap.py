"""The bootstrap behind every range in an eval report: seeded, over documents, stdlib only."""

from __future__ import annotations

import pytest

from openodke.eval.bootstrap import bootstrap, draws, percentile_range
from openodke.eval.report import Metric, prf


def micro(units: list[tuple[int, int, int]]) -> dict[str, Metric]:
    """Precision, recall and F1 over `(tp, fp, fn)` per document, pooled."""
    scores = prf(*(sum(u[i] for u in units) for i in range(3)))
    return {k: scores[k] for k in ("precision", "recall", "f1")}


def test_a_seeded_bootstrap_draws_the_same_documents_on_every_machine() -> None:
    """Pinned: Python promises `random()`'s sequence for a seed, and nothing else is used."""
    assert next(draws(5, seed=0)) == [4, 3, 2, 1, 2]
    assert list(draws(5, resamples=3, seed=7)) == list(draws(5, resamples=3, seed=7))
    assert list(draws(5, resamples=3, seed=7)) != list(draws(5, resamples=3, seed=8))
    assert all(0 <= i < 5 for draw in draws(5, resamples=50) for i in draw)


def test_a_seeded_bootstrap_gives_the_same_range_twice() -> None:
    units = [(3, 1, 0), (0, 2, 2), (5, 0, 1), (1, 1, 1), (2, 0, 3)]
    first = bootstrap(units, micro, seed=11)
    assert first == bootstrap(units, micro, seed=11)
    assert first != bootstrap(units, micro, seed=12)
    low, high = first["precision"]
    assert low is not None and high is not None
    assert low <= micro(units)["precision"] <= high  # type: ignore[operator]


def test_the_range_is_the_central_95_percent_interpolated() -> None:
    # 101 evenly spaced values: the 2.5th and 97.5th percentiles fall at 3.5 and 98.5.
    assert percentile_range([float(v) for v in range(1, 102)]) == pytest.approx((3.5, 98.5))
    assert percentile_range([0.25]) == (0.25, 0.25)
    with pytest.raises(ValueError, match="no values"):
        percentile_range([])
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile_range([1.0], level=95)


def test_a_statistic_undefined_on_every_draw_has_no_range() -> None:
    nothing_predicted = [(0, 0, 2), (0, 0, 1)]
    ranges = bootstrap(nothing_predicted, micro)
    assert ranges["precision"] == (None, None)
    assert ranges["recall"] == (0.0, 0.0)
    assert bootstrap([], micro)["f1"] == (None, None)


def test_a_statistic_can_compare_two_runs_on_the_same_draws() -> None:
    """How a paired comparison (#142) reuses it: each unit is one document in both runs."""
    paired = [((1, 0, 0), (1, 0, 0)), ((0, 1, 1), (1, 0, 0)), ((1, 0, 0), (1, 0, 0))]

    def gain(draw: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> dict[str, Metric]:
        before = micro([a for a, _ in draw])["f1"]
        after = micro([b for _, b in draw])["f1"]
        return {"f1": None if before is None or after is None else after - before}

    low, high = bootstrap(paired, gain)["f1"]
    assert low is not None and high is not None and 0.0 <= low <= high <= 1.0
