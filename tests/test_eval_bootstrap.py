"""The bootstrap behind every range in an eval report: seeded, over documents, stdlib only."""

from __future__ import annotations

import random

import pytest

from openodke.eval import stats
from openodke.eval.bootstrap import RESAMPLES, bootstrap, draws, percentile_range
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


# --------------------------------------------------------------------------- #
# One bootstrap: the report's ranges and a comparison's draw the same documents
# --------------------------------------------------------------------------- #


def _documents(n: int, seed: int) -> list[tuple[int, int, int]]:
    """`n` documents' counts, all different, so the resampler draws them one by one."""
    rng = random.Random(seed)
    return [(rng.randint(0, 9) + 10 * i, rng.randint(0, 4), rng.randint(0, 4)) for i in range(n)]


def test_one_run_s_interval_in_stats_is_the_report_s_range() -> None:
    documents = _documents(40, seed=1)
    ranges = bootstrap(documents, micro, resamples=500, seed=3)
    for name, metric in (
        ("precision", stats.precision),
        ("recall", stats.recall),
        ("f1", stats.f1),
    ):
        assert stats.bootstrap_interval(documents, metric, n=500, seed=3) == ranges[name]


def test_a_comparison_draws_each_run_on_the_report_s_draws() -> None:
    a, b = _documents(30, seed=2), _documents(30, seed=4)
    paired = stats.paired_bootstrap(a, b, stats.f1, n=400, seed=9)
    assert paired.a_interval == bootstrap(a, micro, resamples=400, seed=9)["f1"]
    assert paired.b_interval == bootstrap(b, micro, resamples=400, seed=9)["f1"]


def test_the_resampler_s_totals_are_the_draws_summed() -> None:
    documents = _documents(12, seed=5)
    resampled = [totals for (totals,) in stats._resample([documents], 50, 6)]
    expected = [
        tuple(sum(documents[i][k] for i in picked) for k in range(3))
        for picked in draws(12, resamples=50, seed=6)
    ]
    assert resampled == expected


def test_both_read_the_same_percentiles() -> None:
    values = sorted(random.Random(8).random() for _ in range(997))
    assert stats._percentile(values, 0.05) == percentile_range(values, 0.95)
    assert stats.RESAMPLES == RESAMPLES == 2000
