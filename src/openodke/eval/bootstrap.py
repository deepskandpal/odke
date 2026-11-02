"""Ranges by resampling documents: a seeded percentile bootstrap, standard library only.

A precision measured on forty documents is one draw. Run the same pipeline on
forty others and it moves, and a report that prints 0.812 without saying by how
much it could move invites a reader to compare two runs that cannot be told
apart. So every precision, recall and F1 in an eval report carries a 95% range.

The unit resampled is the **document**, never the fact. Facts from one document
share one reading of it: an extractor that misreads a page misreads several of
its facts at once, so facts are not independent draws, and resampling them
would print a range narrower than the truth.

The method is the plain percentile bootstrap. Draw as many documents as there
are, with replacement; compute the statistic on the draw; repeat; read the
2.5th and 97.5th percentiles. A draw on which the statistic is undefined (no
predictions at all, so no precision) is left out rather than counted as zero.

Seeded, and the same on every machine. Only `random.Random.random()` is used,
because Python guarantees its sequence for a given integer seed across
versions; `randrange` and `choices` carry no such promise. The same units,
statistic, seed and resample count give the same range, byte for byte, which
is what lets a CI gate compare two reports at all.

Paired comparisons (`openodke.eval.stats`, #142) draw from here too: their
resampler takes each draw's indices from `draws` and reads every interval with
`percentile_range`, so a report's ranges and a comparison's come from one
bootstrap. Where items repeat, as pass/fail outcomes do, it counts how many of
each outcome a draw takes instead, an exact shortcut with the same distribution.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TypeVar

# One count for every range and every comparison (`openodke.eval.stats`).
RESAMPLES = 2000
SEED = 0
LEVEL = 0.95

Unit = TypeVar("Unit")
Range = tuple[float | None, float | None]


def draws(n: int, *, resamples: int = RESAMPLES, seed: int = SEED) -> Iterator[list[int]]:
    """`resamples` lists of `n` indices into `range(n)`, drawn with replacement.

    The same `n`, `resamples` and `seed` give the same lists on any machine.
    """
    if n < 0 or resamples < 0:
        raise ValueError("n and resamples cannot be negative")
    rng = random.Random(seed)
    for _ in range(resamples):
        yield [min(int(rng.random() * n), n - 1) for _ in range(n)]


def percentile_range(values: Sequence[float], level: float = LEVEL) -> tuple[float, float]:
    """The central `level` of `values`: its lower and upper percentiles, interpolated.

    Linear interpolation between order statistics, as `numpy.percentile` does by
    default, so the bounds agree with what a reader recomputes elsewhere.
    """
    if not values:
        raise ValueError("no values to take a range of")
    if not 0 < level < 1:
        raise ValueError(f"level is a share between 0 and 1, got {level}")
    ordered = sorted(values)
    # Rounded so 95% reads the 2.5th percentile exactly, not 0.025000000000000022.
    tail = round((1 - level) / 2, 12)
    return _at(ordered, tail), _at(ordered, 1 - tail)


def bootstrap(
    units: Sequence[Unit],
    statistic: Callable[[Sequence[Unit]], Mapping[str, float | None]],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> dict[str, Range]:
    """A range for every number `statistic` returns, from one set of draws.

    `units` are the documents, each in whatever form `statistic` reads: counts
    for a micro-average, a per-document score for a mean. `statistic` takes a
    resampled list of them and returns named numbers; each name gets its own
    `(low, high)`, or `(None, None)` when it was undefined on every draw. With
    no units there is nothing to resample, and every range is `(None, None)`.
    """
    names = list(statistic(units))
    values: dict[str, list[float]] = {name: [] for name in names}
    if units:
        for picked in draws(len(units), resamples=resamples, seed=seed):
            scored = statistic([units[i] for i in picked])
            for name in names:
                value = scored.get(name)
                if value is not None and math.isfinite(value):
                    values[name].append(float(value))
    return {
        name: percentile_range(found, level) if found else (None, None)
        for name, found in values.items()
    }


def _at(ordered: Sequence[float], q: float) -> float:
    position = q * (len(ordered) - 1)
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


__all__ = ["LEVEL", "RESAMPLES", "SEED", "Range", "bootstrap", "draws", "percentile_range"]
