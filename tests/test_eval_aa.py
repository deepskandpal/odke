"""Evaluating the Evaluator: an A/A comparison, a planted regression and a planted tiny change.

The same system run twice differs only by noise, so its comparison must come
back inconclusive about 95% of the time: a 95% interval excludes zero by chance
one time in twenty. Much more often, and the comparison cries wolf; much less,
and its intervals are too wide to catch anything. A planted drop well above
the detection limit must come back worse, and one well below it inconclusive,
with the limit printed instead of "no regression".

The simulated system gives each item a chance of passing: most items are
settled either way, and one in five is a coin toss, so about 15% of items flip
between two runs of the same system. A document is a handful of gold facts,
each with its own chance of being found, and three chances of a spurious fact.
Everything is seeded, so the measured rate is the same on every run.
"""

from __future__ import annotations

import math
import random

from openodke.eval.compare import ItemRow, compare_items
from openodke.eval.stats import detection_limit

SETTLED_PASS, SETTLED_FAIL, COIN = 0.97, 0.03, 0.5


def _system(rng: random.Random, items: int) -> list[float]:
    """Each item's chance of passing under one system."""
    return [rng.choice((SETTLED_PASS,) * 3 + (SETTLED_FAIL, COIN)) for _ in range(items)]


def _run(
    rng: random.Random, system: list[float], broken: frozenset[int] = frozenset()
) -> list[ItemRow]:
    """One run of the system: each item passes by its own chance, unless planted broken."""
    return [
        ItemRow(id=f"i{n}", correct=n not in broken and rng.random() < p)
        for n, p in enumerate(system)
    ]


def test_an_aa_comparison_is_inconclusive_about_95_percent_of_the_time() -> None:
    """300 A/A comparisons of 100 items: 95% inconclusive, within three binomial SDs.

    Measured at this seed: 98.3% (295 of 300), with McNemar's exact test on
    the flips gating a pass/fail verdict; 96.7% on the bootstrap alone.
    Offline, at 1,500 comparisons each (seed 2026): 96.4% for 300 items at
    2,000 resamples and 97.9% for 100 items at 1,000 (95.1% and 96.1% on the
    bootstrap alone); 94.7% and 94.9% for corpus F1 over 150 and 400
    documents, which the gate does not touch.
    """
    rng, reps = random.Random(147), 300
    inconclusive = 0
    for rep in range(reps):
        system = _system(rng, 100)
        a, b = _run(rng, system), _run(rng, system)
        inconclusive += compare_items(a, b, resamples=1000, seed=rep).verdict == "inconclusive"
    tolerance = 3 * math.sqrt(0.95 * 0.05 / reps)
    assert abs(inconclusive / reps - 0.95) <= tolerance, f"{inconclusive} of {reps} inconclusive"


def _false_verdicts(flips: int, items: int = 300) -> float:
    """How often a change that is not there gets a verdict, when `flips` of `items` flip.

    With no real change each flip goes either way on a fair coin, so `gained`
    flips of `flips` has probability C(flips, gained) / 2^flips. Each split is
    compared, so the rate is exact, not sampled. A split and its mirror image
    get mirror verdicts, and a split nearer even leans less, so the splits are
    tried from the most lopsided in, until one is inconclusive.
    """
    both = (items - flips) * 2 // 3
    neither = items - flips - both
    false = 0.0
    for gained in range(flips // 2 + 1):
        lost = flips - gained
        a = [True] * both + [False] * gained + [True] * lost + [False] * neither
        b = [True] * both + [True] * gained + [False] * lost + [False] * neither
        rows_a = [ItemRow(id=f"i{n}", correct=x) for n, x in enumerate(a)]
        rows_b = [ItemRow(id=f"i{n}", correct=x) for n, x in enumerate(b)]
        if compare_items(rows_a, rows_b).verdict == "inconclusive":
            break
        mirrors = 1 if gained == lost else 2
        false += mirrors * math.comb(flips, gained) / 2**flips
    return false


def test_a_handful_of_flips_gives_no_verdict_beyond_the_stated_rate() -> None:
    """Release audit 10: four one-way flips read as "better", though McNemar's p is 0.125.

    On its own the bootstrap gave a verdict on noise 12.5% of the time at 4 flips and
    6.2% at 5, against the 5% it states. A pass/fail verdict needs McNemar's exact
    test on the flips to agree, so at every flip count it stays at or under 5%.
    """
    rates = {flips: _false_verdicts(flips) for flips in (*range(1, 13), 16, 20, 30, 60)}
    over = {k: f"{v:.1%}" for k, v in rates.items() if v > 0.05}
    assert not over, f"a verdict on no change, by flip count: {over}"
    assert rates[4] == 0.0 and rates[5] == 0.0, rates
    # Not vacuous: at 20 flips, 2 of 20 one way is a real lean, and is called.
    assert rates[20] > 0.0, rates


def test_a_planted_regression_well_above_the_limit_is_worse() -> None:
    """A quarter of the items broken in B: about 17 points down, about twice the limit."""
    for seed in range(10):
        rng = random.Random(seed)
        system = _system(rng, 300)
        broken = frozenset(rng.sample(range(300), 75))
        comparison = compare_items(_run(rng, system), _run(rng, system, broken), seed=seed)
        limit = comparison.primary.detection_limit
        assert comparison.verdict == "worse", seed
        assert limit is not None and comparison.primary.difference < -1.5 * limit


def test_a_planted_corpus_regression_is_worse() -> None:
    """F1 over 300 documents, with B missing every fifth gold fact it would have found."""
    rng = random.Random(5)
    docs = [_system(rng, rng.randint(1, 8)) for _ in range(300)]

    def run(drop: bool) -> list[ItemRow]:
        rows = []
        for n, facts in enumerate(docs):
            found = sum(rng.random() < p and not (drop and k % 5 == 0) for k, p in enumerate(facts))
            spurious = sum(rng.random() < 0.05 for _ in range(3))
            rows.append(ItemRow(id=f"d{n}", tp=found, fp=spurious, fn=len(facts) - found))
        return rows

    comparison = compare_items(run(drop=False), run(drop=True))
    assert (comparison.metric, comparison.unit, comparison.verdict) == ("f1", "document", "worse")
    assert comparison.guardrails["recall"].verdict == "worse"


def test_a_planted_tiny_change_is_inconclusive_and_prints_the_limit() -> None:
    """Three of 300 items broken in B: a point at most, under a limit of about 6 points."""
    rng = random.Random(42)
    system = _system(rng, 300)
    broken = frozenset(rng.sample(range(300), 3))
    comparison = compare_items(_run(rng, system), _run(rng, system, broken))
    primary = comparison.primary
    assert comparison.verdict == "inconclusive"
    assert primary.detection_limit is not None
    assert primary.detection_limit == detection_limit(300, primary.flip_share)
    assert primary.detection_limit > 3 / 300
    text = comparison.render()
    assert (
        "inconclusive: with this set the smallest change it can detect is "
        f"{primary.detection_limit * 100:.1f} points"
    ) in text
    assert "no regression" not in text
