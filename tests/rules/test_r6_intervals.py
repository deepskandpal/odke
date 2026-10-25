"""R6. Intervals behave.

Every interval openodke prints beside a share claims a level, 95% by default.
Two did not hold it (release audit): the refusal audit pooled an equal-per-
stratum sample as if it were one random sample, so its range held the truth
0 times in 2,000 (finding 5); and the PPI-corrected precision's range shrank to
zero width when every label agreed, so at 99% precision it held the truth 71%
of the time with 100 labels (finding 6).

The check, for every closed-form interval on a share (`stats.wilson`, PPI's
corrected and labels-only ranges, the refusal precision):

- **width above zero**, whatever the labels, as property tests (hypothesis);
- **coverage near the stated level**: exact for Wilson, simulated and seeded
  for the others, never below the level less 5 points.

Not here: the percentile ranges `openodke.eval.bootstrap` puts beside a
precision, recall or F1. They resample documents, and are zero width when every
resample agrees (one document, or every fact right), which the release audit
checked as intended.

Deterministic: hypothesis is derandomised and every simulation is seeded, so a
failure here is the code's, never the draw's.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openodke.eval.formats import Refusal, RefusalLabel
from openodke.eval.ppi import prediction_powered
from openodke.eval.refusals import report_refusals, write_sample
from openodke.eval.stats import wilson

LEVEL = 0.95
# An interval may be conservative; it may not claim 95% and hold the truth 90% of the time.
FLOOR = LEVEL - 0.05
FEW = settings(max_examples=40, derandomize=True, deadline=None, database=None)


# --------------------------------------------------------------------------- #
# Wilson: the interval every share uses
# --------------------------------------------------------------------------- #


@FEW
@given(n=st.integers(1, 2000), data=st.data())
def test_a_wilson_interval_has_width_and_holds_its_share(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(0, n))
    interval = wilson(k, n, LEVEL)
    assert interval is not None
    low, high = interval
    assert low < high and low <= k / n <= high, (
        f"wilson({k}, {n}) is [{low}, {high}]: a share's interval must have width and hold "
        "the share it is around."
    )


@pytest.mark.parametrize("n", [10, 30, 100, 300])
def test_a_wilson_interval_covers_its_level(n: int) -> None:
    """Exact coverage, summed over the binomial, at shares from even to near certain."""
    for p in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        covered = 0.0
        for k in range(n + 1):
            low, high = wilson(k, n, LEVEL) or (0.0, 0.0)
            if low <= p <= high:
                covered += math.comb(n, k) * p**k * (1 - p) ** (n - k)
        assert covered >= FLOOR, (
            f"wilson at n={n} holds a share of {p} {covered:.1%} of the time, under the "
            f"{LEVEL:.0%} it claims. Use the score interval with the exact z."
        )


# --------------------------------------------------------------------------- #
# PPI: the judge corrected by a labelled sample
# --------------------------------------------------------------------------- #


@st.composite
def _judged_and_labelled(draw: st.DrawFn) -> tuple[list[bool], list[tuple[bool, bool]]]:
    """A judge's verdict on N facts, and (judge, label) on n of them drawn from those."""
    cells = draw(st.lists(st.integers(0, 80), min_size=4, max_size=4).filter(lambda c: sum(c) > 1))
    pairs = ((True, True), (True, False), (False, True), (False, False))
    labelled = [pair for pair, count in zip(pairs, cells, strict=True) for _ in range(count)]
    unlabelled = draw(st.lists(st.booleans(), max_size=2000))
    return [f for f, _ in labelled] + unlabelled, labelled


@FEW
@given(case=_judged_and_labelled())
def test_ppi_ranges_have_width(case: tuple[list[bool], list[tuple[bool, bool]]]) -> None:
    judged, labelled = case
    _, corrected, labels = prediction_powered(judged, labelled, level=LEVEL)
    for name, found in (("corrected", corrected), ("labels-only", labels)):
        assert found.low is not None and found.high is not None
        assert found.low < found.high, (
            f"The {name} precision from {len(labelled)} labels is "
            f"{found.value:.3f} [{found.low:.3f}, {found.high:.3f}]: a range of zero width. "
            "Labels that all agree still leave doubt: use plus-two counts in the PPI variance, "
            "(Σy+2)/(n+4), and a Wilson interval for the labels-only row."
        )


def _ppi_coverage(precision: float, wrong: float, n: int, *, facts: int = 1000) -> float:
    """How often the corrected range holds the facts' true precision, over seeded draws."""
    rng = random.Random(0)
    reps, held = 300, 0
    for _ in range(reps):
        truth = [rng.random() < precision for _ in range(facts)]
        judge = [y if rng.random() >= wrong else not y for y in truth]
        drawn = rng.sample(range(facts), n)
        _, corrected, _ = prediction_powered(judge, [(judge[i], truth[i]) for i in drawn])
        assert corrected.low is not None and corrected.high is not None
        held += corrected.low <= sum(truth) / facts <= corrected.high
    return held / reps


@pytest.mark.parametrize(
    ("precision", "wrong", "n"),
    [
        # The regime DECISIONS #36 simulated: it holds there.
        pytest.param(0.90, 0.10, 100, id="90%-precise, judge 10% wrong, 100 labels"),
        pytest.param(0.99, 0.01, 100, id="99%-precise, 100 labels"),
        pytest.param(0.99, 0.01, 150, id="99%-precise, 150 labels"),
    ],
)
def test_ppi_covers_its_level(precision: float, wrong: float, n: int) -> None:
    covered = _ppi_coverage(precision, wrong, n)
    assert covered >= FLOOR, (
        f"At {precision:.0%} precision with a judge {wrong:.0%} wrong and {n} labels, the "
        f"corrected {LEVEL:.0%} range held the truth {covered:.1%} of the time. A precise "
        "pipeline's labels mostly agree, and their measured variance understates the doubt: "
        "use plus-two counts in the variance."
    )


# --------------------------------------------------------------------------- #
# The refusal audit: a stratified sample
# --------------------------------------------------------------------------- #


def _refusals(strata: Sequence[tuple[str, int, int]]) -> tuple[list[Refusal], dict[str, bool]]:
    """`(dataset, size, right)` per stratum: the refusals, and which of them were right."""
    refusals, right = [], {}
    for dataset, size, correct in strata:
        for i in range(size):
            rid = f"{dataset}-{i:05d}"
            refusals.append(
                Refusal(id=rid, claim="c", text="t", verdict="not_found", reason="r",
                        predicate="p", dataset=dataset)
            )  # fmt: skip
            right[rid] = i < correct
    return refusals, right


def _labelled(out: Path, right: dict[str, bool]) -> list[RefusalLabel]:
    """What `odke eval refusals --make-sheet` wrote to `out`, ticked by someone who knows."""
    rows = [json.loads(line) for line in (out / "refusals.jsonl").read_text("utf-8").splitlines()]
    return [
        RefusalLabel.model_validate(
            {**row, "judgement": "refusal_correct" if right[row["id"]] else "refusal_wrong"}
        )
        for row in rows
    ]


def _overall(labels: list[RefusalLabel]) -> tuple[float, float, float]:
    (stage,) = report_refusals(labels).stages
    found = stage.breakdown["all"]
    return float(found["precision"]), float(found["low"]), float(found["high"])  # type: ignore[arg-type]


def test_the_refusal_precision_covers_its_level(tmp_path: Path) -> None:
    """The audit's case: 1,000 refusals 90% right, and 50 that are 30% right."""
    refusals, right = _refusals([("big", 1000, 900), ("small", 50, 15)])
    truth = sum(right.values()) / len(right)
    reps, held = 150, 0
    for seed in range(reps):
        write_sample(refusals, tmp_path / str(seed), n=100, seed=seed)
        _, low, high = _overall(_labelled(tmp_path / str(seed), right))
        held += low <= truth <= high
    assert held / reps >= FLOOR, (
        f"The refusal precision's {LEVEL:.0%} range held the truth ({truth:.3f}) "
        f"{held / reps:.1%} of the time. The sample takes an equal share of each stratum, so "
        "pooling it overweights the small ones: store each stratum's size with the sample, "
        "weight each stratum by it, and take the range from a stratified variance."
    )


@settings(max_examples=15, derandomize=True, deadline=None, database=None)
@given(
    big=st.integers(1, 400),
    small=st.integers(1, 400),
    n=st.integers(2, 120),
)
def test_a_stratified_sample_estimates_the_whole_population(big: int, small: int, n: int) -> None:
    """Strata that are all right and all wrong: any draw of them knows the truth exactly."""
    import tempfile

    refusals, right = _refusals([("all-right", big, big), ("all-wrong", small, 0)])
    truth = big / (big + small)
    with tempfile.TemporaryDirectory() as out:
        write_sample(refusals, out, n=n, seed=0)
        found, low, high = _overall(_labelled(Path(out), right))
    assert math.isclose(found, truth) and low <= truth <= high, (
        f"{big} refusals all right and {small} all wrong: the precision is {truth:.3f}, but "
        f"a sample of {n} reports {found:.3f} [{low:.3f}, {high:.3f}]. Weight each stratum by "
        "its size, not by how many of it the sample drew."
    )
