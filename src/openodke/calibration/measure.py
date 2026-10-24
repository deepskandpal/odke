"""Cards from per-item outcomes: a judge's answers beside a person's labels.

Provisional (DECISIONS #48), as `openodke.calibration` is: the card format and the bars settle
with the first published cards.

Nothing here calls a model. A caller runs the judge (`odke calibrate`,
`bench/calibrate.py`), and hands over what each labelled item got: the
person's label and the judge's answer in each run, or in each order. The same
arithmetic makes every card, so a card a user makes from their own labels
reads like the ones the package ships.

κ is Cohen's, over the items the judge decided, with a 95% interval from the
package's own bootstrap (`openodke.eval.stats.bootstrap_interval`, 2,000
resamples, seed 0): an item is one cell of the confusion matrix, so a resample
is a multinomial draw over the cells. Shares carry Wilson intervals
(`openodke.eval.stats.wilson`), which stay inside [0, 1] on the small samples
hand labels are.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from openodke.calibration.card import (
    BARS,
    Band,
    Bar,
    Card,
    Equivalence,
    Grounding,
    Kappa,
    LabelSet,
    Pairs,
    Share,
    Vote,
)
from openodke.eval.stats import bootstrap_interval, wilson

VERDICTS = ("supported", "contradicted", "not_found")
UNCHECKED = "unchecked"
# Passage lengths, in words, that a grounder's results are split by (roadmap §9.4,
# "length"): a sentence, a paragraph, and a document or more.
BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("sentence", 0, 40),
    ("paragraph", 41, 150),
    ("document", 151, None),
)
# Above this share of items changing verdict between runs, a single call is not
# stable enough to drop facts on, and "refuse only if 2 of 3 runs refuse" is
# measured beside it (roadmap §9.4, "non-determinism").
FLIP_LIMIT = 0.05
SELF_AGREEMENT_NOTE = (
    "judged on κ alone: the bar's other half, within 0.1 of the labeller's agreement with "
    "themselves, was not checked, because self-agreement has not been measured"
)


@dataclass(frozen=True)
class JudgedFact:
    """One labelled fact under a grounder: the person's verdict and the judge's, run by run.

    `label` is supported, contradicted or not_found; for a planted fact it is
    what the planting expects, and `planted` is set. Each of `runs` is the
    judge's verdict, or `unchecked` where it gave none. `words` is the length
    of the passage the judge read.
    """

    id: str
    label: str
    runs: tuple[str, ...]
    words: int = 0
    planted: bool = False


@dataclass(frozen=True)
class JudgedPair:
    """One labelled pair under a two-order judge: the person's `same`, and both answers.

    `forward` and `backward` are same, different, unsure, or None where an
    answer could not be read. For the pair judge they are (A, B) and (B, A);
    for the fact-equivalence judge, Fact 1 first and Fact 2 first.
    """

    id: str
    same: bool
    forward: str | None
    backward: str | None


@dataclass(frozen=True)
class Made:
    """What a card names besides its numbers: the prompt, the model, the labels."""

    prompt: str
    prompts: Mapping[str, str]
    model: str
    labels: LabelSet
    claim_format: str | None = None
    cost_per_1000: float | None = None
    date: str = field(default_factory=lambda: _dt.date.today().isoformat())
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def share(k: int, n: int) -> Share:
    """`k` of `n` with its Wilson 95% interval."""
    interval = wilson(k, n)
    if interval is None:
        return Share(k=k, n=n)
    return Share(k=k, n=n, value=k / n, low=interval[0], high=interval[1])


def kappa(pairs: Sequence[tuple[str, str]], classes: Sequence[str]) -> Kappa:
    """Cohen's κ between two raters' labels, each pair (person, judge), with its interval."""
    index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cells = [index[a] * k + index[b] for a, b in pairs]

    def statistic(totals: tuple[int | float, ...], n: int) -> float | None:
        return _kappa(totals, k)

    totals = tuple(cells.count(c) for c in range(k * k))
    value = _kappa(totals, k)
    if value is None or not cells:
        return Kappa(value=value, n=len(cells))
    vectors = [tuple(1 if c == cell else 0 for c in range(k * k)) for cell in cells]
    interval = bootstrap_interval(vectors, statistic)
    if interval is None:
        return Kappa(value=value, n=len(cells))
    return Kappa(
        value=value, low=min(interval[0], value), high=max(interval[1], value), n=len(cells)
    )


def _kappa(totals: Sequence[int | float], k: int) -> float | None:
    n = sum(totals)
    if not n:
        return None
    observed = sum(totals[i * k + i] for i in range(k)) / n
    rows = [sum(totals[i * k + j] for j in range(k)) / n for i in range(k)]
    cols = [sum(totals[j * k + i] for j in range(k)) / n for i in range(k)]
    expected = sum(r * c for r, c in zip(rows, cols, strict=True))
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def swap_rule(forward: str | None, backward: str | None) -> str:
    """same when both orders say same, different when both say different, else unsure."""
    if forward == backward and forward in ("same", "different"):
        return forward
    return "unsure"


def voted(runs: Sequence[str], vote: Vote) -> str:
    """The verdict a fact gets: its first run's, or refused only if 2 of 3 runs refuse.

    Under the vote a fact is refused when at least two runs refuse it, with the
    refusal most of them gave (contradicted on a tie); otherwise it is kept, as
    supported when any run supported it and unchecked when none decided.
    """
    if vote == "single":
        return runs[0]
    refusals = [r for r in runs if r in ("contradicted", "not_found")]
    if len(refusals) >= 2:
        counted = Counter(refusals)
        return max(("contradicted", "not_found"), key=lambda v: counted[v])
    return "supported" if "supported" in runs else UNCHECKED


def flips(runs: Sequence[Sequence[str]]) -> tuple[Share, Share] | None:
    """The share of items whose verdict changed between runs, and whose keep/refuse did."""
    if not runs or len(runs[0]) < 2:
        return None
    changed = sum(1 for r in runs if len(set(r)) > 1)
    turned = sum(1 for r in runs if len({_refused(v) for v in r}) > 1)
    return share(changed, len(runs)), share(turned, len(runs))


def _refused(verdict: str) -> bool:
    return verdict in ("contradicted", "not_found")


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


def grounding_card(
    items: Sequence[JudgedFact],
    made: Made,
    *,
    planted: Sequence[JudgedFact] = (),
    vote: Vote = "single",
    self_agreement: float | None = None,
) -> Card:
    """A grounder's card from its labelled facts, and the planted ones beside them.

    Keep is "supported"; refuse is "contradicted" or "not_found". κ is over the
    facts the judge decided; fail-recall, false refusals and the planted catch
    rate count an abstention as a fact kept, as the gate keeps it.
    """
    if not items:
        raise ValueError("a grounding card needs at least one labelled fact")
    runs = {len(item.runs) for item in (*items, *planted)}
    if len(runs) != 1:
        raise ValueError(f"every fact needs the same number of runs; got {sorted(runs)}")
    (count,) = runs
    if vote == "refuse_2_of_3" and count != 3:
        raise ValueError(f"the 2-of-3 vote reads three runs; got {count}")
    results = _grounding(items, vote)
    caught = None
    if planted:
        refused = sum(1 for p in planted if _refused(voted(p.runs, vote)))
        caught = share(refused, len(planted))
    found = flips([item.runs for item in items])
    by_length = []
    for name, low, high in BANDS:
        inside = [i for i in items if low <= i.words and (high is None or i.words <= high)]
        if inside:
            band = _grounding(inside, vote)
            by_length.append(
                Band(
                    name=name,
                    low=low,
                    high=high,
                    n=len(inside),
                    kappa=band.kappa,
                    fail_recall=band.fail_recall,
                    false_refusals=band.false_refusals,
                )
            )
    grounding = Grounding(
        kappa=results.kappa,
        kappa_three_way=results.kappa_three_way,
        confusion=results.confusion,
        fail_recall=results.fail_recall,
        false_refusals=results.false_refusals,
        planted=caught,
        abstain=results.abstain,
        flips=found[0] if found else None,
        flips_keep_refuse=found[1] if found else None,
        runs=count,
        vote=vote,
        by_length=tuple(by_length),
    )
    value = grounding.kappa.value
    note = SELF_AGREEMENT_NOTE if self_agreement is None else None
    met = value is not None and value >= BARS["ground"][1]
    if self_agreement is not None and value is not None:
        met = met and value >= self_agreement - 0.1
    return _card(
        "ground",
        made,
        vote=vote,
        grounding=grounding,
        bar=Bar(
            metric=BARS["ground"][0], threshold=BARS["ground"][1], value=value, met=met, note=note
        ),
        self_agreement=self_agreement,
    )


@dataclass(frozen=True)
class _Results:
    kappa: Kappa
    kappa_three_way: Kappa
    confusion: dict[str, dict[str, int]]
    fail_recall: Share
    false_refusals: Share
    abstain: Share


def _grounding(items: Sequence[JudgedFact], vote: Vote) -> _Results:
    pairs = [(item.label, voted(item.runs, vote)) for item in items]
    decided = [(p, j) for p, j in pairs if j != UNCHECKED]
    keep = [
        ("keep" if p == "supported" else "refuse", "keep" if j == "supported" else "refuse")
        for p, j in decided
    ]
    unsupported = [j for p, j in pairs if p != "supported"]
    supported = [j for p, j in pairs if p == "supported"]
    confusion = {p: dict.fromkeys((*VERDICTS, UNCHECKED), 0) for p in VERDICTS}
    for p, j in pairs:
        confusion[p][j] += 1
    return _Results(
        kappa=kappa(keep, ("keep", "refuse")),
        kappa_three_way=kappa(decided, VERDICTS),
        confusion=confusion,
        fail_recall=share(sum(1 for j in unsupported if _refused(j)), len(unsupported)),
        false_refusals=share(sum(1 for j in supported if _refused(j)), len(supported)),
        abstain=share(sum(1 for _, j in pairs if j == UNCHECKED), len(pairs)),
    )


def pair_card(
    items: Sequence[JudgedPair],
    made: Made,
    *,
    audited: int = 0,
    label_noise: Share | None = None,
) -> Card:
    """The pair judge's card: the precision of "same" under the swap rule, and its lower bound."""
    if not items:
        raise ValueError("a pair card needs at least one labelled pair")
    decided = [(item.same, swap_rule(item.forward, item.backward)) for item in items]
    said_same = [same for same, d in decided if d == "same"]
    truly_same = [d for same, d in decided if same]
    precision = share(sum(1 for same in said_same if same), len(said_same))
    confusion = {
        label: {d: 0 for d in ("same", "different", "unsure")} for label in ("same", "different")
    }
    for same, d in decided:
        confusion["same" if same else "different"][d] += 1
    pairs = Pairs(
        precision_same=precision,
        recall_same=share(sum(1 for d in truly_same if d == "same"), len(truly_same)),
        unsure=share(sum(1 for _, d in decided if d == "unsure"), len(decided)),
        order_disagreement=share(sum(1 for i in items if i.forward != i.backward), len(items)),
        confusion=confusion,
        audited=audited,
        label_noise=label_noise,
    )
    metric, threshold = BARS["pair"]
    low = precision.low
    return _card(
        "pair",
        made,
        pairs=pairs,
        bar=Bar(
            metric=metric, threshold=threshold, value=low, met=low is not None and low >= threshold
        ),
    )


def equivalence_card(
    items: Sequence[JudgedPair], made: Made, *, lenient_minus_strict: float | None = None
) -> Card:
    """The fact-equivalence judge's card: κ on same against not, under the swap rule."""
    if not items:
        raise ValueError("an equivalence card needs at least one labelled pair")
    decided = [(item.same, swap_rule(item.forward, item.backward)) for item in items]
    as_labels = [
        ("same" if same else "not", "same" if d == "same" else "not") for same, d in decided
    ]
    confusion = {
        label: {d: 0 for d in ("same", "different", "unsure")} for label in ("same", "different")
    }
    for same, d in decided:
        confusion["same" if same else "different"][d] += 1
    found = kappa(as_labels, ("same", "not"))
    equivalence = Equivalence(
        kappa=found,
        unsure=share(sum(1 for _, d in decided if d == "unsure"), len(decided)),
        order_disagreement=share(sum(1 for i in items if i.forward != i.backward), len(items)),
        judge_same=share(sum(1 for _, d in decided if d == "same"), len(decided)),
        person_same=share(sum(1 for same, _ in decided if same), len(decided)),
        confusion=confusion,
        lenient_minus_strict=lenient_minus_strict,
    )
    metric, threshold = BARS["fact_equiv"]
    value = found.value
    return _card(
        "fact_equiv",
        made,
        equivalence=equivalence,
        bar=Bar(
            metric=metric,
            threshold=threshold,
            value=value,
            met=value is not None and value >= threshold,
        ),
    )


def _card(judge: Literal["ground", "pair", "fact_equiv"], made: Made, **parts: object) -> Card:
    return Card(
        judge=judge,
        prompt=made.prompt,
        prompts=dict(made.prompts),
        model=made.model,
        claim_format=made.claim_format,
        date=made.date,
        openodke=importlib.metadata.version("openodke"),
        labels=made.labels,
        cost_per_1000=made.cost_per_1000,
        notes=made.notes,
        **parts,  # type: ignore[arg-type]
    )


__all__ = [
    "BANDS",
    "FLIP_LIMIT",
    "SELF_AGREEMENT_NOTE",
    "UNCHECKED",
    "VERDICTS",
    "JudgedFact",
    "JudgedPair",
    "Made",
    "equivalence_card",
    "flips",
    "grounding_card",
    "kappa",
    "pair_card",
    "share",
    "swap_rule",
    "voted",
]
