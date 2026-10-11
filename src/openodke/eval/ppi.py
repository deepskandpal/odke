"""Precision with no gold: a judge grades every fact, a person a random sample (#144).

On your own documents there is no gold to score against. A judge, the
grounder or any other, can still grade every fact a pipeline wrote, and the
share it calls `supported` is a precision. It is the judge's precision,
though, not the facts': a lenient judge reads high and a strict one low, and
raw agreement on a sample does not say which way. So a person labels a small
random sample, and **prediction-powered inference** corrects the judge's
number by the mean gap between the labels and the judge on that sample
(PPI: Angelopoulos et al., *Science* 382, 2023, arXiv 2301.09633; ARES uses it
to correct an LLM judge, Saad-Falcon et al. 2023, arXiv 2311.09476):

    corrected = (1/N) Σᵢ fᵢ + (1/n) Σⱼ (yⱼ − fⱼ)

`f` is 1 where the judge said supported and `y` where the person did; `i`
runs over the N judged facts and `j` over the n labelled ones. Whatever the
judge's bias, the correction removes it on average, and the interval is
narrower than the labels' own whenever the judge mostly agrees with them.

**The interval is PPI's closed form for a mean**, normal at the given level.
Here the labelled facts are drawn from the judged ones, so the two terms share
them, and the variance is

    Var(y)/N + Var(y − f)·(1/n − 1/N)

which is the paper's Var(f)/N + Var(y − f)/n with that overlap taken out. It
is exact when every fact is labelled: the corrected number is then the labels'
mean, with the labels' variance. Both variances are read off the labelled
sample. **Labels only** is the classical interval on the same labels,
ȳ ± z·s/√n, and **judge only** is the judge's share with no interval at all.

Facts are treated as independent here, where the report's ranges resample
documents (`openodke.eval.bootstrap`). The sample is drawn fact by fact, so
few labelled facts share a document, and the judged set's own term is a small
part of the variance. A corpus of a few long documents is where this
interval would be too narrow.

**Under 100 labels the corrected number is an uncalibrated estimate**, and
the report says so: a normal interval on fewer labels is not one to rely on.

`sample` draws the labels' facts, seeded, and `write_sample` writes them as
`odke label` sheets. Stratified by predicate, each predicate gets its share of
the sample, rounded by largest remainder. The sample then stays
self-weighting, and the estimator reads it as a simple random one, which
makes the interval a little wide if anything.

**Recall is never claimed.** Without gold nobody knows what was missed. The
coverage report (`openodke.coverage`) stands in for it: the sentences naming
two known entities that no fact covers, and the entities no fact names.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import NormalDist, variance

from openodke.eval.bootstrap import LEVEL, SEED
from openodke.eval.eval_report import (
    CoverageTotals,
    Dataset,
    Estimate,
    EvalReport,
    JudgedPrecision,
    Run,
)
from openodke.eval.formats import GroundingLabel
from openodke.eval.grounding import evaluate_grounding
from openodke.eval.report import StageReport
from openodke.eval.sheets import Made, make_sheets
from openodke.ontology import Ontology
from openodke.types import Document, Fact, GroundingVerdict

# The sample `odke eval precision --make-sheet` draws, and the fewest labels
# whose corrected number the report calls calibrated.
SAMPLE = 150
MINIMUM_LABELS = 100
SAMPLE_FILE = "sample.jsonl"
TITLE = "precision"

RECALL = "recall is not claimed: with no gold, nothing says what was missed"

DESCRIPTION = f"""\
odke eval precision --facts JUDGED [--labels LABELS] [--documents DOCS] [--ontology FILE]
odke eval precision --facts JUDGED --make-sheet DIR --documents DOCS [--n {SAMPLE}]
                    [--seed 0] [--by-predicate]

Precision with no gold. JUDGED is a run's facts after a judge graded them: a
facts.jsonl, or the directory a sink wrote, whose `verdict` the grounder set.
A fact the judge called supported counts as correct; any other verdict as
wrong.

--make-sheet draws a random sample of the facts (--n, seeded by --seed;
--by-predicate gives each predicate its share) and writes it to DIR as
`odke label` grounding sheets, with the drawn rows in DIR/{SAMPLE_FILE}.
Tick them, read them back with `odke label read DIR -o labels.jsonl`, and
pass the labels here as --labels: they join the facts on Fact.id.

The report shows three numbers: the judge's own precision, the precision
corrected by prediction-powered inference (Angelopoulos et al. 2023) with its
95% interval, and the labels' own precision and interval. Under
{MINIMUM_LABELS} labels the corrected number is an uncalibrated estimate.
Recall is never claimed; with --documents the coverage report stands in."""


# --------------------------------------------------------------------------- #
# The estimate
# --------------------------------------------------------------------------- #


def prediction_powered(
    judged: Sequence[bool], labelled: Sequence[tuple[bool, bool]], *, level: float = LEVEL
) -> tuple[Estimate, Estimate, Estimate]:
    """The judge's share, the PPI-corrected share and the labels' share, as estimates.

    `judged` holds the judge's verdict on every fact, true for supported.
    `labelled` holds `(judge, label)` for each fact a person labelled, drawn at
    random from the judged ones. The corrected and label shares carry their
    normal intervals at `level`, clipped to [0, 1], and so does the corrected
    number itself; with one label there is no variance and no interval, and
    with none, no number.
    """
    if not 0 < level < 1:
        raise ValueError(f"level is a share between 0 and 1, got {level}")
    big, small = len(judged), len(labelled)
    if small > big:
        raise ValueError(
            f"{small} labelled facts but only {big} judged: the labels are a sample of the facts"
        )
    judge = sum(map(bool, judged)) / big if big else None
    if judge is None or not small:
        return Estimate(value=judge), Estimate(value=None), Estimate(value=None)
    truth = [float(y) for _, y in labelled]
    gap = [float(y) - float(f) for f, y in labelled]
    mean_truth, mean_gap = sum(truth) / small, sum(gap) / small
    corrected = judge + mean_gap
    if small < 2:
        return Estimate(value=judge), _clipped(corrected, None), _clipped(mean_truth, None)
    z = NormalDist().inv_cdf((1 + level) / 2)
    spread_truth, spread_gap = variance(truth), variance(gap)
    ppi = spread_truth / big + spread_gap * (1 / small - 1 / big)
    classical = spread_truth / small
    return (
        Estimate(value=judge),
        _clipped(corrected, z * math.sqrt(max(ppi, 0.0))),
        _clipped(mean_truth, z * math.sqrt(classical)),
    )


def _clipped(value: float, half: float | None) -> Estimate:
    def clip(x: float) -> float:
        return min(1.0, max(0.0, x))

    if half is None:
        return Estimate(value=clip(value))
    return Estimate(value=clip(value), low=clip(value - half), high=clip(value + half))


def judged_precision(
    facts: Sequence[Fact],
    labels: Sequence[GroundingLabel] = (),
    *,
    documents: Sequence[Document] | None = None,
    ontology: Ontology | None = None,
    level: float = LEVEL,
    minimum: int = MINIMUM_LABELS,
) -> tuple[JudgedPrecision, list[str]]:
    """The judged facts, corrected by the labels joined to them on `Fact.id`, and notes.

    A label whose fact is not among `facts` is left out and counted; so is a
    second label on one fact. With `documents`, the coverage report over the
    facts comes too, with `ontology`'s relations when one is given.
    """
    ids = Counter(fact.id for fact in facts)
    if shared := [i for i, n in ids.items() if n > 1]:
        raise ValueError(
            f"{len(shared)} fact id(s) appear more than once, e.g. {shared[0]}: the labels "
            "join the facts on Fact.id, so each must be one fact"
        )
    said = {fact.id: fact.verdict is GroundingVerdict.SUPPORTED for fact in facts}
    pairs: list[tuple[bool, bool]] = []
    seen: set[str] = set()
    strays = repeats = 0
    for row in labels:
        if row.fact.id not in said:
            strays += 1
        elif row.fact.id in seen:
            repeats += 1
        else:
            seen.add(row.fact.id)
            pairs.append((said[row.fact.id], row.verdict == "supported"))
    judge, corrected, alone = prediction_powered(list(said.values()), pairs, level=level)
    verdicts = Counter(fact.verdict.value for fact in facts)
    order = ("supported", "contradicted", "not_found", "unchecked")
    notes = [
        "judged: "
        + ", ".join(f"{v} {verdicts.get(v, 0)}" for v in order)
        + "; only supported counts as correct"
    ]
    if verdicts.get(GroundingVerdict.UNCHECKED.value):
        notes.append(
            f"{verdicts[GroundingVerdict.UNCHECKED.value]} fact(s) were never judged and count "
            "as wrong in the judge's number; the labels correct for that too"
        )
    if strays:
        notes.append(f"{strays} label(s) name no judged fact and were left out")
    if repeats:
        notes.append(f"{repeats} label(s) repeat a fact already labelled and were left out")
    if pairs and all(f == y for f, y in pairs):
        notes.append(
            "the judge agreed with every label, so the correction is zero and the "
            "interval is only the labels' own spread over the judged facts"
        )
    coverage = None
    if documents is not None:
        coverage = coverage_totals(documents, facts, ontology)
        notes.append(f"{RECALL}; the coverage report stands in for it")
    else:
        notes.append(f"{RECALL}; with the documents, the coverage report stands in for it")
    return (
        JudgedPrecision(
            facts=len(facts),
            supported=sum(said.values()),
            labels=len(pairs),
            labelled_supported=sum(y for _, y in pairs),
            false_support=sum(f and not y for f, y in pairs),
            lost_support=sum(y and not f for f, y in pairs),
            judge_only=judge,
            corrected=corrected,
            labels_only=alone,
            level=level,
            minimum=minimum,
            calibrated=len(pairs) >= minimum,
            coverage=coverage,
        ),
        notes,
    )


def coverage_totals(
    documents: Sequence[Document], facts: Sequence[Fact], ontology: Ontology | None = None
) -> CoverageTotals:
    """`openodke.coverage.measure` over `facts`, as the report's totals."""
    from openodke.coverage import measure

    found = measure(documents, facts, ontology if ontology is not None else Ontology())
    return CoverageTotals(
        documents=len(found.documents),
        sentences=found.sentences,
        uncovered=found.uncovered,
        missed_entities=found.missed,
        not_offered=found.not_offered,
        unused=found.unused,
    )


def report_precision(
    facts: Sequence[Fact],
    labels: Sequence[GroundingLabel] = (),
    *,
    documents: Sequence[Document] | None = None,
    ontology: Ontology | None = None,
    dataset: Dataset | None = None,
    level: float = LEVEL,
) -> EvalReport:
    """`judged_precision` as the eval report, with the judge scored on the labels beside it.

    The stage beside the section is `evaluate_grounding` over the labels and
    the judged facts: the judge's accuracy and confusion on the sample, the
    same arithmetic `odke eval ground` prints.
    """
    section, notes = judged_precision(
        facts, labels, documents=documents, ontology=ontology, level=level
    )
    stages: tuple[StageReport, ...] = ()
    if section.labels:
        joined = {fact.id: fact for fact in facts}
        # The labels the section used: each judged fact's first, as `judged_precision` keeps it.
        first: dict[str, GroundingLabel] = {}
        for row in labels:
            if row.fact.id in joined:
                first.setdefault(row.fact.id, row)
        kept = list(first.values())
        stages = (
            evaluate_grounding(kept, [joined[row.fact.id] for row in kept]).model_copy(
                update={"stage": "the judge on the labels"}
            ),
        )
    return EvalReport(
        title=TITLE,
        n=len(facts),
        run=Run(dataset=dataset),
        stages=stages,
        notes=tuple(notes),
        judged_precision=section,
    )


# --------------------------------------------------------------------------- #
# The sample
# --------------------------------------------------------------------------- #


def sample(
    facts: Sequence[Fact],
    n: int = SAMPLE,
    *,
    seed: int = SEED,
    by: Callable[[Fact], str] | None = None,
) -> list[Fact]:
    """`n` of `facts` drawn at random without replacement, in the order they came.

    Seeded, and drawn with `random.Random.random()` alone, whose sequence for a
    given seed Python keeps across versions, so the same facts and seed draw
    the same sample anywhere. `by` stratifies: each group (a predicate, say)
    gets its share of `n`, rounded by largest remainder with ties to the group
    named first, so the sample keeps the facts' proportions. Asking for at
    least as many as there are draws them all.
    """
    if n < 1:
        raise ValueError(f"a sample needs at least one fact, got n={n}")
    if n >= len(facts):
        return list(facts)
    rng = random.Random(seed)
    if by is None:
        return [facts[i] for i in sorted(_pick(rng, range(len(facts)), n))]
    groups: dict[str, list[int]] = {}
    for index, fact in enumerate(facts):
        groups.setdefault(by(fact), []).append(index)
    names = sorted(groups)
    exact = {name: n * len(groups[name]) / len(facts) for name in names}
    quota = {name: math.floor(exact[name]) for name in names}
    left = n - sum(quota.values())
    for name in sorted(names, key=lambda g: (-(exact[g] - quota[g]), g))[:left]:
        quota[name] += 1
    chosen = [i for name in names for i in _pick(rng, groups[name], quota[name])]
    return [facts[i] for i in sorted(chosen)]


def _pick(rng: random.Random, pool: Sequence[int], k: int) -> list[int]:
    """`k` of `pool` without replacement: the first `k` steps of a Fisher–Yates shuffle."""
    items = list(pool)
    for i in range(k):
        j = i + min(int(rng.random() * (len(items) - i)), len(items) - i - 1)
        items[i], items[j] = items[j], items[i]
    return items[:k]


def predicate(fact: Fact) -> str:
    """What `--by-predicate` stratifies on."""
    return fact.predicate


def write_sample(
    facts: Sequence[Fact],
    documents: Sequence[Document],
    out: str | Path,
    *,
    n: int = SAMPLE,
    seed: int = SEED,
    by_predicate: bool = False,
    per_sheet: int = 50,
) -> tuple[list[Fact], Made]:
    """Draw the sample, and write it to `out` as grounding sheets to label by hand.

    Every judged fact must cite one of `documents`, since any of them may be
    drawn and a sheet shows the text a fact cites. The drawn facts go out with
    their verdict reset to unchecked and their confidence to zero, so nothing
    on a sheet or beside it shows what the judge said. `out/sample.jsonl`
    holds the rows the sheets were made from. A directory that already holds a
    sample or sheets is refused: those may be hours of ticks.
    """
    target = Path(out)
    if target.is_dir() and ((target / SAMPLE_FILE).exists() or any(target.glob("sheet-*"))):
        raise ValueError(f"{target} already has a sample or sheets; write to an empty directory")
    texts = {doc.id: doc for doc in documents}
    homes: dict[str, Document] = {}
    homeless = []
    for fact in facts:
        doc = next((texts[e.doc_id] for e in fact.evidence if e.doc_id in texts), None)
        if doc is None:
            homeless.append(fact.id)
        else:
            homes[fact.id] = doc
    if homeless:
        raise ValueError(
            f"{len(homeless)} judged fact(s) cite none of the documents, e.g. {homeless[0]}: "
            "any fact may be drawn, and a sheet shows the text it cites"
        )
    drawn = sample(facts, n, seed=seed, by=predicate if by_predicate else None)
    rows = []
    for fact in drawn:
        blind = fact.model_copy(update={"verdict": GroundingVerdict.UNCHECKED, "confidence": 0.0})
        doc = homes[fact.id]
        rows.append({"text": doc.text, "fact": blind.model_dump(mode="json"), "doc_id": doc.id})
    target.mkdir(parents=True, exist_ok=True)
    items = target / SAMPLE_FILE
    items.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return drawn, make_sheets("grounding", items, target, per_sheet=per_sheet)


__all__ = [
    "DESCRIPTION",
    "MINIMUM_LABELS",
    "SAMPLE",
    "SAMPLE_FILE",
    "coverage_totals",
    "judged_precision",
    "prediction_powered",
    "predicate",
    "report_precision",
    "sample",
    "write_sample",
]
