"""A calibration card: one judge prompt on one dated model, measured against human labels.

A judge's number is only as good as its agreement with a person. A card holds
that agreement for one prompt version (DECISIONS #27), one dated model id
(#7a) and one version of one label set, and nothing else: change any of the
three and the card no longer describes the judge that runs, so it expires
(roadmap §9.4). Cards ship in the package (`cards/`, beside this file); the
labels they were measured on never do.

Each card names its primary metric and its bar, and whether the bar is met:

- a grounder (P1, its challengers, P2): Cohen's κ on keep against refuse, at
  least 0.6. Fail-recall (the share of facts a person did not mark supported
  that the judge refuses) and false refusals (the share of facts a person
  marked supported that it refuses) sit beside it, each with a Wilson
  interval, then the planted-error catch rate, the abstain rate, the rerun
  flip rate and the results by passage length;
- the pair judge (P4): the precision of "same", whose lower 95% bound must be
  at least 0.9, beside the recall of "same", the unsure rate and the rate at
  which the two orders disagree;
- the fact-equivalence judge (P5): κ on same against not, at least 0.6.

Raw agreement is never reported alone: 88% agreement can be κ 0.27 (D01 §5c).

The grounder's bar has a second half, "within 0.1 of the labeller's agreement
with themselves". Until that is measured, `self_agreement` is None and the bar
is judged on κ alone, which the card's `bar.note` says.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# A card's format, major.minor: a minor version adds fields, a major one changes them.
CARD_VERSION = "1.0"

Judge = Literal["ground", "pair", "fact_equiv"]
Vote = Literal["single", "refuse_2_of_3"]

# Each judge's primary metric, and the bar a card must clear to make it a default.
BARS: dict[str, tuple[str, float]] = {
    "ground": ("kappa_keep_refuse", 0.6),
    "pair": ("precision_same_low", 0.9),
    "fact_equiv": ("kappa_same", 0.6),
}
UNCALIBRATED = "UNCALIBRATED"


class _Read(BaseModel):
    """A card's parts: frozen, and tolerant of fields a newer minor version adds."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class Share(_Read):
    """`k` of `n`, with its Wilson 95% interval. `value` and the interval are None at n 0."""

    k: int = Field(ge=0)
    n: int = Field(ge=0)
    value: float | None = None
    low: float | None = None
    high: float | None = None

    def text(self) -> str:
        if self.value is None:
            return "n/a (n 0)"
        return f"{self.value:.2f} [{self.low:.2f}, {self.high:.2f}] ({self.k}/{self.n})"


class Kappa(_Read):
    """Cohen's κ over `n` decided items, with its 95% bootstrap interval; None where undefined."""

    value: float | None = None
    low: float | None = None
    high: float | None = None
    n: int = Field(default=0, ge=0)

    def text(self) -> str:
        if self.value is None:
            return f"undefined (n {self.n})"
        if self.low is None or self.high is None:
            return f"{self.value:.2f} (n {self.n})"
        return f"{self.value:.2f} [{self.low:.2f}, {self.high:.2f}] (n {self.n})"


class LabelSet(_Read):
    """The labels a card was measured on: which set, which version, which split, how many.

    `sha256` is over the labels scored, by item id (`label_hash`), so a label
    that changes changes the hash whatever file it lives in.
    """

    name: str
    version: str
    sha256: str
    split: str
    n: int = Field(ge=0)


class Bar(_Read):
    """The primary metric, its threshold, its value, and whether the card clears it."""

    metric: str
    threshold: float
    value: float | None = None
    met: bool = False
    note: str | None = None


class Band(_Read):
    """A grounder's results on passages of one length: `words` from `low` to `high`."""

    name: str
    low: int
    high: int | None
    n: int = Field(ge=0)
    kappa: Kappa
    fail_recall: Share
    false_refusals: Share


class Grounding(_Read):
    """A grounder's results against G-format labels (supported, contradicted, not found).

    `confusion` is the person's verdict, then the judge's, then the count; the
    judge's include `unchecked`, its abstentions. `vote` says which verdict was
    read: the first run's, or "refuse only if 2 of 3 runs refuse".
    """

    kappa: Kappa
    kappa_three_way: Kappa
    confusion: dict[str, dict[str, int]]
    fail_recall: Share
    false_refusals: Share
    planted: Share | None = None
    abstain: Share
    flips: Share | None = None
    flips_keep_refuse: Share | None = None
    runs: int = Field(ge=1)
    vote: Vote = "single"
    by_length: tuple[Band, ...] = ()


class Pairs(_Read):
    """The pair judge's results against same/different labels, both orders asked.

    `decided` is the swap rule's: "same" only when both orders say same.
    `label_noise` is how often a person's audit disagreed with the dataset's
    own labels, which the labels here were corrected with where audited.
    """

    precision_same: Share
    recall_same: Share
    unsure: Share
    order_disagreement: Share
    confusion: dict[str, dict[str, int]]
    audited: int = Field(default=0, ge=0)
    label_noise: Share | None = None


class Equivalence(_Read):
    """The fact-equivalence judge's results against a person's same/different labels.

    `judge_same` and `person_same` are the shares each calls the same fact: the
    lenient score's bias in small, since a pair judged the same is a hit there.
    `lenient_minus_strict` is the gap it opens on a scored run, where one was given.
    """

    kappa: Kappa
    unsure: Share
    order_disagreement: Share
    judge_same: Share
    person_same: Share
    confusion: dict[str, dict[str, int]]
    lenient_minus_strict: float | None = None


class Card(_Read):
    """One prompt version on one dated model, against one version of one label set."""

    card_version: str = CARD_VERSION
    judge: Judge
    prompt: str
    # Every registered prompt the calls sent, by key, with its SHA-256 (DECISIONS #27).
    prompts: dict[str, str]
    model: str
    # How the fact was written: inline, labelled, sentence or triple; None for P4 and P5.
    claim_format: str | None = None
    vote: Vote = "single"
    date: str
    openodke: str
    labels: LabelSet
    cost_per_1000: float | None = None
    grounding: Grounding | None = None
    pairs: Pairs | None = None
    equivalence: Equivalence | None = None
    bar: Bar
    # The labeller's agreement with themselves: None until it is measured.
    self_agreement: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """The card's file name, without `.json`: prompt, model, claim format, vote."""
        parts = [self.prompt, _slug(self.model), self.claim_format or "-"]
        if self.vote != "single":
            parts.append(self.vote)
        return "__".join(parts)

    @property
    def meets_bar(self) -> bool:
        return self.bar.met

    def line(self) -> str:
        """The one line a run report prints for this judge."""
        return f"{_who(self.prompt, self.model, self.claim_format)}: {self.summary()}"

    def summary(self) -> str:
        bar = BAR_TEXT[self.judge]
        standing = f"meets its bar ({bar})" if self.bar.met else f"below its bar ({bar})"
        where = f"label set {self.labels.name} {self.labels.split}, n {self.labels.n}, {self.date}"
        if self.grounding is not None:
            g = self.grounding
            head = f"κ {g.kappa.text()}"
            rest = f"fail-recall {_v(g.fail_recall)}, false refusals {_v(g.false_refusals)}" + (
                f", planted caught {_v(g.planted)}" if g.planted is not None else ""
            )
        elif self.pairs is not None:
            p = self.pairs
            low = p.precision_same.low
            head = f"precision of same {_v(p.precision_same)}, low {_f(low)}"
            rest = f"recall {_v(p.recall_same)}, unsure {_v(p.unsure)}"
        elif self.equivalence is not None:
            e = self.equivalence
            head = f"κ {e.kappa.text()}"
            rest = f"unsure {_v(e.unsure)}, orders disagree {_v(e.order_disagreement)}"
        else:
            head, rest = "no results", ""
        return f"{head}, {standing}; {rest} ({where})" if rest else f"{head}, {standing}"

    def render(self) -> str:
        """Every number on the card, as `odke calibrate` prints it."""
        lines = [f"calibration card  {self.id}", f"  {self.line()}"]
        labels = self.labels
        lines.append(
            f"  labels      {labels.name} v{labels.version}, split {labels.split}, n {labels.n}, "
            f"sha256 {labels.sha256[:12]}"
        )
        lines.append(
            f"  prompts     {', '.join(f'{k} ({v[:12]})' for k, v in self.prompts.items())}"
        )
        bar = self.bar
        lines.append(
            f"  bar         {bar.metric} ≥ {bar.threshold:g}: {_f(bar.value)}, "
            + ("met" if bar.met else "not met")
            + (f" ({bar.note})" if bar.note else "")
        )
        if self.grounding is not None:
            g = self.grounding
            lines += [
                f"  κ           keep vs refuse {g.kappa.text()}",
                f"  κ 3-way     {g.kappa_three_way.text()}",
                f"  fail-recall {g.fail_recall.text()}",
                f"  false ref.  {g.false_refusals.text()}",
                f"  planted     {g.planted.text() if g.planted else 'not measured'}",
                f"  abstained   {g.abstain.text()}",
                f"  flips       {g.flips.text() if g.flips else 'one run'} over {g.runs} run(s)"
                + (f", vote {g.vote}" if g.vote != "single" else ""),
            ]
            for band in g.by_length:
                span = f"{band.low}-{band.high}" if band.high is not None else f"{band.low}+"
                lines.append(
                    f"  {band.name:<11} {span} words, n {band.n}: κ {band.kappa.text()}, "
                    f"fail-recall {_v(band.fail_recall)}, false refusals {_v(band.false_refusals)}"
                )
            lines.append(f"  confusion   {_confusion(g.confusion)}")
        if self.pairs is not None:
            p = self.pairs
            lines += [
                f"  precision   same {p.precision_same.text()}",
                f"  recall      same {p.recall_same.text()}",
                f"  unsure      {p.unsure.text()}",
                f"  orders      disagree {p.order_disagreement.text()}",
                f"  audit       {p.audited} labels from the audit"
                + (f", label noise {p.label_noise.text()}" if p.label_noise else ""),
                f"  confusion   {_confusion(p.confusion)}",
            ]
        if self.equivalence is not None:
            e = self.equivalence
            gap = e.lenient_minus_strict
            lines += [
                f"  κ           same vs not {e.kappa.text()}",
                f"  unsure      {e.unsure.text()}",
                f"  orders      disagree {e.order_disagreement.text()}",
                f"  same        judge {e.judge_same.text()}; person {e.person_same.text()}",
                f"  lenient     minus strict {'not measured' if gap is None else f'{gap:+.3f}'}",
                f"  confusion   {_confusion(e.confusion)}",
            ]
        cost = "unknown" if self.cost_per_1000 is None else f"${self.cost_per_1000:.4f}"
        lines.append(f"  cost        {cost} per 1,000 items")
        lines.append(
            "  self-agree  "
            + ("not measured" if self.self_agreement is None else f"{self.self_agreement:.2f}")
        )
        lines += [f"  - {note}" for note in self.notes]
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        """The card as JSON at `path`, a `.json` file; any other path is a directory for it."""
        target = Path(path)
        if target.is_dir() or target.suffix != ".json":
            target = target / f"{self.id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return target


BAR_TEXT = {
    "ground": "κ ≥ 0.6",
    "pair": "lower 95% bound of the precision of same ≥ 0.9",
    "fact_equiv": "κ ≥ 0.6",
}


def load_card(path: str | Path) -> Card:
    """A card written by `Card.write`. A card of another major version is refused by name."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("card_version") if isinstance(data, dict) else None
    if not isinstance(version, str) or version.split(".")[0] != CARD_VERSION.split(".")[0]:
        raise ValueError(f"{path}: card_version {version!r}; this package reads {CARD_VERSION}")
    return Card.model_validate(data)


def label_hash(labels: Iterable[tuple[str, Any]]) -> str:
    """The SHA-256 of `(item id, label)` pairs, sorted by id: a label set's content hash."""
    rows = sorted([str(item), label] for item, label in labels)
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def uncalibrated_line(prompt: str, model: str | None, claim_format: str | None = None) -> str:
    """What a report prints for a judge with no card."""
    return f"{_who(prompt, model, claim_format)}: {UNCALIBRATED} (no card)"


def _who(prompt: str, model: str | None, claim_format: str | None) -> str:
    shown = f"{prompt} on {model or 'an unnamed model'}"
    return f"{shown} ({claim_format})" if claim_format else shown


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model)


def _v(share: Share | None) -> str:
    return "n/a" if share is None or share.value is None else f"{share.value:.2f}"


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _confusion(table: Mapping[str, Mapping[str, int]]) -> str:
    return "; ".join(
        f"{person}: " + ", ".join(f"{judge} {n}" for judge, n in row.items() if n)
        for person, row in table.items()
    )


__all__ = [
    "BARS",
    "BAR_TEXT",
    "CARD_VERSION",
    "UNCALIBRATED",
    "Band",
    "Bar",
    "Card",
    "Equivalence",
    "Grounding",
    "Judge",
    "Kappa",
    "LabelSet",
    "Pairs",
    "Share",
    "Vote",
    "label_hash",
    "load_card",
    "uncalibrated_line",
]
