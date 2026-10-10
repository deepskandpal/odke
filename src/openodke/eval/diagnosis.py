"""Where a pipeline loses facts: every miss in one cause bucket (#140).

A recall of 14% says how much was lost and nothing about where. The run itself
says more: what the extractor was offered, what the gate refused, which
predictions came near a gold fact, how long each document was. So every gold
fact the scorer counted missed goes into exactly one bucket, the first of these
its evidence fits (DECISIONS #38):

1. **relation never offered**: the trace says the extractor was never shown the
   relation, for the subject's type;
2. **refused by the Validator**: a prediction matching it was made, and the gate
   refused it;
3. **wrong relation**: a prediction links the same pair with another relation;
4. **inverse direction**: a prediction links the pair the other way round, with
   the same relation or its inverse (`Ontology.inverses`, DECISIONS #28);
5. **surface form**: a prediction has the relation and one end, and the other
   end is a near name;
6. **same triple, scored apart**: a prediction states the triple, but it is
   typed, negated or qualified otherwise, or the scorer paired it with another
   gold fact (one prediction finds one);
7. **cross-sentence**: nothing came near, and the evidence spans sentences;
8. **output saturation**: nothing came near, the run's output is flat against
   length, and the document is a long one;
9. **entity never extracted**: an end is in no prediction for the document;
10. **both seen but not linked**: both ends are, never in one prediction.

Three tiers, in order: what the pipeline did (1–6), the condition the fact or
its document was in (7–8), and what is merely missing (9–10). A bucket that says
what the pipeline did with a fact beats one that describes the fact, and one
that describes it beats one that only names the symptom: every miss whose
evidence fits nothing earlier is, trivially, an entity never extracted or a pair
never linked.

- **Predictions** are everything the pipeline produced: the facts it wrote and,
  where the gate's record is given, the ones it refused. An entity in a refused
  fact was extracted.
- **Surface form** needs the fact-equivalence judge (#143). Here is the
  deterministic pre-filter that makes its candidates, and an `EquivalenceJudge`
  hook. With no judge the bucket is *surface form (unconfirmed)*; with one, a
  candidate it rejects falls through to the buckets after.
- **Cross-sentence** reads the dataset's evidence sentences where it gives them,
  Re-DocRED's; otherwise the two ends are named in the text, by the span
  locator's rules, and never in one sentence together.
- **Output saturation** is a run-level signal. The extraction count per document
  is regressed on length, log on log, and so is the gold count. When the
  output grows at less than half the gold's rate, the run is saturated, and the
  misses of its documents longer than the median land here.

The precision side is counted too: every false positive before the gate, split
into the ones the gate refused, the ones written that their passage supports
(possibly true and missing from the gold, which #145 adjudicates through the
`Adjudicator` hook), and the rest.

The input is a `View`: gold facts and predictions as one scorer compares them.
`gold_view` makes one from `GoldFact` labels with `match_extraction`'s
matching, and a benchmark's own scorer makes its own (Re-DocRED's
`diagnosis_view`), so every bucket count sums to the report's misses, to the
fact.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Literal, Protocol, runtime_checkable

from openodke.chunking import sentences
from openodke.corroborate.normalize import name_key
from openodke.coverage import NameMatcher, identity
from openodke.eval.extraction import _scoped, match_extraction, normalise_value, per_document
from openodke.eval.formats import GoldFact
from openodke.eval.report import _table
from openodke.ontology import Ontology
from openodke.types import Entity, Fact, GroundingVerdict
from openodke.types import Frozen as _Frozen

# Examples kept per bucket, unless asked otherwise.
EXAMPLES = 3
# A near name: this similarity of name keys, or one inside the other as whole
# words with the shorter at least four characters.
NEAR = 0.85
# The output is saturated when it grows with length at less than this share of
# the gold's rate.
SATURATED = 0.5

Side = Literal["recall", "precision"]
Claim = tuple[str, str, str]

# Recall buckets, in the order a miss is tried against them (DECISIONS #38).
RECALL: tuple[tuple[str, str], ...] = (
    ("never_offered", "relation never offered"),
    ("refused", "refused by the Validator"),
    ("wrong_relation", "wrong relation"),
    ("inverse", "inverse direction"),
    ("surface_form", "surface form"),
    ("scored_apart", "same triple, scored apart"),
    ("cross_sentence", "cross-sentence"),
    ("saturation", "output saturation"),
    ("entity_missing", "entity never extracted"),
    ("not_linked", "both seen but not linked"),
)
# The false positives before the gate, split three ways.
PRECISION: tuple[tuple[str, str], ...] = (
    ("refused_not_in_gold", "refused, not in gold"),
    ("written_supported", "written, not in gold, supported by its passage"),
    ("written_other", "written, not in gold, the rest"),
)
BUCKETS = dict(RECALL + PRECISION)
UNCONFIRMED = "surface form (unconfirmed)"


# --------------------------------------------------------------------------- #
# The view: gold facts and predictions as one scorer compares them
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One end of a gold fact: the keys a prediction's end must equal, and the names as written."""

    keys: frozenset[str]
    # As written: to find it in the text, to test a near name, and to print.
    forms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GoldItem:
    """One gold fact, and whether the scorer counted it found."""

    id: str
    doc_id: str
    subject: Endpoint
    relation: str
    object: Endpoint
    found: bool
    subject_type: str | None = None
    # What else the scorer compares, beside the triple: type, polarity, qualifiers.
    detail: Hashable = None
    # The dataset's evidence sentences, where it gives them.
    evidence: tuple[int, ...] = ()
    text: str = ""

    @property
    def claim(self) -> Claim:
        return (_first(self.subject), self.relation, _first(self.object))


@dataclass(frozen=True, slots=True)
class Said:
    """One prediction: its ends and relation as the scorer compares them, and as written."""

    doc_id: str | None
    subject: str
    relation: str
    object: str
    # Subject and object as written.
    forms: tuple[str, str] = ("", "")
    # The scorer matched it to a gold fact. A refused prediction is never a hit.
    hit: bool = False
    detail: Hashable = None
    # The grounder's verdict, where the prediction carries one.
    verdict: str | None = None
    # The gate's reason, for a refused prediction.
    reason: str | None = None
    text: str = ""
    # Its object is an entity, so it can have an inverse partner.
    edge: bool = True

    @property
    def claim(self) -> Claim:
        return (self.forms[0] or self.subject, self.relation, self.forms[1] or self.object)


@dataclass
class View:
    """What a diagnosis reads: one row's gold facts, predictions and texts.

    `refused` is what the gate refused, `None` when no gate's record was kept;
    `texts` the documents by id, `None` when they are unknown. `relation` puts a
    predicate's name in the form the scorer compares relations in, for the
    trace's offered relations and the ontology's inverses, and `names` turns
    that form back into one to print.
    """

    gold: list[GoldItem]
    said: list[Said]
    refused: list[Said] | None = None
    texts: dict[str, str] | None = None
    relation: Callable[[str], str] = str
    row: str = ""
    # A relation as the scorer compares it -> as a person reads it, where they differ.
    names: dict[str, str] = field(default_factory=dict)

    def name(self, relation: str) -> str:
        return self.names.get(relation, relation)


# --------------------------------------------------------------------------- #
# What the extractor was offered
# --------------------------------------------------------------------------- #


class Offered(_Frozen):
    """The relations the extractor was shown: per subject type where known, else those withheld.

    A model extractor shows one snippet per entity type (DECISIONS #6), and
    refuses a fact whose predicate is not in its subject type's snippet. So
    `by_type` is the truth when it is known; a coverage report knows only the
    relations no type was shown (`withheld`).
    """

    by_type: dict[str, tuple[str, ...]] | None = None
    withheld: tuple[str, ...] = ()
    source: str = ""

    def shown(self, relation: str, subject_type: str | None, key: Callable[[str], str]) -> bool:
        """Whether a fact with this relation and subject type could have been written."""
        if self.by_type is not None:
            names = self.by_type.get(subject_type) if subject_type else None
            if names is None:
                names = tuple(p for shown in self.by_type.values() for p in shown)
            return relation in {key(p) for p in names}
        return relation not in {key(p) for p in self.withheld}


def offered_by(extractor: object, ontology: Ontology, source: str = "") -> Offered | None:
    """What `extractor` shows its model: its snippets per type, or its `offered(ontology)`."""
    snippets = getattr(extractor, "snippets", None)
    if callable(snippets):
        by_type = {s.type_name: tuple(p.name for p in s.predicates) for s in snippets(ontology)}
        return Offered(by_type=by_type, source=source)
    offered = getattr(extractor, "offered", None)
    shown = offered(ontology) if callable(offered) else None
    if shown is None:
        return None
    withheld = tuple(sorted(set(ontology.predicates) - set(shown)))
    return Offered(withheld=withheld, source=source)


def read_trace(path: str | Path) -> tuple[Offered | None, dict[str, Any] | None]:
    """What a trace says the extractor was offered, and the run config when the trace is one.

    A trace is a run's `manifest.json` or stats (`KnowledgeGraph.stats`, whose
    coverage report names the relations never offered), or the run config that
    produced the predictions, whose extractor is built, with no model call, to
    read its snippets per type.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    data: Any
    if source.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: a trace is a manifest, a run's stats or a run config")
    stats: dict[str, Any] = data["stats"] if isinstance(data.get("stats"), dict) else data
    coverage = stats.get("coverage")
    if isinstance(coverage, dict):
        withheld = coverage.get("not_offered")
        if withheld is None:
            return None, None
        return Offered(withheld=tuple(withheld), source=source.name), None
    if "stages" in data and "ontology" in data:
        from openodke.run.build import build
        from openodke.run.config import load_config

        built = build(load_config(source))
        return offered_by(built.stages["extractor"], built.ontology, source.name), data
    raise ValueError(
        f"{source}: no coverage report and no stages in it; a trace is a run's manifest.json, "
        "its stats, or the run config that produced the predictions"
    )


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #


@runtime_checkable
class EquivalenceJudge(Protocol):
    """Is the prediction the gold fact in other words? (#143)

    Asked about each surface-form candidate: True puts the miss in *surface
    form*, False or None sends it on to the buckets after.
    """

    def equivalent(self, gold: Claim, predicted: Claim, text: str | None) -> bool | None: ...


@runtime_checkable
class Adjudicator(Protocol):
    """Does the passage support a written prediction the gold lacks? (#145)

    Without one, the prediction's own grounder verdict says, and the bucket is
    unadjudicated.
    """

    def supports(self, predicted: Claim, text: str | None) -> bool | None: ...


# --------------------------------------------------------------------------- #
# The report's entries
# --------------------------------------------------------------------------- #


class Example(_Frozen):
    """One miss or false positive, with the prediction nearest to it."""

    doc_id: str | None = None
    gold: str | None = None
    nearest: str | None = None
    why: str | None = None


class Bucket(_Frozen):
    """One cause bucket: how many, what share, and a few examples.

    `count` and `share` are None when the run cannot say, and `note` says why:
    no trace of what was offered, no record of what the gate refused.
    `share` is of the row's misses on the recall side, and of its false
    positives before the gate on the precision side.
    """

    row: str
    side: Side
    bucket: str
    label: str
    count: int | None
    share: float | None
    examples: tuple[Example, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class Saturation:
    """The run-level signal: does the output grow with the text as the gold does?"""

    flagged: bool
    # Log-log slopes of the extraction and gold counts per document on its length.
    output: float | None
    gold: float | None
    median: float
    ceiling: int
    long: frozenset[str]

    def note(self) -> str:
        if self.output is None or self.gold is None:
            return "too few documents of different lengths to say"
        grows = (
            f"the output grows with length at {self.output:+.2f} (log-log slope) against the "
            f"gold's {self.gold:+.2f}; at most {self.ceiling} per document"
        )
        if self.gold <= 0:
            return f"not flagged: the gold does not grow with length here; {grows}"
        if not self.flagged:
            return f"not flagged: {grows}"
        return f"flagged: {grows}, so misses in documents over {self.median:,.0f} characters"


@dataclass
class Diagnosis:
    """One row diagnosed: the buckets, each miss's bucket, and the rates the fixes read."""

    view: View
    buckets: tuple[Bucket, ...]
    # Gold id -> its bucket, for every miss.
    assigned: dict[str, str]
    gold: int
    hits: int
    # The row's own (hits, gold) where a cause does not reach, by name: `overall`,
    # `offered`, `same_sentence`, `short`. What a fix is assumed to recover at.
    rates: dict[str, tuple[int, int]] = field(default_factory=dict)
    saturation: Saturation | None = None
    # (gold relation, relation written) for each miss a prediction linked with
    # another relation, the same way round (`confusions`) or the other (`swaps`).
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)
    swaps: Counter[tuple[str, str]] = field(default_factory=Counter)
    inverses: dict[str, str] = field(default_factory=dict)
    offered: Offered | None = None
    judged: bool = False

    @property
    def misses(self) -> int:
        return self.gold - self.hits

    def count(self, bucket: str) -> int:
        return sum(1 for b in self.assigned.values() if b == bucket)

    def render(self) -> list[str]:
        return render(self.buckets)


# --------------------------------------------------------------------------- #
# Diagnosing
# --------------------------------------------------------------------------- #


def diagnose(
    view: View,
    *,
    offered: Offered | None = None,
    inverses: Mapping[str, str] | None = None,
    judge: EquivalenceJudge | None = None,
    adjudicator: Adjudicator | None = None,
    examples: int = EXAMPLES,
) -> Diagnosis:
    """Every miss of `view` in one bucket, and the false positives split three ways.

    `offered` is the trace of what the extractor was shown (`read_trace`,
    `offered_by`); `inverses` maps a predicate to its inverse, as
    `Ontology.inverses` does, for the inverse-direction bucket. `examples` per
    bucket are taken one document at a time, so they come from as many
    documents as they can.
    """
    key = view.relation
    pairs = {key(a): key(b) for a, b in (inverses or {}).items()}
    by_doc: dict[str | None, list[Said]] = defaultdict(list)
    for said in [*view.said, *(view.refused or ())]:
        by_doc[said.doc_id].append(said)
    refused_by_doc: dict[str | None, list[Said]] = defaultdict(list)
    for said in view.refused or ():
        refused_by_doc[said.doc_id].append(said)
    seen: dict[str | None, set[str]] = defaultdict(set)
    for doc, made in by_doc.items():
        for said in made:
            seen[doc].update((said.subject, said.object))

    saturation = _saturation(view, by_doc)
    crossing = _Crossing(view.gold, view.texts)
    out = Diagnosis(
        view=view,
        buckets=(),
        assigned={},
        gold=len(view.gold),
        hits=sum(1 for g in view.gold if g.found),
        saturation=saturation,
        inverses=pairs,
        offered=offered,
        judged=judge is not None,
    )
    found: dict[str, list[Example]] = defaultdict(list)
    for g in view.gold:
        cross = crossing.crosses(g)
        _rate(out.rates, "overall", g, True)
        _rate(out.rates, "same_sentence", g, cross is False)
        _rate(
            out.rates,
            "offered",
            g,
            offered is None or offered.shown(g.relation, g.subject_type, key),
        )
        long = saturation is not None and g.doc_id in saturation.long
        _rate(out.rates, "short", g, saturation is not None and not long)
        if g.found:
            continue
        nearby = _Near(by_doc[g.doc_id], refused_by_doc[g.doc_id], seen[g.doc_id], pairs)
        bucket, example = _bucket(g, nearby, out, judge=judge, cross=cross, long=long)
        out.assigned[g.id] = bucket
        found[bucket].append(example)

    misses = out.misses
    recall = []
    for name, label in RECALL:
        unknown = _unknown(name, view, offered, saturation, crossing)
        count = None if unknown else len(found[name])
        recall.append(
            Bucket(
                row=view.row,
                side="recall",
                bucket=name,
                label=UNCONFIRMED if name == "surface_form" and judge is None else label,
                count=count,
                share=count / misses if count is not None and misses else None,
                examples=_spread(found[name], examples),
                note=unknown or _note(name, out, crossing),
            )
        )
    out.buckets = (*recall, *_precision(view, adjudicator, examples))
    return out


@dataclass(frozen=True)
class _Near:
    """What one document's pipeline produced, for trying its misses against the buckets."""

    made: list[Said]
    refused: list[Said]
    # Every end of every prediction: the entities the document's extraction named.
    seen: set[str]
    # Relation -> its inverse, as the scorer names relations.
    pairs: Mapping[str, str]


def _bucket(
    g: GoldItem, near: _Near, out: Diagnosis, *, judge: EquivalenceJudge | None,
    cross: bool | None, long: bool,
) -> tuple[str, Example]:  # fmt: skip
    """The first bucket `g`'s evidence fits, and the example it is shown by."""
    s, o, made = g.subject.keys, g.object.keys, near.made

    def example(nearest: Said | None = None, why: str | None = None) -> Example:
        shown = nearest.text if nearest else None
        return Example(doc_id=g.doc_id, gold=g.text, nearest=shown, why=why)

    offered = out.offered
    if offered is not None and not offered.shown(g.relation, g.subject_type, out.view.relation):
        return "never_offered", example(why=f"not offered for {g.subject_type or 'any type'}")
    for r in near.refused:
        if _matches(g, r):
            return "refused", example(r, r.reason)
    forward = [p for p in made if p.subject in s and p.object in o]
    backward = [p for p in made if p.subject in o and p.object in s]
    turned = {g.relation, near.pairs.get(g.relation, g.relation)}
    other = [p for p in forward if p.relation != g.relation]
    other += [p for p in backward if p.relation not in turned]
    out.swaps.update((g.relation, p.relation) for p in dict.fromkeys(backward))
    if other:
        out.confusions.update((g.relation, p.relation) for p in dict.fromkeys(other))
        return "wrong_relation", example(other[0], f"written as {out.view.name(other[0].relation)}")
    inverse = [p for p in backward if p.relation in turned]
    if inverse:
        return "inverse", example(inverse[0], "the other way round")
    for p in _near(g, made):
        if judge is None:
            return "surface_form", example(p, "a near name")
        text = out.view.texts.get(g.doc_id) if out.view.texts else None
        if judge.equivalent(g.claim, p.claim, text) is True:
            return "surface_form", example(p, "judged the same fact")
    same = [p for p in forward if p.relation == g.relation]
    if same:
        return "scored_apart", example(same[0], "paired with another gold fact, or details differ")
    if cross:
        return "cross_sentence", example(why="the evidence spans sentences")
    if long:
        return "saturation", example(why="a long document in a saturated run")
    if not (s & near.seen) or not (o & near.seen):
        absent = g.subject if not (s & near.seen) else g.object
        return "entity_missing", example(why=f"{_first(absent)} is in no prediction")
    return "not_linked", example(why="both ends predicted, never together")


def _matches(g: GoldItem, p: Said) -> bool:
    return (
        p.relation == g.relation
        and p.subject in g.subject.keys
        and p.object in g.object.keys
        and (g.detail is None or p.detail == g.detail)
    )


def _near(g: GoldItem, made: Sequence[Said]) -> list[Said]:
    """The surface-form candidates: the relation, one end equal, the other a near name."""
    out = []
    for p in made:
        if p.relation != g.relation:
            continue
        subject, obj = p.subject in g.subject.keys, p.object in g.object.keys
        near_object = subject and not obj and _close(p.forms[1] or p.object, g.object.forms)
        near_subject = obj and not subject and _close(p.forms[0] or p.subject, g.subject.forms)
        if near_object or near_subject:
            out.append(p)
    return out


def near(a: str, b: str) -> bool:
    """Whether two names are close enough to be one name written two ways.

    Their name keys (`name_key`: case, accents, punctuation and a leading
    article dropped) are equal, one is inside the other as whole words with the
    shorter at least four characters, or they are 85% alike.
    """
    x, y = name_key(a), name_key(b)
    if not x or not y:
        return False
    if x == y:
        return True
    short, long = sorted((x, y), key=len)
    if len(short) >= 4 and f" {short} " in f" {long} ":
        return True
    return SequenceMatcher(None, x, y).ratio() >= NEAR


def _close(name: str, forms: Iterable[str]) -> bool:
    return any(near(name, form) for form in forms)


def _rate(rates: dict[str, tuple[int, int]], name: str, g: GoldItem, counted: bool) -> None:
    if counted:
        hits, total = rates.get(name, (0, 0))
        rates[name] = (hits + g.found, total + 1)


def _unknown(
    name: str,
    view: View,
    offered: Offered | None,
    saturation: Saturation | None,
    crossing: _Crossing,
) -> str | None:
    """Why a bucket cannot be counted on this run; None when it can."""
    if name == "never_offered" and offered is None:
        return "unknown: no trace of what the extractor was offered (--trace)"
    if name == "refused" and view.refused is None:
        return "unknown: no record of what a gate refused (run with --validator)"
    if name == "cross_sentence" and crossing.method() is None:
        return "unknown: no evidence sentences, and no text naming both ends"
    if name == "saturation" and saturation is None:
        return (
            "unknown: the texts were not given"
            if view.texts is None
            else "unknown: too few documents to read a trend"
        )
    return None


def _note(name: str, out: Diagnosis, crossing: _Crossing) -> str | None:
    if name == "saturation" and out.saturation is not None:
        return out.saturation.note()
    if name == "cross_sentence":
        return crossing.method()
    if name == "never_offered" and out.offered is not None:
        return f"from {out.offered.source}" if out.offered.source else None
    return None


def _spread(found: Sequence[Example], n: int) -> tuple[Example, ...]:
    """Up to `n` examples, one document at a time: the first of each, then the second..."""
    rank: Counter[str | None] = Counter()
    order = []
    for i, e in enumerate(found):
        order.append((rank[e.doc_id], i, e))
        rank[e.doc_id] += 1
    return tuple(e for _, _, e in sorted(order, key=lambda t: (t[0], t[1]))[:n])


# --------------------------------------------------------------------------- #
# The precision side
# --------------------------------------------------------------------------- #


def _precision(view: View, adjudicator: Adjudicator | None, n: int) -> list[Bucket]:
    by_doc: dict[str, list[GoldItem]] = defaultdict(list)
    for g in view.gold:
        by_doc[g.doc_id].append(g)
    caught = [
        Example(doc_id=r.doc_id, nearest=r.text, why=r.reason)
        for r in view.refused or ()
        if not any(_matches(g, r) for g in by_doc.get(r.doc_id or "", ()))
    ]
    supported: list[Example] = []
    rest: list[Example] = []
    verdicts = adjudicator is not None or any(s.verdict for s in view.said)
    for s in view.said:
        if s.hit:
            continue
        if adjudicator is not None:
            text = view.texts.get(s.doc_id or "") if view.texts else None
            said = adjudicator.supports(s.claim, text)
        else:
            said = s.verdict == "supported"
        (supported if said else rest).append(
            Example(doc_id=s.doc_id, nearest=s.text, why=s.verdict and f"grounded {s.verdict}")
        )
    total = len(caught) + len(supported) + len(rest)
    found = {
        "refused_not_in_gold": None if view.refused is None else caught,
        "written_supported": supported if verdicts else None,
        "written_other": rest if verdicts else [*supported, *rest],
    }
    notes = {
        "refused_not_in_gold": "unknown: no record of what a gate refused (run with --validator)",
        "written_supported": "unknown: the predictions carry no grounder verdict",
    }
    out = []
    for name, label in PRECISION:
        items = found[name]
        if name == "written_supported" and adjudicator is not None:
            label = "written, not in gold, supported (adjudicated)"
        elif name == "written_supported" and items is not None:
            label += " (unadjudicated, #145)"
        elif name == "written_other" and not verdicts:
            label = "written, not in gold"
        out.append(
            Bucket(
                row=view.row,
                side="precision",
                bucket=name,
                label=label,
                count=None if items is None else len(items),
                share=None if items is None else (len(items) / total if total else None),
                examples=_spread(items or [], n),
                note=notes[name] if items is None else None,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Cross-sentence and saturation
# --------------------------------------------------------------------------- #


class _Crossing:
    """Whether a gold fact's evidence spans sentences: the dataset's evidence, else its names."""

    def __init__(self, gold: Sequence[GoldItem], texts: Mapping[str, str] | None) -> None:
        self.texts = texts
        self.ends: dict[str, list[Endpoint]] = defaultdict(list)
        for g in gold:
            self.ends[g.doc_id] += [g.subject, g.object]
        self.by_evidence = 0
        self.by_names = 0
        self._places: dict[str, dict[str, set[int]]] = {}

    def crosses(self, g: GoldItem) -> bool | None:
        """True across sentences, False within one, None when neither the data nor the text says."""
        if g.evidence:
            self.by_evidence += 1
            return len(set(g.evidence)) > 1
        if self.texts is None or g.doc_id not in self.texts:
            return None
        places = self._where(g.doc_id)
        a, b = places.get(_identity(g.subject) or ""), places.get(_identity(g.object) or "")
        if not a or not b:
            return None
        self.by_names += 1
        return not (a & b)

    def method(self) -> str | None:
        parts = []
        if self.by_evidence:
            parts.append(f"{self.by_evidence} gold facts read by the dataset's evidence sentences")
        if self.by_names:
            parts.append(f"{self.by_names} by the sentences that name both ends")
        return "; ".join(parts) or None

    def _where(self, doc_id: str) -> dict[str, set[int]]:
        """Every gold end of the document, by its identity, to the sentences that name it."""
        if doc_id not in self._places:
            assert self.texts is not None
            text = self.texts[doc_id]
            starts = [start for start, _ in sentences(text)]
            places: dict[str, set[int]] = defaultdict(set)
            entities = [_entity(e) for e in dict.fromkeys(self.ends[doc_id]) if e.forms]
            for keys, start, _ in NameMatcher(entities).find(text):
                for k in keys:
                    places[k].add(max(bisect_right(starts, start) - 1, 0))
            self._places[doc_id] = places
        return self._places[doc_id]


def _identity(endpoint: Endpoint) -> str | None:
    return identity(_entity(endpoint)) if endpoint.forms else None


def _entity(endpoint: Endpoint) -> Entity:
    first, *rest = endpoint.forms
    return Entity(key=first, type="Thing", label=first, aliases=tuple(rest))


def _saturation(view: View, by_doc: Mapping[str | None, list[Said]]) -> Saturation | None:
    """Does the extraction count per document grow with its length as the gold count does?"""
    if view.texts is None:
        return None
    gold = Counter(g.doc_id for g in view.gold)
    docs = [d for d in gold if d in view.texts and view.texts[d]]
    if len(docs) < 3:
        return None
    length = {d: len(view.texts[d]) for d in docs}
    x = [math.log(length[d]) for d in docs]
    made = {d: len(by_doc.get(d, ())) for d in docs}
    output = _slope(x, [math.log1p(made[d]) for d in docs])
    expected = _slope(x, [math.log1p(gold[d]) for d in docs])
    middle = median(length.values())
    flagged = _flat(output, expected)
    return Saturation(
        flagged=flagged,
        output=output,
        gold=expected,
        median=middle,
        ceiling=max(made.values()),
        long=frozenset(d for d in docs if length[d] > middle) if flagged else frozenset(),
    )


def _flat(output: float | None, gold: float | None) -> bool:
    """The output grows with length at under half the gold's rate, where the gold grows at all."""
    if output is None or gold is None or gold <= 0:
        return False
    return output < SATURATED * gold


def _slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    """The least-squares slope of `y` on `x`; None when `x` does not vary."""
    mx, my = sum(x) / len(x), sum(y) / len(y)
    spread = sum((a - mx) ** 2 for a in x)
    if spread == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / spread


# --------------------------------------------------------------------------- #
# A view from GoldFact labels
# --------------------------------------------------------------------------- #


def gold_view(
    gold: Sequence[GoldFact],
    predictions: Iterable[Fact],
    *,
    refused: Iterable[tuple[Fact, str]] | None = None,
    texts: Mapping[str, str] | None = None,
    row: str = "",
) -> View:
    """`GoldFact` labels and predicted `Fact`s, matched as `match_extraction` matches them.

    A gold fact is found when the matcher paired it as correct; a prediction is
    a hit on the same terms. `refused` is the gate's record: each fact it
    refused and why. A prediction citing only unlabelled documents is in no
    number, here as in the report.
    """
    labelled = {g.doc_id for g in gold}
    outcomes = match_extraction(gold, list(predictions))
    found = {id(o.gold) for o in outcomes if o.gold is not None and o.kind == "correct"}
    items: list[GoldItem] = []
    seen: Counter[str] = Counter()
    for g in gold:
        fact = g.fact
        seen[g.doc_id] += 1
        items.append(
            GoldItem(
                id=f"{g.doc_id}#{seen[g.doc_id]}",
                doc_id=g.doc_id,
                subject=Endpoint(frozenset({fact.subject.key}), _forms(fact.subject)),
                relation=fact.predicate,
                object=_object_end(fact),
                found=id(fact) in found,
                subject_type=fact.subject.type,
                detail=_detail(fact),
                text=_text(fact),
            )
        )
    said = [
        _said(o.predicted, o.doc_id, o.kind == "correct")
        for o in outcomes
        if o.predicted is not None
    ]
    kept = None
    if refused is not None:
        kept = [
            _said(fact, doc, False, reason)
            for whole, reason in refused
            for fact in per_document([whole])
            for doc in [fact.evidence[0].doc_id if fact.evidence else None]
            if doc in labelled
        ]
    return View(gold=items, said=said, refused=kept, texts=dict(texts) if texts else None, row=row)


def _forms(entity: Entity) -> tuple[str, ...]:
    return tuple(dict.fromkeys(n for n in (entity.label, *entity.aliases, entity.key) if n))


def _object_end(fact: Fact) -> Endpoint:
    if fact.object_entity is not None:
        return Endpoint(frozenset({fact.object_entity.key}), _forms(fact.object_entity))
    value = fact.object_value
    return Endpoint(frozenset({f"value:{normalise_value(value)}"}), (str(value),))


def _object_key(fact: Fact) -> str:
    if fact.object_entity is not None:
        return fact.object_entity.key
    return f"value:{normalise_value(fact.object_value)}"


def _detail(fact: Fact) -> Hashable:
    return (fact.subject.type, fact.polarity, _scoped(fact))


def _text(fact: Fact) -> str:
    obj = (
        (fact.object_entity.label or fact.object_entity.key)
        if fact.object_entity is not None
        else str(fact.object_value)
    )
    return f"{fact.subject.label or fact.subject.key} · {fact.predicate} · {obj}"


def _said(fact: Fact, doc: str | None, hit: bool, reason: str | None = None) -> Said:
    obj = fact.object_entity
    return Said(
        doc_id=doc,
        subject=fact.subject.key,
        relation=fact.predicate,
        object=_object_key(fact),
        forms=(
            fact.subject.label or fact.subject.key,
            (obj.label or obj.key) if obj is not None else str(fact.object_value),
        ),
        hit=hit,
        detail=_detail(fact),
        verdict=None if fact.verdict is GroundingVerdict.UNCHECKED else fact.verdict.value,
        reason=reason,
        text=_text(fact),
        edge=obj is not None,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _first(endpoint: Endpoint) -> str:
    return endpoint.forms[0] if endpoint.forms else next(iter(sorted(endpoint.keys)), "")


def render(buckets: Sequence[Bucket]) -> list[str]:
    """The tables `odke eval` prints: misses by bucket, an example of each, then false positives."""
    if not buckets:
        return []
    recall = [b for b in buckets if b.side == "recall"]
    misses = sum(b.count or 0 for b in recall)
    noun = "miss" if misses == 1 else "misses"
    lines = [f"where it loses facts  ({buckets[0].row}: {misses:,} {noun})"]
    lines += _bucket_table(recall, "misses")
    lines += _first_examples(recall)
    fp = [b for b in buckets if b.side == "precision"]
    total = sum(b.count or 0 for b in fp)
    lines += ["", f"  false positives before the gate: {total:,}"]
    lines += _bucket_table(fp, "facts")
    notes = [f"{b.label}: {b.note}" for b in buckets if b.note]
    if notes:
        lines += ["", *(f"  - {note}" for note in notes)]
    return lines


def _bucket_table(buckets: Sequence[Bucket], noun: str) -> list[str]:
    body = [
        [
            b.label,
            "—" if b.count is None else f"{b.count:,}",
            "—" if b.share is None else f"{b.share:.1%}",
        ]
        for b in buckets
    ]
    return _table(["", noun, "share"], body)


def _first_examples(buckets: Iterable[Bucket]) -> list[str]:
    lines = []
    for b in buckets:
        if not b.examples:
            continue
        e = b.examples[0]
        near = f"; nearest: {e.nearest}" if e.nearest else ""
        why = f" ({e.why})" if e.why else ""
        lines.append(f"  e.g. {b.label}: {e.doc_id}: {e.gold}{near}{why}")
    return [""] + lines if lines else []


__all__ = [
    "BUCKETS",
    "EXAMPLES",
    "PRECISION",
    "RECALL",
    "UNCONFIRMED",
    "Adjudicator",
    "Bucket",
    "Claim",
    "Diagnosis",
    "Endpoint",
    "EquivalenceJudge",
    "Example",
    "GoldItem",
    "Offered",
    "Said",
    "Saturation",
    "View",
    "diagnose",
    "gold_view",
    "near",
    "offered_by",
    "read_trace",
    "render",
]
