"""What should change: each bucket's fixes, ranked by the recall the arithmetic gives them (#141).

A diagnosis says where the facts went; this says what to change first. Every
bucket maps to a fix that names its config knob or command, and every fix has an
expected recall gain computed from the bucket counts, never a model's opinion
(DECISIONS #38). Two kinds of arithmetic:

- **Exact**, where the fix can be replayed offline on the run's own output.
  Accepting what the gate refused writes exactly the refused facts back.
  Declaring inverse pairs adds exactly the partners `Pipeline`'s inverse step
  would add (DECISIONS #28), which are scored against the gold, so the recall
  gained and the precision lost are both counts.
- **At the run's own rate**, where it cannot. The bucket's misses are assumed to
  be found as often as the run finds the facts the cause does not touch: a
  relation never offered at the recall on relations that were; cross-sentence
  facts at the recall on same-sentence ones; a saturated run's long documents
  at the recall on its short ones; the rest at the run's recall. That is the
  `expected` gain, and it ranks the fixes. Every miss in the bucket found is the
  `ceiling`.

Gains are recall points, as shares of the gold facts scored, and they overlap:
two fixes can recover the same fact, so they do not add. A `reference` run, such
as another extractor on the same gold, gives a third number: the gain if the
bucket were found as often as that run finds the same facts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from pydantic import Field

from openodke.eval.diagnosis import Diagnosis, GoldItem, Said
from openodke.eval.report import _table
from openodke.types import Frozen


@dataclass(frozen=True)
class Remedy:
    """One fix in the catalogue: the buckets it answers, what to change, and how it is priced."""

    id: str
    buckets: tuple[str, ...]
    action: str
    knob: str
    # Run-config keys whose change applies it, for the track record's config diff.
    knobs: tuple[str, ...]
    # The rate the bucket is assumed recovered at, or `exact` / `judge`.
    rate: str


CATALOGUE: tuple[Remedy, ...] = (
    Remedy(
        "offer-relations",
        ("never_offered",),
        "Offer every relation the subject's type can take: raise the snippet limit, or name "
        "the types",
        "stages.extractor.snippet_limit",
        ("stages.extractor.snippet_limit", "stages.extractor.types"),
        "offered",
    ),
    Remedy(
        "audit-refusals",
        ("refused",),
        "Audit what the gate refused (odke eval refusals, #113), and loosen it if the "
        "refusals are wrong",
        "stages.gate.refuse_not_found",
        ("stages.gate", "stages.validator"),
        "exact",
    ),
    Remedy(
        "relation-descriptions",
        ("wrong_relation",),
        "Tell the confused relations apart: a description and examples for each in the ontology",
        "ontology predicates.<name>.description",
        (),
        "overall",
    ),
    Remedy(
        "inverses",
        ("inverse",),
        "Add inverse partners: declare the pairs (inverse_of, symmetric) and keep the step on",
        "ontology predicates.<name>.inverse_of; inverses: true",
        ("inverses",),
        "exact",
    ),
    Remedy(
        "normalise-names",
        ("surface_form",),
        "Normalise names, or give entities the aliases the gold uses",
        "stages.normalizer; Entity.aliases",
        ("stages.normalizer",),
        "judge",
    ),
    Remedy(
        "check-details",
        ("scored_apart",),
        "Check the types, polarity and identity qualifiers written: a prediction finds one gold "
        "fact, and only when they agree",
        "ontology types and qualifiers",
        (),
        "overall",
    ),
    Remedy(
        "wider-context",
        ("cross_sentence",),
        "Let a fact span sentences: whole-document context, wider windows, or evidence that "
        "cites two sentences",
        "stages.chunker",
        ("stages.chunker",),
        "same_sentence",
    ),
    Remedy(
        "smaller-chunks",
        ("saturation",),
        "Ask less of each call: smaller chunks, so the output can grow with the text",
        "stages.chunker; models.extract.max_tokens",
        ("stages.chunker", "models.extract.max_tokens"),
        "short",
    ),
    Remedy(
        "reextract",
        ("entity_missing", "not_linked"),
        "Hand the coverage report's gaps, missed entities and uncovered sentences, back to the "
        "extractor (#102)",
        "reextract",
        ("reextract",),
        "overall",
    ),
)
REMEDIES = {remedy.id: remedy for remedy in CATALOGUE}

# What each rate is, in the words a basis line uses.
_RATES = {
    "overall": "its recall",
    "offered": "its recall on the relations it was offered",
    "same_sentence": "its recall on same-sentence facts",
    "short": "its recall on the shorter half of the documents",
}


class Gain(Frozen):
    """A fix's recall gain, in points as a share of the gold: the expected, its ceiling, and how.

    `exact` is a replay of the fix on the run's own output; otherwise
    `expected` assumes the bucket is found at the run's own rate where the
    cause does not reach, which `basis` names with its numbers. `reference` is
    the gain at a reference run's rate on the same facts. `precision` is the
    change in precision, where the replay knows it.
    """

    expected: float
    ceiling: float
    exact: bool = False
    basis: str
    reference: float | None = None
    precision: float | None = None


class Measured(Frozen):
    """One prediction beside its measurement, from the track record (`openodke.eval.track`)."""

    run: str
    against: str
    applied: str
    expected: float
    ceiling: float
    measured: float
    low: float
    high: float
    verdict: str


class Fix(Frozen):
    """One fix for one row, ranked: what to change, where, and what the arithmetic expects."""

    id: str
    rank: int = Field(ge=1)
    row: str
    buckets: tuple[str, ...]
    misses: int = Field(ge=0)
    action: str
    knob: str
    recall: Gain
    detail: tuple[str, ...] = ()
    record: tuple[Measured, ...] = ()


def fixes(diagnosis: Diagnosis, *, reference: Collection[str] | None = None) -> list[Fix]:
    """The fixes for `diagnosis`'s buckets, best first, each with its expected recall gain.

    `reference` is the ids of the gold facts another run found on the same gold,
    for each fix's gain at that run's rate. A fix with nothing to recover is
    left out. Ties go to the larger ceiling, then to catalogue order.
    """
    found: list[tuple[float, float, int, Fix]] = []
    for index, remedy in enumerate(CATALOGUE):
        made = _price(remedy, diagnosis, reference)
        if made is not None:
            found.append((-made.recall.expected, -made.recall.ceiling, index, made))
    ranked = [fix for *_, fix in sorted(found, key=lambda t: t[:3])]
    return [fix.model_copy(update={"rank": n}) for n, fix in enumerate(ranked, 1)]


def render(found: Sequence[Fix]) -> list[str]:
    """The ranked table `odke eval` prints, with each fix's arithmetic under it."""
    if not found:
        return []
    row = found[0].row
    lines = [f"what should change  ({row}: ranked by expected recall gain; gains overlap)"]
    body = [
        [
            str(fix.rank),
            fix.id,
            _points(fix.recall.expected) + ("" if not fix.recall.exact else " exact"),
            _points(fix.recall.ceiling),
            "—" if fix.recall.reference is None else _points(fix.recall.reference),
            fix.knob,
        ]
        for fix in found
    ]
    lines += _table(["", "fix", "expected", "ceiling", "reference", "knob"], body)
    lines.append("")
    for fix in found:
        lines.append(f"  {fix.rank}. {fix.action}. {fix.recall.basis}")
        lines += [f"     - {line}" for line in fix.detail]
        lines += [
            f"     - track record: expected {_points(m.expected)} (ceiling {_points(m.ceiling)}), "
            f"measured {_points(m.measured)} [{_points(m.low)}, {_points(m.high)}] {m.verdict}"
            for m in fix.record
        ]
    return lines


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #


def _price(remedy: Remedy, d: Diagnosis, reference: Collection[str] | None) -> Fix | None:
    misses = sum(d.count(bucket) for bucket in remedy.buckets)
    gold = d.gold
    if not gold:
        return None
    if remedy.id == "inverses":
        return _inverses(remedy, d, misses)
    if remedy.id == "audit-refusals":
        return _refusals(remedy, d, misses) if misses else None
    if not misses:
        return None
    detail: tuple[str, ...] = ()
    if remedy.rate == "judge":
        if d.judged:
            gain = Gain(
                expected=misses / gold,
                ceiling=misses / gold,
                exact=True,
                basis=f"{_n(misses, 'near name')} the judge confirmed / {gold:,} gold",
            )
        else:
            gain = Gain(
                expected=0.0,
                ceiling=misses / gold,
                basis="unconfirmed: no fact-equivalence judge (#143); the ceiling counts "
                f"{_n(misses, 'candidate')} as the gold fact",
            )
    else:
        which = remedy.rate if d.rates.get(remedy.rate, (0, 0))[1] else "overall"
        hits, total = d.rates.get(which, (0, 0))
        rate = hits / total if total else 0.0
        gain = Gain(
            expected=misses * rate / gold,
            ceiling=misses / gold,
            basis=f"{_n(misses, 'miss')} × {rate:.1%} ({_RATES[which]}: {hits:,} of {total:,}) "
            f"/ {gold:,} gold",
        )
    if reference is not None:
        caught = sum(1 for g, b in d.assigned.items() if b in remedy.buckets and g in reference)
        gain = gain.model_copy(update={"reference": caught / gold})
    if remedy.id == "offer-relations":
        detail = _never_offered(d)
    elif remedy.id == "relation-descriptions":
        name = d.view.name
        top = d.confusions.most_common(5)
        detail = tuple(f"{name(r)} written as {name(w)}: {n}" for (r, w), n in top)
    return _fix(remedy, d, misses, gain, detail)


def _fix(
    remedy: Remedy, d: Diagnosis, misses: int, gain: Gain, detail: tuple[str, ...] = ()
) -> Fix:
    return Fix(
        id=remedy.id,
        rank=1,
        row=d.view.row,
        buckets=remedy.buckets,
        misses=misses,
        action=remedy.action,
        knob=remedy.knob,
        recall=gain,
        detail=detail,
    )


def _refusals(remedy: Remedy, d: Diagnosis, misses: int) -> Fix:
    """Accepting every refused fact: the gold ones come back, and so do the rest."""
    refused = d.view.refused or []
    written, hits = len(d.view.said), d.hits
    before = hits / written if written else None
    after = (hits + misses) / (written + len(refused)) if written + len(refused) else None
    change = None if before is None or after is None else after - before
    lost = f"; precision {_share(before)} → {_share(after)}" if change is not None else ""
    gain = Gain(
        expected=misses / d.gold,
        ceiling=misses / d.gold,
        exact=True,
        basis=f"accepting the {_n(len(refused), 'refused fact')} writes back "
        f"{_n(misses, 'gold fact')} / {d.gold:,} gold{lost}",
        precision=change,
    )
    return _fix(remedy, d, misses, gain)


def _inverses(remedy: Remedy, d: Diagnosis, misses: int) -> Fix | None:
    """The inverse step replayed on the written output (DECISIONS #28).

    The pairs are the ontology's, and a relation a miss was written with the
    other way round, which declaring symmetric would recover. Each written edge
    on one of them gets its partner unless its document states it already, as
    the step does, and a partner that is a missed gold fact recovers it. A
    relation written the other way round as some other relation is a wrong
    relation, never a pair guessed here: the ontology says which are inverses.
    """
    pairs = dict(d.inverses)
    pairs.update({r: r for r, said in d.swaps if r == said and r not in pairs})
    if not pairs:
        return None
    recovered, added = _replay(d, pairs)
    total = sum(recovered.values())
    if not total:
        return None
    written, hits = len(d.view.said), d.hits
    partners = sum(added.values())
    before = hits / written if written else None
    after = (hits + total) / (written + partners) if written + partners else None
    change = None if before is None or after is None else after - before
    gain = Gain(
        expected=total / d.gold,
        ceiling=total / d.gold,
        exact=True,
        basis=f"the inverse step replayed: {_n(partners, 'partner')}, {total:,} of them missed "
        f"gold facts / {d.gold:,} gold; precision {_share(before)} → {_share(after)}",
        precision=change,
    )
    name = d.view.name
    detail = []
    for pair in sorted(recovered, key=lambda p: (-recovered[p], p)):
        a, b = pair
        how = f"{name(a)} symmetric" if a == b else f"{name(a)} ↔ {name(b)}"
        kind = "declared" if a in d.inverses else "written the other way round"
        detail.append(f"{how} ({kind}): {recovered[pair]} gold, {added[pair]} partners")
    return _fix(remedy, d, misses, gain, tuple(detail[:8]))


def _replay(
    d: Diagnosis, pairs: dict[str, str]
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Gold facts recovered and partners added, per pair, by the inverse step on the output."""
    gold: dict[str, list[GoldItem]] = defaultdict(list)
    for g in d.view.gold:
        if not g.found:
            gold[g.doc_id].append(g)
    by_doc: dict[str | None, list[Said]] = defaultdict(list)
    for said in d.view.said:
        by_doc[said.doc_id].append(said)
    recovered: dict[tuple[str, str], int] = defaultdict(int)
    added: dict[tuple[str, str], int] = defaultdict(int)
    for doc, made in by_doc.items():
        stated = {(p.subject, p.relation, p.object) for p in made}
        open_ = list(gold.get(doc or "", ()))
        for p in made:
            partner = pairs.get(p.relation)
            if partner is None or not p.edge:
                continue
            triple = (p.object, partner, p.subject)
            if triple in stated:
                continue
            stated.add(triple)
            key = tuple(sorted((p.relation, partner)))
            pair = (key[0], key[1])
            added[pair] += 1
            hit = next(
                (
                    g
                    for g in open_
                    if g.relation == partner
                    and p.object in g.subject.keys
                    and p.subject in g.object.keys
                ),
                None,
            )
            if hit is not None:
                open_.remove(hit)
                recovered[pair] += 1
    return dict(recovered), dict(added)


def _never_offered(d: Diagnosis) -> tuple[str, ...]:
    """The relations behind the bucket, most missed first, by the subject's type."""
    counts: dict[tuple[str | None, str], int] = defaultdict(int)
    for g in d.view.gold:
        if d.assigned.get(g.id) == "never_offered":
            counts[(g.subject_type, g.relation)] += 1
    top = sorted(counts.items(), key=lambda t: (-t[1], str(t[0])))[:5]
    name = d.view.name
    return tuple(f"{kind or 'any type'} · {name(relation)}: {n}" for (kind, relation), n in top)


def _n(count: int, noun: str) -> str:
    plural = noun + ("es" if noun.endswith("s") else "s")
    return f"{count:,} {noun if count == 1 else plural}"


def _points(share: float) -> str:
    return f"{share * 100:+.1f}"


def _share(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


__all__ = ["CATALOGUE", "REMEDIES", "Fix", "Gain", "Measured", "Remedy", "fixes", "render"]
