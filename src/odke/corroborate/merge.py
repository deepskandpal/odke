"""Corroboration: one claim per signature, and a decided winner per contested value.

Two jobs, in order.

**Merge.** Facts sharing a `signature` are one claim. The signature already
carries polarity and identity-bearing qualifiers (DECISIONS #14, #15), so a
denial never merges with its assertion and uptime at p50 never merges with
uptime at p95; everything else about the claim is reconciled. Evidence is
unioned. Reconcilable qualifiers agree, or take the interval union
(`start_time` the earliest, `end_time` the latest), or take the best-ranked
source's value. The valid clock takes the earliest start and latest end any
source gave. `support` is the count of **independent sources**: forty pages
from one host are one source, two chunks of one document are one.

**Contest.** Two claims conflict when they share subject and predicate, the
predicate is single-valued in the ontology, the objects differ and their valid
intervals overlap — or when a denial and an assertion share an object, on any
predicate. Every claim gets a rank:

    rank      = trust × agreement
    trust     = max over its evidence of tier.weight × freshness
    freshness = floor + (1 − floor) × 0.5 ^ (age / half_life)
    agreement = 1 + ln(1 + Σ_s 1 / (1 + ln n_s))

where `age` is measured back from the newest evidence in the batch and `n_s` is
the number of claims source `s` backs in the batch. The highest rank wins. A
loser is **kept**, never dropped. Its confidence is multiplied by
`rank / winning rank`, and `qualifiers["odke.conflict"]` records a sentence
saying why, so a validator or a person can see the losing claim.

Agreement is volume-normalised. A raw count of agreeing sources — Pasternack and
Roth's *Sums* — rewards whoever is loudest, and a scraper emitting ten thousand
facts outvotes a careful filing on every one. Their *Average·Log* weighs a
source by the log of how much it asserts rather than by the raw volume. The same
shape, taken in one pass, is to discount each source's vote on a claim by
`1 + ln` of its claim count. One pass rather than their iteration to a
fixpoint, because the tier already supplies the prior that the iteration
exists to learn. The log outside the sum keeps agreement sub-linear, so trust
decides between tiers unless many independent sources disagree with it.
Freshness can at most halve trust by default, so a stale curated record still
beats a fresh scrape.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from odke.corroborate.provenance import CONFLICT, SOURCE_FORM, source_forms, unstamped
from odke.ontology import Ontology
from odke.types import Evidence, Fact, GroundingVerdict, Polarity, SourceTier

SourceKey = Callable[[Evidence], str]

# Which reconcilable qualifiers are interval bounds, and which end of a
# disagreement each keeps. A caller whose ontology names them differently
# passes their own.
DEFAULT_INTERVALS: Mapping[str, Literal["min", "max"]] = {
    "start_time": "min",
    "start": "min",
    "end_time": "max",
    "end": "max",
}

# The most informative verdict among a claim's members wins: one supporting
# span supports the claim; otherwise a contradiction outweighs silence.
_VERDICTS = (
    GroundingVerdict.SUPPORTED,
    GroundingVerdict.CONTRADICTED,
    GroundingVerdict.NOT_FOUND,
    GroundingVerdict.UNCHECKED,
)


def source_of(evidence: Evidence) -> str:
    """Which independent source a piece of evidence came from.

    The host of its URI when it has one — forty pages from one scraper are one
    source, not forty — otherwise its document.
    """
    if evidence.uri and (host := urlsplit(evidence.uri).hostname):
        return host.removeprefix("www.")
    return f"doc:{evidence.doc_id}"


def independent_sources(fact: Fact, source: SourceKey = source_of) -> set[str]:
    """The distinct sources behind a fact, by `source`.

    A fact with no evidence has no receipts to tell its sources apart, so it
    counts as many as its `support` already claims, and never fewer than one.
    """
    if not fact.evidence:
        return {f"fact:{fact.id}:{i}" for i in range(max(1, fact.support))}
    return {source(e) for e in fact.evidence}


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _bound(values: Sequence[datetime | None], *, latest: bool) -> datetime | None:
    known = [v for v in values if v is not None]
    if not known:
        return None
    return max(known, key=_aware) if latest else min(known, key=_aware)


def _overlap(a: Fact, b: Fact) -> bool:
    # Unknown bounds are open. A CEO from 2019 to 2024 and another from 2024 on
    # is a value that changed, not a conflict — DECISIONS #17's two clocks.
    for earlier, later in ((a, b), (b, a)):
        end, start = earlier.valid_to, later.valid_from
        if end is not None and start is not None and _aware(end) <= _aware(start):
            return False
    return True


def _claim(fact: Fact) -> str:
    obj = (
        (fact.object_entity.label or fact.object_entity.key)
        if fact.object_entity is not None
        else fact.object_value
    )
    return f"not '{obj}'" if fact.polarity is Polarity.DENIED else f"'{obj}'"


@dataclass(frozen=True)
class _Standing:
    rank: float
    sources: int
    tier: SourceTier | None
    retrieved: datetime | None

    def describe(self) -> str:
        count = (
            "1 independent source" if self.sources == 1 else f"{self.sources} independent sources"
        )
        if self.tier is None or self.retrieved is None:
            return f"{count} with no cited evidence"
        strongest = "" if self.sources == 1 else "the strongest "
        return f"{count} ({strongest}{self.tier.value}, retrieved {self.retrieved:%Y-%m-%d})"


class SignatureCorroborator:
    """Merge by signature, count independent sources, decide contested values.

    `ontology` supplies predicate cardinality. Without one, or for a predicate
    it does not declare, no value is contested: picking a winner among values
    that may all be true is the more harmful mistake. Denials against
    assertions are contested either way. `source` decides what counts as one
    independent source. `half_life_days` and `freshness_floor` shape how much
    age discounts trust. `intervals` names the qualifiers that reconcile as
    bounds.
    """

    def __init__(
        self,
        ontology: Ontology | None = None,
        *,
        source: SourceKey = source_of,
        half_life_days: float = 365.0,
        freshness_floor: float = 0.5,
        intervals: Mapping[str, Literal["min", "max"]] = DEFAULT_INTERVALS,
    ) -> None:
        self.ontology = ontology
        self.source = source
        self.half_life_days = half_life_days
        self.freshness_floor = freshness_floor
        self.intervals = intervals

    def corroborate(self, facts: Iterable[Fact]) -> list[Fact]:
        groups: dict[tuple[Any, ...], list[Fact]] = {}
        for fact in facts:
            fact = unstamped(fact)
            groups.setdefault(fact.signature, []).append(fact)
        return self._contest([self._merge(members) for members in groups.values()])

    # ----------------------------------------------------------------------- #
    # Merge
    # ----------------------------------------------------------------------- #

    def _merge(self, members: list[Fact]) -> Fact:
        first = members[0]
        evidence: dict[tuple[Any, ...], Evidence] = {}
        for fact in members:
            for e in fact.evidence:
                span = (e.span.start, e.span.end) if e.span else None
                evidence.setdefault((e.doc_id, span, e.uri), e)
        sources: set[str] = set()
        for fact in members:
            sources |= independent_sources(fact, self.source)
        # Stable, so the first-seen member wins a tie.
        ranked = sorted(members, key=self._strength, reverse=True)
        extractors = sorted({f.extractor for f in members})
        present = {f.verdict for f in members}
        return first.model_copy(
            update={
                "evidence": tuple(evidence.values()),
                "support": len(sources),
                "qualifiers": self._qualifiers(members, ranked),
                "valid_from": _bound([f.valid_from for f in members], latest=False),
                "valid_to": _bound([f.valid_to for f in members], latest=True),
                "extractor": "+".join(extractors),
                "confidence": max(f.confidence for f in members),
                "verdict": next(v for v in _VERDICTS if v in present),
            }
        )

    @staticmethod
    def _strength(fact: Fact) -> tuple[float, datetime]:
        tier = max((e.tier.weight for e in fact.evidence), default=SourceTier.UNVERIFIED.weight)
        moments = [_aware(e.retrieved_at) for e in fact.evidence]
        return tier, max(moments, default=datetime.min.replace(tzinfo=UTC))

    def _qualifiers(self, members: list[Fact], ranked: list[Fact]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        forms: dict[str, list[str]] = {}
        for key in dict.fromkeys(k for f in members for k in f.qualifiers):
            if key == SOURCE_FORM:
                for fact in members:
                    for field, spellings in source_forms(fact.qualifiers.get(key)).items():
                        bucket = forms.setdefault(field, [])
                        bucket.extend(s for s in spellings if s not in bucket)
                continue
            distinct = list(
                {
                    repr(f.qualifiers[key]): f.qualifiers[key]
                    for f in members
                    if key in f.qualifiers
                }.values()
            )
            if len(distinct) == 1:
                out[key] = distinct[0]
                continue
            end = self.intervals.get(key)
            if end is not None:
                try:
                    out[key] = min(distinct) if end == "min" else max(distinct)
                    continue
                except TypeError:
                    pass
            out[key] = next(f.qualifiers[key] for f in ranked if key in f.qualifiers)
        if forms:
            out[SOURCE_FORM] = {field: tuple(spellings) for field, spellings in forms.items()}
        return out

    # ----------------------------------------------------------------------- #
    # Contest
    # ----------------------------------------------------------------------- #

    def _single(self, predicate: str) -> bool:
        found = self.ontology.predicates.get(predicate) if self.ontology else None
        return found is not None and found.cardinality == "single"

    def _freshness(self, moment: datetime, newest: datetime) -> float:
        age_days = max(0.0, (newest - moment).total_seconds() / 86_400)
        decay = 0.5 ** (age_days / self.half_life_days)
        return self.freshness_floor + (1.0 - self.freshness_floor) * decay

    def _standing(
        self, fact: Fact, sources: set[str], volume: Counter[str], newest: datetime | None
    ) -> _Standing:
        # No receipts: the lowest tier at the floor of freshness.
        trust = SourceTier.UNVERIFIED.weight * self.freshness_floor
        tier: SourceTier | None = None
        retrieved: datetime | None = None
        if fact.evidence and newest is not None:
            trust = -1.0
            for e in fact.evidence:
                moment = _aware(e.retrieved_at)
                candidate = e.tier.weight * self._freshness(moment, newest)
                if candidate > trust:
                    trust, tier, retrieved = candidate, e.tier, moment
        votes = sum(1.0 / (1.0 + math.log(volume[s])) for s in sources)
        agreement = 1.0 + math.log1p(votes)
        return _Standing(trust * agreement, len(sources), tier, retrieved)

    def _contest(self, facts: list[Fact]) -> list[Fact]:
        sources = [independent_sources(f, self.source) for f in facts]
        volume = Counter(s for found in sources for s in found)
        moments = [_aware(e.retrieved_at) for f in facts for e in f.evidence]
        newest = max(moments, default=None)
        standing = [
            self._standing(f, found, volume, newest)
            for f, found in zip(facts, sources, strict=True)
        ]

        contests: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for i, fact in enumerate(facts):
            subject, kind, predicate, obj, _, scope = fact.signature
            if fact.polarity is not Polarity.PARTIAL:
                contests[("polarity", subject, kind, predicate, obj, scope)].append(i)
            if fact.polarity is Polarity.ASSERTED and self._single(predicate):
                contests[("value", subject, kind, predicate, scope)].append(i)

        outcomes: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for key, members in contests.items():
            if len(members) < 2:
                continue
            for i in members:
                rivals = [j for j in members if self._rivals(facts[i], facts[j], key[0])]
                if not rivals:
                    continue
                best = max(rivals, key=lambda j: standing[j].rank)
                mine, top = standing[i].rank, standing[best].rank
                if math.isclose(mine, top, rel_tol=1e-9):
                    outcomes[i].append(("tied", best))
                elif top > mine:
                    outcomes[i].append(("lost", best))
                else:
                    outcomes[i].append(("won", best))

        return [
            self._stamp(i, facts, standing, outcomes[i]) if i in outcomes else fact
            for i, fact in enumerate(facts)
        ]

    @staticmethod
    def _rivals(a: Fact, b: Fact, axis: str) -> bool:
        if a is b or not _overlap(a, b):
            return False
        if axis == "value":
            return a.signature[3] != b.signature[3]
        return a.polarity is not b.polarity

    def _stamp(
        self,
        i: int,
        facts: list[Fact],
        standing: list[_Standing],
        outcomes: list[tuple[str, int]],
    ) -> Fact:
        fact, mine = facts[i], standing[i]
        by_status: dict[str, list[int]] = defaultdict(list)
        for status, j in outcomes:
            by_status[status].append(j)
        update: dict[str, Any] = {}
        if losses := by_status["lost"]:
            j = max(losses, key=lambda k: standing[k].rank)
            ratio = mine.rank / standing[j].rank
            conflict: dict[str, Any] = {
                "status": "lost",
                "to": _claim(facts[j]),
                "ratio": round(ratio, 4),
                "confidence_before": fact.confidence,
                "reason": self._explain(facts[j], standing[j], fact, mine),
            }
            update["confidence"] = fact.confidence * ratio
        elif ties := by_status["tied"]:
            other = facts[ties[0]]
            conflict = {
                "status": "tied",
                "with": _claim(other),
                "reason": f"Could not choose between {_claim(fact)} and {_claim(other)}: "
                f"they rank equally at {mine.rank:.2f}, so both are kept at full "
                f"confidence for a validator or a person to settle.",
            }
        else:
            beaten = by_status["won"]
            j = max(beaten, key=lambda k: standing[k].rank)
            conflict = {
                "status": "won",
                "over": sorted({_claim(facts[k]) for k in beaten}),
                "reason": self._explain(fact, mine, facts[j], standing[j]),
            }
        update["qualifiers"] = {**fact.qualifiers, CONFLICT: conflict}
        return fact.model_copy(update=update)

    @staticmethod
    def _explain(winner: Fact, won: _Standing, loser: Fact, lost: _Standing) -> str:
        return (
            f"Kept {_claim(winner)} over {_claim(loser)}: {_claim(winner)} is backed by "
            f"{won.describe()} and {_claim(loser)} by {lost.describe()}. Ranked on source "
            f"trust, freshness, and agreement discounted by how much each source asserts: "
            f"{won.rank:.2f} to {lost.rank:.2f}."
        )


__all__ = [
    "DEFAULT_INTERVALS",
    "SignatureCorroborator",
    "independent_sources",
    "source_of",
]
