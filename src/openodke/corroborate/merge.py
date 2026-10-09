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
from one host are one source, two chunks of one document are one, and a page
copied to another host counts once with its original when the corroborator has
the texts to compare (`duplicates`). `supported_by` names those sources, one
`Support` each, so `support` is its length: what a reconciler reads when a
source changes or disappears.

**Contest.** Two claims conflict when they share subject and predicate, the
predicate is single-valued in the ontology, they agree on its `scope_keys` (the
qualifiers a single value is unique within, which the Neo4j constraint checks
too), the objects differ and their valid intervals overlap — or when a denial
and an assertion share an object, on any predicate. Every claim gets a rank:

    rank      = trust × agreement
    trust     = max over its evidence of tier.weight × freshness
    freshness = floor + (1 − floor) × 0.5 ^ (age / half_life)
    agreement = 1 + ln(1 + Σ_s 1 / (1 + ln n_s))

where `age` is measured back from the newest evidence in the batch and `n_s` is
the number of claims source `s` backs in the batch. The highest rank wins. A
loser is **kept**, never dropped. Its confidence is multiplied by
`rank / winning rank`, and `qualifiers["odke.conflict"]` records a sentence
saying why, so the gate or a person can see the losing claim.

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

from openodke.corroborate.duplicates import DEFAULT_THRESHOLD, NearDuplicates
from openodke.corroborate.provenance import (
    CONFLICT,
    NEAR_DUPLICATES,
    SOURCE_FORM,
    near_duplicates,
    source_forms,
    unstamped,
)
from openodke.ontology import Ontology
from openodke.types import (
    Document,
    Evidence,
    Fact,
    GroundingVerdict,
    Polarity,
    SourceTier,
    Support,
)

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

    The documents in one `odke.near_duplicates` group count as one source
    between them, under the least of their keys. A fact with no evidence has
    no receipts to tell its sources apart, so it counts as many as its
    `support` already claims, and never fewer than one.
    """
    if not fact.evidence:
        return {f"fact:{fact.id}:{i}" for i in range(max(1, fact.support))}
    found = {source(e) for e in fact.evidence}
    groups = near_duplicates(fact.qualifiers.get(NEAR_DUPLICATES))
    return _collapse(found, fact.evidence, groups, source) if groups else found


def _union(groups: Iterable[Iterable[str]]) -> list[frozenset[str]]:
    """Groups that share a member joined into one, until none do."""
    out: list[frozenset[str]] = []
    for group in groups:
        joined, apart = frozenset(group), []
        for other in out:
            if other & joined:
                joined |= other
            else:
                apart.append(other)
        out = [*apart, joined] if joined else apart
    return out


def _collapse(
    sources: set[str],
    evidence: Sequence[Evidence],
    groups: Iterable[Iterable[str]],
    source: SourceKey,
) -> set[str]:
    """`sources`, with the sources of each group of documents counted once."""
    return set(_joined(sources, evidence, groups, source).values())


def _joined(
    sources: Iterable[str],
    evidence: Sequence[Evidence],
    groups: Iterable[Iterable[str]],
    source: SourceKey,
) -> dict[str, str]:
    """Each source, mapped to the key it counts under: the least of its group's, or its own."""
    joined: list[set[str]] = [{s} for s in sources]
    for group in groups:
        docs = set(group)
        joined.append({source(e) for e in evidence if e.doc_id in docs})
    return {member: min(g) for g in _union(joined) for member in g}


def support_of(
    evidence: Sequence[Evidence],
    groups: Iterable[Iterable[str]] = (),
    source: SourceKey = source_of,
) -> tuple[Support, ...]:
    """The support list a fact citing `evidence` carries: one `Support` per independent source.

    Sources are counted as `independent_sources` counts them: by `source`, and
    each group of near-duplicate documents in `groups` as one, under the least
    of their keys. Entries are sorted by source key, and each names its
    documents, its best tier and its newest clock.
    """
    if not evidence:
        return ()
    own = {source(e) for e in evidence}
    joined = _joined(own, evidence, groups, source)
    by_key: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        by_key[joined[source(e)]].append(e)
    return tuple(
        Support(
            source=key,
            doc_ids=tuple(sorted({e.doc_id for e in cited})),
            tier=max((e.tier for e in cited), key=lambda tier: tier.weight),
            retrieved_at=max((e.retrieved_at for e in cited), key=_aware),
        )
        for key, cited in sorted(by_key.items())
    )


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

    `documents` are the texts the evidence cites, by `Document.id`. Two of a
    claim's documents from different sources whose 5-word shingles have a
    Jaccard similarity of at least `near_duplicates` count as one source, and
    `odke.near_duplicates` records the group; `None` turns the check off.
    Without the texts nothing is compared. `odke run` and the Validator hand
    over the documents they were given, and `stats["near_duplicates"]` counts
    the pairs compared and found.
    """

    def __init__(
        self,
        ontology: Ontology | None = None,
        *,
        source: SourceKey = source_of,
        half_life_days: float = 365.0,
        freshness_floor: float = 0.5,
        intervals: Mapping[str, Literal["min", "max"]] = DEFAULT_INTERVALS,
        documents: Mapping[str, Document] | Iterable[Document] | None = None,
        near_duplicates: float | None = DEFAULT_THRESHOLD,
    ) -> None:
        if near_duplicates is not None and not 0.0 < near_duplicates <= 1.0:
            raise ValueError(f"near_duplicates must be in (0, 1] or None, got {near_duplicates}")
        self.ontology = ontology
        self.source = source
        self.half_life_days = half_life_days
        self.freshness_floor = freshness_floor
        self.intervals = intervals
        items = documents.values() if isinstance(documents, Mapping) else documents or ()
        self.documents: dict[str, Document] = {doc.id: doc for doc in items}
        self.near_duplicates = near_duplicates
        self.stats: dict[str, Any] = {"near_duplicates": {"compared": 0, "found": 0}}

    def corroborate(self, facts: Iterable[Fact]) -> list[Fact]:
        groups: dict[tuple[Any, ...], list[Fact]] = {}
        for fact in facts:
            fact = unstamped(fact)
            groups.setdefault(fact.signature, []).append(fact)
        copies = None
        if self.near_duplicates is not None and self.documents:
            copies = NearDuplicates(self.documents, self.near_duplicates)
        merged = [self._merge(members, copies) for members in groups.values()]
        if copies is not None:
            counts = self.stats["near_duplicates"]
            counts["compared"] += copies.compared
            counts["found"] += copies.found
        return self._contest(merged)

    # ----------------------------------------------------------------------- #
    # Merge
    # ----------------------------------------------------------------------- #

    def _merge(self, members: list[Fact], copies: NearDuplicates | None) -> Fact:
        first = members[0]
        evidence: dict[tuple[Any, ...], Evidence] = {}
        for fact in members:
            for e in fact.evidence:
                span = (e.span.start, e.span.end) if e.span else None
                evidence.setdefault((e.doc_id, span, e.uri), e)
        cited = tuple(evidence.values())
        # Every member's sources as cited; which of them are copies is decided below.
        sources: set[str] = set()
        for fact in members:
            own = {self.source(e) for e in fact.evidence}
            sources |= own if own else independent_sources(fact, self.source)
        # A group a member already carries is kept, so a later batch without
        # those texts still counts the copy once. One whose texts are all in
        # hand is decided again, at this threshold.
        found: list[Iterable[str]] = [
            g for f in members for g in near_duplicates(f.qualifiers.get(NEAR_DUPLICATES))
        ]
        if copies is not None:
            found = [g for g in found if not all(d in copies.documents for d in g)]
            found.extend(copies.clusters(cited, self.source))
        groups = tuple(sorted(tuple(sorted(g)) for g in _union(found) if len(g) > 1))
        if groups:
            sources = _collapse(sources, cited, groups, self.source)
        # Stable, so the first-seen member wins a tie.
        ranked = sorted(members, key=self._strength, reverse=True)
        qualifiers = self._qualifiers(members, ranked)
        if groups:
            qualifiers[NEAR_DUPLICATES] = groups
        extractors = sorted({f.extractor for f in members})
        present = {f.verdict for f in members}
        # A source with no evidence has no name: a claim any of whose members
        # cites nothing keeps its count and names none of its sources.
        named = support_of(cited, groups, self.source) if all(f.evidence for f in members) else ()
        return first.model_copy(
            update={
                "evidence": cited,
                "support": len(named) if named else len(sources),
                "supported_by": named,
                "qualifiers": qualifiers,
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
            if key == NEAR_DUPLICATES:
                continue
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

    def _scope(self, predicate: str) -> tuple[str, ...] | None:
        # The qualifier keys a single value is unique within — the grouping the
        # Neo4j constraint compiler checks too — or None when nothing is single.
        found = self.ontology.predicates.get(predicate) if self.ontology else None
        if found is None or found.cardinality != "single":
            return None
        return found.scope_keys

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
            keys = self._scope(predicate)
            if fact.polarity is Polarity.ASSERTED and keys is not None:
                # Scoped by the schema, not by what the extractor stamped: a
                # price per tier is not a conflict across tiers either way.
                bounded = tuple(
                    sorted((k, repr(fact.qualifiers[k])) for k in keys if k in fact.qualifiers)
                )
                contests[("value", subject, kind, predicate, bounded)].append(i)

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
    "support_of",
]
