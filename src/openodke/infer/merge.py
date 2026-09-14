"""Near-duplicates, merged conservatively and reported.

The failure that makes naive inference useless: `works_at`, `employer`,
`employed_by` and `worksFor` all appear, and the resulting graph cannot be
queried. So candidates are clustered, one name is kept, and every other
spelling becomes an alias — which is also what lets extraction still match a
column headed `worksFor` (`openodke.extract._common.PredicateNames`).

Two candidates are linked by one of four signals, strongest first:

1. **the same name** once case, separators, plural and tense are forgotten
   (`names.normal_form`);
2. **a claim** — one lists the other's name among its aliases, which is how the
   model proposer says "these are the one I named";
3. **similar names** — `difflib` ratio of the normal forms at or above a
   threshold;
4. **shared evidence**, for predicates — at least `min_shared` of the same
   `(subject, value)` pairs, and at least half of the smaller side's. This is
   the signal that finds `works_at` and `employer`, whose names share nothing.

And a link is refused, with the reason recorded, when merging would change what
the data means:

- **ranges conflict** — `date` against `string`, or two entity types neither of
  which is a kind of the other. `integer` with `number` and `date` with
  `datetime` widen instead;
- **domains do not overlap**, for any link but the same name: `birth_place` on a
  Person and `berth_place` on a Ship are two predicates;
- **one type is a kind of the other**: `Car` never merges into `Vehicle`.

Extraction matches predicates by folded name — `birth_date` is `birthdate` to
it — so two candidates whose names fold alike cannot both stay when a refusal
kept them apart: the better-supported one is kept and the other is reported as
dropped.

Deterministic throughout: the same proposals merge the same way, and every
decision says which names, which signal and which refusal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from openodke.infer.candidates import PredicateCandidate, Proposals, TypeCandidate
from openodke.infer.names import fold, normal_form
from openodke.ontology import _JSON_TYPES, Cardinality
from openodke.types import Span

_NUMERIC = frozenset({"integer", "number", "float"})
_TEMPORAL = frozenset({"date", "datetime"})

C = TypeVar("C", TypeCandidate, PredicateCandidate)


class MergeDecision(BaseModel):
    """One thing the merge step did or refused to do, for the reviewer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["type", "predicate"]
    action: Literal["merged", "blocked", "dropped"]
    into: str
    names: tuple[str, ...]
    reason: str

    def __str__(self) -> str:
        return f"{self.action} {self.kind} {', '.join(self.names)} → {self.into}: {self.reason}"


class Merged(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposals: Proposals
    decisions: tuple[MergeDecision, ...] = ()


def merge(
    proposals: Proposals,
    *,
    type_threshold: float = 0.9,
    predicate_threshold: float = 0.85,
    min_shared: int = 2,
) -> Merged:
    """Cluster near-duplicate types, then predicates, and keep one name for each."""
    decisions: list[MergeDecision] = []
    types = _merge_types(list(proposals.types), type_threshold, decisions)
    renamed = _type_names(types)
    parents = {t.name: set(t.parents) for t in types}

    def lineage(name: str) -> set[str]:
        seen: set[str] = set()
        stack = [name]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(parents.get(current, ()))
        return seen

    predicates = [
        p.model_copy(
            update={
                "domain": tuple(dict.fromkeys(renamed.get(normal_form(d), d) for d in p.domain)),
                "range": p.range
                if p.range in _JSON_TYPES
                else renamed.get(normal_form(p.range), p.range),
            }
        )
        for p in proposals.predicates
    ]
    merged_predicates = _merge_predicates(
        predicates, predicate_threshold, min_shared, lineage, decisions
    )
    return Merged(
        proposals=Proposals(types=tuple(types), predicates=tuple(merged_predicates)),
        decisions=tuple(decisions),
    )


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


class _Clusters:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.members = {i: [i] for i in range(n)}
        self.reasons: dict[int, list[str]] = {i: [] for i in range(n)}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int, reason: str) -> None:
        keep, gone = sorted((self.find(a), self.find(b)))
        self.parent[gone] = keep
        self.members[keep] += self.members.pop(gone)
        self.reasons[keep] += [*self.reasons.pop(gone), reason]

    def groups(self) -> list[tuple[list[int], list[str]]]:
        return [(sorted(m), self.reasons[root]) for root, m in sorted(self.members.items())]


def _name_link(a: C, b: C, threshold: float) -> tuple[int, float, str] | None:
    na, nb = normal_form(a.name), normal_form(b.name)
    if na == nb:
        return 3, 1.0, "same name"
    if nb in {normal_form(x) for x in a.aliases} or na in {normal_form(x) for x in b.aliases}:
        return 2, 1.0, "claimed as an alias"
    ratio = SequenceMatcher(None, na, nb).ratio()
    if ratio >= threshold:
        return 1, ratio, f"similar names ({ratio:.2f})"
    return None


def _cluster(
    items: Sequence[C],
    link: Callable[[C, C], tuple[int, float, str] | None],
    refuse: Callable[[list[C], list[C], int], str | None],
    kind: Literal["type", "predicate"],
    decisions: list[MergeDecision],
) -> list[tuple[list[C], list[str]]]:
    links = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            found = link(items[i], items[j])
            if found is not None:
                links.append((-found[0], -found[1], i, j, found))
    clusters = _Clusters(len(items))
    reported: set[frozenset[str]] = set()
    for _, _, i, j, (priority, _, reason) in sorted(links):
        ri, rj = clusters.find(i), clusters.find(j)
        if ri == rj:
            continue
        left = [items[k] for k in clusters.members[ri]]
        right = [items[k] for k in clusters.members[rj]]
        refused = refuse(left, right, priority)
        if refused is None:
            clusters.union(i, j, f"{items[i].name} ~ {items[j].name}: {reason}")
            continue
        pair = frozenset({items[i].name, items[j].name})
        if pair not in reported:
            reported.add(pair)
            decisions.append(
                MergeDecision(
                    kind=kind,
                    action="blocked",
                    into=items[i].name,
                    names=(items[i].name, items[j].name),
                    reason=f"{reason}, but {refused}",
                )
            )
    return [([items[k] for k in members], reasons) for members, reasons in clusters.groups()]


def _preference(candidate: TypeCandidate | PredicateCandidate) -> tuple[int, int, int, int, str]:
    """Which member names a cluster: the model's, then the best supported, then the shortest."""
    return (
        0 if candidate.proposer == "llm" else 1,
        candidate.rank if candidate.rank is not None else 1 << 30,
        -candidate.support,
        len(candidate.name),
        candidate.name,
    )


def _union_evidence(members: Sequence[TypeCandidate | PredicateCandidate]) -> tuple[Span, ...]:
    spans = {(s.doc_id, s.start, s.end): s for m in members for s in m.evidence}
    return tuple(spans.values())


def _aliases(name: str, members: Sequence[TypeCandidate | PredicateCandidate]) -> tuple[str, ...]:
    spellings = (a for m in members for a in (m.name, *m.aliases))
    out: dict[str, str] = {}
    for alias in spellings:
        if fold(alias) != fold(name):
            out.setdefault(fold(alias), alias)
    return tuple(out.values())


def _ranks(members: Sequence[TypeCandidate | PredicateCandidate]) -> int | None:
    ranks = [m.rank for m in members if m.rank is not None]
    return min(ranks) if ranks else None


def _proposers(members: Sequence[TypeCandidate | PredicateCandidate]) -> str:
    return "+".join(sorted({p for m in members for p in m.proposer.split("+")}))


def _record(
    kind: Literal["type", "predicate"],
    into: str,
    members: Sequence[TypeCandidate | PredicateCandidate],
    reasons: list[str],
    decisions: list[MergeDecision],
) -> None:
    names = tuple(dict.fromkeys(m.name for m in members))
    if len(members) > 1:
        decisions.append(
            MergeDecision(
                kind=kind, action="merged", into=into, names=names, reason="; ".join(reasons)
            )
        )


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


def _merge_types(
    items: list[TypeCandidate], threshold: float, decisions: list[MergeDecision]
) -> list[TypeCandidate]:
    ancestry: dict[str, set[str]] = {}
    for t in items:
        ancestry.setdefault(normal_form(t.name), set()).update(map(normal_form, t.parents))

    def above(name: str) -> set[str]:
        seen: set[str] = set()
        stack = list(ancestry.get(name, ()))
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(ancestry.get(current, ()))
        return seen

    def refuse(left: list[TypeCandidate], right: list[TypeCandidate], _: int) -> str | None:
        ln = {normal_form(t.name) for t in left}
        rn = {normal_form(t.name) for t in right}
        if any(ln & above(n) for n in rn) or any(rn & above(n) for n in ln):
            return "one is a kind of the other"
        return None

    out: list[TypeCandidate] = []
    for members, reasons in _cluster(
        items, lambda a, b: _name_link(a, b, threshold), refuse, "type", decisions
    ):
        lead = min(members, key=_preference)
        names = {normal_form(m.name) for m in members}
        merged = TypeCandidate(
            name=lead.name,
            proposer=_proposers(members),
            description=next(
                (m.description for m in sorted(members, key=_preference) if m.description), None
            ),
            parents=tuple(
                dict.fromkeys(p for m in members for p in m.parents if normal_form(p) not in names)
            ),
            aliases=_aliases(lead.name, members),
            keys=tuple(dict.fromkeys(k for m in members for k in m.keys)),
            examples=tuple(dict.fromkeys(e for m in members for e in m.examples)),
            evidence=_union_evidence(members),
            rank=_ranks(members),
        )
        _record("type", lead.name, members, reasons, decisions)
        out.append(merged)
    renamed = _type_names(out)
    return [
        t.model_copy(
            update={
                "parents": tuple(
                    dict.fromkeys(
                        p
                        for p in (renamed.get(normal_form(q), q) for q in t.parents)
                        if p != t.name
                    )
                )
            }
        )
        for t in out
    ]


def _type_names(types: Sequence[TypeCandidate]) -> dict[str, str]:
    """Every spelling a merged type answered to, by normal form, to the name it kept."""
    renamed: dict[str, str] = {}
    for t in types:
        renamed[normal_form(t.name)] = t.name
    for t in types:
        for alias in t.aliases:
            renamed.setdefault(normal_form(alias), t.name)
    return renamed


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #


def _merge_predicates(
    items: list[PredicateCandidate],
    threshold: float,
    min_shared: int,
    lineage: Callable[[str], set[str]],
    decisions: list[MergeDecision],
) -> list[PredicateCandidate]:
    def link(a: PredicateCandidate, b: PredicateCandidate) -> tuple[int, float, str] | None:
        named = _name_link(a, b, threshold)
        if named is not None:
            return named
        shared = len(set(a.observations) & set(b.observations))
        smaller = min(len(a.observations), len(b.observations))
        if shared >= min_shared and smaller and shared >= 0.5 * smaller:
            return 1, shared / smaller, f"{shared} shared (subject, value) pairs"
        return None

    def refuse(
        left: list[PredicateCandidate], right: list[PredicateCandidate], priority: int
    ) -> str | None:
        for a in left:
            for b in right:
                if not _compatible(a.range, b.range, lineage):
                    return f"ranges conflict ({a.name}: {a.range}, {b.name}: {b.range})"
        if priority < 3:
            ld = {d for p in left for d in p.domain}
            rd = {d for p in right for d in p.domain}
            if ld and rd and not ld & rd:
                return f"domains do not overlap ({', '.join(sorted(ld))} / {', '.join(sorted(rd))})"
        return None

    out: list[PredicateCandidate] = []
    for members, reasons in _cluster(items, link, refuse, "predicate", decisions):
        lead = min(members, key=_preference)
        cardinality: Cardinality = (
            "multi" if any(m.cardinality == "multi" for m in members) else "single"
        )
        merged = PredicateCandidate(
            name=lead.name,
            proposer=_proposers(members),
            description=next(
                (m.description for m in sorted(members, key=_preference) if m.description), None
            ),
            domain=tuple(
                dict.fromkeys(d for m in sorted(members, key=_preference) for d in m.domain)
            ),
            range=_widest([m.range for m in sorted(members, key=_preference)], lineage),
            cardinality=cardinality,
            aliases=_aliases(lead.name, members),
            examples=tuple(dict.fromkeys(e for m in members for e in m.examples))[:5],
            observations=tuple(dict.fromkeys(o for m in members for o in m.observations)),
            evidence=_union_evidence(members),
            rank=_ranks(members),
        )
        _record("predicate", lead.name, members, reasons, decisions)
        out.append(merged)

    # Same name, conflicting range: the ontology can hold only one of them.
    kept: dict[str, PredicateCandidate] = {}
    for candidate in sorted(out, key=_preference):
        winner = kept.get(fold(candidate.name))
        if winner is None:
            kept[fold(candidate.name)] = candidate
            continue
        decisions.append(
            MergeDecision(
                kind="predicate",
                action="dropped",
                into=winner.name,
                names=(candidate.name,),
                reason=(
                    f"{winner.name!r} ({winner.range}, support {winner.support}) was kept; "
                    f"this one's range {candidate.range!r} conflicts with it"
                ),
            )
        )
    survivors = {id(c) for c in kept.values()}
    return [c for c in out if id(c) in survivors]


def _compatible(a: str, b: str, lineage: Callable[[str], set[str]]) -> bool:
    if a == b or {a, b} <= _NUMERIC or {a, b} <= _TEMPORAL:
        return True
    if a in _JSON_TYPES or b in _JSON_TYPES:
        return False
    return a in lineage(b) or b in lineage(a)


def _widest(ranges: Sequence[str], lineage: Callable[[str], set[str]]) -> str:
    distinct = list(dict.fromkeys(ranges))
    if len(distinct) == 1:
        return distinct[0]
    if set(distinct) <= _NUMERIC:
        return "number"
    if set(distinct) <= _TEMPORAL:
        return "datetime"
    # Entity types that passed `_compatible` are one lineage: keep the ancestor.
    return next((r for r in distinct if all(r in lineage(o) for o in distinct)), distinct[0])


__all__ = ["MergeDecision", "Merged", "merge"]
