"""Entity resolution with blocking: propose links, change keys only on proof.

The recipe is the standard one, in its standard order: normalise, block on cheap
keys, match on strong identifiers, score weak ones, let a strong identifier that
*disagrees* kill the match, nudge on non-identifying attributes, drop what falls
below the threshold. Two parts of it are what the surveyed resolvers leave out.

**The disagreement rule.** Two companies called "Acme Corporation" and "Acme
Corporation Ltd" with registration numbers DE-114322 and GB-889401 are two
companies, and no amount of name similarity outweighs that. Evidence against
beats evidence for, so the pair is recorded as `DIFFERENT` with a `reason` that
names both identifiers (DECISIONS #16) — the rejection is a query, not a mystery.

**Nothing is destroyed.** Only a `SAME_AS` — a shared external id or domain,
which is proof rather than resemblance — re-keys facts onto one canonical
entity, and the losing keys, labels and aliases survive as that entity's
aliases. A `SIMILAR` is a proposal with a score and changes no key: a wrong
merge silently corrupts every query that touches the node, and a missed one
costs a duplicate that the link still points at. Thresholds are wrong on the
first try, and a link can be re-run at a new threshold; a merge cannot.

Blocking is what keeps this from being all-pairs. Only entities of one type
sharing the first or last token of a name, a domain or an external id are ever
compared, so the work grows with block sizes rather than with the square of the
corpus. A name block too large to compare — a first token like "bank" — is split
by `country` where entities carry one, and skipped where it is still too large;
exact-key blocks that large are compared as a star, since `SAME_AS` is transitive.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from odke.corroborate.normalize import name_key
from odke.corroborate.provenance import NAME_KEY
from odke.stages import EntityIndex
from odke.types import Entity, EntityLink, Fact, LinkKind, Resolution

# What `Entity.resolution.linker` says when this resolver decided.
LINKER = "odke.native"

_DOMAIN = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?"
    r"((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,24})\.?(?:[:/?#]\S*)?",
    re.I,
)
_SCHEME = re.compile(r"[A-Za-z][\w.-]*")


def domain_of(text: str) -> str | None:
    """The host an alias names, lowercased and without `www.`, or `None`.

    `"https://www.acme.com/about"` and `"acme.com"` are both `acme.com`. Text with
    a space in it is a name, not a domain.
    """
    m = _DOMAIN.fullmatch(text.strip())
    return m[1].lower() if m else None


def _identifier(raw: str) -> tuple[str, str]:
    # "wikidata:Q95" is scheme "wikidata"; a bare "DE-114322" has none. Ids from
    # two different authorities neither match nor disagree — a Q-id and a
    # registration number are not two answers to one question.
    scheme, sep, value = raw.partition(":")
    if sep and value and not value.startswith("//") and _SCHEME.fullmatch(scheme):
        return scheme.casefold(), "".join(value.split()).casefold()
    return "", "".join(raw.split()).casefold()


def name_similarity(a: str, b: str) -> float:
    """How alike two name keys are, in [0, 1]: the better of two cheap measures.

    `difflib`'s ratio catches spelling variation; a token overlap catches
    reordering. An initial matches a word it could abbreviate for half a token,
    so "j smith" and "john smith" come out near 0.82 — close, and below the
    default threshold, which is the right answer for two people who may differ.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return max(ratio, _token_overlap(a.split(), b.split()))


def _token_overlap(left: list[str], right: list[str]) -> float:
    remaining = list(right)
    matched = 0.0
    for token in left:
        if token in remaining:
            remaining.remove(token)
            matched += 1
            continue
        for other in remaining:
            if (len(token) == 1 or len(other) == 1) and token[0] == other[0]:
                remaining.remove(other)
                matched += 0.5
                break
    union = len(left) + len(right) - matched
    return matched / union if union else 0.0


@dataclass(frozen=True)
class _Profile:
    """What resolution compares about one entity, computed once."""

    entity: Entity
    names: tuple[str, ...]
    # scheme -> (compacted value, value as the source wrote it)
    ids: dict[str, tuple[str, str]]
    domains: frozenset[str]
    attributes: dict[str, Any]

    @property
    def key(self) -> str:
        return self.entity.key


def _profile(entity: Entity) -> _Profile:
    primary = entity.attributes.get(NAME_KEY)
    names = [primary if isinstance(primary, str) else name_key(entity.label or entity.key)]
    domains: set[str] = set()
    for alias in entity.aliases:
        if found := domain_of(alias):
            domains.add(found)
        elif (alias_key := name_key(alias)) not in names:
            names.append(alias_key)
    ids = {}
    if entity.external_id:
        scheme, value = _identifier(entity.external_id)
        ids[scheme] = (value, entity.external_id)
    # Everything the caller put in attributes that is not our own bookkeeping
    # is a non-identifying attribute: it nudges a score, it never decides.
    attributes = {k: v for k, v in entity.attributes.items() if not k.startswith("odke.")}
    return _Profile(entity, tuple(n for n in names if n), ids, frozenset(domains), attributes)


def _fold(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value


def _candidate_indices(profiles: Sequence[_Profile], max_block: int) -> set[tuple[int, int]]:
    name_blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    exact_blocks: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, p in enumerate(profiles):
        kind = p.entity.type
        tokens: set[str] = set()
        for name in p.names:
            words = name.split()
            tokens.update((words[0], words[-1]))
        for token in tokens:
            name_blocks[(kind, token)].append(i)
        for found in p.domains:
            exact_blocks[("domain", kind, found)].append(i)
        for scheme, (value, _) in p.ids.items():
            exact_blocks[("id", kind, scheme, value)].append(i)

    pairs: set[tuple[int, int]] = set()
    for members in exact_blocks.values():
        if len(members) <= max_block:
            pairs.update(combinations(members, 2))
        else:
            pairs.update((members[0], j) for j in members[1:])
    for members in name_blocks.values():
        if len(members) <= max_block:
            pairs.update(combinations(members, 2))
            continue
        by_country: dict[Any, list[int]] = defaultdict(list)
        for i in members:
            by_country[_fold(profiles[i].attributes.get("country"))].append(i)
        for sub in by_country.values():
            if len(sub) <= max_block:
                pairs.update(combinations(sub, 2))
    return pairs


def _distinct(entities: Iterable[Entity]) -> list[Entity]:
    seen: dict[str, Entity] = {}
    for entity in entities:
        seen.setdefault(entity.key, entity)
    return [seen[k] for k in sorted(seen)]


def candidate_pairs(entities: Iterable[Entity], *, max_block: int = 500) -> set[tuple[str, str]]:
    """The key pairs blocking lets through — every comparison resolution makes.

    Exposed so the cost of a configuration can be measured before it is run:
    the size of this set, not the corpus size squared, is the work.
    """
    profiles = [_profile(e) for e in _distinct(entities)]
    return {(profiles[i].key, profiles[j].key) for i, j in _candidate_indices(profiles, max_block)}


class NativeResolver:
    """The dependency-free resolver: blocking, strong ids, a scored name match.

    `threshold` is the weak-match bar for `SIMILAR` — high by default, because a
    link proposed wrongly is a false lead for whoever acts on it. `nudge_up` and
    `nudge_down` are what one agreeing or disagreeing non-identifying attribute
    (a country, an industry) adds or takes away; disagreement costs more, for the
    same reason the disagreement rule exists. `max_block` caps a block before it
    is split or skipped.

    Returns every fact with `Entity.resolution` stamped — `external_id` where a
    shared identifier settled a merge, `linker="odke.native"` otherwise, with
    `score=1.0` on a merge decided by a shared domain — and every link, including
    each `DIFFERENT`. An identity decided upstream keeps its own provenance, and
    an entity the caller keyed (`method="caller"`) is the canonical one whenever
    it is in a merge.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.9,
        nudge_up: float = 0.05,
        nudge_down: float = 0.15,
        max_block: int = 500,
    ) -> None:
        self.threshold = threshold
        self.nudge_up = nudge_up
        self.nudge_down = nudge_down
        self.max_block = max_block

    def resolve(
        self, facts: Iterable[Fact], index: EntityIndex
    ) -> tuple[list[Fact], list[EntityLink]]:
        facts = list(facts)
        mentioned = [e for f in facts for e in (f.subject, f.object_entity) if e is not None]
        profiles = [_profile(e) for e in _distinct([*index.values(), *mentioned])]

        links: list[EntityLink] = []
        proofs: list[tuple[int, int, EntityLink, bool]] = []
        for i, j in sorted(_candidate_indices(profiles, self.max_block)):
            decided = self._judge(profiles[i], profiles[j])
            if decided is None:
                continue
            link, by_id = decided
            if link.kind is LinkKind.SAME_AS:
                proofs.append((i, j, link, by_id))
            else:
                links.append(link)

        parent = list(range(len(profiles)))
        cluster_ids = [dict(p.ids) for p in profiles]
        # Whether every proof in the cluster was a shared external id.
        by_id_only = [True] * len(profiles)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j, link, by_id in proofs:
            ri, rj = find(i), find(j)
            if ri != rj:
                # SAME_AS is transitive, and that is exactly how two entities
                # whose ids disagree get joined through a third that has none.
                clash = [
                    f"external_id mismatch: {cluster_ids[ri][s][1]} vs {cluster_ids[rj][s][1]}"
                    for s in sorted(cluster_ids[ri].keys() & cluster_ids[rj].keys())
                    if cluster_ids[ri][s][0] != cluster_ids[rj][s][0]
                ]
                if clash:
                    links.append(
                        EntityLink(
                            source_key=link.source_key,
                            target_key=link.target_key,
                            kind=LinkKind.DIFFERENT,
                            score=self._weak(profiles[i], profiles[j]),
                            reason=f"{'; '.join(clash)} (refused a {link.reason} that "
                            f"would have joined them)",
                        )
                    )
                    continue
                parent[rj] = ri
                cluster_ids[ri] = {**cluster_ids[rj], **cluster_ids[ri]}
                by_id_only[ri] = by_id_only[ri] and by_id_only[rj]
            root = find(i)
            by_id_only[root] = by_id_only[root] and by_id
            links.append(link)

        clusters: dict[int, list[Entity]] = defaultdict(list)
        for i, p in enumerate(profiles):
            clusters[find(i)].append(p.entity)
        known = set(index.keys())
        replacement: dict[str, Entity] = {}
        for root, members in clusters.items():
            if len(members) > 1:
                merged = _merge(members, known, by_id_only[root])
                replacement.update(dict.fromkeys((m.key for m in members), merged))

        unmerged = Resolution(method="linker", linker=LINKER)

        def settle(entity: Entity) -> Entity:
            if entity.key in replacement:
                return replacement[entity.key]
            if entity.resolution is None:
                return entity.model_copy(update={"resolution": unmerged})
            return entity

        resolved = [
            f.model_copy(
                update={
                    "subject": settle(f.subject),
                    "object_entity": settle(f.object_entity) if f.object_entity else None,
                }
            )
            for f in facts
        ]
        return resolved, links

    def _judge(self, a: _Profile, b: _Profile) -> tuple[EntityLink, bool] | None:
        agree: list[str] = []
        against: list[str] = []
        by_id = False
        for scheme in sorted(a.ids.keys() & b.ids.keys()):
            (value_a, raw_a), (value_b, raw_b) = a.ids[scheme], b.ids[scheme]
            if value_a == value_b:
                agree.append(f"external_id match: {raw_a}")
                by_id = True
            else:
                against.append(f"external_id mismatch: {raw_a} vs {raw_b}")
        if a.domains and b.domains:
            if shared := a.domains & b.domains:
                agree.append(f"domain match: {min(shared)}")
            else:
                against.append(
                    f"domain mismatch: {', '.join(sorted(a.domains))} "
                    f"vs {', '.join(sorted(b.domains))}"
                )

        score = self._weak(a, b)
        if against:
            # Evidence against beats evidence for. Recorded when there was a
            # match to kill — a strong one, or a name above the bar — so that
            # every pair of unrelated entities in a block is not a link.
            if agree or score >= self.threshold:
                return self._link(a, b, LinkKind.DIFFERENT, score, "; ".join(against)), False
            return None
        if agree:
            return self._link(a, b, LinkKind.SAME_AS, 1.0, "; ".join(agree)), by_id
        if score >= self.threshold:
            return self._link(a, b, LinkKind.SIMILAR, score, None), False
        return None

    def _weak(self, a: _Profile, b: _Profile) -> float:
        best = max((name_similarity(x, y) for x in a.names for y in b.names), default=0.0)
        if best == 0.0:
            return 0.0
        for key in sorted(a.attributes.keys() & b.attributes.keys()):
            same = _fold(a.attributes[key]) == _fold(b.attributes[key])
            best += self.nudge_up if same else -self.nudge_down
        return round(min(1.0, max(0.0, best)), 4)

    @staticmethod
    def _link(
        a: _Profile, b: _Profile, kind: LinkKind, score: float, reason: str | None
    ) -> EntityLink:
        return EntityLink(source_key=a.key, target_key=b.key, kind=kind, score=score, reason=reason)


def _merge(members: list[Entity], known: set[str], by_id: bool) -> Entity:
    """One entity for a `SAME_AS` cluster, keeping every name any member had."""

    def preference(e: Entity) -> tuple[bool, bool, str]:
        # The caller's own key first, then one already in the store, then the
        # smallest key, so a re-run picks the same canonical entity.
        caller = e.resolution is not None and e.resolution.method == "caller"
        return (not caller, e.key not in known, e.key)

    canonical = min(members, key=preference)
    others = sorted((m for m in members if m is not canonical), key=lambda e: e.key)
    aliases = list(canonical.aliases)
    seen: set[str | None] = {canonical.key, canonical.label, *aliases}
    attributes: dict[str, Any] = {}
    for other in others:
        for name in (other.key, other.label, *other.aliases):
            if name and name not in seen:
                seen.add(name)
                aliases.append(name)
        attributes = {**other.attributes, **attributes}
    external_id = canonical.external_id or next(
        (o.external_id for o in others if o.external_id), None
    )

    if canonical.resolution is not None and canonical.resolution.method == "caller":
        resolution = canonical.resolution
    elif by_id:
        resolution = Resolution(method="external_id")
    else:
        resolution = Resolution(method="linker", linker=LINKER, score=1.0)
    return canonical.model_copy(
        update={
            "aliases": tuple(aliases),
            "attributes": {**attributes, **canonical.attributes},
            "external_id": external_id,
            "resolution": resolution,
        }
    )


__all__ = ["LINKER", "NativeResolver", "candidate_pairs", "domain_of", "name_similarity"]
