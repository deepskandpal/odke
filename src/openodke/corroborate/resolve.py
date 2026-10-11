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

**Against the store.** Given a `StoreLookup`, the batch is also resolved against
what the store already holds, without loading it: the lookup returns, per
entity, the store's entities sharing a block key with it, and each pair is
judged by the same rules. A proof re-keys the *incoming* facts onto the store's
key, carrying the store's entity exactly as the store holds it, so writing it
changes nothing on the node; anything weaker is a `SIMILAR` link. Store
entities are never compared with each other (DECISIONS #31).

**The pair judge.** Given a `PairJudge`, the pairs no rule settled whose name
score is in the band just below the threshold are put to a model, in both
orders (DECISIONS #34). Its "same" is a `SIMILAR` link and never a merge, its
"different" a `DIFFERENT` link, and its unsure no link at all.

**Normalising a batch.** With `normalize_batch`, two mentions the batch
itself introduces may become one entity on resemblance, not only on
proof, when nothing keeps them apart (DECISIONS #43): a pair above the
threshold whose names carry no different number or legal form and, with
`embed`, whose sentences are alike, or a pair in the judge's band the judge
calls the same. They are re-keyed onto one canonical entity that keeps every
name as an alias, before anything is written. A key the store holds is never
part of such a merge.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

from openodke.corroborate.normalize import legal_form, name_key
from openodke.corroborate.provenance import NAME_KEY
from openodke.ontology import Ontology
from openodke.stages import EntityIndex, StoreLookup
from openodke.types import Document, Entity, EntityLink, Fact, LinkKind, Resolution

if TYPE_CHECKING:
    # Imported where it is used: the judge reaches `openodke.ground`, whose
    # span locator imports this module.
    from openodke.corroborate.judge import PairJudge

# What `Entity.resolution.linker` says when this resolver decided.
LINKER = "odke.native"
# Texts in, one vector per text out, in order. Any embedding model fits.
Embed: TypeAlias = Callable[[Sequence[str]], Sequence[Sequence[float]]]
# The cosine between two look-alikes' sentences below which they stay two
# entities. Set before anything was measured: the caller's model sets the scale.
DEFAULT_CONTEXT_FLOOR = 0.5
# A Roman numeral to 39, the ones that number kings, wars and sequels. Longer
# runs of these letters are words ("mix", "civil").
_ROMAN = re.compile(r"(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})")

_DOMAIN = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?"
    r"((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,24})\.?(?::\d+)?/?",
    re.I,
)
_SCHEME = re.compile(r"[A-Za-z][\w.-]*")


def domain_of(text: str) -> str | None:
    """The host an alias names, lowercased and without `www.`, or `None`.

    `"https://www.acme.com/"` and `"acme.com"` are both `acme.com`. A URL with a
    path names a page on a host, not the host: two LinkedIn profiles share
    `linkedin.com` and nothing else, so `"https://www.linkedin.com/in/alice-chen"`
    is not a domain and is compared as a name. Nor is text with a space in it.
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


class BlockKeys(NamedTuple):
    """What blocking compares an entity on. Within one type, one shared key is enough."""

    # The first and last token of each name key.
    tokens: frozenset[str]
    # (scheme, compacted value): `wikidata:Q95` is ("wikidata", "q95").
    ids: frozenset[tuple[str, str]]
    domains: frozenset[str]


def block_keys(entity: Entity) -> BlockKeys:
    """The keys the resolver blocks `entity` on, which a `StoreLookup` looks up by."""
    return _keys(_profile(entity))


def _numbers(entity: Entity) -> tuple[str, ...]:
    """The numbers in an entity's name, sorted: tokens with a digit, and Roman numerals.

    Read from the label's organisation-style key, which keeps the "II" a
    person's key drops, so "Robert II" and "Robert III" still differ here.
    """
    words = name_key(entity.label or entity.key).split()
    return tuple(sorted(w for w in words if any(c.isdigit() for c in w) or _ROMAN.fullmatch(w)))


def _disagree(a: Entity, b: Entity) -> tuple[str, str] | None:
    """What in two look-alike names says they are two things, and the reason; or None.

    Different numbers ("1900" and "1902 County Championship", "Robert II" and
    "Robert III") or different legal forms ("Acme GmbH" and "Acme Ltd"), each
    only when both names carry one: a name without a number may be either.
    """
    numbers = _numbers(a), _numbers(b)
    if all(numbers) and numbers[0] != numbers[1]:
        said = " vs ".join(" ".join(n) for n in numbers)
        return "numbers", f"the names carry different numbers ({said})"
    forms = legal_form(a.label or a.key), legal_form(b.label or b.key)
    if forms[0] and forms[1] and forms[0] != forms[1]:
        return "forms", f"the names carry different legal forms ({forms[0]} vs {forms[1]})"
    return None


def _fold_type(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def _type_names(ontology: Ontology | None) -> dict[str, str]:
    """Each type's name and alias, case- and punctuation-blind, to the type's own name."""
    names: dict[str, str] = {}
    if ontology is None:
        return names
    for key in ontology.types:
        names.setdefault(_fold_type(key), key)
    for key, entity_type in ontology.types.items():
        for alias in entity_type.aliases:
            names.setdefault(_fold_type(alias), key)
    return names


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    return [float(x) / norm for x in vector] if norm else [0.0 for _ in vector]


def _keys(p: _Profile) -> BlockKeys:
    tokens: set[str] = set()
    for name in p.names:
        words = name.split()
        tokens.update((words[0], words[-1]))
    ids = frozenset((scheme, value) for scheme, (value, _) in p.ids.items())
    return BlockKeys(frozenset(tokens), ids, p.domains)


def _fold(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value


def _candidate_indices(
    profiles: Sequence[_Profile], max_block: int, types: Mapping[str, str] | None = None
) -> set[tuple[int, int]]:
    name_blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    exact_blocks: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, p in enumerate(profiles):
        kind = p.entity.type
        if types:
            # An alias of an ontology type is that type, however it is spelled.
            kind = types.get(_fold_type(kind), kind)
        keys = _keys(p)
        for token in keys.tokens:
            name_blocks[(kind, token)].append(i)
        for found in keys.domains:
            exact_blocks[("domain", kind, found)].append(i)
        for scheme, value in keys.ids:
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


def candidate_pairs(entities: Iterable[Entity], *, max_block: int = 100) -> set[tuple[str, str]]:
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
    is split or skipped. A block is compared pair by pair, so the cap is what
    bounds the work: 100 entities are 4,950 comparisons, 500 would be 124,750.

    Returns every fact with `Entity.resolution` stamped — `external_id` where a
    shared identifier settled a merge, `linker="odke.native"` otherwise, with
    `score=1.0` on a merge decided by a shared domain — and every link, including
    each `DIFFERENT`. An identity decided upstream keeps its own provenance, and
    an entity the caller keyed (`method="caller"`) is the canonical one whenever
    it is in a merge.

    `lookup` resolves the batch against the store as well (DECISIONS #31). Each
    entity of the batch is compared with the candidates the lookup returns for
    it, within its type, by the same rules and the same `threshold`. On proof,
    its facts take the store's key and the store's entity exactly as the store
    holds it, so writing them changes nothing on the node. A batch that states
    the stored key itself, or a key the caller chose, keeps its own entity, as
    it would with no store. Anything weaker is a `SIMILAR` link to the store
    entity. Store entities are never compared with each other, and `stats`
    counts what the last call found in the store. Without a lookup, nothing
    here differs from a resolver that has never heard of one.

    `judge` puts each pair no rule settled, and whose score is at least the
    judge's `low` (0.7) and below `threshold`, to a `PairJudge` (DECISIONS
    #34): asked in both orders, its "same" is a `SIMILAR` link (a merge when
    the batch is normalised and both are its own mentions), its "different"
    a `DIFFERENT` link, and anything else no link.
    A pair with a disagreeing id or domain is never asked. `stats["judge"]`
    is the judge's own count, and `documents` is where it reads contexts.
    Without a judge, no pair below the threshold is asked about.

    `normalize_batch` makes look-alikes the batch introduces one entity when
    nothing keeps them apart (DECISIONS #43). Blocked as
    above, by type (an ontology type's aliases are that type, given
    `ontology`) and a shared token, id or domain, a pair of mentions no store
    holds is merged when it is a `SIMILAR` whose names carry no different
    numbers ("1900" and "1902 County Championship") or legal forms ("Acme
    GmbH" and "Acme Ltd") and, given `embed`, whose sentences' cosine is at
    least `context_floor`; or when the judge, or a person in its place, says
    the pair in its band is the same and the names do not disagree. A merge
    is a `SAME_AS` link with the name score and a reason starting "in-batch
    merge", and re-keys every mention onto one canonical entity as a proof
    does, every name kept as an alias and `Entity.resolution.score` the
    weakest score the group was joined at. A mention alike to two entities
    kept apart merges with neither. Merges are made strongest first, and one
    that would join a pair kept apart, two keys a caller chose, or two
    disagreeing ids is refused. A pair kept apart keeps its `SIMILAR` link,
    with a reason that says why. `stats["batch"]` counts it all.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.9,
        nudge_up: float = 0.05,
        nudge_down: float = 0.15,
        max_block: int = 100,
        lookup: StoreLookup | None = None,
        judge: PairJudge | None = None,
        normalize_batch: bool = False,
        embed: Embed | None = None,
        context_floor: float = DEFAULT_CONTEXT_FLOOR,
        ontology: Ontology | None = None,
    ) -> None:
        if judge is not None and judge.low > threshold:
            raise ValueError(
                f"the judge's band starts at {judge.low}, above the threshold {threshold}"
            )
        if embed is not None and not normalize_batch:
            raise ValueError("embed is the batch's context check: it needs normalize_batch=True")
        if not -1.0 <= context_floor <= 1.0:
            raise ValueError(f"context_floor is a cosine, in [-1, 1], got {context_floor}")
        self.threshold = threshold
        self.nudge_up = nudge_up
        self.nudge_down = nudge_down
        self.max_block = max_block
        self.lookup = lookup
        self.judge = judge
        self.normalize_batch = normalize_batch
        self.embed = embed
        self.context_floor = context_floor
        self.ontology = ontology
        self._types = _type_names(ontology)
        self._documents: dict[str, Document] = {}
        # What the last `resolve()` found in the store, what the judge has
        # done and what the batch merged; None while none of the three is on.
        self.stats: dict[str, Any] | None = None

    @property
    def documents(self) -> dict[str, Document] | None:
        """The texts the judge and the context check read, filled as the corroborator's are."""
        if self.judge is not None:
            return self.judge.documents
        return self._documents if self.embed is not None else None

    def resolve(
        self, facts: Iterable[Fact], index: EntityIndex
    ) -> tuple[list[Fact], list[EntityLink]]:
        facts = list(facts)
        mentioned = [e for f in facts for e in (f.subject, f.object_entity) if e is not None]
        profiles = [_profile(e) for e in _distinct([*index.values(), *mentioned])]
        size = len(profiles)
        pairs = sorted(_candidate_indices(profiles, self.max_block, self._types))
        # The same node by key: a store entity the batch also states.
        same_node: list[tuple[int, int]] = []
        if self.lookup is not None:
            store_pairs, same_node = self._stored(profiles, _distinct(mentioned))
            pairs += store_pairs
        # What a batch merge may join: an entity a fact of the batch states, by a
        # key the store does not hold. The index is the batch's own, as the
        # pipeline passes it, and an entity only the index has is not a mention.
        fresh: set[int] = set()
        if self.normalize_batch:
            stated = {e.key for e in mentioned} - {profiles[i].key for i, _ in same_node}
            fresh = {i for i in range(size) if profiles[i].key in stated}

        links: list[EntityLink] = []
        proofs: list[tuple[int, int, EntityLink, bool]] = []
        # Pairs no rule settled, scored in the judge's band, with their score.
        open_pairs: list[tuple[int, int, float]] = []
        # Batch pairs that may merge, and whether the judge or a person read them.
        alike: list[tuple[int, int, EntityLink, bool]] = []
        # Batch pairs a merge must never join, through any chain.
        apart: list[tuple[int, int]] = []
        # Links to a store entity, by kind: what `stats` reports.
        to_store: Counter[str] = Counter()

        def add(i: int, j: int, link: EntityLink) -> None:
            links.append(link)
            if link.kind is LinkKind.DIFFERENT:
                apart.append((i, j))
            if j >= size:
                to_store[link.kind.value] += 1

        for i, j in pairs:
            decided = self._judge(profiles[i], profiles[j])
            if isinstance(decided, float):
                if self.judge is not None and decided >= self.judge.low:
                    open_pairs.append((i, j, decided))
                continue
            if decided is None:
                continue
            link, by_id = decided
            if link.kind is LinkKind.SAME_AS:
                proofs.append((i, j, link, by_id))
            elif link.kind is LinkKind.SIMILAR and i in fresh and j in fresh:
                alike.append((i, j, link, False))
            else:
                add(i, j, link)
        if self.judge is not None and open_pairs:
            asked = self._ask(profiles, open_pairs, facts)
            for (i, j, _), judged in zip(open_pairs, asked, strict=True):
                if judged is None:
                    continue
                if judged.kind is LinkKind.SIMILAR and i in fresh and j in fresh:
                    alike.append((i, j, judged, True))
                else:
                    add(i, j, judged)

        # The context check: what keeps two look-alikes apart, else a merge.
        kept: Counter[str] = Counter()
        # The pairs to merge: the link as the rules or the judge made it, and what it rests on.
        merges: list[tuple[int, int, EntityLink, str]] = []
        vectors = self._embedded(profiles, alike, facts) if alike and self.embed else {}
        for i, j, link, read in alike:
            a, b = profiles[i], profiles[j]
            context = _cosine(vectors.get(a.key), vectors.get(b.key)) if not read else None
            if disagreed := _disagree(a.entity, b.entity):
                kept[disagreed[0]] += 1
                why = disagreed[1]
            elif context is not None and context < self.context_floor:
                kept["context"] += 1
                why = f"the contexts differ ({context:.4f} < {self.context_floor})"
            else:
                said = link.reason or f"names {link.score}"
                if context is not None:
                    said += f", contexts {context:.4f}"
                merges.append((i, j, link, said))
                continue
            add(i, j, _not_merged(link, why))
            apart.append((i, j))

        parent = list(range(len(profiles)))
        cluster_ids = [dict(p.ids) for p in profiles]
        # Whether every proof in the cluster was a shared external id.
        by_id_only = [True] * len(profiles)
        # The weakest score a merge joined the cluster at, and whether a caller keyed it.
        weakest = [1.0] * len(profiles)
        caller = [_caller(p.entity) for p in profiles]
        members = {i: [i] for i in range(len(profiles))}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def join(ri: int, rj: int) -> None:
            parent[rj] = ri
            cluster_ids[ri] = {**cluster_ids[rj], **cluster_ids[ri]}
            by_id_only[ri] = by_id_only[ri] and by_id_only[rj]
            weakest[ri] = min(weakest[ri], weakest[rj])
            caller[ri] = caller[ri] or caller[rj]
            members[ri].extend(members.pop(rj))

        def clash(ri: int, rj: int) -> list[str]:
            return [
                f"external_id mismatch: {cluster_ids[ri][s][1]} vs {cluster_ids[rj][s][1]}"
                for s in sorted(cluster_ids[ri].keys() & cluster_ids[rj].keys())
                if cluster_ids[ri][s][0] != cluster_ids[rj][s][0]
            ]

        for i, j in same_node:
            ri, rj = find(i), find(j)
            if ri != rj:
                join(ri, rj)

        for i, j, link, by_id in proofs:
            ri, rj = find(i), find(j)
            if ri != rj:
                # SAME_AS is transitive, and that is exactly how two entities
                # whose ids disagree get joined through a third that has none.
                if found := clash(ri, rj):
                    links.append(
                        EntityLink(
                            source_key=link.source_key,
                            target_key=link.target_key,
                            kind=LinkKind.DIFFERENT,
                            score=self._weak(profiles[i], profiles[j]),
                            reason=f"{'; '.join(found)} (refused a {link.reason} that "
                            f"would have joined them)",
                        )
                    )
                    if j >= size:
                        to_store[LinkKind.DIFFERENT.value] += 1
                    continue
                join(ri, rj)
            root = find(i)
            by_id_only[root] = by_id_only[root] and by_id
            links.append(link)
            if j >= size:
                to_store[LinkKind.SAME_AS.value] += 1

        # Batch merges, strongest first: a merge is transitive too, so one that
        # would join a pair kept apart, two caller keys or two ids is refused.
        partners: dict[int, list[int]] = defaultdict(list)
        for i, j in apart:
            partners[i].append(j)
            partners[j].append(i)

        def refusal(ri: int, rj: int) -> list[str]:
            found = clash(ri, rj)
            if caller[ri] and caller[rj]:
                found.append("two keys a caller chose")
            if any(find(y) == rj for x in members[ri] for y in partners[x]):
                found.append("a pair in the two is kept apart")
            return found

        # A mention alike to two entities that are kept apart could be either,
        # so it merges with neither: "Halden Robotics" beside its GmbH and Ltd.
        alike_to: dict[int, set[int]] = defaultdict(set)
        for i, j, _, _ in merges:
            alike_to[i].add(find(j))
            alike_to[j].add(find(i))
        torn: dict[int, str] = {}
        for x, roots in alike_to.items():
            either = sorted(roots - {find(x)}, key=lambda r: profiles[r].key)
            for r1, r2 in combinations(either, 2):
                if refusal(r1, r2):
                    torn[x] = f"alike to both {profiles[r1].key} and {profiles[r2].key}, kept apart"
                    break
        merged: set[int] = set()
        for i, j, link, said in sorted(
            merges, key=lambda m: (-(m[2].score or 0.0), m[2].source_key, m[2].target_key)
        ):
            if i in torn or j in torn:
                kept["ambiguous"] += 1
                add(i, j, _not_merged(link, torn.get(i) or torn[j]))
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                refused = refusal(ri, rj)
                if refused:
                    kept["refused"] += 1
                    add(i, j, _not_merged(link, "; ".join(refused)))
                    continue
                join(ri, rj)
                root = find(ri)
                by_id_only[root] = False
                weakest[root] = min(weakest[root], link.score or 0.0)
                merged.add(root)
            links.append(
                link.model_copy(
                    update={"kind": LinkKind.SAME_AS, "reason": f"in-batch merge: {said}"}
                )
            )

        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(profiles)):
            clusters[find(i)].append(i)
        known = set(index.keys())
        replacement: dict[str, Entity] = {}
        for root, cluster in clusters.items():
            batch = [profiles[i].entity for i in cluster if i < size]
            stored = [profiles[i].entity for i in cluster if i >= size]
            if stored and not _states(batch, stored):
                # Proof against the store: the incoming facts take the stored
                # entity as it is, and none of the batch's names land on it.
                canonical = min(stored, key=lambda e: e.key)
                replacement.update(dict.fromkeys((m.key for m in batch), canonical))
            elif len(batch) > 1:
                prefer = {e.key for e in stored} if stored else known
                merged_entity = _merge(
                    batch, prefer, by_id_only[root], score=weakest[root], types=self._types
                )
                replacement.update(dict.fromkeys((m.key for m in batch), merged_entity))

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
        stats: dict[str, Any] | None = None
        if self.lookup is not None:
            stats = self._store_stats(profiles[size:], to_store, replacement, mentioned)
        if self.judge is not None:
            stats = stats or {}
            own = self.judge.stats
            stats["judge"] = own
            # Where `prompts_sent` and a run report look for the keys a stage sent.
            stats["prompts"] = own["prompts"]
        if self.normalize_batch:
            stats = stats or {}
            groups = {find(root) for root in merged}
            stats["batch"] = {
                "alike": len(alike),
                "merged": len(alike) - sum(kept.values()),
                "groups": len(groups),
                "mentions": sum(1 for root in groups for i in clusters[root] if i < size),
                "numbers": kept["numbers"],
                "forms": kept["forms"],
                "context": kept["context"],
                "ambiguous": kept["ambiguous"],
                "refused": kept["refused"],
                "embedded": len(vectors),
            }
        self.stats = stats
        return resolved, links

    def _embedded(
        self,
        profiles: Sequence[_Profile],
        alike: Sequence[tuple[int, int, EntityLink, bool]],
        facts: Sequence[Fact],
    ) -> dict[str, list[float]]:
        """The unit vector of each look-alike's sentence, by key, for those that have one.

        A mention's sentence is where the first fact citing it mentions it,
        read from `documents` or a span's quote. Every distinct sentence is
        embedded once, in one call.
        """
        from openodke.corroborate.judge import cited_context

        assert self.embed is not None
        wanted = {k for i, j, _, read in alike if not read for k in (i, j)}
        citing: dict[str, list[Fact]] = defaultdict(list)
        for fact in facts:
            for entity in (fact.subject, fact.object_entity):
                if entity is not None:
                    citing[entity.key].append(fact)
        documents = self.documents or {}
        sentence: dict[str, str] = {}
        for k in sorted(wanted):
            entity = profiles[k].entity
            found = (cited_context(entity, f, documents, around=0) for f in citing[entity.key])
            if text := next((t for t in found if t), None):
                sentence[entity.key] = text
        if not sentence:
            return {}
        texts = sorted(set(sentence.values()))
        vectors = dict(zip(texts, map(_unit, self.embed(texts)), strict=True))
        return {key: vectors[text] for key, text in sentence.items()}

    def _ask(
        self,
        profiles: Sequence[_Profile],
        open_pairs: Sequence[tuple[int, int, float]],
        facts: Sequence[Fact],
    ) -> list[EntityLink | None]:
        """The judge's link for each open pair, or None where it made none."""
        from openodke.corroborate.judge import link_for

        assert self.judge is not None
        involved = [profiles[k].entity for i, j, _ in open_pairs for k in (i, j)]
        mentions = self.judge.mentions(involved, facts)
        decisions = self.judge.judge_many(
            [
                (mentions[profiles[i].key], mentions[profiles[j].key], score)
                for i, j, score in open_pairs
            ]
        )
        return [link_for(decision) for decision in decisions]

    def _stored(
        self, profiles: list[_Profile], incoming: list[Entity]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """The store's candidates appended to `profiles`, and the pairs to judge.

        Each incoming entity is paired with each candidate the lookup returned for
        it, within its type. A candidate with the incoming entity's own key is the
        same node and is never judged. A store entity whose key the batch states
        is that node too, so it joins the batch's entity without a link.
        """
        assert self.lookup is not None
        found = self.lookup.candidates(incoming)
        at = {p.key: i for i, p in enumerate(profiles)}
        # A key is unique within a type, so a stored entity is known by both.
        stored: dict[tuple[str, str], int] = {}

        def add(candidate: Entity) -> int:
            known = (candidate.type, candidate.key)
            if known not in stored:
                stored[known] = len(profiles)
                profiles.append(_profile(candidate))
            return stored[known]

        pairs: set[tuple[int, int]] = set()
        for entity in incoming:
            for candidate in found.get(entity.key, ()):
                if candidate.type != entity.type:
                    continue
                if candidate.key == entity.key:
                    add(candidate)
                else:
                    pairs.add((at[entity.key], add(candidate)))
        same_node = [
            (at[key], index)
            for (kind, key), index in stored.items()
            if key in at and profiles[at[key]].entity.type == kind
        ]
        return sorted(pairs), sorted(same_node)

    def _store_stats(
        self,
        stored: Sequence[_Profile],
        to_store: Mapping[str, int],
        replacement: Mapping[str, Entity],
        mentioned: Sequence[Entity],
    ) -> dict[str, Any]:
        keys = {p.key for p in stored}
        incoming = {e.key for e in mentioned}
        # An incoming key whose facts now carry a key the store already held.
        rekeyed = sum(
            1 for key in incoming if key in replacement and replacement[key].key in keys - {key}
        )
        stats: dict[str, Any] = {
            "store": {
                "looked_up": len(incoming),
                "candidates": len(keys),
                "rekeyed": rekeyed,
                **{kind.value: to_store.get(kind.value, 0) for kind in LinkKind},
            }
        }
        own = getattr(self.lookup, "stats", None)
        if isinstance(own, Mapping):
            stats["lookup"] = dict(own)
        return stats

    def _judge(self, a: _Profile, b: _Profile) -> tuple[EntityLink, bool] | float | None:
        """The rules' link and whether an id proved it; the score when no rule
        settled the pair; None when evidence against left nothing to kill."""
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
        return score

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


def _caller(entity: Entity) -> bool:
    return entity.resolution is not None and entity.resolution.method == "caller"


def _states(batch: Sequence[Entity], stored: Sequence[Entity]) -> bool:
    """Whether the batch keeps its own entity in a cluster with the store: it
    states a stored key itself, or the caller chose one of its keys."""
    keys = {e.key for e in stored}
    return any(_caller(e) or e.key in keys for e in batch)


def _not_merged(link: EntityLink, why: str) -> EntityLink:
    """The `SIMILAR` a look-alike kept apart keeps, saying why it was not merged."""
    said = f"not merged: {why}"
    return link.model_copy(update={"reason": f"{link.reason}; {said}" if link.reason else said})


def _cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return round(sum(x * y for x, y in zip(a, b, strict=True)), 4)


def _merge(
    members: list[Entity],
    known: set[str],
    by_id: bool,
    *,
    score: float = 1.0,
    types: Mapping[str, str] | None = None,
) -> Entity:
    """One entity for a `SAME_AS` cluster, keeping every name any member had."""
    named = types or {}

    def preference(e: Entity) -> tuple[bool, bool, bool, bool, str]:
        # The caller's own key first, then one already in the store, then one
        # an identifier names, then one typed with the ontology's own name
        # rather than an alias, then the smallest key, so a re-run picks the
        # same canonical entity.
        caller = e.resolution is not None and e.resolution.method == "caller"
        aliased = named.get(_fold_type(e.type), e.type) != e.type
        return (not caller, e.key not in known, e.external_id is None, aliased, e.key)

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
        resolution = Resolution(method="linker", linker=LINKER, score=score)
    return canonical.model_copy(
        update={
            "aliases": tuple(aliases),
            "attributes": {**attributes, **canonical.attributes},
            "external_id": external_id,
            "resolution": resolution,
        }
    )


__all__ = [
    "DEFAULT_CONTEXT_FLOOR",
    "LINKER",
    "BlockKeys",
    "Embed",
    "NativeResolver",
    "block_keys",
    "candidate_pairs",
    "domain_of",
    "name_similarity",
]
