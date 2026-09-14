"""Scoring a Resolver: pairwise P/R over labelled pairs, B-cubed over clusters.

Two numbers because they fail differently. Pairwise precision and recall ask,
for each pair you labelled, whether the links put the two keys together; they
need no exhaustive labels, and they punish a wrong merge of two big clusters
exactly once. B-cubed asks, for each key, how much of its predicted cluster
is really its cluster and how much of its real cluster was found; it sees
that same wrong merge from every member's side, which is what a user of the
graph sees too.

Links are taken as plain `(a, b, kind)` triples from any source — an
`EntityLink` a Resolver proposed here, a line of a sink's `links.jsonl`, or a
platform's merges read back out of the store (DECISIONS #21). A merge that
replaced nodes leaves groups of original keys behind; `links_from_clusters`
turns each group into `same_as` triples, and from there it is scored exactly
like a link. `same_as` chains are followed, so a platform linking every
member to one survivor and a resolver linking them in a line score the same.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

from openodke.eval.formats import LinkRow, PairLabel
from openodke.eval.report import Metric, StageReport, accuracy, prf, ratio
from openodke.types import EntityLink, LinkKind

# Anything a link can arrive as.
LinkLike: TypeAlias = EntityLink | LinkRow | tuple[str, str, str]


def as_triples(links: Iterable[LinkLike]) -> list[tuple[str, str, LinkKind]]:
    """Every link as `(a, b, kind)`, whatever resolver or store it came from."""
    triples = []
    for link in links:
        if isinstance(link, EntityLink):
            triples.append((link.source_key, link.target_key, link.kind))
        elif isinstance(link, LinkRow):
            triples.append((link.a, link.b, link.kind))
        else:
            a, b, kind = link
            triples.append((a, b, LinkKind(kind)))
    return triples


def links_from_clusters(groups: Iterable[Iterable[str]]) -> list[LinkRow]:
    """A platform's merge groups as `same_as` links, each member to the first.

    For a store that replaced nodes instead of linking them: read back which
    original keys ended up as one node, pass the groups here, and score the
    result with `evaluate_resolution` like any other resolver's links.
    """
    rows: list[LinkRow] = []
    for group in groups:
        first, *rest = list(group)
        rows.extend(LinkRow(a=first, b=member, kind=LinkKind.SAME_AS) for member in rest)
    return rows


def clusters(keys: Iterable[str], same: Iterable[tuple[str, str]]) -> dict[str, frozenset[str]]:
    """Each key's cluster under the transitive closure of `same`."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for key in keys:
        find(key)
    for a, b in same:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a
    groups: dict[str, set[str]] = {}
    for key in parent:
        groups.setdefault(find(key), set()).add(key)
    return {key: frozenset(groups[find(key)]) for key in parent}


def evaluate_resolution(
    labels: Sequence[PairLabel],
    links: Iterable[LinkLike],
    *,
    similar_as_same: bool = False,
) -> StageReport:
    """Pairwise P/R/F1 over the labelled pairs, and B-cubed over the labelled keys.

    `similar` links are no decision unless `similar_as_same` says they are a
    merge. `different` links never merge anything; one whose keys a `same_as`
    chain joins anyway is a resolver disagreeing with itself, and is noted.
    """
    triples = as_triples(links)
    universe = list(dict.fromkeys(k for row in labels for k in (row.a, row.b)))
    in_universe = set(universe)
    merging = {LinkKind.SAME_AS, LinkKind.SIMILAR} if similar_as_same else {LinkKind.SAME_AS}

    gold = clusters(universe, ((r.a, r.b) for r in labels if r.same))
    link_keys = [k for a, b, _ in triples for k in (a, b)]
    # Unlabelled keys stay in the closure so a chain through them still joins two labelled keys.
    predicted = clusters([*universe, *link_keys], ((a, b) for a, b, k in triples if k in merging))

    confusion = {"same": {"same": 0, "different": 0}, "different": {"same": 0, "different": 0}}
    for row in labels:
        together = row.b in predicted[row.a]
        confusion["same" if row.same else "different"]["same" if together else "different"] += 1
    tp, fn = confusion["same"]["same"], confusion["same"]["different"]
    fp, tn = confusion["different"]["same"], confusion["different"]["different"]
    pairwise = prf(tp, fp, fn)

    precisions: dict[str, float] = {}
    recalls: dict[str, float] = {}
    for key in universe:
        mine = predicted[key] & in_universe
        overlap = len(gold[key] & mine)
        precisions[key] = overlap / len(mine)
        recalls[key] = overlap / len(gold[key])
    b3_precision = ratio(sum(precisions.values()), len(universe))
    b3_recall = ratio(sum(recalls.values()), len(universe))
    b3_f1 = (
        2 * b3_precision * b3_recall / (b3_precision + b3_recall)
        if b3_precision and b3_recall
        else None
    )

    breakdown: dict[str, dict[str, Metric]] = {}
    for group in dict.fromkeys(gold[k] for k in universe):
        members = sorted(group)
        breakdown[", ".join(members)] = {
            "size": len(members),
            "bcubed_precision": sum(precisions[m] for m in members) / len(members),
            "bcubed_recall": sum(recalls[m] for m in members) / len(members),
            # More than one: the resolver split this entity.
            "predicted_clusters": len({predicted[m] & in_universe for m in members}),
        }

    kinds = [k for _, _, k in triples]
    metrics: dict[str, Metric] = {
        "pairwise_precision": pairwise["precision"],
        "pairwise_recall": pairwise["recall"],
        "pairwise_f1": pairwise["f1"],
        "pairwise_accuracy": accuracy(confusion),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "bcubed_precision": b3_precision,
        "bcubed_recall": b3_recall,
        "bcubed_f1": b3_f1,
        "gold_clusters": len(set(gold[k] for k in universe)),
        "predicted_clusters": len({predicted[k] & in_universe for k in universe}),
        "links_same_as": kinds.count(LinkKind.SAME_AS),
        "links_similar": kinds.count(LinkKind.SIMILAR),
        "links_different": kinds.count(LinkKind.DIFFERENT),
    }
    return StageReport(
        stage="resolve",
        n=len(labels),
        metrics=metrics,
        breakdown=breakdown,
        confusion=confusion,
        notes=tuple(_notes(labels, triples, gold, predicted, in_universe, similar_as_same)),
    )


def _notes(
    labels: Sequence[PairLabel],
    triples: Sequence[tuple[str, str, LinkKind]],
    gold: dict[str, frozenset[str]],
    predicted: dict[str, frozenset[str]],
    in_universe: set[str],
    similar_as_same: bool,
) -> list[str]:
    notes = []
    inconsistent = sum(1 for r in labels if not r.same and r.b in gold[r.a])
    if inconsistent:
        notes.append(
            f"{inconsistent} pair(s) labelled different are joined by a chain of pairs "
            "labelled same; B-cubed follows the chain"
        )
    torn = sum(1 for a, b, k in triples if k is LinkKind.DIFFERENT and b in predicted[a])
    if torn:
        notes.append(f"{torn} different link(s) join keys a same_as chain merges anyway")
    similar = sum(1 for _, _, k in triples if k is LinkKind.SIMILAR)
    if similar and not similar_as_same:
        notes.append(f"{similar} similar link(s) counted as no decision")
    outside = {k for a, b, _ in triples for k in (a, b)} - in_universe
    if outside:
        notes.append(f"{len(outside)} linked key(s) appear in no labelled pair and were not scored")
    return notes


__all__ = [
    "LinkLike",
    "as_triples",
    "clusters",
    "evaluate_resolution",
    "links_from_clusters",
]
