"""Resolving against a store, measured offline on Re-DocRED's entity clusters (#114, #149).

    python bench/store_lookup.py /path/to/redocred/test_revised.json [--out DIR]

No model and no database: the store is a `MemoryLookup`, and the resolver is
`NativeResolver(lookup=...)`, exactly as a run uses them. `odke bench fetch
redocred` downloads the split.

Re-DocRED's gold identity is within a document. Each document's mentions are
grouped into entity clusters, and nothing ties a cluster in one document to
one in another: there is no id. A store built from some documents and probed
with others could only be scored by guessing. So the split is inside each
document, by sentence:

- **The store** holds what the first half of each document's sentences said:
  one entity per cluster mentioned there, typed by its mentions' majority
  type, labelled by its first mention, its other names as aliases, and its
  document as its tenant.
- **The batch** is the second half: one entity per distinct name and type a
  cluster is mentioned under there.
- Each document's batch is resolved against the one shared store, scoped to
  its tenant, so every candidate is in the same document and every pair can
  be judged. Every (mention, stored entity) pair of a document is labelled, so
  the labels are exhaustive within it.

TIME and NUM clusters are values, not entities, and are left out, as
`odke bench` leaves them out of the ontology's types.

Reported, for each `SIMILAR` threshold (0.9 is the default, chosen before this
ran, and not tuned on it):

- **link precision**: of the links the resolver proposed to the store, the
  share between a mention and its own entity;
- **link recall**: of the mentions whose entity is in the store, the share
  linked to it;
- pairwise P/R and B-cubed from `openodke.eval.evaluate_resolution`, with
  `SIMILAR` counted as a match. B-cubed is over every labelled key, and most
  are singletons nothing links (a stored entity the second half never names,
  a mention of one the first half never named), so it sits near 1 and moves
  little. The link columns are the ones to read.

Re-DocRED has no external ids or domains, so no pair is ever proven and every
link is a `SIMILAR`: this measures the weak path alone. A proof is exact by
construction; the multi-source benchmark (#117, T-REx with Wikidata ids, in
0.6.0) is where it, and identity across documents, get measured. The unscoped
run below shows why that is needed: with no tenant, a mention also links to
entities of other documents, which this gold cannot call right or wrong.

`--out` writes `store_lookup.json` and `store_lookup.md`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from openodke import Entity, EntityLink, Fact
from openodke.corroborate import MemoryLookup, NativeResolver
from openodke.eval import evaluate_resolution
from openodke.eval.datasets.redocred import TYPES
from openodke.eval.formats import PairLabel

THRESHOLDS = (0.8, 0.85, 0.9, 0.95, 1.0)
DEFAULT = 0.9
# DocRED's value types: not entities, so neither stored nor looked up.
VALUES = frozenset({"TIME", "NUM"})


class Document(NamedTuple):
    """One document's store entities and batch mentions, with each one's gold cluster."""

    id: str
    stored: list[Entity]
    mentions: list[Entity]
    cluster: dict[str, int]


def split(raw: Sequence[Mapping[str, Any]], *, split_name: str = "test") -> list[Document]:
    """Each document's first half of sentences as its store, the second half as its batch."""
    documents = []
    for i, doc in enumerate(raw):
        doc_id = f"{split_name}_{i:04d}"
        half = (len(doc["sents"]) + 1) // 2
        out = Document(doc_id, [], [], {})
        for c, cluster in enumerate(doc["vertexSet"]):
            mentions = sorted(
                (m for m in cluster if m["type"] not in VALUES),
                key=lambda m: (m["sent_id"], m["pos"][0]),
            )
            first = [m for m in mentions if m["sent_id"] < half]
            second = [m for m in mentions if m["sent_id"] >= half]
            if first:
                kind = Counter(m["type"] for m in first).most_common(1)[0][0]
                names = list(dict.fromkeys(m["name"] for m in first))
                key = f"{doc_id}#e{c}"
                out.stored.append(
                    Entity(
                        key=key,
                        type=TYPES[kind],
                        label=names[0],
                        aliases=tuple(names[1:]),
                        attributes={"tenant": doc_id},
                    )
                )
                out.cluster[key] = c
            seen = dict.fromkeys((m["name"], m["type"]) for m in second)
            for j, (name, kind) in enumerate(seen):
                key = f"{doc_id}#m{c}.{j}"
                out.mentions.append(Entity(key=key, type=TYPES[kind], label=name))
                out.cluster[key] = c
        documents.append(out)
    return documents


def labels(doc: Document) -> list[PairLabel]:
    """Every (mention, stored entity) pair of the document, and whether they are one entity."""
    return [
        PairLabel(a=m.key, b=s.key, same=doc.cluster[m.key] == doc.cluster[s.key])
        for m in doc.mentions
        for s in doc.stored
    ]


def resolve(doc: Document, lookup: MemoryLookup, threshold: float) -> tuple[list[EntityLink], int]:
    """The document's batch resolved against the store: links into it, and pairs judged."""
    resolver = NativeResolver(threshold=threshold, lookup=lookup)
    facts = [Fact(subject=m, predicate="mention", object_value=m.label) for m in doc.mentions]
    _, links = resolver.resolve(facts, {})
    found = lookup.candidates(doc.mentions)
    judged = sum(len(found[m.key]) for m in doc.mentions)
    stored = {key for keys in found.values() for key in (e.key for e in keys)}
    return [link for link in links if link.target_key in stored], judged


def measure(
    documents: Sequence[Document], threshold: float, lookups: Mapping[str, MemoryLookup]
) -> dict[str, Any]:
    """Link precision and recall, pairwise P/R and B-cubed, at one threshold."""
    gold: list[PairLabel] = []
    links: list[EntityLink] = []
    judged = 0
    for doc in documents:
        gold += labels(doc)
        found, compared = resolve(doc, lookups[doc.id], threshold)
        links += found
        judged += compared
    cluster = {key: (doc.id, c) for doc in documents for key, c in doc.cluster.items()}
    right = sum(cluster[link.source_key] == cluster[link.target_key] for link in links)
    findable = {
        (m.key, s.key)
        for doc in documents
        for m in doc.mentions
        for s in doc.stored
        if doc.cluster[m.key] == doc.cluster[s.key]
    }
    found_pairs = {(link.source_key, link.target_key) for link in links}
    metrics = evaluate_resolution(gold, links, similar_as_same=True).metrics
    return {
        "threshold": threshold,
        "links": len(links),
        "kinds": dict(sorted(Counter(link.kind.value for link in links).items())),
        "link_precision": right / len(links) if links else None,
        "link_recall": len(findable & found_pairs) / len(findable) if findable else None,
        "findable": len(findable),
        "pairwise_precision": metrics["pairwise_precision"],
        "pairwise_recall": metrics["pairwise_recall"],
        "bcubed_precision": metrics["bcubed_precision"],
        "bcubed_recall": metrics["bcubed_recall"],
        "bcubed_f1": metrics["bcubed_f1"],
        "pairs_judged": judged,
    }


def unscoped(documents: Sequence[Document], threshold: float = DEFAULT) -> dict[str, Any]:
    """The same batches against the whole store with no tenant: how many links cross documents."""
    store = {e.key: e for doc in documents for e in doc.stored}
    lookup = MemoryLookup(store)
    links: list[EntityLink] = []
    judged = 0
    for doc in documents:
        found, compared = resolve(doc, lookup, threshold)
        links += found
        judged += compared
    across = sum(link.target_key.split("#")[0] != link.source_key.split("#")[0] for link in links)
    return {
        "threshold": threshold,
        "links": len(links),
        "cross_document": across,
        "pairs_judged": judged,
    }


def summary(documents: Sequence[Document]) -> dict[str, int]:
    stored = sum(len(doc.stored) for doc in documents)
    mentions = sum(len(doc.mentions) for doc in documents)
    return {
        "documents": len(documents),
        "stored_entities": stored,
        "mentions": mentions,
        "all_pairs_within_documents": sum(len(d.stored) * len(d.mentions) for d in documents),
        "all_pairs_across_the_store": stored * mentions,
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def table(result: Mapping[str, Any]) -> str:
    """The numbers as the Markdown a PR or the docs quote."""
    size = result["summary"]
    lines = [
        f"Re-DocRED {result['split']}: {size['documents']} documents, "
        f"{size['stored_entities']} stored entities, {size['mentions']} mentions "
        f"({result['findable']} with their entity in the store).",
        "",
        "| threshold | links | link precision | link recall | pairwise P | pairwise R "
        "| B³ P | B³ R | B³ F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["thresholds"]:
        mark = " (default)" if row["threshold"] == DEFAULT else ""
        lines.append(
            f"| {row['threshold']}{mark} | {row['links']} | {_pct(row['link_precision'])} "
            f"| {_pct(row['link_recall'])} | {_pct(row['pairwise_precision'])} "
            f"| {_pct(row['pairwise_recall'])} | {_pct(row['bcubed_precision'])} "
            f"| {_pct(row['bcubed_recall'])} | {_pct(row['bcubed_f1'])} |"
        )
    loose = result["unscoped"]
    default = next(r for r in result["thresholds"] if r["threshold"] == DEFAULT)
    lines += [
        "",
        f"Pairs judged at the default: {default['pairs_judged']}, against "
        f"{size['all_pairs_within_documents']} for every mention against every stored "
        "entity of its document.",
        f"With no tenant, the same batches judged {loose['pairs_judged']} pairs against a store "
        f"of {size['stored_entities']} (all-pairs: {size['all_pairs_across_the_store']}) and "
        f"proposed {loose['links']} links, {loose['cross_document']} of them to another "
        "document's entity, which this gold cannot judge.",
    ]
    return "\n".join(lines)


def run(path: Path, *, split_name: str = "test") -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    documents = split(raw, split_name=split_name)
    started = time.perf_counter()
    # One store for every document, each document's batch scoped to its own tenant.
    store = {e.key: e for doc in documents for e in doc.stored}
    lookups = {doc.id: MemoryLookup(store, tenant=doc.id) for doc in documents}
    rows = [measure(documents, t, lookups) for t in THRESHOLDS]
    loose = unscoped(documents)
    return {
        "dataset": "redocred",
        "split": split_name,
        "summary": summary(documents),
        "findable": rows[0]["findable"],
        "thresholds": rows,
        "unscoped": loose,
        "seconds": round(time.perf_counter() - started, 2),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="Re-DocRED's test_revised.json (or dev_revised)")
    parser.add_argument("--split", default="test", help="names the documents test_0000, …")
    parser.add_argument("--out", type=Path, help="write store_lookup.json and .md here")
    args = parser.parse_args(argv)
    result = run(args.data, split_name=args.split)
    text = table(result)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "store_lookup.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "store_lookup.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = ["Document", "labels", "measure", "resolve", "run", "split", "table", "unscoped"]
