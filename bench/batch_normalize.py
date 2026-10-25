"""Normalising mentions in a batch, measured on label set R (#148).

    python bench/batch_normalize.py [bench/labels/R] [--out DIR]

No model, no download: label set R (#151, `bench/labels/R/`) carries each
mention's name, type and passage, and its gold comes from Re-DocRED's own
entity clusters. Each R document is one batch: every mention R drew from it,
stated by one fact that cites its passage. Its gold is within the document,
and the batch is too.

Scored on R's labelled pairs, dev and gate apart, with
`openodke.eval.evaluate_resolution`: of the pairs a setting put in one entity,
the share R calls one (precision), and of the pairs R calls one, the share it
put together (recall). The settings:

- **current**: `NativeResolver()` as it was. It merges on proof only, and R
  has no ids, so it merges nothing; its "same" is a `SIMILAR` link, counted
  here as merged.
- **batch, judge-free**: `NativeResolver(normalize_batch=True)`: the rules
  and the name score alone.
- **batch, names alone**: the same with the rule that keeps names with
  different numbers or legal forms apart switched off, to show what it does.
- **current + perfect judge**, **batch + perfect judge**: a pair judge whose
  every answer is R's gold, as a person's decisions in its place
  (`PairJudge(reviewed=...)`). It is never called, so it is free, and it
  bounds what a judge in the band can add. The real judge's numbers come
  from its calibration card on R.

R is drawn to be hard, and leaves out two names with one name key, which are
most of what the default merges. `--redocred` adds the batches as they come:
each of Re-DocRED's documents is one batch, one mention per name and cluster,
every pair of them labelled by the clusters. It describes the default, and
decided nothing: the rule was R's gate split.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

from openodke import Document, Entity, EntityLink, Evidence, Fact, Span
from openodke.corroborate import NativeResolver, PairJudge
from openodke.corroborate import resolve as resolve_module
from openodke.eval import evaluate_resolution
from openodke.eval.datasets.redocred import TYPES
from openodke.eval.formats import PairLabel

R = Path(__file__).parent / "labels" / "R"
SPLITS = ("dev", "gate")
SETTINGS = (
    "current",
    "batch, judge-free",
    "batch, names alone",
    "current + perfect judge",
    "batch + perfect judge",
)


class Mention(NamedTuple):
    """One mention R drew: its key, type, name, passage, and Re-DocRED cluster."""

    key: str
    type: str
    label: str
    context: str
    cluster: int


class Batch(NamedTuple):
    """One R document: its split, its mentions, and R's labelled pairs in it."""

    doc: str
    split: str
    mentions: list[Mention]
    labels: list[PairLabel]


def load(folder: Path = R) -> list[Batch]:
    """R's items, labels and private rows, grouped by document, in document order."""

    def rows(name: str) -> list[dict[str, Any]]:
        lines = (folder / name).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    items, labels, private = rows("items.jsonl"), rows("labels.jsonl"), rows("items.private.jsonl")
    mentions: dict[str, dict[str, Mention]] = defaultdict(dict)
    pairs: dict[str, list[PairLabel]] = defaultdict(list)
    split: dict[str, str] = {}
    for item, label, row in zip(items, labels, private, strict=True):
        doc = row["doc"]
        split[doc] = row["split"]
        for side, cluster in zip(("a", "b"), row["clusters"], strict=True):
            m = item[side]
            found = Mention(m["key"], m["type"], m["label"], m["context"], cluster)
            if mentions[doc].setdefault(found.key, found) != found:
                raise ValueError(f"{found.key} is in two clusters")
        pairs[doc].append(PairLabel.model_validate(label))
    return [
        Batch(doc, split[doc], [mentions[doc][k] for k in sorted(mentions[doc])], pairs[doc])
        for doc in sorted(mentions)
    ]


def facts(batch: Batch) -> tuple[list[Fact], list[Document]]:
    """One fact per mention, citing its passage, and the passages as documents."""
    out, texts = [], []
    for m in batch.mentions:
        start = max(m.context.find(m.label), 0)
        span = Span(doc_id=m.key, start=start, end=start + len(m.label))
        entity = Entity(key=m.key, type=m.type, label=m.label)
        out.append(
            Fact(
                subject=entity,
                predicate="mentioned",
                object_value=m.label,
                evidence=(Evidence(doc_id=m.key, span=span),),
            )
        )
        texts.append(Document(id=m.key, text=m.context))
    return out, texts


def gold(batch: Batch) -> list[dict[str, Any]]:
    """Every pair of the document's mentions and whether they are one entity: the perfect judge."""
    return [
        {"a": a.key, "b": b.key, "same": a.cluster == b.cluster}
        for a, b in combinations(batch.mentions, 2)
    ]


class _Never:
    """A client the perfect judge holds and never calls: every pair is decided before."""

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the perfect judge called a model")


@contextmanager
def _names_alone() -> Iterator[None]:
    disagree = resolve_module._disagree
    resolve_module._disagree = lambda a, b: None
    try:
        yield
    finally:
        resolve_module._disagree = disagree


def resolver(setting: str, batch: Batch) -> NativeResolver:
    judge = None
    if setting.endswith("perfect judge"):
        judge = PairJudge(client=_Never(), reviewed=gold(batch))  # type: ignore[arg-type]
    return NativeResolver(normalize_batch=setting.startswith("batch"), judge=judge)


def measure(batches: Sequence[Batch], setting: str) -> dict[str, Any]:
    """One setting on every batch: P/R of merged pairs on each split, and what it did."""
    links: list[EntityLink] = []
    did: dict[str, int] = defaultdict(int)
    for batch in batches:
        found, texts = facts(batch)
        run = resolver(setting, batch)
        if run.documents is not None:
            run.documents.update({doc.id: doc for doc in texts})
        if setting == "batch, names alone":
            with _names_alone():
                _, made = run.resolve(found, {})
        else:
            _, made = run.resolve(found, {})
        links += made
        own = run.stats or {}
        for key, value in (own.get("batch") or {}).items():
            did[key] += int(value)
        if run.judge is not None:
            did["judge_pairs"] += run.judge.stats["pairs"]
            did["judge_calls"] += run.judge.stats["calls"]
    as_merged = setting.startswith("current")
    out: dict[str, Any] = {"setting": setting, "stats": dict(sorted(did.items()))}
    for split in SPLITS:
        labels = [label for b in batches if b.split == split for label in b.labels]
        metrics = evaluate_resolution(labels, links, similar_as_same=as_merged).metrics
        out[split] = {
            "pairs": len(labels),
            "same": sum(label.same for label in labels),
            "merged": int(metrics["tp"]) + int(metrics["fp"]),
            "right": int(metrics["tp"]),
            "precision": metrics["pairwise_precision"],
            "recall": metrics["pairwise_recall"],
        }
    return out


def run(folder: Path = R) -> dict[str, Any]:
    batches = load(folder)
    return {
        "documents": {s: sum(b.split == s for b in batches) for s in SPLITS},
        "mentions": sum(len(b.mentions) for b in batches),
        "settings": [measure(batches, setting) for setting in SETTINGS],
    }


def natural(raw: Sequence[Mapping[str, Any]], *, split_name: str = "test") -> dict[str, Any]:
    """Each Re-DocRED document as one batch: the current resolver against the default.

    One mention per name and cluster, TIME and NUM left out as values, every
    pair of a document's mentions labelled: one entity when one cluster.
    """
    labels: list[PairLabel] = []
    found: dict[str, list[EntityLink]] = {"current": [], "batch, judge-free": []}
    mentions = 0
    for i, doc in enumerate(raw):
        doc_id = f"{split_name}_{i:04d}"
        entities, cluster = [], {}
        for c, vertex in enumerate(doc["vertexSet"]):
            named = dict.fromkeys((m["name"], m["type"]) for m in vertex if m["type"] in TYPES)
            for j, (name, kind) in enumerate(named):
                if kind in ("TIME", "NUM"):
                    continue
                entity = Entity(key=f"{doc_id}#m{c}.{j}", type=TYPES[kind], label=name)
                entities.append(entity)
                cluster[entity.key] = c
        mentions += len(entities)
        labels += [
            PairLabel(a=a.key, b=b.key, same=cluster[a.key] == cluster[b.key])
            for a, b in combinations(entities, 2)
        ]
        facts = [Fact(subject=e, predicate="mentioned", object_value=e.label) for e in entities]
        for setting in found:
            run = NativeResolver(normalize_batch=setting.startswith("batch"))
            found[setting] += run.resolve(facts, {})[1]
    rows = []
    for setting, links in found.items():
        metrics = evaluate_resolution(labels, links, similar_as_same=setting == "current").metrics
        rows.append(
            {
                "setting": setting,
                "merged": int(metrics["tp"]) + int(metrics["fp"]),
                "right": int(metrics["tp"]),
                "precision": metrics["pairwise_precision"],
                "recall": metrics["pairwise_recall"],
            }
        )
    same = sum(label.same for label in labels)
    return {"documents": len(raw), "mentions": mentions, "pairs": len(labels), "same": same,
            "settings": rows}  # fmt: skip


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def table(result: Mapping[str, Any]) -> str:
    """The numbers as the Markdown a PR or the docs quote."""
    first = result["settings"][0]
    lines = [
        f"Label set R: {first['dev']['pairs']} dev pairs ({first['dev']['same']} same) in "
        f"{result['documents']['dev']} documents, {first['gate']['pairs']} gate pairs "
        f"({first['gate']['same']} same) in {result['documents']['gate']}; one batch a "
        f"document, {result['mentions']} mentions.",
        "",
        "| setting | dev merged | dev P | dev R | gate merged | gate P | gate R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["settings"]:
        dev, gate = row["dev"], row["gate"]
        lines.append(
            f"| {row['setting']} | {dev['merged']} ({dev['right']} right) "
            f"| {_pct(dev['precision'])} | {_pct(dev['recall'])} "
            f"| {gate['merged']} ({gate['right']} right) "
            f"| {_pct(gate['precision'])} | {_pct(gate['recall'])} |"
        )
    if natural := result.get("natural"):
        lines += [
            "",
            f"Re-DocRED test as it comes: {natural['documents']} documents, one batch each, "
            f"{natural['mentions']} mentions, {natural['pairs']} pairs ({natural['same']} same).",
            "",
            "| setting | merged | P | R |",
            "|---|---:|---:|---:|",
        ]
        for row in natural["settings"]:
            lines.append(
                f"| {row['setting']} | {row['merged']} ({row['right']} right) "
                f"| {_pct(row['precision'])} | {_pct(row['recall'])} |"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, nargs="?", default=R, help="label set R")
    parser.add_argument("--out", type=Path, help="write batch_normalize.json and .md here")
    parser.add_argument("--redocred", type=Path, help="Re-DocRED's test_revised.json, as it comes")
    args = parser.parse_args(argv)
    result = run(args.folder)
    if args.redocred is not None:
        result["natural"] = natural(json.loads(args.redocred.read_text(encoding="utf-8")))
    text = table(result)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "batch_normalize.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "batch_normalize.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "Batch",
    "Mention",
    "facts",
    "gold",
    "load",
    "measure",
    "natural",
    "resolver",
    "run",
    "table",
]
