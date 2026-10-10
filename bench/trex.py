"""T-REx, measured from a run `odke bench run trex` saved (#117).

    python bench/trex.py lookup runs/trex/set80                       # no model
    python bench/trex.py lookup runs/trex/set80 --judge --budget-usd 0.10
    python bench/trex.py tables runs/trex/set80 runs/trex/set80/competitors/lgt

`lookup` measures the store lookup across documents (#31, #114), which a run
never exercises: it resolves one batch and has no store. Every other abstract
the run read is the store, its entities in a `MemoryLookup`, and the rest are
resolved against it by `NativeResolver(lookup=...)`, as a second batch meets a
graph already written. Re-DocRED could only score this within one document
(`store_lookup.py`); here Wikidata ids are gold identity across documents.

- A batch entity whose key the store holds is the same node, by key.
- A link to the store is right when both names stand for one id (`links` in
  `openodke.eval.datasets.trex`).
- A mention is owed a link when its name stands for one id and the store holds
  that id under another name; recall is the share of those linked.

`--judge` adds the pair judge (#34) on the set's ground model, under
`--budget-usd`, answered from the run's response cache where it can be. A
stored entity's context is the sentence its store abstract names it in, and one
either side. It writes `lookup.json` beside the run.

`tables` prints the Markdown tables `docs/benchmarks.md` publishes, from each
run's `report.json`, `corroboration.json` and `lookup.json`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openodke import Entity, Fact
from openodke.chunking import sentences
from openodke.corroborate import MemoryLookup, NativeResolver
from openodke.eval.datasets import trex
from openodke.eval.datasets._common import read_jsonl
from openodke.types import Document, GroundingVerdict

REFUSED = {GroundingVerdict.NOT_FOUND, GroundingVerdict.CONTRADICTED}


def load(run: Path) -> tuple[trex.Gold, list[Fact]]:
    """The set's gold, and the grounded facts the gate kept.

    The prepared sets refuse `not_found` as well as `contradicted` (`--paper`).
    """
    meta = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
    gold = trex.Gold(read_jsonl(run / "gold.jsonl"), meta["relation_labels"].values())
    grounded = [Fact.model_validate(r) for r in read_jsonl(run / "facts" / "grounded.jsonl")]
    return gold, [f for f in grounded if f.verdict not in REFUSED]


def _name(entity: Entity) -> str:
    return entity.label or entity.key.partition(":")[2] or entity.key


def _documents(run: Path) -> dict[str, Document]:
    from openodke.run.build import build
    from openodke.run.config import load_config

    return {doc.id: doc for doc in build(load_config(run / "odke.json")).documents()}


def store_lookup(
    run: Path,
    gold: trex.Gold,
    gated: list[Fact],
    *,
    judge: bool = False,
    budget: float | None = None,
) -> dict[str, Any]:
    """Every other abstract as the store, the rest resolved against it; links scored on the ids."""
    docs = _documents(run)
    order = sorted({e.doc_id for f in gated for e in f.evidence})
    stored_docs = set(order[::2])
    in_store = [f for f in gated if f.evidence and f.evidence[0].doc_id in stored_docs]
    batch = [f for f in gated if f.evidence and f.evidence[0].doc_id not in stored_docs]
    store: dict[str, Entity] = {}
    where: dict[str, str] = {}
    for fact in in_store:
        for entity in (fact.subject, fact.object_entity):
            if entity is not None and entity.key not in store:
                store[entity.key] = entity
                where[entity.key] = fact.evidence[0].doc_id
    mentions: dict[str, Entity] = {}
    for fact in batch:
        for entity in (fact.subject, fact.object_entity):
            if entity is not None:
                mentions.setdefault(entity.key, entity)

    pair_judge = None
    spent: dict[str, Any] = {}
    if judge:
        pair_judge = _judge(run, budget)
        pair_judge.documents.update(docs)
        pair_judge.store_context = lambda entity: _context(entity, where, docs)
    resolver = NativeResolver(lookup=MemoryLookup(store), judge=pair_judge)
    _, found = resolver.resolve(batch, mentions)
    links = [
        k for k in found if (k.source_key in store) != (k.target_key in store)
    ]  # one end in the store, the other a batch mention the store does not hold by key
    if pair_judge is not None:
        stats = pair_judge.stats
        spent = {
            "calls": stats.get("calls", 0),
            "cached": stats.get("cached", 0),
            "input_tokens": stats.get("prompt_tokens", 0),
            "output_tokens": stats.get("completion_tokens", 0),
            "usd": stats.get("cost_usd", 0.0),
            "unasked": stats.get("unasked", 0),
            "disagreed": stats.get("disagreed", 0),
        }

    # Mentions owed a link: one id, held by the store under another key.
    held: dict[str, set[str]] = {}
    for key, entity in store.items():
        for qid in gold.of(_name(entity)):
            held.setdefault(qid, set()).add(key)
    by_key = [k for k in mentions if k in store]
    owed = {
        key: held[qid]
        for key, entity in mentions.items()
        if key not in store and len(ids := gold.of(_name(entity))) == 1
        for qid in ids
        if qid in held
    }
    linked = {
        (k.source_key if k.source_key in mentions else k.target_key)
        for k in links
        if k.kind.value in ("same_as", "similar")
        and (
            k.target_key in owed.get(k.source_key, ()) or k.source_key in owed.get(k.target_key, ())
        )
    }
    facts = [*in_store, *batch]
    return {
        "store": {"documents": len(stored_docs), "entities": len(store)},
        "batch": {"documents": len(order) - len(stored_docs), "entities": len(mentions)},
        "same_node_by_key": len(by_key),
        "owed_a_link": len(owed),
        "linked_of_owed": len(linked),
        "recall": len(linked) / len(owed) if owed else None,
        "links": trex.links(gold, links, facts),
        "judge": spent,
    }


def _judge(run: Path, budget: float | None) -> Any:
    """The set's own pair judge, on its ground model, cache and meter, under `budget` USD."""
    from openodke.run.build import build
    from openodke.run.config import load_config

    config = load_config(run / "odke.json")
    if budget is not None:
        config = config.with_budget(usd=budget)
    resolver = build(config).stages["resolver"]
    judge = getattr(resolver, "judge", None)
    if judge is None:
        raise SystemExit(f"{run}/odke.json has no pair judge: prepare the set with --judge")
    return judge


def _context(entity: Entity, where: Mapping[str, str], docs: Mapping[str, Document]) -> str | None:
    """The sentence of its store abstract that names a stored entity, and one either side."""
    doc = docs.get(where.get(entity.key, ""))
    if doc is None:
        return None
    spans = sentences(doc.text)
    name = _name(entity).casefold()
    for i, (start, end) in enumerate(spans):
        if name in doc.text[start:end].casefold():
            lo, hi = spans[max(0, i - 1)][0], spans[min(len(spans) - 1, i + 1)][1]
            return doc.text[lo:hi]
    return None


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """The 95% Wilson interval of `hits` in `n`; None with nothing to count."""
    if not n:
        return None
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _share(hits: int, n: int) -> str:
    """`85.3 [69.9, 93.6]`: a share of edges with its Wilson interval."""
    interval = wilson(hits, n)
    if interval is None:
        return "—"
    return f"{100 * hits / n:.1f} [{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]"


def tables(runs: Sequence[Path]) -> str:
    """The published tables: documents, the graph off and on, support, sources, the resolver."""
    docs = [
        "| System | Row | P (gold) | P (factual) | R | R, 1 source | R, 2 | R, 3+ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    graph = [
        "| System | Graph | Edges | P (factual) | In gold | Found, 1 source | 2 | 3+ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    support = ["| System | Support | Edges | P (factual) |", "|---|---|---|---|"]
    counted = [
        "| System | Gold sources | On the graph | 2+ sources by name | 2+ sources by id |",
        "|---|---|---|---|---|",
    ]
    resolver = [
        "| System | Links | Where | Count | Right | Wrong | Not checkable |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        system = "LLMGraphTransformer" if run.name == "lgt" else "openodke"
        report = json.loads((run / "report.json").read_text())
        for name, row in report["stages"][0]["breakdown"].items():
            docs.append(
                f"| {system} | {name} | {_pct(row['precision'])} | "
                f"{_pct(row.get('precision_factual'))} | {_pct(row['recall'])} | "
                + " | ".join(_pct(row.get(f"recall_{b}")) for b in trex.BUCKETS)
                + " |"
            )
        saved = json.loads((run / "corroboration.json").read_text())
        found = saved["graph"]
        shown = [
            ("off: + grounding", found["off"]),
            ("on: + corroboration", found["on"]),
            ("on, 2+ sources", found["on_two_or_more"]),
        ]
        for name, g in shown:
            graph.append(
                f"| {system} | {name} | {g['edges']} | {_share(g['factual'], g['edges'])} | "
                f"{g['in_gold']} | "
                + " | ".join(
                    f"{g['recall'][b]['found']}/{g['recall'][b]['gold']}" for b in trex.BUCKETS
                )
                + " |"
            )
        for b in trex.BUCKETS:
            g = found["by_support"][b]
            support.append(
                f"| {system} | {b} | {g['edges']} | {_share(g['factual'], g['edges'])} |"
            )
            c = found["counted"][b]
            by = f"{c['two_or_more']} | {c['two_or_more_pooled']}"
            counted.append(f"| {system} | {b} | {c['found']} | {by} |")
        lookup = _read(run / "lookup.json")
        places = [("in the batch", saved["links"])]
        if lookup:
            places.append(("against the store", lookup["links"]))
        for where, kinds in places:
            for kind, k in sorted(kinds.items()):
                resolver.append(
                    f"| {system} | {kind} | {where} | {k['links']} | {k['right']} | "
                    f"{k['wrong']} | {k['uncheckable']} |"
                )
    return "\n\n".join("\n".join(t) for t in (docs, graph, support, counted, resolver)) + "\n"


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("lookup", help="the store lookup across documents, from a saved run")
    one.add_argument("run", type=Path)
    one.add_argument("--judge", action="store_true", help="the pair judge too (model calls)")
    one.add_argument("--budget-usd", type=float, help="the most the judge may spend")
    many = sub.add_parser("tables", help="the Markdown tables, from saved runs")
    many.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "tables":
        print(tables(args.runs), end="")
        return
    gold, gated = load(args.run)
    found = store_lookup(args.run, gold, gated, judge=args.judge, budget=args.budget_usd)
    (args.run / "lookup.json").write_text(json.dumps(found, indent=2) + "\n")
    print(
        f"store lookup: {found['same_node_by_key']} the same node by key, "
        f"{found['linked_of_owed']} of {found['owed_a_link']} owed a link linked; "
        f"links {json.dumps(found['links'])}; judge {json.dumps(found['judge'])}"
    )


if __name__ == "__main__":
    main()
