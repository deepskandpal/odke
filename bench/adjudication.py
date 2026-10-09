"""How often gold adjudication is right (#145), measured on label set G's not-in-gold items.

    python bench/adjudication.py --verdicts runs/adjudication/G.verdicts.jsonl
    python bench/adjudication.py --model anthropic/claude-haiku-4-5-20251001 --out runs/adjudication

`odke eval pipeline --adjudicate` lists a prediction the gold lacks as
possibly missing from gold when the grounder supports it in 2 of 3 runs.
Label set G (`bench/labels/G`) holds 200 real predictions that Text2KGBench's
or Re-DocRED's gold does not count, 100 per dataset, and 100 planted false
facts, each with a person's label: does the passage support it? That is the
question the three runs vote on. So, for the 200:

- **precision of the list**: of the items it lists, the share the person
  labelled supported. How often "possibly missing from gold" is right.
- **recall of the list**: of the items the person labelled supported, the
  share it lists. How much of the gold's gap it finds.

each with a 95% Wilson interval, over both datasets and per dataset. And for
the planted facts, false by construction, how many it lists, which should be
none.

The verdicts come from `--verdicts`, a file an earlier run wrote, or from a
live run with `--model`: each item grounded three times with `LLMGrounder`,
the call carrying its run index, so 900 calls for the 300 items. They are
saved under `--out` as `G.verdicts.jsonl`, so a second run reads them instead
of paying again. Keys come from the environment, under the names LiteLLM
reads.

Until `bench/labels/G/labels.jsonl` exists it says why and exits 0. The sheets
are ticked by hand and read back with `odke label read`
(bench/labels/README.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openodke import Document, Fact
from openodke.eval.adjudication import NEEDED, RUNS, ask
from openodke.stages import Grounder

G = Path(__file__).parent / "labels" / "G"
VERDICTS = "G.verdicts.jsonl"
DATASETS = ("text2kgbench", "redocred")
Z = 1.959963984540054


def wilson(hits: int, n: int) -> tuple[float, float] | None:
    """The 95% Wilson interval of `hits` in `n`; None with nothing to count."""
    if not n:
        return None
    p = hits / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def not_in_gold(folder: Path = G) -> list[dict[str, Any]]:
    """G's items the gold does not count, real and planted: each sheet row with its private row."""
    private = {row["id"]: row for row in _jsonl(folder / "items.private.jsonl")}
    items = _jsonl(folder / "items.jsonl")
    out = []
    for number, item in enumerate(items, start=1):
        meta = private[f"G-{number:04d}"]
        if meta["fact_id"] != item["fact"]["id"]:
            raise ValueError(f"G-{number:04d}: items.jsonl and items.private.jsonl disagree")
        if meta["gold_status"] == "not_in_gold":
            out.append({**meta, "item": item})
    return out


def collect(
    rows: Sequence[Mapping[str, Any]], grounder: Grounder, out: Path, *, runs: int = RUNS
) -> dict[str, list[str]]:
    """Every row's verdicts in each run, asked of `grounder` unless `out` already holds them."""
    path = out / VERDICTS
    saved = {row["fact_id"]: row["verdicts"] for row in _jsonl(path)} if path.is_file() else {}
    todo = [row for row in rows if row["fact_id"] not in saved]
    questions = [
        (
            Fact.model_validate(row["item"]["fact"]),
            Document(id=row["item"]["doc_id"], text=row["item"]["text"]),
        )
        for row in todo
    ]
    for row, verdicts in zip(todo, ask(questions, grounder, runs=runs), strict=True):
        saved[row["fact_id"]] = [v.value for v in verdicts]
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"fact_id": k, "verdicts": v}) + "\n" for k, v in saved.items()),
        encoding="utf-8",
    )
    return saved


def measure(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    verdicts: Mapping[str, Sequence[str]],
    *,
    needed: int = NEEDED,
) -> dict[str, Any]:
    """The list's precision and recall against the labels, per dataset, and the planted it lists.

    `labels` maps a fact id to the person's verdict, `verdicts` to the
    grounder's in each run. An item with no label, or no verdicts, is counted
    and left out.
    """
    counts = {
        name: {"labelled": 0, "supported": 0, "listed": 0, "listed_supported": 0}
        for name in ("all", *DATASETS)
    }
    planted = {"items": 0, "listed": 0}
    unlabelled = unjudged = 0
    for row in rows:
        said = verdicts.get(row["fact_id"])
        if said is None:
            unjudged += 1
            continue
        listed = sum(v == "supported" for v in said) >= needed
        if row.get("planted"):
            planted["items"] += 1
            planted["listed"] += listed
            continue
        label = labels.get(row["fact_id"])
        if label is None:
            unlabelled += 1
            continue
        true = label == "supported"
        for name in ("all", row["dataset"]):
            group = counts[name]
            group["labelled"] += 1
            group["supported"] += true
            group["listed"] += listed
            group["listed_supported"] += listed and true
    groups = {
        name: {
            **group,
            "precision": _share(group["listed_supported"], group["listed"]),
            "recall": _share(group["listed_supported"], group["supported"]),
        }
        for name, group in counts.items()
    }
    return {
        "groups": groups,
        "planted": planted,
        "unlabelled": unlabelled,
        "unjudged": unjudged,
        "needed": needed,
    }


def _share(hits: int, n: int) -> dict[str, Any]:
    return {"hits": hits, "n": n, "value": hits / n if n else None, "interval": wilson(hits, n)}


def render(found: Mapping[str, Any]) -> str:
    def cell(share: Mapping[str, Any]) -> str:
        if share["value"] is None:
            return "—"
        low, high = share["interval"]
        return f"{share['value']:.2f} [{low:.2f}, {high:.2f}] ({share['hits']}/{share['n']})"

    lines = [
        f"gold adjudication on label set G: listed when supported in {found['needed']} of "
        f"{RUNS} runs",
        "",
        f"  {'':13}{'labelled':>9}{'supported':>10}{'listed':>8}  precision of the list"
        "            recall of the list",
    ]
    for name, group in found["groups"].items():
        lines.append(
            f"  {name:13}{group['labelled']:>9}{group['supported']:>10}{group['listed']:>8}  "
            f"{cell(group['precision']):33}{cell(group['recall'])}"
        )
    planted = found["planted"]
    lines += [
        "",
        f"  planted false facts listed: {planted['listed']} of {planted['items']} (should be 0)",
    ]
    if found["unlabelled"] or found["unjudged"]:
        lines.append(
            f"  left out: {found['unlabelled']} with no label, {found['unjudged']} with no verdicts"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", type=Path, default=G, help="the label set's folder")
    parser.add_argument("--labels", type=Path, help="its labels (default: SET/labels.jsonl)")
    parser.add_argument("--verdicts", type=Path, help="G.verdicts.jsonl from an earlier run")
    parser.add_argument("--model", help="ground live with this model, three runs per item")
    parser.add_argument("--out", type=Path, default=Path("runs/adjudication"))
    parser.add_argument("--json", action="store_true", help="print the numbers as JSON")
    args = parser.parse_args(argv)
    if args.labels is None:
        args.labels = args.set / "labels.jsonl"
    if not args.labels.is_file():
        print(
            f"skipped: {args.labels} is not there yet. Label set G's sheets are ticked by hand "
            "and read back with `odke label read` (bench/labels/README.md); until then there "
            "is nothing to measure adjudication against."
        )
        return 0
    if (args.verdicts is None) == (args.model is None):
        parser.error("give --verdicts (an earlier run's) or --model (a live run), one of them")
    rows = not_in_gold(args.set)
    labels = {row["fact"]["id"]: row["verdict"] for row in _jsonl(args.labels)}
    if args.verdicts is not None:
        verdicts = {row["fact_id"]: row["verdicts"] for row in _jsonl(args.verdicts)}
    else:
        from openodke.ground import LLMGrounder
        from openodke.llm import ModelRoles

        verdicts = collect(rows, LLMGrounder(ModelRoles.single(args.model)), args.out)
    found = measure(rows, labels, verdicts)
    print(json.dumps(found, indent=2) if args.json else render(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
