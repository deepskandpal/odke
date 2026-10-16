"""Does the grounder read the claim better with its parts named? (label set G)

    uv run python bench/labels/claim_format.py --cache .odke-cache   # the default grounder
    uv run python bench/labels/claim_format.py --model openai/gpt-…  # any other
    uv run python bench/labels/claim_format.py --limit 100           # a cheaper first look

Grounds label set G's 600 items twice with `LLMGrounder`, once per
`claim_format` ("inline", the default, and "labelled"), and counts each verdict
by what the item is: a real fact the dataset's gold confirms (`match`), one it
doesn't list (`not_in_gold`), a gold fact no extractor wrote (`gold_only`), or a
planted false fact. No hand label is read, so the check can't leak into the
calibration that decides the default; it only says whether that calibration
should test "labelled" at all.

What to look for: "labelled" should call more `match` facts supported without
supporting more planted ones. Gold is distantly supervised, so neither column
is truth; the calibration on the hand labels is (roadmap §9).

Cost: two calls per item at the grounder's price, about $0.70 for all 600 on
the default Haiku. `--cache DIR` keeps the answers, so a rerun of the same
items costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from openodke.eval.sheets import GroundingItem
from openodke.ground import LLMGrounder
from openodke.llm import CachedClient, ModelRoles, ModelSpec

HERE = Path(__file__).parent / "G"
FORMATS = ("inline", "labelled")
VERDICTS = ("supported", "contradicted", "not_found", "unchecked")


def load(limit: int | None, seed: int) -> list[tuple[str, GroundingItem]]:
    rows = (HERE / "items.jsonl").read_text(encoding="utf-8").splitlines()
    private = (HERE / "items.private.jsonl").read_text(encoding="utf-8").splitlines()
    items = []
    for row, about in zip(rows, private, strict=True):
        meta = json.loads(about)
        kind = "planted" if meta["planted"] else meta["gold_status"]
        items.append((kind, GroundingItem.model_validate_json(row)))
    if limit is not None:
        items = random.Random(seed).sample(items, limit)
    return items


def run(
    items: list[tuple[str, GroundingItem]], roles: ModelRoles, claim_format: str, cache: Path | None
) -> Counter:
    client = CachedClient(roles.client_for("ground"), cache)
    grounder = LLMGrounder(roles, client=client, claim_format=claim_format)  # type: ignore[arg-type]
    counts: Counter = Counter()
    for kind, item in items:
        fact = grounder.ground(item.fact, item.document())
        counts[kind, fact.verdict.value] += 1
    print(f"{claim_format}: {grounder.stats['calls']} calls, cost {grounder.stats['cost_usd']}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="the grounder model; default: the package default")
    parser.add_argument("--limit", type=int, help="a seeded random sample of this many items")
    parser.add_argument("--seed", type=int, default=133)
    parser.add_argument("--cache", type=Path, help="the response cache's directory")
    args = parser.parse_args()
    roles = ModelRoles()
    if args.model:
        roles = ModelRoles(ground=ModelSpec(model=args.model, max_tokens=256))
    items = load(args.limit, args.seed)
    kinds = sorted({kind for kind, _ in items})
    results = {f: run(items, roles, f, args.cache) for f in FORMATS}

    print(f"\nmodel {roles.ground.model}, {len(items)} items")
    print("| kind | n | " + " | ".join(f"{f} supported" for f in FORMATS) + " |")
    print("|---|---:|" + "---:|" * len(FORMATS))
    for kind in kinds:
        n = sum(1 for k, _ in items if k == kind)
        cells = [
            f"{results[f][kind, 'supported']} ({results[f][kind, 'supported'] / n:.0%})"
            for f in FORMATS
        ]
        print(f"| {kind} | {n} | " + " | ".join(cells) + " |")
    print("\nfull counts:")
    for f in FORMATS:
        for kind in kinds:
            print(f"  {f:8} {kind:12} " + "  ".join(f"{v}={results[f][kind, v]}" for v in VERDICTS))


if __name__ == "__main__":
    main()
