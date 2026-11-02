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
items costs nothing. The run is held to the spend guard (`bench/spend.py`):
priced first from what it would send, less the cache, and refused past its
provider's cap; its real spend goes in the ledger after.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openodke.eval.sheets import GroundingItem
from openodke.ground import LLMGrounder
from openodke.llm import CachedClient, ModelRoles, ModelSpec

# bench/spend.py and bench/calibrate.py: the ledger, the cap and the estimate.
sys.path.insert(0, str(Path(__file__).parent.parent))
import calibrate  # noqa: E402
import spend  # noqa: E402

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
    items: list[tuple[str, GroundingItem]],
    roles: ModelRoles,
    claim_format: str,
    cache: Path | None,
    ledger: Path = spend.LEDGER,
) -> Counter:
    client = CachedClient(roles.client_for("ground"), cache)
    grounder = LLMGrounder(roles, client=client, claim_format=claim_format)  # type: ignore[arg-type]
    counts: Counter = Counter()
    for kind, item in items:
        fact = grounder.ground(item.fact, item.document())
        counts[kind, fact.verdict.value] += 1
    stats = grounder.stats
    print(f"{claim_format}: {stats['calls']} calls, cost {stats['cost_usd']}")
    live = int(stats["calls"]) - int(stats.get("cached", 0))
    if live:
        spend.record(
            f"claim-format check: G {len(items)} items, {claim_format}",
            roles.ground.model,
            calls=live,
            input_tokens=int(stats["prompt_tokens"]),
            output_tokens=int(stats["completion_tokens"]),
            usd=stats["cost_usd"],
            ledger=ledger,
        )
    return counts


def estimate(items: list[tuple[str, GroundingItem]], model: str, cache: Path | None) -> float:
    """What grounding `items` in both formats would cost, less what the cache holds."""
    store = CachedClient(calibrate._Never(), cache).store if cache is not None else None
    rows = [
        calibrate.GItem(str(at), _labelled(item), "", False, kind)
        for at, (kind, item) in enumerate(items)
    ]
    plans = [calibrate.plan_grounding(model, f"span-{f}", rows, 1, store) for f in FORMATS]
    return sum(job["usd"] or 0.0 for job in calibrate.priced(model, plans))


def _labelled(item: GroundingItem) -> Any:
    from openodke.eval.formats import GroundingLabel

    return GroundingLabel(text=item.text, fact=item.fact, doc_id=item.doc_id, verdict="not_found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="the grounder model; default: the package default")
    parser.add_argument("--limit", type=int, help="a seeded random sample of this many items")
    parser.add_argument("--seed", type=int, default=133)
    parser.add_argument("--cache", type=Path, help="the response cache's directory")
    parser.add_argument(
        "--cap-usd", type=float, help="the provider's cap; bench/spend.py's by default"
    )
    parser.add_argument("--ledger", type=Path, default=spend.LEDGER)
    args = parser.parse_args()
    roles = ModelRoles()
    if args.model:
        roles = ModelRoles(ground=ModelSpec(model=args.model, max_tokens=256))
    items = load(args.limit, args.seed)
    try:
        print(
            spend.check(
                roles.ground.model,
                estimate(items, roles.ground.model, args.cache),
                cap=args.cap_usd,
                ledger=args.ledger,
            )
        )
    except spend.SpendRefused as refused:
        raise SystemExit(f"refused: {refused}") from refused
    kinds = sorted({kind for kind, _ in items})
    results = {f: run(items, roles, f, args.cache, args.ledger) for f in FORMATS}

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
