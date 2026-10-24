"""The spend guard: one ledger of every live run, and a cap no run may pass (harness §9).

    uv run python bench/spend.py                    # what each provider has spent, against its cap
    uv run python bench/spend.py --seed RAW.jsonl   # add a raw log's rows the ledger lacks

`spend.jsonl`, beside this file, is the ledger: one row per live run, written by
the bench script that made it, in the shape every row has always had:

    {"who": "...", "model": "provider/model", "calls": 0, "input_tokens": 0,
     "output_tokens": 0, "usd": 0.0, "date": "YYYY-MM-DD"}

(`date` is new; older rows lack it, and a raw log's tokens may be null.)

Before a live run calls anything, it estimates what it will cost and asks
`check`: the ledger's total for the run's provider, plus the estimate, must stay
within the provider's cap, or the run refuses to start. Providers are billed
apart, so each has its own cap: Anthropic's is the $20 the 1.0 work was given
(roadmap §10); any other provider has none until a run names one, and a run
with no cap for its provider is refused. After the run, `record` appends what
it really spent, from the provider's own counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LEDGER = Path(__file__).with_name("spend.jsonl")
# USD per provider for every live run of 1.0. OpenAI is billed apart and has none
# until a run names one.
CAPS: dict[str, float] = {"anthropic": 20.0}
# USD per million tokens, input and output, at list price: what an estimate uses
# where a run gives no price of its own.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5-20251001": (1.0, 5.0),
    "openai/gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "openai/gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "anthropic/claude-sonnet-5-5": (2.0, 10.0),
    # Up to 100K tokens a prompt, which every calibration message is.
    "anthropic/claude-haiku-5-5": (0.10, 0.50),
}


class SpendRefused(RuntimeError):
    """A live run that would take its provider's ledger past the cap, refused before any call."""


def provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else "openai"


def rows(ledger: Path = LEDGER) -> list[dict[str, Any]]:
    """Every row of the ledger; none when it does not exist yet."""
    if not ledger.is_file():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


def spent(ledger: Path = LEDGER) -> dict[str, float]:
    """USD the ledger holds, by provider."""
    totals: dict[str, float] = defaultdict(float)
    for row in rows(ledger):
        totals[provider(str(row["model"]))] += float(row.get("usd") or 0.0)
    return dict(totals)


def cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: Mapping[str, tuple[float, float]] | None = None,
) -> float:
    """USD for `input_tokens` and `output_tokens` on `model`, at its list price."""
    table = PRICES if prices is None else prices
    if model not in table:
        raise SpendRefused(f"no price for {model}: give the run a price so it can be estimated")
    price_in, price_out = table[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def check(
    model: str,
    estimate_usd: float,
    *,
    cap: float | None = None,
    ledger: Path = LEDGER,
) -> str:
    """Refuse a run whose estimate would take its provider past the cap; else say what is left.

    `cap` is the provider's cap in USD, `CAPS`' by default. A provider with no
    cap refuses every live run until one is given.
    """
    name = provider(model)
    limit = CAPS.get(name) if cap is None else cap
    if limit is None:
        raise SpendRefused(f"no cap for {name}: it is billed apart; name one (--cap-usd)")
    before = spent(ledger).get(name, 0.0)
    if before + estimate_usd > limit:
        raise SpendRefused(
            f"{name}: ${before:.2f} spent + ${estimate_usd:.2f} estimated = "
            f"${before + estimate_usd:.2f}, over the ${limit:.2f} cap; nothing was called"
        )
    return (
        f"{name}: ${before:.2f} spent + ${estimate_usd:.2f} estimated, "
        f"${limit - before - estimate_usd:.2f} left of ${limit:.2f}"
    )


def record(
    who: str,
    model: str,
    *,
    calls: int,
    input_tokens: int | None,
    output_tokens: int | None,
    usd: float | None,
    ledger: Path = LEDGER,
) -> dict[str, Any]:
    """Append one run's real spend to the ledger, and return the row."""
    row = {
        "who": who,
        "model": model,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": round(usd, 6) if usd is not None else None,
        "date": dt.date.today().isoformat(),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def seed(raw: Path, ledger: Path = LEDGER) -> int:
    """Add the rows of a raw log the ledger does not hold yet; how many were added."""
    held = {json.dumps(r, sort_keys=True) for r in rows(ledger)}
    added = 0
    with ledger.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows(raw):
            line = json.dumps(row, sort_keys=True)
            if line not in held:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                held.add(line)
                added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--seed", type=Path, help="a raw log whose rows to add, once each")
    args = parser.parse_args()
    if args.seed is not None:
        print(f"added {seed(args.seed, args.ledger)} row(s) from {args.seed}")
    totals = spent(args.ledger)
    for name in sorted(set(totals) | set(CAPS)):
        cap = CAPS.get(name)
        left = f", ${cap - totals.get(name, 0.0):.2f} left of ${cap:.2f}" if cap else ", no cap"
        print(f"{name}: ${totals.get(name, 0.0):.2f} spent{left}")


if __name__ == "__main__":
    main()
