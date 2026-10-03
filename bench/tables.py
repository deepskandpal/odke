"""The comparison tables: every system x configuration, every metric, per dataset.

    python bench/tables.py /path/to/runs/cmp

Text2KGBench rows average the per-ontology scores (each ontology has the same
number of sentences); counts are summed. Costs are at list price per million
tokens: Sonnet 5.5 extracting ($2 in, $10 out) and Haiku 4.5 grounding ($1, $5).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROWS = ("extraction alone", "+ grounding", "+ corroboration")
SYSTEMS = (
    ("openodke", ""),
    ("LLMGraphTransformer", "competitors/lgt"),
    ("neo4j-graphrag", "competitors/neo4j"),
)
EXTRACT, GROUND = (2.0, 10.0), (1.0, 5.0)


def load(path: Path) -> dict[str, Any] | None:
    """A report, skipping anything a library printed to stdout before the JSON."""
    try:
        text = path.read_text()
    except OSError:
        return None
    start = 0 if text.startswith("{") else text.find("\n{") + 1
    try:
        return json.loads(text[start:]) if text.strip() else None
    except json.JSONDecodeError:
        return None


def rejected(report: dict[str, Any]) -> int:
    note = next((n for n in report["notes"] if n.startswith("grounder verdicts")), "")
    found = re.search(r"not_found (\d+)", note)
    return int(found.group(1)) if found else 0


def cost(report: dict[str, Any], folder: Path, system_dir: str) -> tuple[float, float]:
    e, g = report["breakdown"][ROWS[0]], report["breakdown"][ROWS[1]]
    if system_dir:
        usage = load(folder / "usage.json") or {}
        extract = (
            usage.get("input_tokens", 0) * EXTRACT[0] / 1e6
            + usage.get("output_tokens", 0) * EXTRACT[1] / 1e6
        )
    else:
        extract = e["prompt_tokens"] * EXTRACT[0] / 1e6 + e["completion_tokens"] * EXTRACT[1] / 1e6
    ground = (g["prompt_tokens"] - e["prompt_tokens"]) * GROUND[0] / 1e6 + (
        g["completion_tokens"] - e["completion_tokens"]
    ) * GROUND[1] / 1e6
    return extract, ground


def pct(v: Any) -> str:
    return "–" if v is None else f"{v:.1%}"


def table(title: str, sets: list[Path], keys: list[str], counts: list[str]) -> list[str]:
    out = [f"## {title}", ""]
    out.append(
        "| System | Configuration | " + " | ".join(keys + counts) + " | rejected by grounder |"
    )
    out.append("|---" * (len(keys) + len(counts) + 3) + "|")
    spend: dict[str, tuple[float, float]] = {}
    for system, sub in SYSTEMS:
        reports = [(d / sub, r) for d in sets if (r := load(d / sub / "report.json")) is not None]
        if not reports:
            out.append(f"| {system} | (no report) |" + " |" * (len(keys) + len(counts) + 1))
            continue
        for row in ROWS:
            means = [
                sum(r["breakdown"][row].get(k) or 0 for _, r in reports) / len(reports)
                for k in keys
            ]
            sums = [sum(r["breakdown"][row].get(k) or 0 for _, r in reports) for k in counts]
            rej = sum(rejected(r) for _, r in reports) if row != ROWS[0] else 0
            label = f"**{system}**" if row == ROWS[0] else ""
            out.append(
                f"| {label} | {row} | "
                + " | ".join(pct(v) for v in means)
                + " | "
                + " | ".join(f"{v:.0f}" for v in sums)
                + f" | {rej if row != ROWS[0] else '–'} |"
            )
        costs = [cost(r, f, sub) for f, r in reports]
        spend[system] = (sum(c[0] for c in costs), sum(c[1] for c in costs))
        missing = len(sets) - len(reports)
        if missing:
            out.append(
                f"| | ({missing} of {len(sets)} sets missing) |"
                + " |" * (len(keys) + len(counts) + 1)
            )
    out.append("")
    out.append(
        "Cost: "
        + "; ".join(f"{s} extraction ${e:.2f} + grounding ${g:.2f}" for s, (e, g) in spend.items())
    )
    out.append("")
    return out


def main(root: Path) -> None:
    t2k = sorted((root / "t2k").glob("ont_*"), key=lambda p: int(p.name.split("_")[1]))
    lines = table(
        f"Text2KGBench (Wikidata-TekGen) — {len(t2k)} ontologies x 20 sentences",
        t2k,
        ["precision", "recall", "f1", "onto_conf", "rel_halluc", "sub_halluc", "obj_halluc"],
        ["triples", "hallucinated_triples"],
    )
    lines += table(
        "Re-DocRED — 50 test documents",
        [root / "redocred"],
        ["precision", "recall", "f1", "onto_conf"],
        ["triples", "gold_triples", "hallucinated_triples"],
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
