"""The comparison tables: every system x configuration, every metric, per dataset.

    python bench/tables.py /path/to/runs/cmp

Text2KGBench rows average the per-ontology scores (each ontology has the same
number of sentences); counts are summed. Costs are at list price, from LiteLLM's
price table for whichever models each set ran (its `odke.json`, and a competitor's
`usage.json`); a model LiteLLM has no price for shows its tokens instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROWS = ("extraction alone", "+ grounding", "+ corroboration")
SYSTEMS = (
    ("openodke", ""),
    ("LLMGraphTransformer", "competitors/lgt"),
    ("neo4j-graphrag", "competitors/neo4j"),
)


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
    """Facts the grounding gate removed: extracted, less what + grounding kept.

    Not a verdict count: which verdicts the gate refuses depends on the mode.
    Paper mode refuses not_found too; three-way mode keeps it and refuses only
    contradicted, so counting not_found there counts facts that were kept.
    """
    rows = report["breakdown"]
    return int(rows[ROWS[0]]["facts"] or 0) - int(rows[ROWS[1]]["facts"] or 0)


def failed(folder: Path) -> int:
    """Documents a competitor's extraction lost, from its `usage.json`; 0 for openodke's."""
    return int((load(folder / "usage.json") or {}).get("failed_documents") or 0)


def models(folder: Path) -> tuple[str | None, str | None]:
    """The extraction and grounding model strings a set's config names."""
    config = load(folder / "odke.json") or {}
    spec = config.get("models", {})
    extract = spec.get("extract")
    if isinstance(extract, dict):
        extract = extract.get("model")
    ground = spec.get("ground")
    if isinstance(ground, dict):
        ground = ground.get("model")
    # A config that names only `extract` grounds on it too (`ModelRoles`).
    return extract, ground or extract


def price(model: str | None, prompt: float, completion: float) -> float | None:
    """List price of the tokens, from LiteLLM's table; None when it has no price."""
    if not model or not (prompt or completion):
        return 0.0 if model else None
    try:
        import litellm

        p, c = litellm.cost_per_token(
            model=model, prompt_tokens=int(prompt), completion_tokens=int(completion)
        )
    except Exception:
        return None
    return p + c


def cost(
    report: dict[str, Any], folder: Path, system_dir: str
) -> tuple[float | None, float | None]:
    e, g = report["breakdown"][ROWS[0]], report["breakdown"][ROWS[1]]
    extract_model, ground_model = models(folder)
    if system_dir:
        usage = load(folder / "usage.json") or {}
        extract = price(
            usage.get("model") or extract_model,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
    else:
        extract = price(extract_model, e["prompt_tokens"], e["completion_tokens"])
    ground = price(
        ground_model,
        g["prompt_tokens"] - e["prompt_tokens"],
        g["completion_tokens"] - e["completion_tokens"],
    )
    return extract, ground


def dollars(values: list[float | None]) -> str:
    return "n/a" if any(v is None for v in values) else f"${sum(v or 0 for v in values):.2f}"


def pct(v: Any) -> str:
    return "–" if v is None else f"{v:.1%}"


def table(title: str, sets: list[Path], keys: list[str], counts: list[str]) -> list[str]:
    out = [f"## {title}", ""]
    out.append(
        "| System | Configuration | " + " | ".join(keys + counts) + " | rejected by grounder |"
    )
    out.append("|---" * (len(keys) + len(counts) + 3) + "|")
    spend: dict[str, tuple[str, str]] = {}
    for system, sub in SYSTEMS:
        # A set whose extraction lost documents scores them as empty answers, which
        # is not the system's score: it is left out, as a set with no report is.
        dropped = [d for d in sets if failed(d / sub)]
        reports = [
            (d / sub, r)
            for d in sets
            if d not in dropped and (r := load(d / sub / "report.json")) is not None
        ]
        gone = f"({len(dropped)} of {len(sets)} sets dropped: documents failed to extract)"
        if not reports:
            empty = gone if dropped else "(no report)"
            out.append(f"| {system} | {empty} |" + " |" * (len(keys) + len(counts) + 1))
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
        spend[system] = (dollars([c[0] for c in costs]), dollars([c[1] for c in costs]))
        missing = len(sets) - len(reports) - len(dropped)
        if missing:
            out.append(
                f"| | ({missing} of {len(sets)} sets missing) |"
                + " |" * (len(keys) + len(counts) + 1)
            )
        if dropped:
            out.append(f"| | {gone} |" + " |" * (len(keys) + len(counts) + 1))
    out.append("")
    out.append(
        "Cost: " + "; ".join(f"{s} extraction {e} + grounding {g}" for s, (e, g) in spend.items())
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
