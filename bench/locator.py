"""The span locator (#112) on a published Re-DocRED comparison: what it places, and what it costs.

    python bench/locator.py runs/cmp/redocred --out OUT            # free: placement only
    python bench/locator.py runs/cmp/redocred --out OUT --live     # + one live grounding run

The saved run is read, never written. For each competitor under `competitors/`,
its triples are rebuilt as facts the way `odke bench run` built them (the
triples stage and the set's ontology, so every fact's span is the whole
document) and the locator is run over them: how many it placed, and how wide
the windows are.

`--live` then makes two runs, both in the saved run's own mode (the paper's
True/False prompt, its ground model) so that only the passage differs:

- **located**: every placed fact, through `LLMGrounder(locate=True)`, shown its
  located window instead of the whole document;
- **retest**: a fixed random sample of the placed facts, shown the whole
  document again, exactly as the saved run did. Its disagreement with the
  saved verdicts is the bench's own noise, since the ground model runs at the
  provider's default temperature.

The saved whole-document verdicts are read off the run's predictions: a triple
in `predictions/grounding.jsonl` was supported, and one only in
`extraction-alone.jsonl` was not (paper mode reads False as `not_found`).
Precision is the dataset's own `score` against gold.

The rule, fixed before the run: the locator is on by default only if, for
both competitors,

1. located-vs-saved agreement is at least the lower end of the 95% Wilson
   interval of retest-vs-saved agreement; and
2. neither the 95% paired-bootstrap interval of the precision difference
   (located minus whole document) nor that of the difference in gold-correct
   facts kept lies wholly below zero.

Keys come from the environment as LiteLLM reads them; `ODKE_ENV_FILE` or
`--env-file` names a file of KEY=value lines to load first. The run is
estimated from token counts before any call and refused above `--max-usd`.
Results are kept under OUT, so a second run reads them instead of paying again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openodke import Document, Fact, GroundingVerdict, Ontology, SpanOrigin
from openodke.chunking import _segment
from openodke.eval.cost import CostMeter, StageCost
from openodke.eval.datasets.redocred import _norm, score
from openodke.ground import LLMGrounder, locate_span
from openodke.ground.llm import build_messages
from openodke.interop.triples import read_triples, to_fact
from openodke.llm import ModelRoles, ModelSpec

SYSTEMS = (("lgt", "LLMGraphTransformer"), ("neo4j", "neo4j-graphrag"))
MODEL = "anthropic/claude-haiku-4-5-20251001"
RETEST = 150
SEED = 112
BOOTSTRAP = 2000
SUPPORTED, NOT_FOUND = GroundingVerdict.SUPPORTED.value, GroundingVerdict.NOT_FOUND.value


# --------------------------------------------------------------------------- #
# the saved run
# --------------------------------------------------------------------------- #


class System:
    """One competitor's saved run: its facts, their documents, and the saved verdicts."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        ontology = Ontology.from_dict(json.loads((folder / "ontology.json").read_text()))
        self.labels: dict[str, str] = json.loads((folder / "dataset.json").read_text())[
            "relation_labels"
        ]
        self.docs = {
            p.stem: Document(id=p.stem, text=p.read_text(encoding="utf-8"))
            for p in sorted((folder / "docs").glob("*.txt"))
        }
        rows = read_triples(folder / "facts.jsonl")
        self.facts = [to_fact(row, self.docs[row.doc], ontology) for row in rows]
        self.doc_of = [row.doc for row in rows]
        kept = _predictions(folder / "predictions" / "grounding.jsonl")
        self.whole = [
            SUPPORTED if self.triple(i) in kept.get(self.doc_of[i], set()) else NOT_FOUND
            for i in range(len(self.facts))
        ]
        self.gold = [json.loads(line) for line in (folder / "gold.jsonl").read_text().splitlines()]
        self.correct = _correct(self)

    def triple(self, i: int) -> tuple[str, str, str]:
        """The fact as the scorer reads it: labels, the dataset's relation, the object."""
        fact = self.facts[i]
        obj = fact.object_entity
        return (
            fact.subject.label or fact.subject.key,
            self.labels.get(fact.predicate, fact.predicate),
            (obj.label or obj.key) if obj is not None else str(fact.object_value),
        )

    def precision(self, indices: Sequence[int]) -> float | None:
        """The dataset's own precision over these facts' triples."""
        predicted: dict[str, list[tuple[str, str, str]]] = {}
        for i in indices:
            predicted.setdefault(self.doc_of[i], []).append(self.triple(i))
        return score(self.gold, predicted)["precision"] if indices else None


def _predictions(path: Path) -> dict[str, set[tuple[str, str, str]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["id"]: {tuple(t) for t in row["triples"]} for row in rows}


def _correct(system: System) -> list[int]:
    """1 for a fact the gold has, matched as `score` matches: each gold fact once, in order."""
    out = [0] * len(system.facts)
    by_doc: dict[str, list[int]] = {}
    for i, doc in enumerate(system.doc_of):
        by_doc.setdefault(doc, []).append(i)
    for row in system.gold:
        names = [{_norm(n) for n in cluster} for cluster in row["entities"]]
        facts = {(h, _norm(r), t) for h, r, t in row["facts"]}
        found: set[tuple[int, str, int]] = set()
        seen: set[tuple[str, str, str]] = set()
        for i in by_doc.get(row["id"], []):
            triple = system.triple(i)
            if triple in seen:
                continue
            seen.add(triple)
            s, rel, o = (_norm(part) for part in triple)
            heads = [k for k, cluster in enumerate(names) if s in cluster]
            tails = [k for k, cluster in enumerate(names) if o in cluster]
            match = next(
                ((h, rel, t) for h in heads for t in tails if (h, rel, t) in facts - found), None
            )
            if match is not None:
                found.add(match)
                out[i] = 1
    return out


# --------------------------------------------------------------------------- #
# placement: free
# --------------------------------------------------------------------------- #


def place(system: System) -> tuple[list[int], dict[str, Any]]:
    """The facts the locator placed, and the widths of their windows."""
    placed, widths, two = [], [], 0
    for i, fact in enumerate(system.facts):
        doc = system.docs[system.doc_of[i]]
        span = locate_span(fact, doc)
        if span is None:
            continue
        placed.append(i)
        widths.append(span.end - span.start)
        # A window is one sentence unless a sentence break falls inside it.
        two += _sentences(span.quote or "") > 1
    doc_widths = [len(system.docs[system.doc_of[i]].text) for i in range(len(system.facts))]
    q1, q2, q3 = statistics.quantiles(widths, n=4, method="inclusive")
    return placed, {
        "facts": len(system.facts),
        "placed": len(placed),
        "placed_rate": len(placed) / len(system.facts),
        "two_sentence_windows": two,
        "width": {"min": min(widths), "p25": q1, "median": q2, "p75": q3, "max": max(widths)},
        "document_median": statistics.median(doc_widths),
        "literal_objects_placed": sum(system.facts[i].object_entity is None for i in placed),
        "literal_objects": sum(f.object_entity is None for f in system.facts),
    }


def _sentences(text: str) -> int:
    return sum(1 for _ in _segment(text))


# --------------------------------------------------------------------------- #
# the estimate, and the live runs
# --------------------------------------------------------------------------- #


def _tokens(messages: list[Any]) -> int:
    import litellm

    return int(
        litellm.token_counter(
            model=MODEL, messages=[{"role": m.role, "content": m.content} for m in messages]
        )
    )


def estimate(systems: dict[str, System], placed: dict[str, list[int]]) -> dict[str, Any]:
    """Tokens and USD the live runs will take, calibrated on the saved run's own bill.

    The token counter is not Anthropic's tokenizer and does not see the output
    schema, so its count of the saved whole-document prompts is fitted to what
    the saved run was billed two ways, in proportion and as a fixed overhead per
    call, and the dearer is taken.
    """
    import litellm

    counted = billed = completions = calls = 0
    asked = 0
    for name, system in systems.items():
        report = json.loads((system.folder / "report.json").read_text())["breakdown"]
        billed += report["+ grounding"]["prompt_tokens"]
        completions += report["+ grounding"]["completion_tokens"]
        calls += report["+ grounding"]["model_calls"]
        for i, fact in enumerate(system.facts):
            doc = system.docs[system.doc_of[i]]
            counted += _tokens(build_messages(fact, doc.text, binary=True))
        for i in placed[name]:
            doc = system.docs[system.doc_of[i]]
            span = locate_span(system.facts[i], doc)
            assert span is not None
            asked += _tokens(build_messages(system.facts[i], span.resolve(doc), binary=True))
        for i in _sample(placed[name]):
            doc = system.docs[system.doc_of[i]]
            asked += _tokens(build_messages(system.facts[i], doc.text, binary=True))
    n_calls = sum(len(p) + len(_sample(p)) for p in placed.values())
    prompt = max(asked * billed / counted, asked + n_calls * (billed - counted) / calls)
    completion = n_calls * completions / calls
    usd_in, usd_out = litellm.cost_per_token(
        model=MODEL, prompt_tokens=int(prompt), completion_tokens=int(completion)
    )
    return {
        "calls": n_calls,
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "usd": usd_in + usd_out,
    }


def _sample(placed: Sequence[int]) -> list[int]:
    return sorted(random.Random(SEED).sample(list(placed), min(RETEST, len(placed))))


def ground(
    system: System, indices: Sequence[int], *, locate: bool, meter: CostMeter, stage: str
) -> list[str]:
    """Each fact's verdict from one live run in the saved run's mode, but for the passage."""
    roles = ModelRoles(ground=ModelSpec(model=MODEL, max_tokens=256))
    grounder = LLMGrounder(
        roles,
        client=meter.client(roles.client_for("ground"), stage),
        max_workers=4,
        context="span" if locate else "document",
        verdicts="binary",
        locate=locate,
    )
    batches: dict[str, list[int]] = {}
    for i in indices:
        batches.setdefault(system.doc_of[i], []).append(i)
    work = [([system.facts[i] for i in rows], system.docs[doc]) for doc, rows in batches.items()]
    verdicts: dict[int, Fact] = {}
    for rows, grounded in zip(batches.values(), grounder.ground_documents(work), strict=True):
        verdicts.update(zip(rows, grounded, strict=True))
    if locate:
        origins = {verdicts[i].evidence[0].span_origin for i in indices}
        assert origins == {SpanOrigin.LOCATED}, origins
    return [verdicts[i].verdict.value for i in indices]


def cached_run(
    out: Path,
    system: System,
    indices: Sequence[int],
    *,
    locate: bool,
    meter: CostMeter,
    stage: str,
) -> list[str]:
    """`ground`, unless OUT already holds this run's verdicts: a rerun never pays twice."""
    path = out / f"{stage}.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [r["fact"] for r in rows] == list(indices):
            return [r["verdict"] for r in rows]
    verdicts = ground(system, indices, locate=locate, meter=meter, stage=stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"fact": i, "verdict": v}) + "\n"
            for i, v in zip(indices, verdicts, strict=True)
        )
    )
    cost = StageCost.of(stage, [r for r in meter.records if r.stage == stage])
    (out / f"{stage}.cost.json").write_text(cost.model_dump_json(indent=2))
    return verdicts


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of a proportion."""
    if not n:
        return 0.0, 1.0
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def bootstrap(
    correct: Sequence[int], whole: Sequence[int], located: Sequence[int]
) -> dict[str, tuple[float, float]]:
    """Paired 95% intervals: precision difference, and gold-correct facts kept, located - whole."""
    rng = random.Random(SEED)
    n = len(correct)
    precision, kept = [], []
    for _ in range(BOOTSTRAP):
        draw = [rng.randrange(n) for _ in range(n)]
        kw = sum(whole[i] for i in draw)
        kl = sum(located[i] for i in draw)
        true = sum(correct[i] for i in draw)
        hit_w = sum(correct[i] * whole[i] for i in draw)
        hit_l = sum(correct[i] * located[i] for i in draw)
        if kw and kl:
            precision.append(hit_l / kl - hit_w / kw)
        if true:
            kept.append((hit_l - hit_w) / true)

    def interval(xs: list[float]) -> tuple[float, float]:
        # Undefined (one side kept nothing in every draw) fails the rule: nan >= 0 is False.
        if not xs:
            return math.nan, math.nan
        xs.sort()
        return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]

    return {"precision_diff": interval(precision), "correct_kept_diff": interval(kept)}


def compare(
    system: System,
    placed: Sequence[int],
    located: Sequence[str],
    retest: Sequence[str],
    sample: Sequence[int],
    costs: dict[str, StageCost],
) -> dict[str, Any]:
    whole = [system.whole[i] for i in placed]
    answered = [k for k, v in enumerate(located) if v != "unchecked"]
    agree = sum(whole[k] == located[k] for k in answered)
    confusion = Counter(f"whole {whole[k]} / located {located[k]}" for k in range(len(placed)))
    at = {i: k for k, i in enumerate(placed)}
    pairs = [(system.whole[i], retest[j], located[at[i]]) for j, i in enumerate(sample)]
    retest_agree = sum(w == r for w, r, _ in pairs if r != "unchecked")
    retest_n = sum(r != "unchecked" for _, r, _ in pairs)
    sample_agree = sum(w == loc for w, _, loc in pairs if loc != "unchecked")
    correct = [system.correct[i] for i in placed]
    kept_w = [int(v == SUPPORTED) for v in whole]
    kept_l = [int(v == SUPPORTED) for v in located]
    intervals = bootstrap(correct, kept_w, kept_l)
    lo, hi = wilson(retest_agree, retest_n)
    agreement = agree / len(answered) if answered else 0.0
    per_fact = {
        stage: (c.cost_usd if c.cost_usd is not None else _price(c)) / c.calls if c.calls else None
        for stage, c in costs.items()
    }
    return {
        "facts": len(placed),
        "unchecked": len(placed) - len(answered),
        "agreement": agreement,
        "agreement_ci": wilson(agree, len(answered)),
        "confusion": dict(sorted(confusion.items())),
        "retest_sample": len(sample),
        "retest_agreement": retest_agree / retest_n if retest_n else None,
        "retest_ci": (lo, hi),
        "located_agreement_on_sample": sample_agree / len(pairs) if pairs else None,
        "precision_placed_ungrounded": system.precision(placed),
        "precision_whole": system.precision([i for i, k in zip(placed, kept_w, strict=True) if k]),
        "precision_located": system.precision(
            [i for i, k in zip(placed, kept_l, strict=True) if k]
        ),
        "kept_whole": sum(kept_w),
        "kept_located": sum(kept_l),
        "correct_kept_whole": sum(c * k for c, k in zip(correct, kept_w, strict=True)),
        "correct_kept_located": sum(c * k for c, k in zip(correct, kept_l, strict=True)),
        "gold_correct_placed": sum(correct),
        **intervals,
        "usd_per_fact": per_fact,
        "prompt_tokens_per_fact": {
            stage: c.prompt_tokens / c.calls if c.calls else None for stage, c in costs.items()
        },
        "passes": agreement >= lo
        and intervals["precision_diff"][1] >= 0
        and intervals["correct_kept_diff"][1] >= 0,
    }


def _price(cost: StageCost) -> float:
    import litellm

    usd_in, usd_out = litellm.cost_per_token(
        model=MODEL, prompt_tokens=cost.prompt_tokens, completion_tokens=cost.completion_tokens
    )
    return float(usd_in + usd_out)


def saved_usd_per_fact(system: System) -> float:
    """What the saved whole-document run cost per fact, over all its facts."""
    import litellm

    row = json.loads((system.folder / "report.json").read_text())["breakdown"]["+ grounding"]
    usd_in, usd_out = litellm.cost_per_token(
        model=MODEL, prompt_tokens=row["prompt_tokens"], completion_tokens=row["completion_tokens"]
    )
    return float(usd_in + usd_out) / row["model_calls"]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("redocred", type=Path, help="The prepared Re-DocRED set of a comparison.")
    parser.add_argument("--out", type=Path, required=True, help="Where results are written.")
    parser.add_argument("--live", action="store_true", help="Make the two grounding runs.")
    parser.add_argument("--max-usd", type=float, default=0.80, help="Refuse a dearer estimate.")
    parser.add_argument(
        "--env-file", default=os.environ.get("ODKE_ENV_FILE"), help="KEY=value lines to load."
    )
    args = parser.parse_args()

    systems = {name: System(args.redocred / "competitors" / name) for name, _ in SYSTEMS}
    placed: dict[str, list[int]] = {}
    report: dict[str, Any] = {"model": MODEL, "placement": {}, "grounding": {}}
    for name, system in systems.items():
        placed[name], report["placement"][name] = place(system)
        report["placement"][name]["saved_usd_per_fact"] = saved_usd_per_fact(system)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.live:
        from competitors import load_env

        load_env(args.env_file)
        cost = estimate(systems, placed)
        report["estimate"] = cost
        print(f"estimate: {cost['calls']} calls, ${cost['usd']:.3f}", file=sys.stderr)
        cached = all(
            (args.out / name / f"{stage}.jsonl").exists()
            for name in systems
            for stage in ("located", "retest")
        )
        if cost["usd"] > args.max_usd and not cached:
            raise SystemExit(f"estimated ${cost['usd']:.2f} is over --max-usd {args.max_usd}")
        spent: list[StageCost] = []
        for name, system in systems.items():
            # One meter per competitor: a stage's cost file is that competitor's calls alone.
            meter = CostMeter()
            folder = args.out / name
            sample = _sample(placed[name])
            located = cached_run(
                folder, system, placed[name], locate=True, meter=meter, stage="located"
            )
            retest = cached_run(folder, system, sample, locate=False, meter=meter, stage="retest")
            costs = {
                stage: StageCost.model_validate_json((folder / f"{stage}.cost.json").read_text())
                for stage in ("located", "retest")
            }
            spent += costs.values()
            report["grounding"][name] = compare(
                system, placed[name], located, retest, sample, costs
            )
        report["spend"] = {
            "model": MODEL,
            "calls": sum(c.calls for c in spent),
            "input_tokens": sum(c.prompt_tokens for c in spent),
            "output_tokens": sum(c.completion_tokens for c in spent),
            "usd": sum(c.cost_usd if c.cost_usd is not None else _price(c) for c in spent),
        }
        report["on_by_default"] = all(g["passes"] for g in report["grounding"].values())

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
