"""The grounder contest and the calibration cards: label sets G, R and F (#134, #135).

    uv run python bench/calibrate.py --estimate --cache DIR   # what a run would cost; no call
    uv run python bench/calibrate.py --cache DIR -o OUT       # the default Haiku, every judge
    uv run python bench/calibrate.py --model openai/gpt-5.4-mini-2026-03-17 --cap openai=10 ...
    uv run python bench/calibrate.py --model A --model B ...  # Claude and another: self-preference

Grounding (label set G, `bench/labels/G`): four grounder variants, `ground.span@1`
with the claim inline (P1, the default) and labelled, `ground.onyx@1` and
`ground.statement@1`, each over the hand-labelled real facts and the 100 planted
false ones, three runs each (runs 2 and 3 are keyed apart in the cache,
`ModelSpec.repeat`). Each variant gets a card on the gate split, and its dev
numbers are reported apart. Then:

- **the contest**: each challenger against P1, paired item by item on the gate
  split (an item passes when the judge's keep or refuse, run 1, matches the
  person's), with the Evaluator's own comparison (`openodke.eval.compare`:
  flips, the paired bootstrap, McNemar's exact test, three verdicts and the
  detection limit). A challenger replaces P1 only when it is better by more
  than the detection limit; an inconclusive result keeps P1. Dev numbers never
  pick a winner;
- **gold adjudication**: on G's hand-labelled `not_in_gold` facts, how often
  "P1 supports it in 2 of 3 runs" agrees with the person's "supported";
- **non-determinism**: the share of facts whose verdict changed over the
  three runs; above 5%, the "refuse only if 2 of 3 runs refuse" card too;
- **self-preference**, when a Claude model and a non-Claude one are both run:
  P1's fail-recall on the same Claude-extracted facts, judged by each, paired.

The pair judge (P4, label set R's gate split, both orders) and the
fact-equivalence judge (P5, F's labelled pairs, both orders) get a card each.

Every live run is held to the spend guard (`bench/spend.py`): it estimates its
cost from the messages it would send, minus what the cache already holds, and
refuses to start past its provider's cap; after it, the ledger gets its real
spend. `--estimate` prints the estimate and stops. Writes, under `-o`: `cards/`
(the gate cards), `cards-dev/`, `items/` (every item's verdicts, per model and
judge), `results.md` and `results.json`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openodke.calibration import (
    FLIP_LIMIT,
    Card,
    JudgedFact,
    JudgedPair,
    LabelSet,
    Made,
    equivalence_card,
    grounding_card,
    label_hash,
    pair_card,
    share,
    voted,
)
from openodke.calibration.judging import ground_labelled, prompt_hashes
from openodke.corroborate import judge as pair_module
from openodke.corroborate.judge import Mention, PairJudge
from openodke.eval import equivalence as fact_module
from openodke.eval.compare import Comparison, ItemRow, compare_items
from openodke.eval.equivalence import FactJudge
from openodke.eval.formats import FactPair, GroundingLabel
from openodke.eval.stats import mcnemar
from openodke.ground.llm import LLMGrounder, _schema_for, build_messages, render_claim
from openodke.ground.span import located
from openodke.llm import CachedClient, Completion, ModelRoles, ModelSpec, RecordedClient
from openodke.llm.base import LLMClient, Message
from openodke.llm.budget import Budget, BudgetExceeded, Ledger, estimate_tokens
from openodke.llm.cache import cache_key, replay_directory
from openodke.llm.roles import DEFAULT_GROUND
from openodke.prompts import get as get_prompt

# bench/spend.py, beside this script: the ledger and the cap.
sys.path.insert(0, str(Path(__file__).parent))
import spend  # noqa: E402

LABELS = Path(__file__).parent / "labels"
# Each grounder variant: the prompt it sends and how it writes the claim.
VARIANTS: dict[str, tuple[str, str]] = {
    "span-inline": ("ground.span@1", "inline"),
    "span-labelled": ("ground.span@1", "labelled"),
    "onyx": ("ground.onyx@1", "inline"),
    "statement": ("ground.statement@1", "inline"),
}
P1 = "span-inline"
RUNS = 3
# Output tokens an answer takes, for an estimate where the cache has no answer:
# the mean Haiku gave for ground.span@1 on G (9.3), and the shape of the others.
OUT_TOKENS = {
    "ground.span": 10,
    "ground.statement": 10,
    "ground.onyx": 2,
    "pair": 40,
    "fact_equiv": 10,
}
G_VERSION = "sheets 01-04 (labelled 20-24 Oct 2026)"
R_VERSION = "Re-DocRED's own labels, the 100-pair audit over them (17 Oct 2026)"
F_VERSION = "sheets 01-02 (labelled 17 Oct 2026)"


# --------------------------------------------------------------------------- #
# The label sets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GItem:
    """One G item: the row to ground (its label or, planted, what it must not be), and where."""

    id: str
    row: GroundingLabel
    split: str
    planted: bool
    gold_status: str | None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_g(*, every: bool = False) -> list[GItem]:
    """G's hand-labelled real facts and all its planted ones; with `every`, every real fact too.

    A real fact with no hand label is grounded but scores nothing: its answers
    wait in the cache for the sheets still to be ticked.
    """
    rows = _jsonl(LABELS / "G" / "items.jsonl")
    private = _jsonl(LABELS / "G" / "items.private.jsonl")
    labels = {r["fact"]["id"]: r["verdict"] for r in _jsonl(LABELS / "G" / "labels.jsonl")}
    out = []
    for row, about in zip(rows, private, strict=True):
        planted = about["planted"] is not None
        verdict = "not_found" if planted else labels.get(about["fact_id"])
        if verdict is None and not every:
            continue
        label = GroundingLabel.model_validate({**row, "verdict": verdict or "not_found"})
        split = "planted" if planted else about["split"]
        if verdict is None:
            split = f"{split}-unlabelled"
        out.append(GItem(about["id"], label, split, planted, about["gold_status"]))
    return out


@dataclass(frozen=True)
class RItem:
    id: str
    a: Mention
    b: Mention
    same: bool
    split: str
    audited: bool


def load_r(splits: Iterable[str] = ("gate",)) -> tuple[list[RItem], dict[str, Any]]:
    """R's pairs in `splits`, each labelled by the dataset or, where audited, by the person.

    Beside them, the audit's disagreement with the dataset on the pairs the
    person decided: the label noise the card reports.
    """
    wanted = set(splits)
    items = _jsonl(LABELS / "R" / "items.jsonl")
    dataset = _jsonl(LABELS / "R" / "labels.jsonl")
    private = _jsonl(LABELS / "R" / "items.private.jsonl")
    audit = {
        frozenset((r["a"], r["b"])): r["same"] for r in _jsonl(LABELS / "R" / "audit.labels.jsonl")
    }
    out, disagree, decided = [], 0, 0
    for item, label, about in zip(items, dataset, private, strict=True):
        pair = frozenset((label["a"], label["b"]))
        person = audit.get(pair)
        if person is not None:
            decided += 1
            disagree += person != label["same"]
        if about["split"] not in wanted:
            continue
        out.append(
            RItem(
                about["id"],
                Mention.model_validate(item["a"]),
                Mention.model_validate(item["b"]),
                label["same"] if person is None else person,
                about["split"],
                person is not None,
            )
        )
    return out, {"noise": share(disagree, decided), "audited": sum(i.audited for i in out)}


@dataclass(frozen=True)
class FItem:
    id: str
    row: FactPair
    same: bool | None
    split: str


def load_f() -> list[FItem]:
    """F's labelled pairs: the person's decided ones, and the unsure ones (no label)."""
    private = {p["fact_id"]: p for p in _jsonl(LABELS / "F" / "items.private.jsonl")}
    out = []
    for name, decided in (("labels.jsonl", True), ("labels.unsure.jsonl", False)):
        for row in _jsonl(LABELS / "F" / name):
            pair = FactPair.model_validate({k: v for k, v in row.items() if k != "same"})
            about = private[pair.id]
            out.append(FItem(about["id"], pair, row["same"] if decided else None, about["split"]))
    return sorted(out, key=lambda item: item.id)


# --------------------------------------------------------------------------- #
# Models, the cache and the meter
# --------------------------------------------------------------------------- #


def roles_for(model: str) -> ModelRoles:
    """The default roles for the default grounder, so its cache keys are `odke run`'s."""
    if model == DEFAULT_GROUND:
        return ModelRoles()
    return ModelRoles(ground=ModelSpec(model=model, max_tokens=256))


class Meter:
    """Wraps a cached client: live calls and their usage, and every answer's usage as paid.

    A cached answer costs nothing now; what it cost when it was asked is in its
    entry, and is what a card's cost per 1,000 items is priced from.
    """

    def __init__(self, cached: CachedClient) -> None:
        self.cached = cached
        self._lock = threading.Lock()
        self.live: dict[str, float] = defaultdict(float)
        self.priced: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        completion = self.cached.complete(messages, spec=spec, schema=schema)
        used = (completion.prompt_tokens, completion.completion_tokens)
        if completion.cached:
            entry = self.cached.store.get(cache_key(messages, spec=spec, schema=schema)) or {}
            stored = entry.get("completion", {})
            used = (
                int(stored.get("prompt_tokens") or 0),
                int(stored.get("completion_tokens") or 0),
            )
        usage = completion.raw.get("usage") if isinstance(completion.raw, dict) else None
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        with self._lock:
            if not completion.cached:
                self.live["calls"] += 1
                self.live["input_tokens"] += completion.prompt_tokens
                self.live["output_tokens"] += completion.completion_tokens
                self.live["reasoning_tokens"] += int(reasoning or 0)
                self.live["usd"] += completion.cost_usd or 0.0
                self.live["unpriced"] += completion.cost_usd is None
            draw = self.priced[spec.repeat]
            draw[0] += used[0]
            draw[1] += used[1]
            draw[2] += 1
        return completion

    def per_1000(self, model: str, draw: int = 0) -> float | None:
        """What 1,000 items cost asked once, from draw `draw`'s usage at list price."""
        tokens_in, tokens_out, calls = self.priced.get(draw, [0, 0, 0])
        if not calls or model not in spend.PRICES:
            return None
        return spend.cost(model, tokens_in, tokens_out) / calls * 1000


def client_for(
    model: str, cache: Path | None, replay: Path | None = None, ledger: Ledger | None = None
) -> Meter:
    """The model's client behind the cache, or, with `replay`, recorded answers (no call).

    Recorded answers are cached apart, under `<cache>/replay/<the file's hash>`,
    never under the live model's key: a later live run is never answered from a
    recording (rule R4, release audit finding 1). The folder is the one `odke`
    caches the same recording in (`openodke.llm.cache.replay_directory`).
    """
    if replay is None:
        inner: LLMClient = roles_for(model).client_for("ground")
        if ledger is not None:
            # The run's budget, in front of the provider and behind the cache: a
            # cached answer costs nothing, and a call past the budget is never made.
            inner = ledger.client(inner)
        return Meter(CachedClient(inner, cache))
    recorded = RecordedClient.from_fixture(replay)
    return Meter(
        CachedClient(recorded, replay_directory(cache, replay) if cache is not None else None)
    )


# --------------------------------------------------------------------------- #
# What a run would send, and what it would cost
# --------------------------------------------------------------------------- #


@dataclass
class Plan:
    """The calls one job would make on one model: how many, how many the cache holds, tokens.

    Input tokens are reckoned as the budget reckons them (characters / 4, plus
    4 a message), without the schema. A call sent with a schema costs more: a
    provider's structured output adds its own framing. That overhead is
    measured on the answers the cache holds, as each answer's real input
    tokens less the reckoning (`overheads`), and added to each uncached call
    sent with one.
    """

    job: str
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    schema_calls: int = 0
    schema_tokens: int = 0
    overheads: list[int] = field(default_factory=list)

    def add(
        self, messages: list[Message], spec: ModelSpec, schema: Any, out: int, store: Any
    ) -> None:
        self.calls += 1
        plain = estimate_tokens(messages)
        key = cache_key(messages, spec=spec, schema=schema)
        entry = store.get(key) if store is not None else None
        if entry is not None:
            self.cached += 1
            stored = entry.get("completion", {}).get("prompt_tokens")
            if stored and schema is not None:
                self.overheads.append(int(stored) - plain)
            return
        self.input_tokens += plain
        self.output_tokens += out
        if schema is not None:
            self.schema_calls += 1
            self.schema_tokens += estimate_tokens(messages, schema) - plain


def plan_grounding(model: str, variant: str, items: Sequence[GItem], runs: int, store: Any) -> Plan:
    prompt, claim_format = VARIANTS[variant]
    chosen = get_prompt(prompt)
    spec = roles_for(model).ground
    found = Plan(f"ground {variant}")
    for draw in range(runs):
        drawn = spec.model_copy(update={"repeat": draw})
        for item in items:
            doc = item.row.as_document()
            evidence = located(item.row.fact, doc)
            if evidence is None or evidence.span is None:
                continue
            messages = build_messages(
                item.row.fact,
                evidence.span.resolve(doc),
                claim_format=claim_format,  # type: ignore[arg-type]
                prompt=chosen,
            )
            found.add(messages, drawn, _schema_for(chosen), OUT_TOKENS[chosen.id], store)
    return found


def plan_pairs(model: str, items: Sequence[RItem], store: Any) -> Plan:
    spec = roles_for(model).ground
    found = Plan("pair judge (R gate, both orders)")
    for item in items:
        for a, b in ((item.a, item.b), (item.b, item.a)):
            found.add(
                pair_module.build_messages(a, b),
                spec,
                pair_module.PAIR_SCHEMA,
                OUT_TOKENS["pair"],
                store,
            )
    return found


def plan_facts(model: str, items: Sequence[FItem], store: Any) -> Plan:
    spec = roles_for(model).ground
    found = Plan("fact-equivalence judge (F labelled, both orders)")
    for item in items:
        row = item.row
        first, second = render_claim(row.first), render_claim(row.second)
        for one, two in ((first, second), (second, first)):
            messages = fact_module.build_messages(
                row.relation, row.description, one, two, row.passage
            )
            found.add(messages, spec, fact_module.EQUIV_SCHEMA, OUT_TOKENS["fact_equiv"], store)
    return found


def priced(
    model: str,
    plans: Sequence[Plan],
    *,
    overhead: float | None = None,
    scale: float = 1.0,
    max_tokens: int = 256,
) -> list[dict[str, Any]]:
    """Each plan's calls and estimated USD, and the worst case.

    Input tokens are the reckoning plus the schema overhead the cache measured
    (`overhead`, a provider's own when this model has no cached answer to
    measure), times `scale`: a model with another tokenizer counts the same text
    as more or fewer tokens, measured with the provider's own counter. Without
    any measured overhead, a schema costs what its characters reckon to. The
    worst case is every uncached reply running to `max_tokens`.
    """
    own = [o for p in plans for o in p.overheads]
    measured = sum(own) / len(own) if own else overhead
    out = []
    for p in plans:
        framing = round(p.schema_calls * measured) if measured is not None else p.schema_tokens
        tokens_in = round((p.input_tokens + framing) * scale)
        asked = p.calls - p.cached
        priceable = model in spend.PRICES
        out.append(
            {
                "job": p.job,
                "calls": p.calls,
                "cached": p.cached,
                "input_tokens": tokens_in,
                "output_tokens": p.output_tokens,
                "usd": spend.cost(model, tokens_in, p.output_tokens) if priceable else None,
                "worst_usd": (
                    spend.cost(model, tokens_in, asked * max_tokens) if priceable else None
                ),
                "schema_overhead": None if measured is None else round(measured, 1),
                "scale": scale,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def grounder_for(model: str, variant: str, client: LLMClient) -> LLMGrounder:
    prompt, claim_format = VARIANTS[variant]
    return LLMGrounder(
        roles_for(model),
        client=client,
        prompt=prompt,
        claim_format=claim_format,  # type: ignore[arg-type]
    )


def labels_of(name: str, version: str, split: str, facts: Sequence[JudgedFact]) -> LabelSet:
    return LabelSet(
        name=name,
        version=version,
        sha256=label_hash((f.id, f.label) for f in facts),
        split=split,
        n=len(facts),
    )


def keep(verdict: str) -> bool:
    """What the gate does with a verdict when refusals drop: keeps all but a refusal."""
    return verdict not in ("contradicted", "not_found")


def passes(fact: JudgedFact, vote: str = "single") -> bool:
    """An item passes when the judge's keep or refuse matches the person's; no answer fails."""
    found = voted(fact.runs, vote)  # type: ignore[arg-type]
    if found == "unchecked":
        return False
    return keep(found) == (fact.label == "supported")


def contest(by_variant: Mapping[str, Sequence[JudgedFact]]) -> dict[str, dict[str, Any]]:
    """Each challenger against P1 on the same items: the comparison, and whether it replaces P1."""
    base = {f.id: f for f in by_variant[P1]}
    out: dict[str, dict[str, Any]] = {}
    for name, facts in by_variant.items():
        if name == P1:
            continue
        rows_a = [ItemRow(id=f.id, correct=passes(base[f.id])) for f in facts]
        rows_b = [ItemRow(id=f.id, correct=passes(f)) for f in facts]
        compared: Comparison = compare_items(
            rows_a, rows_b, metric="accuracy", a_name=P1, b_name=name
        )
        p = compared.primary
        limit = p.detection_limit
        wins = p.verdict == "better" and limit is not None and p.difference > limit
        out[name] = {
            "comparison": compared,
            "replaces_p1": wins,
            "decision": (
                f"{name} replaces {P1}: better by {p.difference * 100:.1f} points, more than "
                f"the {limit * 100:.1f}-point detection limit"
                if wins
                else f"{P1} stays: {p.verdict}"
                + (f" (detection limit {limit * 100:.1f} points)" if limit is not None else "")
            ),
        }
    return out


def adjudication(p1: Sequence[JudgedFact], items: Mapping[str, GItem]) -> dict[str, Any]:
    """On G's labelled not_in_gold facts: "P1 supports it in 2 of 3 runs" against the person."""
    out: dict[str, Any] = {}
    for split in ("gate", "dev", "all"):
        chosen = [
            f
            for f in p1
            if items[f.id].gold_status == "not_in_gold"
            and not items[f.id].planted
            and (split == "all" or items[f.id].split == split)
        ]
        if not chosen:
            continue
        said = [sum(r == "supported" for r in f.runs) >= 2 for f in chosen]
        truth = [f.label == "supported" for f in chosen]
        agree = sum(s == t for s, t in zip(said, truth, strict=True))
        listed = [t for s, t in zip(said, truth, strict=True) if s]
        out[split] = {
            "n": len(chosen),
            "runs": len(chosen[0].runs),
            "agreement": share(agree, len(chosen)).model_dump(),
            "precision": share(sum(listed), len(listed)).model_dump(),
            "recall": share(
                sum(1 for s, t in zip(said, truth, strict=True) if s and t), sum(truth)
            ).model_dump(),
        }
    return out


def self_preference(
    claude: Sequence[JudgedFact], other: Sequence[JudgedFact], names: tuple[str, str]
) -> dict[str, Any]:
    """P1's fail-recall by a Claude judge and by another, on the facts the person did not keep."""
    theirs = {f.id: f for f in other}
    pairs = [(f, theirs[f.id]) for f in claude if f.id in theirs and f.label != "supported"]
    refused_a = [not keep(a.runs[0]) for a, _ in pairs]
    refused_b = [not keep(b.runs[0]) for _, b in pairs]
    rows_a = [ItemRow(id=a.id, correct=r) for (a, _), r in zip(pairs, refused_a, strict=True)]
    rows_b = [ItemRow(id=a.id, correct=r) for (a, _), r in zip(pairs, refused_b, strict=True)]
    found: dict[str, Any] = {
        "n": len(pairs),
        "fail_recall": {
            names[0]: share(sum(refused_a), len(pairs)).model_dump(),
            names[1]: share(sum(refused_b), len(pairs)).model_dump(),
        },
    }
    if pairs:
        exact = mcnemar(refused_a, refused_b)
        found["mcnemar"] = exact.model_dump()
        found["comparison"] = compare_items(
            rows_a, rows_b, metric="accuracy", a_name=names[0], b_name=names[1]
        ).model_dump(mode="json")
    return found


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run_model(
    model: str,
    args: argparse.Namespace,
    out: Path,
    g: list[GItem],
    r: list[RItem],
    noise: Mapping[str, Any],
    f: list[FItem],
) -> dict[str, Any]:
    """Every asked judge on one model: cards, items, and the results the table reads.

    What each job spent goes in the ledger even when a job stops partway (a
    budget stop, a provider error). A job with a failed or unasked call makes
    no card (rule R1): its numbers would score answers that were never given.
    """
    result: dict[str, Any] = {"model": model, "grounding": {}, "cards": [], "failed": []}
    spent_rows: list[tuple[str, Meter]] = []
    try:
        _judge_all(model, args, out, g, r, noise, f, result, spent_rows)
    finally:
        result["spent"] = _record(model, args, spent_rows)
    return result


def _judge_all(
    model: str,
    args: argparse.Namespace,
    out: Path,
    g: list[GItem],
    r: list[RItem],
    noise: Mapping[str, Any],
    f: list[FItem],
    result: dict[str, Any],
    spent_rows: list[tuple[str, Meter]],
) -> None:
    slug = Card.model_construct(prompt="", model=model).id.split("__")[1]  # type: ignore[call-arg]
    by_id = {item.id: item for item in g}
    ledger = args.ledgers.get(spend.provider(model))
    asked = judges_for(args, model)
    if "ground" in asked:
        per_variant: dict[str, list[JudgedFact]] = {}
        meters = {}
        for variant in variants_for(args, model):
            meter = client_for(model, args.cache, args.replay, ledger)
            meters[variant] = meter
            spent_rows.append((f"ground {variant}", meter))
            grounder = grounder_for(model, variant, meter)
            judged = ground_labelled(
                [item.row for item in g],
                grounder,
                runs=args.runs,
                planted=[item.planted for item in g],
                ids=[item.id for item in g],
            )
            _write_items(out / "items" / slug / f"ground-{variant}.jsonl", judged.facts, by_id)
            failed = sum(int(each.stats.get("failed", 0)) for each in judged.grounders)
            if failed:
                result["failed"].append(f"ground {variant}: {failed} failed call(s), no card")
                continue
            per_variant[variant] = judged.facts
        for variant, facts in per_variant.items():
            prompt, claim_format = VARIANTS[variant]
            grounder = grounder_for(model, variant, meters[variant])
            planted = [x for x in facts if by_id[x.id].planted]
            summary: dict[str, Any] = {}
            for split in ("gate", "dev"):
                scored = [x for x in facts if by_id[x.id].split == split]
                if not scored:
                    continue
                made = Made(
                    prompt=prompt,
                    prompts=prompt_hashes(grounder.prompt_keys),
                    model=model,
                    claim_format=grounder.claim_line,
                    labels=labels_of("G", G_VERSION, split, scored),
                    cost_per_1000=meters[variant].per_1000(model),
                )
                card = grounding_card(scored, made, planted=planted)
                folder = "cards" if split == "gate" else "cards-dev"
                card.write(out / folder)
                summary[split] = card
                grounding = card.grounding
                if split == "gate" and grounding is not None and grounding.flips is not None:
                    flipped = grounding.flips.value or 0.0
                    if flipped > FLIP_LIMIT and args.runs == 3:
                        voted_card = grounding_card(
                            scored, made, planted=planted, vote="refuse_2_of_3"
                        )
                        voted_card.write(out / folder)
                        summary["gate-2of3"] = voted_card
            result["grounding"][variant] = summary
        gate = {
            v: [x for x in facts if by_id[x.id].split == "gate"] for v, facts in per_variant.items()
        }
        dev = {
            v: [x for x in facts if by_id[x.id].split == "dev"] for v, facts in per_variant.items()
        }
        result["by_variant"] = per_variant
        if P1 in per_variant:
            result["contest"] = contest(gate)
            result["contest_dev"] = contest(dev)
            if args.runs >= 2:
                result["adjudication"] = adjudication(per_variant[P1], by_id)
            result["p1"] = per_variant[P1]
    if "pair" in asked and r:
        meter = client_for(model, args.cache, args.replay, ledger)
        spent_rows.append(("pair judge", meter))
        judge = PairJudge(roles_for(model), client=meter, on_decision="link")
        decisions = judge.judge_many([(i.a, i.b, None) for i in r])
        judged_pairs = [
            JudgedPair(
                id=i.id,
                same=i.same,
                forward=d.forward.decision if d.forward else None,
                backward=d.backward.decision if d.backward else None,
            )
            for i, d in zip(r, decisions, strict=True)
        ]
        _write_pairs(out / "items" / slug / "pair.jsonl", judged_pairs)
        broken = int(judge.stats.get("failed", 0)) + int(judge.stats.get("unasked", 0))
        if broken:
            result["failed"].append(f"pair judge: {broken} failed or unasked call(s), no card")
        else:
            made = Made(
                prompt=pair_module.PROMPT.key,
                prompts=prompt_hashes([pair_module.PROMPT.key, pair_module.FRAME.key]),
                model=model,
                labels=LabelSet(
                    name="R",
                    version=R_VERSION,
                    sha256=label_hash((i.id, i.same) for i in r),
                    split="gate",
                    n=len(r),
                ),
                cost_per_1000=meter.per_1000(model),
            )
            card = pair_card(
                judged_pairs, made, audited=int(noise["audited"]), label_noise=noise["noise"]
            )
            card.write(out / "cards")
            result["pair"] = card
    if "fact" in asked and f:
        meter = client_for(model, args.cache, args.replay, ledger)
        spent_rows.append(("fact-equivalence judge", meter))
        fact_judge = FactJudge(roles_for(model), client=meter)
        decided = fact_judge.judge_rows([item.row for item in f])
        judged_facts = [
            JudgedPair(
                id=item.id,
                same=bool(item.same),
                forward=e.gold_first.decision if e.gold_first else None,
                backward=e.prediction_first.decision if e.prediction_first else None,
            )
            for item, e in zip(f, decided, strict=True)
        ]
        _write_pairs(out / "items" / slug / "fact.jsonl", judged_facts)
        broken = int(fact_judge.stats.get("failed", 0)) + int(fact_judge.stats.get("unasked", 0))
        if broken:
            result["failed"].append(
                f"fact-equivalence judge: {broken} failed or unasked call(s), no card"
            )
            return
        labelled = {item.id: item for item in f}
        result["fact"] = {}
        for split in ("gate", "dev"):
            scored = [
                j
                for j in judged_facts
                if labelled[j.id].same is not None and labelled[j.id].split == split
            ]
            if not scored:
                continue
            unsure = [
                j
                for j in judged_facts
                if labelled[j.id].same is None and labelled[j.id].split == split
            ]
            made = Made(
                prompt=fact_module.PROMPT.key,
                prompts=prompt_hashes([fact_module.PROMPT.key, fact_module.FRAME.key]),
                model=model,
                labels=LabelSet(
                    name="F",
                    version=F_VERSION,
                    sha256=label_hash((j.id, j.same) for j in scored),
                    split=split,
                    n=len(scored),
                ),
                cost_per_1000=meter.per_1000(model),
                notes=(
                    f"{len(unsure)} pair(s) the person left unsure were asked and are not scored",
                ),
            )
            card = equivalence_card(scored, made)
            card.write(out / ("cards" if split == "gate" else "cards-dev"))
            result["fact"][split] = card


def _record(
    model: str, args: argparse.Namespace, spent_rows: Sequence[tuple[str, Meter]]
) -> list[dict[str, Any]]:
    """One ledger row per job that made a live call: the model and what it cost."""
    rows = []
    for job, meter in spent_rows:
        live = meter.live
        if not live.get("calls") or args.replay is not None:
            continue
        usd = (
            live["usd"]
            if not live.get("unpriced")
            else spend.cost(model, int(live["input_tokens"]), int(live["output_tokens"]))
        )
        reasoning = int(live.get("reasoning_tokens", 0))
        note = f", {reasoning} of them reasoning" if reasoning else ""
        rows.append(
            spend.record(
                f"calibration #134/#135: {job}{note}{f' ({args.who})' if args.who else ''}",
                model,
                calls=int(live["calls"]),
                input_tokens=int(live["input_tokens"]),
                output_tokens=int(live["output_tokens"]),
                usd=usd,
                ledger=args.ledger,
            )
        )
    return rows


def _write_items(path: Path, facts: Sequence[JudgedFact], items: Mapping[str, GItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for x in facts:
            about = items[x.id]
            row = {
                "id": x.id,
                "label": x.label,
                "runs": list(x.runs),
                "words": x.words,
                "split": about.split,
                "planted": about.planted,
                "gold_status": about.gold_status,
            }
            fh.write(json.dumps(row) + "\n")


def _write_pairs(path: Path, pairs: Sequence[JudgedPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(
                json.dumps(
                    {"id": p.id, "same": p.same, "forward": p.forward, "backward": p.backward}
                )
                + "\n"
            )


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def table(results: Sequence[Mapping[str, Any]], extra: Mapping[str, Any]) -> str:
    lines = ["# Calibration results", ""]
    for found in results:
        model = found["model"]
        lines += [f"## {model}", ""]
        if found["grounding"]:
            lines += [
                "### Grounding, gate split (G)",
                "",
                "| variant | n | κ keep/refuse | κ 3-way | fail-recall | false refusals "
                "| planted caught | abstain | flips | bar |",
                "|---|---:|---|---|---|---|---|---|---|---|",
            ]
            for variant, cards in found["grounding"].items():
                for key in ("gate", "gate-2of3"):
                    card = cards.get(key)
                    if card is None or card.grounding is None:
                        continue
                    g = card.grounding
                    name = variant + (" (refuse 2 of 3)" if key == "gate-2of3" else "")
                    lines.append(
                        f"| {name} | {card.labels.n} | {g.kappa.text()} | "
                        f"{g.kappa_three_way.text()} | "
                        f"{g.fail_recall.text()} | {g.false_refusals.text()} | "
                        f"{g.planted.text() if g.planted else '-'} | {g.abstain.text()} | "
                        f"{g.flips.text() if g.flips else '-'} | "
                        f"{'met' if card.bar.met else 'not met'} |"
                    )
            lines += ["", "Dev split, reported apart (never picks a winner):", ""]
            for variant, cards in found["grounding"].items():
                card = cards.get("dev")
                if card is not None and card.grounding is not None:
                    lines.append(f"- {variant}: n {card.labels.n}, κ {card.grounding.kappa.text()}")
            lines.append("")
        if "contest" in found:
            lines += ["### The contest: each challenger against P1, gate split, run 1", ""]
            for verdict in found["contest"].values():
                lines += [
                    "```text",
                    verdict["comparison"].render(),
                    "```",
                    f"**{verdict['decision']}**",
                    "",
                ]
        if "adjudication" in found:
            lines += [
                "### Gold adjudication: P1 supports in 2 of 3 runs, on G's labelled "
                "not_in_gold facts",
                "",
            ]
            for split, a in found["adjudication"].items():
                agree, prec = a["agreement"], a["precision"]
                lines.append(
                    f"- {split}: n {a['n']}, agrees with the person {agree['k']}/{agree['n']} "
                    f"({_pct(agree['value'])}); listed facts the person supports "
                    f"{prec['k']}/{prec['n']}"
                )
            lines.append("")
        if "pair" in found:
            lines += ["### Pair judge (R gate)", "", f"- {found['pair'].line()}", ""]
        for split, card in (found.get("fact") or {}).items():
            lines += [f"### Fact-equivalence judge (F {split})", "", f"- {card.line()}", ""]
        for failed in found.get("failed", ()):
            lines.append(f"- **failed, no card:** {failed}")
        for row in found.get("spent", ()):
            lines.append(f"- spent: {row['who']}: {row['calls']} calls, ${row['usd']}")
        lines.append("")
    if extra.get("head_to_head"):
        lines += ["## Model against model, same prompt, gate split, run 1", ""]
        for name, compared in extra["head_to_head"].items():
            lines += [f"### {name}", "", "```text", compared.render(), "```", ""]
    if extra.get("stopped"):
        lines += [f"**Stopped:** {extra['stopped']}", ""]
    if extra.get("self_preference"):
        sp = extra["self_preference"]
        lines += ["## Self-preference: P1's fail-recall on the same facts", ""]
        for name, value in sp["fail_recall"].items():
            lines.append(f"- {name}: {value['k']}/{value['n']} ({_pct(value['value'])})")
        if "mcnemar" in sp:
            lines.append(f"- McNemar exact p = {sp['mcnemar']['p_value']:.3g}")
        lines.append("")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Card):
        return value.model_dump(mode="json")
    if isinstance(value, Comparison):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping) and "by_variant" in value:
        value = {k: v for k, v in value.items() if k != "by_variant"}
    if isinstance(value, JudgedFact):
        return {"id": value.id, "label": value.label, "runs": list(value.runs)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# The published tables: generated from the shipped cards, never typed
# --------------------------------------------------------------------------- #

TABLES_BEGIN = "<!-- judge calibration: generated by `bench/calibrate.py --tables`; do not edit -->"
TABLES_END = "<!-- judge calibration: end -->"
# The aggregates the cards do not hold (the contest, adjudication, the
# head-to-head, self-preference), written by a run with `--aggregates`.
AGGREGATES = Path(__file__).parent / "calibration" / "results.json"
CARDS = Path(__file__).parent.parent / "src" / "openodke" / "calibration" / "cards"


def _comparison(compared: Comparison) -> dict[str, Any]:
    p = compared.primary
    return {
        "a": compared.a,
        "b": compared.b,
        "items": p.items,
        "a_accuracy": p.a,
        "b_accuracy": p.b,
        "difference": p.difference,
        "interval": list(p.interval),
        "detection_limit": p.detection_limit,
        "verdict": p.verdict,
        "mcnemar_p": compared.mcnemar.p_value if compared.mcnemar is not None else None,
        "flips": p.flips,
    }


def aggregates(results: Sequence[Mapping[str, Any]], extra: Mapping[str, Any]) -> dict[str, Any]:
    """What the tables read beside the cards, with no per-item answer in it."""
    out: dict[str, Any] = {
        "date": _dt.date.today().isoformat(),
        "models": [x["model"] for x in results],
        "contest": {},
        "adjudication": {},
        "failed": {x["model"]: list(x.get("failed", ())) for x in results if x.get("failed")},
        "head_to_head": {
            name: _comparison(compared) for name, compared in extra.get("head_to_head", {}).items()
        },
    }
    for x in results:
        if "contest" in x:
            out["contest"][x["model"]] = {
                name: {**_comparison(v["comparison"]), "replaces_p1": v["replaces_p1"],
                       "decision": v["decision"]}
                for name, v in x["contest"].items()
            }  # fmt: skip
        if "adjudication" in x:
            out["adjudication"][x["model"]] = x["adjudication"]
    if "self_preference" in extra:
        sp = dict(extra["self_preference"])
        sp.pop("comparison", None)
        out["self_preference"] = sp
    return out


def _cell(share: Any) -> str:
    return "–" if share is None or share.value is None else f"{share.value:.2f}"


def _ci(estimate: Any) -> str:
    if estimate is None or estimate.value is None:
        return "–"
    if estimate.low is None or estimate.high is None:
        return f"{estimate.value:.2f}"
    return f"{estimate.value:.2f} [{estimate.low:.2f}, {estimate.high:.2f}]"


def _money(value: float | None) -> str:
    return "–" if value is None else f"${value:.2f}"


def _short(model: str) -> str:
    return f"`{model}`"


def render_tables(cards: Sequence[Card], found: Mapping[str, Any]) -> str:
    """The Judge calibration section: one table per judge, then the contest."""
    models = list(found.get("models", ()))
    order = {m: i for i, m in enumerate(models)}

    def pick(judge: str, prompt: str, claim_format: str | None) -> list[Card]:
        chosen = [
            c
            for c in cards
            if c.judge == judge
            and c.prompt == prompt
            and c.vote == "single"
            and c.labels.split == "gate"
            and (claim_format is None or c.claim_format == claim_format)
        ]
        return sorted(chosen, key=lambda c: order.get(c.model, len(order)))

    date = found.get("date", "")
    lines = [
        TABLES_BEGIN,
        "",
        "## Judge calibration",
        "",
        f"Each model judge, measured on {date} against the maintainer's hand labels: label sets "
        "G (grounding), R (entity pairs) and F (fact pairs), public data only "
        "(`bench/labels/`). How a card is made and read, and the bars: "
        "[Calibration cards](calibration.md). These tables are generated from the cards the "
        "package ships (`src/openodke/calibration/cards/`) and `bench/calibration/results.json` "
        "by `bench/calibrate.py --tables`, and a test holds them to both.",
        "",
        "### The grounder, `ground.span@1`, claim inline",
        "",
        "Label set G's gate split: the maintainer's hand labels on facts three extractors "
        "wrote, and 100 planted false facts for the catch rate. Keep is `supported`; refuse is "
        "`contradicted` or `not_found`. Three runs of each question.",
        "",
        "| Model | n | κ keep vs refuse [95%] | Meets the bar (κ ≥ 0.6) | Flips over 3 runs "
        "| Planted caught | False refusals | Adjudication agrees | κ, ≤ 40 words / > 150 words "
        "| $ per 1,000 facts |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for c in pick("ground", "ground.span@1", "inline"):
        g = c.grounding
        assert g is not None
        bands = {b.name: b for b in g.by_length}
        short = bands.get("sentence")
        long = bands.get("document")
        split = f"{_cell(short.kappa) if short else '–'} / {_cell(long.kappa) if long else '–'}"
        adjudicated = found.get("adjudication", {}).get(c.model, {}).get("gate")
        agrees = (
            f"{adjudicated['agreement']['value']:.2f} ({adjudicated['agreement']['k']}/"
            f"{adjudicated['agreement']['n']})"
            if adjudicated and adjudicated["agreement"]["value"] is not None
            else "–"
        )
        lines.append(
            f"| {_short(c.model)} | {c.labels.n} | {_ci(g.kappa)} | "
            f"{'yes' if c.bar.met else 'no'} | {_cell(g.flips)} | {_cell(g.planted)} | "
            f"{_cell(g.false_refusals)} | {agrees} | {split} | {_money(c.cost_per_1000)} |"
        )
    lines += [
        "",
        "Not measured: the maintainer's agreement with themselves, the bar's other half "
        "(within 0.1 of it), so each bar here is κ ≥ 0.6 alone. Adjudication agrees: on G's "
        'gate facts the dataset\'s gold lacks, how often "supported in 2 of 3 runs" matches '
        'the maintainer\'s "supported". Cost is measured token use at list price.',
        "",
        "### The pair judge, `pair@1`, both orders",
        "",
        "Label set R's gate split: Re-DocRED's own mention clusters, with the maintainer's "
        '100-pair audit replacing a dataset label where it decided one. "Same" counts only '
        "when both orders say same.",
        "",
        '| Model | n | Precision of "same" [95%] | Meets the bar (lower bound ≥ 0.9) '
        '| Recall of "same" | Unsure | Orders disagree | $ per 1,000 pairs |',
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for c in pick("pair", "pair@1", None):
        pr = c.pairs
        assert pr is not None
        lines.append(
            f"| {_short(c.model)} | {c.labels.n} | {_ci(pr.precision_same)} | "
            f"{'yes' if c.bar.met else 'no'} | {_cell(pr.recall_same)} | {_cell(pr.unsure)} | "
            f"{_cell(pr.order_disagreement)} | {_money(c.cost_per_1000)} |"
        )
    noise = next((c.pairs.label_noise for c in pick("pair", "pair@1", None) if c.pairs), None)
    lines += [
        "",
        "Not measured: the maintainer's agreement with themselves. "
        + (
            f"The audit disagreed with the dataset's label on {noise.k} of {noise.n} pairs it "
            f"decided ({noise.value:.2f}): the labels the judge is scored on are noisy."
            if noise is not None and noise.value is not None
            else ""
        ),
        "",
        "### The fact-equivalence judge, `fact_equiv@1`, both orders",
        "",
        "Label set F's gate split: the pairs the maintainer decided same or different (the "
        'unsure ones are asked and not scored). "Same" counts only when both orders say same.',
        "",
        "| Model | n | κ same vs not [95%] | Meets the bar (κ ≥ 0.6) | Unsure "
        '| Orders disagree | Judge "same" / person "same" | $ per 1,000 pairs |',
        "|---|---:|---|---|---:|---:|---|---:|",
    ]
    for c in pick("fact_equiv", "fact_equiv@1", None):
        e = c.equivalence
        assert e is not None
        lines.append(
            f"| {_short(c.model)} | {c.labels.n} | {_ci(e.kappa)} | "
            f"{'yes' if c.bar.met else 'no'} | {_cell(e.unsure)} | "
            f"{_cell(e.order_disagreement)} | {_cell(e.judge_same)} / {_cell(e.person_same)} | "
            f"{_money(c.cost_per_1000)} |"
        )
    lines += [
        "",
        "Not measured: the maintainer's agreement with themselves.",
        "",
        "### The contest, and model against model",
        "",
        "On G's gate split, run 1: an item passes when the judge's keep or refuse matches the "
        "maintainer's, compared item by item with the Evaluator's own comparison "
        "([Was the change real?](evaluation.md#was-the-change-real)). A challenger replaces "
        "`ground.span@1` only when it is better by more than the detection limit.",
        "",
        "| Model | Against | Accuracy | Difference [95%] | Detection limit | McNemar p | Verdict |",
        "|---|---|---|---|---:|---:|---|",
    ]
    names = {
        "span-labelled": "`ground.span@1` labelled",
        "onyx": "`ground.onyx@1`",
        "statement": "`ground.statement@1`",
    }
    for model in models:
        for variant, v in found.get("contest", {}).get(model, {}).items():
            lines.append(_contest_row(_short(model), f"{names.get(variant, variant)} vs inline", v))
    for name, v in found.get("head_to_head", {}).items():
        if not name.endswith(": span-inline"):
            continue
        lines.append(_contest_row(f"`{v['b']}`", f"`{v['a']}`, `ground.span@1` inline", v))
    lines += ["", TABLES_END]
    return "\n".join(lines)


def _contest_row(model: str, against: str, v: Mapping[str, Any]) -> str:
    low, high = v["interval"]
    limit = "–" if v["detection_limit"] is None else f"{v['detection_limit'] * 100:.1f} pts"
    p = "–" if v["mcnemar_p"] is None else f"{v['mcnemar_p']:.3g}"
    return (
        f"| {model} | {against} | {v['a_accuracy']:.3f} → {v['b_accuracy']:.3f} | "
        f"{v['difference'] * 100:+.1f} pts [{low * 100:+.1f}, {high * 100:+.1f}] | {limit} | "
        f"{p} | {v['verdict']} |"
    )


def shipped_cards(directory: Path = CARDS) -> list[Card]:
    from openodke.calibration import load_card

    return [load_card(path) for path in sorted(directory.glob("*.json"))]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def estimate(
    args: argparse.Namespace, g: list[GItem], r: list[RItem], f: list[FItem]
) -> dict[str, list[dict[str, Any]]]:
    """Per model, each job's calls, the share the cache holds, tokens and USD. No call is made."""
    store = CachedClient(_Never(), args.cache).store if args.cache is not None else None
    planned: dict[str, list[Plan]] = {}
    for model in args.model:
        plans: list[Plan] = []
        asked = judges_for(args, model)
        if "ground" in asked:
            plans += [
                plan_grounding(model, v, g, args.runs, store) for v in variants_for(args, model)
            ]
        if "pair" in asked:
            plans.append(plan_pairs(model, r, store))
        if "fact" in asked:
            plans.append(plan_facts(model, f, store))
        planned[model] = plans
    # A provider's schema overhead, from any of its models' cached answers.
    by_provider: dict[str, list[int]] = defaultdict(list)
    for model, plans in planned.items():
        by_provider[spend.provider(model)] += [o for p in plans for o in p.overheads]
    if store is not None:
        for provider, model in getattr(args, "overhead_from", {}).items():
            if not by_provider.get(provider):
                plans = [plan_grounding(model, P1, g, 1, store)] if g else []
                by_provider[provider] += [o for p in plans for o in p.overheads]
    return {
        model: priced(
            model,
            plans,
            overhead=(
                sum(by_provider[spend.provider(model)]) / len(by_provider[spend.provider(model)])
                if by_provider[spend.provider(model)]
                else None
            ),
            scale=getattr(args, "scale", {}).get(model, 1.0),
            max_tokens=roles_for(model).ground.max_tokens,
        )
        for model, plans in planned.items()
    }


def variants_for(args: argparse.Namespace, model: str) -> list[str]:
    """The grounder variants `model` runs: `--only`'s for it, else every one asked."""
    return list(getattr(args, "only", {}).get(model, args.variants))


def judges_for(args: argparse.Namespace, model: str) -> set[str]:
    """The judges `model` runs: `--judges-for`'s for it, else every one asked."""
    return set(getattr(args, "judges_for", {}).get(model, args.judges))


class _Never:
    """A client for an estimate: it is never called."""

    def complete(self, messages: Any, *, spec: Any, schema: Any = None) -> Completion:
        raise RuntimeError("an estimate makes no call")


def _pairs(given: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for item in given:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"{what} takes KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", help="a model to calibrate; repeat for more")
    parser.add_argument("--cache", type=Path, help="the response cache's directory")
    parser.add_argument("--out", "-o", type=Path, default=Path("calibration-out"))
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--judges", default="ground,pair,fact", help="ground, pair, fact")
    parser.add_argument("--variants", default=",".join(VARIANTS), help=", ".join(VARIANTS))
    parser.add_argument(
        "--only", action="append", default=[], help="MODEL=v1,v2: the variants that model runs"
    )
    parser.add_argument(
        "--judges-for", action="append", default=[], help="MODEL=j1,j2: the judges that model runs"
    )
    parser.add_argument(
        "--scale",
        action="append",
        default=[],
        help="MODEL=FACTOR: its input tokens against the reckoning, measured with the "
        "provider's token counter (a newer tokenizer counts more)",
    )
    parser.add_argument("--every", action="store_true", help="ground G's unlabelled facts too")
    parser.add_argument("--estimate", action="store_true", help="print the estimate and stop")
    parser.add_argument(
        "--cap", action="append", default=[], help="PROVIDER=USD, a cap for this run"
    )
    parser.add_argument("--ledger", type=Path, default=spend.LEDGER)
    parser.add_argument("--who", default="", help="a note for the ledger rows")
    parser.add_argument(
        "--replay", type=Path, help="recorded answers (a RecordedClient fixture): no call, no spend"
    )
    parser.add_argument(
        "--aggregates",
        type=Path,
        help="also write the tables' aggregates here (no per-item answer)",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="print the Judge calibration tables from the shipped cards and aggregates",
    )
    args = parser.parse_args(argv)
    if args.tables:
        found = json.loads(AGGREGATES.read_text(encoding="utf-8"))
        print(render_tables(shipped_cards(), found))
        return 0
    args.model = args.model or [DEFAULT_GROUND]
    args.judges = {j.strip() for j in args.judges.split(",") if j.strip()}
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.only = {
        m: [v.strip() for v in vs.split(",") if v.strip()]
        for m, vs in _pairs(args.only, "--only").items()
    }
    args.scale = {m: float(v) for m, v in _pairs(args.scale, "--scale").items()}
    args.judges_for = {
        m: {j.strip() for j in js.split(",") if j.strip()}
        for m, js in _pairs(args.judges_for, "--judges-for").items()
    }
    args.overhead_from = {"anthropic": DEFAULT_GROUND}
    asked = [*args.variants, *(v for vs in args.only.values() for v in vs)]
    unknown = [v for v in asked if v not in VARIANTS]
    if unknown or not args.judges <= {"ground", "pair", "fact"}:
        parser.error(f"unknown variant or judge: {unknown or sorted(args.judges)}")
    caps = {name: float(usd) for name, usd in _pairs(args.cap, "--cap").items()}

    g = load_g(every=args.every) if "ground" in args.judges else []
    r, noise = load_r() if "pair" in args.judges else ([], {"noise": None, "audited": 0})
    f = load_f() if "fact" in args.judges else []
    estimated = estimate(args, g, r, f)
    by_provider: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    for model, jobs in estimated.items():
        print(
            f"\n{model}: {len(g)} G items, {len(r)} R pairs, {len(f)} F pairs, {args.runs} run(s)"
            + (f", input scaled ×{args.scale[model]}" if model in args.scale else "")
        )
        for job in jobs:
            usd = "unpriced" if job["usd"] is None else f"${job['usd']:.3f}"
            worst = "" if job["worst_usd"] is None else f" (worst ${job['worst_usd']:.3f})"
            print(
                f"  {job['job']:<52} {job['calls']:>5} calls, {job['cached']:>5} cached, "
                f"{job['input_tokens']:>8} in, {job['output_tokens']:>6} out  {usd}{worst}"
            )
            by_provider[spend.provider(model)] += job["usd"] or 0.0
            calls[spend.provider(model)] += job["calls"] - job["cached"]
        total = sum(j["usd"] or 0.0 for j in jobs)
        worst_total = sum(j["worst_usd"] or 0.0 for j in jobs)
        print(f"  {'total':<52} ${total:.3f} (worst ${worst_total:.3f})")
    if args.estimate:
        return 0
    args.ledgers = {}
    for name, usd in by_provider.items() if args.replay is None else ():
        model = next(m for m in args.model if spend.provider(m) == name)
        try:
            print(spend.check(model, usd, cap=caps.get(name), ledger=args.ledger))
        except spend.SpendRefused as refused:
            print(f"refused: {refused}", file=sys.stderr)
            return 3
        # The cap holds during the run too: a call that would pass it is never made.
        limit = caps.get(name, spend.CAPS.get(name))
        left = (limit or 0.0) - spend.spent(args.ledger).get(name, 0.0)
        args.ledgers[name] = Ledger(Budget(usd=max(left, 0.0), calls=2 * calls[name] + 100))
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    stopped: str | None = None
    for model in args.model:
        try:
            results.append(run_model(model, args, args.out, g, r, noise, f))
        except BudgetExceeded as exc:
            stopped = f"{model}: the budget stopped the run, no card from it: {exc}"
            print(f"stopped: {stopped}", file=sys.stderr)
            break
    extra: dict[str, Any] = {"head_to_head": head_to_head(results, g)}
    claude = [x for x in results if spend.provider(x["model"]) == "anthropic" and "p1" in x]
    others = [x for x in results if spend.provider(x["model"]) != "anthropic" and "p1" in x]
    if claude and others:
        gate = {i.id for i in g if i.split == "gate"}
        a = [x for x in claude[0]["p1"] if x.id in gate]
        b = [x for x in others[0]["p1"] if x.id in gate]
        extra["self_preference"] = self_preference(a, b, (claude[0]["model"], others[0]["model"]))
    if stopped:
        extra["stopped"] = stopped
    (args.out / "results.md").write_text(table(results, extra) + "\n", encoding="utf-8")
    (args.out / "results.json").write_text(
        json.dumps(
            {"results": _jsonable(results), **_jsonable(extra), "estimate": estimated}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    if args.aggregates is not None:
        args.aggregates.parent.mkdir(parents=True, exist_ok=True)
        args.aggregates.write_text(
            json.dumps(aggregates(results, extra), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"\nwrote {args.out / 'results.md'}")
    return 3 if stopped else 0


def head_to_head(results: Sequence[Mapping[str, Any]], g: Sequence[GItem]) -> dict[str, Any]:
    """Each later model against the first, prompt by prompt, on the same gate items, paired.

    The Evaluator's own comparison (`openodke.eval.compare`), as the contest:
    an item passes when the judge's keep or refuse, run 1, matches the person's.
    """
    gate = {i.id for i in g if i.split == "gate"}
    with_items = [x for x in results if x.get("by_variant")]
    if len(with_items) < 2:
        return {}
    first = with_items[0]
    out: dict[str, Any] = {}
    for other in with_items[1:]:
        for variant, facts in other["by_variant"].items():
            base = first["by_variant"].get(variant)
            if base is None:
                continue
            a = {x.id: x for x in base if x.id in gate}
            rows_a = [ItemRow(id=i, correct=passes(a[i])) for i in sorted(a)]
            b = {x.id: x for x in facts if x.id in a}
            rows_b = [ItemRow(id=i, correct=passes(b[i])) for i in sorted(a)]
            compared = compare_items(
                rows_a, rows_b, metric="accuracy", a_name=first["model"], b_name=other["model"]
            )
            out[f"{other['model']} vs {first['model']}: {variant}"] = compared
    return out


if __name__ == "__main__":
    raise SystemExit(main())
