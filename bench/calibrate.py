"""The grounder contest and the calibration cards: label sets G, R and F (#134, #135).

    uv run python bench/calibrate.py --estimate --cache DIR   # what a run would cost; no call
    uv run python bench/calibrate.py --cache DIR -o OUT       # the default Haiku, every judge
    uv run python bench/calibrate.py --model openai/gpt-4.1-mini-2025-04-14 --cap openai=5 ...
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
import hashlib
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
from openodke.llm.budget import estimate_tokens
from openodke.llm.cache import cache_key
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
        with self._lock:
            if not completion.cached:
                self.live["calls"] += 1
                self.live["input_tokens"] += completion.prompt_tokens
                self.live["output_tokens"] += completion.completion_tokens
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


def client_for(model: str, cache: Path | None, replay: Path | None = None) -> Meter:
    """The model's client behind the cache, or, with `replay`, recorded answers (no call).

    Recorded answers are cached apart, under `<cache>/replay/<the file's hash>`,
    never under the live model's key: a later live run is never answered from a
    recording (rule R4, release audit finding 1).
    """
    if replay is None:
        return Meter(CachedClient(roles_for(model).client_for("ground"), cache))
    recorded = RecordedClient.from_fixture(replay)
    digest = hashlib.sha256(replay.read_bytes()).hexdigest()[:16]
    return Meter(CachedClient(recorded, cache / "replay" / digest if cache is not None else None))


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


def priced(model: str, plans: Sequence[Plan]) -> list[dict[str, Any]]:
    """Each plan's calls and estimated USD, with the schema overhead the cache measured.

    Without a cached answer to measure it on, a schema costs what its
    characters reckon to.
    """
    overheads = [o for p in plans for o in p.overheads]
    overhead = sum(overheads) / len(overheads) if overheads else None
    out = []
    for p in plans:
        framing = round(p.schema_calls * overhead) if overhead is not None else p.schema_tokens
        tokens_in = p.input_tokens + framing
        usd = spend.cost(model, tokens_in, p.output_tokens) if model in spend.PRICES else None
        out.append(
            {
                "job": p.job,
                "calls": p.calls,
                "cached": p.cached,
                "input_tokens": tokens_in,
                "output_tokens": p.output_tokens,
                "usd": usd,
                "schema_overhead": None if overhead is None else round(overhead, 1),
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
    """Every asked judge on one model: cards, items, and the results the table reads."""
    slug = Card.model_construct(prompt="", model=model).id.split("__")[1]  # type: ignore[call-arg]
    result: dict[str, Any] = {"model": model, "grounding": {}, "cards": []}
    by_id = {item.id: item for item in g}
    spent_rows = []
    if "ground" in args.judges:
        per_variant: dict[str, list[JudgedFact]] = {}
        meters = {}
        for variant in args.variants:
            meter = client_for(model, args.cache, args.replay)
            meters[variant] = meter
            grounder = grounder_for(model, variant, meter)
            judged = ground_labelled(
                [item.row for item in g],
                grounder,
                runs=args.runs,
                planted=[item.planted for item in g],
                ids=[item.id for item in g],
            )
            per_variant[variant] = judged.facts
            _write_items(out / "items" / slug / f"ground-{variant}.jsonl", judged.facts, by_id)
            spent_rows.append((f"ground {variant}", meter))
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
        if P1 in per_variant:
            result["contest"] = contest(gate)
            result["contest_dev"] = contest(dev)
            if args.runs >= 2:
                result["adjudication"] = adjudication(per_variant[P1], by_id)
            result["p1"] = per_variant[P1]
    if "pair" in args.judges and r:
        meter = client_for(model, args.cache, args.replay)
        judge = PairJudge(roles_for(model), client=meter, on_decision="link")
        decisions = judge.judge_many([(i.a, i.b, None) for i in r])
        judged = [
            JudgedPair(
                id=i.id,
                same=i.same,
                forward=d.forward.decision if d.forward else None,
                backward=d.backward.decision if d.backward else None,
            )
            for i, d in zip(r, decisions, strict=True)
        ]
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
        card = pair_card(judged, made, audited=int(noise["audited"]), label_noise=noise["noise"])
        card.write(out / "cards")
        result["pair"] = card
        _write_pairs(out / "items" / slug / "pair.jsonl", judged)
        spent_rows.append(("pair judge", meter))
    if "fact" in args.judges and f:
        meter = client_for(model, args.cache, args.replay)
        judge = FactJudge(roles_for(model), client=meter)
        decided = judge.judge_rows([item.row for item in f])
        judged = [
            JudgedPair(
                id=item.id,
                same=bool(item.same),
                forward=e.gold_first.decision if e.gold_first else None,
                backward=e.prediction_first.decision if e.prediction_first else None,
            )
            for item, e in zip(f, decided, strict=True)
        ]
        _write_pairs(out / "items" / slug / "fact.jsonl", judged)
        labelled = {item.id: item for item in f}
        result["fact"] = {}
        for split in ("gate", "dev"):
            scored = [
                j
                for j in judged
                if labelled[j.id].same is not None and labelled[j.id].split == split
            ]
            if not scored:
                continue
            unsure = [
                j for j in judged if labelled[j.id].same is None and labelled[j.id].split == split
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
        spent_rows.append(("fact-equivalence judge", meter))
    result["spent"] = []
    for job, meter in spent_rows:
        live = meter.live
        if live.get("calls") and args.replay is None:
            usd = (
                live["usd"]
                if not live.get("unpriced")
                else spend.cost(model, int(live["input_tokens"]), int(live["output_tokens"]))
            )
            row = spend.record(
                f"calibration #134/#135: {job}{f' ({args.who})' if args.who else ''}",
                model,
                calls=int(live["calls"]),
                input_tokens=int(live["input_tokens"]),
                output_tokens=int(live["output_tokens"]),
                usd=usd,
                ledger=args.ledger,
            )
            result["spent"].append(row)
    return result


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
        for row in found.get("spent", ()):
            lines.append(f"- spent: {row['who']}: {row['calls']} calls, ${row['usd']}")
        lines.append("")
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
    if isinstance(value, JudgedFact):
        return {"id": value.id, "label": value.label, "runs": list(value.runs)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def estimate(
    args: argparse.Namespace, g: list[GItem], r: list[RItem], f: list[FItem]
) -> dict[str, list[dict[str, Any]]]:
    """Per model, each job's calls, the share the cache holds, tokens and USD. No call is made."""
    store = CachedClient(_Never(), args.cache).store if args.cache is not None else None
    out = {}
    for model in args.model:
        plans: list[Plan] = []
        if "ground" in args.judges:
            plans += [plan_grounding(model, v, g, args.runs, store) for v in args.variants]
        if "pair" in args.judges:
            plans.append(plan_pairs(model, r, store))
        if "fact" in args.judges:
            plans.append(plan_facts(model, f, store))
        out[model] = priced(model, plans)
    return out


class _Never:
    """A client for an estimate: it is never called."""

    def complete(self, messages: Any, *, spec: Any, schema: Any = None) -> Completion:
        raise RuntimeError("an estimate makes no call")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", help="a model to calibrate; repeat for more")
    parser.add_argument("--cache", type=Path, help="the response cache's directory")
    parser.add_argument("--out", "-o", type=Path, default=Path("calibration-out"))
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--judges", default="ground,pair,fact", help="ground, pair, fact")
    parser.add_argument("--variants", default=",".join(VARIANTS), help=", ".join(VARIANTS))
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
    args = parser.parse_args(argv)
    args.model = args.model or [DEFAULT_GROUND]
    args.judges = {j.strip() for j in args.judges.split(",") if j.strip()}
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown or not args.judges <= {"ground", "pair", "fact"}:
        parser.error(f"unknown variant or judge: {unknown or sorted(args.judges)}")
    caps = {}
    for given in args.cap:
        name, _, usd = given.partition("=")
        caps[name] = float(usd)

    g = load_g(every=args.every) if "ground" in args.judges else []
    r, noise = load_r() if "pair" in args.judges else ([], {"noise": None, "audited": 0})
    f = load_f() if "fact" in args.judges else []
    estimated = estimate(args, g, r, f)
    by_provider: dict[str, float] = defaultdict(float)
    for model, jobs in estimated.items():
        print(
            f"\n{model}: {len(g)} G items, {len(r)} R pairs, {len(f)} F pairs, {args.runs} run(s)"
        )
        for job in jobs:
            usd = "unpriced" if job["usd"] is None else f"${job['usd']:.3f}"
            print(
                f"  {job['job']:<52} {job['calls']:>5} calls, {job['cached']:>5} cached, "
                f"{job['input_tokens']:>8} in, {job['output_tokens']:>6} out  {usd}"
            )
            by_provider[spend.provider(model)] += job["usd"] or 0.0
    if args.estimate:
        return 0
    for name, usd in by_provider.items() if args.replay is None else ():
        model = next(m for m in args.model if spend.provider(m) == name)
        try:
            print(spend.check(model, usd, cap=caps.get(name), ledger=args.ledger))
        except spend.SpendRefused as refused:
            print(f"refused: {refused}", file=sys.stderr)
            return 3
    args.out.mkdir(parents=True, exist_ok=True)
    results = [run_model(model, args, args.out, g, r, noise, f) for model in args.model]
    extra: dict[str, Any] = {}
    claude = [x for x in results if spend.provider(x["model"]) == "anthropic" and "p1" in x]
    others = [x for x in results if spend.provider(x["model"]) != "anthropic" and "p1" in x]
    if claude and others:
        gate = {i.id for i in g if i.split == "gate"}
        a = [x for x in claude[0]["p1"] if x.id in gate]
        b = [x for x in others[0]["p1"] if x.id in gate]
        extra["self_preference"] = self_preference(a, b, (claude[0]["model"], others[0]["model"]))
    (args.out / "results.md").write_text(table(results, extra) + "\n", encoding="utf-8")
    (args.out / "results.json").write_text(
        json.dumps(
            {"results": _jsonable(results), **_jsonable(extra), "estimate": estimated}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
