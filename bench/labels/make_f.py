"""Label set F: 300 pairs of facts for the fact-equivalence judge (issue #143).

    uv run python bench/labels/make_f.py RUNS/cmp DATA/redocred/test_revised.json

Reads the comparison `bench/run_all.sh` wrote, as `make_g.py` does, and
Re-DocRED's own test file, for its entity clusters and the evidence sentences
its gold lists. Calls no model. A pair is a gold fact an extractor missed, by
the dataset's own scoring, and a triple of that extractor's the scoring did not
match, which the judge's pre-filter lets through (`surface_pair`): the same
relation, one end the same name after normalisation, and the other end not
exactly equal. Writes to `bench/labels/F/`:

- `items.jsonl`: the rows `odke label make fact` turns into sheets, in sheet
  order, so row n is item F-n: the relation, the two facts in an order drawn
  per item, and the gold fact's evidence, with nothing that says which is gold;
- `items.private.jsonl`: what each item is, which no sheet shows;
- `gate.jsonl`: the gate split's documents, which the prompt-leakage test reads.

The same inputs and seed write the same bytes. bench/labels/README.md says how
the pairs are drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from openodke.eval.datasets import redocred
from openodke.eval.equivalence import surface_pair
from openodke.eval.formats import FactPair
from openodke.extract._common import entity_key


def _sibling(name: str) -> ModuleType:
    """`make_g.py` beside this file: its reader of the comparison is F's too."""
    path = Path(__file__).with_name(f"{name}.py")
    found = sys.modules.get(f"odke_bench_{name}")
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(f"odke_bench_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses look their module up while it loads.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


g = _sibling("make_g")

SEED = 143
T2K, REDOCRED = g.T2K, g.REDOCRED
DATASETS: tuple[str, ...] = g.DATASETS
EXTRACTORS: tuple[str, ...] = tuple(g.EXTRACTORS)
SIDES = ("subject", "object")

Triple = tuple[str, str, str]


@dataclass(frozen=True)
class Plan:
    """How many. The defaults are issue #143's."""

    per_dataset: int = 150  # a third per extractor, as the data allows
    dev: int = 100  # of all the pairs, in whole documents; the rest are the gate
    # The most pairs one gold fact, or one prediction, is shown in. Once each
    # gives 243 pairs; Text2KGBench has 107 distinct ones in all.
    reuse: int = 2


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Item:
    """One pair: a gold fact an extractor missed, and a triple of its the scoring left."""

    dataset: str
    name: str  # the set: a Text2KGBench ontology, or redocred
    doc: str
    gold: int  # the gold fact's place in its document's gold
    triple: Triple  # the prediction, as the extractor wrote it
    shown: Triple  # the gold fact, spelled as the text spells it (`make_g.surface`)
    written: Triple  # the gold fact, as the dataset writes it
    extractor: str
    side: str  # the end of the prediction that differs: subject or object

    @property
    def status(self) -> str:
        """The stratum `make_g.split` balances dev on, beside dataset and extractor."""
        return self.side

    @property
    def claim(self) -> tuple[str, str, tuple[str, ...]]:
        """The prediction as a reader sees it: two items never show the same one."""
        return (self.dataset, self.doc, tuple(g.fold(x) for x in self.triple))

    def fact_id(self, seed: int) -> str:
        key = [seed, self.dataset, self.doc, self.gold, self.extractor, list(self.triple)]
        return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]


def _ends(s: Any, doc: str, index: int) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    """A gold fact's ends as every name it goes by: a Re-DocRED cluster's mentions, or the
    Text2KGBench string as written and as the text spells it."""
    written, shown = s.gold(doc, as_written=True)[index], s.gold(doc)[index]
    if s.dataset == REDOCRED:
        head, _, tail = s.rows[doc]["facts"][index]
        clusters = s.raw[doc]["vertexSet"]
        subjects = tuple(dict.fromkeys([shown[0], *(m["name"] for m in clusters[head])]))
        objects = tuple(dict.fromkeys([shown[2], *(m["name"] for m in clusters[tail])]))
        return subjects, written[1], objects
    return (written[0], shown[0]), written[1], (written[2], shown[2])


def candidates(s: Any) -> list[Item]:
    """Every pair the pre-filter lets through in one set, per extractor.

    The gold end's names include its spelling in the text (`make_g.surface`),
    so a prediction that only undoes the dataset's tokenised name or padded
    date ("2010" for "01 January 2010") is exactly equal to one of them and
    makes no pair: there is nothing for a person to judge.
    """
    out: list[Item] = []
    for extractor in EXTRACTORS:
        for doc in sorted(s.rows):
            said = s.predicted[extractor].get(doc, [])
            unmatched = [t for t in said if s.matched(doc, t) is None]
            if not unmatched:
                continue
            written, shown = s.gold(doc, as_written=True), s.gold(doc)
            for index in range(len(written)):
                if s.score(doc, said, index) >= 1:
                    continue
                ends = _ends(s, doc, index)
                for triple in unmatched:
                    side = surface_pair(ends, triple)
                    if side is not None:
                        out.append(
                            Item(
                                s.dataset,
                                s.name,
                                doc,
                                index,
                                triple,
                                shown[index],
                                written[index],
                                extractor,
                                side,
                            )
                        )
    return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


@dataclass
class Draw:
    """What has been drawn so far: no pair twice, and no gold fact or prediction past `reuse`."""

    reuse: int = 2
    pairs: set[Any] = field(default_factory=set)
    golds: Counter[tuple[str, str, int]] = field(default_factory=Counter)
    claims: Counter[Any] = field(default_factory=Counter)

    def take(self, item: Item) -> bool:
        gold = (item.dataset, item.doc, item.gold)
        pair = (gold, item.claim)
        if pair in self.pairs or max(self.golds[gold], self.claims[item.claim]) >= self.reuse:
            return False
        self.pairs.add(pair)
        self.golds[gold] += 1
        self.claims[item.claim] += 1
        return True


def _shuffled(items: Iterable[Item], rng: random.Random) -> list[Item]:
    out = sorted(items, key=lambda i: (i.doc, i.gold, i.triple, i.extractor))
    rng.shuffle(out)
    return out


def draw(pool: Sequence[Item], plan: Plan, seed: int, notes: list[str]) -> list[Item]:
    """`plan.per_dataset` per dataset, a third per extractor, as evenly as the data allows.

    The extractors draw in turn (`make_g._fill`), so a pair two of them share
    goes to one of them. A shortfall is made up by the dataset's other
    extractors, then, once each dataset has drawn its own, by the other
    dataset; each is noted.
    """
    rng = random.Random(f"{seed}:draw")
    pools: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in pool:
        pools[(item.dataset, item.extractor)].append(item)
    for key in sorted(pools):
        pools[key] = _shuffled(pools[key], rng)
    taken = Draw(reuse=plan.reuse)
    got: dict[str, list[Item]] = {}
    for cell, dataset in enumerate(DATASETS):
        quotas = [plan.per_dataset // len(EXTRACTORS)] * len(EXTRACTORS)
        for k in range(plan.per_dataset % len(EXTRACTORS)):
            quotas[(cell + k) % len(EXTRACTORS)] += 1
        drawn = g._fill([pools[(dataset, e)] for e in EXTRACTORS], quotas, taken)
        for extractor, items, quota in zip(EXTRACTORS, drawn, quotas, strict=True):
            if len(items) < quota:
                notes.append(
                    f"{dataset}: {extractor} gave {len(items)} of {quota}; "
                    "its dataset's other extractors made up what they could"
                )
        got[dataset] = [i for items in drawn for i in items]
    for dataset in DATASETS:
        short = plan.per_dataset - len(got[dataset])
        if short <= 0:
            continue
        others = [d for d in DATASETS if d != dataset]
        other = [pools[(d, e)] for d in others for e in EXTRACTORS]
        # Shared among the other dataset's extractors, as evenly as they allow.
        quotas = [short // len(other) + (k < short % len(other)) for k in range(len(other))]
        more = [i for items in g._fill(other, quotas, taken) for i in items]
        notes.append(
            f"{dataset}: {len(got[dataset])} of {plan.per_dataset}; "
            f"{len(more)} more from {', '.join(others)}"
        )
        for item in more:
            got[item.dataset].append(item)
    return [i for dataset in DATASETS for i in got[dataset]]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _fact(s: Any, doc: str, triple: Triple, label: str) -> dict[str, Any]:
    """A triple as a fact, typed as `make_g` types a claim; the same for gold and prediction."""
    subject, _, obj = triple
    p = s.predicate(label)
    s_type = g._type(s, doc, subject, p["domain"])
    fact: dict[str, Any] = {
        "subject": {"key": entity_key(s_type, subject), "type": s_type, "label": subject},
        "predicate": s.predicates[label],
    }
    if p["range"] in g.LITERALS:
        fact["object_value"] = obj
    else:
        o_type = g._type(s, doc, obj, [p["range"]])
        fact["object_entity"] = {"key": entity_key(o_type, obj), "type": o_type, "label": obj}
    return fact


def passage(s: Any, item: Item) -> tuple[str, list[int] | None]:
    """The gold fact's evidence, and the Re-DocRED sentences it is, as the judge is shown it.

    A Text2KGBench item's text is one sentence. A Re-DocRED item's is the
    sentences its gold lists as evidence, with `…` between ones that are not
    adjacent; with none listed, where both its ends are named, or else where
    the less mentioned one is (`make_g._focus`).
    """
    if s.dataset != REDOCRED:
        return s.texts[item.doc], None
    focus = g._focus(s, item)
    if not focus:
        return s.texts[item.doc], None
    sentences = [redocred.detokenize([tokens]) for tokens in s.raw[item.doc]["sents"]]
    parts: list[str] = []
    for at, k in enumerate(focus):
        if at and k != focus[at - 1] + 1:
            parts.append("…")
        parts.append(sentences[k])
    return " ".join(parts), focus


@dataclass
class Made:
    """What `make` drew: the rows of every file, and the counts the README quotes."""

    rows: list[dict[str, Any]]
    private: list[dict[str, Any]]
    gate: list[dict[str, Any]]
    candidates: dict[tuple[str, str], int]
    notes: list[str]

    def table(self) -> str:
        """Counts by dataset and extractor, with the candidates and the splits; then by side."""
        heads = ["openodke", "LGT", "neo4j-graphrag"]
        lines = [
            "| Dataset | " + " | ".join(heads) + " | Total | dev | gate | Candidates |",
            "|---|" + "---:|" * (len(heads) + 4),
        ]
        for dataset in DATASETS:
            rows = [p for p in self.private if p["dataset"] == dataset]
            counts = [sum(p["extractor"] == e for p in rows) for e in EXTRACTORS]
            dev = sum(p["split"] == "dev" for p in rows)
            pool = sum(self.candidates.get((dataset, e), 0) for e in EXTRACTORS)
            cells = [dataset, *map(str, counts), str(len(rows)), str(dev), str(len(rows) - dev)]
            lines.append("| " + " | ".join([*cells, str(pool)]) + " |")
        counts = [sum(p["extractor"] == e for p in self.private) for e in EXTRACTORS]
        dev = sum(p["split"] == "dev" for p in self.private)
        total = len(self.private)
        pool = sum(self.candidates.values())
        cells = ["**all**", *map(str, counts), str(total), str(dev), str(total - dev), str(pool)]
        lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "| Differs | " + " | ".join(DATASETS) + " | Total |", "|---|---:|---:|---:|"]
        for side in SIDES:
            counts = [
                sum(p["differs"] == side and p["dataset"] == d for p in self.private)
                for d in DATASETS
            ]
            lines.append(f"| {side} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
        first = sum(p["gold_side"] == "first" for p in self.private)
        lines += ["", f"Gold is Fact 1 in {first} of {total} and Fact 2 in {total - first}."]
        return "\n".join(lines)


def make(sets: Sequence[Any], plan: Plan | None = None, seed: int = SEED) -> Made:
    """Draw, split and shuffle the pairs; nothing is written."""
    plan = plan or Plan()
    notes: list[str] = []
    pool = [item for s in sets for item in candidates(s)]
    counted = Counter((i.dataset, i.extractor) for i in pool)
    drawn = draw(pool, plan, seed, notes)
    splits = g.split(drawn, plan, seed)
    by_name = {s.name: s for s in sets}
    shared: dict[tuple[str, str, int, Any], list[str]] = defaultdict(list)
    for item in pool:
        key = (item.dataset, item.doc, item.gold, item.claim)
        if item.extractor not in shared[key]:
            shared[key].append(item.extractor)
    everything = sorted(drawn, key=lambda i: i.fact_id(seed))
    random.Random(f"{seed}:order").shuffle(everything)
    width = max(4, len(str(len(everything))))
    rows, private = [], []
    for n, item in enumerate(everything, start=1):
        s = by_name[item.name]
        fact_id = item.fact_id(seed)
        label = item.written[1]
        gold = _fact(s, item.doc, item.shown, label)
        said = _fact(s, item.doc, item.triple, label)
        first = random.Random(f"{seed}:side:{fact_id}").random() < 0.5
        text, kept = passage(s, item)
        predicate = s.predicates[label]
        row = {
            "id": fact_id,
            "relation": predicate,
            "description": s.predicate(label).get("description"),
            "first": gold if first else said,
            "second": said if first else gold,
            "passage": text,
        }
        FactPair.model_validate(row)
        rows.append(row)
        also = shared[(item.dataset, item.doc, item.gold, item.claim)]
        private.append(
            {
                "id": f"F-{n:0{width}d}",
                "fact_id": fact_id,
                "dataset": item.dataset,
                "set": item.name,
                "doc": item.doc,
                "extractor": item.extractor,
                "also_written_by": [e for e in also if e != item.extractor],
                "gold": list(item.written),
                "gold_shown": list(item.shown),
                "predicted": list(item.triple),
                "gold_side": "first" if first else "second",
                "differs": item.side,
                "split": splits[item],
                "sentences": kept,
            }
        )
    gate = sorted({(item.dataset, item.name, item.doc) for item in drawn if splits[item] == "gate"})
    gate_rows = [{"doc": doc, "text": by_name[name].texts[doc]} for _, name, doc in gate]
    return Made(rows, private, gate_rows, dict(sorted(counted.items())), notes)


def write(made: Made, out: Path) -> list[Path]:
    """The three files under `out`, byte-identical for the same `made`."""
    out.mkdir(parents=True, exist_ok=True)

    def lines(rows: Iterable[dict[str, Any]]) -> str:
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    files = {
        "items.jsonl": lines(made.rows),
        "items.private.jsonl": lines(made.private),
        "gate.jsonl": lines(made.gate),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return [out / name for name in files]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("cmp", type=Path, help="the comparison: t2k/ont_* and redocred/")
    parser.add_argument("redocred_raw", type=Path, help="Re-DocRED's test_revised.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "F")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    made = make(g.load_all(args.cmp, args.redocred_raw), seed=args.seed)
    for path in write(made, args.out):
        print(f"wrote {path}", file=sys.stderr)
    print(made.table())
    for note in made.notes:
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
