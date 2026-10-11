"""Label set R: 600 entity pairs from Re-DocRED's mention clusters (issue #151).

    uv run python bench/labels/make_r.py DATA/redocred/test_revised.json

Reads Re-DocRED's test file, which groups each document's mentions into entity
clusters, with their types and positions. Calls no model. Within a document,
two mentions of one cluster are the same entity, and two clusters of one type
are two, so every label is the dataset's own. Writes to `bench/labels/R/`:

- `items.jsonl`: one row `odke label make pair` reads per pair, in id order,
  so row n is item R-n: each mention's opaque key, type, name and context;
- `labels.jsonl`: each item's label from the dataset, a `PairLabel`;
- `items.private.jsonl`: what each item is, which no sheet shows;
- `gate.jsonl`: the gate split's passages, which the prompt-leakage test reads;
- `audit.jsonl`: the 100 items the owner labels by hand, in sheet order.

Identity across documents needs ids, and Re-DocRED has none; it comes from
T-REx with Wikidata ids (#117, in 0.6.0), and `Plan.across` is its slot. The
same input and seed write the same bytes. bench/labels/README.md says how the
pairs are drawn.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from openodke.corroborate import name_key, name_similarity
from openodke.eval.datasets.redocred import TYPES, detokenize
from openodke.eval.formats import PairLabel
from openodke.eval.sheets import PairItem

SEED = 151
STRATA = ("same", "hard", "other")
# DocRED's value types: not entities, so never in a pair.
VALUES = frozenset({"TIME", "NUM"})
# The pair judge's default band (#150): the pairs it is asked about.
BAND = (0.7, 0.9)
# A different pair is hard when blocking lets it through, or its names are this alike.
HARD = 0.5


@dataclass(frozen=True)
class Plan:
    """How many of each. The defaults are issue #151's."""

    same: int = 300  # two mentions of one entity, under two names
    hard: int = 200  # two entities of one type whose names are alike
    other: int = 100  # two entities of one type, any other names
    dev: int = 200  # of all the pairs, split by document; the rest are the gate
    audit_same: int = 50  # the owner's audit: half same
    audit_hard: int = 40  # and hard negatives over their share of R
    audit_other: int = 10
    # Pairs across documents. Re-DocRED has no ids, so none: T-REx (#117, 0.6.0).
    across: int = 0

    @property
    def total(self) -> int:
        return self.same + self.hard + self.other

    def audit(self, stratum: str) -> int:
        return int(getattr(self, f"audit_{stratum}"))


@dataclass(frozen=True)
class Mention:
    """One mention of an entity: where it is, and the name as the text spells it."""

    doc: str
    sent: int
    start: int  # token offsets in the sentence, as Re-DocRED gives them
    end: int
    label: str

    @property
    def key(self) -> str:
        """Its place in the document, which says nothing about its entity."""
        return f"{self.doc}:s{self.sent}:{self.start}-{self.end}"


@dataclass
class Cluster:
    """One entity of a document: its type and its mentions under each name key."""

    index: int
    type: str
    names: dict[str, list[Mention]] = field(default_factory=dict)


@dataclass
class Doc:
    """One Re-DocRED document, its entity clusters and its text."""

    id: str
    sents: list[list[str]]
    clusters: list[Cluster]

    @property
    def text(self) -> str:
        return detokenize(self.sents)

    def context(self, mention: Mention) -> str:
        """The mention's sentence and one either side, as the pair judge reads it."""
        lo = max(0, mention.sent - 1)
        return detokenize(self.sents[lo : mention.sent + 2])


def load(raw: Sequence[dict[str, Any]], *, split: str = "test") -> list[Doc]:
    """Each document's entity clusters, ids as `odke bench` gives them (`test_0042`).

    A cluster is typed by its mentions' majority type; TIME and NUM mentions
    are values and are left out. A name is the mention's tokens joined as the
    text joins them, keyed by `name_key`.
    """
    docs = []
    for i, doc in enumerate(raw):
        doc_id = f"{split}_{i:04d}"
        clusters = []
        for c, vertex in enumerate(doc["vertexSet"]):
            found = [m for m in vertex if m["type"] not in VALUES]
            if not found:
                continue
            kind = Counter(m["type"] for m in found).most_common(1)[0][0]
            cluster = Cluster(c, TYPES[kind])
            for m in sorted(found, key=lambda m: (m["sent_id"], m["pos"][0])):
                start, end = m["pos"]
                label = detokenize([doc["sents"][m["sent_id"]][start:end]])
                if key := name_key(label):
                    mention = Mention(doc_id, m["sent_id"], start, end, label)
                    cluster.names.setdefault(key, []).append(mention)
            if cluster.names:
                clusters.append(cluster)
        docs.append(Doc(doc_id, [list(s) for s in doc["sents"]], clusters))
    return docs


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pair:
    """Two mentions of one document, and what the dataset says of them."""

    stratum: str
    a: Mention
    b: Mention
    type: str
    clusters: tuple[int, int]
    score: float  # the resolver's name score, as it is with no normaliser

    @property
    def doc(self) -> str:
        return self.a.doc

    @property
    def same(self) -> bool:
        return self.stratum == "same"


def blocked(a: str, b: str) -> bool:
    """Whether the resolver's blocking compares two name keys: a shared first or last token."""
    left, right = a.split(), b.split()
    return bool({left[0], left[-1]} & {right[0], right[-1]})


def candidates(doc: Doc, seed: int) -> dict[str, list[Pair]]:
    """Every name pair of the document, by stratum, each with one mention per name.

    Two names with one name key are never a pair: the rules settle them, so
    they never reach a judge. Which mention stands for a name is drawn per
    pair, so the same input and seed give the same mentions.
    """
    out: dict[str, list[Pair]] = {s: [] for s in STRATA}

    def pair(stratum: str, ca: Cluster, ka: str, cb: Cluster, kb: str) -> Pair:
        rng = random.Random(f"{seed}:{doc.id}:{ca.index}:{ka}:{cb.index}:{kb}")
        a, b = rng.choice(ca.names[ka]), rng.choice(cb.names[kb])
        score = round(name_similarity(ka, kb), 4)
        return Pair(stratum, a, b, ca.type, (ca.index, cb.index), score)

    for cluster in doc.clusters:
        for ka, kb in combinations(sorted(cluster.names), 2):
            out["same"].append(pair("same", cluster, ka, cluster, kb))
    for ca, cb in combinations(doc.clusters, 2):
        if ca.type != cb.type:
            continue
        for ka in sorted(ca.names):
            for kb in sorted(cb.names):
                if ka == kb:
                    continue
                hard = blocked(ka, kb) or name_similarity(ka, kb) >= HARD
                out["hard" if hard else "other"].append(
                    pair("hard" if hard else "other", ca, ka, cb, kb)
                )
    return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """`total` shared in proportion to `sizes`, largest remainder first, summing exactly."""
    whole = sum(sizes.values())
    exact = {k: total * v / whole for k, v in sizes.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def split_docs(docs: Sequence[Doc], plan: Plan, seed: int) -> dict[str, str]:
    """Each document's split: a shuffled `dev / total` share of them is dev, the rest gate."""
    ids = sorted(d.id for d in docs if d.clusters)
    random.Random(f"{seed}:split").shuffle(ids)
    cut = round(len(ids) * plan.dev / plan.total)
    return {doc: ("dev" if n < cut else "gate") for n, doc in enumerate(ids)}


def _draw(by_doc: dict[str, list[Pair]], quota: int, rng: random.Random) -> list[Pair]:
    """`quota` pairs, one per document a pass, so they spread over the documents."""
    pools = {doc: rng.sample(pairs, len(pairs)) for doc, pairs in sorted(by_doc.items()) if pairs}
    order = sorted(pools)
    rng.shuffle(order)
    drawn: list[Pair] = []
    while len(drawn) < quota and order:
        for doc in list(order):
            if len(drawn) == quota:
                break
            drawn.append(pools[doc].pop())
            if not pools[doc]:
                order.remove(doc)
    if len(drawn) < quota:
        raise ValueError(f"wanted {quota} pairs, the documents have {len(drawn)}")
    return drawn


@dataclass
class Made:
    """What `make` drew: the rows of every file, and the counts the README quotes."""

    rows: list[dict[str, Any]]
    labels: list[dict[str, Any]]
    private: list[dict[str, Any]]
    gate: list[dict[str, Any]]
    audit: list[dict[str, Any]]

    def table(self) -> str:
        """Counts by stratum and split, how many are in the judge's band, and the audit."""
        lines = [
            "| Stratum | Label | dev | gate | Total | In the band | Audited |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for stratum in (*STRATA, None):
            rows = [p for p in self.private if stratum is None or p["stratum"] == stratum]
            dev = sum(p["split"] == "dev" for p in rows)
            band = sum(p["band"] for p in rows)
            audited = sum(p["audit"] is not None for p in rows)
            name = f"**{stratum or 'all'}**" if stratum is None else stratum
            label = "" if stratum is None else ("same" if stratum == "same" else "different")
            cells = [name, label, dev, len(rows) - dev, len(rows), band, audited]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
        return "\n".join(lines)


def make(docs: Sequence[Doc], plan: Plan | None = None, seed: int = SEED) -> Made:
    """Draw, split, shuffle and pick the audit; nothing is written."""
    plan = plan or Plan()
    if plan.across:
        raise ValueError(
            "pairs across documents need ids, and Re-DocRED has none: they come from T-REx "
            "with Wikidata ids (#117, 0.6.0)"
        )
    splits = split_docs(docs, plan, seed)
    found = {doc.id: candidates(doc, seed) for doc in docs if doc.id in splits}
    sizes = {s: int(getattr(plan, s)) for s in STRATA}
    dev = _allocate(sizes, plan.dev)
    drawn: list[tuple[str, Pair]] = []
    for name in ("dev", "gate"):
        for stratum in STRATA:
            quota = dev[stratum] if name == "dev" else sizes[stratum] - dev[stratum]
            by_doc = {d: c[stratum] for d, c in found.items() if splits[d] == name}
            rng = random.Random(f"{seed}:{name}:{stratum}")
            drawn += [(name, p) for p in _draw(by_doc, quota, rng)]
    random.Random(f"{seed}:order").shuffle(drawn)

    audited: dict[int, str] = {}
    picked = []
    for stratum in STRATA:
        at = [n for n, (_, p) in enumerate(drawn) if p.stratum == stratum]
        picked += random.Random(f"{seed}:audit:{stratum}").sample(at, plan.audit(stratum))
    random.Random(f"{seed}:audit").shuffle(picked)
    width = max(4, len(str(len(picked))))
    audited = {n: f"P-{k:0{width}d}" for k, n in enumerate(picked, start=1)}

    by_id = {doc.id: doc for doc in docs}
    width = max(4, len(str(len(drawn))))
    rows, labels, private = [], [], []
    for n, (name, pair) in enumerate(drawn):
        doc = by_id[pair.doc]
        row = {
            side: {
                "key": mention.key,
                "type": pair.type,
                "label": mention.label,
                "context": doc.context(mention),
            }
            for side, mention in (("a", pair.a), ("b", pair.b))
        }
        PairItem.model_validate(row)
        rows.append(row)
        label = {"a": pair.a.key, "b": pair.b.key, "same": pair.same}
        PairLabel.model_validate(label)
        labels.append(label)
        private.append(
            {
                "id": f"R-{n + 1:0{width}d}",
                "doc": pair.doc,
                "split": name,
                "scope": "within",
                "stratum": pair.stratum,
                "same": pair.same,
                "clusters": list(pair.clusters),
                "names": [pair.a.label, pair.b.label],
                "score": pair.score,
                "band": BAND[0] <= pair.score < BAND[1],
                "audit": audited.get(n),
            }
        )
    gate = [{"doc": d.id, "text": d.text} for d in docs if splits.get(d.id) == "gate"]
    audit = [rows[n] for n in picked]
    return Made(rows, labels, private, gate, audit)


def write(made: Made, out: Path) -> list[Path]:
    """The five files under `out`, byte-identical for the same `made`."""
    out.mkdir(parents=True, exist_ok=True)

    def lines(rows: Iterable[dict[str, Any]]) -> str:
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    files = {
        "items.jsonl": lines(made.rows),
        "labels.jsonl": lines(made.labels),
        "items.private.jsonl": lines(made.private),
        "gate.jsonl": lines(made.gate),
        "audit.jsonl": lines(made.audit),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return [out / name for name in files]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("redocred_raw", type=Path, help="Re-DocRED's test_revised.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "R")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    raw = json.loads(args.redocred_raw.read_text(encoding="utf-8"))
    made = make(load(raw), seed=args.seed)
    for path in write(made, args.out):
        print(f"wrote {path}", file=sys.stderr)
    print(made.table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
