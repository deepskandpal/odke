"""Label set G: 600 grounding items from the published comparison (issue #133).

    uv run python bench/labels/make_g.py RUNS/cmp DATA/redocred/test_revised.json

Reads the comparison `bench/run_all.sh` wrote (Text2KGBench under `t2k/ont_*`,
Re-DocRED under `redocred/`) and Re-DocRED's own test file, which has the
evidence sentences its gold lists. Calls no model. Writes to `bench/labels/G/`:

- `items.jsonl`: the rows `odke label make grounding` turns into sheets, in
  sheet order, so row n is item G-n;
- `items.private.jsonl`: what each item is, which no sheet shows;
- `gate.jsonl`: the gate split's rows, which the prompt-leakage test reads;
- `selfagreement.ids`: the items labelled a second time, a week later.

The same inputs and seed write the same bytes. bench/labels/README.md says how
the items are drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openodke.eval.datasets import redocred, text2kgbench
from openodke.eval.sheets import GroundingItem
from openodke.extract._common import entity_key

SEED = 133
T2K, REDOCRED = "text2kgbench", "redocred"
DATASETS = (T2K, REDOCRED)
# Each extractor's own triples, before openodke's grounding touched them. The
# folder is where a set keeps that extractor's predictions.
EXTRACTORS = {
    "openodke": "predictions",
    "lgt": "competitors/lgt/predictions",
    "neo4j": "competitors/neo4j/predictions",
}
REFERENCE = "extraction-alone.jsonl"
STATUSES = ("match", "not_in_gold", "gold_only")
KINDS = ("object_swap", "subject_object_swap", "relation_swap", "polarity", "value_change")
# What a planted item may be labelled: by construction, anything but supported.
NOT_SUPPORTED = ["contradicted", "not_found"]
LITERALS = {"string", "date", "number", "integer", "boolean"}
# Relations that read the same both ways round: swapping them changes nothing.
SYMMETRIC = {
    "spouse",
    "sibling",
    "sister city",
    "shares border with",
    "partner",
    "twinned administrative body",
    "diplomatic relation",
}
# Relations one passage often states together ("London, United Kingdom" says
# country and located-in at once). A swap within a group could still be
# supported, so it is never planted.
NEAR = (
    {
        "country",
        "located in the administrative territorial entity",
        "continent",
        "location",
        "headquarters location",
        "basin country",
        "located in or next to body of water",
        "located on terrain feature",
        "location of formation",
        "applies to jurisdiction",
        "contains administrative territorial entity",
        "capital",
        "capital of",
        "country of origin",
        "country of citizenship",
        "territory claimed by",
        "work location",
        "residence",
        "narrative location",
        "filming location",
    },
    {"place of birth", "residence", "work location"},
    {"place of death", "residence", "work location"},
    {"original language of work", "languages spoken, written or signed", "official language"},
    {"publication date", "inception", "point in time", "start time"},
    {"end time", "dissolved, abolished or demolished", "point in time"},
    {"head of government", "head of state", "chairperson", "founded by"},
    {
        "author",
        "creator",
        "screenwriter",
        "director",
        "producer",
        "composer",
        "lyrics by",
        "performer",
        "developer",
    },
    {"instance of", "genre", "series", "subclass of", "parent taxon"},
    {
        "member of",
        "part of",
        "member of sports team",
        "league",
        "member of political party",
        "military branch",
        "legislative body",
        "participant of",
        "employer",
    },
    {"subsidiary", "parent organization", "owned by", "operator", "part of", "has part"},
    {
        "publisher",
        "manufacturer",
        "production company",
        "record label",
        "original network",
        "operator",
        "owned by",
        "developer",
    },
    {"award received", "nominated for"},
    {"elected in", "candidacy in election"},
    {"cast member", "characters", "participant", "present in work"},
    {"notable work", "has part", "product or material produced"},
)
YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
NUMBER = re.compile(r"\d[\d,]*\d|\d")

Triple = tuple[str, str, str]


@dataclass(frozen=True)
class Plan:
    """How many of each. The defaults are issue #133's."""

    match: int = 100  # per dataset: predictions the dataset's scoring counts as gold
    not_in_gold: int = 100  # per dataset: predictions it does not
    gold_only: int = 50  # per dataset: gold facts no extractor wrote
    planted: int = 50  # per dataset, spread over KINDS
    dev: int = 200  # of the real items; the rest are the gate
    self_agreement: int = 100  # of the real items
    trim_words: int = 350  # a Re-DocRED passage longer than this is cut to its evidence

    def wanted(self, status: str) -> int:
        """How many of one gold status each dataset gives."""
        return int(getattr(self, status))


def fold(text: str) -> str:
    """Case, spaces and punctuation ignored, as Re-DocRED's scoring compares names."""
    return re.sub(r"[\W_]+", "", text.casefold())


# --------------------------------------------------------------------------- #
# The published comparison, read back
# --------------------------------------------------------------------------- #


@dataclass
class Docset:
    """One prepared set of the comparison: its gold, texts and every extractor's triples."""

    dataset: str
    name: str
    meta: dict[str, Any]
    ontology: dict[str, Any]
    rows: dict[str, dict[str, Any]]  # doc id -> gold row
    texts: dict[str, str]
    predicted: dict[str, dict[str, list[Triple]]]  # extractor -> doc -> triples
    written: dict[str, list[Triple]]  # doc -> every triple any configuration kept
    raw: dict[str, dict[str, Any]] = field(default_factory=dict)  # Re-DocRED's own docs

    @property
    def predicates(self) -> dict[str, str]:
        """Relation label -> predicate name."""
        return {label: name for name, label in self.meta["relation_labels"].items()}

    def predicate(self, label: str) -> dict[str, Any]:
        return dict(self.ontology["predicates"][self.predicates[label]])

    def gold(self, doc: str, *, as_written: bool = False) -> list[Triple]:
        """The doc's gold facts, in the gold row's order, named as the text names them.

        `as_written` keeps the dataset's own strings, tokenised spacing and all.
        """
        row = self.rows[doc]
        if self.dataset == T2K:
            triples = [(t["sub"], t["rel"], t["obj"]) for t in row["triples"]]
        else:
            names = [_first(cluster) for cluster in self.raw[doc]["vertexSet"]]
            triples = [(names[h], rel, names[t]) for h, rel, t in row["facts"]]
        if as_written:
            return triples
        text = self.texts[doc]
        return [(surface(s, text), r, surface(o, text)) for s, r, o in triples]

    def score(self, doc: str, triples: Sequence[Triple], only: int | None = None) -> float:
        """The dataset's own precision for `triples` (recall when `only` names one gold fact)."""
        row = self.rows[doc]
        if only is not None:
            key = "triples" if self.dataset == T2K else "facts"
            row = {**row, key: [row[key][only]]}
        predicted = {doc: list(triples)}
        if self.dataset == T2K:
            metrics = text2kgbench.score(
                [row], predicted, self.meta["ontology"], hallucination=False
            )
        else:
            metrics = redocred.score([row], predicted)
        return float(metrics["recall" if only is not None else "precision"] or 0)

    def matched(self, doc: str, triple: Triple) -> int | None:
        """The gold fact `triple` is, by the dataset's own matching, or None."""
        if self.score(doc, [triple]) < 1:
            return None
        return next(i for i in range(len(self.gold(doc))) if self.score(doc, [triple], i) >= 1)

    def found(self, doc: str, index: int) -> bool:
        """Whether any extractor, in any configuration, wrote gold fact `index`."""
        return self.score(doc, self.written.get(doc, []), index) >= 1

    def scored(self, doc: str, triple: Triple) -> bool | None:
        """Text2KGBench scores a triple only if the sentence's gold uses its relation."""
        if self.dataset != T2K:
            return None
        used = {t["rel"].replace(" ", "_") for t in self.rows[doc]["triples"]}
        return triple[1].replace(" ", "_") in used


def surface(name: str, text: str) -> str:
    """A gold name as `text` spells it, so no claim gives away that it came from gold.

    Both datasets keep names tokenised ("Assassin 's Creed", "Bleach : Hell
    Verse"), and Text2KGBench pads a date it has no day for: "01 January 2010",
    "00 June 1962". An extractor writes what the text says. So a name the text
    spells with other spacing takes the text's spelling, a padded date loses
    its padding, and any other name is detokenised as Re-DocRED's text is.
    """
    if name in text:
        return name
    spaced = re.search(r"\s*".join(map(re.escape, name.replace(" ", ""))), text)
    if spaced is not None:
        return spaced.group()
    # Text2KGBench's stand-ins for an unknown day or month; its own scoring drops "01 January".
    bare = re.sub(r"^(?:00|01 January)\s+", "", name)
    if bare != name:
        return bare
    return redocred.detokenize([name.split()])


def _first(cluster: Sequence[dict[str, Any]]) -> str:
    """An entity's first mention in the document: what a reader meets it as."""
    return str(min(cluster, key=lambda m: (m["sent_id"], m["pos"][0]))["name"])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _triples(path: Path) -> dict[str, list[Triple]]:
    out: dict[str, list[Triple]] = {}
    for row in _jsonl(path):
        out[row["id"]] = list(dict.fromkeys((s, r, o) for s, r, o in row["triples"]))
    return out


def load(folder: Path, dataset: str, raw: Sequence[dict[str, Any]] = ()) -> Docset:
    """One prepared set and its predictions. `raw` is Re-DocRED's test file, in order."""
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    gold = _jsonl(folder / "gold.jsonl")
    texts = {row["id"]: (folder / "docs" / f"{row['id']}.txt").read_text("utf-8") for row in gold}
    predicted = {name: _triples(folder / sub / REFERENCE) for name, sub in EXTRACTORS.items()}
    written: dict[str, list[Triple]] = defaultdict(list)
    for sub in EXTRACTORS.values():
        for path in sorted((folder / sub).glob("*.jsonl")):
            for doc, triples in _triples(path).items():
                written[doc].extend(t for t in triples if t not in written[doc])
    docs: dict[str, dict[str, Any]] = {}
    if dataset == REDOCRED:
        for row in gold:
            source = raw[int(row["id"].rsplit("_", 1)[1])]
            facts = [[x["h"], redocred.RELATIONS[x["r"]], x["t"]] for x in source["labels"]]
            if source["title"] != row["title"] or facts != row["facts"]:
                raise ValueError(f"{row['id']}: the Re-DocRED file is not the one prepared")
            docs[row["id"]] = source
    return Docset(
        dataset=dataset,
        name=meta.get("ontology_id", dataset),
        meta=meta,
        ontology=json.loads((folder / "ontology.json").read_text(encoding="utf-8")),
        rows={row["id"]: row for row in gold},
        texts=texts,
        predicted=predicted,
        written=dict(written),
        raw=docs,
    )


def load_all(cmp: Path, redocred_raw: Path) -> list[Docset]:
    raw = json.loads(redocred_raw.read_text(encoding="utf-8"))
    sets = [load(folder, T2K) for folder in sorted((cmp / "t2k").glob("ont_*"))]
    return [*sets, load(cmp / "redocred", REDOCRED, raw)]


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Item:
    """One fact to label, before it is written out."""

    dataset: str
    name: str  # the set: a Text2KGBench ontology, or redocred
    doc: str
    triple: Triple
    status: str  # match, not_in_gold or gold_only; planted ones are not_in_gold
    extractor: str | None = None
    gold: int | None = None  # the gold fact it matches, is, or was made from
    scored: bool | None = None
    also: tuple[str, ...] = ()  # other extractors that wrote the same triple
    planted: str | None = None
    denied: bool = False

    @property
    def claim(self) -> tuple[Any, ...]:
        """What the item says, as a reader sees it: two items never say the same."""
        return (self.dataset, self.doc, *(fold(x) for x in self.triple), self.denied)

    def fact_id(self, seed: int) -> str:
        key = [seed, self.dataset, self.doc, self.status, self.extractor, self.planted]
        return hashlib.sha256(json.dumps([*key, self.triple]).encode()).hexdigest()[:16]


def predictions(s: Docset) -> dict[str, list[Item]]:
    """Every extractor's triples as items, `match` or `not_in_gold`, by extractor."""
    by_claim: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for name, docs in s.predicted.items():
        for doc, triples in docs.items():
            for t in triples:
                key = (doc, tuple(fold(x) for x in t))
                if name not in by_claim[key]:
                    by_claim[key].append(name)
    out: dict[str, list[Item]] = {}
    for name, docs in s.predicted.items():
        items = []
        for doc in sorted(docs):
            for t in docs[doc]:
                index = s.matched(doc, t)
                others = by_claim[(doc, tuple(fold(x) for x in t))]
                items.append(
                    Item(
                        dataset=s.dataset,
                        name=s.name,
                        doc=doc,
                        triple=t,
                        status="match" if index is not None else "not_in_gold",
                        extractor=name,
                        gold=index,
                        scored=s.scored(doc, t),
                        also=tuple(o for o in others if o != name),
                    )
                )
        out[name] = items
    return out


def gold_only(s: Docset) -> list[Item]:
    """Gold facts no extractor wrote in any configuration, by the dataset's own matching."""
    return [
        Item(s.dataset, s.name, doc, t, "gold_only", gold=i)
        for doc in sorted(s.rows)
        for i, t in enumerate(s.gold(doc))
        if not s.found(doc, i)
    ]


# --------------------------------------------------------------------------- #
# Planted false facts
# --------------------------------------------------------------------------- #


def _near(a: str, b: str) -> bool:
    return any(a in group and b in group for group in NEAR)


def _changed(value: str, text: str, rng: random.Random) -> str | None:
    """`value` with its year moved, or else its first number changed, to one the text lacks."""
    year = YEAR.search(value)
    if year is not None:
        found = year
        options = [int(year.group()) + d for d in (-4, -3, -2, -1, 1, 2, 3, 4)]
    else:
        number = NUMBER.search(value)
        if number is None:
            return None
        found = number
        digits = number.group().replace(",", "")
        step = 10 ** max(0, len(digits) - 2)
        options = [int(digits) + d * step for d in range(1, 10)]
    rng.shuffle(options)

    def spelled(n: int) -> str:
        return f"{n:,}" if "," in found.group() else str(n)

    fresh = [n for n in options if n > 0 and spelled(n) not in text] or options
    return value[: found.start()] + spelled(fresh[0]) + value[found.end() :]


def _entities(s: Docset, doc: str) -> list[tuple[str, str]]:
    """Each entity the doc's gold names, with its type: (name, type)."""
    if s.dataset == REDOCRED:
        clusters = s.raw[doc]["vertexSet"]
        return [(surface(_first(c), s.texts[doc]), redocred.TYPES[c[0]["type"]]) for c in clusters]
    seen: dict[str, tuple[str, str]] = {}
    for subject, label, obj in s.gold(doc):
        p = s.predicate(label)
        seen.setdefault(fold(subject), (subject, (p["domain"] or ["Thing"])[0]))
        if p["range"] not in LITERALS:
            seen.setdefault(fold(obj), (obj, p["range"]))
    return list(seen.values())


def corruptions(s: Docset, doc: str, index: int, seed: int) -> list[Item]:
    """Every corruption of gold fact `index` that is not itself gold or any extractor's triple."""
    subject, label, obj = s.gold(doc)[index]
    p = s.predicate(label)
    literal = p["range"] in LITERALS
    entities = _entities(s, doc)
    kinds = {fold(n): t for n, t in entities}
    s_type = kinds.get(fold(subject), (p["domain"] or ["Thing"])[0])
    o_type = kinds.get(fold(obj), p["range"])
    rng = random.Random(f"{seed}:{s.dataset}:{doc}:{index}")
    out: list[tuple[str, Triple]] = []
    if not literal:
        for name, kind in entities:
            if kind == o_type and fold(name) not in (fold(subject), fold(obj)):
                out.append(("object_swap", (subject, label, name)))
        if label not in SYMMETRIC and fold(subject) != fold(obj):
            out.append(("subject_object_swap", (obj, label, subject)))
    for other, name in sorted(s.predicates.items()):
        q = s.ontology["predicates"][name]
        same = q["range"] == p["range"] and s_type in q["domain"]
        # Text2KGBench's literals are all strings: "cost" for "publication date" is noise.
        typed = not literal or (s.dataset == REDOCRED and p["range"] == "date")
        if other != label and same and typed and not _near(label, other):
            out.append(("relation_swap", (subject, other, obj)))
    if literal and (value := _changed(obj, s.texts[doc], rng)) is not None:
        out.append(("value_change", (subject, label, value)))
    written = {tuple(fold(x) for x in t) for t in s.written.get(doc, [])}
    items = [
        Item(
            s.dataset,
            s.name,
            doc,
            (subject, label, obj),
            "not_in_gold",
            gold=index,
            planted="polarity",
            denied=True,
        )
    ]
    for kind, t in out:
        if s.matched(doc, t) is None and tuple(fold(x) for x in t) not in written:
            items.append(Item(s.dataset, s.name, doc, t, "not_in_gold", gold=index, planted=kind))
    return items


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


@dataclass
class Draw:
    """What has been drawn so far, so nothing is shown twice."""

    claims: set[Any] = field(default_factory=set)
    golds: set[tuple[str, str, int]] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def take(self, item: Item) -> bool:
        gold = (item.dataset, item.doc, item.gold) if item.gold is not None else None
        if item.claim in self.claims or (gold is not None and gold in self.golds):
            return False
        self.claims.add(item.claim)
        if gold is not None:
            self.golds.add(gold)
        return True


def _shuffled(items: Iterable[Item], rng: random.Random) -> list[Item]:
    out = sorted(items, key=lambda i: (i.doc, i.triple, i.extractor or "", i.planted or ""))
    rng.shuffle(out)
    return out


def _fill(pools: list[list[Item]], quotas: list[int], draw: Draw) -> list[list[Item]]:
    """Each pool's quota, one item from each in turn; then any shortfall from the others.

    In turn, so pools that share candidates (one triple from two extractors, one
    gold fact open to two corruptions) share them out rather than the first
    pool taking all.
    """
    got: list[list[Item]] = [[] for _ in pools]

    def one(k: int) -> bool:
        while pools[k]:
            item = pools[k].pop(0)
            if draw.take(item):
                got[k].append(item)
                return True
        return False

    while any(len(got[k]) < quotas[k] and pools[k] for k in range(len(pools))):
        for k in range(len(pools)):
            if len(got[k]) < quotas[k]:
                one(k)
    short = sum(quotas) - sum(map(len, got))
    while short > 0 and any(pools):
        for k in range(len(pools)):
            if short > 0 and one(k):
                short -= 1
    return got


def draw_real(sets: Sequence[Docset], plan: Plan, seed: int, draw: Draw) -> list[Item]:
    """`match` and `not_in_gold` per extractor as evenly as the data allows, then `gold_only`."""
    rng = random.Random(f"{seed}:real")
    names = list(EXTRACTORS)
    pools: dict[tuple[str, str, str], list[Item]] = defaultdict(list)
    golds: dict[str, list[Item]] = defaultdict(list)
    for s in sets:
        for name, items in predictions(s).items():
            for item in items:
                pools[(s.dataset, item.status, name)].append(item)
        golds[s.dataset].extend(gold_only(s))
    for key in sorted(pools):
        pools[key] = _shuffled(pools[key], rng)
    out: list[Item] = []
    cell = 0
    for status in ("match", "not_in_gold"):
        for dataset in DATASETS:
            wanted = plan.wanted(status)
            # A third each; the odd item goes to each extractor in turn across the cells.
            quotas = [wanted // 3] * 3
            for k in range(wanted % 3):
                quotas[(cell + k) % 3] += 1
            cell += 1
            got = _fill([pools[(dataset, status, n)] for n in names], quotas, draw)
            for name, items, quota in zip(names, got, quotas, strict=True):
                if len(items) < quota:
                    draw.notes.append(
                        f"{dataset} {status}: {name} wrote only {len(items)} of {quota}; "
                        "the other extractors made up the rest"
                    )
            taken = sum(map(len, got))
            if taken < wanted:
                other = [pools[(d, status, n)] for d in DATASETS if d != dataset for n in names]
                got += _fill(other, [wanted - taken] + [0] * (len(other) - 1), draw)
                draw.notes.append(
                    f"{dataset} {status}: {taken} of {wanted}; the rest from the other dataset"
                )
            out.extend(i for g in got for i in g)
    unfound = {dataset: _shuffled(golds[dataset], rng) for dataset in DATASETS}
    for dataset in DATASETS:
        (chosen,) = _fill([unfound[dataset]], [plan.gold_only], draw)
        if len(chosen) < plan.gold_only:
            other = [unfound[d] for d in DATASETS if d != dataset]
            (extra,) = _fill(other, [plan.gold_only - len(chosen)], draw)
            draw.notes.append(
                f"{dataset} gold_only: {len(chosen)} of {plan.gold_only}; "
                f"{len(extra)} more from the other dataset"
            )
            chosen += extra
        out.extend(chosen)
    return out


def draw_planted(sets: Sequence[Docset], plan: Plan, seed: int, draw: Draw) -> list[Item]:
    """`plan.planted` per dataset, the kinds in turn, one per gold fact, none shown already."""
    rng = random.Random(f"{seed}:planted")
    pools: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for s in sets:
        for doc in sorted(s.rows):
            for index in range(len(s.gold(doc))):
                if (s.dataset, doc, index) in draw.golds:
                    continue
                for item in corruptions(s, doc, index, seed):
                    pools[(s.dataset, item.planted or "")].append(item)
    out: list[Item] = []
    for dataset in DATASETS:
        # The scarcest kind draws first, so a gold fact that allows several
        # corruptions goes to the kind with the fewest to choose from.
        kinds = sorted(KINDS, key=lambda k: (len(pools[(dataset, k)]), KINDS.index(k)))
        lists = [_shuffled(pools[(dataset, kind)], rng) for kind in kinds]
        each, extra = divmod(plan.planted, len(KINDS))
        quotas = [each + (KINDS.index(kind) < extra) for kind in kinds]
        got = _fill(lists, quotas, draw)
        for kind, items, quota in zip(kinds, got, quotas, strict=True):
            if len(items) < quota:
                draw.notes.append(
                    f"{dataset} planted {kind}: {len(items)} of {quota}; "
                    "the other kinds made up the rest"
                )
        out.extend(i for g in got for i in g)
    return out


def _allocate(sizes: dict[Any, int], total: int) -> dict[Any, int]:
    """`total` shared out in proportion to `sizes`, largest remainders first."""
    whole = sum(sizes.values())
    if not whole:
        return dict.fromkeys(sizes, 0)
    exact = {k: n * total / whole for k, n in sizes.items()}
    out = {k: int(v) for k, v in exact.items()}
    largest = sorted(exact, key=lambda k: (-(exact[k] - out[k]), str(k)))
    for k in largest[: total - sum(out.values())]:
        out[k] += 1
    return out


def _stratum(item: Item) -> tuple[str, str, str]:
    return (item.dataset, item.status, item.extractor or "-")


def split(real: Sequence[Item], plan: Plan, seed: int) -> dict[Item, str]:
    """dev or gate per item, whole documents at a time, `plan.dev` items in dev.

    A document is never in both: a prompt written while reading dev has then
    never seen a gate passage. Each dataset gives dev its share, and within it
    every stratum (gold status by extractor) as near its share as whole
    documents allow.
    """
    rng = random.Random(f"{seed}:split")
    out: dict[Item, str] = {}
    per_dataset = _allocate(Counter(i.dataset for i in real), plan.dev)
    for dataset in DATASETS:
        docs: dict[str, list[Item]] = defaultdict(list)
        for item in real:
            if item.dataset == dataset:
                docs[item.doc].append(item)
        order = sorted(docs)
        rng.shuffle(order)
        dev = _dev_docs(docs, order, per_dataset.get(dataset, 0))
        for doc in order:
            for item in docs[doc]:
                out[item] = "dev" if doc in dev else "gate"
    return out


def _dev_docs(docs: dict[str, list[Item]], order: list[str], target: int) -> list[str]:
    """Documents whose items come to `target`, each stratum as near its share as they allow.

    In `order`, a document is taken when it brings the strata nearer their
    shares; then any that still fits; then one swap settles the last few items,
    and documents of one size trade places while that helps. Taken in order
    rather than best first, so dev does not fill up with the documents that
    have the most items.
    """
    total = sum(map(len, docs.values()))
    counts = Counter(_stratum(i) for items in docs.values() for i in items)
    share = {k: n * target / max(1, total) for k, n in counts.items()}
    need = {doc: Counter(map(_stratum, items)) for doc, items in docs.items()}
    dev: list[str] = []
    have: Counter[tuple[str, str, str]] = Counter()

    def off(counts: Counter[tuple[str, str, str]]) -> float:
        return sum((counts[k] - share[k]) ** 2 for k in share)

    for closer in (True, False):
        for doc in order:
            fits = sum(have.values()) + len(docs[doc]) <= target
            if doc not in dev and fits and (off(have + need[doc]) < off(have) or not closer):
                dev.append(doc)
                have.update(need[doc])
    gap = target - sum(have.values())
    gate = [d for d in order if d not in dev]
    swap = next(((a, b) for a in dev for b in gate if len(docs[b]) - len(docs[a]) == gap), None)
    if gap and swap is not None:
        dev[dev.index(swap[0])] = swap[1]
        gate[gate.index(swap[1])] = swap[0]
        have = have - need[swap[0]] + need[swap[1]]
    # Then documents of one size trade places while that brings the strata nearer.
    better = True
    while better:
        better = False
        for a, b in ((a, b) for a in dev for b in gate if len(docs[a]) == len(docs[b])):
            moved = have - need[a] + need[b]
            if off(moved) < off(have) - 1e-9:
                dev[dev.index(a)], gate[gate.index(b)], have = b, a, moved
                better = True
                break
    return dev


def self_agreement(real: Sequence[Item], plan: Plan, seed: int) -> set[Item]:
    """`plan.self_agreement` real items, in proportion to each stratum."""
    rng = random.Random(f"{seed}:self")
    strata: dict[tuple[str, str, str], list[Item]] = defaultdict(list)
    for item in real:
        strata[_stratum(item)].append(item)
    picks = _allocate({k: len(v) for k, v in strata.items()}, plan.self_agreement)
    out: set[Item] = set()
    for key in sorted(strata):
        out.update(_shuffled(strata[key], rng)[: picks[key]])
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _cluster(raw: dict[str, Any], name: str) -> int | None:
    """The Re-DocRED entity `name` names: a mention spelled the same, else the longest overlap."""
    folded = fold(name)
    names = [{fold(m["name"]) for m in cluster} for cluster in raw["vertexSet"]]
    exact = next((i for i, n in enumerate(names) if folded in n), None)
    if exact is not None or not folded:
        return exact
    best, length = None, 0
    for i, cluster in enumerate(names):
        for n in cluster:
            if len(n) >= 3 and (n in folded or folded in n) and len(n) > length:
                best, length = i, len(n)
    return best


def _type(s: Docset, doc: str, name: str, allowed: Sequence[str]) -> str:
    """The type an entity is shown with: Re-DocRED's own if the doc names it, else the schema's."""
    if s.dataset == REDOCRED:
        raw = s.raw[doc]
        cluster = _cluster(raw, name)
        if cluster is not None:
            kind = redocred.TYPES[raw["vertexSet"][cluster][0]["type"]]
            if kind not in LITERALS:
                return kind
        if len(allowed) > 1:
            return "Misc" if "Misc" in allowed else allowed[0]
    return allowed[0] if allowed else "Thing"


def _fact(s: Docset, item: Item, fact_id: str, text: str) -> dict[str, Any]:
    subject, label, obj = item.triple
    p = s.predicate(label)
    s_type = _type(s, item.doc, subject, p["domain"])
    fact: dict[str, Any] = {
        "id": fact_id,
        "subject": {"key": entity_key(s_type, subject), "type": s_type, "label": subject},
        "predicate": s.predicates[label],
    }
    if p["range"] in LITERALS:
        fact["object_value"] = obj
    else:
        o_type = _type(s, item.doc, obj, [p["range"]])
        fact["object_entity"] = {"key": entity_key(o_type, obj), "type": o_type, "label": obj}
    if item.denied:
        fact["polarity"] = "denied"
    # Nobody cited a span: the fact is grounded against the whole text it is shown with.
    span = {"doc_id": item.doc, "start": 0, "end": len(text)}
    fact["evidence"] = [{"doc_id": item.doc, "span": span, "span_origin": "context"}]
    return fact


def _focus(s: Docset, item: Item) -> list[int]:
    """The Re-DocRED sentences an item is about: its gold evidence, else where its names are."""
    raw = s.raw[item.doc]
    if item.gold is not None:
        label = raw["labels"][item.gold]
        if label["evidence"]:
            return sorted(label["evidence"])
        clusters = [label["h"], label["t"]]
    else:
        found = (_cluster(raw, item.triple[0]), _cluster(raw, item.triple[2]))
        clusters = [c for c in found if c is not None]
    where = [{m["sent_id"] for m in raw["vertexSet"][c]} for c in clusters]
    if not where:
        return []
    # Where both are named; else where the less mentioned one is, since the
    # other is usually the passage's own subject, named everywhere.
    return sorted(set.intersection(*where) or min(where, key=len))


def text_of(s: Docset, item: Item, plan: Plan) -> tuple[str, list[int] | None]:
    """The text the item is shown with, and the sentences kept when it was cut."""
    full = s.texts[item.doc]
    if s.dataset != REDOCRED or len(full.split()) <= plan.trim_words:
        return full, None
    sentences = [redocred.detokenize([tokens]) for tokens in s.raw[item.doc]["sents"]]
    focus = _focus(s, item)
    if not focus:
        return full, None
    keep = sorted({j for f in focus for j in (f - 1, f, f + 1) if 0 <= j < len(sentences)})
    parts = ["…"] if keep[0] > 0 else []
    for at, j in enumerate(keep):
        if at and j != keep[at - 1] + 1:
            parts.append("…")
        parts.append(sentences[j])
    if keep[-1] < len(sentences) - 1:
        parts.append("…")
    return " ".join(parts), keep


@dataclass
class Made:
    """What `make` wrote, and the counts the README and the PR quote."""

    rows: list[dict[str, Any]]
    private: list[dict[str, Any]]
    notes: list[str]

    def table(self) -> str:
        """Counts by dataset, gold status and extractor; then splits; then the planted kinds."""
        extractors = [*EXTRACTORS, None]
        heads = ["openodke", "LGT", "neo4j-graphrag", "none"]
        lines = [
            "| Dataset | Gold status | " + " | ".join(heads) + " | Total | dev | gate |",
            "|---|---|" + "---:|" * (len(heads) + 3),
        ]
        real = [p for p in self.private if p["planted"] is None]
        for dataset in DATASETS:
            for status in STATUSES:
                rows = [p for p in real if p["dataset"] == dataset and p["gold_status"] == status]
                counts = [sum(p["extractor"] == e for p in rows) for e in extractors]
                dev = sum(p["split"] == "dev" for p in rows)
                cells = [dataset, status, *map(str, counts), str(len(rows)), str(dev)]
                lines.append("| " + " | ".join([*cells, str(len(rows) - dev)]) + " |")
        dev = sum(p["split"] == "dev" for p in real)
        totals = [str(sum(p["extractor"] == e for p in real)) for e in extractors]
        cells = ["**all**", "", *totals, str(len(real)), str(dev), str(len(real) - dev)]
        lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "| Planted | " + " | ".join(DATASETS) + " | Total |", "|---|---:|---:|---:|"]
        planted = [p for p in self.private if p["planted"] is not None]
        for kind in KINDS:
            counts = [
                sum(p["planted"] == kind and p["dataset"] == d for p in planted) for d in DATASETS
            ]
            lines.append(f"| {kind} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
        counts = [sum(p["dataset"] == d for p in planted) for d in DATASETS]
        lines.append("| **all** | " + " | ".join(map(str, counts)) + f" | {len(planted)} |")
        return "\n".join(lines)


def make(sets: Sequence[Docset], plan: Plan | None = None, seed: int = SEED) -> Made:
    """Draw, split and shuffle the items; nothing is written."""
    plan = plan or Plan()
    draw = Draw()
    real = draw_real(sets, plan, seed, draw)
    planted = draw_planted(sets, plan, seed, draw)
    splits = split(real, plan, seed)
    again = self_agreement(real, plan, seed)
    by_name = {s.name: s for s in sets}
    everything = sorted([*real, *planted], key=lambda i: i.fact_id(seed))
    random.Random(f"{seed}:order").shuffle(everything)
    width = max(4, len(str(len(everything))))
    rows, private = [], []
    for n, item in enumerate(everything, start=1):
        s = by_name[item.name]
        fact_id = item.fact_id(seed)
        text, kept = text_of(s, item, plan)
        row = {"text": text, "fact": _fact(s, item, fact_id, text), "doc_id": item.doc}
        GroundingItem.model_validate(row)
        rows.append(row)
        gold = s.gold(item.doc, as_written=True)[item.gold] if item.gold is not None else None
        private.append(
            {
                "id": f"G-{n:0{width}d}",
                "fact_id": fact_id,
                "dataset": item.dataset,
                "set": item.name,
                "doc": item.doc,
                "triple": list(item.triple),
                "denied": item.denied,
                "extractor": item.extractor,
                "also_written_by": list(item.also),
                "gold_status": item.status,
                "scored": item.scored,
                "gold": list(gold) if gold is not None else None,
                "planted": item.planted,
                "expected": NOT_SUPPORTED if item.planted else None,
                "split": splits.get(item, "planted"),
                "self_agreement": item in again,
                "trimmed": kept is not None,
                "sentences": kept,
            }
        )
    return Made(rows, private, draw.notes)


def write(made: Made, out: Path) -> list[Path]:
    """The four files under `out`, byte-identical for the same `made`."""
    out.mkdir(parents=True, exist_ok=True)

    def lines(rows: Iterable[dict[str, Any]]) -> str:
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    gate = [r for r, p in zip(made.rows, made.private, strict=True) if p["split"] == "gate"]
    again = [p["id"] for p in made.private if p["self_agreement"]]
    files = {
        "items.jsonl": lines(made.rows),
        "items.private.jsonl": lines(made.private),
        "gate.jsonl": lines(gate),
        "selfagreement.ids": "".join(f"{i}\n" for i in again),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return [out / name for name in files]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("cmp", type=Path, help="the comparison: t2k/ont_* and redocred/")
    parser.add_argument("redocred_raw", type=Path, help="Re-DocRED's test_revised.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "G")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    made = make(load_all(args.cmp, args.redocred_raw), seed=args.seed)
    for path in write(made, args.out):
        print(f"wrote {path}", file=sys.stderr)
    print(made.table())
    for note in made.notes:
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
