"""T-REx (Elsahar et al., LREC 2018): one Wikidata fact, told by several abstracts.

Provisional (DECISIONS #48): a benchmark adapter, as `openodke.eval.datasets`
says. It may change in a minor release, with a CHANGELOG line.

T-REx aligns Wikidata triples to sentences of Wikipedia abstracts, and a fact
about two entities can be aligned in more than one abstract: "Bavaria is in
Germany" in Bavaria's own abstract and in Abensberg's. So it gives what
Text2KGBench and Re-DocRED cannot: facts with a known number of independent
sources, which is what corroboration counts (#117). Its entities carry Wikidata
ids, which is gold identity for the resolver across documents.
<https://hadyelsahar.io/t-rex/>, CC BY-SA 4.0. `fetch` downloads the official
10,000-abstract sample (the full set's first file, 21 MB) and the English labels
of the properties it uses from Wikidata's API; nothing is copied into this
package.

**Which alignments are gold.** T-REx has three aligners, and they differ in
what a sentence has to say:

- `NoSubject-Triple-aligner`: the subject is the abstract's own entity and the
  object is named in the sentence;
- `SPOAligner`: both entities and a lexicalisation of the property are in the
  sentence;
- `Simple-Aligner`: both entities are named in the sentence, the property
  anywhere. Most of its multi-abstract alignments are co-mentions: Indonesia
  and the United States in a list of countries, aligned to "diplomatic
  relation".

So the gold, and each fact's source count, come from the first two by default
(`aligners`). Every alignment of all three is still a Wikidata triple, so all of
them count as known facts when precision is checked.

**The set.** `prepare` picks documents so that multi-source facts are well
represented: facts aligned in three or more abstracts first, then two, in a
seeded order, each with up to three of its abstracts, no relation more than
`per_relation` times among the facts it picks, until `documents` abstracts are
in. Each gold fact's number of sources is then counted inside the set. Single-
source facts come with every abstract, as controls. Abstracts longer than
`max_chars` are left out, which keeps a run cheap.

**The ontology** is the Wikidata properties the set's gold uses, one entity
type (`Entity`) for every end. One type is deliberate: the resolver matches
within a type (#24), and a fact told by two abstracts must not stay two claims
because the extractor typed France as a Country once and a Location once.

**Scoring.** A predicted `(subject, relation, object)` names an entity when its
name, case and punctuation ignored, is a surface form T-REx linked to that
Wikidata id anywhere in the file, or the title of its abstract. Then:

- **recall**: each document's gold facts found in it, each once, micro-averaged
  over documents, and split by the fact's number of sources;
- **precision**: a predicted triple is right when it is one of its document's
  gold facts, as Re-DocRED counts it (`precision`), and factually right when
  T-REx aligns that Wikidata triple anywhere in the file (`precision_factual`).
  That is closed-world on a 2017 slice of Wikidata, so it is a lower bound: a
  true fact T-REx never aligned counts as wrong, and so does a fact whose
  entities T-REx never linked (`checkable` says how many it could check).

`corroboration` reads the graph rather than the documents: one edge per
distinct claim, each with the support the corroborator gave it, precision by
support, and recall of the set's gold facts by their number of sources, with
corroboration off and on. `links` scores the resolver's links against the ids.
"""

from __future__ import annotations

import io
import json
import math
import random
import re
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import IO, Any

from openodke.eval.ablation import AblationRun, ablate
from openodke.eval.bootstrap import Range
from openodke.eval.cost import CallRecord
from openodke.eval.datasets import _common
from openodke.eval.datasets._common import (
    Opener,
    Triple,
    check_out,
    download,
    read_jsonl,
    run_config,
    snake,
    write_documents,
    write_json,
    write_jsonl,
)
from openodke.eval.eval_report import (
    Conformance,
    Counts,
    Dataset,
    EvalReport,
    Hallucination,
    Row,
    performance,
    spend,
)
from openodke.eval.report import Metric, StageReport
from openodke.types import EntityLink, Fact, LinkKind

NAME = "trex"
SAMPLE_URL = "https://www.dropbox.com/s/jlszpvnwklmeysl/T-REx_sample.zip?dl=1"
SAMPLE = "T-REx_sample.zip"
PROPERTIES = "properties.json"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "openodke-bench/1.0 (https://github.com/deepskandpal/odke)"
ENTITY = "http://www.wikidata.org/entity/"
CHECKED = ("NoSubject-Triple-aligner", "SPOAligner")
ALIGNERS = (*CHECKED, "Simple-Aligner")
# The entity annotations that name an entity; coreference ("He") and dates do not.
LINKER = "Wikidata_Spotlight_Entity_Linker"
TYPE = "Entity"
# A gold fact's sources, bucketed as the report splits them.
BUCKETS = ("1", "2", "3+")

Fact3 = tuple[str, str, str]


def _open(url: str) -> Any:
    """`url` opened with a generic User-Agent, which Wikimedia's API policy asks for."""
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}))


def fetch(dest: str | Path, *, opener: Opener | None = None) -> Path:
    """The T-REx sample and its properties' labels, under `dest`.

    The sample is the full set's first file: 10,000 abstracts in T-REx's JSON.
    The labels are English, from Wikidata's API, 50 properties a request.
    """
    root = Path(dest)
    open_url = opener if opener is not None else _open
    download(SAMPLE_URL, root / SAMPLE, opener=open_url)
    target = root / PROPERTIES
    if not target.exists():
        used = sorted(
            {_tail(t["predicate"]["uri"]) for doc in abstracts(root) for t in doc["triples"]},
            key=_number,
        )
        write_json(target, properties(used, opener=open_url))
    return root


def properties(ids: Sequence[str], *, opener: Opener | None = None) -> dict[str, dict[str, str]]:
    """Each property id -> its English `label` and `description`, from Wikidata."""
    open_url = opener if opener is not None else _open
    out: dict[str, dict[str, str]] = {}
    for at in range(0, len(ids), 50):
        query = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(ids[at : at + 50]),
                "props": "labels|descriptions",
                "languages": "en",
                "format": "json",
            }
        )
        with open_url(f"{WIKIDATA_API}?{query}") as response:
            entities = json.loads(response.read().decode("utf-8")).get("entities", {})
        for pid, entity in entities.items():
            label = entity.get("labels", {}).get("en", {}).get("value")
            if label:
                description = entity.get("descriptions", {}).get("en", {}).get("value", "")
                out[pid] = {"label": label, "description": description}
    return out


# --------------------------------------------------------------------------- #
# reading T-REx
# --------------------------------------------------------------------------- #


def objects(stream: IO[str], size: int = 1 << 20) -> Iterator[Any]:
    """Each object of a JSON array, read a megabyte at a time rather than all at once.

    A T-REx file is one array of documents, 148 MB for the sample; parsed whole
    it would take over a gigabyte.
    """
    decoder = json.JSONDecoder()
    buffer, at, done = "", 0, False
    while True:
        while at < len(buffer) and buffer[at] in " \t\r\n,[]":
            at += 1
        if at < len(buffer):
            try:
                found, at = decoder.raw_decode(buffer, at)
            except json.JSONDecodeError:
                if done:
                    raise
            else:
                yield found
                continue
        if done:
            return
        chunk = stream.read(size)
        done = not chunk
        buffer, at = buffer[at:] + chunk, 0


def abstracts(root: str | Path) -> Iterator[dict[str, Any]]:
    """Every T-REx document under `root`: in its `.zip` files' `.json` members, then its `.json`."""
    base = Path(root)
    for path in sorted(base.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
                with archive.open(name) as raw:
                    yield from objects(io.TextIOWrapper(raw, encoding="utf-8"))
    for path in sorted(base.glob("*.json")):
        if path.name != PROPERTIES:
            with path.open(encoding="utf-8") as text:
                yield from objects(text)


def _tail(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _number(pid: str) -> int:
    return int(pid[1:]) if pid[1:].isdigit() else 0


def alignments(doc: Mapping[str, Any], aligners: Iterable[str] = ALIGNERS) -> set[Fact3]:
    """The document's aligned `(subject, property, object)` ids, entities at both ends.

    A literal object (a date, a number) is left out, and so is a triple whose
    two ends are one entity.
    """
    wanted = set(aligners)
    out: set[Fact3] = set()
    for triple in doc["triples"]:
        obj = triple["object"]["uri"] or ""
        if triple.get("annotator") not in wanted or not obj.startswith(ENTITY):
            continue
        subject, pid, target = (
            _tail(triple["subject"]["uri"]),
            _tail(triple["predicate"]["uri"]),
            _tail(obj),
        )
        if subject != target:
            out.add((subject, pid, target))
    return out


def share(root: str | Path) -> dict[str, dict[str, Any]]:
    """How many distinct facts are aligned in one, two, and three or more abstracts.

    Counted twice: with every aligner, as T-REx is published, and with the two
    that check what the sentence says (`CHECKED`).
    """
    seen = {"all": Counter[Fact3](), "checked": Counter[Fact3]()}
    for doc in abstracts(root):
        _tally(seen, doc)
    return _shares(seen)


def _tally(seen: Mapping[str, Counter[Fact3]], doc: Mapping[str, Any]) -> None:
    # Each alignment set is a set, so a document adds one to each fact it aligns.
    seen["all"].update(alignments(doc))
    seen["checked"].update(alignments(doc, CHECKED))


def _shares(seen: Mapping[str, Counter[Fact3]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, found in seen.items():
        sizes = Counter(found.values())
        by = {"1": sizes[1], "2": sizes[2], "3+": sum(c for n, c in sizes.items() if n >= 3)}
        total, multi = sum(by.values()), by["2"] + by["3+"]
        out[name] = {
            "facts": total,
            **by,
            "multi_source": multi,
            "multi_share": multi / total if total else 0.0,
        }
    return out


def _bucket(n: int) -> str:
    return BUCKETS[min(max(n, 1), 3) - 1]


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #


def select(
    sources: Mapping[Fact3, set[str]],
    lengths: Mapping[str, int],
    *,
    documents: int,
    seed: int = 0,
    max_chars: int = 2500,
    per_relation: int = 6,
) -> list[str]:
    """The abstracts a set is built from, in the order they were picked.

    Facts with three or more short abstracts first, then two, each group in a
    seeded order; each fact brings up to three of its abstracts, at most
    `per_relation` facts of one relation are picked, and a fact that would
    take the set past `documents` is passed over.
    """
    rng = random.Random(seed)
    short = {
        fact: sorted(d for d in docs if lengths.get(d, max_chars + 1) <= max_chars)
        for fact, docs in sources.items()
    }
    multi = sorted(fact for fact, docs in short.items() if len(docs) >= 2)
    rng.shuffle(multi)
    multi.sort(key=lambda fact: len(short[fact]) < 3)
    chosen: list[str] = []
    taken: Counter[str] = Counter()
    for fact in multi:
        if taken[fact[1]] >= per_relation:
            continue
        docs = list(short[fact])
        rng.shuffle(docs)
        new = [d for d in docs[:3] if d not in chosen]
        if len(chosen) + len(new) > documents:
            continue
        chosen += new
        taken[fact[1]] += 1
        if len(chosen) == documents:
            break
    return chosen


def prepare(
    root: str | Path,
    out: str | Path,
    *,
    documents: int = 80,
    limit: int | None = None,
    seed: int = 0,
    max_chars: int = 2500,
    per_relation: int = 6,
    aligners: Sequence[str] = CHECKED,
    extract_model: str | None = None,
    ground_model: str | None = None,
    paper: bool = False,
    max_tokens: int = 16000,
    judge: bool = False,
) -> Path:
    """A runnable directory: `docs/`, `ontology.json`, `gold.jsonl`, `dataset.json`, `odke.json`.

    Two passes over the file and no model. `limit` keeps the first N documents
    picked, for a pilot, and counts sources among those. The run config adds
    the native resolver, so its links can be scored against the ids, and
    `judge` its pair judge (#34), which asks the ground model.
    """
    base = Path(root)
    names_file = base / PROPERTIES
    if not any(base.glob("*.zip")) and not any(p != names_file for p in base.glob("*.json")):
        raise FileNotFoundError(f"no T-REx files under {base} — run fetch first")
    unknown = sorted(set(aligners) - set(ALIGNERS))
    if unknown:
        raise ValueError(f"unknown aligner {unknown[0]!r}; one of {', '.join(ALIGNERS)}")
    labels = json.loads(names_file.read_text("utf-8")) if names_file.exists() else {}

    # Pass 1: each abstract's length and checked alignments, and each fact's abstracts.
    lengths: dict[str, int] = {}
    gold: dict[str, set[Fact3]] = {}
    sources: dict[Fact3, set[str]] = defaultdict(set)
    seen = {"all": Counter[Fact3](), "checked": Counter[Fact3]()}
    for doc in abstracts(base):
        qid = _tail(doc["docid"])
        lengths[qid] = len(doc["text"])
        gold[qid] = alignments(doc, aligners)
        for fact in gold[qid]:
            sources[fact].add(qid)
        _tally(seen, doc)
    chosen = select(
        sources,
        lengths,
        documents=documents,
        seed=seed,
        max_chars=max_chars,
        per_relation=per_relation,
    )
    if limit:
        chosen = chosen[:limit]
    kept = set(chosen)
    count = {f: len(docs & kept) for f, docs in sources.items() if docs & kept}
    used = sorted({f[1] for qid in chosen for f in gold[qid]}, key=_number)
    relation = _relation_names(used, labels)

    # Pass 2: the chosen abstracts' text and entities, every name of those
    # entities anywhere in the file, and every triple T-REx aligns between them.
    rows: dict[str, dict[str, Any]] = {}
    for doc in abstracts(base):
        qid = _tail(doc["docid"])
        if qid in kept:
            mentioned = {qid} | {
                _tail(e["uri"]) for e in doc["entities"] if e.get("annotator") == LINKER
            }
            rows[qid] = {"title": doc["title"], "text": doc["text"], "mentioned": mentioned}
    entities = {q for row in rows.values() for q in row["mentioned"]}
    entities |= {q for qid in chosen for f in gold[qid] for q in (f[0], f[2])}
    names: dict[str, set[str]] = defaultdict(set)
    known: set[Fact3] = set()
    for doc in abstracts(base):
        qid = _tail(doc["docid"])
        if qid in entities and doc.get("title"):
            names[qid].add(doc["title"])
        for e in doc["entities"]:
            linked = e.get("annotator") == LINKER and e.get("surfaceform")
            if linked and (q := _tail(e["uri"])) in entities:
                names[q].add(e["surfaceform"])
        known |= {
            f for f in alignments(doc) if f[0] in entities and f[2] in entities and f[1] in relation
        }

    target = Path(out)
    check_out(target)
    gold_rows = []
    for qid in chosen:
        row = rows[qid]
        facts = sorted(gold[qid], key=lambda f: (_number(f[1]), f))
        ends = row["mentioned"] | {q for f in facts for q in (f[0], f[2])}
        gold_rows.append(
            {
                "id": qid,
                "title": row["title"],
                "text": row["text"],
                "entities": {q: sorted(names[q]) for q in sorted(ends, key=_number)},
                "facts": [[s, relation[p]["label"], o] for s, p, o in facts],
                "sources": [count[f] for f in facts],
                "known": [
                    [s, relation[p]["label"], o]
                    for s, p, o in sorted(known)
                    if s in ends and o in ends
                ],
            }
        )
    write_json(target / "ontology.json", to_ontology(relation, gold_rows))
    write_documents(target, ((row["id"], row["text"]) for row in gold_rows))
    write_jsonl(target / "gold.jsonl", gold_rows)
    in_set = Counter(_bucket(count[f]) for qid in chosen for f in gold[qid])
    write_json(
        target / "dataset.json",
        {
            "dataset": NAME,
            "documents": len(gold_rows),
            "aligners": list(aligners),
            "selection": {
                "documents": documents,
                "seed": seed,
                "max_chars": max_chars,
                "per_relation": per_relation,
                "limit": limit,
            },
            # Each distinct gold fact once, by its number of sources in the set.
            "gold_facts": {b: sum(1 for f in count if _bucket(count[f]) == b) for b in BUCKETS},
            # Each (document, fact) pair: what recall is computed over.
            "gold_rows": {b: in_set[b] for b in BUCKETS},
            # The whole file's facts, by the abstracts they are aligned in.
            "share": _shares(seen),
            "relation_labels": {spec["name"]: spec["label"] for spec in relation.values()},
        },
    )
    config = run_config(
        extract_model=extract_model,
        ground_model=ground_model,
        paper=paper,
        relations=len(relation),
        max_tokens=max_tokens,
    )
    config["stages"]["resolver"] = {"use": "native", **({"judge": True} if judge else {})}
    write_json(target / "odke.json", config)
    return target


def _relation_names(
    pids: Sequence[str], labels: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, str]]:
    """Each property id -> its predicate `name`, `label` and `description`.

    The name is the label in snake case; two properties whose labels fold to
    one name keep them apart with the id.
    """
    out: dict[str, dict[str, str]] = {}
    taken: set[str] = set()
    for pid in pids:
        found = labels.get(pid, {})
        label = found.get("label") or pid
        name = snake(label)
        if name in taken:
            name = f"{name}_{pid.lower()}"
        taken.add(name)
        out[pid] = {"name": name, "label": label, "description": found.get("description", "")}
    return out


def to_ontology(
    relation: Mapping[str, Mapping[str, str]], gold: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The set's properties, over one `Entity` type, ranked by how often the gold uses them."""
    used = Counter(r for row in gold for _, r, _ in row["facts"])
    most = max(used.values(), default=0) or 1
    predicates: dict[str, dict[str, Any]] = {}
    for pid, spec in relation.items():
        seen = used[spec["label"]]
        about = f"{spec['label']} ({pid})"
        predicates[spec["name"]] = {
            "description": f"{about}: {spec['description']}" if spec["description"] else about,
            "domain": [TYPE],
            "range": TYPE,
            "aliases": [spec["label"]],
            "importance": round(0.05 + 0.95 * math.log1p(seen) / math.log1p(most), 4),
        }
    return {
        "name": "trex",
        "version": "t-rex-sample",
        "types": {TYPE: {"description": "Anything with a Wikidata item"}},
        "predicates": predicates,
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


class Gold:
    """A prepared set's gold, read once: names to ids, gold facts, known facts, sources.

    `relations` are the set's relation labels, for conformance; empty, every
    relation conforms.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], relations: Iterable[str] = ()) -> None:
        self.rows = list(rows)
        self.relations = {_norm(r) for r in relations}
        self.ids: dict[str, set[str]] = defaultdict(set)
        self.sources: dict[Fact3, int] = {}
        self.known: set[Fact3] = set()
        # Each abstract's own entities, by its id.
        self.mentioned: dict[str, set[str]] = {}
        for row in self.rows:
            self.mentioned[row["id"]] = set(row["entities"])
            for qid, names in row["entities"].items():
                for name in names:
                    if key := _norm(name):
                        self.ids[key].add(qid)
            for (s, r, o), n in zip(row["facts"], row["sources"], strict=True):
                self.sources[(s, _norm(r), o)] = n
            self.known |= {(s, _norm(r), o) for s, r, o in row["known"]}
        self.known |= set(self.sources)

    def of(self, name: str, within: str | None = None) -> set[str]:
        """The ids a name may stand for; empty when T-REx never linked it.

        `within` names an abstract: an ambiguous name ("India": the republic,
        British India, ...) then stands for the ids that abstract mentions, when
        it mentions any of them.
        """
        ids = self.ids.get(_norm(name), set())
        if within is not None and len(ids) > 1:
            return (ids & self.mentioned.get(within, set())) or ids
        return ids

    def candidates(self, triple: Triple) -> list[Fact3]:
        """Every id triple a predicted triple could stand for."""
        subject, relation, obj = triple
        rel = _norm(relation)
        return [(s, rel, o) for s in sorted(self.of(subject)) for o in sorted(self.of(obj))]


def documents(
    gold: Gold | Sequence[Mapping[str, Any]], predicted: Mapping[str, Sequence[Triple]]
) -> list[dict[str, Any]]:
    """Each document's counts before they are pooled: what `aggregate` sums.

    `tp` is the document's gold facts found, each once, also by their number of
    sources (`tp_2`, `gold_2`, ...); `correct` the predicted triples that are a
    known Wikidata fact; `checkable` those whose two ends name an entity T-REx
    linked; `conformant` those whose relation is the set's; `hallucinated`
    those with an end nowhere in the abstract, or a relation outside the set.
    """
    read = gold if isinstance(gold, Gold) else Gold(gold)
    out = []
    for doc in read.rows:
        facts = [(s, _norm(r), o) for s, r, o in doc["facts"]]
        bucket = {f: _bucket(n) for f, n in zip(facts, doc["sources"], strict=True)}
        text = _norm(doc["text"])
        triples = list(dict.fromkeys(predicted.get(doc["id"], ())))
        found: set[Fact3] = set()
        unit: dict[str, Any] = {"id": doc["id"], "predicted": len(triples), "gold": len(bucket)}
        counts = Counter[str]()
        for subject, relation, obj in triples:
            options = read.candidates((subject, relation, obj))
            fits = not read.relations or _norm(relation) in read.relations
            counts["checkable"] += bool(read.of(subject) and read.of(obj))
            counts["correct"] += any(o in read.known for o in options)
            counts["conformant"] += fits
            counts["hallucinated"] += (
                _norm(subject) not in text or _norm(obj) not in text or not fits
            )
            match = next((o for o in options if o in bucket and o not in found), None)
            if match is not None:
                found.add(match)
        unit |= {k: counts[k] for k in ("correct", "checkable", "conformant", "hallucinated")}
        unit["tp"] = len(found)
        for b in BUCKETS:
            unit[f"gold_{b}"] = sum(1 for f in bucket.values() if f == b)
            unit[f"tp_{b}"] = sum(1 for f in found if bucket[f] == b)
        out.append(unit)
    return out


def aggregate(units: Sequence[Mapping[str, Any]]) -> dict[str, Metric]:
    """The pooled metrics: micro P/R/F1 on the gold, factual precision, recall by sources."""
    total = Counter[str]()
    for unit in units:
        total.update({k: v for k, v in unit.items() if isinstance(v, int)})
    tp, n_pred, n_gold = total["tp"], total["predicted"], total["gold"]
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out: dict[str, Metric] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_factual": _rate(total["correct"], n_pred),
        "checkable": _rate(total["checkable"], n_pred),
        "onto_conf": total["conformant"] / n_pred if n_pred else 1.0,
        "triples": n_pred,
        "gold_triples": n_gold,
        "hallucinated_triples": total["hallucinated"],
    }
    for b in BUCKETS:
        out[f"recall_{b}"] = _rate(total[f"tp_{b}"], total[f"gold_{b}"])
    return out


def score(
    gold: Sequence[Mapping[str, Any]], predicted: Mapping[str, Sequence[Triple]]
) -> dict[str, Metric]:
    """`aggregate` over `documents`, for triples from any system."""
    return aggregate(documents(gold, predicted))


HALLUCINATION = (
    "T-REx, openodke's scoring: the subject or object, case and punctuation ignored, is "
    "nowhere in the abstract, or the relation is not one of the set's"
)


def row(
    name: str,
    units: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Metric],
    ranges: Mapping[str, Range],
    calls: Sequence[CallRecord] | None,
) -> Row:
    """One configuration's eval report row, counted as Re-DocRED's are."""
    tp, n_pred, n_gold, conformant, hallucinated = (
        sum(u[k] for u in units) for k in ("tp", "predicted", "gold", "conformant", "hallucinated")
    )
    return Row(
        name=name,
        performance=performance(metrics, ranges),
        counts=Counts(
            hits=tp,
            over_extraction=n_pred - tp,
            under_extraction=n_gold - tp,
            predicted=n_pred,
            gold=n_gold,
            documents=len(units),
        ),
        conformance=Conformance(
            rate=float(metrics["onto_conf"] or 0.0),
            conformant=conformant,
            facts=n_pred,
            checks=("predicate",),
        ),
        hallucination=Hallucination(
            definition=HALLUCINATION,
            hallucinated=hallucinated,
            facts=n_pred,
            rate=_rate(hallucinated, n_pred),
        ),
        **spend(name, calls),
    )


# --------------------------------------------------------------------------- #
# the graph: corroboration off and on, and the resolver's links
# --------------------------------------------------------------------------- #


def _edge(fact: Fact, labels: Mapping[str, str]) -> Triple:
    obj = (
        (fact.object_entity.label or fact.object_entity.key)
        if fact.object_entity is not None
        else str(fact.object_value)
    )
    return (fact.subject.label or fact.subject.key, labels.get(fact.predicate, fact.predicate), obj)


def corroboration(
    gold: Gold,
    gated: Sequence[Fact],
    corroborated: Sequence[Fact],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """The graph each row writes, one edge per distinct claim, scored against the set.

    Off is "+ grounding": every claim any document made is an edge, with no
    count of who else made it. On is "+ corroboration": the same claims merged,
    each with the independent sources the corroborator counted; `on_two_or_more`
    keeps only the edges two or more sources back, which is what ranking on the
    score does here, since the score rises with support alone once every kept
    fact is supported. An edge is right when it is a known Wikidata fact
    (`precision_factual`), and in the gold when it is one of the set's gold
    facts. Recall is over the set's distinct gold facts, by their number of
    sources. `counted` is, for each bucket of gold facts found on, how many the
    corroborator gave two or more sources, and how many two or more abstracts
    state under any name (`two_or_more_pooled`): the support a resolver that
    knew the Wikidata ids would have counted, since each abstract is a source.
    """
    support: dict[Triple, int] = {}
    cited: dict[Triple, set[str]] = defaultdict(set)
    for fact in corroborated:
        edge = _edge(fact, labels)
        support[edge] = max(support.get(edge, 0), fact.support)
        cited[edge] |= {e.doc_id for e in fact.evidence}
    off = {_edge(f, labels) for f in gated}

    def scored(edges: Iterable[Triple]) -> dict[str, Any]:
        edges = list(edges)
        factual = in_gold = 0
        found: set[Fact3] = set()
        for edge in edges:
            options = gold.candidates(edge)
            factual += any(o in gold.known for o in options)
            hits = [o for o in options if o in gold.sources]
            in_gold += bool(hits)
            found.update(hits)
        recall: dict[str, Any] = {}
        for b in BUCKETS:
            facts = [f for f, n in gold.sources.items() if _bucket(n) == b]
            hit = sum(1 for f in facts if f in found)
            recall[b] = {"found": hit, "gold": len(facts), "recall": _rate(hit, len(facts))}
        return {
            "edges": len(edges),
            "factual": factual,
            "precision_factual": _rate(factual, len(edges)),
            "in_gold": in_gold,
            "precision_gold": _rate(in_gold, len(edges)),
            "recall": recall,
        }

    # Each gold fact found on the graph: the most sources one edge naming it
    # has, and the abstracts every edge naming it cites, pooled, which is what
    # a resolver that knew the ids would have counted.
    most: dict[Fact3, int] = {}
    pooled: dict[Fact3, set[str]] = defaultdict(set)
    for edge, n in support.items():
        for option in gold.candidates(edge):
            if option in gold.sources:
                most[option] = max(most.get(option, 0), n)
                pooled[option] |= cited[edge]
    counted = {
        b: {
            "found": sum(1 for f in most if _bucket(gold.sources[f]) == b),
            "two_or_more": sum(
                1 for f, n in most.items() if _bucket(gold.sources[f]) == b and n >= 2
            ),
            "two_or_more_pooled": sum(
                1 for f, docs in pooled.items() if _bucket(gold.sources[f]) == b and len(docs) >= 2
            ),
        }
        for b in BUCKETS
    }
    return {
        "off": scored(off),
        "on": scored(support),
        "on_two_or_more": scored(e for e, n in support.items() if n >= 2),
        "by_support": {
            b: scored(e for e, n in support.items() if _bucket(n) == b) for b in BUCKETS
        },
        "counted": counted,
    }


def links(gold: Gold, found: Sequence[EntityLink], facts: Sequence[Fact]) -> dict[str, Any]:
    """The resolver's links scored against Wikidata ids: right, wrong, or not checkable.

    A `SAME_AS` or `SIMILAR` link is right when its two ends name one id, a
    `DIFFERENT` one when they share none. An end whose name T-REx never linked
    cannot be checked. Grouped by kind, and by who decided: the rules, the pair
    judge, or a person in its place (each link's reason says).
    """
    label: dict[str, str] = {}
    for fact in facts:
        for entity in (fact.subject, fact.object_entity):
            if entity is not None:
                label.setdefault(entity.key, entity.label or entity.key)
    out: dict[str, dict[str, Any]] = {}
    for link in found:
        ends = [gold.of(label.get(k, k)) for k in (link.source_key, link.target_key)]
        reason = link.reason or ""
        who = (
            "person"
            if reason.startswith("person:")
            else "judge"
            if "pair judge" in reason
            else "rules"
        )
        bucket = out.setdefault(
            f"{link.kind.value} ({who})", {"links": 0, "right": 0, "wrong": 0, "uncheckable": 0}
        )
        bucket["links"] += 1
        if not ends[0] or not ends[1]:
            bucket["uncheckable"] += 1
            continue
        same = bool(ends[0] & ends[1])
        bucket["right" if same is (link.kind is not LinkKind.DIFFERENT) else "wrong"] += 1
    for bucket in out.values():
        bucket["precision"] = _rate(bucket["right"], bucket["right"] + bucket["wrong"])
    return out


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #


def run(prepared: str | Path) -> StageReport:
    """Run `prepared/odke.json` three ways and score each (see `score`)."""
    return evaluate(prepared).stages[0]


def evaluate(prepared: str | Path) -> EvalReport:
    """`run`, as the eval report, with `corroboration.json` written beside it."""
    from openodke.run.config import load_config

    folder = Path(prepared)
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    gold = read_jsonl(folder / "gold.jsonl")
    ablation = ablate(load_config(folder / "odke.json"))
    return report_run(ablation, gold, meta, save_to=folder, path=folder)


def score_run(
    ablation: AblationRun,
    gold: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    *,
    save_to: Path | None = None,
) -> StageReport:
    """Score an `AblationRun` already made — `run` without the model calls."""
    return report_run(ablation, gold, meta, save_to=save_to).stages[0]


def report_run(
    ablation: AblationRun,
    gold: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    *,
    save_to: Path | None = None,
    path: str | Path | None = None,
) -> EvalReport:
    """`score_run`, as the eval report. The graph's numbers go in its notes and in `save_to`."""
    labels: Mapping[str, str] = meta["relation_labels"]
    read = Gold(gold, labels.values())
    graph = corroboration(read, ablation.gated, ablation.corroborated, labels)
    resolved = links(read, ablation.links, ablation.candidates)
    if save_to is not None:
        write_json(save_to / "corroboration.json", {"graph": graph, "links": resolved})
        # The facts themselves, for what is measured offline (bench/trex.py).
        for name, facts in (("grounded", ablation.grounded), ("graph", ablation.corroborated)):
            write_jsonl(
                save_to / "facts" / f"{name}.jsonl", (f.model_dump(mode="json") for f in facts)
            )
        write_jsonl(
            save_to / "facts" / "links.jsonl", (k.model_dump(mode="json") for k in ablation.links)
        )

    def notes(
        raw: Mapping[str, Metric], gated: Mapping[str, Metric], full: Mapping[str, Metric]
    ) -> list[str]:
        return _notes(meta, raw, gated, graph, resolved)

    return _common.report_run(
        ablation, gold, scoring(gold, meta, path=path), notes=notes, save_to=save_to
    )


def scoring(
    gold: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    *,
    hallucination: bool = True,
    path: str | Path | None = None,
) -> _common.Scoring:
    """How a prepared set scores facts from any source. `hallucination` is accepted for symmetry."""
    read = Gold(gold, meta["relation_labels"].values())
    return _common.Scoring(
        stage=NAME,
        meta=meta,
        units=partial(documents, read),
        aggregate=aggregate,
        row=row,
        dataset=Dataset(
            name=NAME,
            path=None if path is None else str(path),
            documents=len(gold),
            labels=sum(len(r["facts"]) for r in gold),
            details={f"gold_facts_{b}": int(meta.get("gold_facts", {}).get(b, 0)) for b in BUCKETS},
        ),
    )


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _notes(
    meta: Mapping[str, Any],
    raw: Mapping[str, Metric],
    gated: Mapping[str, Metric],
    graph: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    facts = meta.get("gold_facts", {})
    lines = [
        "gold facts by sources in the set: "
        + ", ".join(f"{b}: {facts.get(b, 0)}" for b in BUCKETS),
        f"factual precision: {_pct(raw['precision_factual'])} extracted, "
        f"{_pct(gated['precision_factual'])} after the gate"
        " (a known Wikidata fact; ODKE+ reports 91% raw, 98.8% ranked)",
        "recall by sources after the gate: "
        + ", ".join(f"{b}: {_pct(gated[f'recall_{b}'])}" for b in BUCKETS),
    ]
    for name in ("off", "on", "on_two_or_more"):
        g = graph[name]
        lines.append(
            f"graph, corroboration {name.replace('_', ' ')}: {g['edges']} edges, factual precision "
            f"{_pct(g['precision_factual'])}, recall by sources "
            + ", ".join(f"{b}: {_pct(g['recall'][b]['recall'])}" for b in BUCKETS)
        )
    counted = graph["counted"]
    lines.append(
        "gold facts on the graph given two or more sources (under any name): "
        + ", ".join(
            f"{b}: {counted[b]['two_or_more']} ({counted[b]['two_or_more_pooled']}) "
            f"of {counted[b]['found']}"
            for b in BUCKETS
        )
    )
    for kind, bucket in resolved.items():
        lines.append(
            f"resolver links {kind}: {bucket['links']}, right {bucket['right']}, "
            f"wrong {bucket['wrong']}, not checkable {bucket['uncheckable']}"
        )
    return lines


__all__ = [
    "ALIGNERS",
    "BUCKETS",
    "CHECKED",
    "NAME",
    "SAMPLE_URL",
    "Gold",
    "aggregate",
    "alignments",
    "corroboration",
    "documents",
    "abstracts",
    "evaluate",
    "fetch",
    "links",
    "objects",
    "prepare",
    "properties",
    "report_run",
    "row",
    "run",
    "score",
    "score_run",
    "scoring",
    "select",
    "share",
    "to_ontology",
]
