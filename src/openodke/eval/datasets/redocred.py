"""Re-DocRED (Tan et al., EMNLP 2022): relations across a whole Wikipedia passage.

DocRED's documents — the opening paragraphs of Wikipedia articles, annotated
with 96 Wikidata relations — with the missing labels that made DocRED's
precision unmeasurable restored. The closest public stand-in for ODKE+'s own
setting, one entity's page at a time. <https://github.com/tonytan48/Re-DocRED>
(MIT); `fetch` downloads it, nothing is copied into this package.

The task here is the paper's, not DocRED's: no entity list is given, the model
reads the passage and states facts in surface strings. So `score` is ours, and
says so. A predicted `(subject, relation, object)` is correct when the relation
matches and the subject and object each name a mention of the gold head and tail
entities (case and punctuation ignored); each gold fact counts once. Precision
and recall are micro-averaged over documents, as DocRED's are. An entity is
hallucinated when its name, normalised the same way, is nowhere in the passage.

The ontology's domains and ranges are read off the dev split's labels, so the
test split's are never looked at to build the schema.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openodke.eval.ablation import AblationRun, ablate
from openodke.eval.datasets._common import (
    Opener,
    Triple,
    change,
    doc_names,
    download,
    read_jsonl,
    report,
    run_config,
    snake,
    triples_by_doc,
    write_documents,
    write_json,
    write_jsonl,
)
from openodke.eval.report import Metric, StageReport

NAME = "redocred"
BASE_URL = "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data"
SPLITS = ("dev", "test", "train")

# DocRED's entity types, as openodke types; TIME and NUM are values, not entities.
TYPES = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Misc",
    "TIME": "date",
    "NUM": "number",
}

# DocRED's `rel_info.json`: the 96 relations, Wikidata property id -> label.
RELATIONS: dict[str, str] = {
    "P6": "head of government",
    "P17": "country",
    "P19": "place of birth",
    "P20": "place of death",
    "P22": "father",
    "P25": "mother",
    "P26": "spouse",
    "P27": "country of citizenship",
    "P30": "continent",
    "P31": "instance of",
    "P35": "head of state",
    "P36": "capital",
    "P37": "official language",
    "P39": "position held",
    "P40": "child",
    "P50": "author",
    "P54": "member of sports team",
    "P57": "director",
    "P58": "screenwriter",
    "P69": "educated at",
    "P86": "composer",
    "P102": "member of political party",
    "P108": "employer",
    "P112": "founded by",
    "P118": "league",
    "P123": "publisher",
    "P127": "owned by",
    "P131": "located in the administrative territorial entity",
    "P136": "genre",
    "P137": "operator",
    "P140": "religion",
    "P150": "contains administrative territorial entity",
    "P155": "follows",
    "P156": "followed by",
    "P159": "headquarters location",
    "P161": "cast member",
    "P162": "producer",
    "P166": "award received",
    "P170": "creator",
    "P171": "parent taxon",
    "P172": "ethnic group",
    "P175": "performer",
    "P176": "manufacturer",
    "P178": "developer",
    "P179": "series",
    "P190": "sister city",
    "P194": "legislative body",
    "P205": "basin country",
    "P206": "located in or next to body of water",
    "P241": "military branch",
    "P264": "record label",
    "P272": "production company",
    "P276": "location",
    "P279": "subclass of",
    "P355": "subsidiary",
    "P361": "part of",
    "P364": "original language of work",
    "P400": "platform",
    "P403": "mouth of the watercourse",
    "P449": "original network",
    "P463": "member of",
    "P488": "chairperson",
    "P495": "country of origin",
    "P527": "has part",
    "P551": "residence",
    "P569": "date of birth",
    "P570": "date of death",
    "P571": "inception",
    "P576": "dissolved, abolished or demolished",
    "P577": "publication date",
    "P580": "start time",
    "P582": "end time",
    "P585": "point in time",
    "P607": "conflict",
    "P674": "characters",
    "P676": "lyrics by",
    "P706": "located on terrain feature",
    "P710": "participant",
    "P737": "influenced by",
    "P740": "location of formation",
    "P749": "parent organization",
    "P800": "notable work",
    "P807": "separated from",
    "P840": "narrative location",
    "P937": "work location",
    "P1001": "applies to jurisdiction",
    "P1056": "product or material produced",
    "P1198": "unemployment rate",
    "P1336": "territory claimed by",
    "P1344": "participant of",
    "P1365": "replaces",
    "P1366": "replaced by",
    "P1376": "capital of",
    "P1412": "languages spoken, written or signed",
    "P1441": "present in work",
    "P3373": "sibling",
}


def fetch(
    dest: str | Path, *, splits: Sequence[str] = ("dev", "test"), opener: Opener | None = None
) -> Path:
    """Download `splits` under `dest`. `dev` is needed: the ontology is built from it."""
    root = Path(dest)
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; one of {', '.join(SPLITS)}")
        name = f"{split}_revised.json"
        download(f"{BASE_URL}/{name}", root / name, opener=opener)
    return root


def to_ontology(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The 96 relations as predicates, with domains and ranges seen in `documents`.

    A relation's domain is every head type it was seen with; its range is the
    tail type it was seen with most. TIME and NUM ranges become date and number
    literals. A relation never seen keeps an open domain and a string range.
    """
    heads: dict[str, Counter[str]] = {r: Counter() for r in RELATIONS}
    tails: dict[str, Counter[str]] = {r: Counter() for r in RELATIONS}
    for doc in documents:
        clusters = doc["vertexSet"]
        for label in doc["labels"]:
            if label["r"] in RELATIONS:
                heads[label["r"]][clusters[label["h"]][0]["type"]] += 1
                tails[label["r"]][clusters[label["t"]][0]["type"]] += 1
    entity_types = {name for name in TYPES.values() if name[:1].isupper()}
    predicates: dict[str, dict[str, Any]] = {}
    for pid, label in RELATIONS.items():
        domain = sorted({TYPES[t] for t in heads[pid] if TYPES.get(t, "") in entity_types})
        tail = tails[pid].most_common(1)[0][0] if tails[pid] else ""
        predicates[snake(label)] = {
            "description": f"{label} ({pid})",
            "domain": domain,
            "range": TYPES.get(tail, "string"),
            "aliases": [label],
        }
    return {
        "name": "redocred",
        "version": "re-docred",
        "types": {
            name: {"description": code} for code, name in TYPES.items() if name in entity_types
        },
        "predicates": predicates,
    }


def detokenize(sentences: Sequence[Sequence[str]]) -> str:
    """DocRED's tokens joined back into readable text: no space before , . ) 's and the like."""
    text = " ".join(" ".join(tokens) for tokens in sentences)
    text = re.sub(r" ([,.;:!?)\]}%])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    return re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)


def prepare(
    root: str | Path,
    out: str | Path,
    *,
    split: str = "test",
    limit: int | None = None,
    extract_model: str | None = None,
    ground_model: str | None = None,
    paper: bool = False,
) -> Path:
    """A runnable directory for `split`: `out/docs/`, `ontology.json`, `gold.jsonl`, `odke.json`."""
    base = Path(root)
    data_path, dev_path = base / f"{split}_revised.json", base / "dev_revised.json"
    missing = [str(p) for p in (data_path, dev_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"not fetched yet: {', '.join(missing)} — run fetch first")
    documents = json.loads(data_path.read_text(encoding="utf-8"))
    if limit:
        documents = documents[:limit]
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    target = Path(out)
    write_json(target / "ontology.json", to_ontology(dev))
    gold = [_gold_row(f"{split}_{i:04d}", doc) for i, doc in enumerate(documents)]
    write_documents(target, ((row["id"], row["text"]) for row in gold))
    write_jsonl(target / "gold.jsonl", gold)
    labels = {snake(label): label for label in RELATIONS.values()}
    write_json(
        target / "dataset.json",
        {"dataset": NAME, "split": split, "documents": len(gold), "relation_labels": labels},
    )
    write_json(
        target / "odke.json",
        run_config(extract_model=extract_model, ground_model=ground_model, paper=paper),
    )
    return target


def _gold_row(doc_id: str, doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": doc["title"],
        "text": detokenize(doc["sents"]),
        "entities": [sorted({m["name"] for m in cluster}) for cluster in doc["vertexSet"]],
        "facts": [
            [label["h"], RELATIONS[label["r"]], label["t"]]
            for label in doc["labels"]
            if label["r"] in RELATIONS
        ],
    }


def _norm(text: str) -> str:
    # Underscores too: `place_of_birth` and `place of birth` are one relation.
    return re.sub(r"[\W_]+", "", text.casefold())


def score(
    gold: Sequence[Mapping[str, Any]], predicted: Mapping[str, Sequence[Triple]]
) -> dict[str, Metric]:
    """Micro precision, recall and F1, relation conformance, and entity hallucination."""
    relations = {_norm(label) for label in RELATIONS.values()}
    tp = n_pred = n_gold = conformant = hallucinated = 0
    for row in gold:
        names = [{_norm(n) for n in cluster} for cluster in row["entities"]]
        facts = {(h, _norm(r), t) for h, r, t in row["facts"]}
        text = _norm(row["text"])
        found: set[tuple[int, str, int]] = set()
        triples = list(dict.fromkeys(predicted.get(row["id"], ())))
        n_pred += len(triples)
        n_gold += len(facts)
        for subject, relation, obj in triples:
            rel, s, o = _norm(relation), _norm(subject), _norm(obj)
            conformant += rel in relations
            hallucinated += (s not in text) or (o not in text) or rel not in relations
            heads = [i for i, cluster in enumerate(names) if s in cluster]
            tails = [i for i, cluster in enumerate(names) if o in cluster]
            match = next(
                ((h, rel, t) for h in heads for t in tails if (h, rel, t) in facts - found), None
            )
            if match is not None:
                found.add(match)
                tp += 1
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onto_conf": conformant / n_pred if n_pred else 1.0,
        "triples": n_pred,
        "gold_triples": n_gold,
        "hallucinated_triples": hallucinated,
    }


def run(prepared: str | Path) -> StageReport:
    """Run `prepared/odke.json` three ways and score each (see `score`)."""
    from openodke.run.config import load_config

    folder = Path(prepared)
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    gold = read_jsonl(folder / "gold.jsonl")
    return score_run(ablate(load_config(folder / "odke.json")), gold, meta)


def score_run(
    ablation: AblationRun, gold: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]
) -> StageReport:
    """Score an `AblationRun` already made — `run` without the model calls."""
    labels: Mapping[str, str] = meta["relation_labels"]
    names = doc_names(ablation.documents)
    rows = []
    scored: dict[str, dict[str, Metric]] = {}
    for name, facts, calls in ablation.configurations():
        metrics = score(gold, triples_by_doc(facts, names, lambda p: labels.get(p, p)))
        scored[name] = metrics
        rows.append((name, metrics, calls, len(facts)))
    raw, gated, full = (scored[n] for n, _, _ in ablation.configurations())
    before = float(raw["hallucinated_triples"] or 0)
    after = float(gated["hallucinated_triples"] or 0)
    precisions = ", ".join(
        f"{p:.1%}" if p is not None else "n/a"
        for p in (raw["precision"], gated["precision"], full["precision"])
    )
    notes = [
        f"hallucinated triples: {before:.0f} extracted, {after:.0f} after the gate — "
        f"{change(before, after)} (ODKE+ reports -35%)",
        f"precision: {precisions} (extracted, after the gate, after corroboration;"
        " ODKE+ reports 91% raw, 98.8% ranked)",
        *ablation.notes,
    ]
    return report(f"{NAME}:{meta['split']}", len(gold), rows, notes)


__all__ = [
    "BASE_URL",
    "NAME",
    "RELATIONS",
    "SPLITS",
    "TYPES",
    "fetch",
    "prepare",
    "run",
    "score",
    "score_run",
    "to_ontology",
]
