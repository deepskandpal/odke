"""Text2KGBench (Mihindukulasooriya et al., ISWC 2023): ontology-guided extraction from sentences.

Given an ontology and a sentence, extract the facts the sentence states, using
only the ontology's relations. Two sources: Wikidata-TekGen (10 ontologies,
13,474 sentences) and DBpedia-WebNLG (19 ontologies, 4,860 sentences).
<https://github.com/cenguix/Text2KGBench> — the data is CC BY-SA 4.0 and is
downloaded from there by `fetch`, never copied into this package.

`score` is the benchmark's own `src/evaluation/run_eval.py`, so its numbers sit
next to the published baselines:

- precision, recall and F1 per sentence, on triples normalised by removing
  underscores and whitespace and lowercasing, counting only predicted triples
  whose relation the sentence's gold uses; averaged over sentences;
- ontology conformance: the share of predicted triples whose relation is in the
  ontology, and relation hallucination, its complement;
- subject and object hallucination: the share whose subject (object), stemmed,
  is not in the stemmed sentence plus the ontology's concept labels.

Two differences, both deliberate. Every test sentence is scored, an empty answer
included (the original skips a sentence a system wrote no line for). Tokenising
uses NLTK's Treebank tokenizer, which `word_tokenize` wraps, so no tokenizer
data has to be downloaded; the hallucination metrics need `openodke[bench]`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openodke.eval.ablation import AblationRun, ablate
from openodke.eval.datasets._common import (
    LITERALS,
    Opener,
    Triple,
    change,
    doc_names,
    download,
    pascal,
    read_jsonl,
    report,
    run_config,
    save_predictions,
    snake,
    triples_by_doc,
    verdicts,
    write_documents,
    write_json,
    write_jsonl,
)
from openodke.eval.report import Metric, StageReport

NAME = "text2kgbench"
BASE_URL = "https://raw.githubusercontent.com/cenguix/Text2KGBench/main/data"
SOURCES: dict[str, tuple[str, ...]] = {
    "wikidata_tekgen": tuple(
        f"ont_{n}"
        for n in (
            "1_movie",
            "2_music",
            "3_sport",
            "4_book",
            "5_military",
            "6_computer",
            "7_space",
            "8_politics",
            "9_nature",
            "10_culture",
        )
    ),
    "dbpedia_webnlg": tuple(
        f"ont_{n}"
        for n in (
            "1_university",
            "2_musicalwork",
            "3_airport",
            "4_building",
            "5_athlete",
            "6_politician",
            "7_company",
            "8_celestialbody",
            "9_astronaut",
            "10_comicscharacter",
            "11_meanoftransportation",
            "12_monument",
            "13_food",
            "14_writtenwork",
            "15_sportsteam",
            "16_city",
            "17_artist",
            "18_scientist",
            "19_film",
        )
    ),
}


def files(source: str, ontology_id: str) -> dict[str, str]:
    """The three files one ontology needs, as paths under the dataset's `data/`."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; one of {', '.join(SOURCES)}")
    if ontology_id not in SOURCES[source]:
        raise ValueError(
            f"{ontology_id!r} is not a {source} ontology; one of {', '.join(SOURCES[source])}"
        )
    stem = ontology_id.removeprefix("ont_")
    return {
        "ontology": f"{source}/ontologies/{stem}_ontology.json",
        "test": f"{source}/test/{ontology_id}_test.jsonl",
        "gold": f"{source}/ground_truth/{ontology_id}_ground_truth.jsonl",
    }


def fetch(
    dest: str | Path,
    *,
    source: str = "wikidata_tekgen",
    ontologies: Sequence[str] | None = None,
    opener: Opener | None = None,
) -> Path:
    """Download `ontologies` (all of `source`'s by default) under `dest`. Idempotent."""
    root = Path(dest)
    for ontology_id in ontologies or SOURCES.get(source, ()):
        for relative in files(source, ontology_id).values():
            download(f"{BASE_URL}/{relative}", root / relative, opener=opener)
    return root


def to_ontology(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """The benchmark's ontology in openodke's JSON form, and predicate name -> relation label.

    Concepts become types (`film production company` -> `FilmProductionCompany`);
    relations become predicates (`cast member` -> `cast_member`) with the domain
    and range the benchmark gives; a label listed twice is one predicate with
    both domains. A range that is no concept is a literal type,
    or, if it is not one of those either (DBpedia's `Party`), a type of its own.
    """
    concepts = {c["qid"]: pascal(c["label"]) for c in raw["concepts"]}
    types: dict[str, dict[str, Any]] = {
        pascal(c["label"]): {"description": f"{c['label']} ({c['qid']})"} for c in raw["concepts"]
    }

    def as_type(ref: str) -> str:
        if ref in concepts:
            return concepts[ref]
        name = pascal(ref)
        types.setdefault(name, {"description": ref})
        return name

    predicates: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    for rel in raw["relations"]:
        name = snake(rel["label"])
        domain = [as_type(rel["domain"])] if rel.get("domain") else []
        if name in predicates:
            # The same label twice with another domain (`league` for a team and
            # for an athlete) is one relation: the benchmark scores by label.
            known = predicates[name]["domain"]
            known.extend(d for d in domain if d not in known)
            continue
        literal = LITERALS.get(str(rel.get("range", "")).strip().lower())
        predicates[name] = {
            "description": f"{rel['label']} ({rel['pid']})",
            "domain": domain,
            "range": literal if literal is not None else as_type(rel["range"]),
            "aliases": [rel["label"]],
        }
        labels[name] = rel["label"]
    ontology = {
        "name": raw.get("id", "text2kgbench"),
        "version": "text2kgbench",
        "types": types,
        "predicates": predicates,
    }
    return ontology, labels


def prepare(
    root: str | Path,
    ontology_id: str,
    out: str | Path,
    *,
    source: str = "wikidata_tekgen",
    limit: int | None = None,
    extract_model: str | None = None,
    ground_model: str | None = None,
    paper: bool = False,
    max_tokens: int = 16000,
) -> Path:
    """A runnable directory for one ontology's test sentences.

    `out/docs/<sentence id>.txt`, `out/ontology.json` (openodke's form),
    `out/gold.jsonl` (the benchmark's rows), `out/dataset.json` (what `run`
    needs to score) and `out/odke.json`, the run config. `limit` keeps the first
    `limit` sentences, which is how a pilot stays cheap.
    """
    base = Path(root)
    paths = {k: base / v for k, v in files(source, ontology_id).items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"not fetched yet: {', '.join(missing)} — run fetch first")
    raw = json.loads(paths["ontology"].read_text(encoding="utf-8"))
    gold = read_jsonl(paths["gold"])[:limit] if limit else read_jsonl(paths["gold"])
    target = Path(out)
    ontology, labels = to_ontology(raw)
    write_json(target / "ontology.json", ontology)
    write_documents(target, ((row["id"], row["sent"]) for row in gold))
    write_jsonl(target / "gold.jsonl", gold)
    write_json(
        target / "dataset.json",
        {
            "dataset": NAME,
            "source": source,
            "ontology_id": ontology_id,
            "sentences": len(gold),
            "relation_labels": labels,
            "ontology": raw,
        },
    )
    write_json(
        target / "odke.json",
        run_config(
            extract_model=extract_model,
            ground_model=ground_model,
            paper=paper,
            relations=len(ontology["predicates"]),
            max_tokens=max_tokens,
        ),
    )
    return target


# --------------------------------------------------------------------------- #
# the benchmark's metrics
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    return re.sub(r"(_|\s+)", "", text).lower()


def _key(triple: Triple) -> str:
    return f"{_norm(triple[0])}{_norm(triple[1])}{_norm(triple[2])}"


def _prf(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not pred:
        return 0.0, 0.0, 0.0
    p = len(gold & pred) / len(pred)
    r = len(gold & pred) / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r)) if p + r > 0 else 0.0


class _Stems:
    """The benchmark's `clean_entity_string`, NLTK's Porter stemmer and Treebank tokenizer."""

    def __init__(self) -> None:
        try:
            from nltk.stem import PorterStemmer
            from nltk.tokenize import TreebankWordTokenizer
        except ImportError as exc:
            raise ImportError(
                'the hallucination metrics need NLTK: pip install "openodke[bench]"'
            ) from exc
        self._stem = PorterStemmer().stem
        self._tokens = TreebankWordTokenizer().tokenize

    def clean(self, text: str) -> str:
        stemmed = "".join(self._stem(word) for word in self._tokens(text))
        return re.sub(r"(_|\s+)", "", stemmed).lower().replace("01januari", "")


def score(
    gold: Sequence[Mapping[str, Any]],
    predicted: Mapping[str, Sequence[Triple]],
    ontology: Mapping[str, Any],
    *,
    hallucination: bool = True,
) -> dict[str, Metric]:
    """The benchmark's seven metrics for `predicted` (sentence id -> triples).

    `hallucination=False` skips the two that need NLTK.
    """
    relations = {r["label"].replace(" ", "_") for r in ontology["relations"]}
    concepts = " ".join(c["label"] for c in ontology["concepts"])
    stems = _Stems() if hallucination else None
    totals = dict.fromkeys(
        ("precision", "recall", "f1", "onto_conf", "rel_halluc", "sub_halluc", "obj_halluc"), 0.0
    )
    hallucinated = 0
    for row in gold:
        gold_triples = [(t["sub"], t["rel"], t["obj"]) for t in row["triples"]]
        system = [(s, r.replace(" ", "_"), o) for s, r, o in predicted.get(row["id"], ())]
        gold_relations = {t[1].replace(" ", "_") for t in gold_triples}
        filtered = [t for t in system if t[1] in gold_relations]
        p, r, f = _prf({_key(t) for t in gold_triples}, {_key(t) for t in filtered})
        totals["precision"] += p
        totals["recall"] += r
        totals["f1"] += f
        if system:
            conformant = sum(1 for t in system if t[1] in relations)
            totals["onto_conf"] += conformant / len(system)
            totals["rel_halluc"] += 1 - conformant / len(system)
        else:
            totals["onto_conf"] += 1.0
        if stems is not None and system:
            context = stems.clean(row["sent"] + concepts)
            subj = [stems.clean(t[0]) not in context for t in system]
            obj = [stems.clean(t[2]) not in context for t in system]
            totals["sub_halluc"] += sum(subj) / len(system)
            totals["obj_halluc"] += sum(obj) / len(system)
            hallucinated += sum(
                1
                for t, s, o in zip(system, subj, obj, strict=True)
                if s or o or t[1] not in relations
            )
        elif stems is None:
            hallucinated += sum(1 for t in system if t[1] not in relations)
    n = len(gold) or 1
    out: dict[str, Metric] = {k: v / n for k, v in totals.items()}
    if stems is None:
        out["sub_halluc"] = out["obj_halluc"] = None
    out["triples"] = sum(len(predicted.get(row["id"], ())) for row in gold)
    out["hallucinated_triples"] = hallucinated
    return out


# --------------------------------------------------------------------------- #
# run a prepared directory three ways
# --------------------------------------------------------------------------- #


def run(prepared: str | Path, *, hallucination: bool = True) -> StageReport:
    """Run `prepared/odke.json` three ways and score each with the benchmark's metrics.

    Extraction and grounding are each called once (`ablate`); the + grounding and
    + corroboration rows replay them. Notes put the two numbers the ODKE+ paper
    reports beside ours: grounding cut hallucinated extractions by 35%, and
    corroboration took precision from 91% to 98.8% (§4, p. 6).
    """
    from openodke.run.config import load_config

    if hallucination:
        # Scoring needs NLTK; without it, say so before the models are paid, not after.
        _Stems()
    folder = Path(prepared)
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    gold = read_jsonl(folder / "gold.jsonl")
    ablation = ablate(load_config(folder / "odke.json"))
    return score_run(ablation, gold, meta, hallucination=hallucination, save_to=folder)


def score_run(
    ablation: AblationRun,
    gold: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    *,
    hallucination: bool = True,
    save_to: Path | None = None,
) -> StageReport:
    """Score an `AblationRun` already made — `run` without the model calls."""
    labels: Mapping[str, str] = meta["relation_labels"]
    names = doc_names(ablation.documents)
    rows = []
    scored: dict[str, dict[str, Metric]] = {}
    predictions = {}
    for name, facts, calls in ablation.configurations():
        predicted = triples_by_doc(facts, names, lambda p: labels.get(p, p))
        predictions[name] = predicted
        metrics = score(gold, predicted, meta["ontology"], hallucination=hallucination)
        scored[name] = metrics
        rows.append((name, metrics, calls, len(facts)))
    if save_to is not None:
        save_predictions(save_to, predictions)
    return report(f"{NAME}:{meta['ontology_id']}", len(gold), rows, _notes(scored, ablation))


def _notes(scored: Mapping[str, Mapping[str, Metric]], ablation: AblationRun) -> list[str]:
    names = [name for name, _, _ in ablation.configurations()]
    raw, gated, full = (scored[n] for n in names)
    before, after = _n(raw["hallucinated_triples"]), _n(gated["hallucinated_triples"])
    notes = [
        "hallucinated triples (subject or object not in the sentence, or relation not in "
        f"the ontology): {before:.0f} extracted, {after:.0f} after the gate — "
        f"{change(before, after)} (ODKE+ reports -35%)",
        f"precision: {_pct(raw['precision'])} extracted, {_pct(gated['precision'])} after the gate,"
        f" {_pct(full['precision'])} after corroboration (ODKE+ reports 91% raw, 98.8% ranked)",
    ]
    notes.append(f"grounder verdicts on the candidates: {verdicts(ablation.grounded)}")
    return notes + list(ablation.notes)


def _n(value: Metric) -> float:
    return float(value or 0)


def _pct(value: Metric) -> str:
    return "n/a" if value is None else f"{value:.1%}"


__all__ = [
    "BASE_URL",
    "NAME",
    "SOURCES",
    "fetch",
    "files",
    "prepare",
    "run",
    "score",
    "score_run",
    "to_ontology",
]
