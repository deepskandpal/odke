"""Inverse and symmetric partners on saved predictions: scored with and without (#106).

    python bench/inverses.py /path/to/runs/cmp [--out DIR]

No model is called. `odke bench run` saves every configuration's triples
(`predictions/<row>.jsonl`), for openodke and for each competitor under
`competitors/`. Each file is scored as it is, and again with the partner of
every triple added by `openodke.corroborate.partners`, the step `Pipeline`
runs, using the dataset's own scorer.

The prepared ontologies declare no inverses, so the pairs are added to a copy
here: the ones Wikidata declares, an "inverse property" (P1696) or a symmetric
constraint, with both ends among the dataset's relations. Re-DocRED has
thirteen, measured twice: the six #106 named, then all thirteen. No
Text2KGBench ontology holds both ends of any pair, so its numbers cannot move,
and the script says so rather than printing a table of zeros. A symmetric
predicate's domain is narrowed to its range in the copy, and an inverse's
domain widened to its partner's range where `validate()` asks, so the copy
loads strictly; the partners do not depend on it.

`--out` writes each dataset's copy of the ontology and the numbers as JSON.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openodke import Entity, Evidence, Fact, Ontology
from openodke.corroborate import derived_from, partners
from openodke.eval.datasets import redocred
from openodke.eval.datasets._common import read_jsonl, snake

Triple = tuple[str, str, str]

ROWS = {
    "extraction alone": "extraction-alone",
    "+ grounding": "grounding",
    "+ corroboration": "corroboration",
}
SYSTEMS = (
    ("openodke", ""),
    ("LLMGraphTransformer", "competitors/lgt"),
    ("neo4j-graphrag", "competitors/neo4j"),
)

# Wikidata's own declarations among Re-DocRED's 96 relations, checked against
# wbgetentities on 9 Oct 2026. A pid mapped to itself is symmetric.
ISSUE = {
    "P131": "P150",
    "P361": "P527",
    "P155": "P156",
    "P710": "P1344",
    "P26": "P26",
    "P3373": "P3373",
}
WIKIDATA = {
    **ISSUE,
    "P36": "P1376",
    "P355": "P749",
    "P674": "P1441",
    "P1365": "P1366",
    "P170": "P800",
    "P176": "P1056",
    "P190": "P190",
}
PAIR_SETS = {"redocred": {"#106's six": ISSUE, "all Wikidata declares": WIKIDATA}}


def with_pairs(
    ontology: Mapping[str, Any], pairs: Mapping[str, str], names: Mapping[str, str]
) -> Ontology:
    """The prepared ontology plus `pairs` (pid -> pid), repaired just enough to load strictly."""
    data = json.loads(json.dumps(ontology))
    predicates = data["predicates"]
    for pid, other in pairs.items():
        name, partner = names[pid], names[other]
        if name == partner:
            predicates[name]["symmetric"] = True
            predicates[name]["domain"] = [predicates[name]["range"]]
        else:
            predicates[name]["inverse_of"] = partner
            for a, b in ((name, partner), (partner, name)):
                domain = predicates[b].setdefault("domain", [])
                if domain and predicates[a]["range"] not in domain:
                    domain.append(predicates[a]["range"])
    return Ontology.from_dict(data)


def derive(
    triples: Sequence[Triple], ontology: Ontology, labels: Mapping[str, str]
) -> list[tuple[Triple, Triple]]:
    """`(stated, partner)` for each triple the ontology implies a partner of.

    `labels` maps predicate names to the dataset's relation labels. A partner
    the batch already states is not derived, as in `Pipeline`.
    """
    names = {label: name for name, label in labels.items()}
    stated: dict[Any, Triple] = {}
    facts = []
    for s, r, o in triples:
        fact = Fact(
            subject=Entity(key=s, type="Thing", label=s),
            predicate=names.get(r, r),
            object_entity=Entity(key=o, type="Thing", label=o),
            evidence=(Evidence(doc_id="doc"),),
        )
        stated.setdefault(fact.signature, (s, r, o))
        facts.append(fact)
    return [
        (stated[derived_from(f)], (f.subject.key, labels.get(f.predicate, f.predicate), obj.key))
        for f in partners(facts, ontology)
        if (obj := f.object_entity) is not None
    ]


def predictions(path: Path) -> dict[str, list[Triple]]:
    return {row["id"]: [(s, r, o) for s, r, o in row["triples"]] for row in read_jsonl(path)}


def measure_redocred(folder: Path) -> dict[str, Any]:
    """Every system and row on one prepared Re-DocRED set, before and after, per pair set.

    Each partner added is also sorted by what the gold says: it is a gold fact;
    it is the partner of a fact the gold calls right but the gold leaves the
    other direction out; or its stated fact is one the gold calls wrong. A
    triple is right when the dataset's own scorer, given it alone, says so.
    """
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    gold = read_jsonl(folder / "gold.jsonl")
    labels = meta["relation_labels"]
    names = {pid: snake(label) for pid, label in redocred.RELATIONS.items()}
    prepared = json.loads((folder / "ontology.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"ontologies": {}, "results": []}
    for set_name, pairs in PAIR_SETS["redocred"].items():
        ontology = with_pairs(prepared, pairs, names)
        out["ontologies"][set_name] = ontology.model_dump(mode="json", exclude_defaults=True)
        for system, sub in SYSTEMS:
            for row, slug in ROWS.items():
                path = folder / sub / "predictions" / f"{slug}.jsonl"
                if not path.is_file():
                    continue
                before = predictions(path)
                after: dict[str, list[Triple]] = {}
                sorted_by_gold = {"in gold": 0, "unlabelled": 0, "from a wrong fact": 0}
                for doc in gold:
                    triples = before.get(doc["id"], [])
                    derived = derive(triples, ontology, labels)
                    after[doc["id"]] = list(dict.fromkeys([*triples, *(p for _, p in derived)]))
                    for partner, source in {p: s for s, p in derived}.items():
                        if partner in triples:
                            continue
                        if _right(doc, partner):
                            sorted_by_gold["in gold"] += 1
                        elif _right(doc, source):
                            sorted_by_gold["unlabelled"] += 1
                        else:
                            sorted_by_gold["from a wrong fact"] += 1
                result = _result(
                    set_name,
                    system,
                    row,
                    redocred.score(gold, before),
                    redocred.score(gold, after),
                )
                out["results"].append({**result, "partners": sorted_by_gold})
    return out


def _right(doc: Mapping[str, Any], triple: Triple) -> bool:
    return redocred.score([doc], {doc["id"]: [triple]})["precision"] == 1.0


def text2kgbench_pairs(root: Path) -> list[str]:
    """Text2KGBench ontologies holding both ends of a Wikidata pair: none, on the published sets."""
    found = []
    for folder in sorted(root.glob("ont_*")):
        meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        pids = {r["pid"] for r in meta["ontology"]["relations"]}
        found += [f"{folder.name}: {a}/{b}" for a, b in WIKIDATA.items() if a in pids and b in pids]
    return found


def _result(
    pairs: str, system: str, row: str, before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    tp = [round(m["precision"] * m["triples"]) for m in (before, after)]
    return {
        "pairs": pairs,
        "system": system,
        "row": row,
        **{f"{k}_before": before[k] for k in ("precision", "recall", "f1", "triples")},
        **{f"{k}_after": after[k] for k in ("precision", "recall", "f1", "triples")},
        "gold_recovered": tp[1] - tp[0],
    }


def table(results: Sequence[Mapping[str, Any]]) -> list[str]:
    out = [
        "| Pairs | System | Configuration | precision | recall | f1 "
        "| partners added: in gold / unlabelled / from a wrong fact |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cells = [
            f"{r[f'{k}_before']:.1%} → {r[f'{k}_after']:.1%} "
            f"({(r[f'{k}_after'] - r[f'{k}_before']) * 100:+.1f})"
            for k in ("precision", "recall", "f1")
        ]
        split = " / ".join(str(n) for n in r["partners"].values())
        added = r["triples_after"] - r["triples_before"]
        out.append(
            f"| {r['pairs']} | {r['system']} | {r['row']} | {' | '.join(cells)} "
            f"| {added}: {split} |"
        )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="the directory holding t2k/ont_* and redocred")
    parser.add_argument("--out", type=Path, help="write the ontology copies and the numbers here")
    args = parser.parse_args(argv)

    measured = measure_redocred(args.root / "redocred")
    lines = ["## Re-DocRED — 50 test documents, with and without inverse partners", ""]
    lines += table(measured["results"])
    t2k = text2kgbench_pairs(args.root / "t2k")
    lines += [
        "",
        "## Text2KGBench",
        "",
        "Pairs with both ends in one ontology: "
        + (", ".join(t2k) if t2k else "none, so no score can move."),
    ]
    print("\n".join(lines))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for set_name, ontology in measured["ontologies"].items():
            slug = "issue" if set_name.startswith("#") else "wikidata"
            (args.out / f"redocred-ontology-{slug}.json").write_text(
                json.dumps(ontology, indent=2) + "\n"
            )
        (args.out / "inverses.json").write_text(
            json.dumps({"redocred": measured["results"], "text2kgbench_pairs": t2k}, indent=2)
            + "\n"
        )


if __name__ == "__main__":
    main()
