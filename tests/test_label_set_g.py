"""`bench/labels/make_g.py`: label set G drawn from a tiny made-up comparison. No network.

The comparison is prepared by the datasets' own `prepare`, from a handful of
invented Text2KGBench sentences and Re-DocRED documents, with three
extractors' triples written beside it the way `bench/run_all.sh` leaves them.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openodke.eval.datasets import redocred, text2kgbench
from openodke.eval.sheets import make_sheets

SCRIPT = Path(__file__).parent.parent / "bench" / "labels" / "make_g.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("odke_bench_make_g", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses look their module up while it loads.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


g = _load()
PLAN = g.Plan(
    match=3, not_in_gold=3, gold_only=2, planted=5, dev=6, self_agreement=4, trim_words=20
)

FILMS = ["Aster", "Birch", "Cedar", "Dahlia", "Elder", "Fennel", "Garnet", "Hazel"]
TOWNS = ["Oakford", "Pinebury", "Quarrow", "Rushden", "Selby", "Tilbury"]
LANDS = ["Veland", "Wexia", "Yorvan", "Zantor", "Ulmar", "Kestia"]


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _predictions(folder: Path, by_extractor: dict[str, dict[str, list[list[str]]]]) -> None:
    for name, sub in g.EXTRACTORS.items():
        rows = [{"id": doc, "triples": t} for doc, t in by_extractor[name].items()]
        _jsonl(folder / sub / g.REFERENCE, rows)


def _t2k(root: Path, cmp: Path) -> None:
    source = root / "t2k" / "wikidata_tekgen"
    ontology = {
        "id": "ont_1_movie",
        "concepts": [{"qid": "Q11424", "label": "film"}, {"qid": "Q5", "label": "human"}],
        "relations": [
            {"pid": "P57", "label": "director", "domain": "Q11424", "range": "Q5"},
            {"pid": "P58", "label": "screenwriter", "domain": "Q11424", "range": "Q5"},
            {"pid": "P161", "label": "cast member", "domain": "Q11424", "range": "Q5"},
            {"pid": "P577", "label": "publication date", "domain": "Q11424", "range": ""},
        ],
    }
    (source / "ontologies").mkdir(parents=True)
    (source / "ontologies" / "1_movie_ontology.json").write_text(json.dumps(ontology))
    rows: list[dict[str, Any]] = []
    by: dict[str, dict[str, list[list[str]]]] = {name: {} for name in g.EXTRACTORS}
    for i, film in enumerate(FILMS, start=1):
        doc, title = f"ont_1_movie_test_{i}", f"{film} Story"
        boss, star, lead = f"Dana {film}son", f"Ezra {film}ley", f"Ivy {film}ton"
        rows.append(
            {
                "id": doc,
                "sent": f"{title} is a {1990 + i} film directed by {boss} and starring "
                f"{star} and {lead}.",
                "triples": [
                    {"sub": title, "rel": "director", "obj": boss},
                    {"sub": title, "rel": "cast member", "obj": star},
                    {"sub": title, "rel": "cast member", "obj": lead},
                    {"sub": title, "rel": "publication date", "obj": f"01 January {1990 + i}"},
                ],
            }
        )
        by["openodke"][doc] = [[title, "director", boss], [title, "director", star]]
        by["lgt"][doc] = [[title, "cast member", star], [title, "cast member", boss]]
        by["neo4j"][doc] = [[title, "cast member", lead], [title, "screenwriter", boss]]
    _jsonl(source / "ground_truth" / "ont_1_movie_ground_truth.jsonl", rows)
    out = text2kgbench.prepare(root / "t2k", "ont_1_movie", cmp / "t2k" / "ont_1_movie")
    _predictions(out, by)


def _mention(name: str, sent: int, at: int, kind: str) -> dict[str, Any]:
    return {"name": name, "sent_id": sent, "pos": [at, at + len(name.split())], "type": kind}


def _redocred(root: Path, cmp: Path) -> None:
    docs = []
    by: dict[str, dict[str, list[list[str]]]] = {name: {} for name in g.EXTRACTORS}
    for i, (town, land) in enumerate(zip(TOWNS, LANDS, strict=True)):
        person, works, year = f"Mara {town}er", f"{land} Works", str(1950 + i)
        sents = [
            [*person.split(), "was", "born", "in", town, "in", year, "."],
            [town, "is", "a", "town", "in", land, "."],
            [*person.split(), "worked", "for", *works.split(), "."],
            [*works.split(), "is", "based", "in", town, "."],
        ]
        docs.append(
            {
                "title": person,
                "sents": sents,
                "vertexSet": [
                    [_mention(person, 0, 0, "PER"), _mention(person, 2, 0, "PER")],
                    [_mention(town, 0, 5, "LOC"), _mention(town, 1, 0, "LOC")],
                    [_mention(year, 0, 7, "TIME")],
                    [_mention(land, 1, 5, "LOC")],
                    [_mention(works, 2, 4, "ORG"), _mention(works, 3, 0, "ORG")],
                ],
                "labels": [
                    {"r": "P19", "h": 0, "t": 1, "evidence": [0]},
                    {"r": "P569", "h": 0, "t": 2, "evidence": [0]},
                    {"r": "P17", "h": 1, "t": 3, "evidence": [1]},
                    {"r": "P131", "h": 1, "t": 3, "evidence": [1]},
                    {"r": "P108", "h": 0, "t": 4, "evidence": [2]},
                    {"r": "P159", "h": 4, "t": 1, "evidence": [3]},
                    {"r": "P17", "h": 4, "t": 3, "evidence": []},
                ],
            }
        )
        doc = f"test_{i:04d}"
        by["openodke"][doc] = [[person, "place of birth", town], [person, "residence", town]]
        by["lgt"][doc] = [[person, "date of birth", year], [works, "country", town]]
        by["neo4j"][doc] = [[town, "country", land], [person, "employer", town]]
    raw = root / "redocred"
    raw.mkdir(parents=True)
    for split in ("dev", "test"):
        (raw / f"{split}_revised.json").write_text(json.dumps(docs))
    _predictions(redocred.prepare(raw, cmp / "redocred"), by)


@pytest.fixture(scope="module")
def cmp(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("data")
    cmp = tmp_path_factory.mktemp("cmp")
    _t2k(root, cmp)
    _redocred(root, cmp)
    return cmp, root / "redocred" / "test_revised.json"


@pytest.fixture(scope="module")
def made(cmp: tuple[Path, Path]) -> Any:
    return g.make(g.load_all(*cmp), PLAN)


@pytest.fixture(scope="module")
def written(made: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("G")
    g.write(made, out)
    return out


def _real(made: Any) -> list[dict[str, Any]]:
    return [p for p in made.private if p["planted"] is None]


def test_each_stratum_gets_its_count_and_the_extractors_share_it(made: Any) -> None:
    real = _real(made)
    for dataset in g.DATASETS:
        for status in g.STATUSES:
            rows = [p for p in real if p["dataset"] == dataset and p["gold_status"] == status]
            assert len(rows) == PLAN.wanted(status), (dataset, status)
            if status != "gold_only":
                assert Counter(p["extractor"] for p in rows) == dict.fromkeys(g.EXTRACTORS, 1)
            else:
                assert {p["extractor"] for p in rows} == {None}
        planted = [p for p in made.private if p["planted"] and p["dataset"] == dataset]
        assert len(planted) == PLAN.planted
    assert len(made.rows) == len(made.private) == 2 * (3 + 3 + 2 + 5)
    # Every item says something the others do not.
    claims = Counter((p["doc"], tuple(p["triple"]), p["denied"]) for p in made.private)
    assert max(claims.values()) == 1


def test_no_planted_fact_is_in_gold(made: Any, cmp: tuple[Path, Path]) -> None:
    sets = {s.name: s for s in g.load_all(*cmp)}
    planted = [p for p in made.private if p["planted"]]
    assert {p["planted"] for p in planted} >= {"polarity", "object_swap", "value_change"}
    for p in planted:
        s, triple = sets[p["set"]], tuple(p["triple"])
        assert p["expected"] == ["contradicted", "not_found"]
        if p["planted"] == "polarity":
            # The gold fact itself, denied.
            assert p["denied"] and s.matched(p["doc"], triple) is not None
            continue
        assert not p["denied"]
        assert s.matched(p["doc"], triple) is None, p
        folded = {tuple(g.fold(x) for x in t) for t in s.gold(p["doc"])}
        assert tuple(g.fold(x) for x in triple) not in folded


def test_dev_gate_and_planted_never_overlap(made: Any) -> None:
    by_split: dict[str, list[dict[str, Any]]] = {"dev": [], "gate": [], "planted": []}
    for p in made.private:
        by_split[p["split"]].append(p)
    assert len(by_split["dev"]) == PLAN.dev
    assert len(by_split["gate"]) == len(_real(made)) - PLAN.dev
    assert all(p["planted"] for p in by_split["planted"])
    assert not any(p["planted"] for p in by_split["dev"] + by_split["gate"])
    ids = {name: {p["id"] for p in rows} for name, rows in by_split.items()}
    assert not ids["dev"] & ids["gate"] and not ids["planted"] & (ids["dev"] | ids["gate"])
    # Whole documents: no passage is read in both dev and gate.
    docs = {name: {p["doc"] for p in rows} for name, rows in by_split.items()}
    assert not docs["dev"] & docs["gate"]
    again = {p["id"] for p in made.private if p["self_agreement"]}
    assert len(again) == PLAN.self_agreement and again <= ids["dev"] | ids["gate"]


def test_the_sheets_show_nothing_private(written: Path, tmp_path: Path) -> None:
    made = make_sheets("grounding", written / "items.jsonl", tmp_path / "sheets", per_sheet=5)
    private = [json.loads(line) for line in (written / "items.private.jsonl").open()]
    sheets = "".join(p.read_text(encoding="utf-8") for p in made.sheets)
    secret = r"openodke|lgt|LLMGraphTransformer|neo4j|graphrag|planted|gold|dev|gate"
    words = "|".join([secret, *g.KINDS, *g.STATUSES, "extractor", "split"])
    assert not re.search(rf"\b({words})\b", sheets, re.IGNORECASE)
    # The sidecars beside the sheets hold the input rows and nothing more.
    sidecars = [
        json.loads(line)
        for path in sorted((tmp_path / "sheets").glob("*.items.jsonl"))
        for line in path.open()
    ]
    assert [s["id"] for s in sidecars] == [p["id"] for p in private]
    for side in sidecars:
        assert set(side["row"]) == {"text", "fact", "doc_id"}
        assert set(side["row"]["fact"]) <= {
            "id",
            "subject",
            "predicate",
            "object_entity",
            "object_value",
            "polarity",
            "evidence",
        }
        (evidence,) = side["row"]["fact"]["evidence"]
        assert evidence["span_origin"] == "context"
    assert "**" not in sheets


def test_the_same_seed_writes_the_same_bytes(cmp: tuple[Path, Path], tmp_path: Path) -> None:
    for name in ("a", "b"):
        g.write(g.make(g.load_all(*cmp), PLAN), tmp_path / name)
    g.write(g.make(g.load_all(*cmp), PLAN, seed=7), tmp_path / "c")
    files = ["items.jsonl", "items.private.jsonl", "gate.jsonl", "selfagreement.ids"]
    a, b, c = ({f: (tmp_path / n / f).read_bytes() for f in files} for n in "abc")
    assert a == b
    assert a["items.jsonl"] != c["items.jsonl"]


def test_a_long_passage_is_cut_to_its_evidence(made: Any) -> None:
    trimmed = [(r, p) for r, p in zip(made.rows, made.private, strict=True) if p["trimmed"]]
    assert trimmed and all(p["dataset"] == "redocred" for _, p in trimmed)
    for row, p in trimmed:
        assert p["sentences"] and len(row["text"]) < 160
        assert row["fact"]["evidence"][0]["span"]["end"] == len(row["text"])


def test_gold_names_are_spelled_as_the_text_spells_them() -> None:
    assert g.surface("Assassin 's Creed", "Assassin's Creed is a game.") == "Assassin's Creed"
    assert g.surface("Bleach : Hell Verse", "a film") == "Bleach: Hell Verse"
    assert g.surface("01 January 2010", "a 2010 film") == "2010"
    assert g.surface("00 June 1962", "a 1962 song") == "June 1962"
    assert g.surface("16 September 2008", "in 2008") == "16 September 2008"
