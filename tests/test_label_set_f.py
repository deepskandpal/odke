"""`bench/labels/make_f.py`: label set F drawn from a tiny made-up comparison. No network.

The comparison is prepared by the datasets' own `prepare`, from a handful of
invented Text2KGBench sentences and Re-DocRED documents, with three
extractors' triples written beside it the way `bench/run_all.sh` leaves them.
Each extractor misses a gold fact in a way the pre-filter lets through: a short
form of a name, another name for the subject, another value for the object.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openodke.eval.datasets import redocred, text2kgbench
from openodke.eval.equivalence import surface_pair
from openodke.eval.formats import FactPair
from openodke.eval.sheets import make_sheets

SCRIPT = Path(__file__).parent.parent / "bench" / "labels" / "make_f.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("odke_bench_make_f", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses look their module up while it loads.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


f = _load()
g = f.g
PLAN = f.Plan(per_dataset=9, dev=6, reuse=2)

FILMS = ["Aster", "Birch", "Cedar", "Dahlia", "Elder", "Fennel"]
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
            {"pid": "P161", "label": "cast member", "domain": "Q11424", "range": "Q5"},
            {"pid": "P577", "label": "publication date", "domain": "Q11424", "range": ""},
        ],
    }
    (source / "ontologies").mkdir(parents=True)
    (source / "ontologies" / "1_movie_ontology.json").write_text(json.dumps(ontology))
    rows: list[dict[str, Any]] = []
    by: dict[str, dict[str, list[list[str]]]] = {name: {} for name in g.EXTRACTORS}
    for i, film in enumerate(FILMS, start=1):
        doc, title, year = f"ont_1_movie_test_{i}", f"{film} Story", str(1990 + i)
        boss, star, lead = f"Dana {film}son", f"Ezra {film}ley", f"Ivy {film}ton"
        rows.append(
            {
                "id": doc,
                "sent": f"{title} is a {year} film directed by {boss} and starring "
                f"{star} and {lead}.",
                "triples": [
                    {"sub": title, "rel": "director", "obj": boss},
                    {"sub": title, "rel": "cast member", "obj": star},
                    {"sub": title, "rel": "cast member", "obj": lead},
                    {"sub": title, "rel": "publication date", "obj": f"01 January {year}"},
                ],
            }
        )
        # A short form of the director; and the year the gold pads, which reads
        # the same once the gold is spelled as the text spells it: dropped.
        by["openodke"][doc] = [
            [title, "director", f"D. {film}son"],
            [title, "publication date", year],
        ]
        # The star is right; the director is not in the cast.
        by["lgt"][doc] = [[title, "cast member", star], [title, "cast member", boss]]
        # Another name for the film.
        by["neo4j"][doc] = [["the film", "director", boss], [title, "cast member", lead]]
    _jsonl(source / "ground_truth" / "ont_1_movie_ground_truth.jsonl", rows)
    out = text2kgbench.prepare(root / "t2k", "ont_1_movie", cmp / "t2k" / "ont_1_movie")
    _predictions(out, by)


def _mention(name: str, sent: int, at: int, kind: str) -> dict[str, Any]:
    return {"name": name, "sent_id": sent, "pos": [at, at + len(name.split())], "type": kind}


def _redocred(root: Path, cmp: Path) -> None:
    docs = []
    by: dict[str, dict[str, list[list[str]]]] = {name: {} for name in g.EXTRACTORS}
    for i, (town, land) in enumerate(zip(TOWNS, LANDS, strict=True)):
        person, works = f"Mara {town}er", f"{land} Works"
        sents = [
            [*person.split(), "was", "born", "in", town, "."],
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
                    [_mention(land, 1, 5, "LOC")],
                    [_mention(works, 2, 4, "ORG"), _mention(works, 3, 0, "ORG")],
                ],
                "labels": [
                    {"r": "P19", "h": 0, "t": 1, "evidence": [0]},
                    {"r": "P17", "h": 1, "t": 2, "evidence": [1]},
                    {"r": "P108", "h": 0, "t": 3, "evidence": [2]},
                    {"r": "P17", "h": 3, "t": 2, "evidence": []},
                ],
            }
        )
        doc = f"test_{i:04d}"
        by["openodke"][doc] = [[f"M. {town}er", "place of birth", town]]
        by["lgt"][doc] = [[works, "country", town]]
        by["neo4j"][doc] = [[person, "employer", town]]
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
def sets(cmp: tuple[Path, Path]) -> list[Any]:
    return list(g.load_all(*cmp))


@pytest.fixture(scope="module")
def made(sets: list[Any]) -> Any:
    return f.make(sets, PLAN)


@pytest.fixture(scope="module")
def written(made: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("F")
    f.write(made, out)
    return out


def test_each_dataset_and_extractor_gets_its_share_as_the_data_allows(made: Any) -> None:
    counts = Counter((p["dataset"], p["extractor"]) for p in made.private)
    assert len(made.private) == 2 * PLAN.per_dataset
    # Three a cell where the data has three; Text2KGBench's openodke has six pairs.
    assert counts == {
        (dataset, extractor): 3 for dataset in f.DATASETS for extractor in f.EXTRACTORS
    }
    assert made.candidates == {
        ("redocred", "lgt"): 6,
        ("redocred", "neo4j"): 6,
        ("redocred", "openodke"): 6,
        ("text2kgbench", "lgt"): 6,
        ("text2kgbench", "neo4j"): 6,
        ("text2kgbench", "openodke"): 6,
    }
    # The padded year reads the same once the gold is spelled as the text spells it.
    assert not [p for p in made.private if p["gold"][1] == "publication date"]


def test_every_pair_is_a_miss_the_pre_filter_lets_through(made: Any, sets: list[Any]) -> None:
    by_name = {s.name: s for s in sets}
    for p in made.private:
        s = by_name[p["set"]]
        written = [tuple(t) for t in s.gold(p["doc"], as_written=True)]
        index = written.index(tuple(p["gold"]))
        said = s.predicted[p["extractor"]][p["doc"]]
        assert s.score(p["doc"], said, index) < 1, p  # the extractor missed the gold fact
        assert s.matched(p["doc"], tuple(p["predicted"])) is None, p  # and its triple is no hit
        ends = f._ends(s, p["doc"], index)
        assert surface_pair(ends, tuple(p["predicted"])) == p["differs"], p
        assert tuple(map(g.fold, p["predicted"])) != tuple(map(g.fold, p["gold_shown"]))


def test_no_gold_fact_or_prediction_past_its_reuse(made: Any) -> None:
    golds = Counter((p["dataset"], p["doc"], tuple(p["gold"])) for p in made.private)
    said = Counter((p["dataset"], p["doc"], tuple(p["predicted"])) for p in made.private)
    pairs = Counter((p["doc"], tuple(p["gold"]), tuple(p["predicted"])) for p in made.private)
    assert max(golds.values()) <= PLAN.reuse and max(said.values()) <= PLAN.reuse
    assert max(pairs.values()) == 1


def test_a_shortfall_comes_from_the_other_extractors_then_the_other_dataset(
    sets: list[Any],
) -> None:
    made = f.make(sets, f.Plan(per_dataset=15, dev=6, reuse=1))
    counts = Counter(p["dataset"] for p in made.private)
    # Text2KGBench's two directors are one gold fact: once each, 12 of its 18
    # pairs. Re-DocRED draws its 15, then the 3 it has left make up the rest.
    assert counts == {"text2kgbench": 12, "redocred": 18}
    assert "text2kgbench: 12 of 15; 3 more from redocred" in made.notes


def test_the_sides_are_drawn_per_item_and_kept_private(made: Any) -> None:
    sides = Counter(p["gold_side"] for p in made.private)
    assert set(sides) == {"first", "second"}
    for row, p in zip(made.rows, made.private, strict=True):
        pair = FactPair.model_validate(row)
        gold = pair.first if p["gold_side"] == "first" else pair.second
        said = pair.second if p["gold_side"] == "first" else pair.first
        assert gold.subject.label == p["gold_shown"][0]
        assert said.subject.label == p["predicted"][0]
        assert row["id"] == p["fact_id"]


def test_dev_and_gate_never_share_a_document(made: Any, written: Path) -> None:
    splits: dict[str, set[str]] = {}
    for p in made.private:
        splits.setdefault(p["doc"], set()).add(p["split"])
    assert all(len(found) == 1 for found in splits.values())
    assert sum(p["split"] == "dev" for p in made.private) == PLAN.dev
    gate = [json.loads(line) for line in (written / "gate.jsonl").read_text().splitlines()]
    assert {row["doc"] for row in gate} == {p["doc"] for p in made.private if p["split"] == "gate"}
    assert all(row["text"] for row in gate)


def test_the_passage_is_the_gold_fact_s_evidence(made: Any) -> None:
    for row, p in zip(made.rows, made.private, strict=True):
        if p["dataset"] == "redocred" and p["gold"][1] == "place of birth":
            assert row["passage"] == f"{p['gold_shown'][0]} was born in {p['gold_shown'][2]}."
            assert p["sentences"] == [0]
        if p["dataset"] == "redocred" and p["gold"][0].endswith("Works"):
            # No evidence listed: where both ends are named, or the rarer one is.
            assert p["sentences"] == [1] and " … " not in row["passage"]


def test_the_sheets_show_nothing_private(written: Path, tmp_path: Path) -> None:
    made = make_sheets("fact", written / "items.jsonl", tmp_path / "sheets", per_sheet=5)
    assert made.items == 2 * PLAN.per_dataset and made.first == "F-0001"
    text = "\n".join(sheet.read_text(encoding="utf-8") for sheet in made.sheets)
    for word in ("gold", "openodke", "lgt", "neo4j", "extractor", "dev", "gate", "stratum"):
        assert word not in text.lower(), word
    private = [
        json.loads(line) for line in (written / "items.private.jsonl").read_text().splitlines()
    ]
    assert [p["id"] for p in private] == [f"F-{n:04d}" for n in range(1, len(private) + 1)]


def test_the_same_seed_writes_the_same_bytes(
    sets: list[Any], written: Path, tmp_path: Path
) -> None:
    again = tmp_path / "again"
    f.write(f.make(sets, PLAN), again)
    for name in ("items.jsonl", "items.private.jsonl", "gate.jsonl"):
        assert (again / name).read_bytes() == (written / name).read_bytes(), name
    other = tmp_path / "other"
    f.write(f.make(sets, PLAN, seed=1), other)
    assert (other / "items.jsonl").read_bytes() != (written / "items.jsonl").read_bytes()
