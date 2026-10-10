"""Public benchmark adapters: fetch, prepare, score — with no network and no model."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact
from openodke.cli.main import app
from openodke.eval.ablation import CONFIGURATIONS, AblationRun
from openodke.eval.datasets import DATASETS, redocred, text2kgbench
from openodke.eval.datasets._common import change, pascal, snake, triples_by_doc
from openodke.eval.eval_report import EvalReport, read_report
from openodke.eval.report import StageReport
from openodke.ontology import Ontology
from openodke.run.build import build
from openodke.run.config import load_config
from openodke.types import GroundingVerdict

ONTOLOGY = {
    "title": "Movie Ontology",
    "id": "ont_1_movie",
    "concepts": [
        {"qid": "Q5", "label": "human"},
        {"qid": "Q11424", "label": "film"},
        {"qid": "Q1762059", "label": "film production company"},
    ],
    "relations": [
        {"pid": "P57", "label": "director", "domain": "Q11424", "range": "Q5"},
        {"pid": "P577", "label": "publication date", "domain": "Q11424", "range": ""},
        {"pid": "P272", "label": "production company", "domain": "Q11424", "range": "Party"},
    ],
}
GOLD = [
    {
        "id": "ont_1_movie_test_1",
        "sent": "Bleach: Hell Verse is a 2010 Japanese animated film directed by Noriyuki Abe.",
        "triples": [
            {"sub": "Bleach : Hell Verse", "rel": "director", "obj": "Noriyuki Abe"},
            {"sub": "Bleach : Hell Verse", "rel": "publication date", "obj": "2010"},
        ],
    },
    {
        "id": "ont_1_movie_test_2",
        "sent": "Keyboard Cat was made in 1984 by Charlie Schmidt.",
        "triples": [{"sub": "Keyboard Cat", "rel": "director", "obj": "Charlie Schmidt"}],
    },
]


def test_names_become_types_and_predicates() -> None:
    assert pascal("film production company") == "FilmProductionCompany"
    assert pascal("University") == "University"
    assert snake("cast member") == "cast_member"
    assert snake("academicStaffSize") == "academic_staff_size"
    assert change(20, 13) == "-35%"
    assert change(0, 3) == "n/a"


def test_the_benchmark_ontology_loads_as_an_openodke_ontology() -> None:
    data, labels = text2kgbench.to_ontology(ONTOLOGY)
    ontology = Ontology.model_validate(data)
    assert set(ontology.types) == {"Human", "Film", "FilmProductionCompany", "Party"}
    assert ontology.predicates["director"].range == "Human"
    assert ontology.predicates["publication_date"].range == "string"  # no range given
    assert ontology.predicates["production_company"].range == "Party"  # not a concept: a type
    assert labels["publication_date"] == "publication date"


def test_a_label_listed_twice_is_one_predicate_with_both_domains() -> None:
    raw = {
        "id": "ont_3_sport",
        "concepts": [
            {"qid": "Q1", "label": "sports team"},
            {"qid": "Q2", "label": "athlete"},
            {"qid": "Q3", "label": "sports league"},
        ],
        "relations": [
            {"pid": "P118", "label": "league", "domain": "Q1", "range": "Q3"},
            {"pid": "P118", "label": "league", "domain": "Q2", "range": "Q3"},
        ],
    }
    data, labels = text2kgbench.to_ontology(raw)
    ontology = Ontology.model_validate(data)
    assert list(ontology.predicates) == ["league"]
    assert ontology.predicates["league"].domain == ("SportsTeam", "Athlete")
    assert labels == {"league": "league"}


def test_score_is_the_benchmarks_own_arithmetic() -> None:
    predicted = {
        # one right, one wrong object, one relation the gold never uses (filtered from P/R)
        "ont_1_movie_test_1": [
            ("Bleach: Hell Verse", "director", "Noriyuki Abe"),
            ("Bleach: Hell Verse", "publication_date", "2011"),
            ("Bleach: Hell Verse", "made_up", "Japan"),
        ],
        # nothing: scored as zeros, not skipped
    }
    m = text2kgbench.score(GOLD, predicted, ONTOLOGY)
    # sentence 1: P = 1/2 (the filtered pair), R = 1/2; sentence 2: 0, 0.
    assert m["precision"] == pytest.approx(0.25)
    assert m["recall"] == pytest.approx(0.25)
    assert m["f1"] == pytest.approx(0.25)
    # conformance: 2 of 3 in sentence 1; an empty answer conforms.
    assert m["onto_conf"] == pytest.approx((2 / 3 + 1) / 2)
    assert m["rel_halluc"] == pytest.approx((1 / 3) / 2)
    assert m["triples"] == 3
    # Hallucination is a substring test on stemmed text, as the benchmark's is:
    # the subject is in the sentence; "Noriyuki Abe" is, "2011" is not, and
    # "japan" is inside "japanes", the stem of "Japanese". Sentence 2 has no triples.
    assert m["sub_halluc"] == 0.0
    assert m["obj_halluc"] == pytest.approx((1 / 3) / 2)
    # "2011" (its object) and "made_up" (its relation).
    assert m["hallucinated_triples"] == 2


def test_hallucination_metrics_are_optional() -> None:
    m = text2kgbench.score(GOLD, {}, ONTOLOGY, hallucination=False)
    assert m["sub_halluc"] is None and m["obj_halluc"] is None


def _fake_opener(payloads: dict[str, bytes]) -> Any:
    def open_url(url: str) -> io.BytesIO:
        for suffix, body in payloads.items():
            if url.endswith(suffix):
                return io.BytesIO(body)
        raise OSError(f"no fixture for {url}")

    return open_url


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _t2k_raw(tmp_path: Path) -> Path:
    opener = _fake_opener(
        {
            "1_movie_ontology.json": json.dumps(ONTOLOGY).encode(),
            "ont_1_movie_ground_truth.jsonl": _jsonl(GOLD),
        }
    )
    return text2kgbench.fetch(tmp_path / "raw", ontologies=["ont_1_movie"], opener=opener)


def test_fetch_then_prepare_writes_a_config_that_builds(tmp_path: Path) -> None:
    # The ontology and the ground truth, which carries each sentence: the
    # benchmark's `test/` files are the same sentences without the triples.
    assert set(text2kgbench.files("wikidata_tekgen", "ont_1_movie")) == {"ontology", "gold"}
    root = _t2k_raw(tmp_path)
    out = text2kgbench.prepare(
        root, "ont_1_movie", tmp_path / "set", limit=1, ground_model="test/small", paper=True
    )
    assert sorted(p.name for p in (out / "docs").iterdir()) == ["ont_1_movie_test_1.txt"]
    built = build(load_config(out / "odke.json"))
    assert len(built.documents()) == 1
    grounder = built.stages["grounder"]
    assert grounder.context == "document" and grounder.binary
    assert GroundingVerdict.NOT_FOUND in built.stages["gate"].refused
    with pytest.raises(ValueError, match="not a wikidata_tekgen ontology"):
        text2kgbench.files("wikidata_tekgen", "ont_99_nothing")


def test_prepare_refuses_a_directory_it_did_not_write(tmp_path: Path) -> None:
    """`--out .` at a repository's root must not take the site's `docs/` with it."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "index.md").write_text("# the site\n")
    root = _t2k_raw(tmp_path)
    with pytest.raises(ValueError, match="dataset.json"):
        text2kgbench.prepare(root, "ont_1_movie", repo)
    args = ["bench", "prepare", "text2kgbench", str(root), "--ontology", "ont_1_movie"]
    result = CliRunner().invoke(app, [*args, "--out", str(repo)])
    assert result.exit_code == 2 and "dataset.json" in result.output
    assert [p.relative_to(repo).as_posix() for p in sorted(repo.rglob("*"))] == [
        "docs",
        "docs/index.md",
    ]
    assert (repo / "docs" / "index.md").read_text() == "# the site\n"


def test_prepare_again_replaces_its_own_documents(tmp_path: Path) -> None:
    root = _t2k_raw(tmp_path)
    out = text2kgbench.prepare(root, "ont_1_movie", tmp_path / "set")
    (out / "docs" / "notes.md").write_text("kept\n")
    text2kgbench.prepare(root, "ont_1_movie", out, limit=1)
    assert sorted(p.name for p in (out / "docs").iterdir()) == [
        "notes.md",
        "ont_1_movie_test_1.txt",
    ]


def _run(names: dict[str, str], candidates: list[Fact], gated: list[Fact]) -> AblationRun:
    docs = [Document(id=i, text="", uri=f"file:///set/docs/{n}.txt") for i, n in names.items()]
    return AblationRun(
        documents=docs,
        ontology=Ontology(),
        candidates=candidates,
        grounded=candidates,
        gated=gated,
        corroborated=gated,
        extraction_calls=[],
        all_calls=[],
        gate=None,
    )


def _fact(doc: str, subject: str, predicate: str, value: str) -> Fact:
    return Fact(
        subject=Entity(key=subject.lower(), type="Film", label=subject),
        predicate=predicate,
        object_value=value,
        evidence=(Evidence(doc_id=doc),),
    )


def test_an_ablation_run_is_scored_row_by_row_against_the_papers_numbers() -> None:
    right = _fact("d1", "Bleach : Hell Verse", "director", "Noriyuki Abe")
    wrong = _fact("d1", "Bleach : Hell Verse", "director", "Somebody Invented")
    run = _run({"d1": "ont_1_movie_test_1"}, [right, wrong], [right])
    _, labels = text2kgbench.to_ontology(ONTOLOGY)
    meta = {"ontology_id": "ont_1_movie", "relation_labels": labels, "ontology": ONTOLOGY}
    report = text2kgbench.score_run(run, GOLD[:1], meta)
    assert list(report.breakdown) == list(CONFIGURATIONS)
    assert report.breakdown[CONFIGURATIONS[0]]["precision"] == pytest.approx(0.5)
    assert report.breakdown[CONFIGURATIONS[1]]["precision"] == pytest.approx(1.0)
    assert "-100%" in report.notes[0] and "ODKE+ reports -35%" in report.notes[0]


def test_a_run_without_nltk_stops_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hallucination metrics need NLTK; finding that out after the bill is too late."""
    calls: list[Any] = []

    def ablate(config: Any) -> AblationRun:  # every model call a run makes is in here
        calls.append(config)
        return _run({}, [], [])

    monkeypatch.setattr(text2kgbench, "ablate", ablate)
    for name in ("nltk", "nltk.stem", "nltk.tokenize"):
        monkeypatch.setitem(sys.modules, name, None)
    out = text2kgbench.prepare(_t2k_raw(tmp_path), "ont_1_movie", tmp_path / "set")
    with pytest.raises(ImportError, match=r"openodke\[bench\]"):
        text2kgbench.run(out)
    result = CliRunner().invoke(app, ["bench", "run", "text2kgbench", str(out)])
    assert result.exit_code == 2 and "openodke[bench]" in result.output
    assert calls == []


def test_triples_use_labels_and_land_in_every_cited_document() -> None:
    fact = _fact("d1", "Keyboard Cat", "director", "Charlie Schmidt")
    fact = fact.model_copy(update={"evidence": (Evidence(doc_id="d1"), Evidence(doc_id="d2"))})
    got = triples_by_doc([fact], {"d1": "a", "d2": "b"}, lambda p: p.replace("_", " "))
    assert got == {
        "a": [("Keyboard Cat", "director", "Charlie Schmidt")],
        "b": [("Keyboard Cat", "director", "Charlie Schmidt")],
    }


# --------------------------------------------------------------------------- #
# Re-DocRED
# --------------------------------------------------------------------------- #

DOC = {
    "title": "Rihanna",
    "sents": [
        ["Rihanna", "was", "born", "in", "Saint", "Michael", ",", "Barbados", "."],
        ["She", "did", "n't", "stop", "(", "ever", ")", "."],
    ],
    "vertexSet": [
        [{"name": "Rihanna", "type": "PER", "sent_id": 0, "pos": [0, 1]}],
        [{"name": "Saint Michael", "type": "LOC", "sent_id": 0, "pos": [4, 6]}],
        [{"name": "Barbados", "type": "LOC", "sent_id": 0, "pos": [7, 8]}],
    ],
    "labels": [
        {"h": 0, "t": 1, "r": "P19", "evidence": [0]},
        {"h": 1, "t": 2, "r": "P17", "evidence": [0]},
    ],
}


def test_redocred_text_reads_as_prose() -> None:
    assert redocred.detokenize(DOC["sents"]) == (
        "Rihanna was born in Saint Michael, Barbados. She didn't stop (ever)."
    )


def test_redocred_ontology_takes_types_from_the_labels() -> None:
    ontology = Ontology.model_validate(redocred.to_ontology([DOC]))
    assert len(ontology.predicates) == 96
    assert ontology.predicates["place_of_birth"].domain == ("Person",)
    assert ontology.predicates["place_of_birth"].range == "Location"


def test_redocred_score_matches_any_mention_and_counts_each_fact_once(tmp_path: Path) -> None:
    opener = _fake_opener(
        {
            "dev_revised.json": json.dumps([DOC]).encode(),
            "test_revised.json": json.dumps([DOC]).encode(),
        }
    )
    root = redocred.fetch(tmp_path / "raw", opener=opener)
    out = redocred.prepare(root, tmp_path / "set")
    (gold,) = [json.loads(line) for line in (out / "gold.jsonl").read_text().splitlines()]
    predicted = {
        gold["id"]: [
            ("rihanna", "place of birth", "Saint Michael"),  # right, case ignored
            ("Rihanna", "place_of_birth", "Saint Michael"),  # same fact again: not counted twice
            ("Saint Michael", "country", "Narnia"),  # wrong, and Narnia is not in the text
        ]
    }
    m = redocred.score([gold], predicted)
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["hallucinated_triples"] == 1
    assert build(load_config(out / "odke.json")).documents()[0].text.startswith("Rihanna was born")


def test_a_prepared_redocred_set_is_diagnosed_from_the_shell(tmp_path: Path) -> None:
    """Types and evidence go into the gold, and `odke eval pipeline --bench` reads them (#140)."""
    opener = _fake_opener(
        {
            "dev_revised.json": json.dumps([DOC]).encode(),
            "test_revised.json": json.dumps([DOC]).encode(),
        }
    )
    out = redocred.prepare(redocred.fetch(tmp_path / "raw", opener=opener), tmp_path / "set")
    (gold,) = [json.loads(line) for line in (out / "gold.jsonl").read_text().splitlines()]
    assert (gold["types"], gold["evidence"]) == (["Person", "Location", "Location"], [[0], [0]])
    rows = tmp_path / "rows.jsonl"
    said = [
        {
            "doc": gold["id"],
            "subject": "Rihanna",
            "predicate": "place of birth",
            "object": "Saint Michael",
        },
        {
            "doc": gold["id"],
            "subject": "Barbados",
            "predicate": "country",
            "object": "Saint Michael",
        },
    ]
    rows.write_text("".join(json.dumps(r) + "\n" for r in said))
    args = ["eval", "pipeline", "--bench", str(out), "--predictions", str(rows)]
    args += ["--trace", str(out / "odke.json"), "--json"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    report = EvalReport.model_validate_json(result.output[result.output.index("{") :])
    found = {b.bucket: b.count for b in report.diagnosis if b.side == "recall"}
    # Every relation was offered; the country fact was written the other way round.
    assert (found["never_offered"], found["inverse"]) == (0, 1)
    assert sum(n or 0 for n in found.values()) == report.rows[0].counts.under_extraction == 1


def test_the_cli_names_the_datasets_and_refuses_an_unknown_one(tmp_path: Path) -> None:
    assert set(DATASETS) == {"text2kgbench", "redocred"}
    result = CliRunner().invoke(app, ["bench", "prepare", "nothing", str(tmp_path), "--out", "x"])
    assert result.exit_code == 2
    assert "unknown dataset" in result.output


# --------------------------------------------------------------------------- #
# The eval report (#139)
# --------------------------------------------------------------------------- #


def test_text2kgbench_score_is_the_mean_of_its_sentences() -> None:
    predicted = {"ont_1_movie_test_1": [("Bleach: Hell Verse", "director", "Noriyuki Abe")]}
    units = text2kgbench.sentences(GOLD, predicted, ONTOLOGY)
    assert [u["f1"] for u in units] == [pytest.approx(2 / 3), 0.0]
    assert [(u["hits"], u["over"], u["under"]) for u in units] == [(1, 0, 1), (0, 0, 1)]
    assert text2kgbench.aggregate(units) == text2kgbench.score(GOLD, predicted, ONTOLOGY)


def test_a_text2kgbench_row_is_the_benchmark_s_metrics_with_ranges() -> None:
    right = _fact("d1", "Bleach : Hell Verse", "director", "Noriyuki Abe")
    wrong = _fact("d1", "Bleach : Hell Verse", "publication_date", "2011")
    off = _fact("d2", "Keyboard Cat", "made_up", "Charlie Schmidt")
    names = {"d1": "ont_1_movie_test_1", "d2": "ont_1_movie_test_2"}
    run = _run(names, [right, wrong, off], [right])
    _, labels = text2kgbench.to_ontology(ONTOLOGY)
    meta = {"ontology_id": "ont_1_movie", "relation_labels": labels, "ontology": ONTOLOGY}
    report = text2kgbench.report_run(run, GOLD, meta)
    (stage,) = report.stages
    assert stage == text2kgbench.score_run(run, GOLD, meta)
    for row in report.rows:
        line = stage.breakdown[row.name]
        assert row.performance.average == "macro"
        assert row.performance.precision.value == line["precision"]
        assert row.performance.f1.value == line["f1"]
        assert row.conformance is not None and row.conformance.rate == line["onto_conf"]
        assert row.hallucination is not None
        assert row.hallucination.hallucinated == line["hallucinated_triples"]
        assert row.hallucination.subject == line["sub_halluc"]
    extracted = report.rows[0]
    # Sentence 1: a hit and a wrong date (one over, one under); sentence 2: its
    # director missed, and `made_up` unscored, because its gold never uses it.
    c = extracted.counts
    assert (c.hits, c.over_extraction, c.under_extraction, c.unscored) == (1, 1, 2, 1)
    assert extracted.conformance is not None
    assert (extracted.conformance.conformant, extracted.conformance.facts) == (2, 3)
    assert report.bootstrap is not None and report.bootstrap.units == 2
    assert report.run.dataset is not None and report.run.dataset.labels == 3


def test_redocred_score_is_its_documents_pooled() -> None:
    gold = [
        {
            "id": "d",
            "text": "Rihanna was born in Saint Michael, Barbados.",
            "entities": [["Rihanna"], ["Saint Michael"], ["Barbados"]],
            "facts": [[0, "place of birth", 1], [1, "country", 2]],
        }
    ]
    predicted = {"d": [("Rihanna", "place of birth", "Saint Michael"), ("x", "country", "y")]}
    (unit,) = redocred.documents(gold, predicted)
    assert (unit["tp"], unit["predicted"], unit["gold"], unit["hallucinated"]) == (1, 2, 2, 1)
    assert redocred.aggregate([unit]) == redocred.score(gold, predicted)
    triples = {"d": [("Rihanna", "place of birth", "Saint Michael")]}
    fact = _fact("d1", "Rihanna", "place_of_birth", "Saint Michael")
    run = _run({"d1": "d"}, [fact], [fact])
    meta = {"split": "test", "relation_labels": {"place_of_birth": "place of birth"}}
    report = redocred.report_run(run, gold, meta)
    row = report.rows[0]
    assert redocred.score(gold, triples)["precision"] == row.performance.precision.value == 1.0
    assert (row.counts.hits, row.counts.under_extraction) == (1, 1)
    assert row.performance.average == "micro"


def test_odke_bench_run_writes_the_report_beside_the_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    right = _fact("d1", "Bleach : Hell Verse", "director", "Noriyuki Abe")
    names = {"d1": "ont_1_movie_test_1", "d2": "ont_1_movie_test_2"}
    monkeypatch.setattr(text2kgbench, "ablate", lambda config: _run(names, [right], [right]))
    prepared = text2kgbench.prepare(_t2k_raw(tmp_path), "ont_1_movie", tmp_path / "set")

    result = CliRunner().invoke(app, ["bench", "run", "text2kgbench", str(prepared)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("text2kgbench:ont_1_movie  (n=2)\n")
    assert result.output.rstrip().endswith(f"wrote {prepared / 'report.json'}")
    report = read_report(prepared / "report.json")
    assert report.run.dataset is not None and report.run.dataset.name == "text2kgbench"
    assert report.rows[0].counts.hits == 1
    assert (prepared / "predictions" / "extraction-alone.jsonl").is_file()

    elsewhere = tmp_path / "elsewhere.json"
    args = ["bench", "run", "text2kgbench", str(prepared), "--json", "--report", str(elsewhere)]
    as_json = CliRunner().invoke(app, args)
    assert as_json.exit_code == 0, as_json.output
    assert StageReport.model_validate_json(as_json.output).stage == "text2kgbench:ont_1_movie"
    assert read_report(elsewhere).title == "text2kgbench:ont_1_movie"
