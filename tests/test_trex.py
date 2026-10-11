"""T-REx: the multi-source benchmark (#117), on a synthetic file — no network, no model."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from openodke import Document, Entity, EntityLink, Evidence, Fact
from openodke.cli.main import app
from openodke.eval.ablation import CONFIGURATIONS, AblationRun
from openodke.eval.datasets import DATASETS, trex
from openodke.eval.datasets._common import read_jsonl
from openodke.ontology import Ontology
from openodke.run.build import build
from openodke.run.config import load_config
from openodke.types import LinkKind

E = trex.ENTITY
LABELS = {
    "P30": {"label": "continent", "description": "continent of which the subject is a part"},
    "P47": {"label": "shares border with", "description": "countries or regions that border"},
}


def _mention(qid: str, name: str, annotator: str = trex.LINKER) -> dict[str, Any]:
    return {"boundaries": [0, 1], "surfaceform": name, "uri": E + qid, "annotator": annotator}


def _triple(s: str, p: str, o: str, aligner: str, *, literal: bool = False) -> dict[str, Any]:
    obj = o if literal else E + o
    return {
        "sentence_id": 0,
        "subject": {"uri": E + s, "surfaceform": s, "annotator": trex.LINKER},
        "predicate": {"uri": f"http://www.wikidata.org/prop/direct/{p}", "annotator": aligner},
        "object": {"uri": obj, "surfaceform": o, "annotator": trex.LINKER},
        "annotator": aligner,
    }


def _doc(qid: str, title: str, text: str, mentions: list, triples: list) -> dict[str, Any]:
    return {
        "docid": E + qid,
        "uri": E + qid,
        "title": title,
        "text": text,
        "sentences_boundaries": [[0, len(text)]],
        "words_boundaries": [],
        "entities": mentions,
        "triples": triples,
    }


SPO, NOSUB, SIMPLE = "SPOAligner", "NoSubject-Triple-aligner", "Simple-Aligner"
# Five short abstracts about Iberia and a long one. Spain borders Portugal in
# three of them; France's continent, Spain's and Portugal's are each in two.
DOCS = [
    _doc(
        "Q142",
        "France",
        "France is a country in Europe. France shares a border with Spain.",
        [_mention("Q142", "France"), _mention("Q46", "Europe"), _mention("Q29", "Spain")],
        [_triple("Q142", "P30", "Q46", NOSUB), _triple("Q142", "P47", "Q29", SPO)],
    ),
    _doc(
        "Q29",
        "Spain",
        "Spain is a country in Europe. Spain shares a border with France and Portugal.",
        [
            _mention("Q29", "Spain"),
            _mention("Q29", "It", "Simple_Coreference"),
            _mention("Q46", "Europe"),
            _mention("Q142", "France"),
            _mention("Q45", "Portugal"),
        ],
        [
            _triple("Q29", "P30", "Q46", NOSUB),
            _triple("Q29", "P47", "Q142", SPO),
            _triple("Q29", "P47", "Q45", SPO),
            # A co-mention: checked aligners would not count it, every aligner would.
            _triple("Q142", "P47", "Q29", SIMPLE),
        ],
    ),
    _doc(
        "Q45",
        "Portugal",
        "Portugal shares a border with the Kingdom of Spain. Portugal is in Europe.",
        [
            _mention("Q45", "Portugal"),
            _mention("Q29", "Kingdom of Spain"),
            _mention("Q46", "Europe"),
        ],
        [
            _triple("Q45", "P47", "Q29", SPO),
            _triple("Q45", "P30", "Q46", NOSUB),
            _triple("Q29", "P47", "Q45", SPO),
        ],
    ),
    _doc(
        "Q46",
        "Europe",
        "Europe is a continent. Spain, France and Portugal are in Europe.",
        [
            _mention("Q46", "Europe"),
            _mention("Q29", "Spain"),
            _mention("Q142", "France"),
            _mention("Q45", "Portugal"),
        ],
        [
            _triple("Q29", "P30", "Q46", SPO),
            _triple("Q142", "P30", "Q46", SPO),
            _triple("Q45", "P30", "Q46", SPO),
        ],
    ),
    _doc(
        "Q12837",
        "Iberian Peninsula",
        "The Iberian Peninsula holds Spain and Portugal, and Spain shares a border with Portugal.",
        [
            _mention("Q12837", "Iberian Peninsula"),
            _mention("Q29", "Spain"),
            _mention("Q45", "Portugal"),
        ],
        [
            _triple("Q29", "P47", "Q45", SPO),
            _triple("Q12837", "P2046", "583254", SPO, literal=True),
        ],
    ),
    _doc(
        "Q999",
        "A long history",
        "Spain shares a border with France. " * 90,
        [_mention("Q29", "Spain"), _mention("Q142", "France")],
        [_triple("Q29", "P47", "Q142", SPO)],
    ),
]


def _zip(docs: list[dict[str, Any]]) -> bytes:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        archive.writestr("re-nlg_0-10000.json", json.dumps(docs))
    return raw.getvalue()


def _opener(asked: list[str]) -> Any:
    def open_url(url: str) -> io.BytesIO:
        asked.append(url)
        if url == trex.SAMPLE_URL:
            return io.BytesIO(_zip(DOCS))
        ids = parse_qs(urlsplit(url).query)["ids"][0].split("|")
        entities = {
            pid: {
                "labels": {"en": {"value": LABELS[pid]["label"]}},
                "descriptions": {"en": {"value": LABELS[pid]["description"]}},
            }
            for pid in ids
            if pid in LABELS
        }
        return io.BytesIO(json.dumps({"entities": entities}).encode())

    return open_url


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    asked: list[str] = []
    root = trex.fetch(tmp_path / "raw", opener=_opener(asked))
    assert asked[0] == trex.SAMPLE_URL and "wbgetentities" in asked[1]
    return root


def test_the_array_is_read_an_object_at_a_time() -> None:
    text = json.dumps([{"a": "x" * 50}, {"b": [1, 2, {"c": "]"}]}, {}])
    assert list(trex.objects(io.StringIO(text), size=7)) == [
        {"a": "x" * 50},
        {"b": [1, 2, {"c": "]"}]},
        {},
    ]


def test_fetch_downloads_the_sample_and_labels_its_properties(raw: Path) -> None:
    assert {p.name for p in raw.iterdir()} == {trex.PROPERTIES, trex.SAMPLE}
    labels = json.loads((raw / trex.PROPERTIES).read_text())
    assert labels["P47"]["label"] == "shares border with"
    assert "P2046" not in labels  # Wikidata did not answer for it; prepare falls back to the id
    asked: list[str] = []
    trex.fetch(raw, opener=_opener(asked))
    assert asked == []  # both files are there already


def test_share_counts_abstracts_per_fact_with_every_aligner_and_the_checked_two(raw: Path) -> None:
    found = trex.share(raw)
    # Checked: Spain-Portugal in three abstracts; four facts in two; two in one.
    assert {k: found["checked"][k] for k in ("facts", "1", "2", "3+")} == {
        "facts": 7,
        "1": 2,
        "2": 4,
        "3+": 1,
    }
    assert found["checked"]["multi_share"] == pytest.approx(5 / 7, abs=1e-4)
    # The co-mention makes France-Spain a second fact in two abstracts.
    assert (found["all"]["1"], found["all"]["2"]) == (1, 5)


def test_prepare_counts_each_facts_sources_inside_the_set(raw: Path, tmp_path: Path) -> None:
    out = trex.prepare(raw, tmp_path / "set", documents=5, ground_model="test/small", paper=True)
    meta = json.loads((out / "dataset.json").read_text())
    # The long abstract is left out, so Spain-France has one source in the set.
    assert sorted(p.stem for p in (out / "docs").iterdir()) == [
        "Q12837",
        "Q142",
        "Q29",
        "Q45",
        "Q46",
    ]
    assert meta["gold_facts"] == {"1": 3, "2": 3, "3+": 1}
    assert meta["share"]["checked"]["multi_source"] == 5
    rows = {row["id"]: row for row in read_jsonl(out / "gold.jsonl")}
    spain = rows["Q29"]
    facts = dict(zip(map(tuple, spain["facts"]), spain["sources"], strict=True))
    assert facts == {
        ("Q29", "continent", "Q46"): 2,
        ("Q29", "shares border with", "Q142"): 1,
        ("Q29", "shares border with", "Q45"): 3,
    }
    # Names come from linked mentions anywhere in the file, and titles; never "It".
    assert spain["entities"]["Q29"] == ["Kingdom of Spain", "Spain"]
    # Known facts are every aligner's, the co-mention included.
    assert ["Q142", "shares border with", "Q29"] in spain["known"]
    ontology = Ontology.model_validate(json.loads((out / "ontology.json").read_text()))
    assert set(ontology.types) == {"Entity"}
    assert ontology.predicates["shares_border_with"].range == "Entity"
    built = build(load_config(out / "odke.json"))
    assert len(built.documents()) == 5
    assert built.stages["resolver"] is not None and built.stages["resolver"].judge is None


def test_a_smaller_set_keeps_the_facts_it_picked_first(raw: Path, tmp_path: Path) -> None:
    out = trex.prepare(raw, tmp_path / "set", documents=3, judge=True, ground_model="test/small")
    meta = json.loads((out / "dataset.json").read_text())
    assert sorted(p.stem for p in (out / "docs").iterdir()) == ["Q12837", "Q29", "Q45"]
    assert meta["gold_facts"] == {"1": 4, "2": 0, "3+": 1}
    config = json.loads((out / "odke.json").read_text())
    assert config["stages"]["resolver"] == {"use": "native", "judge": True}
    with pytest.raises(ValueError, match="unknown aligner"):
        trex.prepare(raw, tmp_path / "other", aligners=("Guess-Aligner",))
    with pytest.raises(FileNotFoundError, match="fetch first"):
        trex.prepare(tmp_path / "empty", tmp_path / "other")


def test_score_is_gold_recall_by_sources_and_factual_precision(raw: Path, tmp_path: Path) -> None:
    out = trex.prepare(raw, tmp_path / "set", documents=5)
    gold = read_jsonl(out / "gold.jsonl")
    predicted = {
        "Q29": [
            ("Spain", "shares border with", "Portugal"),  # gold here, three sources
            ("spain", "Shares_Border_With", "FRANCE"),  # gold here, one source
            ("France", "shares border with", "Spain"),  # true, not this abstract's gold
            ("Spain", "continent", "Asia"),  # T-REx never linked "Asia"
        ],
        "Q45": [("Portugal", "shares border with", "Kingdom of Spain")],
    }
    found = trex.score(gold, predicted)
    assert found["precision"] == pytest.approx(3 / 5)
    assert found["precision_factual"] == pytest.approx(4 / 5)
    assert found["checkable"] == pytest.approx(4 / 5)
    assert found["recall_3+"] == pytest.approx(1 / 3)  # found in Spain, not Portugal or Iberia
    assert found["recall_1"] == pytest.approx(2 / 3)
    assert found["recall_2"] == 0.0
    assert found["recall"] == pytest.approx(3 / 12)


def _fact(doc: str, s: str, p: str, o: str, support: int = 1) -> Fact:
    return Fact(
        subject=Entity(key=f"Entity:{s.lower()}", type="Entity", label=s),
        predicate=p,
        object_entity=Entity(key=f"Entity:{o.lower()}", type="Entity", label=o),
        evidence=(Evidence(doc_id=doc),),
        support=support,
    )


def test_corroboration_on_scores_the_graph_by_support(raw: Path, tmp_path: Path) -> None:
    out = trex.prepare(raw, tmp_path / "set", documents=5)
    meta = json.loads((out / "dataset.json").read_text())
    read = trex.Gold(read_jsonl(out / "gold.jsonl"), meta["relation_labels"].values())
    border = "shares_border_with"
    both = (Evidence(doc_id="d1"), Evidence(doc_id="d2"))
    gated = [
        _fact("d1", "Spain", border, "Portugal"),
        _fact("d2", "Kingdom of Spain", border, "Portugal"),
        _fact("d1", "Spain", border, "Andorra"),
        _fact("d1", "Spain", border, "France"),
        _fact("d2", "Spain", border, "France"),
    ]
    merged = [
        *gated[:3],
        _fact("d1", "Spain", border, "France", 2).model_copy(update={"evidence": both}),
    ]
    graph = trex.corroboration(read, gated, merged, meta["relation_labels"])
    assert graph["off"]["edges"] == 4 and graph["off"]["precision_factual"] == 0.75
    assert graph["on_two_or_more"]["edges"] == 1
    assert graph["on_two_or_more"]["precision_factual"] == 1.0
    assert graph["on_two_or_more"]["recall"]["1"] == {"found": 1, "gold": 3, "recall": 1 / 3}
    assert graph["by_support"]["1"]["precision_factual"] == pytest.approx(2 / 3)
    # Spain-Portugal is stated twice under two names: one source each by name, two pooled.
    assert graph["counted"]["3+"] == {"found": 1, "two_or_more": 0, "two_or_more_pooled": 1}
    assert graph["counted"]["1"] == {"found": 1, "two_or_more": 1, "two_or_more_pooled": 1}


def test_links_are_scored_against_the_ids(raw: Path, tmp_path: Path) -> None:
    out = trex.prepare(raw, tmp_path / "set", documents=5)
    read = trex.Gold(read_jsonl(out / "gold.jsonl"))
    facts = [
        _fact("d1", "Spain", "p", "Kingdom of Spain"),
        _fact("d1", "Portugal", "p", "Atlantis"),
    ]
    keys = {
        name: f"Entity:{name.lower()}"
        for name in ("Spain", "Kingdom of Spain", "Portugal", "Atlantis")
    }
    judged = "pair judge (pair@1, test/small): same in both orders"
    found = [
        EntityLink(
            source_key=keys["Spain"], target_key=keys["Kingdom of Spain"], kind=LinkKind.SIMILAR
        ),
        EntityLink(
            source_key=keys["Spain"],
            target_key=keys["Portugal"],
            kind=LinkKind.SIMILAR,
            reason=judged,
        ),
        EntityLink(source_key=keys["Spain"], target_key=keys["Portugal"], kind=LinkKind.DIFFERENT),
        EntityLink(source_key=keys["Portugal"], target_key=keys["Atlantis"], kind=LinkKind.SIMILAR),
    ]
    scored = trex.links(read, found, facts)
    assert scored["similar (rules)"] == {
        "links": 2,
        "right": 1,
        "wrong": 0,
        "uncheckable": 1,
        "precision": 1.0,
    }
    assert scored["similar (judge)"]["wrong"] == 1
    assert scored["different (rules)"]["right"] == 1


def test_a_run_is_scored_with_the_graph_in_its_notes(raw: Path, tmp_path: Path) -> None:
    out = trex.prepare(raw, tmp_path / "set", documents=5)
    meta = json.loads((out / "dataset.json").read_text())
    gold = read_jsonl(out / "gold.jsonl")
    docs = [
        Document(id=f"d{i}", text="", uri=f"file:///set/docs/{q}.txt")
        for i, q in enumerate(["Q29", "Q45"])
    ]
    border = "shares_border_with"
    candidates = [
        _fact("d0", "Spain", border, "Portugal"),
        _fact("d1", "Portugal", border, "Spain"),
    ]
    merged = [_fact("d0", "Spain", border, "Portugal")]
    link = EntityLink(
        source_key="Entity:spain", target_key="Entity:portugal", kind=LinkKind.DIFFERENT
    )
    run = AblationRun(
        documents=docs,
        ontology=Ontology(),
        candidates=candidates,
        grounded=candidates,
        gated=candidates,
        corroborated=merged,
        extraction_calls=[],
        all_calls=[],
        gate=None,
        links=(link,),
    )
    report = trex.score_run(run, gold, meta, save_to=out)
    assert list(report.breakdown) == list(CONFIGURATIONS)
    assert report.breakdown[CONFIGURATIONS[0]]["precision"] == 1.0
    assert any(
        n.startswith("gold facts by sources in the set: 1: 3, 2: 3, 3+: 1") for n in report.notes
    )
    assert any("resolver links different (rules): 1, right 1" in n for n in report.notes)
    saved = json.loads((out / "corroboration.json").read_text())
    assert saved["graph"]["on"]["edges"] == 1
    assert len(read_jsonl(out / "facts" / "grounded.jsonl")) == 2


def test_the_cli_fetches_and_prepares_trex(raw: Path, tmp_path: Path) -> None:
    assert "trex" in DATASETS
    args = [
        "bench",
        "prepare",
        "trex",
        str(raw),
        "--out",
        str(tmp_path / "set"),
        "--documents",
        "5",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "gold facts by sources in the set: 1: 3, 2: 3, 3+: 1" in result.output
    assert "checked 71.4%" in result.output
