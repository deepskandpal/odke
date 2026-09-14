"""Importance, the cap, freeze, and the whole bootstrap composed (#34)."""

from __future__ import annotations

import inspect
import warnings
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from odke import Document, Inferrer, Ontology, PatternExtractor, Pipeline
from odke.infer import DEFAULT_SAMPLE_WORDS
from odke.infer.build import OntologyInferrer, infer_ontology
from odke.llm import ModelSpec, ReplayClient
from odke.loaders import DirectoryLoader
from odke.ontology import OntologyFreezeError, UnreviewedOntologyWarning

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
SONNET = ModelSpec(model="anthropic/claude-sonnet-5")
PEOPLE = [
    "Ada Lovelace",
    "Grace Hopper",
    "Alan Turing",
    "Linus Torvalds",
    "Barbara Liskov",
    "Edsger Dijkstra",
    "Donald Knuth",
    "Margaret Hamilton",
]
COMPANIES = ["Acme Corp", "Globex", "Initech"]
NOTES = (
    "# Notes\n\n"
    "Ada Lovelace works at Acme Corp. Grace Hopper works at Globex. "
    "Alan Turing works at Initech.\n\n"
    "A mathematician is a scientist. Ada Lovelace was an English mathematician.\n\n"
    "They used languages such as Python, Rust and Go.\n"
)


def _corpus(tmp_path: Path, extra_rows: int = 0) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    rows = [
        f"{n},19{10 + i}-01-0{1 + i},{COMPANIES[i % 3]},{['London', 'Paris'][i % 2]}"
        for i, n in enumerate(PEOPLE)
    ]
    rows += [
        f"Person {i},1950-01-01,{COMPANIES[i % 3]},{['London', 'Paris'][i % 2]}"
        for i in range(extra_rows)
    ]
    (root / "people.csv").write_text(
        "name,born,employer,city\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    (root / "companies.csv").write_text("name\n" + "\n".join(COMPANIES) + "\n", encoding="utf-8")
    (root / "notes.md").write_text(NOTES, encoding="utf-8")
    return root


def _load(root: Path) -> list[Document]:
    return list(DirectoryLoader().load(root))


def test_no_llm_infers_a_valid_ontology_from_a_mixed_corpus(tmp_path: Path) -> None:
    inference = infer_ontology(_load(_corpus(tmp_path)), llm=False)
    ontology = inference.ontology
    assert ontology.inferred and ontology.name == "inferred"
    assert [d.code for d in ontology.validate()] == ["unreviewed"]
    assert {"Person", "Company", "City", "Mathematician", "Scientist", "Language"} <= set(
        ontology.types
    )
    assert ontology.types["Mathematician"].parents == ("Scientist",)
    assert ontology.types["Person"].keys == ("name",)
    assert ontology.types["Company"].keys == ("name",)
    employer = ontology.predicates["employer"]
    assert (employer.domain, employer.range) == (("Person",), "Company")
    assert "works_at" in employer.aliases
    assert ontology.predicates["born"].range == "date"
    assert ontology.predicates["city"].range == "City"
    assert inference.calls == () and inference.settings["model"] is None


def test_importance_comes_from_corpus_support(tmp_path: Path) -> None:
    ontology = infer_ontology(_load(_corpus(tmp_path)), llm=False).ontology
    importance = {name: p.importance for name, p in ontology.predicates.items()}
    # `name` is backed by every person and every company: 11 documents.
    assert importance["name"] == 1.0
    assert 0 < importance["born"] < importance["employer"] < 1.0
    assert ontology.snippet("Person").predicates[0].name == "name"


def test_the_cap_keeps_the_best_supported_and_says_what_went(tmp_path: Path) -> None:
    inference = infer_ontology(_load(_corpus(tmp_path)), llm=False, max_types=2, max_predicates=2)
    ontology = inference.ontology
    assert set(ontology.types) == {"City", "Person"}
    # born and city tie on support (8 rows each); the name breaks the tie.
    assert set(ontology.predicates) == {"name", "born"}
    assert not [d for d in ontology.validate() if d.severity == "error"]
    dropped = {(d.kind, d.name): d.reason for d in inference.dropped}
    assert dropped[("type", "Company")].startswith("over the cap of 2 types")
    assert dropped[("predicate", "employer")] == "its range 'Company' was not kept"
    assert dropped[("predicate", "city")].startswith("over the cap of 2 predicates")


def test_every_entry_is_traceable_to_its_evidence(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    inference = infer_ontology(_load(root), llm=False)
    assert set(inference.evidence) == {
        *(f"types.{t}" for t in inference.ontology.types),
        *(f"predicates.{p}" for p in inference.ontology.predicates),
    }
    employer = inference.evidence["predicates.employer"]
    assert employer.support == 9 and employer.proposers == ("cooccurrence", "records")
    assert employer.aliases == inference.ontology.predicates["employer"].aliases
    rows = [s for s in employer.spans if s.source and s.source.endswith("people.csv")]
    assert rows and all(s.line is not None and s.quote in COMPANIES for s in rows)
    prose = [s for s in employer.spans if s.source and s.source.endswith("notes.md")]
    text = Path(unquote(urlparse(prose[0].source or "").path)).read_text(encoding="utf-8")
    assert text[prose[0].start : prose[0].end] == prose[0].quote
    assert inference.sample.budget_words == DEFAULT_SAMPLE_WORDS
    assert any(d.action == "merged" and d.into == "employer" for d in inference.decisions)


def test_the_same_corpus_infers_the_same_ontology_and_evidence(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    first = infer_ontology(_load(root), llm=False)
    second = infer_ontology(_load(root), llm=False)
    assert first.ontology == second.ontology
    assert first.evidence == second.evidence
    assert first.decisions == second.decisions


def test_doubling_the_sample_does_not_move_the_schema(tmp_path: Path) -> None:
    """#31's done-when, asserted on the thing it is about: the inferred schema."""
    root = _corpus(tmp_path, extra_rows=300)

    def shape(words: int) -> tuple[set[tuple[str, ...]], set[tuple[str, str]]]:
        ontology = infer_ontology(_load(root), llm=False, sample_words=words).ontology
        types = {(name, *t.parents) for name, t in ontology.types.items()}
        return types, {(name, p.range) for name, p in ontology.predicates.items()}

    assert shape(600) == shape(1200)


def test_ontology_infer_and_the_inferrer_stage_compose_the_same_path(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    expected = infer_ontology(_load(root), llm=False).ontology
    assert Ontology.infer(_load(root), llm=False) == expected
    stage = OntologyInferrer(llm=False)
    assert isinstance(stage, Inferrer)
    assert stage.infer(_load(root)) == expected
    assert stage.last is not None and stage.last.ontology == expected


def test_the_model_path_runs_on_a_recorded_response(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "people.csv").write_text(
        "name,employer\nAda Lovelace,Acme Corp\nGrace Hopper,Globex\n", encoding="utf-8"
    )
    (root / "companies.csv").write_text("name\nAcme Corp\nGlobex\n", encoding="utf-8")
    (root / "notes.md").write_text(
        "Ada Lovelace works at Acme Corp. Grace Hopper works at Globex.\n\n"
        "Acme Corp is headquartered in London.\n",
        encoding="utf-8",
    )
    client = ReplayClient(FIXTURES / "infer_people.json")
    inference = infer_ontology(_load(root), client=client, spec=SONNET)
    ontology = inference.ontology
    assert client.exhausted and len(inference.calls) == 1
    assert inference.settings["model"] == "anthropic/claude-sonnet-5"
    assert set(ontology.types) == {"Person", "Company", "Place"}
    assert ontology.types["Person"].description == "A person named in the staff records."
    assert ontology.types["Person"].keys == ("full_name",)
    assert ontology.predicates["employer"].aliases == ("works_at",)
    assert ontology.predicates["full_name"].aliases == ("name",)
    assert not [d for d in ontology.validate() if d.severity == "error"]
    assert {r.name for r in inference.rejections} == {"City", "full_name", "headquarters"}
    assert inference.evidence["types.Person"].proposers == ("llm", "records")


# --------------------------------------------------------------------------- #
# Freeze
# --------------------------------------------------------------------------- #

WHEN = datetime(2026, 9, 14, 9, 30, tzinfo=UTC)


def test_freeze_clears_inferred_and_records_when_and_by_whom(tmp_path: Path) -> None:
    inferred = infer_ontology(_load(_corpus(tmp_path)), llm=False).ontology
    frozen = inferred.freeze(by="Deepak Kandpal", at=WHEN)
    assert (frozen.inferred, frozen.frozen_by, frozen.frozen_at) == (False, "Deepak Kandpal", WHEN)
    assert inferred.inferred and inferred.frozen_at is None
    assert frozen.validate() == []
    assert Ontology.from_json(frozen.model_dump_json()) == frozen
    changes = inferred.diff(frozen)
    assert [c.path for c in changes] == ["inferred", "frozen_at", "frozen_by"]
    assert not any(c.breaking for c in changes)
    stamped = inferred.freeze(by="someone")
    assert stamped.frozen_at is not None and stamped.frozen_at.tzinfo is not None


def test_freeze_refuses_an_ontology_with_validation_errors(tmp_path: Path) -> None:
    inferred = infer_ontology(_load(_corpus(tmp_path)), llm=False).ontology
    broken = inferred.model_copy(deep=True)
    broken.predicates["employer"] = broken.predicates["employer"].model_copy(
        update={"range": "Compnay"}
    )
    with pytest.raises(OntologyFreezeError, match="1 validation error") as info:
        broken.freeze(by="Deepak Kandpal")
    (diagnostic,) = info.value.diagnostics
    assert (diagnostic.code, diagnostic.path) == ("unknown-range", "predicates.employer.range")
    assert broken.inferred


def test_freeze_needs_a_reviewer() -> None:
    with pytest.raises(ValueError, match="who reviewed"):
        Ontology(inferred=True).freeze(by="  ")


def test_an_old_ontology_still_loads_without_the_freeze_fields() -> None:
    loaded = Ontology.from_dict({"name": "old", "types": {"Person": {}}})
    assert (loaded.inferred, loaded.frozen_at, loaded.frozen_by) == (False, None, None)
    assert loaded.validate() == []


# --------------------------------------------------------------------------- #
# The loop, and the things that must not happen
# --------------------------------------------------------------------------- #


def test_infer_review_freeze_then_run_guided(tmp_path: Path) -> None:
    """#34's done-when: the inferred → review → freeze → guided-run loop, end to end."""
    docs = _load(_corpus(tmp_path))
    inferred = Ontology.infer(docs, llm=False)
    reviewed = inferred.model_copy(deep=True)
    reviewed.types["Person"] = reviewed.types["Person"].model_copy(
        update={"description": "Someone on the staff list."}
    )
    frozen = reviewed.freeze(by="Deepak Kandpal", at=WHEN)
    kg = Pipeline(frozen, PatternExtractor(documents=docs)).run(docs)
    edges = {(f.subject.key, f.predicate, f.object_entity.key) for f in kg.edges if f.object_entity}
    assert ("Person:ada lovelace", "employer", "Company:acme corp") in edges
    assert ("Person:grace hopper", "city", "City:paris") in edges


def test_inference_never_runs_from_pipeline_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import odke.infer.build as build
    import odke.infer.sample as sample

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("inference ran")

    monkeypatch.setattr(build, "infer_ontology", refuse)
    monkeypatch.setattr(sample, "sample_corpus", refuse)
    assert not {"inferrer", "infer"} & set(inspect.signature(Pipeline).parameters)

    docs = _load(_corpus(tmp_path))
    empty = Ontology()
    kg = Pipeline(empty, PatternExtractor(documents=docs)).run(docs)
    assert len(kg) == 0 and empty == Ontology()

    inferred = Ontology(name="draft", inferred=True, types={"Person": {"name": "Person"}})
    pipeline = Pipeline(inferred, PatternExtractor(documents=docs))
    pipeline.run(docs)
    assert pipeline.ontology is inferred and inferred.inferred


def test_a_sink_warns_before_shaping_a_store_with_an_unreviewed_schema(tmp_path: Path) -> None:
    from odke.sinks.neo4j import Neo4jSink

    inferred = infer_ontology(_load(_corpus(tmp_path)), llm=False).ontology
    with pytest.warns(UnreviewedOntologyWarning, match="inferred and has not been reviewed"):
        Neo4jSink(driver=object(), ontology=inferred)
    sink = Neo4jSink(driver=object())
    with pytest.warns(UnreviewedOntologyWarning, match="bootstrap"):
        sink.bootstrap(inferred, dry_run=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frozen = inferred.freeze(by="Deepak Kandpal", at=WHEN)
        Neo4jSink(driver=object(), ontology=frozen).bootstrap(frozen, dry_run=True)
