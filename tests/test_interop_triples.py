"""The triples input format (#95): any extractor's output, checked by openodke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from openodke import (
    Document,
    Evidence,
    GroundingVerdict,
    Ontology,
    Pipeline,
    SourceTier,
    SpanOrigin,
)
from openodke.cli.main import app
from openodke.eval.spans import evaluate_spans, load_facts
from openodke.ground import LLMGrounder
from openodke.interop import THING, TripleRow, TriplesExtractor, read_triples, to_fact
from openodke.llm.testing import RecordedClient
from openodke.run import ConfigError, build, parse_config

runner = CliRunner()

TEXT = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. The company builds "
    "warehouse robots, and it opened a second office in Lyon in 2019."
)


def _row(**overrides: Any) -> TripleRow:
    base: dict[str, Any] = {
        "doc": "halden",
        "subject": "Halden Robotics",
        "subject_type": "Company",
        "predicate": "office_in",
        "object": "Lyon",
        "object_type": "City",
    }
    return TripleRow.model_validate({**base, **overrides})


@pytest.fixture
def companies() -> Ontology:
    return Ontology.from_json(Path(__file__).parent.parent / "examples/triples/ontology.json")


@pytest.fixture
def halden() -> Document:
    return Document(id="halden", text=TEXT)


# --------------------------------------------------------------------------- #
# The row
# --------------------------------------------------------------------------- #


def test_the_row_the_bench_already_writes_is_a_valid_row() -> None:
    row = TripleRow.model_validate(
        {
            "doc": "d1",
            "subject": "Acme",
            "subject_type": "Company",
            "predicate": "located_in",
            "object": "Berlin",
            "object_type": "City",
        }
    )
    assert (row.start, row.end, row.quote) == (None, None, None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start": 3}, "start and end come together"),
        ({"start": 9, "end": 3}, "start 9 is after end 3"),
        ({"quote": ""}, "quote is empty"),
        ({"subject": ""}, "at least 1 character"),
        ({"objet": "Lyon"}, "Extra inputs are not permitted"),
    ],
)
def test_a_malformed_row_is_refused_with_the_reason(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _row(**overrides)


def test_reading_a_file_names_the_line_a_bad_row_is_on(tmp_path: Path) -> None:
    good = _row().model_dump_json(exclude_none=True)
    path = tmp_path / "triples.jsonl"
    path.write_text(good + "\n\n" + good.replace('"object":"Lyon",', "") + "\n")
    with pytest.raises(ValueError, match=r"triples.jsonl:3: object: Field required"):
        read_triples(path)
    path.write_text(good + "\n{not json\n")
    with pytest.raises(ValueError, match=r"triples.jsonl:2: not JSON"):
        read_triples(path)
    path.write_text(good + "\n\n" + good + "\n")
    assert len(read_triples(path)) == 2


# --------------------------------------------------------------------------- #
# Evidence: offsets, a quote, or neither
# --------------------------------------------------------------------------- #


def test_offsets_are_a_citation(halden: Document) -> None:
    start = TEXT.index("Lyon")
    (evidence,) = to_fact(_row(start=start, end=start + 4), halden).evidence
    assert evidence.span is not None and evidence.span.resolve(halden) == "Lyon"
    assert evidence.span_origin is SpanOrigin.CITED


def test_a_quote_with_no_offsets_is_found_in_the_text(halden: Document) -> None:
    quote = "opened a second office in Lyon"
    (evidence,) = to_fact(_row(quote=quote), halden).evidence
    assert evidence.span is not None
    assert evidence.span.start == TEXT.index(quote)
    assert evidence.span.is_faithful(halden)
    assert evidence.span_origin is SpanOrigin.CITED


def test_a_quote_that_is_not_in_the_text_gets_no_span(halden: Document) -> None:
    (evidence,) = to_fact(_row(quote="an office in Lyon since 2012"), halden).evidence
    assert evidence.span is None


def test_a_row_with_neither_is_grounded_against_its_whole_text(halden: Document) -> None:
    (evidence,) = to_fact(_row(), halden).evidence
    assert evidence.span is not None
    assert (evidence.span.start, evidence.span.end) == (0, len(TEXT))
    # Not a quote of the whole text on every fact: offsets are enough.
    assert evidence.span.quote is None
    assert evidence.span_origin is SpanOrigin.CONTEXT


def test_evidence_carries_the_documents_provenance() -> None:
    doc = Document(text=TEXT, uri="file:///notes/halden.txt", tier=SourceTier.CURATED)
    (evidence,) = to_fact(_row(), doc).evidence
    assert (evidence.doc_id, evidence.uri, evidence.tier) == (doc.id, doc.uri, doc.tier)


def test_evidence_written_before_the_field_existed_loads_as_cited() -> None:
    evidence = Evidence.model_validate(
        {"doc_id": "d", "span": {"doc_id": "d", "start": 0, "end": 4}}
    )
    assert evidence.span_origin is SpanOrigin.CITED


# --------------------------------------------------------------------------- #
# Types: the ontology decides what the labels cannot
# --------------------------------------------------------------------------- #


def test_the_ontology_makes_an_edge_and_spells_the_names(
    companies: Ontology, halden: Document
) -> None:
    fact = to_fact(
        _row(predicate="OFFICE_IN", subject_type="company", object_type="city"), halden, companies
    )
    assert fact.predicate == "office_in"
    assert (fact.subject.type, fact.subject.key) == ("Company", "Company:halden robotics")
    assert fact.object_entity is not None
    assert (fact.object_entity.type, fact.object_entity.key) == ("City", "City:lyon")


def test_a_literal_range_makes_a_property_whatever_the_extractor_called_the_node(
    companies: Ontology, halden: Document
) -> None:
    # Graph extractors give a literal a node label of their own: "Date", "Number".
    fact = to_fact(
        _row(predicate="founded", object="2014", object_type="Number"), halden, companies
    )
    assert fact.object_entity is None and fact.object_value == "2014"


def test_a_subject_with_no_type_takes_the_predicates_domain(
    companies: Ontology, halden: Document
) -> None:
    fact = to_fact(_row(subject_type=None), halden, companies)
    assert fact.subject.type == "Company"


def test_a_predicate_the_ontology_lacks_is_kept_for_a_later_stage_to_judge(
    companies: Ontology, halden: Document
) -> None:
    fact = to_fact(_row(predicate="rival_of", object_type="Company"), halden, companies)
    assert fact.predicate == "rival_of"
    assert fact.object_entity is not None and fact.object_entity.type == "Company"


@pytest.mark.parametrize(
    ("obj", "object_type", "entity_type", "value"),
    [
        ("Lyon", None, THING, None),
        ("Lyon", "City", "City", None),
        ("2019", "date", None, "2019"),
        (2019, None, None, 2019),
        (True, None, None, True),
    ],
)
def test_without_an_ontology_a_string_is_an_entity_unless_typed_as_a_literal(
    halden: Document, obj: Any, object_type: str | None, entity_type: str | None, value: Any
) -> None:
    fact = to_fact(_row(object=obj, object_type=object_type, subject_type=None), halden)
    assert fact.subject.type == THING
    assert (fact.object_entity.type if fact.object_entity else None) == entity_type
    assert fact.object_value == value


def test_a_rows_own_id_confidence_and_extractor_win(halden: Document) -> None:
    fact = to_fact(
        _row(id="f-1", confidence=0.9, extractor="langchain"),
        halden,
        extractor="triples",
        confidence=0.5,
    )
    assert (fact.id, fact.confidence, fact.extractor) == ("f-1", 0.9, "langchain")
    assert to_fact(_row(), halden).extractor == "triples"


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def test_rows_find_their_text_by_id_or_by_file_name(companies: Ontology) -> None:
    # What a loader makes of corpus/halden.txt: a generated id, and the file's URI.
    by_name = Document(text=TEXT, uri="file:///corpus/halden.txt")
    by_id = Document(id="other", text=TEXT)
    stage = TriplesExtractor(
        [_row(), _row(doc="other", object="Leeds"), _row(doc="missing")],
        documents=[by_name, by_id],
    )
    pipeline = Pipeline(companies, stage)
    kg = pipeline.run([by_name, by_id])
    assert sorted(f.object_entity.label for f in kg.facts if f.object_entity) == ["Leeds", "Lyon"]
    assert stage.stats["unmatched_rows"] == 1
    assert stage.stats["unmatched_docs"] == ["missing"]


def test_triples_arrive_once_with_the_first_chunk(companies: Ontology, halden: Document) -> None:
    from openodke import SentenceChunker

    stage = TriplesExtractor([_row(), _row(object="Leeds")])
    kg = Pipeline(companies, stage, chunker=SentenceChunker(max_words=12)).run([halden])
    assert kg.stats["chunks"] > 1
    assert len(kg.facts) == 2


def test_another_extractors_triples_ground_end_to_end(
    companies: Ontology, halden: Document
) -> None:
    rows = [
        _row(quote="opened a second office in Lyon"),  # cited by quote
        _row(object="Berlin"),  # no citation: the whole text
        _row(object="Leeds", quote="an office in Leeds since 2012"),  # a quote not in the text
    ]
    client = RecordedClient(
        [
            {"match": "— Lyon (City)", "response": {"verdict": "supported"}},
            {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
        ]
    )
    kg = Pipeline(
        companies, TriplesExtractor(rows, documents=[halden]), grounder=LLMGrounder(client=client)
    ).run([halden])
    verdicts = {f.object_entity.label: f.verdict for f in kg.facts if f.object_entity}
    assert verdicts == {
        "Lyon": GroundingVerdict.SUPPORTED,
        "Berlin": GroundingVerdict.NOT_FOUND,
        "Leeds": GroundingVerdict.NOT_FOUND,
    }
    # The invented quote was refused for free: two calls, not three.
    assert len(client.calls) == 2
    # The uncited fact was asked about against the whole text.
    (berlin,) = [messages for messages, _, _ in client.calls if "Berlin" in str(messages)]
    assert TEXT in berlin[-1].content

    report = evaluate_spans(kg)
    # One width: the quote. The whole-text stand-in is not a citation.
    assert (report.metrics["with_span"], report.metrics["no_span"]) == (1, 2)


# --------------------------------------------------------------------------- #
# odke run, and the example
# --------------------------------------------------------------------------- #


def test_the_example_grounds_five_hand_written_triples(example: Path) -> None:
    triples = example.parent / "triples"
    result = runner.invoke(app, ["run", str(triples / "odke.yaml")])
    assert result.exit_code == 0, result.output
    stats = json.loads((triples / "out" / "manifest.json").read_text())["stats"]
    extractor = stats["stages"]["extractor"]
    assert {k: extractor[k] for k in ("rows", "cited", "quoted", "quote_not_found", "context")} == {
        "rows": 5,
        "cited": 1,
        "quoted": 1,
        "quote_not_found": 1,
        "context": 2,
    }
    assert extractor["unmatched_rows"] == 0
    grounder = stats["stages"]["grounder"]
    assert (grounder["calls"], grounder["supported"], grounder["not_found"]) == (4, 3, 1)

    facts = load_facts(triples / "out")
    assert {f.extractor for f in facts} == {"hand-written"}
    not_found = {
        (f.object_entity.label if f.object_entity else f.object_value)
        for f in facts
        if f.verdict is GroundingVerdict.NOT_FOUND
    }
    assert not_found == {"Berlin", 2012}

    report = runner.invoke(app, ["eval", "spans", "--facts", str(triples / "out")])
    assert report.exit_code == 0, report.output
    assert "3 of 5 fact(s) cite no span of their own" in report.output


def _example_config(**extractor: Any) -> dict[str, Any]:
    path = Path(__file__).parent.parent / "examples/triples/odke.yaml"
    config = yaml.safe_load(path.read_text())
    config["stages"]["extractor"] = {"use": "triples", **extractor}
    return config


def test_odke_run_needs_the_triples_path_and_names_the_mistake(tmp_path: Path) -> None:
    base = Path(__file__).parent.parent / "examples/triples"
    with pytest.raises(ConfigError, match=r"extractor.path: the JSON Lines file"):
        build(parse_config(_example_config(), base_dir=base))
    with pytest.raises(ConfigError, match=r"extractor.triples: name the file with path"):
        build(parse_config(_example_config(path="triples.jsonl", triples="x"), base_dir=base))
    with pytest.raises(ConfigError, match=r"extractor.documents: set by odke run"):
        build(parse_config(_example_config(path="triples.jsonl", documents=[]), base_dir=base))
    built = build(
        parse_config(_example_config(path="triples.jsonl", confidence=0.7), base_dir=base)
    )
    assert isinstance(built.stages["extractor"], TriplesExtractor)
    assert built.stages["extractor"].confidence == 0.7
