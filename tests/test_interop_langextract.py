"""LangExtract output (#96): its offsets are citations, its attributes are the triples.

The fixture is a file LangExtract 1.7.1 wrote itself, with no model: five
extractions aligned to the text by its own `Resolver` (two exact, one exact
with no attributes, one fuzzy, one it could not place) and saved by
`lx.io.save_annotated_documents`. The stand-ins below carry exactly the
attribute names of `langextract.core.data` (`CharInterval`, `AlignmentStatus`,
`Extraction`, `AnnotatedDocument`).
"""

from __future__ import annotations

import enum
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from openodke import GroundingVerdict, Ontology, Pipeline, SpanOrigin
from openodke.ground import LLMGrounder
from openodke.interop import TriplesExtractor, attribute_triples, from_langextract
from openodke.llm.testing import RecordedClient

FIXTURE = Path(__file__).parent / "fixtures" / "interop" / "langextract.annotated.jsonl"
TEXT = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. The company builds "
    "warehouse robots, and it opened a second office in Lyon in 2019."
)


class AlignmentStatus(enum.Enum):
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


@dataclass
class CharInterval:
    start_pos: int | None = None
    end_pos: int | None = None


@dataclass
class Extraction:
    extraction_class: str
    extraction_text: str
    char_interval: CharInterval | None = None
    alignment_status: AlignmentStatus | None = None
    extraction_index: int | None = None
    group_index: int | None = None
    description: str | None = None
    attributes: dict[str, str | list[str]] | None = None


@dataclass
class AnnotatedDocument:
    document_id: str
    extractions: list[Extraction] | None = None
    text: str | None = None


def _objects() -> AnnotatedDocument:
    """The fixture, as the objects `lx.extract` returns."""
    exact = AlignmentStatus.MATCH_EXACT
    return AnnotatedDocument(
        document_id="halden",
        text=TEXT,
        extractions=[
            Extraction(
                "company",
                "Halden Robotics",
                CharInterval(0, 15),
                exact,
                attributes={"founded_in": "Leeds", "founded": "2014"},
            ),
            Extraction("person", "Mara Quist", CharInterval(48, 58), exact),
            Extraction(
                "office",
                "a second office in Lyon",
                CharInterval(111, 134),
                exact,
                attributes={"company": "Halden Robotics", "city": "Lyon", "opened": "2019"},
            ),
            Extraction(
                "product",
                "warehouse robot",
                CharInterval(79, 95),
                AlignmentStatus.MATCH_FUZZY,
                attributes={"made_by": "Halden Robotics"},
            ),
            Extraction(
                "office",
                "Halden's Berlin branch",
                attributes={"company": "Halden Robotics", "city": "Berlin"},
            ),
        ],
    )


def offices(extraction: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """A mapping of the caller's: an office names the company it belongs to and its city."""
    attributes = extraction["attributes"] or {}
    if extraction["extraction_class"] == "office":
        yield {
            "subject": attributes["company"],
            "subject_type": "Company",
            "predicate": "office_in",
            "object": attributes["city"],
            "object_type": "City",
        }


@pytest.fixture
def companies() -> Ontology:
    return Ontology.from_dict(
        {
            "name": "companies",
            "types": {"Company": {}, "City": {}},
            "predicates": {
                "office_in": {"domain": ["Company"], "range": "City"},
                "founded_in": {"domain": ["Company"], "range": "City"},
            },
        }
    )


def _read(source: Any, **kw: Any) -> list[dict[str, Any]]:
    with pytest.warns(UserWarning):
        rows, _ = from_langextract(source, **kw)
    return [r.model_dump(exclude_defaults=True) for r in rows]


def test_each_attribute_is_a_triple_about_the_extraction() -> None:
    rows = _read(FIXTURE)
    company = [r for r in rows if r["subject"] == "Halden Robotics"]
    assert company == [
        {
            "doc": "halden",
            "subject": "Halden Robotics",
            "subject_type": "company",
            "predicate": predicate,
            "object": value,
            "object_type": "string",
            "start": 0,
            "end": 15,
            "quote": "Halden Robotics",
        }
        for predicate, value in (("founded_in", "Leeds"), ("founded", "2014"))
    ]
    assert len(rows) == 2 + 3 + 1 + 2


def test_an_extraction_with_no_attributes_is_an_entity_and_is_said_to_be_left_out() -> None:
    with pytest.warns(UserWarning, match=r"1 of 5 LangExtract extractions have no attributes"):
        rows, _ = from_langextract(FIXTURE)
    assert "Mara Quist" not in {r.subject for r in rows}


def test_a_fuzzy_alignment_keeps_its_offsets_and_drops_the_quote_they_do_not_hold() -> None:
    (product,) = [r for r in _read(FIXTURE) if r["predicate"] == "made_by"]
    assert (product["start"], product["end"]) == (79, 95)
    assert TEXT[79:95] == "warehouse robots"
    assert "quote" not in product


def test_an_extraction_langextract_could_not_place_has_no_offsets_and_no_quote() -> None:
    berlin = [r for r in _read(FIXTURE) if r["subject"] == "Halden's Berlin branch"]
    assert len(berlin) == 2
    assert all(not {"start", "end", "quote"} & set(r) for r in berlin)


def test_the_objects_read_as_the_file_does(tmp_path: Path) -> None:
    line = json.loads(FIXTURE.read_text())
    assert _read(_objects()) == _read([_objects()]) == _read(FIXTURE) == _read(line)
    _, (doc,) = from_langextract(FIXTURE, triples=offices)
    assert (doc.id, doc.text) == ("halden", TEXT)


def test_a_mapping_of_the_callers_gets_the_extractions_evidence() -> None:
    rows, _ = from_langextract(FIXTURE, triples=offices)
    lyon, berlin = (r.model_dump(exclude_defaults=True) for r in rows)
    assert lyon == {
        "doc": "halden",
        "subject": "Halden Robotics",
        "subject_type": "Company",
        "predicate": "office_in",
        "object": "Lyon",
        "object_type": "City",
        "start": 111,
        "end": 134,
        "quote": "a second office in Lyon",
    }
    assert "start" not in berlin and berlin["object"] == "Berlin"


def test_the_mapping_is_handed_langextracts_json_form_whatever_was_read() -> None:
    seen: list[Mapping[str, Any]] = []

    def record(extraction: Mapping[str, Any]) -> list[dict[str, Any]]:
        seen.append(extraction)
        return []

    from_langextract(_objects(), triples=record)
    from_file = json.loads(FIXTURE.read_text())["extractions"]
    assert seen == from_file


def test_a_mapping_may_cite_otherwise_and_a_bad_row_names_its_extraction() -> None:
    def sentence(extraction: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [{"subject": "Halden Robotics", "predicate": "is", "object": "a company", "end": 59}]

    # The mapping's `end` meets the extraction's own `start`: 111 is after 59.
    with pytest.raises(ValueError, match=r"document 'halden', extraction 2: .*111 is after end 59"):
        from_langextract(FIXTURE, triples=sentence)

    def whole(extraction: Mapping[str, Any]) -> list[dict[str, Any]]:
        row = {"subject": "Halden Robotics", "predicate": "is", "object": "a company"}
        return [{**row, "start": 0, "end": 59, "quote": None}]

    rows, _ = from_langextract(FIXTURE, triples=whole)
    assert {(r.start, r.end, r.quote) for r in rows} == {(0, 59, None)}


def test_a_document_must_have_its_id_and_text() -> None:
    with pytest.raises(ValueError, match="no text"):
        from_langextract([{"document_id": "d", "extractions": []}])
    with pytest.raises(ValueError, match="no document_id"):
        from_langextract([{"text": TEXT, "extractions": []}])
    other = {"document_id": "halden", "text": "Another text.", "extractions": []}
    with pytest.raises(ValueError, match="'halden'"):
        from_langextract([json.loads(FIXTURE.read_text()), other], triples=offices)


def test_reading_needs_no_langextract() -> None:
    from_langextract(FIXTURE, triples=offices)
    assert "langextract" not in sys.modules


def test_the_default_mapping_is_public_and_reads_lists() -> None:
    rows = attribute_triples(
        {
            "extraction_class": "office",
            "extraction_text": "HQ",
            "attributes": {"city": ["Leeds", ""]},
        }
    )
    assert [(r["subject"], r["predicate"], r["object"]) for r in rows] == [("HQ", "city", "Leeds")]


def test_langextracts_output_grounds_end_to_end(companies: Ontology) -> None:
    rows, texts = from_langextract(FIXTURE, triples=offices)
    client = RecordedClient(
        [
            {"match": "— Lyon (City)", "response": {"verdict": "supported"}},
            {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
        ]
    )
    stage = TriplesExtractor(rows, extractor="langextract", documents=texts)
    # The cited span is LangExtract's own, often just a name: the free check
    # still runs on it, and the model is asked against the whole text.
    grounder = LLMGrounder(client=client, context="document")
    kg = Pipeline(companies, stage, grounder=grounder).run(texts)

    assert (stage.stats["cited"], stage.stats["context"]) == (1, 1)
    by_city = {f.object_entity.label: f for f in kg.facts if f.object_entity}
    lyon, berlin = by_city["Lyon"], by_city["Berlin"]
    assert lyon.verdict is GroundingVerdict.SUPPORTED
    assert berlin.verdict is GroundingVerdict.NOT_FOUND
    assert lyon.evidence[0].span_origin is SpanOrigin.CITED
    assert (
        lyon.evidence[0].span is not None
        and lyon.evidence[0].span.quote == "a second office in Lyon"
    )
    assert berlin.evidence[0].span_origin is SpanOrigin.CONTEXT


def test_an_offset_that_does_not_hold_its_quote_is_refused_for_free(companies: Ontology) -> None:
    line = json.loads(FIXTURE.read_text())
    # Offsets that point one character late: the quote no longer sits there.
    line["extractions"][2]["char_interval"] = {"start_pos": 112, "end_pos": 135}
    line["text"] = TEXT
    rows, texts = from_langextract([line], triples=offices)
    client = RecordedClient([{"match": "— Berlin (City)", "response": {"verdict": "not_found"}}])
    kg = Pipeline(
        companies,
        TriplesExtractor(rows, documents=texts),
        grounder=LLMGrounder(client=client),
    ).run(texts)
    assert {f.verdict for f in kg.facts} == {GroundingVerdict.NOT_FOUND}
    # One call, for Berlin: the shifted span was refused without asking.
    assert len(client.calls) == 1
