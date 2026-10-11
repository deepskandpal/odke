"""The coverage report: what extraction left behind, found with no model (#130)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openodke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    Ontology,
    Pipeline,
    RouteVerdict,
    Span,
    SpanOrigin,
)
from openodke.coverage import NameMatcher, known_entities, measure, offered_by, summary
from openodke.extract import LLMExtractor
from openodke.llm import ScriptedClient
from openodke.run import execute, load_config, parse_config

ONTOLOGY = Ontology.from_dict(
    {
        "name": "people",
        "types": {"Person": {}, "Company": {}, "City": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company", "importance": 0.9},
            "born_in": {"domain": ["Person"], "range": "City", "importance": 0.8},
            "founded": {"domain": ["Company"], "range": "integer", "importance": 0.5},
            "headquarters": {"domain": ["Company"], "range": "City", "importance": 0.1},
        },
    }
)
# Three sentences. The first is cited, the second names two known entities and
# no fact covers it (the planted gap), the third names only one.
TEXT = (
    "Ada Lovelace worked for Analytical Engines Ltd. "
    "Ada Lovelace was born in London. "
    "London is large."
)
FIRST = "Ada Lovelace worked for Analytical Engines Ltd."
GAP = "Ada Lovelace was born in London."
DOC = Document(id="d1", text=TEXT)


def _entity(type_: str, label: str) -> Entity:
    return Entity(key=f"{type_}:{label.casefold()}", type=type_, label=label)


ADA = _entity("Person", "Ada Lovelace")
ENGINES = _entity("Company", "Analytical Engines")
LONDON = _entity("City", "London")


def _cited(doc: Document, quote: str, subject: Entity, predicate: str, obj: Entity) -> Fact:
    start = doc.text.index(quote)
    span = Span(doc_id=doc.id, start=start, end=start + len(quote), quote=quote)
    return Fact(
        subject=subject,
        predicate=predicate,
        object_entity=obj,
        evidence=(Evidence(doc_id=doc.id, span=span),),
    )


EMPLOYED = _cited(DOC, FIRST, ADA, "employer", ENGINES)
# London is known to the batch from a fact in another document.
ELSEWHERE = Document(id="d2", text="Analytical Engines is based in London.")
BASED = _cited(ELSEWHERE, ELSEWHERE.text, ENGINES, "headquarters", LONDON)


def test_a_planted_gap_is_found_and_a_covered_sentence_is_not() -> None:
    report = measure([DOC], [EMPLOYED, BASED], ONTOLOGY)
    (record,) = report.documents
    # Two sentences name two known entities; the third names only London.
    assert record.sentences == 2
    assert [span.quote for span in record.uncovered] == [GAP]
    gap = record.uncovered[0]
    assert DOC.text[gap.start : gap.end] == GAP
    # London is mentioned here and named by no fact of this document.
    assert [span.quote for span in record.missed] == ["London"]
    assert report.uncovered == 1 and report.missed == 1 and report.sentences == 2


def test_a_fact_on_the_gap_covers_it() -> None:
    born = _cited(DOC, GAP, ADA, "born_in", LONDON)
    (record,) = measure([DOC], [EMPLOYED, BASED, born], ONTOLOGY).documents
    assert record.sentences == 2
    assert record.uncovered == ()
    assert record.missed == ()


def test_any_overlap_covers_a_sentence() -> None:
    # A citation narrower than its sentence still covers it: a narrow citation
    # is the grounder's concern, not a gap.
    born = _cited(DOC, "London", ADA, "born_in", LONDON)
    assert born.evidence[0].span is not None and born.evidence[0].span.start > TEXT.index(GAP)
    (record,) = measure([DOC], [EMPLOYED, BASED, born], ONTOLOGY).documents
    assert record.uncovered == ()


def test_a_span_nobody_chose_covers_the_sentence_naming_both_ends_of_its_fact() -> None:
    whole = Evidence(
        doc_id="d1",
        span=Span(doc_id="d1", start=0, end=len(TEXT), quote=TEXT),
        span_origin=SpanOrigin.CONTEXT,
    )
    born = Fact(subject=ADA, predicate="born_in", object_entity=LONDON, evidence=(whole,))
    (record,) = measure([DOC], [EMPLOYED, born], ONTOLOGY).documents
    assert record.uncovered == ()
    # An uncited fact whose ends no sentence names covers nothing, though its
    # span overlaps every sentence.
    elsewhere = born.model_copy(update={"object_entity": _entity("City", "Paris")})
    (record,) = measure([DOC], [EMPLOYED, elsewhere, BASED], ONTOLOGY).documents
    assert [span.quote for span in record.uncovered] == [GAP]


def test_a_span_that_does_not_resolve_covers_nothing() -> None:
    wrong = EMPLOYED.model_copy(
        update={
            "evidence": (
                Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=5, quote="nope!")),
            )
        }
    )
    (record,) = measure([DOC], [wrong, BASED], ONTOLOGY).documents
    assert [span.quote for span in record.uncovered] == [FIRST, GAP]


def test_relations_never_offered_when_the_snippet_is_limited() -> None:
    # A one-predicate snippet per type: `born_in` and `headquarters` fall off the end.
    extractor = LLMExtractor(client=ScriptedClient([]), snippet_limit=1)
    assert offered_by(extractor, ONTOLOGY) == ["founded", "employer"]
    report = measure([DOC], [EMPLOYED, BASED], ONTOLOGY, offered=offered_by(extractor, ONTOLOGY))
    assert report.not_offered == ("born_in", "headquarters")
    # Offered, and no fact used it.
    assert report.unused == ("founded",)


def test_relations_unknown_to_have_been_withheld_are_only_counted_as_unused() -> None:
    assert offered_by(object(), ONTOLOGY) is None
    report = measure([DOC], [EMPLOYED], ONTOLOGY)
    assert report.not_offered is None
    assert report.unused == ("born_in", "founded", "headquarters")


def _named(*labels: str) -> list[Entity]:
    return [Entity(key=f"x:{label}", type="X", label=label) for label in labels]


def test_names_are_found_by_the_span_locator_s_rules() -> None:
    entities = _named("Zoë Smith", "Acme, Inc.", "Acme Cloud", "U.S.", "The Beatles", "Africa", "A")
    matcher = NameMatcher(entities)
    text = "ZOE SMITH met Acme Cloud staff and Acme in the U.S. with the Beatles in South Africa."
    found = [(sorted(keys), text[s:e]) for keys, s, e in matcher.find(text)]
    assert found == [
        (["zoe smith"], "ZOE SMITH"),
        (["acme cloud"], "Acme Cloud"),
        (["acme"], "Acme"),
        (["us"], "U.S."),
        (["beatles"], "Beatles"),
    ]
    # A capitalised name is found only capitalised, and never inside a longer one:
    # `acme cloud` is not Acme Cloud, and `Africa` is not in `South Africa`.
    assert [text[s:e] for _, s, e in matcher.find("acme cloud in South Africa")] == []
    # A one-character name would match nearly everything, so it is left out.
    assert len(matcher) == 6


def test_two_entities_sharing_a_name_are_one_mention() -> None:
    city = Entity(key="City:paris", type="City", label="Paris")
    person = Entity(key="Person:paris", type="Person", label="Paris")
    (mention,) = NameMatcher([city, person]).find("Paris spoke.")
    assert mention[0] == frozenset({"paris"})


def test_known_entities_are_every_subject_and_edge_object_once() -> None:
    assert known_entities([EMPLOYED, BASED, EMPLOYED]) == [ADA, ENGINES, LONDON]


class _Replay:
    """An extractor answering each chunk with the facts planted for its document."""

    def __init__(self, facts: dict[str, list[Fact]]) -> None:
        self.facts = facts

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        return list(self.facts.get(chunk.doc_id, []))

    def offered(self, ontology: Ontology) -> list[str]:
        return ["employer", "headquarters"]


def test_the_pipeline_reports_coverage_per_document_only_when_asked() -> None:
    replay = _Replay({"d1": [EMPLOYED], "d2": [BASED]})
    assert "coverage" not in Pipeline(ONTOLOGY, replay).run([DOC, ELSEWHERE]).stats

    stats = Pipeline(ONTOLOGY, replay, coverage=True).run([DOC, ELSEWHERE]).stats["coverage"]
    assert (stats["sentences"], stats["uncovered"], stats["missed_entities"]) == (3, 1, 1)
    assert stats["not_offered"] == ["born_in", "founded"]
    assert stats["unused"] == []
    by_doc = {d["doc_id"]: d for d in stats["documents"]}
    assert [s["quote"] for s in by_doc["d1"]["uncovered"]] == [GAP]
    assert by_doc["d2"] == {"doc_id": "d2", "sentences": 1, "uncovered": [], "missed": []}
    json.dumps(stats)  # what a manifest writes
    assert summary(stats) == (
        "1 of 3 sentences naming two known entities uncovered, 1 entities in no fact, "
        "2 relations never offered, 0 unused"
    )


class _SkipSecond:
    def route(self, chunk: Chunk) -> RouteVerdict:
        return RouteVerdict(action="skip" if "born" in chunk.text else "extract")


class _Sentences:
    def chunk(self, doc: Document) -> list[Chunk]:
        out, at = [], 0
        for index, part in enumerate(doc.text.split(". ")):
            start = doc.text.index(part, at)
            out.append(
                Chunk(doc_id=doc.id, start=start, end=start + len(part), text=part, index=index)
            )
            at = start + len(part)
        return out


def test_a_chunk_the_router_skipped_is_not_a_gap() -> None:
    replay = _Replay({"d1": [EMPLOYED], "d2": [BASED]})
    pipeline = Pipeline(ONTOLOGY, replay, chunker=_Sentences(), router=_SkipSecond(), coverage=True)
    stats = pipeline.run([DOC, ELSEWHERE]).stats["coverage"]
    assert stats["uncovered"] == 0


def _project(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    (tmp_path / "ontology.json").write_text(ONTOLOGY.model_dump_json(), encoding="utf-8")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text(TEXT, encoding="utf-8")
    return {
        "ontology": "ontology.json",
        "inputs": ["corpus"],
        "stages": {
            "extractor": "test_coverage:Planted",
            "sink": {"use": "jsonl", "directory": "out"},
        },
        **extra,
    }


class Planted:
    """`test_coverage:Planted`: the first sentence's fact, cited, for any document."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        start = chunk.text.find(FIRST)
        if start < 0:
            return []
        at = chunk.start + start
        span = Span(doc_id=chunk.doc_id, start=at, end=at + len(FIRST), quote=FIRST)
        return [
            Fact(
                subject=ADA,
                predicate="employer",
                object_entity=ENGINES,
                evidence=(Evidence(doc_id=chunk.doc_id, span=span),),
            ),
            # A fact that makes London a known name, cited to the first sentence.
            Fact(
                subject=ENGINES,
                predicate="headquarters",
                object_entity=LONDON,
                evidence=(Evidence(doc_id=chunk.doc_id, span=span),),
            ),
        ]


def test_odke_run_reports_coverage_in_its_stats_manifest_and_summary(tmp_path: Path) -> None:
    result = execute(parse_config(_project(tmp_path), base_dir=tmp_path))
    coverage = result.stats["coverage"]
    assert coverage["uncovered"] == 1
    assert [s["quote"] for s in coverage["documents"][0]["uncovered"]] == [GAP]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["coverage"] == coverage
    line = next(row for row in result.render().splitlines() if row.startswith("coverage"))
    assert "1 of 2 sentences naming two known entities uncovered" in line


def test_odke_run_can_leave_coverage_out(tmp_path: Path) -> None:
    result = execute(parse_config(_project(tmp_path, coverage=False), base_dir=tmp_path))
    assert "coverage" not in result.stats
    assert not any(row.startswith("coverage") for row in result.render().splitlines())


def test_the_e2e_example_reports_its_coverage(example: Path) -> None:
    result = execute(load_config(example / "odke.yaml"), dry_run=True)
    coverage = result.stats["coverage"]
    assert coverage["sentences"] >= coverage["uncovered"]
    assert coverage["not_offered"] == []
