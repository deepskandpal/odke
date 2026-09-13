"""The composition, and the ways it is allowed to degrade."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal

from odke import (
    Chunk,
    Document,
    Entity,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    Ontology,
    Pipeline,
    RouteVerdict,
    ValidationVerdict,
)
from odke.sinks import JsonlSink


class _StubExtractor:
    """Emits one property fact and one edge fact per chunk."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        person = Entity(key=f"p:{chunk.doc_id}", type="Person", label="Ada")
        return [
            Fact(subject=person, predicate="name", object_value="Ada"),
            Fact(
                subject=person,
                predicate="employer",
                object_entity=Entity(key="c:1", type="Company"),
            ),
        ]


class _StampingGrounder:
    """Supports names, contradicts employers. Stamps; never drops."""

    def ground(self, fact: Fact, doc: Document) -> Fact:
        verdict = (
            GroundingVerdict.SUPPORTED
            if fact.predicate == "name"
            else GroundingVerdict.CONTRADICTED
        )
        return fact.model_copy(update={"verdict": verdict})


class _RefusingValidator:
    """The gate: what the grounder contradicted does not reach the sink."""

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        if fact.verdict is GroundingVerdict.CONTRADICTED:
            return ValidationVerdict(action="refuse", reason="contradicted by its own span")
        return ValidationVerdict(action="accept")


class _LineChunker:
    """One chunk per line, offsets intact."""

    def chunk(self, doc: Document) -> Iterable[Chunk]:
        start = 0
        for index, line in enumerate(doc.text.splitlines(keepends=True)):
            end = start + len(line)
            yield Chunk(doc_id=doc.id, start=start, end=end, text=line, index=index)
            start = end


class _AdRouter:
    """Skips any chunk that reads like an advert; at document scope if asked."""

    def __init__(self, scope: Literal["chunk", "document"] = "chunk") -> None:
        self.scope = scope

    def route(self, chunk: Chunk) -> RouteVerdict:
        if chunk.text.startswith("AD:"):
            return RouteVerdict(action="skip", label="marketing", scope=self.scope)
        return RouteVerdict(action="extract")


def test_pipeline_runs_with_only_an_extractor() -> None:
    """No grounder and no corroborator must degrade to candidates, not to an error."""
    kg = Pipeline(Ontology(name="demo"), _StubExtractor()).run([Document(id="d1", text="Ada.")])
    assert len(kg) == 2
    assert kg.ontology_name == "demo"
    assert all(f.verdict is GroundingVerdict.UNCHECKED for f in kg.facts)
    assert kg.links == ()
    assert kg.stats["chunks"] == 1


def test_entities_are_collected_from_subjects_and_edge_objects() -> None:
    kg = Pipeline(Ontology(), _StubExtractor()).run([Document(id="d1", text="Ada.")])
    assert {e.key for e in kg.entities} == {"p:d1", "c:1"}


def test_the_grounder_stamps_and_the_validator_gates() -> None:
    """A grounder sets the verdict rather than dropping, so the ablation can count what went."""
    stamped = Pipeline(Ontology(), _StubExtractor(), grounder=_StampingGrounder()).run(
        [Document(id="d1", text="Ada.")]
    )
    assert {f.verdict for f in stamped.facts} == {
        GroundingVerdict.SUPPORTED,
        GroundingVerdict.CONTRADICTED,
    }
    gated = Pipeline(
        Ontology(),
        _StubExtractor(),
        grounder=_StampingGrounder(),
        validator=_RefusingValidator(),
    ).run([Document(id="d1", text="Ada.")])
    assert [f.predicate for f in gated.facts] == ["name"]
    assert gated.facts[0].verdict is GroundingVerdict.SUPPORTED
    assert gated.stats["refused"] == 1
    # The refused fact's object entity is gone with it.
    assert {e.key for e in gated.entities} == {"p:d1"}


def test_a_chunk_scoped_skip_drops_only_that_chunk() -> None:
    doc = Document(id="d1", text="Ada.\nAD: buy now\nLovelace.\n")
    kg = Pipeline(Ontology(), _StubExtractor(), chunker=_LineChunker(), router=_AdRouter()).run(
        [doc]
    )
    assert kg.stats["chunks"] == 3
    assert kg.stats["skipped"] == 1
    assert len(kg) == 4


def test_a_document_scoped_skip_stops_the_document() -> None:
    """A router that recognises a marketing page from its first chunk skips the rest."""
    doc = Document(id="d1", text="AD: buy now\nAda.\nLovelace.\n")
    kg = Pipeline(
        Ontology(), _StubExtractor(), chunker=_LineChunker(), router=_AdRouter("document")
    ).run([doc])
    assert kg.stats["chunks"] == 1
    assert kg.stats["skipped"] == 1
    assert len(kg) == 0


def test_explicit_defaults_and_none_are_the_same_pipeline() -> None:
    """`None` means the pass-through, so naming every default changes nothing."""
    from odke import stages

    docs = [Document(id="d1", text="Ada.")]
    implicit = Pipeline(Ontology(name="demo"), _StubExtractor()).run(docs)
    explicit = Pipeline(
        Ontology(name="demo"),
        _StubExtractor(),
        chunker=stages.PassThroughChunker(),
        router=stages.PassThroughRouter(),
        grounder=stages.PassThroughGrounder(),
        normalizer=stages.PassThroughNormalizer(),
        resolver=stages.PassThroughResolver(),
        corroborator=stages.PassThroughCorroborator(),
        scorer=stages.PassThroughScorer(),
        validator=stages.PassThroughValidator(),
        constrainer=stages.PassThroughConstrainer(),
    ).run(docs)
    assert [f.signature for f in implicit.facts] == [f.signature for f in explicit.facts]
    assert implicit.stats == explicit.stats
    assert Pipeline(Ontology(), _StubExtractor()).constraints() == ()


def test_jsonl_sink_round_trips_a_graph(tmp_path) -> None:
    out = tmp_path / "kg"
    Pipeline(Ontology(name="demo"), _StubExtractor(), sinks=[JsonlSink(out)]).run(
        [Document(id="d1", text="Ada.")]
    )
    facts = [json.loads(line) for line in (out / "facts.jsonl").read_text().splitlines()]
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(facts) == 2
    assert manifest["edges"] == 1
    assert manifest["properties"] == 1
    assert manifest["links"] == 0
    assert (out / "links.jsonl").exists()
    assert manifest["ontology"] == "demo"


def test_the_default_router_passes_everything() -> None:
    """Routing is opt-in: a caller who never asked for it sees no chunk skipped."""
    from odke.pipeline import PassThroughRouter, Router

    router = PassThroughRouter()
    assert isinstance(router, Router)
    verdict = router.route(Chunk(doc_id="d1", start=0, end=4, text="Ada.", index=0))
    assert verdict.action == "extract"
    assert verdict.scope == "chunk"
    assert verdict.label is None


def test_stub_stages_satisfy_the_declared_protocols() -> None:
    """If a stub stops matching, callers' own implementations would break too."""
    from odke.pipeline import Chunker, Extractor, Grounder, Router, Sink, Validator

    assert isinstance(_StubExtractor(), Extractor)
    assert isinstance(_StampingGrounder(), Grounder)
    assert isinstance(_RefusingValidator(), Validator)
    assert isinstance(_LineChunker(), Chunker)
    assert isinstance(_AdRouter(), Router)
    assert isinstance(JsonlSink("/tmp/unused"), Sink)


def test_knowledge_graph_is_serialisable() -> None:
    """Anything that crosses a process boundary has to survive JSON."""
    kg = Pipeline(Ontology(), _StubExtractor()).run([Document(id="d1", text="Ada.")])
    assert KnowledgeGraph.model_validate_json(kg.model_dump_json()).facts[0].predicate == "name"
