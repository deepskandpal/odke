"""The composition, and the ways it is allowed to degrade."""

from __future__ import annotations

import json
from collections.abc import Sequence

from odke import Document, Entity, Fact, GroundingVerdict, KnowledgeGraph, Ontology, Pipeline
from odke.sinks import JsonlSink


class _StubExtractor:
    """Emits one property fact and one edge fact per document."""

    def extract(self, docs: Sequence[Document], ontology: Ontology) -> Sequence[Fact]:
        out: list[Fact] = []
        for doc in docs:
            person = Entity(key=f"p:{doc.id}", type="Person", label="Ada")
            out.append(Fact(subject=person, predicate="name", object_value="Ada"))
            out.append(
                Fact(
                    subject=person,
                    predicate="employer",
                    object_entity=Entity(key="c:1", type="Company"),
                )
            )
        return out


class _RejectingGrounder:
    def ground(self, facts: Sequence[Fact], docs: Sequence[Document]) -> Sequence[Fact]:
        return [
            f.model_copy(update={"verdict": GroundingVerdict.SUPPORTED})
            for f in facts
            if f.predicate != "employer"
        ]


def test_pipeline_runs_with_only_an_extractor() -> None:
    """No grounder and no corroborator must degrade to candidates, not to an error."""
    kg = Pipeline(Ontology(name="demo"), _StubExtractor()).run([Document(id="d1", text="Ada.")])
    assert len(kg) == 2
    assert kg.ontology_name == "demo"
    assert all(f.verdict is GroundingVerdict.UNCHECKED for f in kg.facts)


def test_entities_are_collected_from_subjects_and_edge_objects() -> None:
    kg = Pipeline(Ontology(), _StubExtractor()).run([Document(id="d1", text="Ada.")])
    assert {e.key for e in kg.entities} == {"p:d1", "c:1"}


def test_grounder_can_drop_facts() -> None:
    kg = Pipeline(Ontology(), _StubExtractor(), grounder=_RejectingGrounder()).run(
        [Document(id="d1", text="Ada.")]
    )
    assert [f.predicate for f in kg.facts] == ["name"]
    assert kg.facts[0].verdict is GroundingVerdict.SUPPORTED


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
    assert manifest["ontology"] == "demo"


def test_stub_stages_satisfy_the_declared_protocols() -> None:
    """If a stub stops matching, callers' own implementations would break too."""
    from odke.pipeline import Extractor, Grounder, Sink

    assert isinstance(_StubExtractor(), Extractor)
    assert isinstance(_RejectingGrounder(), Grounder)
    assert isinstance(JsonlSink("/tmp/unused"), Sink)


def test_knowledge_graph_is_serialisable() -> None:
    """Anything that crosses a process boundary has to survive JSON."""
    kg = Pipeline(Ontology(), _StubExtractor()).run([Document(id="d1", text="Ada.")])
    assert KnowledgeGraph.model_validate_json(kg.model_dump_json()).facts[0].predicate == "name"
