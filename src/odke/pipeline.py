"""The five stages, and the seam between each pair.

The ODKE+ paper describes a pipeline of five components. This module declares
each one as a Protocol and composes them. The point of the protocols is that a
stage can be swapped without touching the others: a caller who already has their
own retrieval can pass their own `Retriever`, someone targeting Memgraph writes a
`Sink`, and the extraction contract does not move.

Stage 1 (Initiator) and stage 2 (Retriever) are optional here in a way they are
not in the paper. The paper's system decides *what* to refresh and goes and gets
it; an SDK is usually handed the documents. Both are still declared, because the
streaming/batch refresh loop is the interesting half for anyone maintaining a
live graph, and it should not require a fork to build.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from odke.ontology import Ontology
from odke.types import Document, Fact, KnowledgeGraph


@runtime_checkable
class Initiator(Protocol):
    """Stage 1 — decides what needs extracting. Optional; default is 'all of it'."""

    def targets(self) -> Iterable[str]: ...


@runtime_checkable
class Retriever(Protocol):
    """Stage 2 — turns targets into documents. Optional; default is the caller's list."""

    def retrieve(self, targets: Iterable[str]) -> Iterable[Document]: ...


@runtime_checkable
class Extractor(Protocol):
    """Stage 3 — documents to candidate facts, held to the ontology.

    Implementations are hybrid by design: a pattern extractor handles tables,
    key-value blocks and JSON exactly and for free, and a model handles prose.
    Both emit `Fact`, so nothing downstream branches on which one ran.
    """

    def extract(self, docs: Sequence[Document], ontology: Ontology) -> Sequence[Fact]: ...


@runtime_checkable
class Grounder(Protocol):
    """Stage 4 — drops any fact the evidence does not actually support.

    This is the precision stage. It is cheap by construction: a small model, one
    fact and one evidence span at a time, answering a yes/no question.
    """

    def ground(self, facts: Sequence[Fact], docs: Sequence[Document]) -> Sequence[Fact]: ...


@runtime_checkable
class Corroborator(Protocol):
    """Stage 5 — normalise, merge across sources, resolve conflicts, rank."""

    def corroborate(self, facts: Sequence[Fact]) -> KnowledgeGraph: ...


@runtime_checkable
class Sink(Protocol):
    """Where a finished graph goes. Neo4j, RDF, NetworkX, JSONL, or yours."""

    def write(self, kg: KnowledgeGraph) -> None: ...


class Pipeline:
    """Composes the stages. Deliberately boring — the stages hold the ideas.

    Every stage after extraction is optional so that the pipeline degrades to
    something useful rather than to an error: with no grounder you get candidate
    facts with an `UNCHECKED` verdict, which is the right default for a caller
    who wants recall and will filter themselves.
    """

    def __init__(
        self,
        ontology: Ontology,
        extractor: Extractor,
        *,
        retriever: Retriever | None = None,
        grounder: Grounder | None = None,
        corroborator: Corroborator | None = None,
        sinks: Sequence[Sink] = (),
    ) -> None:
        self.ontology = ontology
        self.extractor = extractor
        self.retriever = retriever
        self.grounder = grounder
        self.corroborator = corroborator
        self.sinks = tuple(sinks)

    def run(self, docs: Sequence[Document]) -> KnowledgeGraph:
        facts = list(self.extractor.extract(docs, self.ontology))
        if self.grounder is not None:
            facts = list(self.grounder.ground(facts, docs))
        kg = (
            self.corroborator.corroborate(facts)
            if self.corroborator is not None
            else KnowledgeGraph(
                entities=_entities_of(facts),
                facts=tuple(facts),
                ontology_name=self.ontology.name,
            )
        )
        for sink in self.sinks:
            sink.write(kg)
        return kg


def _entities_of(facts: Sequence[Fact]) -> tuple:
    """Every distinct entity mentioned as a subject or an edge's object."""
    seen: dict[str, object] = {}
    for f in facts:
        seen.setdefault(f.subject.key, f.subject)
        if f.object_entity is not None:
            seen.setdefault(f.object_entity.key, f.object_entity)
    return tuple(seen.values())


__all__ = [
    "Corroborator",
    "Extractor",
    "Grounder",
    "Initiator",
    "Pipeline",
    "Retriever",
    "Sink",
]
