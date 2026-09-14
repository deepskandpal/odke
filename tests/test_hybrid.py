"""The hybrid extractor: routed by modality, merged by signature, and cheaper for it."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from openodke import (
    Chunk,
    Document,
    Entity,
    Extractor,
    Fact,
    HybridExtractor,
    LLMExtractor,
    Ontology,
    Pipeline,
    Polarity,
    SentenceChunker,
    SourceTier,
)
from openodke.extract.hybrid import merge
from openodke.llm import Completion, Message, ModelSpec, ReplayClient, ScriptedClient
from openodke.loaders import DirectoryLoader, MarkdownLoader

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
SONNET = ModelSpec(model="anthropic/claude-sonnet-5")

NOTES = "# Grace Hopper\n\nGrace Hopper was born in 1906. She worked for the US Navy.\n"
STAFF = (
    "## Staff\n\nAda Lovelace was born on 10 December 1815. She worked for Babbage & Co.\n\n"
    "| Name | Born |\n|------|------|\n| Ada Lovelace | 1815-12-10 |\n"
)


def _corpus(tmp_path: Path) -> list[Document]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "people.csv").write_text(
        "name,born,employer\nAda Lovelace,1815-12-10,Babbage & Co\nAlan Turing,1912-06-23,\n",
        encoding="utf-8",
    )
    (corpus / "notes.md").write_text(NOTES, encoding="utf-8")
    staff = tmp_path / "staff.md"
    staff.write_text(STAFF, encoding="utf-8")
    docs = list(DirectoryLoader(tier=SourceTier.AUTHORITATIVE).load(corpus))
    return docs + MarkdownLoader(modality="semi_structured").load(staff)


class _EmptyReplies:
    """A model that finds nothing, counting what it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        self.calls += 1
        return Completion(text='{"entities": []}', model=spec.model)


def test_a_mixed_corpus_costs_fewer_model_calls_than_model_only(
    tmp_path: Path, people: Ontology
) -> None:
    """The number that is the whole argument for the hybrid design."""
    docs = _corpus(tmp_path)
    assert [d.modality for d in docs] == [
        "unstructured",
        "structured",
        "structured",
        "semi_structured",
    ]
    client = ReplayClient(FIXTURES / "llm_hybrid_corpus.json")
    hybrid = HybridExtractor(LLMExtractor(client=client, spec=SONNET), documents=docs)
    kg = Pipeline(people, hybrid, chunker=SentenceChunker()).run(docs)

    baseline = _EmptyReplies()
    Pipeline(people, LLMExtractor(client=baseline, spec=SONNET), chunker=SentenceChunker()).run(
        docs
    )
    totals = hybrid.totals()
    assert kg.stats["chunks"] == baseline.calls == 4
    assert totals.model_calls == 2
    assert client.exhausted
    assert totals.cost_usd == pytest.approx(0.0062)
    assert (totals.prompt_tokens, totals.completion_tokens) == (1240, 290)

    by_id = {d.id: d for d in docs}
    for fact in kg.facts:
        span = fact.evidence[0].span
        assert span is not None and span.is_faithful(by_id[fact.evidence[0].doc_id])
    assert len(kg) == 11


def test_each_document_reports_the_paths_that_ran(tmp_path: Path, people: Ontology) -> None:
    docs = _corpus(tmp_path)
    notes, ada_row, turing_row, staff = docs
    client = ReplayClient(FIXTURES / "llm_hybrid_corpus.json")
    hybrid = HybridExtractor(LLMExtractor(client=client, spec=SONNET), documents=docs)
    Pipeline(people, hybrid, chunker=SentenceChunker()).run(docs)

    assert hybrid.report[ada_row.id].paths == {"pattern"}
    assert hybrid.report[ada_row.id].model_calls == 0
    assert hybrid.report[turing_row.id].pattern_facts == 2
    assert hybrid.report[notes.id].paths == {"llm"}
    assert (hybrid.report[notes.id].llm_facts, hybrid.report[notes.id].model_calls) == (3, 1)
    mixed = hybrid.report[staff.id]
    assert mixed.paths == {"pattern", "llm"}
    assert (mixed.pattern_facts, mixed.llm_facts, mixed.merged) == (2, 3, 2)


def test_both_paths_on_a_mixed_page_merge_and_the_pattern_fact_wins(
    tmp_path: Path, people: Ontology
) -> None:
    docs = _corpus(tmp_path)
    staff = docs[-1]
    client = ReplayClient(FIXTURES / "llm_hybrid_corpus.json")
    hybrid = HybridExtractor(LLMExtractor(client=client, spec=SONNET), documents=docs)
    chunks = list(SentenceChunker().chunk(staff))
    facts = [f for c in chunks for f in hybrid.extract(c, people)]
    by_predicate = {f.predicate: f for f in facts}
    assert set(by_predicate) == {"full_name", "birth_date", "employer"}
    # The table cell and the sentence said the same thing; the exact read is kept.
    assert by_predicate["birth_date"].extractor == "pattern"
    assert by_predicate["full_name"].extractor == "pattern"
    # Only the sentence names the employer.
    assert by_predicate["employer"].extractor == "llm"
    assert by_predicate["employer"].evidence[0].uri == staff.uri


def test_structured_documents_never_reach_the_model(people: Ontology) -> None:
    doc = Document(text="name: Ada Lovelace\nborn: 1815-12-10", modality="structured")
    chunk = Chunk(doc_id=doc.id, start=0, end=len(doc.text), text=doc.text, index=0)
    # A strict client with nothing queued raises if it is ever called.
    llm = LLMExtractor(client=ScriptedClient([]), spec=SONNET)
    facts = HybridExtractor(llm, documents=[doc]).extract(chunk, people)
    assert {f.predicate for f in facts} == {"full_name", "birth_date"}
    assert llm.calls == []


def test_without_a_model_prose_yields_nothing_and_a_mix_gets_the_pattern_path(
    people: Ontology,
) -> None:
    prose = Document(text="Ada Lovelace was born in 1815.")
    mixed = Document(text="Name: Ada Lovelace\nBorn: 1815-12-10", modality="semi_structured")
    hybrid = HybridExtractor(documents=[prose, mixed])
    for doc, expected in ((prose, 0), (mixed, 2)):
        chunk = Chunk(doc_id=doc.id, start=0, end=len(doc.text), text=doc.text, index=0)
        assert len(hybrid.extract(chunk, people)) == expected


def test_an_unknown_document_cannot_be_routed(people: Ontology) -> None:
    chunk = Chunk(doc_id="nowhere", start=0, end=3, text="Ada", index=0)
    with pytest.raises(LookupError, match="documents="):
        HybridExtractor().extract(chunk, people)


def test_unreported_cost_stays_unknown(people: Ontology) -> None:
    doc = Document(text="Grace Hopper was born in 1906.")
    chunk = Chunk(doc_id=doc.id, start=0, end=len(doc.text), text=doc.text, index=0)
    llm = LLMExtractor(client=ScriptedClient(['{"entities": []}']), spec=SONNET)
    hybrid = HybridExtractor(llm, documents=[doc])
    hybrid.extract(chunk, people)
    assert hybrid.totals().model_calls == 1
    assert hybrid.totals().cost_usd is None


def test_merge_keeps_the_most_confident_and_never_merges_a_denial() -> None:
    subject = Entity(key="Person:ada", type="Person")
    low = Fact(subject=subject, predicate="born", object_value="1815", confidence=0.5)
    high = low.model_copy(update={"confidence": 1.0, "extractor": "pattern"})
    tie = low.model_copy(update={"extractor": "second"})
    denied = low.model_copy(update={"polarity": Polarity.DENIED})
    assert merge([low, high, tie, denied]) == [high, denied]
    assert merge([low, tie]) == [low]


def test_it_is_an_extractor() -> None:
    assert isinstance(HybridExtractor(), Extractor)
