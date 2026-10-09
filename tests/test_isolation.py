"""Per-document error isolation (#162): one bad document never fails the batch.

A document whose chunking, extraction, grounding or normalising raises is left
out whole, named with its reason in `stats["failed"]` and the report, and the
rest go on. A configuration error still stops the run. No model is called.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from openodke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    Ontology,
    Pipeline,
    Span,
)
from openodke.chunking import SentenceChunker
from openodke.extract import HybridExtractor, LLMExtractor
from openodke.ground import LLMGrounder, RetryPolicy
from openodke.llm import (
    Completion,
    Message,
    MissingAPIKey,
    ModelRoles,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
    RecordedClient,
)
from openodke.sinks.jsonl import JsonlSink

SPEC = ModelSpec(model="test/model")
ONCE = RetryPolicy(attempts=1)


def _born(i: int) -> str:
    return f"Person {i} was born in {1800 + i}."


def _reply(i: int) -> dict[str, Any]:
    return {
        "entities": [
            {
                "type": "Person",
                "name": f"Person {i}",
                "facts": [
                    {
                        "predicate": "birth_date",
                        "value": str(1800 + i),
                        "quote": f"born in {1800 + i}",
                    }
                ],
            }
        ]
    }


def _client(bad: Sequence[int], n: int) -> RecordedClient:
    """Answers every passage but the bad ones, which fail as a provider does after its retries."""
    entries: list[dict[str, Any]] = [{"match": "Claim:", "response": {"verdict": "supported"}}]
    entries += [{"match": _born(i), "error": "500 malformed response body"} for i in bad]
    entries += [{"match": _born(i), "response": _reply(i)} for i in range(n) if i not in bad]
    return RecordedClient(entries)


def _documents(n: int) -> list[Document]:
    return [Document(id=f"d{i}", text=_born(i)) for i in range(n)]


@pytest.mark.parametrize("workers", [1, 8])
def test_one_document_s_extraction_failure_leaves_the_rest_to_finish(
    people: Ontology, tmp_path: Path, workers: int
) -> None:
    client = _client([3], 12)
    extractor = LLMExtractor(
        client=client, spec=SPEC, types=["Person"], retry=ONCE, max_workers=workers
    )
    grounder = LLMGrounder(ModelRoles.single("test/model"), client=client)
    kg = Pipeline(people, extractor, grounder=grounder, sinks=[JsonlSink(tmp_path)]).run(
        _documents(12)
    )

    assert set(kg.stats["failed"]) == {"d3"}
    assert kg.stats["failed"]["d3"].startswith("extract: ProviderError: 500 malformed")
    cited = {f.evidence[0].doc_id for f in kg.facts}
    assert cited == {f"d{i}" for i in range(12)} - {"d3"}
    assert {f.verdict for f in kg.facts} == {GroundingVerdict.SUPPORTED}
    # Every other document's call went out: nothing was cancelled for d3.
    assert len([c for c in client.calls if "Claim:" not in c[0][-1].content]) == 12
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["failed"] == kg.stats["failed"]


def test_a_document_that_cannot_be_chunked_is_left_out(people: Ontology) -> None:
    class _Picky(SentenceChunker):
        def chunk(self, doc: Document) -> Iterable[Chunk]:
            if doc.id == "d1":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            return super().chunk(doc)

    client = _client([], 3)
    extractor = LLMExtractor(client=client, spec=SPEC, types=["Person"])
    kg = Pipeline(people, extractor, chunker=_Picky()).run(_documents(3))
    assert list(kg.stats["failed"]) == ["d1"]
    assert kg.stats["failed"]["d1"].startswith("chunk: UnicodeDecodeError")
    assert {f.evidence[0].doc_id for f in kg.facts} == {"d0", "d2"}


class _Grounder:
    """A grounder of someone else's, by document, that cannot read one of them."""

    def ground_many(self, facts: Sequence[Fact], doc: Document) -> list[Fact]:
        if doc.id == "d2":
            raise ValueError("could not parse the grounding reply")
        return [f.model_copy(update={"verdict": GroundingVerdict.SUPPORTED}) for f in facts]

    def ground(self, fact: Fact, doc: Document) -> Fact:  # pragma: no cover - the Protocol's
        return self.ground_many([fact], doc)[0]


class _Given:
    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        span = Span(doc_id=chunk.doc_id, start=0, end=len(chunk.text), quote=chunk.text)
        return [
            Fact(
                subject=Entity(key=f"p:{chunk.doc_id}", type="Person", label=chunk.doc_id),
                predicate="birth_date",
                object_value=chunk.doc_id,
                evidence=(Evidence(doc_id=chunk.doc_id, span=span),),
            )
        ]


def test_one_document_s_grounding_failure_is_its_own(people: Ontology) -> None:
    kg = Pipeline(people, _Given(), grounder=_Grounder()).run(_documents(4))
    assert kg.stats["failed"] == {"d2": "ground: ValueError: could not parse the grounding reply"}
    assert sorted(f.evidence[0].doc_id for f in kg.facts) == ["d0", "d1", "d3"]


def test_a_batch_that_does_not_say_which_document_failed_is_asked_one_by_one(
    people: Ontology,
) -> None:
    class _Silent(_Grounder):
        def ground_documents(
            self, batches: Sequence[tuple[Sequence[Fact], Document]]
        ) -> list[list[Fact]]:
            if len(batches) > 1:
                raise ValueError("something in this batch")
            return [self.ground_many(facts, doc) for facts, doc in batches]

    kg = Pipeline(people, _Given(), grounder=_Silent()).run(_documents(4))
    assert list(kg.stats["failed"]) == ["d2"]
    assert len(kg.facts) == 3


def test_a_hybrid_extractor_isolates_a_chunk_its_model_path_failed_on(people: Ontology) -> None:
    client = _client([1], 4)
    llm = LLMExtractor(client=client, spec=SPEC, types=["Person"], retry=ONCE, max_workers=4)
    docs = _documents(4)
    hybrid = HybridExtractor(llm, documents=docs)
    kg = Pipeline(people, hybrid).run(docs)
    assert list(kg.stats["failed"]) == ["d1"]
    assert {f.evidence[0].doc_id for f in kg.facts} == {"d0", "d2", "d3"}


@pytest.mark.parametrize(
    "error",
    [
        MissingAPIKey("test/model needs an API key in TEST_API_KEY, which is not set"),
        ProviderNotInstalled("litellm is not installed"),
    ],
    ids=["missing key", "missing adapter"],
)
def test_a_configuration_error_still_stops_the_run(people: Ontology, error: Exception) -> None:
    class _Unconfigured:
        def complete(
            self, messages: Sequence[Message], *, spec: ModelSpec, schema: Any = None
        ) -> Completion:
            raise error

    extractor = LLMExtractor(client=_Unconfigured(), spec=SPEC, types=["Person"], max_workers=4)
    with pytest.raises(type(error)):
        Pipeline(people, extractor).run(_documents(6))
    grounder = LLMGrounder(ModelRoles.single("test/model"), client=_Unconfigured())
    with pytest.raises(type(error)):
        Pipeline(people, _Given(), grounder=grounder).run(_documents(3))


def test_the_extractor_s_own_batch_says_which_chunks_failed(people: Ontology) -> None:
    chunks = [Chunk(doc_id=f"d{i}", index=0, text=_born(i), start=0, end=27) for i in range(20)]
    extractor = LLMExtractor(
        client=_client([4, 11], 20), spec=SPEC, types=["Person"], retry=ONCE, max_workers=8
    )
    with pytest.raises(ProviderError, match="500") as raised:
        extractor.extract_many(chunks, people)
    partial = raised.value.partial  # type: ignore[attr-defined]
    assert sorted(raised.value.failures) == [4, 11]  # type: ignore[attr-defined]
    assert [i for i, row in enumerate(partial) if row is None] == [4, 11]
    assert sum(len(row) for row in partial if row is not None) == 18


# --------------------------------------------------------------------------- #
# odke run
# --------------------------------------------------------------------------- #


def test_a_failed_gap_window_costs_that_window_and_the_first_pass_stands() -> None:
    from openodke.reextract import Reextract
    from test_reextract import DOC, ONTOLOGY, _grounder, _Planted

    class _Unreachable(_Planted):
        def reextract(
            self, window: Chunk, relations: list[str], already: list[Fact], ontology: Ontology
        ) -> list[Fact]:
            raise ProviderError("500 malformed response body")

    kg = Pipeline(ONTOLOGY, _Unreachable(), grounder=_grounder(), reextract=Reextract()).run([DOC])
    assert "failed" not in kg.stats
    assert kg.stats["reextract"]["failed"] == 1 and kg.stats["reextract"]["windows"] == 1
    assert {f.predicate for f in kg.facts} == {"employer", "headquarters"}
