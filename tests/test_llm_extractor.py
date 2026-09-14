"""The model extractor on recorded responses: no key, no network, every span checked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odke import Chunk, Document, Extractor, Fact, Ontology, Polarity, SourceTier
from odke.extract import LLMExtractor, response_schema
from odke.llm import ModelRoles, ModelSpec, OpenAICompatClient, ReplayClient, ScriptedClient

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
SONNET = ModelSpec(model="anthropic/claude-sonnet-5")
LLAMA = ModelSpec(model="ollama/llama3.1")
PREFACE = "Preface paragraph.\n\n"
ADA = (
    "Ada Lovelace was born on 10 December 1815 in London. From 1833 she worked with "
    "Charles Babbage at Babbage & Co. She never worked for the Royal Society."
)
GRACE = "Grace Hopper was born in 1906."
ACME = "Acme Corp reported 99.9% uptime at p50 and 99.9% uptime at p95 during 2024."


def _doc_and_chunk(passage: str, prefix: str = "") -> tuple[Document, Chunk]:
    doc = Document(
        text=prefix + passage, uri="https://example.org/ada", tier=SourceTier.AUTHORITATIVE
    )
    chunk = Chunk(doc_id=doc.id, start=len(prefix), end=len(doc.text), text=passage, index=1)
    return doc, chunk


def _triples(facts: list[Fact]) -> set[tuple[str, str, Any, Polarity]]:
    return {
        (
            f.subject.key,
            f.predicate,
            f.object_entity.key if f.object_entity else f.object_value,
            f.polarity,
        )
        for f in facts
    }


def _ada(people: Ontology, spec: ModelSpec = SONNET) -> tuple[LLMExtractor, list[Fact], Document]:
    doc, chunk = _doc_and_chunk(ADA, PREFACE)
    client = ReplayClient(FIXTURES / "llm_ada_lovelace.json")
    extractor = LLMExtractor(client=client, spec=spec, documents=[doc])
    return extractor, extractor.extract(chunk, people), doc


def test_sound_facts_survive_and_every_fault_is_dropped_and_recorded(people: Ontology) -> None:
    extractor, facts, doc = _ada(people)
    assert _triples(facts) == {
        ("Person:ada lovelace", "full_name", "Ada Lovelace", Polarity.ASSERTED),
        ("Person:ada lovelace", "birth_date", "1815-12-10", Polarity.ASSERTED),
        ("Person:ada lovelace", "employer", "Company:babbage & co", Polarity.ASSERTED),
        ("Person:ada lovelace", "employer", "Company:royal society", Polarity.DENIED),
        ("Company:babbage & co", "legal_name", "Babbage & Co", Polarity.ASSERTED),
    }
    assert sorted(r.reason for r in extractor.rejections) == [
        "predicate not in the snippet",
        "quote not in the passage",
        "quote not in the passage",
        "quote not in the passage",
        "type not in the prompt: 'City'",
        "unknown polarity",
    ]
    invented = {r.quote for r in extractor.rejections if r.reason == "quote not in the passage"}
    assert invented == {"born on 11 December 1815", "Augusta Ada King", "ada lovelace"}


def test_every_span_that_survives_is_faithful_to_the_document(people: Ontology) -> None:
    _, facts, doc = _ada(people)
    for fact in facts:
        (evidence,) = fact.evidence
        assert evidence.span is not None and evidence.span.is_faithful(doc)
        assert (evidence.uri, evidence.tier) == (doc.uri, SourceTier.AUTHORITATIVE)
        assert (fact.extractor, fact.confidence) == ("llm", 0.5)
    born = next(f for f in facts if f.predicate == "birth_date")
    # The model claimed offset 3. The quote is really at 25 of the passage, which
    # is 25 + len(PREFACE) of the document: moved, never invented.
    span = born.evidence[0].span
    assert span is not None
    assert span.start == doc.text.index("10 December 1815") == len(PREFACE) + 25


def test_edges_qualifiers_and_identity_keys_come_from_the_ontology(people: Ontology) -> None:
    _, facts, _ = _ada(people)
    worked = next(f for f in facts if f.object_entity and f.polarity is Polarity.ASSERTED)
    company = next(f for f in facts if f.predicate == "legal_name")
    # The edge's object is keyed exactly as the company's own facts are.
    assert worked.object_entity is not None and worked.object_entity.key == company.subject.key
    # Undeclared qualifiers are dropped; start_date is reconcilable, so no identity keys.
    assert worked.qualifiers == {"start_date": "1833"}
    assert worked.identity_keys == ()


def test_identity_qualifiers_keep_two_measurements_apart(people: Ontology) -> None:
    doc, chunk = _doc_and_chunk(ACME)
    client = ReplayClient(FIXTURES / "llm_uptime_percentiles.json")
    facts = LLMExtractor(client=client, spec=SONNET, documents=[doc]).extract(chunk, people)
    p50, p95 = (f for f in facts if f.predicate == "uptime")
    assert p50.identity_keys == p95.identity_keys == ("percentile",)
    assert p50.signature != p95.signature
    later = p50.model_copy(update={"qualifiers": {**p50.qualifiers, "year": "2025"}})
    assert later.signature == p50.signature


def test_the_prompt_is_the_snippet_and_the_contract_is_its_schema(people: Ontology) -> None:
    doc, chunk = _doc_and_chunk(ADA, PREFACE)
    client = ReplayClient(FIXTURES / "llm_ada_lovelace.json")
    extractor = LLMExtractor(client=client, spec=SONNET, documents=[doc])
    extractor.extract(chunk, people)

    ((messages, spec, schema),) = client.calls
    snippets = [people.snippet("Company"), people.snippet("Person")]
    assert spec == SONNET
    assert all(s.render() in messages[0].content for s in snippets)
    assert messages[1].content == chunk.text
    assert schema == response_schema(snippets)

    values = snippets[1].json_schema()["properties"]
    entity_kinds = schema["properties"]["entities"]["items"]["anyOf"]
    person = next(e for e in entity_kinds if e["title"] == "Person")
    fact_kinds = {
        f["properties"]["predicate"]["const"]: f["properties"]
        for f in person["properties"]["facts"]["items"]["anyOf"]
    }
    assert set(fact_kinds) == set(values)
    assert fact_kinds["birth_date"]["value"] == values["birth_date"]
    # Multi-valued: one fact holds one value, so the item schema is the contract.
    assert fact_kinds["employer"]["value"] == values["employer"]["items"]
    assert set(fact_kinds["employer"]["qualifiers"]["properties"]) == {"start_date"}

    (call,) = extractor.calls
    assert (call.prompt_tokens, call.completion_tokens, call.cost_usd) == (812, 431, 0.00891)
    assert (call.doc_id, call.chunk_index, call.repair) == (doc.id, 1, False)


def test_a_local_and_a_hosted_model_extract_identically(people: Ontology) -> None:
    """Nothing in the extractor branches on the provider; only the spec differs."""
    hosted, hosted_facts, _ = _ada(people, SONNET)
    local, local_facts, _ = _ada(people, LLAMA)
    assert _triples(hosted_facts) == _triples(local_facts)

    def spans(facts: list[Fact]) -> list[tuple[int, int, str | None]]:
        return [(s.start, s.end, s.quote) for f in facts if (s := f.evidence[0].span)]

    assert spans(hosted_facts) == spans(local_facts)
    assert len(spans(hosted_facts)) == len(hosted_facts)
    assert hosted.calls[0].model == local.calls[0].model == "claude-sonnet-5"


def test_a_malformed_reply_is_repaired_once(people: Ontology) -> None:
    doc, chunk = _doc_and_chunk(GRACE)
    client = ReplayClient(FIXTURES / "llm_repair.json")
    extractor = LLMExtractor(client=client, spec=LLAMA, documents=[doc])
    (fact,) = extractor.extract(chunk, people)
    span = fact.evidence[0].span
    assert span is not None and span.resolve(doc) == "born in 1906"
    assert [c.repair for c in extractor.calls] == [False, True]
    repair = client.calls[1][0]
    assert [m.role for m in repair] == ["system", "user", "assistant", "user"]
    assert repair[2].content.startswith("Sure!")
    assert client.exhausted
    # Neither reply reported a cost: unknown, not zero.
    assert all(c.cost_usd is None for c in extractor.calls)


def test_an_unrepairable_reply_is_recorded_not_raised(people: Ontology) -> None:
    doc, chunk = _doc_and_chunk(GRACE)
    extractor = LLMExtractor(client=ScriptedClient(["nope", "still nope"]), spec=LLAMA)
    assert extractor.extract(chunk, people) == []
    assert [r.reason for r in extractor.rejections] == ["malformed reply"]
    assert len(extractor.calls) == 2
    single = LLMExtractor(client=ScriptedClient(["nope"]), spec=LLAMA, repairs=0)
    assert single.extract(chunk, people) == []
    assert len(single.calls) == 1


def test_structured_output_is_read_and_an_unregistered_chunk_is_still_checked(
    people: Ontology,
) -> None:
    reply = {
        "entities": [
            {
                "type": "Person",
                "name": "Grace Hopper",
                "facts": [
                    {"predicate": "birth_date", "value": "1906", "quote": "in 1906"},
                    {"predicate": "full_name", "value": "Grace Hopper", "quote": "Grace B. Hopper"},
                ],
            }
        ]
    }
    _, chunk = _doc_and_chunk(GRACE)
    extractor = LLMExtractor(client=ScriptedClient([reply]), spec=SONNET)
    (fact,) = extractor.extract(chunk, people)
    (evidence,) = fact.evidence
    assert evidence.uri is None and evidence.span is not None
    assert GRACE[evidence.span.start : evidence.span.end] == "in 1906"
    assert [r.quote for r in extractor.rejections] == ["Grace B. Hopper"]


def test_types_narrow_the_prompt_and_the_client_resolves_lazily(people: Ontology) -> None:
    extractor = LLMExtractor(roles=ModelRoles.single("ollama/llama3.1"), types=["Person"])
    assert extractor.spec.model == "ollama/llama3.1"
    assert [s.type_name for s in extractor.snippets(people)] == ["Person"]
    assert isinstance(extractor.client, OpenAICompatClient)
    assert LLMExtractor().spec == ModelRoles().extract


def test_nothing_to_ask_costs_nothing_and_no_schema_is_an_error(people: Ontology) -> None:
    client = ScriptedClient([])
    blank = Chunk(doc_id="d", start=0, end=2, text="  ", index=0)
    assert LLMExtractor(client=client, spec=SONNET).extract(blank, people) == []
    assert client.calls == []
    _, chunk = _doc_and_chunk(GRACE)
    with pytest.raises(ValueError, match="no entity types"):
        LLMExtractor(client=client, spec=SONNET).extract(chunk, Ontology())


def test_it_is_an_extractor() -> None:
    assert isinstance(LLMExtractor(client=ScriptedClient()), Extractor)
