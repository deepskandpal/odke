"""The model proposer (#32) on recorded responses: no key, no network, every claim audited."""

from __future__ import annotations

from pathlib import Path

import pytest

from odke import Document
from odke.infer import sample_corpus
from odke.infer.candidates import Proposals
from odke.infer.llm import LLMProposer, render_candidates
from odke.infer.propose import propose
from odke.infer.sample import CorpusSample
from odke.llm import ModelRoles, ModelSpec, ReplayClient, ScriptedClient
from odke.loaders import DirectoryLoader

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
SONNET = ModelSpec(model="anthropic/claude-sonnet-5")
NOTES = (
    "Ada Lovelace works at Acme Corp. Grace Hopper works at Globex.\n\n"
    "Acme Corp is headquartered in London.\n"
)


def _corpus(tmp_path: Path) -> tuple[CorpusSample, Proposals, dict[str, Document]]:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "people.csv").write_text(
        "name,employer\nAda Lovelace,Acme Corp\nGrace Hopper,Globex\n", encoding="utf-8"
    )
    (root / "companies.csv").write_text("name\nAcme Corp\nGlobex\n", encoding="utf-8")
    (root / "notes.md").write_text(NOTES, encoding="utf-8")
    docs = list(DirectoryLoader().load(root))
    sample = sample_corpus(docs, words=10_000)
    return sample, propose(sample), {d.id: d for d in docs}


def _replayed(tmp_path: Path) -> tuple[LLMProposer, Proposals, Proposals, dict[str, Document]]:
    sample, found, docs = _corpus(tmp_path)
    proposer = LLMProposer(client=ReplayClient(FIXTURES / "infer_people.json"), spec=SONNET)
    return proposer, found, proposer.propose(sample, found), docs


def test_the_model_names_what_the_candidates_found(tmp_path: Path) -> None:
    proposer, found, named, docs = _replayed(tmp_path)
    assert [t.name for t in named.types] == ["Person", "Company", "Place"]
    assert [p.name for p in named.predicates] == ["employer", "full_name"]
    assert all(c.proposer == "llm" for c in [*named.types, *named.predicates])
    person = named.types[0]
    (records_person,) = [t for t in found.types if t.name == "Person"]
    assert person.evidence == records_person.evidence
    assert person.keys == ("name",) and person.description == "A person named in the staff records."
    assert [t.rank for t in named.types] == [0, 1, 3]


def test_a_merge_carries_the_claimed_names_as_aliases_and_their_evidence(tmp_path: Path) -> None:
    _, found, named, _ = _replayed(tmp_path)
    employer = named.predicates[0]
    assert (employer.domain, employer.range, employer.cardinality) == (
        ("Person",),
        "Company",
        "single",
    )
    assert employer.aliases == ("works_at",)
    claimed = [p for p in found.predicates if p.name in {"employer", "works_at"}]
    assert employer.support == len({s.doc_id for c in claimed for s in c.evidence})
    assert set(employer.observations) == {o for c in claimed for o in c.observations}


def test_a_claim_by_name_stays_inside_the_domain_the_model_gave(tmp_path: Path) -> None:
    _, found, named, _ = _replayed(tmp_path)
    full_name = named.predicates[1]
    assert full_name.aliases == ("name",)
    (person_name,) = [p for p in found.predicates if p.name == "name" and p.domain == ("Person",)]
    assert full_name.evidence == person_name.evidence


def test_what_has_no_evidence_is_rejected_and_recorded(tmp_path: Path) -> None:
    proposer, _, named, docs = _replayed(tmp_path)
    assert sorted((r.kind, r.name, r.reason) for r in proposer.rejections) == [
        ("predicate", "full_name", "'from' names no candidate: 'ghost'"),
        ("predicate", "headquarters", "quote not in the passages"),
        ("type", "City", "no evidence: claims no candidate and quotes no passage"),
    ]
    place = named.types[2]
    (span,) = place.evidence
    assert span.quote == "headquartered in London" and span.is_faithful(docs[span.doc_id])


def test_the_prompt_shows_candidates_and_passages_and_asks_for_the_contract(
    tmp_path: Path,
) -> None:
    proposer, found, _, _ = _replayed(tmp_path)
    assert isinstance(proposer.client, ReplayClient)
    ((messages, spec, schema),) = proposer.client.calls
    assert spec == SONNET and schema is not None and schema["title"] == "ontology_proposal"
    system, user = messages
    assert "not to invent a schema of your own" in system.content
    assert "- employer: Person -> Company, single (support 2, records)" in user.content
    assert "[1] " in user.content and "notes.md" in user.content
    assert user.content.index("Candidates") < user.content.index("Passages")
    (call,) = proposer.calls
    assert (call.prompt_tokens, call.completion_tokens, call.cost_usd) == (940, 310, 0.0075)
    assert render_candidates(found) in user.content


def test_the_default_model_is_the_infer_role() -> None:
    assert LLMProposer().spec == ModelRoles().infer
    local = ModelRoles.single("ollama/llama3.1")
    assert LLMProposer(roles=local).spec.model == "ollama/llama3.1"


def test_a_malformed_reply_is_repaired_once_then_recorded(tmp_path: Path) -> None:
    sample, found, _ = _corpus(tmp_path)
    repaired = LLMProposer(
        client=ScriptedClient(["not json", {"types": [{"name": "Person"}], "predicates": []}]),
        spec=SONNET,
    )
    assert [t.name for t in repaired.propose(sample, found).types] == ["Person"]
    assert [c.repair for c in repaired.calls] == [False, True]

    broken = LLMProposer(client=ScriptedClient(["no", "still no"]), spec=SONNET)
    assert broken.propose(sample, found) == Proposals()
    assert [(r.kind, r.reason) for r in broken.rejections] == [("reply", "malformed reply")]


def test_passages_are_a_prefix_of_the_sample_capped_by_words(tmp_path: Path) -> None:
    docs = [Document(text=f"Document {i} has exactly six words.") for i in range(20)]
    sample = sample_corpus(docs, words=10_000)
    proposer = LLMProposer(client=ScriptedClient([]), spec=SONNET, prompt_words=13)
    assert proposer.passages(sample) == list(sample.chunks[:3])


def test_nothing_to_name_makes_no_call() -> None:
    proposer = LLMProposer(client=ScriptedClient([]), spec=SONNET)
    assert proposer.propose(sample_corpus([]), Proposals()) == Proposals()
    assert proposer.calls == []


@pytest.mark.parametrize("reply", [{"types": "x", "predicates": []}, {"answer": 1}, "[1, 2]"])
def test_a_reply_of_the_wrong_shape_is_malformed(tmp_path: Path, reply: object) -> None:
    sample, found, _ = _corpus(tmp_path)
    proposer = LLMProposer(client=ScriptedClient([reply, reply]), spec=SONNET)  # type: ignore[list-item]
    assert proposer.propose(sample, found) == Proposals()
