"""The paper's grounder (ODKE+ §3.3.1, App. B): whole context, a triple, True or False."""

from __future__ import annotations

import pytest

from openodke import Document, Entity, Evidence, Fact, Span
from openodke.gate import VerdictGate
from openodke.ground import LLMGrounder
from openodke.ground.llm import (
    BINARY_SCHEMA,
    GROUNDING_SCHEMA,
    PAPER_PROMPT,
    SYSTEM_PROMPT,
    build_messages,
    render_triple,
)
from openodke.llm import ScriptedClient
from openodke.types import GroundingVerdict, Polarity

TEXT = "Ada Lovelace wrote the first algorithm in 1843. She was born in London."
DOC = Document(id="d1", text=TEXT)
ADA = Entity(key="p1", type="Person", label="Ada Lovelace")
QUOTE = "Ada Lovelace wrote the first algorithm in 1843."
WROTE = Fact(
    subject=ADA,
    predicate="wrote",
    object_value="the first algorithm",
    evidence=(Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=len(QUOTE), quote=QUOTE)),),
)


def test_render_triple_has_the_papers_shape() -> None:
    assert render_triple(WROTE) == "<Ada Lovelace, wrote, the first algorithm>"
    dated = WROTE.model_copy(
        update={"qualifiers": {"point_in_time": 1843}, "identity_keys": ("point_in_time",)}
    )
    assert render_triple(dated) == "<Ada Lovelace, wrote(point_in_time: 1843), the first algorithm>"
    denied = WROTE.model_copy(update={"polarity": Polarity.DENIED})
    assert render_triple(denied) == "<Ada Lovelace, not wrote, the first algorithm>"


def test_the_default_is_unchanged_span_and_three_way() -> None:
    system, user = build_messages(WROTE, QUOTE)
    assert system.content == SYSTEM_PROMPT
    assert user.content.startswith("Claim: ")
    client = ScriptedClient([{"verdict": "supported"}])
    grounded = LLMGrounder(client=client).ground(WROTE, DOC)
    assert grounded.verdict is GroundingVerdict.SUPPORTED
    messages, _, schema = client.calls[0]
    assert schema == GROUNDING_SCHEMA
    assert "born in London" not in messages[1].content


def test_paper_mode_sends_the_papers_prompt_the_whole_document_and_a_boolean_schema() -> None:
    client = ScriptedClient([{"verdict": True}])
    grounder = LLMGrounder(client=client, context="document", verdicts="binary")
    grounded = grounder.ground(WROTE, DOC)
    assert grounded.verdict is GroundingVerdict.SUPPORTED
    messages, _, schema = client.calls[0]
    assert messages[0].content == PAPER_PROMPT
    assert messages[1].content == (
        f"**Context:\n{TEXT}\n**triple:\n<Ada Lovelace, wrote, the first algorithm>"
    )
    assert schema == BINARY_SCHEMA


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        ({"verdict": False}, GroundingVerdict.NOT_FOUND),
        ("True", GroundingVerdict.SUPPORTED),
        ("false.", GroundingVerdict.NOT_FOUND),
        ({"verdict": "yes"}, GroundingVerdict.SUPPORTED),
        ("It is probably true", GroundingVerdict.UNCHECKED),
    ],
)
def test_binary_answers_are_read_and_prose_is_not_guessed(
    answer: object, verdict: GroundingVerdict
) -> None:
    client = ScriptedClient([answer])  # type: ignore[list-item]
    grounded = LLMGrounder(client=client, verdicts="binary").ground(WROTE, DOC)
    assert grounded.verdict is verdict


def test_with_the_strict_gate_only_affirmed_facts_survive() -> None:
    """The paper keeps 'only those facts that receive an affirmative grounding judgment'."""
    client = ScriptedClient([{"verdict": False}])
    grounded = LLMGrounder(client=client, context="document", verdicts="binary").ground(WROTE, DOC)
    gate = VerdictGate(refuse_not_found=True)
    assert gate.validate(grounded, ontology=None).action == "refuse"  # type: ignore[arg-type]


@pytest.mark.parametrize(("option", "value"), [("context", "page"), ("verdicts", "yes_no")])
def test_an_unknown_mode_is_refused(option: str, value: str) -> None:
    with pytest.raises(ValueError, match=option):
        LLMGrounder(client=ScriptedClient(), **{option: value})  # type: ignore[arg-type]
