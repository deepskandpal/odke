"""The grounder's two challengers (#134): Onyx's validator verbatim, and its three-way adaptation.

Every answer here is recorded (`RecordedClient`): what each prompt sends, how
its reply is read and repaired, and what the grounder reports. No call is made.
"""

from __future__ import annotations

from typing import Any

import pytest

from openodke import Document, Entity, Evidence, Fact, GroundingVerdict, Polarity, Span
from openodke.ground import LLMGrounder
from openodke.ground.llm import (
    GROUNDING_SCHEMA,
    build_messages,
    parse_verdict,
    render_sentence,
)
from openodke.llm import Completion, RecordedClient, ScriptedClient
from openodke.llm.cache import cache_key
from openodke.prompts import get

TEXT = "Ada Lovelace was born in London in 1815. She worked with Charles Babbage."
DOC = Document(id="d1", text=TEXT)
ADA = Entity(key="Person:ada", type="Person", label="Ada Lovelace")
LONDON = Entity(key="City:london", type="City", label="London")


def _fact(**update: Any) -> Fact:
    base = Fact(
        subject=ADA,
        predicate="born_in",
        object_entity=LONDON,
        evidence=(Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=40)),),
    )
    return base.model_copy(update=update)


def _text(text: str, parsed: dict[str, Any] | None = None) -> Completion:
    return Completion(text=text, parsed=parsed)


# --------------------------------------------------------------------------- #
# The fact as a sentence
# --------------------------------------------------------------------------- #


def test_a_fact_reads_as_one_plain_sentence() -> None:
    assert render_sentence(_fact()) == "Ada Lovelace born in London."
    year = Fact(subject=ADA, predicate="birth_year", object_value=1815)
    assert render_sentence(year) == "Ada Lovelace birth year 1815."
    named = Fact(subject=ADA, predicate="nickname", object_value="Enchantress of Numbers")
    assert render_sentence(named) == "Ada Lovelace nickname Enchantress of Numbers."


def test_a_denial_and_a_partial_claim_say_so_and_identity_qualifiers_stay() -> None:
    assert render_sentence(_fact(polarity=Polarity.DENIED)) == (
        "It is not the case that Ada Lovelace born in London."
    )
    assert render_sentence(_fact(polarity=Polarity.PARTIAL)) == (
        "Only partially, or under a stated condition: Ada Lovelace born in London."
    )
    scoped = _fact(
        qualifiers={"point_in_time": "1815", "odke.score": {}}, identity_keys=("point_in_time",)
    )
    assert render_sentence(scoped) == "Ada Lovelace born in London (point in time: 1815)."


# --------------------------------------------------------------------------- #
# What each prompt sends
# --------------------------------------------------------------------------- #


def test_onyx_is_one_user_message_its_template_filled_once() -> None:
    passage = "A passage with {braces} and {statement} in it."
    (message,) = build_messages(_fact(), passage, prompt="ground.onyx@1")
    head, rest = get("ground.onyx@1").text.split("{answer}")
    middle, tail = rest.split("{statement}")
    assert message.role == "user"
    assert message.content == head + passage + middle + "Ada Lovelace born in London." + tail
    # The passage's own braces are text: only the template's two fields are filled.
    assert "{braces} and {statement} in it." in message.content


def test_the_statement_prompt_is_its_instructions_and_its_frame() -> None:
    system, user = build_messages(_fact(), TEXT[:40], prompt="ground.statement")
    assert system.role == "system" and system.content == get("ground.statement@1").text
    assert user.content == (
        "## Passage\n```\nAda Lovelace was born in London in 1815.\n```\n\n"
        "## Statement\n```\nAda Lovelace born in London.\n```"
    )


def test_span_stays_the_default_message() -> None:
    assert build_messages(_fact(), "p") == build_messages(_fact(), "p", prompt="ground.span@1")


# --------------------------------------------------------------------------- #
# Reading, and repairing, each reply
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("yes", GroundingVerdict.SUPPORTED),
        ("Yes.", GroundingVerdict.SUPPORTED),
        ("**YES**", GroundingVerdict.SUPPORTED),
        ("yes\nThe answer states the birthplace.", GroundingVerdict.SUPPORTED),
        ("no", GroundingVerdict.NOT_FOUND),
        ("No, it does not say so.", GroundingVerdict.NOT_FOUND),
        ("  \nno\n", GroundingVerdict.NOT_FOUND),
        # Onyx reads the first line alone.
        ("Let me check.\nyes", None),
        ("The answer is consistent.", None),
        ("", None),
    ],
)
def test_onyx_s_reply_is_read_as_onyx_reads_it(
    reply: str, verdict: GroundingVerdict | None
) -> None:
    assert parse_verdict(_text(reply), prompt="ground.onyx@1") is verdict


@pytest.mark.parametrize(
    ("completion", "verdict"),
    [
        (_text("", {"verdict": "contradicted"}), GroundingVerdict.CONTRADICTED),
        (_text('```json\n{"verdict": "not_found"}\n```'), GroundingVerdict.NOT_FOUND),
        (_text('Here it is: {"verdict": "supported"}'), GroundingVerdict.SUPPORTED),
        (_text("supported"), GroundingVerdict.SUPPORTED),
        (_text("Not found."), GroundingVerdict.NOT_FOUND),
        # Two answers, or prose, are no answer: never a guess.
        (_text('{"verdict": "supported"} {"verdict": "contradicted"}'), None),
        (_text("not supported"), None),
        (_text('{"verdict": "maybe"}'), None),
    ],
)
def test_the_statement_reply_is_read_and_repaired(
    completion: Completion, verdict: GroundingVerdict | None
) -> None:
    assert parse_verdict(completion, prompt="ground.statement@1") is verdict


def test_span_s_reading_is_unchanged_by_the_repair() -> None:
    # A fenced object is ground.statement's repair; ground.span keeps its own reading.
    assert parse_verdict(_text('```json\n{"verdict": "supported"}\n```')) is None


# --------------------------------------------------------------------------- #
# The grounder
# --------------------------------------------------------------------------- #


def test_the_grounder_sends_onyx_without_a_schema_and_reports_its_key() -> None:
    client = RecordedClient([{"match": "## Statement", "response": "yes"}])
    grounder = LLMGrounder(client=client, prompt="ground.onyx")
    grounded = grounder.ground(_fact(), DOC)
    assert grounded.verdict is GroundingVerdict.SUPPORTED
    (messages, _, schema) = client.calls[0]
    assert [m.role for m in messages] == ["user"] and schema is None
    assert grounder.stats["prompts"] == ["ground.onyx@1"]
    assert grounder.claim_line == "sentence"


def test_the_grounder_sends_the_statement_prompt_with_the_verdict_schema() -> None:
    client = RecordedClient(
        [{"match": "## Passage", "response": '```{"verdict": "contradicted"}```'}]
    )
    grounder = LLMGrounder(client=client, prompt="ground.statement@1")
    assert grounder.ground(_fact(), DOC).verdict is GroundingVerdict.CONTRADICTED
    assert client.calls[0][2] == GROUNDING_SCHEMA
    assert grounder.stats["prompts"] == ["ground.statement@1", "ground.statement.user@1"]


def test_an_unreadable_challenger_answer_leaves_the_fact_unchecked_and_is_counted() -> None:
    grounder = LLMGrounder(client=ScriptedClient(["I cannot tell."]), prompt="ground.onyx")
    assert grounder.ground(_fact(), DOC).verdict is GroundingVerdict.UNCHECKED
    assert grounder.stats["unparseable"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": "ground.onyx", "claim_format": "labelled"}, "plain sentence"),
        ({"prompt": "ground.paper"}, "binary prompt"),
        ({"prompt": "ground.span", "verdicts": "binary"}, "binary prompt"),
        ({"prompt": "extract"}, "prompt must be one of"),
        ({"prompt": "ground.nothing"}, "no prompt"),
    ],
)
def test_a_prompt_that_cannot_be_sent_that_way_is_refused(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValueError, LookupError), match=message):
        LLMGrounder(client=ScriptedClient([]), **kwargs)


def test_the_paper_prompt_still_comes_with_binary_verdicts() -> None:
    grounder = LLMGrounder(client=ScriptedClient([]), verdicts="binary", prompt="ground.paper@1")
    assert grounder.prompt.key == "ground.paper@1" and grounder.claim_line == "triple"


def test_each_prompt_keys_the_cache_apart() -> None:
    spec = LLMGrounder(client=ScriptedClient([])).spec
    keys = {
        cache_key(build_messages(_fact(), TEXT, prompt=p), spec=spec, schema=None)
        for p in ("ground.span@1", "ground.onyx@1", "ground.statement@1")
    }
    assert len(keys) == 3
