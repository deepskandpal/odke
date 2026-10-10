"""The entity-pair judge (#150): both orders, the swap rule, and a review queue.

Every answer comes from a recorded client, matched on which mention the
message names first, so the (A, B) and (B, A) calls answer apart under a
thread pool as they do in a loop.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import (
    Document,
    Entity,
    Evidence,
    Fact,
    LinkKind,
    Ontology,
    Pipeline,
    Resolution,
    Span,
)
from openodke.cli.main import app
from openodke.corroborate import (
    MemoryLookup,
    Mention,
    NativeResolver,
    PairJudge,
    name_key,
    name_similarity,
)
from openodke.corroborate.judge import (
    FRAME,
    MAX_CONTEXT,
    PROMPT,
    Answer,
    context_around,
    parse_answer,
    render_pair,
    swap_rule,
)
from openodke.eval.sheets import PairItem
from openodke.llm import Completion, MissingAPIKey, ModelSpec, RecordedClient
from openodke.prompts import get, read_lock
from openodke.run import build, execute, parse_config
from openodke.run.execute import prompts_sent
from openodke.validator import Validator

TEXT = (
    "Ada Lovelace wrote the first published program. "
    "She worked with Charles Babbage on the Analytical Engine. "
    "Lovelace died in London in 1852. "
    "Charles Dickens visited her that year."
)
DOC = Document(id="d1", text=TEXT)
ADA = Entity(key="p:ada", type="Person", label="Ada Lovelace")
LOVELACE = Entity(key="p:lovelace", type="Person", label="Lovelace")
BABBAGE = Entity(key="p:babbage", type="Person", label="Charles Babbage")
DICKENS = Entity(key="p:dickens", type="Person", label="Charles Dickens")
RECORDED = Resolution(method="linker", linker="odke.native")


def _fact(entity: Entity, quote: str) -> Fact:
    start = TEXT.index(quote)
    span = Span(doc_id="d1", start=start, end=start + len(quote), quote=quote)
    return Fact(
        subject=entity,
        predicate="mentioned",
        object_value=quote,
        evidence=(Evidence(doc_id="d1", span=span),),
    )


FACTS = [
    _fact(ADA, "Ada Lovelace wrote the first published program."),
    _fact(LOVELACE, "Lovelace died in London in 1852."),
]


def _answer(first: str, decision: str, because: str = "") -> dict[str, Any]:
    """A recorded answer for the call that names `first` as mention A."""
    reply = {"because": because or f"{decision} in context", "decision": decision}
    return {"match": f'Mention A: "{first}"', "response": reply}


def _client(forward: str, backward: str, *, a: str = "Ada Lovelace", b: str = "Lovelace") -> Any:
    return RecordedClient([_answer(a, forward), _answer(b, backward)])


def _judge(client: Any, **options: Any) -> PairJudge:
    judge = PairJudge(client=client, **options)
    judge.documents[DOC.id] = DOC
    return judge


def _resolve(
    judge: PairJudge, facts: Sequence[Fact] = FACTS, **options: Any
) -> tuple[list[Fact], list[Any], NativeResolver]:
    resolver = NativeResolver(judge=judge, **options)
    resolved, links = resolver.resolve(facts, {})
    return list(resolved), list(links), resolver


def _orders(client: RecordedClient) -> list[tuple[str, str]]:
    """Which mention each call put first and second, in call order."""
    out = []
    for messages, _, _ in client.calls:
        lines = messages[1].content.split("\n")
        out.append((lines[0].split('"')[1], lines[3].split('"')[1]))
    return out


# --------------------------------------------------------------------------- #
# The swap rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("forward", "backward", "decision"),
    [
        ("same", "same", "same"),
        ("different", "different", "different"),
        ("same", "different", "unsure"),
        ("different", "same", "unsure"),
        ("same", "unsure", "unsure"),
        ("unsure", "different", "unsure"),
        ("unsure", "unsure", "unsure"),
        (None, "same", "unsure"),
        (None, None, "unsure"),
    ],
)
def test_same_and_different_count_only_when_both_orders_say_so(
    forward: Any, backward: Any, decision: str
) -> None:
    assert swap_rule(Answer(decision=forward), Answer(decision=backward)) == decision


def test_same_in_both_orders_is_a_similar_link_with_the_judges_provenance_never_a_merge() -> None:
    assert name_similarity(name_key("Ada Lovelace"), name_key("Lovelace")) == 0.8
    client = _client("same", "same")
    facts, links, resolver = _resolve(_judge(client))

    (link,) = links
    assert (link.source_key, link.target_key, link.kind) == ("p:ada", "p:lovelace", "similar")
    # The resolver's own name score, as on any link it proposes.
    assert link.score == 0.8
    assert link.reason == (
        "pair judge (pair@1, anthropic/claude-haiku-4-5-20251001): same in both orders; "
        "same in context"
    )
    # A link, not a merge: no key moved, and the resolver stamped what it always does.
    assert [f.subject.key for f in facts] == ["p:ada", "p:lovelace"]
    assert all(f.subject.resolution == RECORDED for f in facts)
    assert _orders(client) == [("Ada Lovelace", "Lovelace"), ("Lovelace", "Ada Lovelace")]
    assert resolver.stats is not None
    judged = resolver.stats["judge"]
    assert {k: judged[k] for k in ("pairs", "asked", "calls", "swapped", "disagreed")} == {
        "pairs": 1,
        "asked": 1,
        "calls": 2,
        "swapped": 1,
        "disagreed": 0,
    }
    assert (judged["same"], judged["different"], judged["unsure"]) == (1, 0, 0)


def test_different_in_both_orders_is_a_different_link() -> None:
    _, links, _ = _resolve(_judge(_client("different", "different")))
    (link,) = links
    assert link.kind is LinkKind.DIFFERENT
    assert link.reason is not None and link.reason.startswith(
        "pair judge (pair@1, anthropic/claude-haiku-4-5-20251001): different in both orders"
    )


def test_orders_that_disagree_make_no_link_and_the_pair_is_queued_with_both_answers(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "review" / "pairs.jsonl"
    judge = _judge(_client("same", "different"), queue=queue)
    _, links, resolver = _resolve(judge)

    assert links == []
    (row,) = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    # A pair sheet's row as it stands: `odke label make pair` reads the queue.
    PairItem.model_validate(row)
    assert row["a"] == {
        "key": "p:ada",
        "type": "Person",
        "label": "Ada Lovelace",
        "context": "Ada Lovelace wrote the first published program. "
        "She worked with Charles Babbage on the Analytical Engine.",
    }
    assert row["b"]["context"] == TEXT[TEXT.index("She worked") :]
    assert row["judge"] == {
        "decision": "unsure",
        "by": "judge",
        "score": 0.8,
        "forward": {"decision": "same", "because": "same in context"},
        "backward": {"decision": "different", "because": "different in context"},
        "disagreed": True,
        "why": "the orders disagree",
        "prompt": "pair@1",
        "model": "anthropic/claude-haiku-4-5-20251001",
    }
    assert resolver.stats is not None
    judged = resolver.stats["judge"]
    assert (judged["disagreed"], judged["unsure"], judged["queued"]) == (1, 1, 1)
    (decision,) = judge.decisions
    assert decision.disagreed and decision.decision == "unsure"


def test_unsure_in_both_orders_is_no_link_and_without_a_queue_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    judge = _judge(_client("unsure", "unsure"))
    _, links, resolver = _resolve(judge)
    assert links == []
    assert list(tmp_path.iterdir()) == []
    assert resolver.stats is not None
    assert (resolver.stats["judge"]["unsure"], resolver.stats["judge"]["queued"]) == (1, 0)
    assert judge.decisions[0].why == "unsure in both orders"
    assert not judge.decisions[0].disagreed


def test_a_pair_is_queued_once_and_the_queue_is_only_appended_to(tmp_path: Path) -> None:
    queue = tmp_path / "pairs.jsonl"
    queue.write_text('{"a": {"key": "x", "type": "T"}, "b": {"key": "y", "type": "T"}}\n')
    for _ in range(2):
        _resolve(_judge(_client("same", "unsure"), queue=queue))
    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert [(r["a"]["key"], r["b"]["key"]) for r in rows] == [("x", "y"), ("p:ada", "p:lovelace")]


# --------------------------------------------------------------------------- #
# Where it runs
# --------------------------------------------------------------------------- #


def test_the_judge_is_asked_only_about_pairs_in_the_band_the_rules_left_open() -> None:
    def ids(entity: Entity, external_id: str) -> Entity:
        # Another type, so the copies are compared with each other alone.
        update = {"external_id": external_id, "key": entity.key + "-id", "type": "Writer"}
        return entity.model_copy(update=update)

    halden = Entity(key="c:halden", type="Company", label="Halden Robotics")
    robotic = Entity(key="c:robotic", type="Company", label="Halden Robotic")
    others = [
        BABBAGE,
        DICKENS,  # 0.6: below the band, never asked
        halden,
        robotic,  # 0.97: the rules' SIMILAR, never asked
        ids(ADA, "wikidata:Q7259"),
        ids(LOVELACE, "wikidata:Q1"),  # 0.8, but the ids disagree: never asked
        ids(BABBAGE, "wikidata:Q46633"),
        ids(DICKENS, "WIKIDATA:q46633"),  # a shared id is proof: never asked
    ]
    assert name_similarity(name_key("Charles Babbage"), name_key("Charles Dickens")) == 0.6
    # ADA and LOVELACE, 0.8, are in the band: the one pair asked.
    facts = [*FACTS, *(Fact(subject=e, predicate="named", object_value=e.label) for e in others)]
    client = RecordedClient([_answer("Ada Lovelace", "same"), _answer("Lovelace", "same")])
    _, links, resolver = _resolve(_judge(client), facts)

    assert _orders(client) == [("Ada Lovelace", "Lovelace"), ("Lovelace", "Ada Lovelace")]
    kinds = sorted((link.source_key, link.target_key, link.kind.value) for link in links)
    assert kinds == [
        ("c:halden", "c:robotic", "similar"),
        ("p:ada", "p:lovelace", "similar"),
        ("p:babbage-id", "p:dickens-id", "same_as"),
    ]
    assert resolver.stats is not None
    assert (resolver.stats["judge"]["pairs"], resolver.stats["judge"]["calls"]) == (1, 2)


def test_the_band_starts_at_low_and_stops_below_the_threshold() -> None:
    client = _client("same", "same")
    # 0.8 is below a band from 0.85.
    _, links, _ = _resolve(_judge(client, low=0.85))
    assert links == [] and client.calls == []
    # At a threshold of 0.8 the rules link the pair themselves.
    _, links, _ = _resolve(_judge(client, low=0.7), threshold=0.8)
    assert [link.reason for link in links] == [None] and client.calls == []
    with pytest.raises(ValueError, match="band starts at 0.95, above the threshold 0.9"):
        NativeResolver(judge=PairJudge(client=client, low=0.95))


def test_without_a_judge_nothing_below_the_threshold_is_asked_or_linked() -> None:
    resolver = NativeResolver()
    _, links = resolver.resolve(FACTS, {})
    assert links == [] and resolver.stats is None and resolver.documents is None


# --------------------------------------------------------------------------- #
# What the model reads
# --------------------------------------------------------------------------- #


def test_each_order_sends_the_registered_prompt_and_frame_filled_in() -> None:
    client = _client("same", "same")
    _resolve(_judge(client))
    (system, forward), (_, backward) = (call[0] for call in client.calls)
    assert system.role == "system" and system.content == PROMPT.text
    assert forward.content == (
        'Mention A: "Ada Lovelace" (type: Person)\n'
        "Context A: Ada Lovelace wrote the first published program. She worked with Charles "
        "Babbage on the Analytical Engine.\n"
        "\n"
        'Mention B: "Lovelace" (type: Person)\n'
        "Context B: She worked with Charles Babbage on the Analytical Engine. Lovelace died in "
        "London in 1852. Charles Dickens visited her that year."
    )
    assert backward.content.startswith('Mention A: "Lovelace" (type: Person)\nContext A: She')
    # Structured output, asked of the ground role's small model.
    assert client.calls[0][2] is not None and client.calls[0][2]["required"] == [
        "because",
        "decision",
    ]
    assert client.calls[0][1].max_tokens == 256


def test_a_field_is_filled_once_so_braces_in_a_context_are_kept() -> None:
    a = Mention(key="a", type="Thing", label="{b_surface}", context="Says {a_type} and {x}.")
    b = Mention(key="b", type="Thing", label="B", context="B is here.")
    text = render_pair(a, b)
    assert 'Mention A: "{b_surface}" (type: Thing)' in text
    assert "Context A: Says {a_type} and {x}." in text


def test_a_stored_entity_shows_its_aliases_and_its_store_context() -> None:
    stored = Entity(
        key="p:ada-king",
        type="Person",
        label="Ada King",
        aliases=("Ada Byron Lovelace", "Countess of Lovelace"),
    )
    assert name_similarity(name_key("Ada Lovelace"), name_key("Ada Byron Lovelace")) == 0.8
    client = RecordedClient([_answer("Ada Lovelace", "same"), _answer("Ada King", "same")])
    judge = _judge(client, store_context=lambda e: "The countess  wrote\nthe notes.")
    resolver = NativeResolver(judge=judge, lookup=MemoryLookup({stored.key: stored}))
    _, links = resolver.resolve(FACTS[:1], {})

    (link,) = links
    assert (link.source_key, link.target_key, link.kind) == ("p:ada", "p:ada-king", "similar")
    user = client.calls[0][0][1].content
    assert 'Mention B: "Ada King" (type: Person); also known as: Ada Byron Lovelace, ' in user
    assert "Countess of Lovelace\nContext B: The countess wrote the notes." in user
    assert resolver.stats is not None and resolver.stats["store"]["similar"] == 1


def test_a_side_with_no_context_is_not_asked_and_is_queued_for_a_person(tmp_path: Path) -> None:
    stored = Entity(key="p:lady", type="Person", label="Lady Lovelace")
    client = RecordedClient([])
    judge = _judge(client, queue=tmp_path / "q.jsonl")
    resolver = NativeResolver(judge=judge, lookup=MemoryLookup({stored.key: stored}))
    _, links = resolver.resolve(FACTS[1:], {})
    assert links == [] and client.calls == []
    (row,) = [json.loads(line) for line in (tmp_path / "q.jsonl").read_text().splitlines()]
    assert row["b"] == {"key": "p:lady", "type": "Person", "label": "Lady Lovelace"}
    assert row["judge"]["why"] == "no context"
    assert resolver.stats is not None and resolver.stats["judge"]["no_context"] == 1


def test_the_context_is_the_sentence_and_one_either_side_cut_when_too_long() -> None:
    text = "One. Two has Ada. Three. Four."
    at = text.index("Ada")
    assert context_around(text, at, at + 3) == "One. Two has Ada. Three."
    long = "Start. " + "word " * 400 + "Ada " + "word " * 400 + "end. Last."
    at = long.index("Ada")
    cut = context_around(long, at, at + 3)
    assert len(cut) <= MAX_CONTEXT and "Ada" in cut
    assert not cut.startswith("ord") and not cut.endswith("wor")


def test_a_cited_quote_stands_in_when_the_document_is_not_given() -> None:
    judge = PairJudge(client=RecordedClient([]))
    mentions = judge.mentions([ADA], FACTS[:1])
    assert mentions["p:ada"].context == "Ada Lovelace wrote the first published program."


# --------------------------------------------------------------------------- #
# Answers that are not answers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "parsed", "decision"),
    [
        ("", {"because": "one person", "decision": "same"}, "same"),
        ('{"because": "two firms", "decision": "Different"}', None, "different"),
        ('```json\n{"because": "x", "decision": "unsure"}\n```', None, "unsure"),
        ("same", None, "same"),
        ("They are not the same.", None, None),
        ('{"decision": "maybe"}', None, None),
    ],
)
def test_an_answer_is_read_from_json_or_a_bare_word_never_from_prose(
    text: str, parsed: dict[str, Any] | None, decision: str | None
) -> None:
    assert parse_answer(Completion(text=text, parsed=parsed)).decision == decision


def test_an_unreadable_answer_is_unsure_and_counted() -> None:
    client = RecordedClient(
        [
            {"match": 'Mention A: "Ada Lovelace"', "response": "Probably, I think."},
            _answer("Lovelace", "same"),
        ]
    )
    judge = _judge(client)
    _, links, resolver = _resolve(judge)
    assert links == []
    assert resolver.stats is not None and resolver.stats["judge"]["unparseable"] == 1
    assert judge.decisions[0].why == "an answer could not be read"


def test_a_failed_call_is_no_link_and_is_not_queued(tmp_path: Path) -> None:
    client = RecordedClient(
        [
            {"match": 'Mention A: "Ada Lovelace"', "error": "bad request"},
            _answer("Lovelace", "same"),
        ]
    )
    judge = _judge(client, queue=tmp_path / "q.jsonl")
    _, links, resolver = _resolve(judge)
    assert links == [] and not (tmp_path / "q.jsonl").exists()
    assert judge.decisions[0].decision == "failed"
    assert resolver.stats is not None and resolver.stats["judge"]["failed"] == 1


def test_a_budget_stop_leaves_the_pair_unasked_and_the_rules_still_run(tmp_path: Path) -> None:
    from openodke.llm.budget import Budget, Ledger

    # Room for one of the two orders: the other is refused, never made.
    client = Ledger(Budget(calls=1)).client(_client("same", "same"))
    judge = _judge(client, queue=tmp_path / "q.jsonl")
    facts, links, resolver = _resolve(judge)

    assert links == [] and not (tmp_path / "q.jsonl").exists()
    assert [f.subject.key for f in facts] == ["p:ada", "p:lovelace"]
    (decision,) = judge.decisions
    assert (decision.decision, decision.why) == ("unasked", "the budget stopped the run first")
    assert resolver.stats is not None
    judged = resolver.stats["judge"]
    assert (judged["calls"], judged["unasked"]) == (1, 1)
    assert judged["stopped"]["limit"] == "calls"


def test_calls_a_response_cache_answered_are_counted() -> None:
    from openodke.llm.cache import CachedClient

    cached = CachedClient(_client("same", "same"))
    for _ in range(2):
        _, links, resolver = _resolve(_judge(cached))
        assert [link.kind for link in links] == [LinkKind.SIMILAR]
    assert resolver.stats is not None
    assert (resolver.stats["judge"]["calls"], resolver.stats["judge"]["cached"]) == (2, 2)


def test_a_missing_key_fails_the_run_rather_than_every_pair() -> None:
    class NoKey:
        def complete(self, messages: Any, *, spec: ModelSpec, schema: Any = None) -> Completion:
            raise MissingAPIKey("ANTHROPIC_API_KEY is not set")

    with pytest.raises(MissingAPIKey):
        _resolve(_judge(NoKey()))


# --------------------------------------------------------------------------- #
# The prompts
# --------------------------------------------------------------------------- #


def test_the_prompt_and_its_frame_are_registered_locked_and_reported() -> None:
    assert get("pair@1") is PROMPT and get("pair.user@1") is FRAME
    lock = read_lock()
    assert lock["pair@1"] == PROMPT.sha256 and lock["pair.user@1"] == FRAME.sha256
    _, _, resolver = _resolve(_judge(_client("same", "same")))
    assert resolver.stats is not None
    assert resolver.stats["prompts"] == ["pair@1", "pair.user@1"]
    assert prompts_sent({"resolver": resolver}) == ["pair@1", "pair.user@1"]


def test_no_call_names_no_prompt() -> None:
    _, _, resolver = _resolve(_judge(RecordedClient([]), low=0.85))
    assert resolver.stats is not None and resolver.stats["prompts"] == []


# --------------------------------------------------------------------------- #
# The review queue, through `odke label`
# --------------------------------------------------------------------------- #

runner = CliRunner()


def _tick(sheet: Path, item_id: str, box: str) -> None:
    head, tail = sheet.read_text(encoding="utf-8").split(f"### {item_id}\n", 1)
    sheet.write_text(f"{head}### {item_id}\n{tail.replace(f'- [ ] {box}', f'- [x] {box}', 1)}")


def test_the_queue_round_trips_through_odke_label_into_a_persons_links(tmp_path: Path) -> None:
    queue = tmp_path / "pairs.jsonl"
    babbage = _fact(BABBAGE, "She worked with Charles Babbage on the Analytical Engine.")
    babbage_too = _fact(
        Entity(key="p:charles", type="Person", label="Charles"),
        "Charles Dickens visited her that year.",
    )
    client = RecordedClient(
        [
            _answer("Ada Lovelace", "same"),
            _answer("Lovelace", "unsure"),
            _answer("Charles Babbage", "unsure", "which Charles"),
            _answer("Charles", "unsure"),
        ]
    )
    assert name_similarity(name_key("Charles Babbage"), name_key("Charles")) >= 0.6
    _resolve(_judge(client, queue=queue, low=0.6), [*FACTS, babbage, babbage_too])
    assert len(queue.read_text(encoding="utf-8").splitlines()) == 2

    sheets = tmp_path / "sheets"
    made = runner.invoke(app, ["label", "make", "pair", str(queue), "-o", str(sheets)])
    assert made.exit_code == 0, made.output
    sheet = sheets / "sheet-01.md"
    shown = sheet.read_text(encoding="utf-8")
    # The person decides unanchored: nothing the judge said is on the sheet.
    assert "because" not in shown and "disagree" not in shown and "unsure in" not in shown
    _tick(sheet, "P-0001", "same")
    _tick(sheet, "P-0002", "different")
    labels = tmp_path / "reviewed.jsonl"
    read = runner.invoke(app, ["label", "read", str(sheets), "-o", str(labels)])
    assert read.exit_code == 0, read.output

    # The next run: a person's decision in the judge's place, and no call at all.
    silent = RecordedClient([])
    judge = _judge(silent, queue=queue, reviewed=labels, low=0.6)
    _, links, resolver = _resolve(judge, [*FACTS, babbage, babbage_too])
    assert silent.calls == []
    assert sorted((link.source_key, link.target_key, link.kind, link.reason) for link in links) == [
        ("p:ada", "p:lovelace", LinkKind.SIMILAR, "person: same, from the review queue"),
        ("p:babbage", "p:charles", LinkKind.DIFFERENT, "person: different, from the review queue"),
    ]
    assert resolver.stats is not None and resolver.stats["judge"]["person"] == 2
    assert len(queue.read_text(encoding="utf-8").splitlines()) == 2


def test_reviewed_decisions_can_be_given_as_rows() -> None:
    from openodke.eval import PairLabel

    judge = _judge(RecordedClient([]), reviewed=[PairLabel(a="p:lovelace", b="p:ada", same=False)])
    _, links, _ = _resolve(judge)
    assert [link.kind for link in links] == [LinkKind.DIFFERENT]
    with pytest.raises(ValueError, match="a reviewed pair needs a, b and same"):
        PairJudge(reviewed=[{"a": "x", "b": "y"}])


def test_a_sheet_shows_a_stored_entitys_aliases() -> None:
    from openodke.eval.sheets import PAIR

    item = PairItem(
        a=Mention(key="a", type="Person", label="Lovelace", context="Lovelace died."),
        b=Mention(key="b", type="Person", label="Ada King", aliases=("Ada Lovelace",)),
    )
    assert "B: Ada King (Person), also known as Ada Lovelace" in PAIR.render(item)


# --------------------------------------------------------------------------- #
# The Validator, the pipeline and `odke run`
# --------------------------------------------------------------------------- #

PEOPLE = Ontology.from_dict(
    {
        "name": "people",
        "types": {"Person": {}},
        "predicates": {"death_year": {"domain": ["Person"], "range": "integer"}},
    }
)
ROWS: list[dict[str, Any]] = [
    {"doc": "d1", "subject": "Ada Lovelace", "subject_type": "Person", "predicate": "death_year",
     "object": 1852, "quote": "Ada Lovelace wrote the first published program."},
    {"doc": "d1", "subject": "Lovelace", "subject_type": "Person", "predicate": "death_year",
     "object": 1852, "quote": "Lovelace died in London in 1852."},
]  # fmt: skip
GROUNDED = [
    {"match": "Claim: ", "response": {"verdict": "supported"}},
]


def test_the_validator_asks_the_judge_and_reports_its_counts_calls_and_prompts() -> None:
    judge = PairJudge(client=_client("same", "different"))
    validator = Validator(PEOPLE, client=RecordedClient(GROUNDED), judge=judge)
    kg, report = validator.validate(ROWS, [DOC])

    assert kg.links == ()
    assert report.judge == {
        "pairs": 1,
        "asked": 1,
        "calls": 2,
        "swapped": 1,
        "disagreed": 1,
        "same": 0,
        "different": 0,
        "unsure": 1,
        "person": 0,
        "queued": 0,
        "no_context": 0,
        "failed": 0,
        "unasked": 0,
    }
    # Two grounding calls and the judge's two.
    assert report.calls == 4
    assert report.prompts == ("ground.span@1", "pair@1", "pair.user@1")
    assert (
        "judge         1 pair in the band: 1 asked in both orders (2 calls, 1 swapped), 0 same, "
        "0 different, 1 unsure; orders disagreed on 1"
    ) in report.render()
    # A second job reports itself alone.
    _, again = validator.validate(ROWS, [DOC])
    assert again.judge is not None and again.judge["calls"] == 2 and again.calls == 4


def test_a_budget_the_judge_reaches_first_stops_the_job_during_resolve() -> None:
    from openodke.llm.budget import Budget, Ledger

    judge = PairJudge(client=Ledger(Budget(calls=0)).client(_client("same", "same")))
    validator = Validator(PEOPLE, client=RecordedClient(GROUNDED), judge=judge)
    kg, report = validator.validate(ROWS, [DOC])
    assert kg.links == ()
    assert report.stopped is not None
    assert (report.stopped["stage"], report.stopped["limit"]) == ("resolve", "calls")
    assert report.judge is not None and (report.judge["calls"], report.judge["unasked"]) == (0, 2)
    assert "during resolve" in report.render()
    assert "2 calls the budget refused" in report.render()


def test_a_dry_run_never_asks_the_judge() -> None:
    client = RecordedClient([])
    _, report = Validator(PEOPLE, judge=PairJudge(client=client)).validate(
        ROWS, [DOC], dry_run=True
    )
    assert client.calls == [] and report.judge is None and report.calls == 0


def test_the_validator_takes_a_judge_only_for_its_own_resolver() -> None:
    with pytest.raises(ValueError, match=r"NativeResolver\(judge=...\)"):
        Validator(resolver=NativeResolver(), judge=PairJudge(client=RecordedClient([])))


def test_a_pipeline_hands_the_judge_its_texts_through_the_resolver() -> None:
    class Replay:
        def extract(self, chunk: Any, ontology: Any) -> list[Fact]:
            return list(FACTS)

    judge = PairJudge(client=_client("same", "same"))
    resolver = NativeResolver(judge=judge)
    assert resolver.documents is judge.documents
    from openodke.run.execute import register_documents

    register_documents(resolver, [DOC])
    kg = Pipeline(Ontology(name="x"), Replay(), resolver=resolver).run([DOC])
    assert [link.kind for link in kg.links] == [LinkKind.SIMILAR]


def _run_config(tmp_path: Path, resolver: Any, **extra: Any) -> dict[str, Any]:
    (tmp_path / "ontology.json").write_text(PEOPLE.model_dump_json(), encoding="utf-8")
    (tmp_path / "corpus").mkdir(exist_ok=True)
    (tmp_path / "corpus" / "d1.txt").write_text(TEXT, encoding="utf-8")
    rows = [{**row, "doc": "corpus/d1.txt"} for row in ROWS]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    replay = [*GROUNDED, _answer("Ada Lovelace", "same"), _answer("Lovelace", "unsure")]
    (tmp_path / "ground.json").write_text(json.dumps(replay), encoding="utf-8")
    return {
        "ontology": "ontology.json",
        "inputs": ["corpus"],
        "models": {"replay": {"ground": "ground.json"}, "meter": True},
        "stages": {
            "extractor": {"use": "triples", "path": "rows.jsonl"},
            "grounder": "llm",
            "resolver": resolver,
        },
        **extra,
    }


def test_a_run_config_turns_the_judge_on_with_paths_relative_to_it(tmp_path: Path) -> None:
    config = _run_config(
        tmp_path, {"use": "native", "judge": {"low": 0.75, "queue": "review/pairs.jsonl"}}
    )
    built = build(parse_config(config, base_dir=tmp_path))
    judge = built.stages["resolver"].judge
    assert isinstance(judge, PairJudge)
    assert (judge.low, judge.queue) == (0.75, tmp_path / "review" / "pairs.jsonl")

    result = execute(parse_config(config, base_dir=tmp_path))
    resolver = result.stats["stages"]["resolver"]
    assert (resolver["judge"]["calls"], resolver["judge"]["queued"]) == (2, 1)
    assert resolver["prompts"] == ["pair@1", "pair.user@1"]
    # The meter counts the judge's calls as a stage of their own.
    assert result.stats["cost"]["stages"]["judge"]["calls"] == 2
    assert (tmp_path / "review" / "pairs.jsonl").is_file()


def test_a_run_whose_budget_runs_out_in_the_judge_says_it_stopped_during_resolve(
    tmp_path: Path,
) -> None:
    config = _run_config(tmp_path, {"use": "native", "judge": True})
    # Two grounding calls fit, and one of the judge's two.
    config["models"]["budget"] = {"calls": 3}
    result = execute(parse_config(config, base_dir=tmp_path))
    stopped = result.stats["stopped"]
    assert (stopped["stage"], stopped["limit"], stopped["unchecked"]) == ("resolve", "calls", 0)
    assert result.stats["stages"]["resolver"]["judge"]["unasked"] == 1
    assert "stopped       at budget, calls 3 of 3, during resolve" in result.render()


def test_the_judge_is_off_unless_named(tmp_path: Path) -> None:
    for resolver in ("native", {"use": "native", "judge": False}):
        built = build(parse_config(_run_config(tmp_path, resolver), base_dir=tmp_path))
        assert built.stages["resolver"].judge is None


def test_a_dry_run_writes_no_queue(tmp_path: Path) -> None:
    config = _run_config(tmp_path, {"use": "native", "judge": {"queue": "pairs.jsonl"}})
    result = execute(parse_config(config, base_dir=tmp_path), dry_run=True)
    assert "dry run: the pair judge's review queue is not written" in result.warnings
    assert not (tmp_path / "pairs.jsonl").exists()


@pytest.mark.parametrize(
    ("judge", "message"),
    [
        ("yes", "stages.resolver.judge: true, or the judge's options"),
        ({"lo": 0.7}, "stages.resolver.judge: "),
        ({"reviewed": "missing.jsonl"}, "stages.resolver.judge.reviewed: "),
        ({"store_context": "x"}, "stages.resolver.judge.store_context: set by odke run"),
    ],
)
def test_a_bad_judge_option_is_a_config_error(tmp_path: Path, judge: Any, message: str) -> None:
    from openodke.run import ConfigError

    config = _run_config(tmp_path, {"use": "native", "judge": judge})
    with pytest.raises(ConfigError, match=message):
        build(parse_config(config, base_dir=tmp_path))
