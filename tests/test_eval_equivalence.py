"""The fact-equivalence judge (#143): its pre-filter, its swap rule, and what it reads.

No model is called: a `RecordedClient` answers by which fact a call shows first,
so each order's answer is set apart and the swap rule can be checked.
"""

from __future__ import annotations

import pytest

from openodke import Document, Entity, Evidence, Fact, Ontology
from openodke.eval.equivalence import (
    FRAME,
    MAX_PASSAGE,
    PROMPT,
    FactJudge,
    candidate,
    passage_for,
    render_plain,
    render_question,
    surface_pair,
)
from openodke.ground.llm import render_claim
from openodke.ground.retry import RetryPolicy
from openodke.llm import MissingAPIKey, ModelSpec, RecordedClient
from openodke.llm.roles import ModelRoles
from openodke.prompts import get
from openodke.types import Polarity, Span, SpanOrigin

LOGAN = Entity(key="person:gabby_logan", type="Person", label="Gabby Logan")
GABRIELLE = Entity(key="person:gabrielle_logan", type="Person", label="Gabrielle Nicole Logan")
LEEDS = Entity(key="place:leeds", type="Place", label="Leeds")
WALES = Entity(key="place:wales", type="Place", label="Wales")

TEXT = (
    "Gabrielle Nicole Logan is a British television presenter. "
    "She was born in Leeds on 25 April 1973. "
    "She competed as a rhythmic gymnast for Wales."
)
DOC = Document(id="d1", text=TEXT)
BORN = TEXT.index("She was born")
BORN_END = TEXT.index(" She competed")

GOLD = Fact(
    subject=GABRIELLE,
    predicate="place_of_birth",
    object_entity=LEEDS,
    evidence=(Evidence(doc_id="d1", span=Span(doc_id="d1", start=BORN, end=BORN_END)),),
)
SAID = Fact(subject=LOGAN, predicate="place_of_birth", object_entity=LEEDS)

ONTOLOGY = Ontology.from_dict(
    {
        "name": "people",
        "types": {"Person": {}, "Place": {}},
        "predicates": {
            "place_of_birth": {
                "description": "where a person was born (P19)",
                "domain": ["Person"],
                "range": "Place",
                "aliases": ["place of birth"],
            },
            "date_of_birth": {"domain": ["Person"], "range": "date"},
        },
    }
)


def answer(decision: str) -> dict[str, str]:
    return {"decision": decision}


def orders(
    gold_first: str, prediction_first: str, gold: Fact = GOLD, said: Fact = SAID
) -> RecordedClient:
    """A client that answers by which fact the call shows as Fact 1."""
    return RecordedClient(
        [
            {"match": f"Fact 1: {render_claim(gold)}", "response": answer(gold_first)},
            {"match": f"Fact 1: {render_claim(said)}", "response": answer(prediction_first)},
        ]
    )


def judge(client: RecordedClient, retry: RetryPolicy | None = None) -> FactJudge:
    return FactJudge(
        client=client, ontology=ONTOLOGY, max_workers=1, retry=retry, sleep=lambda _: None
    )


# --------------------------------------------------------------------------- #
# The pre-filter
# --------------------------------------------------------------------------- #


def test_a_pair_with_the_relation_and_one_end_passes_and_says_which_end_differs() -> None:
    assert candidate(GOLD, SAID) == "subject"
    other = Fact(subject=GABRIELLE, predicate="place_of_birth", object_entity=WALES)
    assert candidate(GOLD, other) == "object"


@pytest.mark.parametrize(
    ("said", "why"),
    [
        (Fact(subject=GABRIELLE, predicate="place_of_birth", object_entity=LEEDS), "identical"),
        (Fact(subject=LOGAN, predicate="place_of_birth", object_entity=WALES), "both ends differ"),
        (Fact(subject=GABRIELLE, predicate="residence", object_entity=WALES), "another relation"),
        (
            Fact(
                subject=LOGAN,
                predicate="place_of_birth",
                object_entity=LEEDS,
                polarity=Polarity.DENIED,
            ),
            "a denial is the opposite claim",
        ),
    ],
)
def test_a_pair_that_is_not_a_question_of_wording_never_reaches_the_judge(
    said: Fact, why: str
) -> None:
    assert candidate(GOLD, said) is None, why


def test_an_end_matches_after_normalisation_against_every_name_the_gold_end_goes_by() -> None:
    aka = Entity(key="person:gnl", type="Person", label="G. N. Logan", aliases=("Gabby Logan",))
    gold = GOLD.model_copy(update={"subject": aka})
    # "gabby logan" is an alias's key, so the subject is the one that matches.
    elsewhere = Fact(subject=LOGAN, predicate="place_of_birth", object_entity=WALES)
    assert candidate(gold, elsewhere) == "object"
    # Case and punctuation are the normaliser's, not the judge's.
    shouted = Entity(key="x", type="Place", label="LEEDS.")
    assert candidate(GOLD, SAID.model_copy(update={"object_entity": shouted})) == "subject"


def test_two_values_compare_as_values() -> None:
    gold = Fact(subject=GABRIELLE, predicate="date_of_birth", object_value="25 April 1973")
    said = Fact(subject=GABRIELLE, predicate="date_of_birth", object_value="1973-04-25")
    assert candidate(gold, said) == "object"
    numbers = Fact(subject=LOGAN, predicate="date_of_birth", object_value=1973)
    assert candidate(gold.model_copy(update={"object_value": "1973"}), numbers) == "subject"


def test_the_string_form_reads_a_cluster_of_names_and_a_relation_spelled_two_ways() -> None:
    cluster = ("Gabrielle Nicole Logan", "Gabby")
    gold = (cluster, "place of birth", "Leeds")
    assert surface_pair(gold, ("Gabby", "place_of_birth", "Leeds, England")) == "object"
    assert surface_pair(gold, ("G. Logan", "place_of_birth", "Leeds")) == "subject"
    assert surface_pair(gold, ("Gabby", "place of birth", "Leeds")) is None
    assert surface_pair(gold, ("Gabby", "country", "England")) is None


# --------------------------------------------------------------------------- #
# What the judge reads
# --------------------------------------------------------------------------- #


def test_the_question_is_the_registered_frame_filled_once() -> None:
    braces = "A passage with {fact_1} in it."
    text = render_question("place_of_birth", "where a person was born", "A.", "B.", braces)
    assert text == (
        "Relation: place of birth: where a person was born\n\nFact 1: A.\nFact 2: B.\n\n"
        "Passage:\nA passage with {fact_1} in it."
    )
    assert render_question("x", None, "A.", "B.", "").startswith("Relation: x: no description\n")
    assert PROMPT.key == "fact_equiv@1" and FRAME.key == "fact_equiv.user@1"
    assert get("fact_equiv", 1) is PROMPT and get("fact_equiv.user", 1) is FRAME


def test_the_passage_is_the_gold_fact_s_evidence_sentences() -> None:
    assert passage_for(GOLD, DOC) == "She was born in Leeds on 25 April 1973."
    # Two sentences apart: both, with the gap marked.
    first = Span(doc_id="d1", start=0, end=10)
    last = Span(doc_id="d1", start=TEXT.index("She competed"), end=len(TEXT))
    both = GOLD.model_copy(
        update={"evidence": (Evidence(doc_id="d1", span=first), Evidence(doc_id="d1", span=last))}
    )
    assert passage_for(both, DOC) == (
        "Gabrielle Nicole Logan is a British television presenter. … "
        "She competed as a rhythmic gymnast for Wales."
    )


def test_a_fact_that_cites_nothing_is_given_the_sentences_naming_both_ends() -> None:
    uncited = Fact(subject=GABRIELLE, predicate="place_of_birth", object_entity=LEEDS)
    assert passage_for(uncited, DOC) == (
        "Gabrielle Nicole Logan is a British television presenter. "
        "She was born in Leeds on 25 April 1973."
    )
    # A context span is the whole text, not a citation, so it is not the evidence.
    whole = Span(doc_id="d1", start=0, end=len(TEXT))
    context = uncited.model_copy(
        update={"evidence": (Evidence(doc_id="d1", span=whole, span_origin=SpanOrigin.CONTEXT),)}
    )
    assert passage_for(context, DOC) == passage_for(uncited, DOC)


def test_with_no_window_naming_both_ends_the_passage_is_the_start_of_the_document() -> None:
    nowhere = Fact(
        subject=Entity(key="k", type="Person", label="Nobody"), predicate="p", object_value="x"
    )
    long = Document(id="d2", text="word " * 1000)
    shown = passage_for(nowhere, long)
    assert shown.endswith(" …") and len(shown) <= MAX_PASSAGE + 2


# --------------------------------------------------------------------------- #
# The swap rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("gold_first", "prediction_first", "decision", "why"),
    [
        ("same", "same", "same", None),
        ("different", "different", "different", None),
        ("same", "different", "unsure", "the orders disagree"),
        ("unsure", "unsure", "unsure", "unsure in both orders"),
        ("same", "maybe", "unsure", "an answer could not be read"),
    ],
)
def test_same_counts_only_when_both_orders_say_so(
    gold_first: str, prediction_first: str, decision: str, why: str | None
) -> None:
    client = orders(gold_first, prediction_first)
    decided = judge(client).judge(GOLD, SAID, "She was born in Leeds.")
    assert decided.decision == decision and decided.why == why
    assert decided.gold_first is not None and decided.prediction_first is not None
    assert decided.disagreed is (gold_first != prediction_first)
    assert decided.prompt == "fact_equiv@1"
    # Gold first, then the prediction first: the same two facts, swapped.
    shown = [call[0][1].content for call in client.calls]
    assert [s.split("\n")[2:4] for s in shown] == [
        [f"Fact 1: {render_claim(GOLD)}", f"Fact 2: {render_claim(SAID)}"],
        [f"Fact 1: {render_claim(SAID)}", f"Fact 2: {render_claim(GOLD)}"],
    ]
    assert all(call[0][0].content == PROMPT.text for call in client.calls)


def test_the_relation_s_description_comes_from_the_ontology() -> None:
    client = orders("same", "same")
    judge(client).judge(GOLD, SAID, "She was born in Leeds.")
    user = client.calls[0][0][1].content
    assert user.startswith("Relation: place of birth: where a person was born (P19)\n")
    assert user.endswith("Passage:\nShe was born in Leeds.")


def test_stats_count_both_orders_and_name_the_prompts() -> None:
    fact_judge = judge(orders("same", "different"))
    fact_judge.judge_many([(GOLD, SAID, "p"), (GOLD, SAID, "p")])
    stats = fact_judge.stats
    assert (stats["pairs"], stats["calls"], stats["swapped"], stats["disagreed"]) == (2, 4, 2, 2)
    assert (stats["same"], stats["different"], stats["unsure"]) == (0, 0, 2)
    assert stats["prompts"] == ["fact_equiv@1", "fact_equiv.user@1"]
    assert FactJudge(client=RecordedClient([])).stats["prompts"] == []


def test_a_failed_call_fails_the_pair_and_a_budget_stop_leaves_it_unasked() -> None:
    broken = RecordedClient([{"match": "Fact 1", "error": "boom"}])
    no_retry = RetryPolicy(attempts=1)
    assert judge(broken, retry=no_retry).judge(GOLD, SAID, "p").decision == "failed"

    from openodke.llm.budget import Budget, Ledger

    stopped = FactJudge(
        client=Ledger(Budget(calls=0)).client(orders("same", "same")), max_workers=1
    )
    decided = stopped.judge(GOLD, SAID, "p")
    assert (decided.decision, decided.why) == ("unasked", "the budget stopped the run first")
    assert stopped.stats["unasked"] == 2 and stopped.stats["calls"] == 0
    assert "stopped" in stopped.stats

    class Keyless:
        def complete(self, *args: object, **kwargs: object) -> None:
            raise MissingAPIKey("no key")

    with pytest.raises(MissingAPIKey):
        FactJudge(client=Keyless(), max_workers=1).judge(GOLD, SAID, "p")  # type: ignore[arg-type]


def test_the_diagnosis_hook_answers_true_false_or_none() -> None:
    gold = ("Gabrielle Nicole Logan", "place of birth", "Leeds")
    said = ("Gabby Logan", "place of birth", "Leeds")

    def hook(first: str, second: str) -> bool | None:
        client = RecordedClient(
            [
                {"match": f"Fact 1: {render_plain(gold)}", "response": answer(first)},
                {"match": f"Fact 1: {render_plain(said)}", "response": answer(second)},
            ]
        )
        return judge(client).equivalent(gold, said, "She was born in Leeds.")

    assert hook("same", "same") is True
    assert hook("different", "different") is False
    assert hook("same", "unsure") is None
    # No passage, no question: the prompt may not decide from the facts alone.
    silent = RecordedClient([])
    assert judge(silent).equivalent(gold, said, None) is None and not silent.calls
    assert render_plain(gold) == "Gabrielle Nicole Logan — place of birth — Leeds."


def test_the_judge_asks_the_ground_role_with_the_answer_schema() -> None:
    client = orders("same", "same")
    judge(client).judge(GOLD, SAID, "p")
    _, spec, schema = client.calls[0]
    assert isinstance(spec, ModelSpec) and spec == ModelRoles().ground
    assert schema is not None and schema["properties"]["decision"]["enum"] == [
        "same",
        "different",
        "unsure",
    ]
