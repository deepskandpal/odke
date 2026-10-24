"""Calibration cards (#135): the arithmetic, the card, where cards are found, and the policy (#54).

No model is called: a card is made from per-item outcomes, as `odke calibrate`
and `bench/calibrate.py` hand them over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openodke.calibration import (
    BANDS,
    CARD_VERSION,
    UNCALIBRATED,
    Card,
    CardStatus,
    JudgedFact,
    JudgedPair,
    LabelSet,
    Made,
    card_for,
    cards,
    equivalence_card,
    grounding_card,
    judge_of,
    kappa,
    label_hash,
    load_card,
    lookup,
    pair_card,
    policy,
    resolve,
    share,
    standing,
    status,
    status_of,
    swap_rule,
    voted,
)
from openodke.calibration.measure import SELF_AGREEMENT_NOTE, flips
from openodke.ground import CheckedGrounder, LLMGrounder, SpanGrounder
from openodke.llm import ScriptedClient
from openodke.ontology import Ontology
from openodke.stages import PassThroughGrounder

HAIKU = "anthropic/claude-haiku-4-5-20251001"
LABELS = LabelSet(name="G", version="1", sha256="0" * 64, split="gate", n=0)


def _made(**update: Any) -> Made:
    base = Made(
        prompt="ground.span@1",
        prompts={"ground.span@1": "dbc23eec"},
        model=HAIKU,
        labels=LABELS,
        claim_format="inline",
        date="2026-10-24",
    )
    return Made(**{**base.__dict__, **update})


def _facts(*rows: tuple[str, str], words: int = 20) -> list[JudgedFact]:
    """(person, judge) per item, one run each."""
    return [
        JudgedFact(id=f"g{n}", label=p, runs=(j,), words=words) for n, (p, j) in enumerate(rows)
    ]


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #


def test_kappa_is_cohen_s() -> None:
    pairs = [("keep", "keep")] * 40 + [("refuse", "refuse")] * 10 + [("keep", "refuse")] * 5
    found = kappa(pairs, ("keep", "refuse"))
    # p_o = 50/55; p_e = (45·40 + 10·15)/55²; κ = (p_o − p_e)/(1 − p_e) = 0.7442
    assert found.value == pytest.approx(0.744186, abs=1e-6) and found.n == 55
    assert found.low is not None and found.high is not None
    assert found.low <= found.value <= found.high


def test_kappa_is_undefined_when_both_raters_say_one_thing() -> None:
    assert kappa([("keep", "keep")] * 5, ("keep", "refuse")).value is None
    assert kappa([], ("keep", "refuse")).value is None


def test_eighty_eight_percent_agreement_can_be_a_low_kappa() -> None:
    # D01 §5c: high raw agreement on a skewed set says little.
    pairs = [("keep", "keep")] * 85 + [("refuse", "keep")] * 9 + [("refuse", "refuse")] * 3
    pairs += [("keep", "refuse")] * 3
    found = kappa(pairs, ("keep", "refuse"))
    assert 0.25 < (found.value or 0) < 0.35


def test_a_share_carries_its_wilson_interval() -> None:
    found = share(52, 59)
    assert (found.k, found.n) == (52, 59)
    assert (round(found.low or 0, 2), round(found.high or 0, 2)) == (0.77, 0.94)
    assert share(0, 0).value is None


def test_the_swap_rule_and_the_two_of_three_vote() -> None:
    assert swap_rule("same", "same") == "same"
    assert swap_rule("same", "different") == "unsure"
    assert swap_rule(None, None) == "unsure"
    assert voted(("supported", "not_found", "contradicted"), "single") == "supported"
    assert voted(("supported", "not_found", "contradicted"), "refuse_2_of_3") == "contradicted"
    assert voted(("not_found", "supported", "not_found"), "refuse_2_of_3") == "not_found"
    assert voted(("supported", "supported", "not_found"), "refuse_2_of_3") == "supported"
    assert voted(("unchecked", "unchecked", "not_found"), "refuse_2_of_3") == "unchecked"


def test_flips_count_items_whose_verdict_changed_between_runs() -> None:
    found = flips(
        [("supported", "supported"), ("supported", "not_found"), ("not_found", "contradicted")]
    )
    assert found is not None
    changed, turned = found
    assert (changed.k, turned.k) == (2, 1)
    assert flips([("supported",)]) is None


def test_label_hash_is_the_labels_not_their_order() -> None:
    a = label_hash([("g1", "supported"), ("g2", "not_found")])
    assert a == label_hash([("g2", "not_found"), ("g1", "supported")])
    assert a != label_hash([("g1", "supported"), ("g2", "supported")])


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #


def test_a_grounding_card_reads_keep_against_refuse_and_counts_abstentions_as_kept() -> None:
    items = _facts(
        ("supported", "supported"),
        ("supported", "supported"),
        ("supported", "not_found"),
        ("not_found", "not_found"),
        ("contradicted", "contradicted"),
        ("not_found", "unchecked"),
    )
    planted = [JudgedFact(id="p1", label="not_found", runs=("contradicted",), planted=True)]
    planted.append(JudgedFact(id="p2", label="not_found", runs=("supported",), planted=True))
    card = grounding_card(items, _made(), planted=planted)
    g = card.grounding
    assert g is not None and card.judge == "ground"
    assert (g.fail_recall.k, g.fail_recall.n) == (2, 3)
    assert (g.false_refusals.k, g.false_refusals.n) == (1, 3)
    assert g.kappa.n == 5 and (g.abstain.k, g.abstain.n) == (1, 6)
    assert g.planted is not None and (g.planted.k, g.planted.n) == (1, 2)
    assert g.confusion["not_found"]["unchecked"] == 1
    assert g.runs == 1 and g.flips is None
    assert card.self_agreement is None and card.bar.note == SELF_AGREEMENT_NOTE
    assert card.bar.metric == "kappa_keep_refuse" and card.bar.threshold == 0.6


def test_the_grounding_bar_is_kappa_at_least_point_six() -> None:
    good = _facts(*[("supported", "supported")] * 8, *[("not_found", "not_found")] * 8)
    assert grounding_card(good, _made()).bar.met
    poor = _facts(*[("supported", "supported")] * 8, *[("not_found", "supported")] * 8)
    assert not grounding_card(poor, _made()).bar.met


def test_with_self_agreement_the_bar_needs_kappa_within_a_tenth_of_it() -> None:
    items = _facts(
        *[("supported", "supported")] * 9,
        ("supported", "not_found"),
        *[("not_found", "not_found")] * 6,
    )
    card = grounding_card(items, _made(), self_agreement=0.99)
    assert card.grounding is not None and (card.grounding.kappa.value or 0) > 0.6
    assert not card.bar.met and card.bar.note is None and card.self_agreement == 0.99


def test_results_are_split_by_passage_length() -> None:
    items = [
        JudgedFact(id="a", label="supported", runs=("supported",), words=12),
        JudgedFact(id="b", label="not_found", runs=("supported",), words=90),
        JudgedFact(id="c", label="supported", runs=("not_found",), words=400),
    ]
    card = grounding_card(items, _made())
    assert card.grounding is not None
    assert [(b.name, b.n) for b in card.grounding.by_length] == [
        ("sentence", 1),
        ("paragraph", 1),
        ("document", 1),
    ]
    assert [name for name, _, _ in BANDS] == ["sentence", "paragraph", "document"]


def test_three_runs_give_the_flip_rate_and_the_two_of_three_card() -> None:
    items = [
        JudgedFact(id="a", label="supported", runs=("supported", "not_found", "supported")),
        JudgedFact(id="b", label="not_found", runs=("not_found", "not_found", "supported")),
    ]
    card = grounding_card(items, _made())
    assert card.grounding is not None and card.grounding.flips is not None
    assert card.grounding.flips.k == 2 and card.grounding.runs == 3
    vote = grounding_card(items, _made(), vote="refuse_2_of_3")
    assert vote.vote == "refuse_2_of_3" and vote.id.endswith("__refuse_2_of_3")
    with pytest.raises(ValueError, match="three runs"):
        grounding_card(_facts(("supported", "supported")), _made(), vote="refuse_2_of_3")
    with pytest.raises(ValueError, match="same number of runs"):
        grounding_card([*items, *_facts(("supported", "supported"))], _made())


def test_a_pair_card_bars_the_lower_bound_of_the_precision_of_same() -> None:
    sure = [JudgedPair(id=f"r{n}", same=True, forward="same", backward="same") for n in range(60)]
    apart = [
        JudgedPair(id=f"d{n}", same=False, forward="different", backward="different")
        for n in range(30)
    ]
    torn = [JudgedPair(id="t", same=True, forward="same", backward="different")]
    card = pair_card([*sure, *apart, *torn], _made(prompt="pair@1", claim_format=None))
    p = card.pairs
    assert p is not None and card.judge == "pair"
    assert (p.precision_same.k, p.precision_same.n) == (60, 60)
    assert (p.recall_same.k, p.recall_same.n) == (60, 61)
    assert (p.unsure.k, p.order_disagreement.k) == (1, 1)
    assert card.bar.value == p.precision_same.low and card.bar.met
    small = pair_card(sure[:10], _made(prompt="pair@1", claim_format=None))
    assert not small.bar.met  # 10 of 10 is a lower bound of 0.72


def test_an_equivalence_card_reads_same_against_not() -> None:
    rows = [JudgedPair(id=f"s{n}", same=True, forward="same", backward="same") for n in range(10)]
    rows += [
        JudgedPair(id=f"d{n}", same=False, forward="different", backward="unsure")
        for n in range(10)
    ]
    card = equivalence_card(rows, _made(prompt="fact_equiv@1", claim_format=None))
    e = card.equivalence
    assert e is not None and e.kappa.value == 1.0 and card.bar.met
    assert (e.unsure.k, e.order_disagreement.k, e.judge_same.k, e.person_same.k) == (10, 10, 10, 10)


def test_a_card_round_trips_and_a_newer_minor_reads(tmp_path: Path) -> None:
    card = grounding_card(_facts(("supported", "supported"), ("not_found", "not_found")), _made())
    written = card.write(tmp_path / "card.json")
    assert load_card(written) == card
    data = json.loads(written.read_text())
    data["card_version"] = "1.9"
    data["a_field_from_later"] = True
    written.write_text(json.dumps(data))
    assert load_card(written).prompt == "ground.span@1"
    data["card_version"] = "2.0"
    written.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"this package reads {CARD_VERSION}"):
        load_card(written)


def test_a_card_written_to_a_directory_names_itself(tmp_path: Path) -> None:
    card = grounding_card(_facts(("supported", "supported")), _made())
    path = card.write(tmp_path / "new")
    assert path.name == "ground.span@1__anthropic--claude-haiku-4-5-20251001__inline.json"
    assert "κ" in card.render() and card.line().startswith("ground.span@1 on " + HAIKU)


# --------------------------------------------------------------------------- #
# Where a card is found
# --------------------------------------------------------------------------- #


def _card(meets: bool, **update: Any) -> Card:
    if meets:
        rows = _facts(*[("supported", "supported")] * 8, *[("not_found", "not_found")] * 8)
    else:
        rows = _facts(*[("supported", "supported")] * 8, *[("not_found", "supported")] * 8)
    return grounding_card(rows, _made(**update))


def test_no_card_is_uncalibrated() -> None:
    assert cards() == []
    found = status("ground", "ground.span@1", HAIKU, "inline")
    assert not found.calibrated and found.meets_bar is None
    assert found.line == f"ground.span@1 on {HAIKU} (inline): {UNCALIBRATED} (no card)"


def test_a_shipped_card_is_found_by_prompt_model_and_claim_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _card(True).write(tmp_path)
    _card(False, claim_format="labelled").write(tmp_path)
    monkeypatch.setattr(lookup, "CARDS_DIR", tmp_path)
    assert status("ground", "ground.span@1", HAIKU, "inline").meets_bar is True
    assert status("ground", "ground.span@1", HAIKU, "labelled").meets_bar is False
    assert not status("ground", "ground.span@1", "anthropic/claude-haiku-4-5", "inline").calibrated
    assert not status("ground", "ground.onyx@1", HAIKU, "sentence").calibrated
    assert card_for("ground.span@1", None) is None


def test_a_card_of_yours_in_odke_cards_stands_before_the_package_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped, yours = tmp_path / "shipped", tmp_path / "yours"
    _card(False).write(shipped)
    _card(True).write(yours)
    monkeypatch.setattr(lookup, "CARDS_DIR", shipped)
    assert status("ground", "ground.span@1", HAIKU, "inline").meets_bar is False
    monkeypatch.setenv("ODKE_CARDS", str(yours))
    assert status("ground", "ground.span@1", HAIKU, "inline").meets_bar is True


def test_a_vote_card_is_never_a_single_call_s() -> None:
    items = [JudgedFact(id="a", label="supported", runs=("supported",) * 3)]
    assert grounding_card(items, _made(), vote="refuse_2_of_3").vote == "refuse_2_of_3"


# --------------------------------------------------------------------------- #
# Which judge a stage holds
# --------------------------------------------------------------------------- #


def test_the_judge_is_found_inside_the_free_checks() -> None:
    grounder = LLMGrounder(client=ScriptedClient([]))
    assert judge_of(CheckedGrounder(grounder, ontology=Ontology())) is grounder
    for none in (SpanGrounder(), PassThroughGrounder(), CheckedGrounder(ontology=Ontology()), None):
        assert judge_of(none) is None and status_of(none) is None


def test_a_grounder_of_your_own_is_a_judge_with_no_card() -> None:
    class Mine:
        def ground(self, fact: Any, doc: Any) -> Any:
            return fact

    found = status_of(Mine())
    assert found is not None and found.prompt == "Mine" and not found.calibrated
    assert found.line == f"Mine: {UNCALIBRATED} (no card)"


# --------------------------------------------------------------------------- #
# The policy (DECISIONS #54)
# --------------------------------------------------------------------------- #

BELOW = CardStatus(judge="ground", prompt="p@1", model="m", card="c", meets_bar=False, line="l")
MEETS = BELOW.model_copy(update={"meets_bar": True})
NONE = BELOW.model_copy(update={"card": None, "meets_bar": None})


def test_the_switch_reads_act_until_the_first_cards_are_measured() -> None:
    assert policy.DEFAULT_POLICY == "act"
    assert resolve(NONE) == (True, "the default policy is 'act'")


@pytest.mark.parametrize(
    ("found", "act", "why"),
    [
        (MEETS, True, "its card meets the bar"),
        (BELOW, False, "its card is below the bar, DECISIONS #54"),
        (NONE, False, "it has no card, DECISIONS #54"),
        (None, True, "no model judge"),
    ],
)
def test_under_card_a_judge_acts_only_on_a_card_that_meets_its_bar(
    found: CardStatus | None, act: bool, why: str
) -> None:
    assert resolve(found, "card") == (act, why)


def test_hold_holds_every_judge_and_an_unknown_policy_is_refused() -> None:
    assert resolve(MEETS, "hold").act is False
    assert resolve(None, "hold").act is True
    with pytest.raises(ValueError, match="'card', 'act' or 'hold'"):
        resolve(MEETS, "sometimes")  # type: ignore[arg-type]


def test_the_switch_is_read_when_resolving(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "DEFAULT_POLICY", "card")
    assert resolve(NONE).act is False


def test_standing_reports_each_judge_with_what_its_word_did() -> None:
    from openodke.gate import VerdictGate

    grounder = LLMGrounder(client=ScriptedClient([]))
    gate = VerdictGate(on_refusal="mark").follow_card(grounder)
    (entry,) = standing({"grounder": grounder, "gate": gate})
    assert entry.action == "mark" and entry.why == "on_refusal='mark' given"
    assert entry.text().endswith(
        "refusals marked for review, not dropped (on_refusal='mark' given)"
    )
    assert standing({"grounder": SpanGrounder(), "gate": gate}) == []
