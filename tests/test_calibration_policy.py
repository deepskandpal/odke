"""Below its bar, nothing is dropped on a judge's word (DECISIONS #54).

The gate marks what the grounder refused instead of dropping it, and the pair
judge sends what it decided to review instead of linking it. An option given
wins; left out, the judge's calibration card decides through the one switch,
`openodke.calibration.policy.DEFAULT_POLICY`. Every model answer is recorded.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from openodke import Chunk, Document, Entity, Fact, GroundingVerdict, Ontology, Pipeline
from openodke.calibration import (
    JudgedFact,
    LabelSet,
    Made,
    grounding_card,
    lookup,
    pair_card,
    policy,
)
from openodke.calibration.measure import JudgedPair
from openodke.corroborate import REVIEW, NativeResolver, PairJudge
from openodke.gate import Kept, VerdictGate
from openodke.ground import CheckedGrounder, LLMGrounder
from openodke.llm import RecordedClient, ScriptedClient
from openodke.validator import Validator
from test_pair_judge import DOC as PAIR_DOC
from test_pair_judge import FACTS as PAIR_FACTS
from test_pair_judge import _client as pair_client
from test_validator import PLACES, RECORDED, ROWS, A, B

HAIKU = "anthropic/claude-haiku-4-5-20251001"
ADA = Entity(key="Person:ada", type="Person", label="Ada")


def _fact(verdict: GroundingVerdict, value: str = "1815") -> Fact:
    return Fact(subject=ADA, predicate="born", object_value=value, verdict=verdict)


def _ground_card(meets: bool, prompt: str = "ground.span@1") -> Any:
    """A card whose judge agrees with the person (κ 1), or keeps everything (κ 0)."""
    said = "not_found" if meets else "supported"
    rows = [JudgedFact(id=f"s{n}", label="supported", runs=("supported",)) for n in range(8)]
    rows += [JudgedFact(id=f"n{n}", label="not_found", runs=(said,)) for n in range(8)]
    made = Made(
        prompt=prompt,
        prompts={prompt: "x"},
        model=HAIKU,
        labels=LabelSet(name="G", version="1", sha256="0", split="gate", n=16),
        claim_format="inline",
    )
    return grounding_card(rows, made)


@pytest.fixture
def card_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The switch as phase 2 sets it: each judge's default from its card."""
    monkeypatch.setattr(policy, "DEFAULT_POLICY", "card")


@pytest.fixture
def shipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(lookup, "CARDS_DIR", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_mark_writes_a_refused_fact_and_says_why() -> None:
    gate = VerdictGate(on_refusal="mark", refuse_not_found=True)
    verdict = gate.validate(_fact(GroundingVerdict.CONTRADICTED), Ontology())
    assert verdict.action == "accept" and verdict.review == "on_refusal='mark' given"
    assert verdict.reason == "grounding verdict contradicted: the passage says otherwise"
    gate.validate(_fact(GroundingVerdict.NOT_FOUND), Ontology())
    gate.validate(_fact(GroundingVerdict.SUPPORTED), Ontology())
    assert gate.stats == {
        "accepted": 3,
        "refused": {},
        "marked": {"contradicted": 1, "not_found": 1},
    }


def test_a_refusal_for_the_schema_is_code_and_is_always_made() -> None:
    ontology = Ontology.from_dict({"types": {"Person": {}}, "predicates": {"name": {}}})
    gate = VerdictGate(on_refusal="mark", schema=True)
    assert gate.validate(_fact(GroundingVerdict.SUPPORTED), ontology).action == "refuse"


def test_drop_is_the_gate_as_it_was() -> None:
    gate = VerdictGate(on_refusal="drop")
    assert gate.validate(_fact(GroundingVerdict.CONTRADICTED), Ontology()).action == "refuse"
    assert gate.policy == ("drop", "on_refusal='drop' given")


def test_a_gate_told_no_grounder_drops_as_before_and_a_bad_option_is_refused() -> None:
    assert VerdictGate().policy == ("drop", "no grounder was named, so no card was read")
    with pytest.raises(ValueError, match="'drop' or 'mark'"):
        VerdictGate(on_refusal="keep")  # type: ignore[arg-type]


def test_until_the_first_cards_the_default_drops_and_says_so() -> None:
    gate = VerdictGate().follow_card(LLMGrounder(client=ScriptedClient([])))
    assert gate.policy == ("drop", "the default policy is 'act'")


@pytest.mark.usefixtures("card_policy")
def test_with_no_card_the_default_marks_and_names_the_judge() -> None:
    grounder = CheckedGrounder(LLMGrounder(client=ScriptedClient([])), ontology=Ontology())
    gate = VerdictGate().follow_card(grounder)
    assert gate.policy == ("mark", "it has no card, DECISIONS #54")
    verdict = gate.validate(_fact(GroundingVerdict.CONTRADICTED), Ontology())
    assert verdict.review == f"ground.span@1 on {HAIKU}: it has no card, DECISIONS #54"


@pytest.mark.usefixtures("card_policy")
def test_a_card_that_meets_its_bar_drops_and_one_below_it_marks(shipped: Path) -> None:
    grounder = LLMGrounder(client=ScriptedClient([]))
    _ground_card(meets=True).write(shipped)
    assert VerdictGate().follow_card(grounder).policy == ("drop", "its card meets the bar")
    for path in shipped.iterdir():
        path.unlink()
    _ground_card(meets=False).write(shipped)
    gate = VerdictGate().follow_card(grounder)
    assert gate.policy == ("mark", "its card is below the bar, DECISIONS #54")
    # An option given always wins over the card.
    assert VerdictGate(on_refusal="drop").follow_card(grounder).policy[0] == "drop"


@pytest.mark.usefixtures("card_policy")
def test_a_grounder_with_no_model_holds_nothing_back() -> None:
    gate = VerdictGate().follow_card(CheckedGrounder(ontology=Ontology()))
    assert gate.policy == ("drop", "no model judge")


def test_kept_follows_through_to_the_gate_it_records() -> None:
    kept = Kept(VerdictGate(on_refusal="mark"))
    assert kept.follow_card(None) is kept and kept.policy == ("mark", "on_refusal='mark' given")
    assert kept.validate(_fact(GroundingVerdict.CONTRADICTED), Ontology()).action == "accept"
    assert kept.refused == []


# --------------------------------------------------------------------------- #
# The pipeline, the Validator and odke run
# --------------------------------------------------------------------------- #


class _TwoBirthYears:
    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        return [
            _fact(GroundingVerdict.UNCHECKED, "1815"),
            _fact(GroundingVerdict.UNCHECKED, "1816"),
        ]


class _DoubtsTheSecond:
    def ground(self, fact: Fact, doc: Document) -> Fact:
        doubted = fact.object_value != "1815"
        verdict = GroundingVerdict.CONTRADICTED if doubted else GroundingVerdict.SUPPORTED
        return fact.model_copy(update={"verdict": verdict})


def test_a_marked_fact_reaches_the_graph_with_its_mark() -> None:
    gate = VerdictGate(on_refusal="mark")
    kg = Pipeline(Ontology(), _TwoBirthYears(), grounder=_DoubtsTheSecond(), gate=gate).run(
        [Document(id="d1", text="Ada was born in 1815.")]
    )
    by_value = {f.object_value: f for f in kg.facts}
    assert set(by_value) == {"1815", "1816"} and kg.stats["refused"] == 0
    assert kg.stats["marked"] == 1
    assert by_value["1816"].qualifiers[REVIEW] == {
        "reason": "grounding verdict contradicted: the passage says otherwise",
        "why": "on_refusal='mark' given",
    }
    assert REVIEW not in by_value["1815"].qualifiers


def test_a_fact_now_supported_loses_its_old_mark() -> None:
    marked = _fact(GroundingVerdict.UNCHECKED).model_copy(
        update={"qualifiers": {REVIEW: {"reason": "r", "why": "w"}}}
    )

    class Once:
        def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
            return [marked]

    kg = Pipeline(Ontology(), Once(), grounder=_DoubtsTheSecond(), gate=VerdictGate()).run(
        [Document(id="d1", text="Ada was born in 1815.")]
    )
    assert REVIEW not in kg.facts[0].qualifiers


def test_a_user_grounder_with_no_card_is_marked_under_the_card_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "DEFAULT_POLICY", "card")
    gate = VerdictGate()
    Pipeline(Ontology(), _TwoBirthYears(), grounder=_DoubtsTheSecond(), gate=gate)
    assert gate.policy == ("mark", "it has no card, DECISIONS #54")


def test_the_validator_reports_each_judge_and_what_it_marked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "DEFAULT_POLICY", "card")
    _, report = Validator(PLACES, client=RecordedClient(RECORDED)).validate(ROWS, [A, B])
    # The 2012 founding the text contradicts is written marked; the off-schema row is refused.
    assert report.marked == {"contradicted": 1} and report.refused_by == {"predicate": 1}
    (judged,) = report.calibration
    assert (judged.prompt, judged.model, judged.action) == ("ground.span@1", HAIKU, "mark")
    text = report.render()
    assert "marked        1 written for review, not refused: 1 contradicted (odke.review)" in text
    assert (
        f"calibration   ground.span@1 on {HAIKU} (inline): UNCALIBRATED (no card); refusals "
        "marked for review, not dropped (it has no card, DECISIONS #54)"
    ) in text


def test_the_validator_s_default_today_drops_and_its_report_says_why() -> None:
    _, report = Validator(PLACES, client=RecordedClient(RECORDED)).validate(ROWS, [A, B])
    assert report.marked == {} and report.refused_by == {"contradicted": 1, "predicate": 1}
    assert [(c.action, c.why) for c in report.calibration] == [
        ("drop", "the default policy is 'act'")
    ]
    data = report.model_dump(mode="json")
    assert data["schema_version"] == "1.1" and data["calibration"][0]["card"] is None


def test_a_dry_run_reports_no_judge() -> None:
    _, report = Validator(PLACES).validate(ROWS, [A, B], dry_run=True)
    assert report.calibration == ()


def test_odke_run_prints_the_calibration_line(tmp_path: Path) -> None:
    from openodke.run import load_config
    from openodke.run.build import build
    from openodke.run.execute import render, run_built

    examples = Path(__file__).parent.parent / "examples"
    config = load_config(examples / "run.yaml")
    result = run_built(build(config), dry_run=True)
    assert "calibration" in result.stats
    assert any(line.startswith("calibration") for line in render(result).splitlines())


# --------------------------------------------------------------------------- #
# The pair judge
# --------------------------------------------------------------------------- #


def _resolve(judge: PairJudge) -> list[Any]:
    judge.documents[PAIR_DOC.id] = PAIR_DOC
    _, links = NativeResolver(judge=judge).resolve(PAIR_FACTS, {})
    return list(links)


def test_review_links_nothing_and_queues_every_pair_it_decided(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    judge = PairJudge(client=pair_client("same", "same"), queue=queue, on_decision="review")
    assert _resolve(judge) == []
    (decision,) = judge.decisions
    assert decision.decision == "same" and decision.review == "on_decision='review' given"
    (row,) = [json.loads(line) for line in queue.read_text().splitlines()]
    assert row["judge"]["decision"] == "same" and row["judge"]["review"]
    assert (judge.stats["same"], judge.stats["review"], judge.stats["queued"]) == (1, 1, 1)


def test_link_is_the_judge_as_it_was() -> None:
    judge = PairJudge(client=pair_client("same", "same"), on_decision="link")
    (link,) = _resolve(judge)
    assert link.kind == "similar" and judge.policy == ("link", "on_decision='link' given")
    with pytest.raises(ValueError, match="'link' or 'review'"):
        PairJudge(on_decision="merge")  # type: ignore[arg-type]


@pytest.mark.usefixtures("card_policy")
def test_the_pair_judge_s_default_follows_its_card_on_r(shipped: Path) -> None:
    judge = PairJudge(client=pair_client("same", "same"))
    assert judge.policy == ("review", "it has no card, DECISIONS #54")
    assert _resolve(judge) == []
    rows = [JudgedPair(id=f"r{n}", same=True, forward="same", backward="same") for n in range(60)]
    made = Made(
        prompt="pair@1",
        prompts={"pair@1": "x"},
        model=HAIKU,
        labels=LabelSet(name="R", version="1", sha256="0", split="gate", n=60),
    )
    pair_card(rows, made).write(shipped)
    judge = PairJudge(client=pair_client("same", "same"))
    assert judge.policy == ("link", "its card meets the bar")
    assert len(_resolve(judge)) == 1


def test_a_person_s_decision_always_links() -> None:
    reviewed = [{"a": "p:ada", "b": "p:lovelace", "same": True}]
    judge = PairJudge(client=ScriptedClient([]), reviewed=reviewed, on_decision="review")
    (link,) = _resolve(judge)
    assert link.reason == "person: same, from the review queue"
