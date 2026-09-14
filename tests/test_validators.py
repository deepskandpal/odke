"""The shipped gate: the grounder stamps, `VerdictValidator` refuses."""

from __future__ import annotations

from collections.abc import Iterable

from odke import (
    Chunk,
    Document,
    Entity,
    Fact,
    GroundingVerdict,
    Ontology,
    Pipeline,
    Validator,
    VerdictValidator,
)
from odke.eval.grounding import kept

ADA = Entity(key="Person:ada", type="Person", label="Ada")


def _fact(verdict: GroundingVerdict, value: str = "1815") -> Fact:
    return Fact(subject=ADA, predicate="born", object_value=value, verdict=verdict)


def test_a_contradicted_fact_is_refused_with_a_reason() -> None:
    verdict = VerdictValidator().validate(_fact(GroundingVerdict.CONTRADICTED), Ontology())
    assert verdict.action == "refuse"
    assert verdict.reason is not None and "contradicted" in verdict.reason


def test_not_found_and_unchecked_are_accepted_by_default() -> None:
    """A passage that does not settle a claim is not evidence against it."""
    gate = VerdictValidator()
    for verdict in (
        GroundingVerdict.NOT_FOUND,
        GroundingVerdict.UNCHECKED,
        GroundingVerdict.SUPPORTED,
    ):
        assert gate.validate(_fact(verdict), Ontology()).action == "accept"


def test_refusing_not_found_is_a_choice_and_unchecked_still_passes() -> None:
    gate = VerdictValidator(refuse_not_found=True)
    assert gate.validate(_fact(GroundingVerdict.NOT_FOUND), Ontology()).action == "refuse"
    assert gate.validate(_fact(GroundingVerdict.UNCHECKED), Ontology()).action == "accept"


def test_refused_is_the_drop_set_the_ablation_scores_with() -> None:
    """One definition of the gate, so `odke eval ablation` measures what the run refuses."""
    facts = [_fact(v) for v in GroundingVerdict]
    for gate in (VerdictValidator(), VerdictValidator(refuse_not_found=True)):
        by_gate = [gate.validate(f, Ontology()).action == "accept" for f in facts]
        assert by_gate == [kept(f, gate.refused) for f in facts]


def test_it_counts_what_it_accepted_and_why_it_refused() -> None:
    gate = VerdictValidator(refuse_not_found=True)
    for verdict in (
        GroundingVerdict.CONTRADICTED,
        GroundingVerdict.NOT_FOUND,
        GroundingVerdict.NOT_FOUND,
        GroundingVerdict.SUPPORTED,
    ):
        gate.validate(_fact(verdict), Ontology())
    assert gate.stats == {"accepted": 1, "refused": {"contradicted": 1, "not_found": 2}}


class _TwoBirthYears:
    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        return [
            _fact(GroundingVerdict.UNCHECKED, "1815"),
            _fact(GroundingVerdict.UNCHECKED, "1816"),
        ]


class _DoubtsTheSecond:
    def ground(self, fact: Fact, doc: Document) -> Fact:
        verdict = (
            GroundingVerdict.SUPPORTED
            if fact.object_value == "1815"
            else GroundingVerdict.CONTRADICTED
        )
        return fact.model_copy(update={"verdict": verdict})


def test_in_a_pipeline_the_contradicted_fact_never_reaches_the_graph() -> None:
    gate = VerdictValidator()
    assert isinstance(gate, Validator)
    kg = Pipeline(Ontology(), _TwoBirthYears(), grounder=_DoubtsTheSecond(), validator=gate).run(
        [Document(id="d1", text="Ada was born in 1815.")]
    )
    assert [f.object_value for f in kg.facts] == ["1815"]
    assert kg.stats["refused"] == 1
    assert gate.stats["refused"] == {"contradicted": 1}
