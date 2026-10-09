"""The shipped gate: the grounder stamps, `VerdictGate` refuses."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from collections.abc import Iterable

import pytest

from openodke import (
    Chunk,
    Document,
    Entity,
    Fact,
    Gate,
    GroundingVerdict,
    Ontology,
    Pipeline,
    VerdictGate,
)
from openodke.eval.grounding import kept
from openodke.stages import PassThroughGate

ADA = Entity(key="Person:ada", type="Person", label="Ada")


def _fact(verdict: GroundingVerdict, value: str = "1815") -> Fact:
    return Fact(subject=ADA, predicate="born", object_value=value, verdict=verdict)


def test_a_contradicted_fact_is_refused_with_a_reason() -> None:
    verdict = VerdictGate().validate(_fact(GroundingVerdict.CONTRADICTED), Ontology())
    assert verdict.action == "refuse"
    assert verdict.reason is not None and "contradicted" in verdict.reason


def test_not_found_and_unchecked_are_accepted_by_default() -> None:
    """A passage that does not settle a claim is not evidence against it."""
    gate = VerdictGate()
    for verdict in (
        GroundingVerdict.NOT_FOUND,
        GroundingVerdict.UNCHECKED,
        GroundingVerdict.SUPPORTED,
    ):
        assert gate.validate(_fact(verdict), Ontology()).action == "accept"


def test_refusing_not_found_is_a_choice_and_unchecked_still_passes() -> None:
    gate = VerdictGate(refuse_not_found=True)
    assert gate.validate(_fact(GroundingVerdict.NOT_FOUND), Ontology()).action == "refuse"
    assert gate.validate(_fact(GroundingVerdict.UNCHECKED), Ontology()).action == "accept"


def test_refused_is_the_drop_set_the_ablation_scores_with() -> None:
    """One definition of the gate, so `odke eval ablation` measures what the run refuses."""
    facts = [_fact(v) for v in GroundingVerdict]
    for gate in (VerdictGate(), VerdictGate(refuse_not_found=True)):
        by_gate = [gate.validate(f, Ontology()).action == "accept" for f in facts]
        assert by_gate == [kept(f, gate.refused) for f in facts]


def test_it_counts_what_it_accepted_and_why_it_refused() -> None:
    gate = VerdictGate(refuse_not_found=True)
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
    gate = VerdictGate()
    assert isinstance(gate, Gate)
    kg = Pipeline(Ontology(), _TwoBirthYears(), grounder=_DoubtsTheSecond(), gate=gate).run(
        [Document(id="d1", text="Ada was born in 1815.")]
    )
    assert [f.object_value for f in kg.facts] == ["1815"]
    assert kg.stats["refused"] == 1
    assert gate.stats["refused"] == {"contradicted": 1}


def test_with_schema_it_refuses_what_the_ontology_has_no_room_for() -> None:
    ontology = Ontology.from_dict(
        {
            "types": {"Person": {}, "Company": {}},
            "predicates": {"employer": {"domain": ["Person"], "range": "Company"}},
        }
    )
    gate = VerdictGate(schema=True)
    off = _fact(GroundingVerdict.SUPPORTED)
    verdict = gate.validate(off, ontology)
    assert verdict.action == "refuse" and verdict.reason is not None
    assert verdict.reason.startswith(f"schema: {off.predicate!r} is not a predicate")
    assert gate.stats["refused"] == {"predicate": 1}
    # Without it, the gate reads verdicts alone, as it always has.
    assert VerdictGate().validate(off, ontology).action == "accept"
    # An ontology that declares nothing has nothing to refuse on.
    assert gate.validate(off, Ontology()).action == "accept"


# --------------------------------------------------------------------------- #
# The 0.2 names (DECISIONS #26): each still works, and says it is going
# --------------------------------------------------------------------------- #

OLD_NAMES = [
    ("openodke", "VerdictValidator", VerdictGate),
    ("openodke.stages", "Validator", Gate),
    ("openodke.stages", "PassThroughValidator", PassThroughGate),
    ("openodke.pipeline", "Validator", Gate),
    ("openodke.pipeline", "PassThroughValidator", PassThroughGate),
    ("openodke.validators", "VerdictValidator", VerdictGate),
]


@pytest.mark.parametrize(
    ("module", "old", "new"), OLD_NAMES, ids=lambda v: getattr(v, "__name__", v)
)
def test_every_old_name_is_the_new_class_and_warns(module: str, old: str, new: type) -> None:
    found = importlib.import_module(module)
    with pytest.warns(
        DeprecationWarning, match=rf"{module}\.{old} is deprecated: use .*{new.__name__}"
    ):
        assert getattr(found, old) is new


def test_the_old_names_stay_in_all_beside_the_new_ones() -> None:
    import openodke
    from openodke import stages

    assert {"Gate", "VerdictGate", "Validator", "VerdictValidator"} <= set(openodke.__all__)
    assert {"Gate", "PassThroughGate", "Validator", "PassThroughValidator"} <= set(stages.__all__)


def test_an_old_import_warns_at_the_line_that_wrote_it() -> None:
    with pytest.warns(DeprecationWarning) as record:
        from openodke.validators import VerdictValidator
    assert record[0].filename == __file__
    gate = VerdictValidator(refuse_not_found=True)
    assert isinstance(gate, VerdictGate)
    assert gate.validate(_fact(GroundingVerdict.NOT_FOUND), Ontology()).action == "refuse"


def test_the_top_level_validator_is_the_layer_now_and_the_stage_name_still_warns() -> None:
    """1.0.0 gives `openodke.Validator` to the layer (#129); the gate's old name there is gone."""
    import openodke
    from openodke.validator import Validator

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert openodke.Validator is Validator
        from openodke import Validator as imported
    assert imported is Validator and imported is not Gate
    # It has a gate's method name, so code that still uses it as the gate is told so.
    with pytest.raises(TypeError, match=r"the gate is openodke\.Gate"):
        Validator().validate(_fact(GroundingVerdict.SUPPORTED), Ontology())
    # Where the old name named only the gate, it still does, and still says so.
    from openodke import stages

    with pytest.warns(DeprecationWarning, match=r"openodke\.stages\.Validator is deprecated"):
        assert stages.Validator is Gate


def test_an_unknown_name_is_still_an_attribute_error() -> None:
    import openodke

    with pytest.raises(AttributeError, match="no attribute 'Nothing'"):
        openodke.Nothing  # noqa: B018


def test_nothing_in_the_package_uses_an_old_name() -> None:
    """The warning is for callers: importing openodke itself must not raise one."""
    modules = (
        "openodke, openodke.run, openodke.cli.main, openodke.eval.ablation, openodke.validators"
    )
    subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", f"import {modules}"], check=True
    )
