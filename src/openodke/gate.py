"""The gate: the grounder stamps a verdict, the gate decides what is written.

DECISIONS #20 keeps the grounder from dropping anything, so an ablation can count
what grounding would have removed. Something still has to refuse, or a fact the
cited passage contradicts goes into the graph with `verdict: contradicted` on it
and every query has to remember to filter it out. That something is a `Gate`,
and this is the one `odke run` ships.

In 0.2 it was `openodke.validators.VerdictValidator`. That name still works,
with a warning, until 1.0.0 (DECISIONS #26).
"""

from __future__ import annotations

from typing import Any

from openodke.ground.checks import schema_problem
from openodke.ontology import Ontology
from openodke.types import Fact, GroundingVerdict, ValidationVerdict

_REASONS = {
    GroundingVerdict.CONTRADICTED: "grounding verdict contradicted: the passage says otherwise",
    GroundingVerdict.NOT_FOUND: "grounding verdict not_found: the passage does not settle it",
}


class VerdictGate:
    """Refuses a fact its own cited passage contradicts, and on request one it cannot find.

    `CONTRADICTED` is always refused: the source was read, and it says something
    else. `NOT_FOUND` is accepted by default, because a passage that does not
    settle a claim is not evidence against it. A structured fact is grounded
    against its own cell, which never names the subject, so a careful grounder
    answers `not_found` for most of a CSV. `refuse_not_found=True` refuses those
    too, trading recall for precision; `odke eval ablation` measures that trade
    on your labels before you make it. `UNCHECKED` is accepted: nothing was
    asked, and the pass-through refuses nothing either.

    By default it checks the verdict only. `schema=True` also refuses a fact
    the ontology has no room for: a predicate it does not declare, a subject
    outside the domain, an object outside the range (`schema_problem`, the free
    checks `CheckedGrounder` runs before any model). `stats` counts what it
    accepted and why it refused, because the pipeline's own `refused` count
    cannot say why.
    """

    def __init__(self, *, refuse_not_found: bool = False, schema: bool = False) -> None:
        self.refuse_not_found = refuse_not_found
        self.schema = schema
        self._accepted = 0
        self._refused: dict[str, int] = {}

    @property
    def refused(self) -> frozenset[GroundingVerdict]:
        """The verdicts this gate refuses, in the shape `openodke.eval.grounding.kept` takes."""
        if self.refuse_not_found:
            return frozenset({GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND})
        return frozenset({GroundingVerdict.CONTRADICTED})

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        problem = schema_problem(fact, ontology) if self.schema else None
        if problem is not None:
            check, reason = problem
            self._refused[check] = self._refused.get(check, 0) + 1
            return ValidationVerdict(action="refuse", reason=f"schema: {reason}")
        if fact.verdict in self.refused:
            key = fact.verdict.value
            self._refused[key] = self._refused.get(key, 0) + 1
            return ValidationVerdict(action="refuse", reason=_REASONS[fact.verdict])
        self._accepted += 1
        return ValidationVerdict(action="accept")

    @property
    def stats(self) -> dict[str, Any]:
        return {"accepted": self._accepted, "refused": dict(self._refused)}


__all__ = ["VerdictGate"]
