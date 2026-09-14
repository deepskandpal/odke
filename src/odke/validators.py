"""The gate: the grounder stamps a verdict, the validator decides what is written.

DECISIONS #20 keeps the grounder from dropping anything, so an ablation can count
what grounding would have removed. Something still has to refuse, or a fact the
cited passage contradicts goes into the graph with `verdict: contradicted` on it
and every query has to remember to filter it out. That something is a
`Validator`, and this is the one `odke run` ships.
"""

from __future__ import annotations

from typing import Any

from odke.ontology import Ontology
from odke.types import Fact, GroundingVerdict, ValidationVerdict

_REASONS = {
    GroundingVerdict.CONTRADICTED: "grounding verdict contradicted: the passage says otherwise",
    GroundingVerdict.NOT_FOUND: "grounding verdict not_found: the passage does not settle it",
}


class VerdictValidator:
    """Refuses a fact its own cited passage contradicts, and on request one it cannot find.

    `CONTRADICTED` is always refused: the source was read, and it says something
    else. `NOT_FOUND` is accepted by default, because a passage that does not
    settle a claim is not evidence against it. A structured fact is grounded
    against its own cell, which never names the subject, so a careful grounder
    answers `not_found` for most of a CSV. `refuse_not_found=True` refuses those
    too, trading recall for precision; `odke eval ablation` measures that trade
    on your labels before you make it. `UNCHECKED` is accepted: nothing was
    asked, and the pass-through refuses nothing either.

    Checks the verdict only, not domain or range. `stats` counts what it
    accepted and why it refused, because the pipeline's own `refused` count
    cannot say why.
    """

    def __init__(self, *, refuse_not_found: bool = False) -> None:
        self.refuse_not_found = refuse_not_found
        self._accepted = 0
        self._refused: dict[str, int] = {}

    @property
    def refused(self) -> frozenset[GroundingVerdict]:
        """The verdicts this gate refuses, in the shape `odke.eval.grounding.kept` takes."""
        if self.refuse_not_found:
            return frozenset({GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND})
        return frozenset({GroundingVerdict.CONTRADICTED})

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        if fact.verdict in self.refused:
            key = fact.verdict.value
            self._refused[key] = self._refused.get(key, 0) + 1
            return ValidationVerdict(action="refuse", reason=_REASONS[fact.verdict])
        self._accepted += 1
        return ValidationVerdict(action="accept")

    @property
    def stats(self) -> dict[str, Any]:
        return {"accepted": self._accepted, "refused": dict(self._refused)}


__all__ = ["VerdictValidator"]
