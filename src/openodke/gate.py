"""The gate: the grounder stamps a verdict, the gate decides what is written.

DECISIONS #20 keeps the grounder from dropping anything, so an ablation can count
what grounding would have removed. Something still has to refuse, or a fact the
cited passage contradicts goes into the graph with `verdict: contradicted` on it
and every query has to remember to filter it out. That something is a `Gate`,
and this is the one `odke run` ships.

In 0.2 it was `openodke.validators.VerdictValidator`. That name still works,
with a warning, until 2.0.0 (DECISIONS #26, #48).

A refusal on a grounding verdict is a judge's word, and a judge below its
calibration bar is not trusted to drop facts (DECISIONS #54). So the gate either
**drops** such a fact or **marks** it: writes it, with `qualifiers["odke.review"]`
saying what it would have refused and why it did not. `on_refusal` chooses;
left out, the grounder's calibration card does (`follow_card`), through
`openodke.calibration.policy`. A refusal for the schema is code, not a judge,
and is always made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from openodke.ground.checks import schema_problem
from openodke.ontology import Ontology
from openodke.types import Fact, GroundingVerdict, ValidationVerdict

OnRefusal = Literal["drop", "mark"]

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
    accepted, why it refused, and what it marked, because the pipeline's own
    `refused` count cannot say why.

    `on_refusal` is what a refusal on a grounding verdict does: "drop" leaves
    the fact out, "mark" writes it marked for review (DECISIONS #54). Left as
    None, `follow_card(grounder)` sets it from the grounder's calibration card:
    a card that meets its bar drops, one below it, or none, marks. A pipeline,
    the Validator and `odke run` each call it with their grounder. A gate never
    told its grounder drops, as it did before #54. `policy` says which applies
    and why.
    """

    def __init__(
        self,
        *,
        refuse_not_found: bool = False,
        schema: bool = False,
        on_refusal: OnRefusal | None = None,
    ) -> None:
        if on_refusal not in (None, "drop", "mark"):
            raise ValueError(f"on_refusal must be 'drop' or 'mark', got {on_refusal!r}")
        self.refuse_not_found = refuse_not_found
        self.schema = schema
        self.on_refusal: OnRefusal | None = on_refusal
        # The default as the grounder's card resolved it (`follow_card`).
        self._followed: tuple[OnRefusal, str] | None = None
        self._accepted = 0
        self._refused: dict[str, int] = {}
        self._marked: dict[str, int] = {}

    def follow_card(self, grounder: Any) -> VerdictGate:
        """Take the default from `grounder`'s calibration card, unless `on_refusal` was given.

        Called again with another grounder, it follows that one. Returns the gate.
        """
        if self.on_refusal is None:
            from openodke.calibration import resolve, status_of

            act, why = resolve(status_of(grounder))
            self._followed = ("drop" if act else "mark", why)
        return self

    @property
    def policy(self) -> tuple[OnRefusal, str]:
        """What a refusal on a grounding verdict does, "drop" or "mark", and why."""
        if self.on_refusal is not None:
            return self.on_refusal, f"on_refusal={self.on_refusal!r} was given"
        if self._followed is not None:
            return self._followed
        return "drop", "no grounder was named, so no card was read"

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
            action, why = self.policy
            if action == "mark":
                self._marked[key] = self._marked.get(key, 0) + 1
                self._accepted += 1
                return ValidationVerdict(action="accept", reason=_REASONS[fact.verdict], review=why)
            self._refused[key] = self._refused.get(key, 0) + 1
            return ValidationVerdict(action="refuse", reason=_REASONS[fact.verdict])
        self._accepted += 1
        return ValidationVerdict(action="accept")

    @property
    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"accepted": self._accepted, "refused": dict(self._refused)}
        if self._marked:
            # Accepted, and written marked for review, in place of a refusal (#54).
            out["marked"] = dict(self._marked)
        return out


# What `odke validate -o` writes beside the graph: `Kept.write`.
REFUSED_FILE = "refused.jsonl"


class Kept:
    """A gate that keeps a record: each fact its inner gate refuses, and why.

    It decides nothing itself, and its `stats` are the inner gate's, so a
    report that counts refusals by reason reads them as before. `refused` is
    what `odke validate -o` writes as `refused.jsonl` (`write`), what `odke
    eval refusals` reads, and what the diagnosis reads as the gate's record.
    """

    def __init__(self, gate: Any) -> None:
        self.gate = gate
        self.refused: list[tuple[Fact, str]] = []

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        verdict: ValidationVerdict = self.gate.validate(fact, ontology)
        if verdict.action == "refuse":
            self.refused.append((fact, verdict.reason or ""))
        return verdict

    def follow_card(self, grounder: Any) -> Kept:
        """The inner gate's `follow_card`, when it has one. Returns this gate."""
        follow = getattr(self.gate, "follow_card", None)
        if callable(follow):
            follow(grounder)
        return self

    @property
    def policy(self) -> Any:
        return getattr(self.gate, "policy", None)

    @property
    def stats(self) -> Any:
        return getattr(self.gate, "stats", None)

    def write(self, path: str | Path) -> Path:
        """What was refused, one JSON line each: the gate's `reason` and the `fact`."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            json.dumps({"reason": reason, "fact": fact.model_dump(mode="json")}, ensure_ascii=False)
            for fact, reason in self.refused
        )
        target.write_text("".join(row + "\n" for row in rows), encoding="utf-8", newline="\n")
        return target


__all__ = ["REFUSED_FILE", "Kept", "OnRefusal", "VerdictGate"]
