"""Nothing is dropped on the word of a judge below its bar (DECISIONS #54).

A judge below its calibration bar still runs, and its answers are kept and
reported. What changes is what its answer does:

- the gate **marks** a fact the grounder refused, instead of dropping it: the
  fact is written with `qualifiers["odke.review"]` saying why, for a person;
- the pair judge **sends every pair it judged to review**, instead of linking
  or merging on its decision.

A judge whose card meets its bar acts: the gate drops, the pair judge links.
An option given always wins over the default (`VerdictGate(on_refusal=...)`,
`PairJudge(on_decision=...)`), and every report says which judges are below
their bar and what that changed.

`DEFAULT_POLICY` is the one switch for the default, so that it can be set from
the measured cards and moved again in one line:

- `"card"`: from each judge's card, as above;
- `"act"`: every judge acts, whatever its card says (openodke before #54);
- `"hold"`: no judge acts on its own word.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from openodke.calibration.lookup import CardStatus

Policy = Literal["card", "act", "hold"]
DEFAULT_POLICY: Policy = "card"


class Resolved(NamedTuple):
    """Whether a judge's word stands, and why, in the words a report prints."""

    act: bool
    why: str


def resolve(found: CardStatus | None, policy: Policy | None = None) -> Resolved:
    """Whether the judge `found` describes may act on its own word, under `policy`.

    `found` is None for a stage with no model judge, which has nothing to hold
    back. `policy` defaults to `DEFAULT_POLICY`, read when called.
    """
    chosen = DEFAULT_POLICY if policy is None else policy
    if chosen not in ("card", "act", "hold"):
        raise ValueError(f"policy must be 'card', 'act' or 'hold', got {chosen!r}")
    if found is None:
        return Resolved(True, "no model judge made these verdicts")
    if chosen == "act":
        return Resolved(True, "the default policy lets every judge act, whatever its card says")
    if chosen == "hold":
        return Resolved(False, "the default policy lets no judge act on its own word")
    if found.meets_bar:
        return Resolved(True, f"{found.prompt} on {found.model} meets its calibration bar")
    if found.calibrated:
        return Resolved(False, f"{found.prompt} on {found.model} is below its calibration bar")
    return Resolved(False, f"{found.prompt} on {found.model or 'its model'} is UNCALIBRATED")


__all__ = ["DEFAULT_POLICY", "Policy", "Resolved", "resolve"]
