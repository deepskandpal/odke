"""Nothing is dropped on the word of a judge below its bar (DECISIONS #54).

Provisional (DECISIONS #48), as `openodke.calibration` is: the card format and the bars settle
with the first published cards.

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

It reads `"act"` until the first cards are measured (#134, #135), so nothing
changes for a run before there is a card to read; it is set from them then.
The reports print every judge's card line, or UNCALIBRATED, either way.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from openodke.calibration.lookup import CardStatus

Policy = Literal["card", "act", "hold"]
# The one switch (DECISIONS #54): "act" until the first cards are measured.
DEFAULT_POLICY: Policy = "act"


class Resolved(NamedTuple):
    """Whether a judge's word stands, and why, in the words a report prints."""

    act: bool
    why: str


def resolve(found: CardStatus | None, policy: Policy | None = None) -> Resolved:
    """Whether the judge `found` describes may act on its own word, under `policy`.

    `found` is None for a stage with no model judge, which has nothing to hold
    back. `policy` defaults to `DEFAULT_POLICY`, read when called. `why` is
    short, and names no judge: a report prints it after the judge's line.
    """
    chosen = DEFAULT_POLICY if policy is None else policy
    if chosen not in ("card", "act", "hold"):
        raise ValueError(f"policy must be 'card', 'act' or 'hold', got {chosen!r}")
    if found is None:
        return Resolved(True, "no model judge")
    if chosen == "act":
        return Resolved(True, "the default policy is 'act'")
    if chosen == "hold":
        return Resolved(False, "the default policy is 'hold'")
    if found.meets_bar:
        return Resolved(True, "its card meets the bar")
    if found.calibrated:
        return Resolved(False, "its card is below the bar, DECISIONS #54")
    return Resolved(False, "it has no card, DECISIONS #54")


__all__ = ["DEFAULT_POLICY", "Policy", "Resolved", "resolve"]
