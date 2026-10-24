"""Which card describes a judge, and the line a report prints for it.

A judge is a registered prompt on a model: `ground.span@1` on
`anthropic/claude-haiku-4-5-20251001`, with the claim written inline. Its card
is the one with that prompt key, that exact model id, that claim format, and a
single run's vote. Anything else is UNCALIBRATED: another prompt version,
another snapshot of the model, a model given without its provider.

Cards are read from the package (`cards/`, beside this module) and from every
directory or file `ODKE_CARDS` names, separated as `PATH` is, so a card made
with `odke calibrate` from your own labels is read by every run that sets it.
A card of yours for the same judge stands in front of the package's.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openodke.calibration.card import Card, load_card, uncalibrated_line
from openodke.types import Frozen

# The package's own cards. A module attribute, so a test can point it elsewhere.
CARDS_DIR = Path(__file__).with_name("cards")
# More cards: directories or files, separated as PATH is.
ENV = "ODKE_CARDS"


class CardStatus(Frozen):
    """A judge's standing: its card, or that it has none, and the line a report prints.

    `meets_bar` is None when there is no card. `card` is the card's id.
    """

    judge: str
    prompt: str
    model: str | None = None
    claim_format: str | None = None
    card: str | None = None
    meets_bar: bool | None = None
    line: str

    @property
    def calibrated(self) -> bool:
        return self.card is not None


def _sources() -> Iterator[Path]:
    for entry in os.environ.get(ENV, "").split(os.pathsep):
        if entry.strip():
            yield Path(entry.strip()).expanduser()
    yield CARDS_DIR


def cards() -> list[Card]:
    """Every card that can be read: `ODKE_CARDS`' first, then the package's."""
    found: list[Card] = []
    for source in _sources():
        paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
        for path in paths:
            if path.is_file():
                found.append(load_card(path))
    return found


def card_for(
    prompt: str, model: str | None, claim_format: str | None = None, *, judge: str | None = None
) -> Card | None:
    """The card for this prompt key, model and claim format, read with a single run's vote."""
    if model is None:
        return None
    for card in cards():
        if (
            card.prompt == prompt
            and card.model == model
            and card.vote == "single"
            and (claim_format is None or card.claim_format in (None, claim_format))
            and (judge is None or card.judge == judge)
        ):
            return card
    return None


def status(
    judge: str, prompt: str, model: str | None, claim_format: str | None = None
) -> CardStatus:
    """The standing of `prompt` on `model`: its card's line, or UNCALIBRATED."""
    card = card_for(prompt, model, claim_format, judge=judge)
    if card is None:
        return CardStatus(
            judge=judge,
            prompt=prompt,
            model=model,
            claim_format=claim_format,
            line=uncalibrated_line(prompt, model, claim_format),
        )
    return CardStatus(
        judge=judge,
        prompt=prompt,
        model=model,
        claim_format=claim_format,
        card=card.id,
        meets_bar=card.meets_bar,
        line=card.line(),
    )


def judge_of(stage: Any) -> Any:
    """The model judge inside a stage, through the wrappers that hold one; None for none.

    A stage with `calibration` is a judge. `CheckedGrounder` holds one as
    `grounder`, and a wrapper of your own may hold one as `inner`. The span check,
    the free checks alone and a pass-through make no judgement, so they hold
    none.
    """
    seen = 0
    while stage is not None and seen < 8:
        if hasattr(stage, "calibration"):
            return stage
        inner = next(
            (getattr(stage, name) for name in ("grounder", "inner") if hasattr(stage, name)),
            None,
        )
        if inner is None:
            return None if _makes_no_judgement(stage) else stage
        stage = inner
        seen += 1
    return None


def _makes_no_judgement(stage: Any) -> bool:
    from openodke.ground.checks import CheckedGrounder
    from openodke.ground.span import SpanGrounder
    from openodke.stages import PassThroughGrounder

    return isinstance(stage, SpanGrounder | PassThroughGrounder | CheckedGrounder)


def status_of(stage: Any) -> CardStatus | None:
    """The standing of the judge a stage holds; None when it holds no model judge.

    A grounder of your own is a judge with no card: UNCALIBRATED, under its
    class's name, until `on_refusal` says what to do with what it refuses.
    """
    judge = judge_of(stage)
    if judge is None:
        return None
    found = getattr(judge, "calibration", None)
    if isinstance(found, CardStatus):
        return found
    name = type(judge).__name__
    return CardStatus(
        judge="ground",
        prompt=name,
        line=f"{name}: UNCALIBRATED, a grounder with no calibration card",
    )


__all__ = ["CARDS_DIR", "ENV", "CardStatus", "card_for", "cards", "judge_of", "status", "status_of"]
