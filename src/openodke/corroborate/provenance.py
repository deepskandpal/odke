"""The reserved keys the M3 stages record their work under.

`Fact` has no free-form metadata slot, and adding one to a frozen type is the
migration DECISIONS #4 warns about. `Fact.qualifiers` is already an open mapping
and never enters `signature` unless a key is named in `identity_keys`, which a
namespaced `odke.` key never is. So each stage writes what it did under one
known key: nothing is overwritten, and a caller asking "why does this value look
like that?" queries a key rather than reading code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openodke.types import Fact

# On Fact.qualifiers: {field: (original spelling, ...)} for every value the
# normaliser rewrote. Union-merged by the corroborator, so all spellings survive.
SOURCE_FORM = "odke.source_form"

# On Entity.attributes: the comparison key for the entity's name — casefolded,
# accents and legal suffixes stripped. The label stays the display form.
NAME_KEY = "odke.name_key"

# On Fact.qualifiers: the corroborator's decision about a contested value —
# {"status": "won" | "lost" | "tied", "reason": <a sentence>, ...}.
CONFLICT = "odke.conflict"

# On Fact.qualifiers: the scorer's inputs, so a score can be read back and
# measured against labels, and re-scoring starts from the extractor's number.
SCORE = "odke.score"


def source_forms(value: Any) -> dict[str, tuple[str, ...]]:
    """`qualifiers["odke.source_form"]` read back — tuples in memory, lists after JSON."""
    if not isinstance(value, Mapping):
        return {}
    return {str(k): tuple(v) if isinstance(v, list | tuple) else (v,) for k, v in value.items()}


def unstamped(fact: Fact) -> Fact:
    """The fact as extraction left it, with the conflict and score stamps removed.

    Both stamps are functions of the batch they were computed in. A stage run
    again over its own output has to start from the extractor's confidence, or
    every pass would compound the previous pass's penalty.
    """
    score, conflict = fact.qualifiers.get(SCORE), fact.qualifiers.get(CONFLICT)
    if score is None and conflict is None:
        return fact
    confidence = fact.confidence
    if isinstance(score, Mapping) and isinstance(score.get("extractor"), int | float):
        confidence = float(score["extractor"])
    elif isinstance(conflict, Mapping) and isinstance(
        conflict.get("confidence_before"), int | float
    ):
        confidence = float(conflict["confidence_before"])
    qualifiers = {k: v for k, v in fact.qualifiers.items() if k not in (SCORE, CONFLICT)}
    return fact.model_copy(update={"confidence": confidence, "qualifiers": qualifiers})


__all__ = ["CONFLICT", "NAME_KEY", "SCORE", "SOURCE_FORM", "source_forms", "unstamped"]
