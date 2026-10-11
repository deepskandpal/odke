"""The reserved keys the M3 stages record their work under.

`Fact` has no free-form metadata slot, and adding one to a frozen type is the
migration DECISIONS #4 warns about. `Fact.qualifiers` is already an open mapping
and never enters `signature` unless a key is named in `identity_keys`, which a
namespaced `odke.` key never is. So each stage writes what it did under one
known key: nothing is overwritten, and a caller asking "why does this value look
like that?" queries a key rather than reading code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from openodke.types import Fact

if TYPE_CHECKING:
    from openodke.ontology import Ontology

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

# On Fact.qualifiers: a fact no source stated in this direction, added because
# the ontology declares an inverse or a symmetric predicate (DECISIONS #28) —
# {"rule": "inverse" | "symmetric", "of": <the stated fact's signature>}.
DERIVED = "odke.derived"

# On Fact.qualifiers: one widen-and-retry attempt by the grounder (#102) —
# {"from": [start, end], "to": [start, end], "verdict": <the retry's answer>}.
# The evidence span is the wider one only when that answer was "supported".
WIDEN = "odke.widen"

# On Fact.qualifiers: why the free checks refused a fact before any model was
# asked about it — {"check": "predicate" | "domain" | "range", "reason": <a sentence>}.
CHECK = "odke.check"

# On Fact.qualifiers: the corroborator's groups of evidence documents that are
# near-duplicates of one another (#154) — ((doc_id, ...), ...), each sorted.
# Every document's evidence stays on the fact; each group counts as one source.
NEAR_DUPLICATES = "odke.near_duplicates"

# On Fact.qualifiers: the `Ontology.fingerprint` of the ontology the pipeline
# checked the fact under, stamped on every fact the gate lets through (#163).
# A fact checked again is stamped again, so it names the last ontology that
# checked it. None is stamped when the ontology has no types and no predicates.
ONTOLOGY = "odke.ontology"

# On Fact.qualifiers: the `OntologySnippet.fingerprint` of the schema slice the
# model extractor was shown for the fact's subject type (#163). Absent where
# no slice was shown: a pattern, a triples row, a fact from elsewhere.
SCHEMA_SLICE = "odke.schema_slice"


def source_forms(value: Any) -> dict[str, tuple[str, ...]]:
    """`qualifiers["odke.source_form"]` read back — tuples in memory, lists after JSON."""
    if not isinstance(value, Mapping):
        return {}
    return {str(k): tuple(v) if isinstance(v, list | tuple) else (v,) for k, v in value.items()}


def near_duplicates(value: Any) -> tuple[tuple[str, ...], ...]:
    """`qualifiers["odke.near_duplicates"]` read back — tuples in memory, lists after JSON."""
    if not isinstance(value, list | tuple):
        return ()
    return tuple(tuple(str(d) for d in group) for group in value if isinstance(group, list | tuple))


def checked_by(fact: Fact) -> str | None:
    """The fingerprint of the ontology that last checked `fact`; None for one never stamped."""
    found = fact.qualifiers.get(ONTOLOGY)
    return found if isinstance(found, str) else None


def checked_under(facts: Iterable[Fact], ontology: Ontology | str) -> list[Fact]:
    """The facts `ontology`, or the fingerprint given, last checked: what a reader filters by.

    `odke ontology diff --store` lists these for the old ontology: the facts a
    breaking change may no longer fit, to validate again.
    """
    wanted = ontology if isinstance(ontology, str) else ontology.fingerprint
    return [fact for fact in facts if checked_by(fact) == wanted]


def stamp_checked(fact: Fact, fingerprint: str | None) -> Fact:
    """`fact` stamped with the ontology that checked it; as it was when nothing did."""
    if fingerprint is None or fact.qualifiers.get(ONTOLOGY) == fingerprint:
        return fact
    return fact.model_copy(update={"qualifiers": {**fact.qualifiers, ONTOLOGY: fingerprint}})


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


__all__ = [
    "CHECK",
    "CONFLICT",
    "DERIVED",
    "NAME_KEY",
    "NEAR_DUPLICATES",
    "ONTOLOGY",
    "SCHEMA_SLICE",
    "SCORE",
    "SOURCE_FORM",
    "checked_by",
    "checked_under",
    "near_duplicates",
    "source_forms",
    "stamp_checked",
    "unstamped",
]
