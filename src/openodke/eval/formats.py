"""What to label: one JSONL format per stage.

Each label row model below says what one row is and what "correct" means for
it, because that is the whole contract. A caller who labels a slice of their
own corpus in this shape can score any stage without reading an evaluator;
`odke eval <stage> --describe` prints the same docstrings.

Two shapes recur. A row about a chunk carries the chunk's text and an `id` to
join predictions on. A row about a fact carries the `Fact` itself — only its
identity fields when written by hand, or a line straight out of a sink's
`facts.jsonl` when labelling pipeline output — plus the expected outcome.
Predictions for the fact stages are `Fact` rows joined on `Fact.id`, so the
file a `JsonlSink` wrote is already a predictions file. Where the label row's
own fact already carries the field being scored (`verdict`, `confidence`),
predictions are optional and override it.

Ids are what the join is on. A hand-written fact with no `"id"` gets a fresh
one on every load, so give it one — or use a `run_*` helper, which scores a
stage in-process and needs no join.

The fixtures under `tests/fixtures/eval/` are examples of these formats, a
handful of rows each. They are not a benchmark: no number computed on them
says anything outside this package's own test suite.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, model_validator

from openodke.types import Chunk, Document, Evidence, Fact, Frozen, GroundingVerdict, LinkKind

Action = Literal["extract", "skip", "defer"]
Decision = Literal["accept", "refuse", "conflict"]


# --------------------------------------------------------------------------- #
# route
# --------------------------------------------------------------------------- #


class RouteLabel(Frozen):
    """One labelled chunk: the text a router sees, and what it should say.

    A row is one chunk — the unit of every metric (DECISIONS #19). `action`
    is what the router should return: `extract`, `skip` or `defer`. `label`
    is your own name for the chunk's kind (fact, policy, marketing — the
    package ships no taxonomy) and may be left out. Correct means the
    predicted action equals yours; a predicted label is correct when it
    equals yours as a string. Rows with no `label` count toward the action
    metrics only.
    """

    id: str
    text: str
    action: Action
    label: str | None = None

    def as_chunk(self) -> Chunk:
        return Chunk(doc_id=self.id, start=0, end=len(self.text), text=self.text, index=0)


class RoutePrediction(Frozen):
    """What a router said about the chunk with this `id`."""

    id: str
    action: Action
    label: str | None = None


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


class GoldFact(Frozen):
    """One fact a document states, as you would want it extracted.

    A row is one gold fact in one document. `fact` needs only its identity:
    subject key and type, predicate, `object_entity` or `object_value`,
    `polarity` if not asserted, and identity-bearing qualifiers with their
    `identity_keys`. Evidence, confidence and the rest are ignored. Keys must
    follow the convention your extractor emits: this scores extraction before
    resolution has had a chance to reconcile them.

    Predictions are `Fact` rows, scored inside the first labelled document
    their evidence cites. A prediction that cites no document is matched
    against every labelled document; one that cites only documents you did
    not label is left out and counted in the notes. Correct means sharing the
    gold fact's `signature`, with one concession: literal values compare
    after whitespace is collapsed and case folded, so `1815` and `"1815"`,
    and `"Ada Lovelace"` and `"ada  lovelace"`, are one value.
    """

    doc_id: str
    fact: Fact


# --------------------------------------------------------------------------- #
# ground
# --------------------------------------------------------------------------- #


class GroundingLabel(Frozen):
    """One fact, the passage it cites, and whether the passage supports it.

    A row is one fact whose evidence span indexes into `text`, and the
    verdict a careful reader gives: `supported`, `contradicted` or
    `not_found`. `doc_id` names the document the span points into; it
    defaults to the first evidence's `doc_id`, then to the fact's id.

    Predictions are `Fact` rows whose `verdict` a grounder set, joined on
    `Fact.id`; without them the label row's own fact is scored as it stands.
    Correct means the predicted verdict equals yours. `unchecked` is never
    correct — it is what a grounder that did not decide leaves behind.
    """

    text: str
    fact: Fact
    verdict: Literal["supported", "contradicted", "not_found"]
    doc_id: str | None = None

    @property
    def expected(self) -> GroundingVerdict:
        return GroundingVerdict(self.verdict)

    def as_document(self) -> Document:
        cited = next((e.doc_id for e in self.fact.evidence), None)
        return Document(id=self.doc_id or cited or self.fact.id, text=self.text)


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #


class PairLabel(Frozen):
    """Two entity keys, and whether they name one thing.

    A row is one unordered pair; `same` is true when the keys should resolve
    to one entity. Pairwise, an unlabelled pair is unknown rather than
    different, so labels need not be exhaustive. B-cubed is stricter: it
    scores the clusters your `same` pairs imply over every key you labelled,
    and treats keys not joined by a `same` chain as distinct.

    Correct, pairwise, means the resolver's links put the two keys in one
    cluster exactly when `same` is true — through a chain of `same_as` links
    as readily as a direct one, which is how a platform's merge (every member
    linked to the survivor) scores the same as a link proposed here.
    """

    a: str
    b: str
    same: bool


class LinkRow(Frozen):
    """A link from any resolver, as a plain `(a, b, kind)` triple.

    `kind` is `same_as`, `similar` or `different`. A serialised `EntityLink`
    — a line of a sink's `links.jsonl` — is accepted as it is; so is a
    triple built from a platform's merge log. `similar` counts as no decision
    unless the evaluator is told otherwise: a resolver that said "similar"
    did not merge.
    """

    a: str
    b: str
    kind: LinkKind

    @model_validator(mode="before")
    @classmethod
    def _from_entity_link(cls, value: Any) -> Any:
        if isinstance(value, dict) and "source_key" in value:
            return {"a": value["source_key"], "b": value["target_key"], "kind": value["kind"]}
        return value


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


class CalibrationLabel(Frozen):
    """One scored fact, and whether it turned out to be true.

    A row is one fact from a slice of pipeline output, with the outcome a
    person decided. The confidence measured is `fact.confidence`; predictions
    are optional `Fact` rows joined on `Fact.id` that override it, for scoring
    one labelled slice under several scorers. No row is "correct": the report
    says how far confidence sits from frequency. The package ships this
    measurement and never the labels — calibration computed against the
    pipeline's own output measures nothing.
    """

    fact: Fact
    true: bool


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


class ValidationLabel(Frozen):
    """One fact, and what a validator should decide about it.

    A row is one fact and the action a person takes against the ontology:
    `accept` (write it), `refuse` (do not) or `conflict` (write it and flag
    that it disagrees with something held). Predictions are
    `ValidationPrediction` rows joined on `Fact.id`; a fact carries no
    validation verdict of its own, so they are required. Correct means the
    predicted action equals yours.
    """

    fact: Fact
    action: Decision


class ValidationPrediction(Frozen):
    """What a validator said about the fact with this `id`."""

    id: str
    action: Decision
    reason: str | None = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

M = TypeVar("M", bound=BaseModel)


def load_jsonl(path: str | Path, model: type[M]) -> list[M]:
    """Every non-blank line of `path` as one `model`, or an error naming the line."""
    rows: list[M] = []
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not a {model.__name__} row: {exc}") from exc
    return rows


def dump_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> None:
    """One row per line: how `run_*` output becomes a predictions file."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def cite(fact: Fact, doc_id: str) -> Fact:
    """The fact with `doc_id` as its evidence, when it cites nothing.

    Extraction is scored per document; a fact with no evidence would
    otherwise be matched against every document in the slice.
    """
    if fact.evidence:
        return fact
    return fact.model_copy(update={"evidence": (Evidence(doc_id=doc_id),)})


# The row each side of `odke eval <stage>` reads. A `Fact` predictions file
# is a sink's `facts.jsonl`.
LABEL_FORMATS: dict[str, type[BaseModel]] = {
    "route": RouteLabel,
    "extract": GoldFact,
    "ground": GroundingLabel,
    "resolve": PairLabel,
    "score": CalibrationLabel,
    "validate": ValidationLabel,
}
PREDICTION_FORMATS: dict[str, type[BaseModel]] = {
    "route": RoutePrediction,
    "extract": Fact,
    "ground": Fact,
    "resolve": LinkRow,
    "score": Fact,
    "validate": ValidationPrediction,
}


def describe(stage: str) -> str:
    """What to label for `stage` and what to predict, from the row models' docstrings."""
    if stage not in LABEL_FORMATS:
        raise ValueError(f"unknown stage {stage!r}; expected one of: {', '.join(LABEL_FORMATS)}")
    parts = []
    for role, model in (
        ("labels", LABEL_FORMATS[stage]),
        ("predictions", PREDICTION_FORMATS[stage]),
    ):
        if model is Fact:
            doc = (
                "A `Fact` per line, joined on `id`. The `facts.jsonl` a `JsonlSink` writes is one."
            )
        else:
            doc = inspect.cleandoc(model.__doc__ or "")
        fields = ", ".join(model.model_fields)
        parts.append(f"--{role}: one {model.__name__} per line ({fields})\n\n{doc}")
    return "\n\n".join(parts)


__all__ = [
    "LABEL_FORMATS",
    "PREDICTION_FORMATS",
    "Action",
    "CalibrationLabel",
    "Decision",
    "GoldFact",
    "GroundingLabel",
    "LinkRow",
    "PairLabel",
    "RouteLabel",
    "RoutePrediction",
    "ValidationLabel",
    "ValidationPrediction",
    "cite",
    "describe",
    "dump_jsonl",
    "load_jsonl",
]
