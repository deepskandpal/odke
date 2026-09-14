"""What a proposer hands on: a candidate type or predicate, and the receipts for it.

A schema nobody can audit is a schema nobody should run. So a candidate is never
just a name: it carries the spans that produced it — document id and character
offsets, checked against the document like every other span in this package
(DECISIONS #3) — and `support`, the number of distinct documents behind it, is
computed from those spans rather than asserted.

Candidates are the proposers' vocabulary, not the ontology's. They carry what a
reviewer and the merge step need — examples, observed `(subject, value)` pairs,
which proposer found them — and `odke.infer.build` turns the survivors into
`EntityType` and `Predicate`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from odke.infer.sample import CorpusSample
from odke.ontology import Cardinality
from odke.types import Span


class TypeCandidate(BaseModel):
    """A proposed entity type.

    `examples` holds every instance name seen, uncapped: the co-occurrence
    proposer uses them to find the type in prose. `keys` is set when a record
    set showed a column that names each row.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    proposer: str
    description: str | None = None
    parents: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    evidence: tuple[Span, ...] = ()
    # The model's ordering, when a model named this; a tie-break after support.
    rank: int | None = None

    @property
    def support(self) -> int:
        return len({span.doc_id for span in self.evidence})


class PredicateCandidate(BaseModel):
    """A proposed predicate, with the `(subject, value)` pairs it was seen holding.

    `observations` are folded names, and they are what lets the merge step see
    that `works_at` and `employer` were observed on the same people and the same
    companies — shared evidence, where the names share nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    proposer: str
    description: str | None = None
    domain: tuple[str, ...] = ()
    range: str = "string"
    cardinality: Cardinality = "single"
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    observations: tuple[tuple[str, str], ...] = ()
    evidence: tuple[Span, ...] = ()
    rank: int | None = None

    @property
    def support(self) -> int:
        return len({span.doc_id for span in self.evidence})


class Proposals(BaseModel):
    """Everything proposed so far, in the order it was proposed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    types: tuple[TypeCandidate, ...] = ()
    predicates: tuple[PredicateCandidate, ...] = ()

    def __add__(self, other: Proposals) -> Proposals:
        return Proposals(
            types=self.types + other.types, predicates=self.predicates + other.predicates
        )


@runtime_checkable
class Proposer(Protocol):
    """Reads a sample, and what the proposers before it found, and proposes more.

    Proposers run in order, and each sees the sum of what came before: the
    co-occurrence proposer looks for the instances the record and Hearst
    proposers found, and the model proposer names all of it.
    """

    name: str

    def propose(self, sample: CorpusSample, found: Proposals) -> Proposals: ...


__all__ = ["PredicateCandidate", "Proposals", "Proposer", "TypeCandidate"]
