"""The free checks, run before a model is asked anything about a fact.

Three, and none costs a token:

- **the mention is in the text**: the span check (`SpanGrounder`), which every
  grounder here already runs first;
- **the relation is in the ontology**, when there is one;
- **the types fit**: the subject's type is in the predicate's domain, and the
  object is in its range: an entity of the range type for an edge, a value for
  a property.

`schema_problem` is the second and third. `CheckedGrounder` runs it ahead of a
grounder, so a fact the ontology has no room for never costs a call. It is
stamped with why, under `odke.check`, and keeps the verdict it had, because
nothing read the text about it. With no grounder, `CheckedGrounder` is the free
checks alone, plus the span locator on request: grounding as a dry run, at no
cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openodke.corroborate.provenance import CHECK
from openodke.ground.locate import SpanLocator
from openodke.ground.span import Counts, SpanGrounder
from openodke.ontology import Ontology
from openodke.stages import Grounder
from openodke.types import Document, Fact, GroundingVerdict

# What `schema_problem` checks, in the order it checks them.
CHECKS = ("predicate", "domain", "range")


def schema_problem(fact: Fact, ontology: Ontology) -> tuple[str, str] | None:
    """Why the ontology has no room for `fact`, as `(check, reason)`, or None.

    `check` is one of `CHECKS`. An ontology that declares no predicates says
    nothing about any fact, so it finds no problem. A type fits when it is the
    declared type or a subtype of it.
    """
    if not ontology.predicates:
        return None
    predicate = ontology.predicates.get(fact.predicate)
    if predicate is None:
        return "predicate", f"{fact.predicate!r} is not a predicate of ontology {ontology.name!r}"
    subject = fact.subject.type
    if predicate.domain and not ontology.lineage(subject) & set(predicate.domain):
        return "domain", (
            f"subject type {subject!r} is not in the domain of {predicate.name!r} "
            f"({', '.join(predicate.domain)})"
        )
    obj = fact.object_entity
    if predicate.is_edge_in(ontology):
        if obj is None:
            return "range", f"{predicate.name!r} is an edge to {predicate.range}, not a value"
        if predicate.range not in ontology.lineage(obj.type):
            return "range", (
                f"object type {obj.type!r} is not the range of {predicate.name!r} "
                f"({predicate.range})"
            )
    elif obj is not None:
        return "range", f"{predicate.name!r} holds a value ({predicate.range}), not an entity"
    return None


class CheckedGrounder:
    """The free checks first, then a grounder: a fact they refuse is never asked about.

    Per fact, `schema_problem` runs against `ontology`. A fact it refuses comes
    back with `qualifiers["odke.check"] = {"check": ..., "reason": ...}` and
    its verdict unchanged, and the grounder never sees it. The rest go to
    `grounder`, batched as the pipeline would batch them; an `LLMGrounder`
    then runs the span check, the locator if it has one, and the model.

    With `grounder=None` it is the free checks alone: the span check, then
    `SpanLocator` when `locate=True`. Nothing is called, and a fact the span
    check leaves standing stays `UNCHECKED`. With a grounder, the grounder
    locates: `LLMGrounder(locate=True)`.

    `stats` is the grounder's own report, with the schema check's counts
    under `"checks"`. With no grounder, the span check's counts are under
    `"span"` and the locator's under `"locate"`, where `LLMGrounder` keeps them.
    """

    def __init__(
        self, grounder: Grounder | None = None, *, ontology: Ontology, locate: bool = False
    ) -> None:
        if grounder is not None and locate:
            raise ValueError("with a grounder, the grounder locates: pass LLMGrounder(locate=True)")
        self.grounder = grounder
        self.ontology = ontology
        self.span_grounder = SpanGrounder() if grounder is None else None
        self.locator = SpanLocator() if locate else None
        self._counts = Counts("facts", "refused", *CHECKS)

    def ground(self, fact: Fact, doc: Document) -> Fact:
        return self.ground_documents([([fact], doc)])[0][0]

    def ground_many(self, facts: Sequence[Fact], doc: Document) -> list[Fact]:
        return self.ground_documents([(facts, doc)])[0]

    def ground_documents(
        self, batches: Sequence[tuple[Sequence[Fact], Document]]
    ) -> list[list[Fact]]:
        """Every document's facts checked, and the ones that pass grounded in one batch."""
        out: list[list[Fact]] = []
        passed: list[list[int]] = []
        for facts, _ in batches:
            row = [self._check(fact) for fact in facts]
            out.append(row)
            passed.append(
                [i for i, (fact, kept) in enumerate(zip(facts, row, strict=True)) if kept is fact]
            )
        work = [
            ([row[i] for i in keep], doc)
            for row, keep, (_, doc) in zip(out, passed, batches, strict=True)
        ]
        for row, keep, grounded in zip(out, passed, self._ground(work), strict=True):
            for index, fact in zip(keep, grounded, strict=True):
                row[index] = fact
        return out

    def _check(self, fact: Fact) -> Fact:
        self._counts.bump("facts")
        problem = schema_problem(fact, self.ontology)
        if problem is None:
            return fact
        check, reason = problem
        self._counts.bump("refused")
        self._counts.bump(check)
        stamp = {"check": check, "reason": reason}
        return fact.model_copy(update={"qualifiers": {**fact.qualifiers, CHECK: stamp}})

    def _ground(self, work: list[tuple[list[Fact], Document]]) -> list[list[Fact]]:
        if self.grounder is not None:
            # Here, not at the top: the pipeline imports this package's locator.
            from openodke.pipeline import _ground

            return _ground(self.grounder, work)
        assert self.span_grounder is not None  # set whenever there is no grounder
        out = []
        for facts, doc in work:
            row = []
            for fact in facts:
                checked = self.span_grounder.ground(fact, doc)
                if self.locator is not None and checked.verdict is GroundingVerdict.UNCHECKED:
                    checked = self.locator.locate(checked, doc)
                row.append(checked)
            out.append(row)
        return out

    @property
    def stats(self) -> dict[str, Any]:
        """The grounder's report, and the schema check's counts under `"checks"`."""
        own: dict[str, Any] = {"checks": self._counts.snapshot()}
        if self.span_grounder is not None:
            own["span"] = self.span_grounder.stats
        if self.locator is not None:
            own["locate"] = self.locator.stats
        inner = getattr(self.grounder, "stats", None)
        return {**(dict(inner) if isinstance(inner, Mapping) else {}), **own}


__all__ = ["CHECKS", "CheckedGrounder", "schema_problem"]
