"""Scoring an Extractor: per-predicate P/R/F1 against gold facts per document.

Never only an aggregate. A corpus-wide F1 of 0.8 is what twelve predicates
that work and three that never do look like, and the three are where the next
day of work should go. So the breakdown is per predicate, and every miss is
sorted into one of the four ways extraction goes wrong:

- **wrong value** — the right subject and predicate, a different literal;
- **wrong entity** — the right claim pinned to the wrong node, as subject or
  as object;
- **missing** — a gold fact no prediction came near;
- **spurious** — a prediction no gold fact came near.

A wrong value or entity counts as a false positive *and* a false negative: a
wrong fact was written and the right one was not.

Matching is by `Fact.signature` — subject, predicate, object, polarity and
identity-bearing qualifiers (DECISIONS #11, #14, #15) — with literal values
normalised first, because `1815` from a pattern extractor and `"1815"` from a
model are one value and scoring them apart would measure serialisation. A
flipped polarity is never a near miss: a denial is the opposite claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from odke.eval.formats import GoldFact, cite
from odke.eval.report import Metric, StageReport, macro_f1, prf
from odke.ontology import Ontology
from odke.stages import Chunker, Extractor, PassThroughChunker
from odke.types import Document, Fact, Frozen

Kind = Literal["correct", "wrong_value", "wrong_entity", "missing", "spurious"]
ERRORS: tuple[Kind, ...] = ("wrong_value", "wrong_entity", "missing", "spurious")


class Outcome(Frozen):
    """One gold fact, one prediction, or a pair of them, and what the pairing means."""

    kind: Kind
    predicate: str
    doc_id: str | None = None
    gold: Fact | None = None
    predicted: Fact | None = None


def normalise_value(value: Any) -> str:
    """A literal as the matcher compares it: whitespace collapsed, case folded,
    and numbers — numeric strings included — in one spelling."""
    text = " ".join(str(value).split()).casefold()
    if isinstance(value, bool):
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    return str(int(number)) if number.is_integer() else repr(number)


def run_extract(
    extractor: Extractor,
    documents: Iterable[Document],
    ontology: Ontology,
    *,
    chunker: Chunker | None = None,
) -> list[Fact]:
    """Extract from every document in-process, each fact citing its document.

    No router, grounder or validator: this measures the extractor alone, which
    is the baseline every later stage is an ablation against.
    """
    chunker = chunker or PassThroughChunker()
    return [
        cite(fact, doc.id)
        for doc in documents
        for chunk in chunker.chunk(doc)
        for fact in extractor.extract(chunk, ontology)
    ]


def match_extraction(gold: Sequence[GoldFact], predictions: Iterable[Fact]) -> list[Outcome]:
    """Pair predictions with gold facts, document by document.

    A prediction is scored in the first labelled document its evidence cites.
    One citing none is matched, after every document has been, against the
    gold facts still unmatched anywhere; one citing only unlabelled documents
    is not scored. Within a scope, pairings go exact match first, then same
    subject, then same claim about another subject, each in file order, so
    the result does not depend on how a dict happened to iterate.
    """
    docs = list(dict.fromkeys(g.doc_id for g in gold))
    gold_by_doc: dict[str, list[Fact]] = {d: [] for d in docs}
    for g in gold:
        gold_by_doc[g.doc_id].append(g.fact)
    cited: dict[str, list[Fact]] = {d: [] for d in docs}
    uncited: list[Fact] = []
    for fact in predictions:
        cites = [e.doc_id for e in fact.evidence]
        if not cites:
            uncited.append(fact)
        elif (home := next((d for d in cites if d in gold_by_doc), None)) is not None:
            cited[home].append(fact)

    outcomes: list[Outcome] = []
    leftover: list[tuple[str, Fact]] = []
    for doc_id in docs:
        gold_left, predicted_left = _pair(gold_by_doc[doc_id], cited[doc_id], doc_id, outcomes)
        leftover.extend((doc_id, g) for g in gold_left)
        outcomes.extend(
            Outcome(kind="spurious", predicate=p.predicate, doc_id=doc_id, predicted=p)
            for p in predicted_left
        )
    homes = {id(g): d for d, g in leftover}
    gold_left, predicted_left = _pair([g for _, g in leftover], uncited, None, outcomes, homes)
    outcomes.extend(
        Outcome(kind="missing", predicate=g.predicate, doc_id=homes[id(g)], gold=g)
        for g in gold_left
    )
    outcomes.extend(
        Outcome(kind="spurious", predicate=p.predicate, predicted=p) for p in predicted_left
    )
    return outcomes


def evaluate_extraction(gold: Sequence[GoldFact], predictions: Iterable[Fact]) -> StageReport:
    """Per-predicate P/R/F1 with the four-way error split, and the totals."""
    predictions = list(predictions)
    labelled = {g.doc_id for g in gold}
    unscored = sum(
        1 for p in predictions if p.evidence and not any(e.doc_id in labelled for e in p.evidence)
    )
    outcomes = match_extraction(gold, predictions)

    counts: dict[str, dict[str, int]] = {}
    for o in outcomes:
        row = counts.setdefault(o.predicate, {k: 0 for k in ("correct", *ERRORS)})
        row[o.kind] += 1
    breakdown = {name: _row(counts[name]) for name in sorted(counts)}
    totals: dict[str, int] = {k: sum(c[k] for c in counts.values()) for k in ("correct", *ERRORS)}
    overall = _row(totals)

    notes = []
    dead = [name for name, row in breakdown.items() if row["f1"] == 0.0]
    if dead:
        notes.append(f"F1 is zero for: {', '.join(dead)}")
    if unscored:
        notes.append(
            f"{unscored} prediction(s) cite only documents with no gold facts and were not scored"
        )
    metrics: dict[str, Metric] = {
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "macro_f1": macro_f1(breakdown),
        **{k: overall[k] for k in ("tp", "fp", "fn", *ERRORS)},
        "predicates": len(breakdown),
    }
    return StageReport(
        stage="extract", n=len(gold), metrics=metrics, breakdown=breakdown, notes=tuple(notes)
    )


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def _object(fact: Fact) -> tuple[str, str]:
    if fact.object_entity is not None:
        return ("entity", fact.object_entity.key)
    return ("value", normalise_value(fact.object_value))


def _scoped(fact: Fact) -> tuple[tuple[str, str], ...]:
    # The identity-bearing qualifiers `signature` includes, with values normalised.
    present = (k for k in fact.identity_keys if k in fact.qualifiers)
    return tuple(sorted((k, normalise_value(fact.qualifiers[k])) for k in present))


def _exact(f: Fact) -> tuple[Any, ...]:
    return (f.subject.key, f.subject.type, f.predicate, _object(f), f.polarity, _scoped(f))


def _same_subject(f: Fact) -> tuple[Any, ...]:
    return (f.subject.key, f.predicate, f.polarity, _scoped(f))


def _same_claim(f: Fact) -> tuple[Any, ...]:
    return (f.predicate, _object(f), f.polarity, _scoped(f))


def _pair(
    gold: list[Fact],
    predicted: list[Fact],
    doc_id: str | None,
    outcomes: list[Outcome],
    homes: dict[int, str] | None = None,
) -> tuple[list[Fact], list[Fact]]:
    """Consume matching pairs in three passes; return what is left on each side."""
    passes = (_exact, _same_subject, _same_claim)
    for key in passes:
        still: list[Fact] = []
        for g in gold:
            wanted = key(g)
            hit = next((i for i, p in enumerate(predicted) if key(p) == wanted), None)
            if hit is None:
                still.append(g)
                continue
            p = predicted.pop(hit)
            outcomes.append(
                Outcome(
                    kind=_classify(g, p, key is _exact),
                    predicate=g.predicate,
                    doc_id=doc_id if homes is None else homes[id(g)],
                    gold=g,
                    predicted=p,
                )
            )
        gold = still
    return gold, predicted


def _classify(gold: Fact, predicted: Fact, exact: bool) -> Kind:
    if exact:
        return "correct"
    both_literal = gold.object_entity is None and predicted.object_entity is None
    if (
        gold.subject.key == predicted.subject.key
        and both_literal
        and _object(gold) != _object(predicted)
    ):
        return "wrong_value"
    # A different object node, a different subject, or the right key typed wrong.
    return "wrong_entity"


def _row(c: dict[str, int]) -> dict[str, Metric]:
    wrong = c["wrong_value"] + c["wrong_entity"]
    return {
        **prf(c["correct"], wrong + c["spurious"], wrong + c["missing"]),
        **{k: c[k] for k in ERRORS},
        "support": c["correct"] + wrong + c["missing"],
    }


__all__ = [
    "ERRORS",
    "Outcome",
    "evaluate_extraction",
    "match_extraction",
    "normalise_value",
    "run_extract",
]
