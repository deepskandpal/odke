"""The free half of grounding: does the cited span exist, and does it say what it claims?

DECISIONS #3: spans are offsets, not quotes, precisely so that this check costs
nothing. A model can invent a quote that reads well and appears nowhere in the
source; an offset either resolves to the claimed text or it does not. This runs
before any model is asked, every time — it is the cheapest possible rejection,
and a high rejection rate here is a prompt problem rather than a model problem.

It answers only the *locatability* question. Whether a span that does exist
actually supports the claim is the second question, and the model's
(`odke.ground.llm`). Nothing here can say `SUPPORTED`.
"""

from __future__ import annotations

import logging
import threading
from enum import StrEnum
from typing import Any

from odke.types import Document, Evidence, Fact, GroundingVerdict

log = logging.getLogger("odke.ground")


class SpanStatus(StrEnum):
    """What one piece of evidence turned out to be, checked against its document."""

    # The span lies in this document and resolves to its quote (or has none).
    LOCATED = "located"
    # Nothing to check: a document-level citation, or another document's span.
    # Neither is wrong, and neither locates the claim here.
    NO_SPAN = "no_span"
    FOREIGN = "foreign"
    # Rejections. The evidence claims something the document cannot back.
    OUT_OF_RANGE = "out_of_range"
    EMPTY = "empty"
    QUOTE_MISMATCH = "quote_mismatch"

    @property
    def rejected(self) -> bool:
        return self in _REJECTED


_REJECTED = frozenset({SpanStatus.OUT_OF_RANGE, SpanStatus.EMPTY, SpanStatus.QUOTE_MISMATCH})


def check_span(evidence: Evidence, doc: Document) -> SpanStatus:
    """Classify one piece of evidence against the document it should cite.

    Range is checked before the quote: Python slicing truncates silently, so a
    span running past the end of the text can still "resolve" to its quote
    while citing offsets that do not exist.
    """
    span = evidence.span
    if span is None:
        return SpanStatus.NO_SPAN
    if span.doc_id != doc.id:
        return SpanStatus.FOREIGN
    if span.start < 0 or span.end > len(doc.text) or span.start > span.end:
        return SpanStatus.OUT_OF_RANGE
    if span.start == span.end:
        return SpanStatus.EMPTY
    if not span.is_faithful(doc):
        return SpanStatus.QUOTE_MISMATCH
    return SpanStatus.LOCATED


def located(fact: Fact, doc: Document) -> Evidence | None:
    """The first evidence on `fact` that really cites `doc` — what a model is shown."""
    return next((e for e in fact.evidence if check_span(e, doc) is SpanStatus.LOCATED), None)


class Counts:
    """Integer counters safe to bump from several threads.

    A grounder's counts are the run report for this stage: the pipeline's own
    stats do not see inside a stage, so each grounder keeps its own and exposes a
    snapshot. Under the GIL `d[k] += 1` is still a read-modify-write, and the
    batched grounder bumps from a thread pool.
    """

    def __init__(self, *keys: str) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = dict.fromkeys(keys, 0)

    def bump(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + by

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


class SpanGrounder:
    """Rejects evidence whose span does not resolve, and stamps `NOT_FOUND` when
    nothing survives. Pure: no model, no network, no configuration.

    Per fact: every evidence span is checked with `check_span`. A rejected span
    is dropped from the fact — an offset that does not say what it claims is not
    provenance, and carrying it to the sink would be a lie in the graph — and
    the reason is logged at DEBUG and counted in `stats`. A span-less citation
    or another document's span is kept as it is, since nothing showed it wrong,
    but it does not locate the claim in *this* document. A fact with no located
    evidence is `NOT_FOUND`; one with at least one is left `UNCHECKED` for the
    model, which is the only thing that can say `SUPPORTED`.

    A verdict already on the fact stands. That is what lets an interrupted run
    resume over a partially grounded set, and it costs nothing here.
    """

    def __init__(self) -> None:
        self._counts = Counts("facts", "rejected", "not_found", *(s.value for s in SpanStatus))

    def ground(self, fact: Fact, doc: Document) -> Fact:
        if fact.verdict is not GroundingVerdict.UNCHECKED:
            return fact
        self._counts.bump("facts")
        kept: list[Evidence] = []
        found = False
        for evidence in fact.evidence:
            status = check_span(evidence, doc)
            self._counts.bump(status.value)
            if status.rejected:
                self._counts.bump("rejected")
                _log_rejection(fact, evidence, status)
                continue
            kept.append(evidence)
            found = found or status is SpanStatus.LOCATED
        if found:
            if len(kept) == len(fact.evidence):
                return fact
            return fact.model_copy(update={"evidence": tuple(kept)})
        self._counts.bump("not_found")
        return fact.model_copy(
            update={"evidence": tuple(kept), "verdict": GroundingVerdict.NOT_FOUND}
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Counts so far: facts seen, each `SpanStatus`, rejections, and `NOT_FOUND`s."""
        return self._counts.snapshot()


def _log_rejection(fact: Fact, evidence: Evidence, status: SpanStatus) -> None:
    span = evidence.span
    assert span is not None  # a rejection always has one
    log.debug(
        "rejected span %s[%d:%d] on fact %s (%s %s): %s",
        span.doc_id,
        span.start,
        span.end,
        fact.id,
        fact.subject.key,
        fact.predicate,
        status.value,
    )


__all__ = ["Counts", "SpanGrounder", "SpanStatus", "check_span", "located"]
