"""Route each chunk by its document's modality, and merge what comes back.

The paper's hybrid design, and the whole argument for it: structured input
never costs a model call, prose always gets one, and a document that is both —
Markdown with a table among its paragraphs — gets both. The results are merged
by `Fact.signature`, so a birth date read from a table cell and the same birth
date read from the sentence above it are one candidate, not two.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from openodke.extract._common import Documents, index_documents
from openodke.extract.pattern import PatternExtractor
from openodke.ontology import Ontology
from openodke.stages import Extractor
from openodke.types import Chunk, Document, Fact


@dataclass
class PathReport:
    """Which paths ran for a document and what they cost: the hybrid's own evidence."""

    modality: str | None = None
    paths: set[str] = field(default_factory=set)
    chunks: int = 0
    pattern_facts: int = 0
    llm_facts: int = 0
    # Candidates collapsed into another with the same signature.
    merged: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # None once any call's cost went unreported: a partial sum would pass for the total.
    cost_usd: float | None = 0.0

    def count_call(self, call: Any) -> None:
        self.model_calls += 1
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens
        self.cost_usd = _add_cost(self.cost_usd, call.cost_usd)

    def add(self, other: PathReport) -> None:
        self.paths |= other.paths
        self.chunks += other.chunks
        self.pattern_facts += other.pattern_facts
        self.llm_facts += other.llm_facts
        self.merged += other.merged
        self.model_calls += other.model_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd = _add_cost(self.cost_usd, other.cost_usd)


def merge(facts: Iterable[Fact]) -> list[Fact]:
    """One fact per signature: the most confident, the first on a tie, in first-seen order."""
    kept: dict[tuple[Any, ...], Fact] = {}
    for fact in facts:
        current = kept.get(fact.signature)
        if current is None or fact.confidence > current.confidence:
            kept[fact.signature] = fact
    return list(kept.values())


class HybridExtractor:
    """Pattern for structure, the model for prose, both for a mix; merged by signature.

    `structured` goes to the pattern extractor and never costs a model call.
    `unstructured` goes to the model. `semi_structured` goes to both, and where
    they found the same claim the more confident fact is kept — the pattern
    extractor's, at its default confidence of 1.0.

    The document lookup is required here, unlike for the other extractors:
    routing on modality is the whole job, and a guessed modality would either
    spend model calls on a CSV or skip a page of prose. Each document is handed
    on to the sub-extractors, so their evidence carries its URI and tier. With
    no `llm`, prose yields nothing and a mixed document gets the pattern path.

    `report` holds a `PathReport` per document id, and `totals()` sums them.
    """

    name = "hybrid"

    def __init__(
        self,
        llm: Extractor | None = None,
        *,
        pattern: Extractor | None = None,
        documents: Documents = None,
    ) -> None:
        self.llm = llm
        self.pattern: Extractor = pattern if pattern is not None else PatternExtractor()
        self.documents = index_documents(documents)
        self.report: dict[str, PathReport] = {}

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        doc = self.documents.get(chunk.doc_id)
        if doc is None:
            raise LookupError(
                f"HybridExtractor has no document {chunk.doc_id!r} to route by; pass the "
                "documents given to Pipeline.run as HybridExtractor(documents=...)"
            )
        report = self.report.setdefault(doc.id, PathReport(modality=doc.modality))
        report.chunks += 1
        candidates: list[Fact] = []
        if doc.modality != "unstructured":
            found = _run(self.pattern, chunk, doc, ontology)
            report.paths.add("pattern")
            report.pattern_facts += len(found)
            candidates += found
        if doc.modality != "structured" and self.llm is not None:
            calls = getattr(self.llm, "calls", None)
            seen = len(calls) if isinstance(calls, list) else 0
            found = _run(self.llm, chunk, doc, ontology)
            report.paths.add("llm")
            report.llm_facts += len(found)
            for call in calls[seen:] if isinstance(calls, list) else ():
                report.count_call(call)
            candidates += found
        kept = merge(candidates)
        report.merged += len(candidates) - len(kept)
        return kept

    def totals(self) -> PathReport:
        total = PathReport()
        for report in self.report.values():
            total.add(report)
        return total


def _run(extractor: Extractor, chunk: Chunk, doc: Document, ontology: Ontology) -> list[Fact]:
    lookup = getattr(extractor, "documents", None)
    if isinstance(lookup, dict):
        lookup.setdefault(doc.id, doc)
    return list(extractor.extract(chunk, ontology))


def _add_cost(total: float | None, cost: float | None) -> float | None:
    return None if total is None or cost is None else total + cost


__all__ = ["HybridExtractor", "PathReport", "merge"]
