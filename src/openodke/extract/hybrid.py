"""Route each chunk by its document's modality, and merge what comes back.

The paper's hybrid design, and the whole argument for it: structured input
never costs a model call, prose always gets one, and a document that is both —
Markdown with a table among its paragraphs — gets both. The results are merged
by `Fact.signature`, so a birth date read from a table cell and the same birth
date read from the sentence above it are one candidate, not two.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from openodke._batch import incomplete, is_config_error, told
from openodke.corroborate.normalize import normalize_value
from openodke.extract._common import Documents, index_documents
from openodke.extract.pattern import PatternExtractor
from openodke.llm.budget import BudgetExceeded
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


def merge(facts: Iterable[Fact], *, ontology: Ontology | None = None) -> list[Fact]:
    """One fact per claim: the most confident, the first on a tie, in first-seen order.

    A claim is `Fact.signature` with a literal value in its canonical form
    (`normalize_value`, held to the predicate's range when `ontology` has it).
    The pattern extractor reads a cell's text, "1999" or "true", and the model
    answers in the range's type, 1999 or True: one claim, which the signature's
    `repr` alone would count as two. The fact kept keeps its value as it was read.
    """
    kept: dict[tuple[Any, ...], Fact] = {}
    for fact in facts:
        claim = _claim(fact, ontology)
        current = kept.get(claim)
        if current is None or fact.confidence > current.confidence:
            kept[claim] = fact
    return list(kept.values())


def _claim(fact: Fact, ontology: Ontology | None) -> tuple[Any, ...]:
    if fact.is_edge:
        return fact.signature
    predicate = ontology.predicates.get(fact.predicate) if ontology is not None else None
    literal = predicate.range if predicate is not None else None
    value = normalize_value(fact.object_value, range=literal)
    if literal == "boolean":
        # `normalize_value` leaves a boolean's text verbatim; True and "true" agree here.
        value = str(value).casefold()
    subject, type_name, name, _, polarity, scoped = fact.signature
    return (subject, type_name, name, repr(value), polarity, scoped)


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

    `extract_many` takes many chunks at once and hands the model path's to the
    model extractor together, so one that can batch runs its calls concurrently.
    The facts and the report are what `extract` on each chunk in turn gives.
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
        return self.extract_many([chunk], ontology)[0]

    def extract_many(self, chunks: Sequence[Chunk], ontology: Ontology) -> list[list[Fact]]:
        docs = [self._document(chunk) for chunk in chunks]
        reports = [self.report.setdefault(d.id, PathReport(modality=d.modality)) for d in docs]
        # A chunk that fails on either path fails alone (`openodke._batch`).
        failures: dict[int, Exception] = {}
        # The free path first, chunk by chunk, so its failure costs no model call.
        pattern: list[list[Fact] | None] = []
        for at, (chunk, doc) in enumerate(zip(chunks, docs, strict=True)):
            if doc.modality == "unstructured":
                pattern.append(None)
                continue
            try:
                pattern.append(_run(self.pattern, chunk, doc, ontology))
            except Exception as exc:
                if is_config_error(exc) or isinstance(exc, BudgetExceeded):
                    raise
                pattern.append(None)
                failures[at] = exc
        prose = [
            i for i, doc in enumerate(docs) if doc.modality != "structured" and i not in failures
        ]
        model: dict[int, list[Fact]] = {}
        stop: BudgetExceeded | None = None
        if self.llm is not None and prose:
            try:
                found: list[list[Fact] | None] = list(
                    self._model(
                        self.llm, [chunks[i] for i in prose], [docs[i] for i in prose], ontology
                    )
                )
            except BudgetExceeded as exc:
                # The pattern path's facts cost nothing and stand; the model
                # path keeps the chunks it finished.
                stop = exc
                partial = exc.partial
                found = (
                    list(partial)
                    if isinstance(partial, list) and len(partial) == len(prose)
                    else [None] * len(prose)
                )
                failures.update({prose[j]: e for j, e in exc.failures.items()})
            except Exception as exc:
                if is_config_error(exc) or (got := told(exc, len(prose))) is None:
                    raise
                found, theirs = got
                failures.update({prose[j]: e for j, e in theirs.items()})
            model = {i: facts for i, facts in zip(prose, found, strict=True) if facts is not None}
        out: list[list[Fact]] = []
        for i, report in enumerate(reports):
            report.chunks += 1
            candidates: list[Fact] = []
            if (by_pattern := pattern[i]) is not None:
                report.paths.add("pattern")
                report.pattern_facts += len(by_pattern)
                candidates += by_pattern
            if (by_model := model.get(i)) is not None:
                report.paths.add("llm")
                report.llm_facts += len(by_model)
                candidates += by_model
            kept = merge(candidates, ontology=ontology)
            report.merged += len(candidates) - len(kept)
            out.append(kept)
        if stop is None and not failures:
            return out
        # None for a chunk that failed, and for one with no path finished: the
        # stop reached it first.
        partial_out: list[list[Fact] | None] = [
            None
            if i in failures or (pattern[i] is None and i in prose and i not in model)
            else facts
            for i, facts in enumerate(out)
        ]
        if stop is not None:
            stop.partial = partial_out
            stop.failures = failures
            raise stop
        raise incomplete(failures, partial_out)

    def offered(self, ontology: Ontology) -> list[str] | None:
        """The predicates the model path is shown, or None when it cannot say.

        Only the model path's: the pattern path reads whatever predicate a
        column names, so it withholds nothing a coverage report could name.
        """
        offered = getattr(self.llm, "offered", None)
        if not callable(offered):
            return None
        shown = offered(ontology)
        return None if shown is None else list(shown)

    def reextract(
        self,
        window: Chunk,
        relations: Sequence[str],
        already: Sequence[Fact],
        ontology: Ontology,
    ) -> list[Fact]:
        """The model path's re-extract (#102): a gap is prose, and prose is the model's."""
        hook = getattr(self.llm, "reextract", None)
        if not callable(hook):
            raise TypeError("this HybridExtractor has no model path that can re-extract")
        return list(hook(window, list(relations), list(already), ontology))

    def _document(self, chunk: Chunk) -> Document:
        doc = self.documents.get(chunk.doc_id)
        if doc is None:
            raise LookupError(
                f"HybridExtractor has no document {chunk.doc_id!r} to route by; pass the "
                "documents given to Pipeline.run as HybridExtractor(documents=...)"
            )
        return doc

    def _model(
        self, llm: Extractor, chunks: list[Chunk], docs: list[Document], ontology: Ontology
    ) -> list[list[Fact]]:
        """The model path's facts per chunk, with each call counted on its document's report."""
        calls = getattr(llm, "calls", None)
        made = calls if isinstance(calls, list) else []
        many = getattr(llm, "extract_many", None)
        if not callable(many):
            found: list[list[Fact] | None] = []
            failed: dict[int, Exception] = {}
            for at, (chunk, doc) in enumerate(zip(chunks, docs, strict=True)):
                seen = len(made)
                try:
                    found.append(_run(llm, chunk, doc, ontology))
                except BudgetExceeded as stop:
                    stop.partial = [*found, *([None] * (len(chunks) - len(found)))]
                    stop.failures = failed
                    raise
                except Exception as exc:
                    if is_config_error(exc):
                        raise
                    found.append(None)
                    failed[at] = exc
                finally:
                    for call in made[seen:]:
                        self.report[doc.id].count_call(call)
            if failed:
                raise incomplete(failed, found)
            return [facts for facts in found if facts is not None]
        for doc in docs:
            _register(llm, doc)
        seen = len(made)
        try:
            batched = [list(facts) for facts in many(chunks, ontology)]
        finally:
            # A batch's calls are told apart by the document each one names; a
            # batch a budget stopped still made the calls it made.
            for call in made[seen:]:
                report = self.report.get(getattr(call, "doc_id", ""))
                if report is not None:
                    report.count_call(call)
        return batched

    def totals(self) -> PathReport:
        total = PathReport()
        for report in self.report.values():
            total.add(report)
        return total


def _run(extractor: Extractor, chunk: Chunk, doc: Document, ontology: Ontology) -> list[Fact]:
    _register(extractor, doc)
    return list(extractor.extract(chunk, ontology))


def _register(extractor: Extractor, doc: Document) -> None:
    lookup = getattr(extractor, "documents", None)
    if isinstance(lookup, dict):
        lookup.setdefault(doc.id, doc)


def _add_cost(total: float | None, cost: float | None) -> float | None:
    return None if total is None or cost is None else total + cost


__all__ = ["HybridExtractor", "PathReport", "merge"]
