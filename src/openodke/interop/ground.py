"""A graph openodke did not build, grounded in one call, and what that says (#99).

`ground_graph` is the library half of `odke ground`. It takes any adapter's
`(rows, documents)` and grounds every fact in three steps, cheapest first:

1. the free checks (`CheckedGrounder`): the mention is in the text, and, with
   an ontology, the relation is one of its predicates and the types fit its
   domain and range;
2. the span locator, on request, for facts that cited nothing;
3. the model, unless no grounder is given: a dry run stops after step 2.

It returns the same facts with their verdicts, grouped by text in the order the
rows gave them, and a `GroundSummary`: counts per verdict, what the free checks
refused, the facts with no span of their own, the `odke eval spans` width
split, and the two ways a fact fails the model, named plainly:

- **the evidence does not support the fact**: the model read the passage, and
  it says otherwise or does not say it;
- **the citation is too narrow for its claim**: the model read a citation that
  does not name both the subject and the object, so it could not state the
  claim (DECISIONS #23). The fact may well be true; its citation lost it.

Only a `not_found` the model gave a citation can be the second. A grounder that
reads the whole document (`context="document"`, the paper's mode) never reads
the citation, so its refusals are all the first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import Field

from openodke._batch import failed_summary
from openodke.eval.report import StageReport
from openodke.eval.spans import evaluate_spans
from openodke.ground.checks import REASONS, VERDICTS, CheckedGrounder, refusals
from openodke.ground.locate import locate_span
from openodke.interop.triples import TripleRow, TriplesExtractor, read_triples
from openodke.llm.budget import stopped_summary
from openodke.manifest import Recorder, RunManifest, inputs_of, spent_of
from openodke.observe import JobCounts, Observer
from openodke.ontology import Ontology
from openodke.pipeline import Pipeline
from openodke.stages import Grounder
from openodke.types import Document, Fact, Frozen, GroundingVerdict, SpanOrigin

FACTS_FILE = "facts.jsonl"
SUMMARY_FILE = "summary.json"
# Example facts a failure shape lists in the summary; its `ids` list them all.
EXAMPLES = 5

UNSUPPORTED = "the evidence does not support the fact"
TOO_NARROW = "the citation is too narrow for its claim"


class FailureShape(Frozen):
    """One way facts failed the model: how many, which, and the first few as lines."""

    count: int = 0
    # Every fact of this shape, by id, to find it in facts.jsonl.
    ids: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


class GroundSummary(Frozen):
    """What one `ground_graph` call found, as `odke ground` prints it and writes it."""

    dry_run: bool
    rows: int
    facts: int
    # Rows whose `doc` named no text that was given: never grounded, not in `facts`.
    unmatched_rows: int = 0
    verdicts: dict[str, int] = Field(default_factory=dict)
    # What the free checks refused: `not_in_text` (no citation resolves, so
    # `not_found` for free), then each of `CHECKS` (refused with `odke.check`).
    refused: dict[str, int] = Field(default_factory=dict)
    # Facts that cite no span of their own, and those the locator gave one.
    no_span: int = 0
    located: int = 0
    calls: int = 0
    # Of `calls`, those a response cache answered (`openodke.llm.cache`).
    cached: int = 0
    tokens: int = 0
    cost_usd: float | None = None
    prompts: tuple[str, ...] = ()
    unsupported: FailureShape = FailureShape()
    too_narrow: FailureShape = FailureShape()
    # `odke eval spans` over the same facts.
    spans: StageReport
    # Where a budget stopped the grounding, and why: the pipeline's `stats["stopped"]`.
    stopped: dict[str, Any] | None = None
    # Texts left out because grounding failed for them alone, with why (#162).
    failed: dict[str, str] = Field(default_factory=dict)

    @property
    def job(self) -> JobCounts:
        """Rows in, facts out, and what the free checks refused: nothing is dropped or merged."""
        return JobCounts(
            facts_in=self.rows, facts_out=self.facts, refused=sum(self.refused.values())
        )

    def render(self) -> str:
        lines = ["odke ground — dry run, no model called" if self.dry_run else "odke ground"]
        if self.stopped is not None:
            lines.append(_row("stopped", stopped_summary(self.stopped)))
        if self.failed:
            lines.append(_row("failed", failed_summary(self.failed, noun="text")))
        unmatched = (
            f"; {self.unmatched_rows} name a text that was not given" if self.unmatched_rows else ""
        )
        lines.append(_row("rows", f"{self.rows} ({self.facts} grounded{unmatched})"))
        lines.append(_row("verdicts", ", ".join(f"{k} {v}" for k, v in self.verdicts.items())))
        refused = sum(self.refused.values())
        reasons = ", ".join(f"{v} {REASONS[k]}" for k, v in self.refused.items() if v)
        lines.append(
            _row("free checks", f"{refused} refused" + (f": {reasons}" if reasons else ""))
        )
        located = f"; {self.located} given one by the locator" if self.located else ""
        lines.append(
            _row("no span", f"{self.no_span} of {self.facts} cite no span of their own{located}")
        )
        if self.dry_run:
            lines.append(_row("model", "not asked: a dry run calls nothing"))
        else:
            usd = f"${self.cost_usd:.4f}" if self.cost_usd is not None else "USD unknown"
            prompts = f" ({', '.join(self.prompts)})" if self.prompts else ""
            cached = f" ({self.cached} from the cache)" if self.cached else ""
            lines.append(
                _row("model", f"{self.calls} calls{cached}, {self.tokens} tokens, {usd}{prompts}")
            )
        lines.append("failures")
        for title, shape in ((UNSUPPORTED, self.unsupported), (TOO_NARROW, self.too_narrow)):
            count = "unknown without the model" if self.dry_run else str(shape.count)
            lines.append(f"  {title}: {count}")
            lines.extend(f"    {example}" for example in shape.examples)
            if shape.count > len(shape.examples):
                lines.append(f"    … and {shape.count - len(shape.examples)} more")
        lines += ["", self.spans.render()]
        return "\n".join(lines)


class GroundedGraph(NamedTuple):
    """The facts with their verdicts, and what they add up to."""

    facts: list[Fact]
    summary: GroundSummary

    def write(self, directory: str | Path) -> list[Path]:
        """`facts.jsonl` (what `odke eval spans --facts` reads) and `summary.json`."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        facts, summary = out / FACTS_FILE, out / SUMMARY_FILE
        with facts.open("w", encoding="utf-8") as fh:
            for fact in self.facts:
                fh.write(fact.model_dump_json() + "\n")
        summary.write_text(self.summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return [facts, summary]


def ground_graph(
    rows: str | Path | Iterable[TripleRow | Mapping[str, Any]],
    documents: Iterable[Document] = (),
    *,
    ontology: Ontology | None = None,
    grounder: Grounder | None = None,
    locate: bool = False,
    extractor: str = "triples",
    observer: Observer | None = None,
) -> GroundedGraph:
    """Every row's fact through the free checks, the locator and `grounder`, and a summary.

    `rows` and `documents` are an adapter's output, or a triples file and the
    texts it cites. `grounder=None` is a dry run: the free checks, and the
    locator when `locate=True`, and no model. With a grounder, the grounder
    locates (`LLMGrounder(locate=True)`). Nothing is resolved, merged, derived
    or dropped: one fact comes back for each row whose text was given.
    `observer` is the job's, whose events and spans each stage reports to.
    """
    docs = list(documents)
    read = list(read_triples(rows))
    schema = ontology if ontology is not None else Ontology()
    stage = TriplesExtractor(read, extractor=extractor, documents=docs)
    checked = CheckedGrounder(grounder, ontology=schema, locate=locate)
    kg = Pipeline(schema, stage, grounder=checked, inverses=False, observer=observer).run(docs)
    facts = list(kg.facts)
    reads_span = getattr(grounder, "context", "span") == "span"
    summary = summarize(
        facts,
        docs,
        rows=len(read),
        unmatched=int(stage.stats.get("unmatched_rows", 0)),
        grounding=checked.stats,
        dry_run=grounder is None,
        reads_span=reads_span,
    )
    if isinstance(stopped := kg.stats.get("stopped"), Mapping):
        summary = summary.model_copy(update={"stopped": dict(stopped)})
    if isinstance(failed := kg.stats.get("failed"), Mapping) and failed:
        summary = summary.model_copy(update={"failed": dict(failed)})
    return GroundedGraph(facts, summary)


def ground_manifest(
    recorder: Recorder,
    grounded: GroundedGraph,
    rows: Sequence[Any],
    documents: Iterable[Document],
    *,
    ontology: Ontology | None = None,
    grounder: Grounder | None = None,
    run: str | None = None,
) -> RunManifest:
    """The run manifest of one `ground_graph` call, finished now (#160).

    `rows` and `documents` are what was grounded, and `grounder` the model
    grounder that was asked, whose spend the manifest keeps. `run` is the id
    the job's log events carry. `odke ground` writes it beside `facts.jsonl`
    and `summary.json`.
    """
    summary = grounded.summary
    counts = {
        "rows": summary.rows,
        "facts": summary.facts,
        "unmatched_rows": summary.unmatched_rows,
        "refused": sum(summary.refused.values()),
        **summary.verdicts,
        "no_span": summary.no_span,
        "located": summary.located,
        "unsupported": summary.unsupported.count,
        "too_narrow": summary.too_narrow.count,
    }
    stats = getattr(grounder, "stats", None)
    return recorder.finish(
        inputs=inputs_of(documents, rows),
        ontology=ontology,
        prompts=summary.prompts,
        counts=counts,
        spent=spent_of(stats if isinstance(stats, Mapping) else {}),
        stopped=summary.stopped,
        failed=summary.failed,
        dry_run=summary.dry_run,
        run=run,
        job=summary.job.model_dump(),
    )


def summarize(
    facts: Iterable[Fact],
    documents: Iterable[Document],
    *,
    rows: int | None = None,
    unmatched: int = 0,
    grounding: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    reads_span: bool = True,
) -> GroundSummary:
    """The summary of grounded facts. `grounding` is the grounder's `stats`."""
    facts = list(facts)
    texts = {doc.id: doc for doc in documents}
    stats = dict(grounding or {})
    spans = evaluate_spans(facts)
    verdicts = {v.value: 0 for v in VERDICTS}
    for fact in facts:
        verdicts[fact.verdict.value] += 1
    refused = refusals(stats)
    unsupported: list[Fact] = []
    too_narrow: list[Fact] = []
    for fact in facts:
        if dry_run or fact.verdict not in (
            GroundingVerdict.NOT_FOUND,
            GroundingVerdict.CONTRADICTED,
        ):
            continue
        shown = next((e for e in fact.evidence if e.span is not None), None)
        if shown is None or shown.span is None:
            continue  # refused by the free span check: no citation resolved
        doc = texts.get(shown.doc_id)
        read_citation = (
            reads_span
            and shown.span_origin is SpanOrigin.CITED
            and fact.verdict is GroundingVerdict.NOT_FOUND
        )
        if read_citation and doc is not None and not _names_both(fact, shown.span.resolve(doc)):
            too_narrow.append(fact)
        else:
            unsupported.append(fact)
    tokens = int(stats.get("prompt_tokens", 0)) + int(stats.get("completion_tokens", 0))
    cost = stats.get("cost_usd")
    return GroundSummary(
        dry_run=dry_run,
        rows=len(facts) + unmatched if rows is None else rows,
        facts=len(facts),
        unmatched_rows=unmatched,
        verdicts=verdicts,
        refused=refused,
        no_span=int(spans.metrics.get("no_span") or 0),
        located=int(spans.metrics.get("located") or 0),
        calls=int(stats.get("calls", 0)),
        cached=int(stats.get("cached", 0)),
        tokens=tokens,
        cost_usd=float(cost) if isinstance(cost, int | float) else None,
        prompts=tuple(str(p) for p in stats.get("prompts", ())),
        unsupported=_shape(unsupported),
        too_narrow=_shape(too_narrow),
        spans=spans,
    )


def _names_both(fact: Fact, text: str) -> bool:
    """Whether `text` names the fact's subject and its object, as the locator finds names."""
    return locate_span(fact, Document(id="citation", text=text)) is not None


def _shape(facts: list[Fact]) -> FailureShape:
    return FailureShape(
        count=len(facts),
        ids=tuple(f.id for f in facts),
        examples=tuple(_line(f) for f in facts[:EXAMPLES]),
    )


def _line(fact: Fact) -> str:
    subject = fact.subject.label or fact.subject.key
    obj = fact.object_entity
    target = (obj.label or obj.key) if obj is not None else repr(fact.object_value)
    where = fact.evidence[0].doc_id if fact.evidence else "no evidence"
    return f"{subject} —{fact.predicate}→ {target}  [{fact.verdict.value}, {where}]"


def _row(label: str, text: str) -> str:
    return f"{label:<13} {text}"


__all__ = [
    "EXAMPLES",
    "FACTS_FILE",
    "SUMMARY_FILE",
    "TOO_NARROW",
    "UNSUPPORTED",
    "FailureShape",
    "GroundSummary",
    "GroundedGraph",
    "ground_graph",
    "ground_manifest",
    "summarize",
]
