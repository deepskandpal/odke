"""Span width against the grounder's verdict: the one evaluator that needs no labels.

Every other evaluator here is BYOLD — bring your own labelled dataset. This one
is not, and that is the point. A citation too narrow to carry its claim is an
extraction bug the grounder reports for free: shown `Ireland` and asked whether
`Acme operates_in Ireland` follows from it, a correct grounder answers
`not_found`, and a true fact is thrown away by its own citation.

Where an extractor makes this mistake the two distributions separate: the
citations that merely distinguish one fact from its siblings cluster in
`not_found`, and the clause-width ones in `supported`. Where they separate
like that, **the `not_found`
rate is a usable proxy for citation quality with no gold set at all** — it
scores the "too narrow" error directly, which is the error a span gold set
would be built to find.

What this cannot say: these are the grounder's own verdicts, not labels. The
gap is a diagnostic of the citations, not a score of the facts, and a corpus
whose widths do not separate has been told nothing alarming.

Only citations are split. A span openodke located for a fact that cited
nothing (`SpanOrigin.LOCATED`, the span locator) is one or two sentences by
construction, so its width says nothing about the extractor; it is counted
apart, per verdict, with its own median (DECISIONS #25).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import median, quantiles

from openodke.eval.formats import load_jsonl
from openodke.eval.report import Metric, StageReport, ratio
from openodke.types import Fact, GroundingVerdict, KnowledgeGraph, SpanOrigin

# The order the report reads in: what grounding kept, what it rejected, and
# what it never saw.
VERDICTS: tuple[GroundingVerdict, ...] = (
    GroundingVerdict.SUPPORTED,
    GroundingVerdict.CONTRADICTED,
    GroundingVerdict.NOT_FOUND,
    GroundingVerdict.UNCHECKED,
)
FACTS_FILE = "facts.jsonl"

DESCRIPTION = """\
odke eval spans --facts FACTS

Needs no labelled data — the only evaluator here that does not. It reads a
run's facts and reports span width split by the grounder's verdict: count,
median, quartiles, min and max, plus the share of facts citing no span of their
own (a fact grounded against its whole text, because its extractor cited
nothing, is one of those). Spans openodke located for such facts are counted
apart from citations, in the `located` column, and not in the widths.
Where the not_found widths sit below the supported ones, the not_found rate is
a proxy for citation quality with no gold set: a citation too narrow to carry
its claim is grounded away, and the grounder reports that error for free.

FACTS is the facts.jsonl a run wrote, or the directory a sink wrote it into.
In process, `evaluate_spans(kg.facts)` takes a KnowledgeGraph's facts."""


def span_width(fact: Fact) -> int | None:
    """The width in characters of the span a grounder would be shown, or `None`.

    The first evidence carrying a span, because that is the one the grounder
    reads: `Evidence.span`, never a narrower mention span beside it, since a
    mention is what a highlight points at and not what grounding is asked
    about. A fact citing a document and no offsets has no width, and neither
    does one whose span is only the text it came from (`SpanOrigin.CONTEXT`):
    nobody chose that span, so its width says nothing about a citation.
    """
    evidence = next((e for e in fact.evidence if e.span is not None), None)
    if evidence is None or evidence.span is None or evidence.span_origin is SpanOrigin.CONTEXT:
        return None
    return evidence.span.end - evidence.span.start


def _located(fact: Fact) -> bool:
    """Whether the span `span_width` measures was located by openodke rather than cited."""
    evidence = next((e for e in fact.evidence if e.span is not None), None)
    return evidence is not None and evidence.span_origin is SpanOrigin.LOCATED


def load_facts(path: str | Path) -> list[Fact]:
    """A run's facts: a JSONL file of `Fact` rows, or the directory a `JsonlSink` wrote."""
    source = Path(path)
    return load_jsonl(source / FACTS_FILE if source.is_dir() else source, Fact)


def evaluate_spans(facts: Iterable[Fact] | KnowledgeGraph) -> StageReport:
    """Citation width per verdict, the gap between `not_found` and `supported`,
    and the spans openodke located, counted apart."""
    rows = list(facts.facts if isinstance(facts, KnowledgeGraph) else facts)
    widths: dict[GroundingVerdict, list[int]] = {verdict: [] for verdict in VERDICTS}
    located: dict[GroundingVerdict, list[int]] = {verdict: [] for verdict in VERDICTS}
    missing = dict.fromkeys(VERDICTS, 0)
    for fact in rows:
        width = span_width(fact)
        if width is None:
            missing[fact.verdict] += 1
        else:
            (located if _located(fact) else widths)[fact.verdict].append(width)
    counts = {
        verdict: len(widths[verdict]) + len(located[verdict]) + missing[verdict]
        for verdict in VERDICTS
    }

    breakdown: dict[str, dict[str, Metric]] = {
        verdict.value: {
            "n": counts[verdict],
            "no_span": missing[verdict],
            "located": len(located[verdict]),
            **_distribution(widths[verdict]),
        }
        for verdict in VERDICTS
    }
    supported, not_found = widths[GroundingVerdict.SUPPORTED], widths[GroundingVerdict.NOT_FOUND]
    everything = [w for row in widths.values() for w in row]
    placed = [w for row in located.values() for w in row]
    no_span = sum(missing.values())
    # The denominator for the proxy is what a grounder actually ruled on.
    checked = sum(counts[v] for v in VERDICTS if v is not GroundingVerdict.UNCHECKED)

    metrics: dict[str, Metric] = {
        "with_span": len(everything),
        "no_span": no_span,
        "no_span_rate": ratio(no_span, len(rows)),
        "median_width": median(everything) if everything else None,
        # Spans openodke found for facts that cited nothing: not citations.
        "located": len(placed),
        "located_median": median(placed) if placed else None,
        # The proxy, over the facts a grounder ruled on: where the widths
        # separate, this is the rate at which citations threw true facts away.
        "not_found_rate": ratio(counts[GroundingVerdict.NOT_FOUND], checked),
        "supported_median": median(supported) if supported else None,
        "not_found_median": median(not_found) if not_found else None,
        "median_gap": _gap(not_found, supported),
    }
    return StageReport(
        stage="spans",
        n=len(rows),
        metrics=metrics,
        breakdown=breakdown,
        notes=_notes(
            not_found, supported, no_span=no_span, located=len(placed), n=len(rows), checked=checked
        ),
    )


def _quartiles(widths: Sequence[int]) -> tuple[float, float]:
    """Lower and upper quartile, interpolated. `quantiles` needs two points; one is both."""
    if len(widths) == 1:
        return float(widths[0]), float(widths[0])
    q1, _, q3 = quantiles(widths, n=4, method="inclusive")
    return q1, q3


def _distribution(widths: Sequence[int]) -> dict[str, Metric]:
    """Min, quartiles, median and max. Every key stays, as `None`, on an empty row."""
    if not widths:
        return dict.fromkeys(("min", "p25", "median", "p75", "max"))
    q1, q3 = _quartiles(widths)
    return {"min": min(widths), "p25": q1, "median": median(widths), "p75": q3, "max": max(widths)}


def _gap(not_found: Sequence[int], supported: Sequence[int]) -> Metric:
    """How much wider the supported citations run. Positive means narrow `not_found` spans."""
    if not not_found or not supported:
        return None
    return median(supported) - median(not_found)


def _separate(not_found: Sequence[int], supported: Sequence[int]) -> bool:
    """Whether the narrow distribution really sits below the wide one.

    Three quarters of the `not_found` widths below where the middle half of
    the `supported` widths begins. Medians alone would call two heavily
    overlapping distributions separated on a difference of one character.
    """
    return _quartiles(not_found)[1] < _quartiles(supported)[0]


def _notes(
    not_found: Sequence[int],
    supported: Sequence[int],
    *,
    no_span: int,
    located: int,
    n: int,
    checked: int,
) -> tuple[str, ...]:
    if not n:
        first = "no facts: the run wrote nothing to measure"
    elif not not_found or not supported:
        first = (
            "nothing to compare: no fact carries a width under both not_found and supported"
            if checked
            else "every fact is unchecked, so there is no verdict to split the widths by"
        )
    else:
        medians = f"not_found median {median(not_found):g} chars vs supported {median(supported):g}"
        first = (
            f"{medians} — citations are too narrow"
            if _separate(not_found, supported)
            else f"{medians}: the widths overlap, so width does not explain the verdicts"
        )
    notes = [first]
    if no_span:
        notes.append(f"{no_span} of {n} fact(s) cite no span of their own and have no width")
    if located:
        notes.append(
            f"{located} of {n} fact(s) cited nothing and have a span openodke located; "
            "their widths are not citations and are left out of the split"
        )
    notes.append(
        "no labels were used: these are the grounder's own verdicts, so the gap is a "
        "diagnostic of the citations and not a score of the facts"
    )
    return tuple(notes)


__all__ = [
    "DESCRIPTION",
    "FACTS_FILE",
    "VERDICTS",
    "evaluate_spans",
    "load_facts",
    "span_width",
]
