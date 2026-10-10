"""The ablation: extraction alone, + grounding, + corroboration, on your own labels (#38).

The architecture's justification, measured — or not. Three configurations of one
run config, scored against one labelled extraction set:

1. **extraction alone** — the config's loaders, chunker, router and extractor.
   Every candidate is kept.
2. **+ grounding** — the same candidates through the config's grounder, then its
   gate: the configured one, or `VerdictGate` when it names none.
3. **+ corroboration** — the whole configured pipeline: normalise, resolve,
   add inverse and symmetric partners, corroborate and score, then the same gate.

Extraction runs once and grounding runs once. The later configurations replay the
facts the earlier ones produced through the real `Pipeline`, so the rows differ
only by the stages they add — and a model is not asked the same question twice,
which would double the bill and could get a different answer the second time.

Each row is `evaluate_extraction` against gold facts per document. A fact that
corroboration merged across documents cites each of them and is scored once in
each: the graph now claims every one of those documents states it, and the labels
say whether it does. `grounding_ablation` adds the view from inside the extracted
set: how many true facts the gate kept, how many false ones it let through, and
what refusing `not_found` as well would have changed.

If grounding does not move precision on your labels, the first note says so.
That is a finding about your corpus, not a failure of the command.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from openodke.coverage import offered_by
from openodke.coverage import summary as coverage_summary
from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED
from openodke.eval.cost import CallRecord, StageCost
from openodke.eval.eval_report import (
    Dataset,
    EvalReport,
    Run,
    extraction_rows,
    from_stage,
    models_called,
)
from openodke.eval.extraction import evaluate_extraction, match_extraction, per_document
from openodke.eval.formats import GoldFact, GroundingLabel
from openodke.eval.grounding import DROPPED, grounding_ablation
from openodke.eval.report import Metric, StageReport
from openodke.ontology import Ontology
from openodke.pipeline import Pipeline
from openodke.types import Chunk, Document, Fact, GroundingVerdict

if TYPE_CHECKING:
    from openodke.run.config import RunConfig

EXTRACTION, GROUNDING, CORROBORATION = "extraction alone", "+ grounding", "+ corroboration"
CONFIGURATIONS = (EXTRACTION, GROUNDING, CORROBORATION)

DESCRIPTION = """\
odke eval ablation --config RUN_CONFIG --labels GOLD_FACTS

Runs the config three ways over your labelled documents and scores each against
your gold facts: extraction alone; + grounding and its gate; + normalisation,
resolution, corroboration and scoring. Extraction and grounding each run once,
and nothing is written to the config's sinks. The documents are the config's
inputs, and a gold fact names its document by the id `odke run` gives it: the
input's path relative to the config, plus `#L<line>` for a record."""

ChunkKey = tuple[str, int, int, int]


def _key(chunk: Chunk) -> ChunkKey:
    return (chunk.doc_id, chunk.index, chunk.start, chunk.end)


class _Recording:
    """The config's extractor, keeping what it found for each chunk."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.found: dict[ChunkKey, list[Fact]] = {}

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        facts = list(self.inner.extract(chunk, ontology))
        self.found[_key(chunk)] = facts
        return facts

    def offered(self, ontology: Ontology) -> list[str] | None:
        # What the coverage report reads: the inner extractor's, when it can say.
        return offered_by(self.inner, ontology)

    def extract_many(self, chunks: Sequence[Chunk], ontology: Ontology) -> list[list[Fact]]:
        # Passed on whole, so a benchmark's extraction runs as concurrently as `odke run`'s.
        many = getattr(self.inner, "extract_many", None)
        if not callable(many):
            return [self.extract(chunk, ontology) for chunk in chunks]
        found = [list(facts) for facts in many(chunks, ontology)]
        for chunk, facts in zip(chunks, found, strict=True):
            self.found[_key(chunk)] = facts
        return found


class _Replaying:
    """An extractor that answers each chunk with what the recorded pass found."""

    def __init__(self, found: dict[ChunkKey, list[Fact]]) -> None:
        self.found = found

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        return list(self.found.get(_key(chunk), ()))


class _Stamped:
    """A grounder that hands back the verdicts the grounding pass already stamped."""

    def __init__(self, grounded: Iterable[Fact]) -> None:
        self.by_id = {fact.id: fact for fact in grounded}

    def ground(self, fact: Fact, doc: Document) -> Fact:
        return self.by_id.get(fact.id, fact)


@dataclass(frozen=True)
class AblationRun:
    """The three configurations' facts, before anything scores them.

    `run_ablation` scores them against your `GoldFact` labels; a public benchmark
    scores the same three sets with its own metrics (`openodke.eval.datasets`).
    `grounded` is every candidate with its verdict; `gated` is what the gate let
    through, which is the "+ grounding" row.
    """

    documents: list[Document]
    ontology: Ontology
    candidates: list[Fact]
    grounded: list[Fact]
    gated: list[Fact]
    corroborated: list[Fact]
    extraction_calls: list[CallRecord]
    all_calls: list[CallRecord]
    gate: Any
    notes: list[str] = field(default_factory=list)
    # What extraction left behind (`CoverageReport.stats()`), measured on the candidates.
    coverage: dict[str, Any] | None = None
    # The registered prompt keys the stages sent, and role -> model as configured.
    prompts: tuple[str, ...] = ()
    models: dict[str, str] = field(default_factory=dict)
    # What the extractor refused before anything was grounded (`Rejection`s,
    # #109); None for an extractor that keeps no record of them.
    rejections: list[Any] | None = None

    def configurations(self) -> list[tuple[str, list[Fact], list[CallRecord]]]:
        """Each configuration's name, its facts, and the calls it took to get them."""
        return [
            (EXTRACTION, self.candidates, self.extraction_calls),
            (GROUNDING, self.gated, self.all_calls),
            (CORROBORATION, self.corroborated, self.all_calls),
        ]


def rejections_of(extractor: Any) -> list[Any] | None:
    """What an extractor refused, as it records it: its `rejections`, and its model path's.

    None when neither keeps a list, so "none rejected" and "no record" stay apart.
    """
    found: list[Any] | None = None
    for source in dict.fromkeys(
        s for s in (extractor, getattr(extractor, "llm", None)) if s is not None
    ):
        kept = getattr(source, "rejections", None)
        if isinstance(kept, list):
            found = [*(found or []), *kept]
    return found


def ablate(config: RunConfig) -> AblationRun:
    """Run `config` three ways and keep the facts. Extraction and grounding run once."""
    # Imported here: `openodke.run` imports `openodke.eval.cost`, and so this package.
    from openodke.gate import VerdictGate
    from openodke.run.build import build
    from openodke.run.execute import prompts_sent, register_documents

    # Metered whatever the config says: calls and cost are a column of the table.
    metered = config.model_copy(update={"models": config.models.model_copy(update={"meter": True})})
    built = build(metered)
    meter = built.context.meter
    assert meter is not None
    docs = built.documents()
    stages = built.stages
    register_documents(stages["extractor"], docs)
    register_documents(stages.get("corroborator"), docs)
    register_documents(stages.get("resolver"), docs)
    ontology = built.ontology
    # The first two rows are what the extractor found and what the gate kept of
    # it; inverse partners belong to the whole pipeline, the third row.
    route = {"chunker": stages["chunker"], "router": stages["router"], "inverses": False}
    notes: list[str] = []
    if config.reextract is not None:
        notes.append("reextract is not ablated: every row is the first extraction pass")

    recording = _Recording(stages["extractor"])
    extracted = Pipeline(ontology, recording, coverage=True, **route).run(docs)
    candidates = list(extracted.facts)
    rejections = rejections_of(stages["extractor"])
    extraction_calls = list(meter.records)

    grounder = stages["grounder"]
    if grounder is None:
        grounded = candidates
        notes.append("no grounder is configured, so + grounding differs only by the gate")
    else:
        replay = _Replaying(recording.found)
        grounded = list(Pipeline(ontology, replay, grounder=grounder, **route).run(docs).facts)
    gate = stages["gate"] if stages["gate"] is not None else VerdictGate()
    gated = [f for f in grounded if gate.validate(f, ontology).action != "refuse"]
    all_calls = list(meter.records)

    full = built.pipeline(
        extractor=_Replaying(recording.found), grounder=_Stamped(grounded), gate=gate
    )
    corroborated = list(full.run(docs).facts)
    return AblationRun(
        documents=list(docs),
        ontology=ontology,
        candidates=candidates,
        grounded=grounded,
        gated=gated,
        corroborated=corroborated,
        extraction_calls=extraction_calls,
        all_calls=all_calls,
        gate=gate,
        notes=notes,
        coverage=extracted.stats.get("coverage"),
        prompts=tuple(prompts_sent(stages)),
        rejections=rejections,
        models={
            role: spec.model
            for role in ("extract", "ground", "infer")
            if (spec := getattr(config.models, role)) is not None
        },
    )


def run_ablation(config: RunConfig, gold: Sequence[GoldFact]) -> StageReport:
    """The three configurations of `config`, scored against `gold`. Writes nothing."""
    return score_ablation(ablate(config), gold)


def report_ablation(
    config: RunConfig,
    gold: Sequence[GoldFact],
    *,
    labels: str | Path | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> EvalReport:
    """`run_ablation` as an eval report: the three rows with their ranges. Writes nothing.

    `labels` names the gold file in the report's dataset.
    """
    run = ablate(config)
    rows, how = extraction_rows(
        run.configurations(),
        gold,
        ontology=run.ontology,
        resamples=resamples,
        seed=seed,
        level=level,
    )
    dataset = Dataset(
        name=Path(labels).name if labels is not None else "gold facts",
        path=str(labels) if labels is not None else None,
        documents=len({g.doc_id for g in gold}),
        labels=len(gold),
    )
    return from_stage(
        score_ablation(run, gold),
        rows=rows,
        bootstrap=how,
        run=Run(
            models=models_called(run.all_calls, run.models), prompts=run.prompts, dataset=dataset
        ),
    )


def score_ablation(run: AblationRun, gold: Sequence[GoldFact]) -> StageReport:
    """An `AblationRun` already made, scored against `gold`: `run_ablation` without the run."""
    notes = list(run.notes)
    candidates, grounded, docs, gate = run.candidates, run.grounded, run.documents, run.gate

    breakdown: dict[str, dict[str, Metric]] = {}
    for name, facts, calls in run.configurations():
        scored = evaluate_extraction(gold, facts)
        cost = StageCost.of(name, calls)
        breakdown[name] = {
            **{k: scored.metrics[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
            "facts": len(facts),
            "model_calls": cost.calls,
            "cost_usd": cost.cost_usd,
        }
        notes.extend(f"{name}: {note}" for note in scored.notes)

    precision = {name: breakdown[name]["precision"] for name in CONFIGURATIONS}
    recall = {name: breakdown[name]["recall"] for name in CONFIGURATIONS}
    notes[:0] = [
        _moved("grounding", precision, recall, EXTRACTION, GROUNDING),
        _moved(
            "normalising, resolving and corroborating", precision, recall, GROUNDING, CORROBORATION
        ),
    ]
    notes[2:2] = _gate_view(gold, candidates, grounded, docs, gate)
    if run.coverage is not None:
        notes.append(f"coverage of the candidates: {coverage_summary(run.coverage)}")
    notes.append("a fact merged across documents is scored once in each document it cites")

    metrics: dict[str, Metric] = {}
    for label, name in zip(
        ("extraction", "grounding", "corroboration"), CONFIGURATIONS, strict=True
    ):
        metrics[f"precision_{label}"] = precision[name]
        metrics[f"recall_{label}"] = recall[name]
    return StageReport(
        stage="ablation", n=len(gold), metrics=metrics, breakdown=breakdown, notes=tuple(notes)
    )


def _moved(
    what: str,
    precision: dict[str, Metric],
    recall: dict[str, Metric],
    before: str,
    after: str,
) -> str:
    p0, p1, r0, r1 = precision[before], precision[after], recall[before], recall[after]
    if p0 is None or p1 is None:
        return f"{what}: precision is undefined in one configuration, because nothing was kept"
    recall_text = f"recall {_fmt(r0)} → {_fmt(r1)}"
    if math.isclose(p0, p1, abs_tol=1e-9):
        return f"{what} did not move precision on these labels ({p0:.3f}); {recall_text}"
    return f"{what} moved precision from {p0:.3f} to {p1:.3f}; {recall_text}"


def _gate_view(
    gold: Sequence[GoldFact],
    candidates: Sequence[Fact],
    grounded: Sequence[Fact],
    docs: Sequence[Document],
    gate: Any,
) -> list[str]:
    labels = _labels(gold, candidates, docs)
    if not labels:
        return []
    refused = getattr(gate, "refused", None)
    notes = []
    if isinstance(refused, Collection) and all(isinstance(v, GroundingVerdict) for v in refused):
        drop = frozenset(refused)
    else:
        drop = frozenset(DROPPED)
        notes.append(
            "the configured gate does not say which verdicts it refuses, so the view "
            "below assumes contradicted and not_found"
        )
    off = grounding_ablation(labels, grounded, drop=drop).breakdown
    true, false = off["off"]["tp"], off["off"]["fp"]
    notes.append(
        f"of the {len(labels)} extracted facts, {true} are true: the gate kept "
        f"{off['on']['tp']} of them and {off['on']['fp']} of the {false} false ones"
    )
    if GroundingVerdict.NOT_FOUND not in drop:
        strict = grounding_ablation(labels, grounded, drop=DROPPED).breakdown["on"]
        notes.append(
            f"refusing not_found as well would keep {strict['tp']} of {true} true facts and "
            f"{strict['fp']} of {false} false ones (precision {_fmt(strict['precision'])})"
        )
    return notes


def _labels(
    gold: Sequence[GoldFact], candidates: Sequence[Fact], docs: Sequence[Document]
) -> list[GroundingLabel]:
    """Extracted facts labelled true or false by the extraction gold they were matched to."""
    text = {doc.id: doc.text for doc in docs}
    labels = []
    for outcome in match_extraction(gold, candidates):
        fact = outcome.predicted
        if fact is None:
            continue
        verdict: Literal["supported", "contradicted", "not_found"] = (
            "supported"
            if outcome.kind == "correct"
            else ("not_found" if outcome.kind == "spurious" else "contradicted")
        )
        cited = fact.evidence[0].doc_id if fact.evidence else None
        labels.append(
            GroundingLabel(
                text=text.get(cited, "") if cited else "", fact=fact, verdict=verdict, doc_id=cited
            )
        )
    return labels


def _fmt(value: Metric) -> str:
    return "—" if value is None else f"{value:.3f}"


__all__ = [
    "CONFIGURATIONS",
    "DESCRIPTION",
    "AblationRun",
    "ablate",
    "per_document",
    "report_ablation",
    "run_ablation",
    "score_ablation",
]
