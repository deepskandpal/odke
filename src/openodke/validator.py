"""The Validator: the verification layer between any extractor and the store (#129).

1.0 is two products in one package, and this is the first (DECISIONS #26). Hand
it another extractor's triples, or `Fact`s, and the texts they came from, and it
grounds, normalises, resolves, corroborates, gates and writes them. It is a
small class over `Pipeline`, which does all of that; the facts it is handed
are the extract stage, and nothing is extracted again.

Each stage defaults to what `odke run` builds under its usual name, with that
name's default options:

- grounder: `LLMGrounder` on `roles`, behind the free checks (`CheckedGrounder`),
  so a fact the ontology has no room for, or whose quote is not in its text,
  never costs a call. `locate=True` turns the span locator on;
- normalizer: `ValueNormalizer`; resolver: `NativeResolver`; corroborator:
  `SignatureCorroborator`, handed the texts so that a near-duplicate copy
  counts as one source; scorer: `EvidenceScorer`;
- gate: `VerdictGate(schema=True)`, which refuses what its passage contradicts
  and what the free checks refused. The free checks leave such a fact's
  verdict as it was, so a gate that only read verdicts would write it;
- the inverses step, on when the ontology declares a pair (DECISIONS #28), and
  the coverage report, as `odke run` has them.

A stage passed in replaces its default, and a pass-through from
`openodke.stages` turns one off. A grounder passed in still runs behind the
free checks. The ontology is optional: without one, the checks on the relation
and the types have nothing to check.

`lookup` resolves the batch against what the store already holds as well
(DECISIONS #31): a `StoreLookup`, such as `Neo4jSink.lookup()` or a
`MemoryLookup`, handed to the default resolver. Off unless given.

Writing to a store merges with it (#153). Every sink that can say what it
already holds (a `FactLookup`: `Neo4jSink`, or `JsonlSink(merge=True)`) is
handed to the default corroborator, which merges each incoming fact with the
one stored under its signature before the scorer and the gate see it. A rerun
of a claim from a second source then adds that source to the stored fact's
support list, and writes no second edge. A corroborator passed in merges with
the stores it was given, and a dry run reads no sink.

`validate(..., update=True)` says the texts are new versions of ones the
store already cites (#116). Each is retracted from every sink that can
retract a source first (`openodke.Reconciler`), so the facts the new version
still states merge with the store again and regain its support, and the rest
are left with less, or retired.

`judge` asks a model about the pairs the resolver's rules leave open, in both
orders (DECISIONS #34): a `PairJudge`, handed to the default resolver, which
reads its contexts from the texts given. Off unless given, and never asked in
a dry run.

`normalize_batch=True` has the default resolver normalise the batch
(DECISIONS #43): look-alikes the batch introduces become one entity when
nothing in their names or context keeps them apart. Off unless asked for.
`embed`, a function from texts to vectors, compares the sentences of
look-alikes the names alone would merge.

`validate()` returns the graph and a `ValidationReport` of the job: facts in,
refused, merged, linked and derived; what the pair judge decided; the model
calls, tokens and cost; the registered prompts sent; and the coverage report. A
client held to a budget (`openodke.llm.budget`) that stops the job leaves a
partial graph, written as usual, and the report's `stopped` says where and why.
A dry run asks no model and writes nothing: the free checks, the locator, and
every deterministic stage.

Every job has a run manifest (`openodke.manifest`, #160), as `report.manifest`:
what the Validator was, the models and prompts, the ontology, the inputs, the
times and the counts. It is written into each `JsonlSink`'s `manifest.json`
beside the counts the sink wrote, and to `manifest` when one is named.

Each call is a job on an `Observer` (`openodke.observe`, #161): a `job.start`
event, one per stage, one per model call the default grounder makes, and a
`job.end` whose counts are the report's (`ValidationReport.job`), each a span
too when OpenTelemetry is configured.

`validate(..., batch_size=N)` streams (#158, DECISIONS #45): the facts are
read N at a time, each micro-batch closed where the text its rows cite
changes, run through every stage and written, so a batch of any size runs in
the memory of one micro-batch. The texts are held, because a row may cite any
of them. What comes back is the run's report and a graph with its stats and no
facts: the sinks hold those. `openodke.stream` says what a micro-batch cannot
see.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import Field

from openodke._batch import failed_summary
from openodke.corroborate import (
    EvidenceScorer,
    NativeResolver,
    PairJudge,
    SignatureCorroborator,
    ValueNormalizer,
)
from openodke.corroborate.judge import judge_stop
from openodke.corroborate.resolve import Embed
from openodke.coverage import summary as coverage_summary
from openodke.gate import VerdictGate
from openodke.ground import LLMGrounder
from openodke.ground.checks import REASONS, VERDICTS, CheckedGrounder, refusals
from openodke.interop.triples import TripleRow, TriplesExtractor, _file_name, read_triples
from openodke.llm.base import LLMClient, ModelSpec
from openodke.llm.budget import stopped_summary
from openodke.llm.roles import ModelRoles
from openodke.manifest import FILE, InputsHash, Recorder, RunManifest, inputs_of, spent_of
from openodke.observe import JobCounts, Observer, spend
from openodke.ontology import Ontology
from openodke.pipeline import Pipeline
from openodke.reconcile import Reconciler
from openodke.sinks.jsonl import JsonlSink
from openodke.stages import (
    Corroborator,
    FactLookup,
    Gate,
    Grounder,
    Normalizer,
    PlatformProfile,
    Resolver,
    Scorer,
    Sink,
    StoreLookup,
)
from openodke.stream import Totals, by_text, micro_batches, shape, streams
from openodke.stream import write as stream_write
from openodke.types import Chunk, Document, Fact, Frozen, KnowledgeGraph, LinkKind

Facts = str | Path | Iterable[TripleRow | Mapping[str, Any] | Fact]


class ValidationReport(Frozen):
    """What one `Validator.validate()` call did: the job's counts, its cost and its gaps."""

    dry_run: bool = False
    documents: int = 0
    # Facts handed in whose text was given, and those whose text was not.
    facts_in: int = 0
    unmatched: int = 0
    # Every fact in, by the verdict grounding left it with.
    verdicts: dict[str, int] = Field(default_factory=dict)
    # What the free checks refused before any model was asked (`refusals`).
    checked: dict[str, int] = Field(default_factory=dict)
    # Refused by the gate, and why, by the gate's own count when it keeps one.
    refused: int = 0
    refused_by: dict[str, int] = Field(default_factory=dict)
    # Folded into another fact with the same signature by the corroborator.
    merged: int = 0
    # Merged with a fact the store already held under its signature (#153).
    restated: int = 0
    # With `update`: what retracting the old versions did first (#116).
    retracted: dict[str, int] | None = None
    # Links the resolver proposed between entities, by kind.
    linked: int = 0
    links: dict[str, int] = Field(default_factory=dict)
    # Against the store, with a lookup (DECISIONS #31): entities looked up,
    # store candidates, incoming keys re-keyed onto a stored one, links by kind.
    store: dict[str, int] | None = None
    # The pair judge (DECISIONS #34): pairs handed in and asked, calls, calls in
    # the swapped order, order disagreements, and each decision's count.
    judge: dict[str, int] | None = None
    # The batch normalised (DECISIONS #43): look-alike pairs, those merged, the
    # entities and mentions they made, and those kept apart, and why.
    batch: dict[str, int] | None = None
    # Inverse and symmetric partners the ontology implied (DECISIONS #28).
    derived: int = 0
    facts_out: int = 0
    edges: int = 0
    properties: int = 0
    entities: int = 0
    calls: int = 0
    # Of `calls`, those a response cache answered (`openodke.llm.cache`).
    cached: int = 0
    tokens: int = 0
    cost_usd: float | None = None
    prompts: tuple[str, ...] = ()
    # `KnowledgeGraph.stats["coverage"]`: what extraction left behind (#130).
    coverage: dict[str, Any] | None = None
    # `KnowledgeGraph.stats["stopped"]`: where a budget stopped the job, and why.
    stopped: dict[str, Any] | None = None
    # Texts left out because something failed for them alone, with why (#162).
    failed: dict[str, str] = Field(default_factory=dict)
    # With `batch_size`: the micro-batches run (#158). None when the job was one batch.
    batches: int | None = None
    # The job's run manifest (#160). Not in the report's JSON, which the graph's
    # stats keep: it is a file of its own.
    manifest: RunManifest | None = Field(default=None, exclude=True)

    def __eq__(self, other: object) -> bool:
        """Two reports of one job are equal: the manifest, which says when it ran, is left out."""
        if not isinstance(other, ValidationReport):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    @property
    def job(self) -> JobCounts:
        """Facts in, out, refused, merged, linked and queued for a person: this report's own."""
        queued = self.judge.get("queued", 0) if self.judge is not None else 0
        return JobCounts(
            facts_in=self.facts_in,
            facts_out=self.facts_out,
            refused=self.refused,
            merged=self.merged,
            linked=self.linked,
            review=queued,
        )

    def render(self) -> str:
        lines = ["odke validate — dry run, no model called" if self.dry_run else "odke validate"]
        if self.retracted is not None:
            lines.append(_row("update", _retracted_line(self.retracted)))
        if self.stopped is not None:
            lines.append(_row("stopped", stopped_summary(self.stopped)))
        if self.failed:
            lines.append(
                _row("failed", failed_summary(self.failed, of=self.documents, noun="text"))
            )
        unmatched = f"; {self.unmatched} name a text that was not given" if self.unmatched else ""
        given = f"{_n(self.facts_in, 'fact')} from {_n(self.documents, 'text')}"
        if self.batches is not None:
            given += f", in {self.batches} micro-batch{'' if self.batches == 1 else 'es'}"
        lines.append(_row("in", given + unmatched))
        lines.append(_row("grounded", ", ".join(f"{k} {v}" for k, v in self.verdicts.items())))
        checked = sum(self.checked.values())
        reasons = ", ".join(f"{v} {REASONS[k]}" for k, v in self.checked.items() if v)
        lines.append(
            _row("free checks", f"{checked} refused" + (f": {reasons}" if reasons else ""))
        )
        why = ", ".join(f"{v} {k}" for k, v in self.refused_by.items() if v)
        lines.append(_row("refused", f"{self.refused} by the gate" + (f": {why}" if why else "")))
        held = f"; {self.restated} into one the store holds" if self.restated else ""
        lines.append(_row("merged", f"{self.merged} into a fact with the same signature{held}"))
        kinds = ", ".join(f"{v} {k}" for k, v in self.links.items())
        lines.append(
            _row("linked", _n(self.linked, "entity link") + (f": {kinds}" if kinds else ""))
        )
        if self.store is not None:
            lines.append(_row("store", _store_line(self.store)))
        if self.judge is not None:
            lines.append(_row("judge", _judge_line(self.judge)))
        if self.batch is not None:
            lines.append(_row("batch", _batch_line(self.batch)))
        lines.append(_row("derived", f"{self.derived} inverse and symmetric partners"))
        lines.append(
            _row(
                "out",
                f"{_n(self.facts_out, 'fact')} ({_n(self.edges, 'edge')}, "
                f"{_n(self.properties, 'property')}), {_n(self.entities, 'entity')}",
            )
        )
        if self.dry_run:
            lines.append(_row("cost", "nothing: a dry run calls no model"))
        else:
            usd = f"${self.cost_usd:.4f}" if self.cost_usd is not None else "USD unknown"
            prompts = f" ({', '.join(self.prompts)})" if self.prompts else ""
            cached = f" ({self.cached} from the cache)" if self.cached else ""
            lines.append(
                _row(
                    "cost",
                    f"{self.calls} model calls{cached}, {self.tokens} tokens, {usd}{prompts}",
                )
            )
        if self.coverage is not None:
            lines.append(_row("coverage", coverage_summary(self.coverage)))
        return "\n".join(lines)


class Validated(NamedTuple):
    """The graph a `validate()` call built, and its report."""

    graph: KnowledgeGraph
    report: ValidationReport


class Validator:
    """The verification layer: another extractor's facts grounded, resolved, gated and written.

    ```python
    kg, report = Validator(ontology).validate(rows, documents)
    ```

    `rows` and `documents` are an adapter's output (`openodke.interop`), a
    triples file and the texts it cites, or `Fact`s and their texts. `roles`
    and `client` reach the default grounder's model, as `LLMGrounder` takes
    them. Every stage default is in the module's docstring.
    """

    def __init__(
        self,
        ontology: Ontology | None = None,
        *,
        grounder: Grounder | None = None,
        locate: bool = False,
        roles: ModelRoles | None = None,
        client: LLMClient | None = None,
        normalizer: Normalizer | None = None,
        resolver: Resolver | None = None,
        corroborator: Corroborator | None = None,
        scorer: Scorer | None = None,
        gate: Gate | None = None,
        inverses: bool | None = None,
        coverage: bool = True,
        sinks: Sequence[Sink] = (),
        lookup: StoreLookup | None = None,
        judge: PairJudge | None = None,
        normalize_batch: bool | None = None,
        embed: Embed | None = None,
        manifest: str | Path | None = None,
    ) -> None:
        if grounder is not None and locate:
            raise ValueError("with a grounder of your own, it locates: LLMGrounder(locate=True)")
        if resolver is not None and lookup is not None:
            raise ValueError(
                "with a resolver of your own, it looks the store up: NativeResolver(lookup=...)"
            )
        if resolver is not None and judge is not None:
            raise ValueError(
                "with a resolver of your own, it asks the judge: NativeResolver(judge=...)"
            )
        if resolver is not None and (normalize_batch is not None or embed is not None):
            raise ValueError(
                "with a resolver of your own, it normalises the batch: "
                "NativeResolver(normalize_batch=..., embed=...)"
            )
        self.ontology = ontology if ontology is not None else Ontology()
        self.grounder = grounder
        self.locate = locate
        self.roles = roles
        self.client = client
        self.normalizer = normalizer
        self.resolver = resolver
        self.corroborator = corroborator
        self.scorer = scorer
        self.gate = gate
        self.inverses = inverses
        self.coverage = coverage
        self.sinks = tuple(sinks)
        self.lookup = lookup
        self.judge = judge
        self.normalize_batch = normalize_batch
        self.embed = embed
        # Where each job's run manifest is written, besides every JsonlSink's.
        self.manifest = Path(manifest) if manifest is not None else None

    def validate(
        self,
        facts: Facts,
        documents: Iterable[Document] = (),
        *,
        dry_run: bool = False,
        extractor: str = "triples",
        confidence: float = 0.5,
        update: bool = False,
        recorder: Recorder | None = None,
        observer: Observer | None = None,
        batch_size: int | None = None,
    ) -> Validated:
        """The whole layer over `facts` and the texts they cite, written to every sink.

        Triples rows (or a JSON Lines file of them) become facts as
        `TriplesExtractor` makes them, named `extractor` with `confidence` as
        their prior; `Fact`s are taken as they are, each with the text its
        first evidence cites. A dry run calls no model and writes nothing.

        With `update`, each text is a new version of one the store already
        cites under its id: it is retracted from the sinks first, so what the
        new version no longer states loses that source. A dry run retracts
        nothing.

        `recorder` is the run manifest in the making, when the caller began it
        (`odke validate` does, with its options); left out, the manifest
        records this Validator's stages and the call's options.

        `observer` reports the job's events and spans; left out, a fresh one
        does. `odke validate` passes its own, which its grounder's calls report to.

        With `batch_size`, the facts are read and written a micro-batch at a
        time (#158), and the graph returned holds the stats and no facts.
        """
        if isinstance(facts, Fact) or isinstance(documents, Ontology):
            # It has a gate's method name, so code from before 1.0.0 that used
            # `openodke.Validator` as the gate arrives here (DECISIONS #26).
            raise TypeError(
                "Validator.validate takes a batch of facts and their texts: openodke.Validator "
                "is the verification layer since 1.0.0, and the gate is openodke.Gate "
                "(DECISIONS #26)"
            )
        from openodke.run.execute import handed_in, register_documents

        if recorder is None:
            described = self._described(extractor, confidence, update, batch_size)
            recorder = Recorder("validate", described)
        observer = observer if observer is not None else Observer("validate")
        observer.start(dry_run=dry_run)
        docs = list(documents)
        if batch_size is not None:
            return self._streamed(
                facts,
                docs,
                batch_size,
                dry_run=dry_run,
                extractor=extractor,
                confidence=confidence,
                update=update,
                recorder=recorder,
                observer=observer,
            )
        source = _source(facts, docs, extractor=extractor, confidence=confidence)
        inputs = inputs_of(docs, handed_in(source))
        retracted: dict[str, int] | None = None
        if update and not dry_run:
            retracted = Reconciler(self.sinks).delete([doc.id for doc in docs]).counts()
        stages = self._stages(dry_run, recorder, observer)
        # The corroborator reads the texts to count a near-duplicate copy once,
        # and the pair judge reads its contexts from them.
        register_documents(stages["corroborator"], docs)
        register_documents(stages["resolver"], docs)
        # A stage given is kept between calls, so its counts are read as a difference.
        before = _Before(stages)
        # Stand-ins: the double-stage warning sees each sink's platform, and the
        # real sinks write once the report is in the graph's stats.
        stand_ins = [_StandIn(getattr(sink, "profile", None)) for sink in self.sinks]
        pipeline = Pipeline(
            self.ontology,
            source,
            sinks=stand_ins,
            inverses=self.inverses,
            coverage=self.coverage,
            observer=observer,
            **stages,
        )
        kg = pipeline.run(docs)
        graph = shape(kg)
        report = self._report(kg.stats, graph, source, stages, dry_run, before)
        if retracted is not None:
            report = report.model_copy(update={"retracted": retracted})
        kg = kg.model_copy(update={"stats": _stats(kg.stats, graph, source, stages, report)})
        if not dry_run:
            begun = observer.begin("write")
            for sink in self.sinks:
                sink.write(kg)
            observer.end(begun, {"sinks": len(self.sinks), "facts": len(kg.facts)})
        spent = _job_spent(stages, before)
        manifest = recorder.finish(
            inputs=inputs,
            ontology=self.ontology,
            prompts=report.prompts,
            counts=validated_counts(report),
            spent=spent_of(spent),
            stopped=report.stopped,
            failed=report.failed,
            dry_run=dry_run,
            run=observer.run,
            job=report.job.model_dump(),
        )
        if not dry_run:
            for sink in self.sinks:
                if isinstance(sink, JsonlSink):
                    manifest.write_into(sink.directory / FILE)
            if self.manifest is not None:
                manifest.write(self.manifest, kg)
        observer.finish(
            report.job,
            cost=spend(spent),
            documents=report.documents,
            failed=len(report.failed),
            stopped=report.stopped.get("limit") if report.stopped else None,
        )
        return Validated(kg, report.model_copy(update={"manifest": manifest}))

    def _streamed(
        self,
        facts: Facts,
        docs: list[Document],
        size: int,
        *,
        dry_run: bool,
        extractor: str,
        confidence: float,
        update: bool,
        recorder: Recorder,
        observer: Observer,
    ) -> Validated:
        """`validate` a micro-batch at a time (#158): read, run, write, and on to the next.

        The run manifest hashes every text given and every fact as it is read,
        into the digest one batch takes of them all, and records the run's
        summed counts and its micro-batches.
        """
        from openodke.run.execute import register_documents

        if size < 1:
            raise ValueError("batch_size must be at least 1")
        whole = [type(sink).__name__ for sink in self.sinks if not streams(sink)]
        if whole:
            raise ValueError(
                f"{', '.join(whole)} writes the whole graph on every write, so a run in "
                "micro-batches would leave only the last; leave batch_size out"
            )
        items, replay = _items(facts)
        hashing = InputsHash(facts=True)
        hashing.add_documents(docs)
        items = _hashed(items, hashing)
        source: _Replay | TriplesExtractor = (
            _Replay(())
            if replay
            else TriplesExtractor((), extractor=extractor, confidence=confidence)
        )
        retracted: dict[str, int] | None = None
        if update and not dry_run:
            retracted = Reconciler(self.sinks).delete([doc.id for doc in docs]).counts()
        stages = self._stages(dry_run, recorder, observer)
        # A stage given is kept between calls, so its counts are read as a difference.
        before = _Before(stages)
        stand_ins = [_StandIn(getattr(sink, "profile", None)) for sink in self.sinks]
        pipeline = Pipeline(
            self.ontology,
            source,
            sinks=stand_ins,
            inverses=self.inverses,
            coverage=self.coverage,
            observer=observer,
            **stages,
        )
        texts = _Texts(docs)
        totals = Totals()
        ran: set[str] = set()

        def run(cited: list[Document]) -> None:
            for stage in (source, stages["corroborator"], stages["resolver"]):
                register_documents(stage, cited, replace=True)
            kg = pipeline.run(cited)
            totals.add(kg, resolver=stages["resolver"])
            ran.update(doc.id for doc in cited)
            stats, report = self._so_far(totals, len(ran), source, stages, dry_run, before)
            report = report.model_copy(update={"retracted": retracted})
            graph = totals.graph()
            if not dry_run:
                kg = kg.model_copy(update={"stats": _stats(stats, graph, source, stages, report)})
                begun = observer.begin("write")
                stream_write(self.sinks, kg, first=totals.batches == 1)
                observer.end(begun, {"sinks": len(self.sinks), "facts": len(kg.facts)})

        for batch in by_text(items, size, _text_of):
            source.feed(batch)
            names = {_text_of(item) for item in batch}
            run(texts.cited(names, getattr(source, "served", {})))
        # The texts no row cited are run too, as an unbatched job runs every text given.
        source.feed(())
        for rest in micro_batches((doc for doc in docs if doc.id not in ran), size):
            run(rest)
        if not totals.batches:
            # Nothing at all is one empty micro-batch, written as an unbatched job writes it.
            run([])
        stats, report = self._so_far(totals, len(ran), source, stages, dry_run, before)
        report = report.model_copy(update={"retracted": retracted})
        graph = totals.graph()
        stats = _stats(stats, graph, source, stages, report)
        kg = KnowledgeGraph(ontology_name=self.ontology.name, stats=stats)
        spent = _job_spent(stages, before)
        # The report's counts are the micro-batches' summed, so the job's are too.
        manifest = recorder.finish(
            inputs=hashing.inputs(),
            ontology=self.ontology,
            prompts=report.prompts,
            counts=validated_counts(report),
            spent=spent_of(spent),
            stopped=report.stopped,
            failed=report.failed,
            dry_run=dry_run,
            run=observer.run,
            job=report.job.model_dump(),
        )
        if not dry_run:
            for sink in self.sinks:
                if isinstance(sink, JsonlSink):
                    manifest.write_into(sink.directory / FILE)
            if self.manifest is not None:
                manifest.write(self.manifest, kg, shape=graph)
        observer.finish(
            report.job,
            cost=spend(spent),
            documents=report.documents,
            failed=len(report.failed),
            stopped=report.stopped.get("limit") if report.stopped else None,
        )
        return Validated(kg, report.model_copy(update={"manifest": manifest}))

    def _so_far(
        self,
        totals: Totals,
        documents: int,
        source: Any,
        stages: Mapping[str, Any],
        dry_run: bool,
        before: _Before,
    ) -> tuple[dict[str, Any], ValidationReport]:
        """The run's stats and report after the micro-batches `totals` has added up."""
        stats = {**totals.stats(), "documents": documents}
        report = self._report(
            stats, totals.graph(), source, stages, dry_run, before, store=totals.store
        )
        return stats, report.model_copy(update={"batches": totals.batches})

    def _described(
        self, extractor: str, confidence: float, update: bool, batch_size: int | None = None
    ) -> dict[str, Any]:
        """What the manifest of a job with no recorder of its own records as its config."""
        roles = self.roles if self.roles is not None else ModelRoles()
        return {
            "validator": {
                "grounder": _named(self.grounder) or "LLMGrounder",
                "roles": roles.model_dump(mode="json") if self.grounder is None else None,
                "locate": self.locate,
                **{
                    name: _named(getattr(self, name))
                    for name in ("normalizer", "resolver", "corroborator", "scorer", "gate")
                },
                "judge": _named(self.judge),
                "lookup": _named(self.lookup),
                "inverses": self.inverses,
                "coverage": self.coverage,
                "sinks": [_named(sink) for sink in self.sinks],
            },
            "extractor": extractor,
            "confidence": confidence,
            "update": update,
            "batch_size": batch_size,
        }

    def _stages(self, dry_run: bool, recorder: Recorder, observer: Observer) -> dict[str, Any]:
        """This job's stages: each one given, or its default, built fresh for the job."""
        ontology = self.ontology
        if dry_run:
            locating = self.locate or getattr(self.grounder, "locator", None) is not None
            grounder = CheckedGrounder(ontology=ontology, locate=locating)
        else:
            inner = self.grounder
            if inner is None:
                roles = self.roles if self.roles is not None else ModelRoles()
                client = self.client if self.client is not None else roles.client_for("ground")
                client = recorder.served.client("ground", roles.ground, client)
                client = observer.client(client, "ground")
                inner = LLMGrounder(roles, client=client, locate=self.locate)
            elif isinstance(spec := getattr(inner, "spec", None), ModelSpec):
                recorder.served.note("ground", spec)
            grounder = CheckedGrounder(inner, ontology=ontology)
        resolver = self.resolver
        if resolver is None:
            batch: dict[str, Any] = {"embed": self.embed}
            if self.normalize_batch is not None:
                batch["normalize_batch"] = self.normalize_batch
            resolver = NativeResolver(
                lookup=self.lookup, judge=self.judge, ontology=ontology, **batch
            )
        if dry_run and isinstance(resolver, NativeResolver) and resolver.judge is not None:
            # A dry run asks no model: the rules alone, on a copy.
            resolver = copy.copy(resolver)
            resolver.judge = None
        return {
            "grounder": grounder,
            "normalizer": _given(self.normalizer, ValueNormalizer, ontology),
            "resolver": resolver,
            "corroborator": (
                self.corroborator
                if self.corroborator is not None
                else SignatureCorroborator(ontology, store=() if dry_run else stores_of(self.sinks))
            ),
            "scorer": _given(self.scorer, EvidenceScorer),
            "gate": self.gate if self.gate is not None else VerdictGate(schema=True),
        }

    def _report(
        self,
        stats: Mapping[str, Any],
        graph: Mapping[str, Any],
        source: Any,
        stages: Mapping[str, Any],
        dry_run: bool,
        before: _Before,
        *,
        store: Mapping[str, Any] | None = None,
    ) -> ValidationReport:
        """The report from the pipeline's `stats` and the graph's `shape`, counting each
        stage since `before`; `store`, the resolver's store counts summed over micro-batches."""
        judge_stage = getattr(stages["resolver"], "judge", None)
        spent = _job_spent(stages, before)
        refused_by = {
            reason: count - before.refused.get(reason, 0)
            for reason, count in _refused_by(stages["gate"]).items()
            if count - before.refused.get(reason, 0)
        }
        own: dict[str, Any] = dict(getattr(source, "stats", {}))
        # A triples stage counts rows that found their text; a replay, facts.
        facts_in = int(own.get("rows") or own.get("facts") or 0)
        grounding = stages["grounder"].stats
        derived, refused = int(stats.get("derived", 0)), int(stats.get("refused", 0))
        links: Mapping[str, int] = graph["links"]
        coverage = stats.get("coverage")
        resolved = getattr(stages["resolver"], "stats", None)
        if store is None and isinstance(resolved, Mapping):
            store = resolved.get("store")
        batch = resolved.get("batch") if isinstance(resolved, Mapping) else None
        stopped = stats.get("stopped")
        if not isinstance(stopped, Mapping):
            stopped = judge_stop(resolved)
        failed = stats.get("failed")
        corroborated = getattr(stages["corroborator"], "stats", None)
        held = corroborated.get("store") if isinstance(corroborated, Mapping) else None
        judge = _counted(judge_stage, before.counted)
        prompts = dict.fromkeys(str(p) for p in grounding.get("prompts", ()))
        if judge is not None and judge.get("calls") and isinstance(resolved, Mapping):
            prompts.update(dict.fromkeys(str(p) for p in resolved.get("prompts", ())))
        return ValidationReport(
            dry_run=dry_run,
            documents=int(stats.get("documents", 0)),
            facts_in=facts_in,
            unmatched=int(own.get("unmatched_rows", 0)),
            verdicts={v.value: int(grounding["verdicts"].get(v.value, 0)) for v in VERDICTS},
            checked=refusals(grounding),
            refused=refused,
            refused_by=dict(refused_by),
            merged=facts_in + derived - refused - int(graph["facts"]),
            restated=int(held.get("merged", 0)) if isinstance(held, Mapping) else 0,
            linked=sum(links.values()),
            links=dict(sorted(links.items())),
            store=(
                {str(k): int(v) for k, v in store.items()} if isinstance(store, Mapping) else None
            ),
            judge=judge,
            batch=(
                {str(k): int(v) for k, v in batch.items()} if isinstance(batch, Mapping) else None
            ),
            derived=derived,
            facts_out=int(graph["facts"]),
            edges=int(graph["edges"]),
            properties=int(graph["properties"]),
            entities=int(graph["entities"]),
            calls=int(spent.get("calls", 0)),
            cached=int(spent.get("cached", 0)),
            tokens=int(spent.get("prompt_tokens", 0)) + int(spent.get("completion_tokens", 0)),
            cost_usd=spent.get("cost_usd"),
            prompts=tuple(prompts),
            coverage=dict(coverage) if isinstance(coverage, Mapping) else None,
            stopped=dict(stopped) if isinstance(stopped, Mapping) else None,
            failed=dict(failed) if isinstance(failed, Mapping) else {},
        )


class _Replay:
    """`Fact`s already made, handed back with the first chunk of the text they cite."""

    name = "facts"

    def __init__(self, facts: Iterable[Fact]) -> None:
        self.by_doc: dict[str | None, list[Fact]] = defaultdict(list)
        self._served: set[str] = set()
        self._before: Counter[str] = Counter()
        # The facts as handed in, when they came as a list: what the manifest hashes.
        self.source = facts if isinstance(facts, list | tuple) else None
        self.feed(facts)

    def feed(self, facts: Iterable[Fact]) -> None:
        """The next micro-batch's facts in place of these, which stay counted (#158)."""
        self._before.update(self._counts())
        self.by_doc = defaultdict(list)
        self._served = set()
        for fact in facts:
            self.by_doc[_text_of(fact)].append(fact)

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        if chunk.index != 0:
            return []
        self._served.add(chunk.doc_id)
        return list(self.by_doc.get(chunk.doc_id, ()))

    @property
    def rows(self) -> dict[str | None, list[Fact]]:
        """The facts handed in, by the text they cite, as a triples stage keeps its rows."""
        return self.by_doc

    def _counts(self) -> dict[str, int]:
        served = sum(len(f) for doc, f in self.by_doc.items() if doc in self._served)
        total = sum(len(f) for f in self.by_doc.values())
        return {"facts": served, "unmatched_rows": total - served}

    @property
    def stats(self) -> dict[str, int]:
        return {key: self._before[key] + count for key, count in self._counts().items()}


class _Before:
    """What the stages had counted before a job: a stage given is kept between calls."""

    def __init__(self, stages: Mapping[str, Any]) -> None:
        judge = getattr(stages["resolver"], "judge", None)
        self.spent = _spent(stages["grounder"].grounder)
        self.judged = _spent(judge)
        self.counted = _counted(judge)
        self.refused = _refused_by(stages["gate"])


class _Texts:
    """The texts a streamed job was given, found by the names its rows cite them by."""

    def __init__(self, docs: Sequence[Document]) -> None:
        self.order = {doc.id: at for at, doc in enumerate(docs)}
        self.by_id = {doc.id: doc for doc in docs}
        self.by_name: dict[str, list[Document]] = defaultdict(list)
        for doc in docs:
            if (name := _file_name(doc)) is not None:
                self.by_name[name].append(doc)

    def cited(self, names: Iterable[str | None], served: Mapping[str, str]) -> list[Document]:
        """The texts these names cite, in the order given: by id, else by file name.

        A file name two texts share cites both, so the triples stage settles
        which one its rows go to, as it does unbatched, and once it has, the
        one it chose.
        """
        found: dict[str, Document] = {}
        for name in names:
            if name is None:
                continue
            if name in self.by_id:
                found[name] = self.by_id[name]
            elif name in served and served[name] in self.by_id:
                found[served[name]] = self.by_id[served[name]]
            else:
                found.update({doc.id: doc for doc in self.by_name.get(name, ())})
        return sorted(found.values(), key=lambda doc: self.order[doc.id])


def _job_spent(stages: Mapping[str, Any], before: _Before) -> dict[str, Any]:
    """The model calls, tokens and cost the grounder and the pair judge spent on this job."""
    judge = getattr(stages["resolver"], "judge", None)
    return _added(_spent(stages["grounder"].grounder, before.spent), _spent(judge, before.judged))


def _hashed(items: Iterator[Any], hashing: InputsHash) -> Iterator[Any]:
    """`items`, each hashed into the run manifest's inputs as it is read."""
    for item in items:
        hashing.add_facts((item,))
        yield item


def _text_of(item: Any) -> str | None:
    """The text a triples row or a `Fact` cites: its `doc`, or its first evidence's."""
    if isinstance(item, Fact):
        return item.evidence[0].doc_id if item.evidence else None
    return str(item.doc)


def _items(facts: Facts) -> tuple[Iterator[Any], bool]:
    """The facts to stream, read as they are needed, and whether they are `Fact`s."""
    if isinstance(facts, str | Path):
        return read_triples(facts), False
    items = iter(facts)
    first = next(items, None)
    if first is None:
        return iter(()), False
    replay = isinstance(first, Fact)

    def checked() -> Iterator[Any]:
        for item in chain([first], items):
            if isinstance(item, Fact) != replay:
                raise ValueError("pass triples rows or Facts, not both")
            yield item

    return (checked() if replay else read_triples(checked())), replay


class _StandIn:
    """What the pipeline sees in place of a sink: its profile, and a write that does nothing."""

    def __init__(self, profile: PlatformProfile | None) -> None:
        self.profile = profile

    def write(self, kg: KnowledgeGraph) -> None:
        return None


def _source(facts: Facts, docs: list[Document], *, extractor: str, confidence: float) -> Any:
    if isinstance(facts, str | Path):
        return TriplesExtractor(facts, extractor=extractor, confidence=confidence, documents=docs)
    items = list(facts)
    if items and all(isinstance(item, Fact) for item in items):
        return _Replay([item for item in items if isinstance(item, Fact)])
    if any(isinstance(item, Fact) for item in items):
        raise ValueError("pass triples rows or Facts, not both")
    rows = [item for item in items if not isinstance(item, Fact)]
    return TriplesExtractor(rows, extractor=extractor, confidence=confidence, documents=docs)


def validated_counts(report: ValidationReport) -> dict[str, int]:
    """A job's counts, as its run manifest keeps them: the report's, by the report's names."""
    counts = {key: int(getattr(report, key)) for key in _COUNTED}
    counts["failed"] = len(report.failed)
    if report.batches is not None:
        # A streamed job's counts are its micro-batches' summed (#158).
        counts["batches"] = report.batches
    return counts


_COUNTED = (
    "documents",
    "facts_in",
    "unmatched",
    "refused",
    "merged",
    "restated",
    "linked",
    "derived",
    "facts_out",
    "edges",
    "properties",
    "entities",
)


def _named(stage: Any) -> str | None:
    """A stage by its class, as `module.Class`; None for none."""
    if stage is None:
        return None
    kind = type(stage)
    return f"{kind.__module__}.{kind.__qualname__}"


def stores_of(sinks: Iterable[Any]) -> tuple[FactLookup, ...]:
    """The sinks that can say what they already hold, which a write merges with (#153)."""
    return tuple(sink for sink in sinks if isinstance(sink, FactLookup))


def _given(stage: Any, default: Any, *args: Any) -> Any:
    """The stage given, or a fresh default."""
    return stage if stage is not None else default(*args)


def _refused_by(gate: Any) -> dict[str, int]:
    """A gate's refusals by reason, when it counts them as `VerdictGate` does."""
    stats = getattr(gate, "stats", None)
    refused = stats.get("refused") if isinstance(stats, Mapping) else None
    return {str(k): int(v) for k, v in refused.items()} if isinstance(refused, Mapping) else {}


def _spent(stage: Any, before: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The model calls, tokens and cost a grounder or judge reports: since `before`, when given."""
    stats = getattr(stage, "stats", None)
    stats = stats if isinstance(stats, Mapping) else {}
    counted = ("calls", "cached", "prompt_tokens", "completion_tokens")
    now: dict[str, Any] = {key: int(stats.get(key, 0)) for key in counted}
    cost = stats.get("cost_usd")
    now["cost_usd"] = float(cost) if isinstance(cost, int | float) else None
    if before is None:
        return now
    out = {key: now[key] - int(before.get(key, 0)) for key in counted}
    earlier = before.get("cost_usd")
    out["cost_usd"] = None if now["cost_usd"] is None else now["cost_usd"] - (earlier or 0.0)
    return out


def _added(one: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Two stages' spend together; the cost is unknown only when neither reported one."""
    out: dict[str, Any] = {
        k: int(one.get(k, 0)) + int(other.get(k, 0))
        for k in ("calls", "cached", "prompt_tokens", "completion_tokens")
    }
    costs = [c for c in (one.get("cost_usd"), other.get("cost_usd")) if c is not None]
    out["cost_usd"] = sum(costs) if costs else None
    return out


# What the report says of the pair judge, in this order.
_JUDGED = (
    "pairs",
    "asked",
    "calls",
    "swapped",
    "disagreed",
    "same",
    "different",
    "unsure",
    "person",
    "queued",
    "no_context",
    "failed",
    "unasked",
)


def _counted(judge: Any, before: Mapping[str, int] | None = None) -> dict[str, int] | None:
    """A judge's counts, since `before` when given; None without a judge."""
    stats = getattr(judge, "stats", None)
    if not isinstance(stats, Mapping):
        return None
    earlier = before or {}
    return {key: int(stats.get(key, 0)) - int(earlier.get(key, 0)) for key in _JUDGED}


def _stats(
    pipeline: Mapping[str, Any],
    graph: Mapping[str, Any],
    source: Any,
    stages: Mapping[str, Any],
    report: ValidationReport,
) -> dict[str, Any]:
    """The pipeline's counts, each stage's own and the report: what a JSONL manifest keeps."""
    from openodke.run.execute import jsonable

    stats: dict[str, Any] = dict(pipeline)
    stats["graph"] = {
        "facts": int(graph["facts"]),
        "edges": report.edges,
        "properties": report.properties,
        "entities": report.entities,
        "links": report.links,
    }
    reported = {"extractor": source, **stages}
    stats["stages"] = {
        # The gate's counts keep the run report's key (DECISIONS #26).
        ("validator" if name == "gate" else name): jsonable(own)
        for name, stage in reported.items()
        if isinstance(own := getattr(stage, "stats", None), Mapping)
    }
    stats["validation"] = report.model_dump(mode="json")
    return stats


def _row(label: str, text: str) -> str:
    return f"{label:<13} {text}"


def _retracted_line(counts: Mapping[str, int]) -> str:
    cited = counts.get("cited", 0)
    head = f"the old versions retracted first: {_n(cited, 'fact')} cited them"
    if not cited:
        return head
    gone = counts.get("retired", 0) + counts.get("deleted", 0)
    return f"{head}, {counts.get('lost', 0)} kept a source, {gone} left with none"


def _store_line(store: Mapping[str, int]) -> str:
    kinds = ", ".join(f"{store.get(k.value, 0)} {k.value}" for k in LinkKind)
    return (
        f"{_n(store.get('looked_up', 0), 'entity')} looked up, "
        f"{_n(store.get('candidates', 0), 'candidate')} in the store: "
        f"{store.get('rekeyed', 0)} re-keyed onto a stored key; links {kinds}"
    )


def _judge_line(judged: Mapping[str, int]) -> str:
    asked, calls = judged.get("asked", 0), judged.get("calls", 0)
    decided = ", ".join(f"{judged.get(k, 0)} {k}" for k in ("same", "different", "unsure"))
    line = (
        f"{_n(judged.get('pairs', 0), 'pair')} in the band: {asked} asked in both orders "
        f"({_n(calls, 'call')}, {judged.get('swapped', 0)} swapped), {decided}; "
        f"orders disagreed on {judged.get('disagreed', 0)}"
    )
    extra = [
        f"{judged[k]} {label}"
        for k, label in (
            ("person", "decided by a person"),
            ("queued", "queued"),
            ("no_context", "without context"),
            ("failed", "failed calls"),
            ("unasked", "calls the budget refused"),
        )
        if judged.get(k)
    ]
    return line + (f"; {', '.join(extra)}" if extra else "")


def _batch_line(batch: Mapping[str, int]) -> str:
    merged = batch.get("mentions", 0)
    line = (
        f"{_n(merged, 'mention')} merged into {_n(batch.get('groups', 0), 'entity')}, "
        f"from {_n(batch.get('alike', 0), 'look-alike pair')}"
    )
    why = [
        f"{batch[k]} {label}"
        for k, label in (
            ("numbers", "by a number"),
            ("forms", "by a legal form"),
            ("context", "by context"),
            ("ambiguous", "alike to two"),
            ("refused", "by a chain"),
        )
        if batch.get(k)
    ]
    return line + (f"; kept apart: {', '.join(why)}" if why else "")


def _n(count: int, noun: str) -> str:
    plural = noun[:-1] + "ies" if noun.endswith("y") else noun + "s"
    return f"{count} {noun if count == 1 else plural}"


__all__ = ["ValidationReport", "Validated", "Validator", "stores_of", "validated_counts"]
