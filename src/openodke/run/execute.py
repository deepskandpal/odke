"""Running a built config: bootstrap, load, run, collect every stage's counts, write.

Then the run manifest (`openodke.manifest`): what was asked, what answered and
what was made, into each JSONL sink's `manifest.json` beside the counts the
sink wrote there, and to the config's `manifest` path.

The pipeline writes the moment it has a graph, and a stage's own counts — the
grounder's verdicts, the extractor's rejections, the corroborator's conflicts —
are not in that graph. So the pipeline here is handed stand-ins that carry each
real sink's `PlatformProfile` and write nothing: the double-stage warning still
fires at construction (DECISIONS #21), and the real sinks write once the stats
are in `KnowledgeGraph.stats`, where a `JsonlSink` manifest and a report can
read them. `pipeline.py` is unchanged.

With `batch_size` the run streams (#158): the inputs are read a micro-batch
of documents at a time, each is run and written, and `openodke.stream.Totals`
sums their counts, so the report and the manifest describe the run.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from openodke._batch import failed_summary
from openodke.corroborate.judge import judge_stop
from openodke.corroborate.provenance import CONFLICT
from openodke.coverage import summary as coverage_summary
from openodke.interop.triples import TriplesExtractor, read_triples
from openodke.llm.budget import budget_summary, stopped_summary
from openodke.manifest import (
    FILE,
    Inputs,
    InputsHash,
    Recorder,
    RunManifest,
    inputs_of,
    package,
    spent_of,
)
from openodke.observe import JobCounts, Observer, job_counts_of_run, spend
from openodke.ontology import Ontology
from openodke.reextract import summary as reextract_summary
from openodke.run.build import Built, build, check_streams
from openodke.run.config import STAGES, ConfigError, RunConfig
from openodke.stages import PlatformProfile, Sink
from openodke.stream import PER_CALL, Totals, micro_batches, shape
from openodke.stream import write as stream_write
from openodke.types import Document, Fact, KnowledgeGraph

# How many facts a dry run prints before saying how many more there were.
SAMPLE = 20

# The gate's counts keep their 0.2 key, `stages.validator`: the stats are a
# format a JSONL manifest and other tools read, and renaming a key there waits
# for the report's schema version (DECISIONS #26).
_REPORTED_AS = {"gate": "validator"}


class _StandIn:
    """What the pipeline sees in place of a sink: its profile, and a write that does nothing."""

    def __init__(self, profile: PlatformProfile | None) -> None:
        self.profile = profile

    def write(self, kg: KnowledgeGraph) -> None:
        return None


@dataclass
class RunResult:
    """What a run produced, wrote or would have written, and what it warned about.

    A streamed run keeps no graph: `graph` holds the run's stats and no facts,
    and `sample` the first facts, which a dry run prints.
    """

    graph: KnowledgeGraph
    dry_run: bool
    written: list[str] = field(default_factory=list)
    bootstrap: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # What the run was, and every file it was written to; a dry run writes none.
    manifest: RunManifest | None = None
    manifests: list[str] = field(default_factory=list)
    sample: list[Fact] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, Any]:
        return self.graph.stats

    @property
    def job(self) -> JobCounts:
        """Facts in, out, refused, merged, linked and sent to review: the job's counts."""
        return job_counts_of_run(self.stats)

    def render(self) -> str:
        return render(self)


def execute(
    config: RunConfig,
    *,
    dry_run: bool = False,
    batch_size: int | None = None,
    replaying: RunManifest | None = None,
) -> RunResult:
    """Run the config. A dry run loads, extracts and grounds, and writes nothing.

    `batch_size`, or the config's own, streams the run in micro-batches of
    that many documents (#158). `replaying` is the manifest the config came
    from (`from_manifest`): the run is refused, as a `ConfigError`, when the
    ontology or the inputs are not what it recorded, because then it would not
    be that run again.
    """
    if batch_size is not None:
        config = config.with_batch_size(batch_size)
    return run_built(build(config), dry_run=dry_run, replaying=replaying)


def run_built(
    built: Built,
    *,
    dry_run: bool = False,
    batch_size: int | None = None,
    replaying: RunManifest | None = None,
) -> RunResult:
    config = built.config
    if batch_size is not None:
        # What the manifest records is the run that ran.
        config = config.with_batch_size(batch_size)
    size = config.batch_size
    if size is not None:
        check_streams(built.sinks)
    recorder = Recorder(
        "run",
        config.canonical(),
        config_file=config.source.name if config.source is not None else None,
        base_dir=config.base_dir,
        served=built.context.served,
        cache=built.context.cache.directory if built.context.cache is not None else None,
        budget=built.context.ledger.budget.limits,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline = built.pipeline(sinks=[_StandIn(p.profile) for p in built.sinks])
    result_warnings = [str(w.message) for w in caught]
    if replaying is not None:
        _refuse_unless_same(replaying, ontology=built.ontology)
        if replaying.package.get("openodke") != package()["openodke"]:
            result_warnings.append(
                f"the manifest was written by openodke {replaying.package.get('openodke')}; "
                f"this is {package()['openodke']}"
            )
    observer = built.context.observer
    observer.start(dry_run=dry_run)
    sample: list[Fact] = []
    shape_of: dict[str, Any] | None = None

    opened: list[Sink] = []
    applied: list[str] = []
    if built.lookup is not None and dry_run:
        result_warnings.append(built.lookup.dry_run_note)
    judge = getattr(built.stages.get("resolver"), "judge", None)
    if dry_run and judge is not None and judge.queue is not None:
        # A dry run writes nothing, the review queue included.
        judge.queue = None
        result_warnings.append("dry run: the pair judge's review queue is not written")
    try:
        if built.config.bootstrap:
            constrainer = built.stages["constrainer"]
            if dry_run:
                applied = [
                    s
                    for p in built.sinks
                    if p.can_bootstrap
                    for s in p.ddl(built.ontology, constrainer)
                ]
        if not dry_run:
            opened = [p.open() for p in built.sinks]
            # Before any model is called: a database that refuses the DDL fails
            # the run while it is still free.
            if built.config.bootstrap:
                applied = _bootstrap(built, opened)
            # After the bootstrap, so the indexes it creates are there to read through.
            built.open_lookup(opened)

        if size is not None:
            if replaying is not None:
                # A pass over the inputs before the first micro-batch is written.
                _refuse_unless_same(replaying, inputs=streamed_inputs(built))
            kg, written, sample, inputs = _streamed(
                built, pipeline, opened, size, dry_run=dry_run, observer=observer
            )
            shape_of = kg.stats.get("graph")
        else:
            docs = built.documents()
            inputs = inputs_of(docs, handed_in(built.stages["extractor"]))
            if replaying is not None:
                _refuse_unless_same(replaying, inputs=inputs)
            register_documents(built.stages["extractor"], docs)
            register_documents(built.stages.get("corroborator"), docs)
            register_documents(built.stages.get("resolver"), docs)
            kg = pipeline.run(docs)
            kg = kg.model_copy(update={"stats": collect_stats(built, kg)})

            if dry_run:
                written = [line for p in built.sinks for line in p.describe(kg)]
            else:
                written = []
                begun = observer.begin("write")
                for plan, sink in zip(built.sinks, opened, strict=True):
                    sink.write(kg)
                    written.extend(plan.describe(kg))
                observer.end(begun, {"sinks": len(opened), "facts": len(kg.facts)})
    finally:
        if built.lookup is not None:
            built.lookup.close()
        for sink in opened:
            close = getattr(sink, "close", None)
            if callable(close):
                close()
    stats = kg.stats
    manifest = recorder.finish(
        inputs=inputs,
        ontology=built.ontology,
        prompts=prompts_sent(built.stages),
        counts=run_counts(stats),
        spent=spent_of(stats.get("spent") or {}),
        stopped=stats.get("stopped"),
        failed=stats.get("failed"),
        dry_run=dry_run,
        run=observer.run,
        job=job_counts_of_run(stats).model_dump(),
    )
    manifests = [] if dry_run else write_manifest(built, manifest, kg, shape=shape_of)
    # A streamed run's stats are its micro-batches' summed: so are the job's counts.
    finish(observer, job_counts_of_run(stats), stats)
    return RunResult(
        graph=kg,
        dry_run=dry_run,
        written=written,
        bootstrap=applied,
        warnings=result_warnings,
        manifest=manifest,
        manifests=manifests,
        sample=sample,
    )


def write_manifest(
    built: Built,
    manifest: RunManifest,
    kg: KnowledgeGraph,
    *,
    shape: Mapping[str, Any] | None = None,
) -> list[str]:
    """The run manifest into each JSONL sink's `manifest.json`, and to `manifest:` when named.

    A run with neither writes it beside its config, as `<name>.manifest.json`.
    `shape` is a streamed run's graph counts, which its graph does not hold.
    """
    config = built.config
    into = [
        Path(plan.directory) / FILE  # type: ignore[attr-defined]
        for plan in built.sinks
        if plan.name == "jsonl"
    ]
    written = [str(manifest.write_into(path)) for path in into]
    own = manifest_path(config, jsonl=bool(into))
    if own is not None:
        written.append(str(manifest.write(own, kg, shape=shape)))
    return written


def manifest_path(config: RunConfig, *, jsonl: bool) -> Path | None:
    """Where a run writes its manifest besides its JSONL sinks: `manifest:`, when named.

    Otherwise, with no JSONL sink (`jsonl` is False), beside the config as
    `<name>.manifest.json`, so that every run writes one; else nowhere more.
    """
    if config.manifest is not None:
        return config.resolve(config.manifest)
    if jsonl:
        return None
    stem = config.source.stem if config.source is not None else "odke"
    return config.base_dir / f"{stem}.manifest.json"


def handed_in(extractor: Any) -> list[Any] | None:
    """The rows an extractor replays rather than extracts, a triples file's say; None otherwise.

    In the order they were handed in, when the stage still has them that way
    (a file, or a list), so a run that reads them a micro-batch at a time
    hashes them as one that reads them all (#158); else by the text they cite.
    """
    rows = getattr(extractor, "rows", None)
    if not isinstance(rows, Mapping):
        return None
    source = getattr(extractor, "source", None)
    if isinstance(extractor, TriplesExtractor) and isinstance(source, str | Path | list | tuple):
        return list(read_triples(source))
    if isinstance(source, list | tuple):
        return list(source)
    return [row for group in rows.values() for row in group]


def run_counts(stats: Mapping[str, Any]) -> dict[str, int]:
    """The run's counts, as its manifest keeps them: the pipeline's, and the graph's."""
    graph = stats.get("graph") or {}
    counts = {key: int(stats.get(key, 0)) for key in _COUNTED}
    counts.update({key: int(graph.get(key, 0)) for key in _GRAPH})
    counts["links"] = sum(int(v) for v in (graph.get("links") or {}).values())
    counts["failed"] = len(stats.get("failed") or {})
    if "batches" in stats:
        # A streamed run's counts are its micro-batches' summed (#158).
        counts["batches"] = int(stats["batches"])
    return counts


_COUNTED = ("documents", "chunks", "skipped", "deferred", "empty_extractions", "derived", "refused")
_GRAPH = ("facts", "edges", "properties", "entities")


def _refuse_unless_same(
    manifest: RunManifest, *, ontology: Ontology | None = None, inputs: Inputs | None = None
) -> None:
    problems = manifest.differences(ontology=ontology, inputs=inputs)
    if problems:
        lines = "; ".join(problems)
        raise ConfigError(
            f"--from-manifest: not the run it recorded, so it is not run: {lines}. "
            "Run the config itself for a new run",
            problems=tuple(problems),
        )


def finish(observer: Observer, counts: JobCounts, stats: Mapping[str, Any]) -> None:
    """The job's last event: its counts, and its documents, failures and stop beside them."""
    stopped = stats.get("stopped")
    observer.finish(
        counts,
        cost=spend(stats.get("spent") or {}),
        documents=int(stats.get("documents", 0)),
        failed=len(stats.get("failed") or {}),
        stopped=stopped.get("limit") if isinstance(stopped, Mapping) else None,
    )


def _streamed(
    built: Built,
    pipeline: Any,
    opened: list[Sink],
    size: int,
    *,
    dry_run: bool,
    observer: Observer,
) -> tuple[KnowledgeGraph, list[str], list[Fact], Inputs]:
    """The run a micro-batch of documents at a time: each loaded, run and written in turn.

    Each stage that looks the documents up holds the micro-batch's alone. Each
    micro-batch is handed to the sinks with the run's stats so far, so a JSONL
    manifest describes the run whenever it is read. An input with no
    documents is one empty micro-batch, written as an unbatched run writes it.
    Every micro-batch's documents are hashed into the run manifest's inputs as
    they are read.
    """
    totals = Totals()
    sample: list[Fact] = []
    handed = handed_in(built.stages["extractor"])
    hashing = InputsHash(facts=handed is not None)
    hashing.add_facts(handed or ())
    looking = [built.stages["extractor"], built.stages.get("corroborator")]
    looking.append(built.stages.get("resolver"))
    stats: dict[str, Any] = {}
    batches = micro_batches(built.iter_documents(), size)
    for docs in chain(batches, [[]]):
        if not docs and totals.batches:
            break
        hashing.add_documents(docs)
        for stage in looking:
            register_documents(stage, docs, replace=True)
        kg = pipeline.run(docs)
        totals.add(kg, resolver=built.stages.get("resolver"))
        stats = collect_stats(built, kg, totals=totals)
        if dry_run:
            sample.extend(kg.facts[: SAMPLE - len(sample)])
        else:
            begun = observer.begin("write")
            stream_write(opened, kg.model_copy(update={"stats": stats}), first=totals.batches == 1)
            observer.end(begun, {"sinks": len(opened), "facts": len(kg.facts)})
    written = [
        line for plan in built.sinks for line in plan.streamed(totals.graph(), totals.batches)
    ]
    graph = KnowledgeGraph(ontology_name=built.ontology.name, stats=stats)
    return graph, written, sample, hashing.inputs()


def streamed_inputs(built: Built) -> Inputs:
    """A streamed run's inputs, read through once without running: what a replay checks first."""
    handed = handed_in(built.stages["extractor"])
    hashing = InputsHash(facts=handed is not None)
    hashing.add_documents(built.iter_documents())
    hashing.add_facts(handed or ())
    return hashing.inputs()


def _bootstrap(built: Built, opened: list[Sink]) -> list[str]:
    applied: list[str] = []
    constrainer = built.stages["constrainer"]
    for plan, sink in zip(built.sinks, opened, strict=True):
        if not plan.can_bootstrap:
            continue
        bootstrap = sink.bootstrap  # type: ignore[attr-defined]
        if plan.name == "neo4j":
            applied.extend(bootstrap(built.ontology, constrainer=constrainer))
        else:
            applied.extend(bootstrap(built.ontology) or ())
    return applied


def register_documents(stage: Any, docs: list[Document], *, replace: bool = False) -> None:
    """Hand the loaded documents to a stage that looks them up by id (DECISIONS #19).

    An extractor reads a document's URI and tier; the corroborator reads its
    text, to count a near-duplicate copy once; the resolver's pair judge reads
    the sentences around a mention. With `replace`, the stage holds these
    alone: what a streamed run hands each micro-batch, so no stage keeps every
    text of the run (#158).
    """
    lookup = getattr(stage, "documents", None)
    if isinstance(lookup, dict):
        if replace:
            lookup.clear()
        lookup.update({doc.id: doc for doc in docs})


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def collect_stats(
    built: Built, kg: KnowledgeGraph, *, totals: Totals | None = None
) -> dict[str, Any]:
    """The pipeline's counts, the graph's shape, and every stage's own counts.

    A stage reports by carrying a `stats` mapping, as the grounders and the
    verdict gate do; a user's stage that does the same is reported the
    same way. The extractor's rejections and routing report, and the
    corroborator's conflicts, are read from where those stages keep them.

    With `totals`, a streamed run's: the pipeline's counts and the shape are
    the run's so far, and `batches` says how many micro-batches made them. A
    stage's own counts are its own, which run on across micro-batches.
    """
    stats: dict[str, Any] = dict(kg.stats) if totals is None else totals.stats()
    stats["graph"] = shape(kg) if totals is None else totals.graph()
    if totals is not None:
        stats["batches"] = totals.batches
    stages: dict[str, Any] = {}
    for name in STAGES:
        stage = built.stages.get(name)
        if stage is None:
            continue
        own = getattr(stage, "stats", None)
        report = jsonable(own) if isinstance(own, Mapping) else {}
        if name == "extractor":
            report.update(_extractor_stats(stage))
        if name == "corroborator":
            report["conflicts"] = _conflicts(kg.facts)
        if name == "resolver" and totals is not None:
            # The resolver counts one call; the run's are summed.
            for key in PER_CALL:
                if (summed := totals.resolver(key)) is not None:
                    report[key] = jsonable(summed)
        if report:
            stages[_REPORTED_AS.get(name, name)] = report
    stats["stages"] = stages
    if "stopped" not in stats and (stop := judge_stop(stages.get("resolver"))) is not None:
        # The budget ran out in the pair judge, after everything else had asked.
        stats["stopped"] = stop
    meter = built.context.meter
    if meter is not None:
        cost = meter.report(documents=int(kg.stats.get("documents", 0))).as_stage_report()
        stats["cost"] = jsonable({"metrics": cost.metrics, "stages": cost.breakdown})
    cache = built.context.cache_stats()
    if cache is not None:
        stats["cache"] = cache
    # Every run says what it spent, metered or not.
    stats["spent"] = built.context.spent()
    if limits := built.context.ledger.budget.limits:
        stats["budget"] = limits
    return stats


def _extractor_stats(extractor: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    totals = getattr(extractor, "totals", None)
    if callable(totals):
        out["paths"] = jsonable(totals())
    rejections: Counter[str] = Counter()
    calls = cached = 0
    # Keys of the registered prompts the model path sent (DECISIONS #27).
    prompts: dict[str, None] = {}
    for source in (extractor, getattr(extractor, "llm", None)):
        found = getattr(source, "rejections", None)
        if isinstance(found, list):
            rejections.update(str(getattr(r, "reason", r)) for r in found)
        made = getattr(source, "calls", None)
        if isinstance(made, list):
            calls += len(made)
            cached += sum(1 for call in made if getattr(call, "cached", False) is True)
        sent = getattr(source, "prompts", None)
        if isinstance(sent, list | tuple):
            prompts.update(dict.fromkeys(str(key) for key in sent))
    if rejections:
        out["rejections"] = dict(sorted(rejections.items()))
    if calls and "paths" not in out:
        out["model_calls"] = calls
    if cached:
        # Among the model calls, those the response cache answered.
        out["cached_calls"] = cached
    if prompts:
        out["prompts"] = list(prompts)
    return out


def prompts_sent(stages: Mapping[str, Any]) -> list[str]:
    """The registered prompt keys every stage sent, in stage order (DECISIONS #27).

    Read where the stages keep them: an extractor's `prompts`, its inner model
    extractor's, or a `stats["prompts"]` as the model grounder reports it.
    """
    keys: dict[str, None] = {}
    for stage in stages.values():
        for source in (stage, getattr(stage, "llm", None)):
            if source is None:
                continue
            sent = getattr(source, "prompts", None)
            own = getattr(source, "stats", None)
            if isinstance(own, Mapping) and isinstance(own.get("prompts"), list | tuple):
                sent = [*(sent if isinstance(sent, list | tuple) else ()), *own["prompts"]]
            if isinstance(sent, list | tuple):
                keys.update(dict.fromkeys(str(key) for key in sent))
    return list(keys)


def _conflicts(facts: tuple[Fact, ...]) -> dict[str, int]:
    found = Counter(
        str(stamp.get("status"))
        for fact in facts
        if isinstance(stamp := fact.qualifiers.get(CONFLICT), Mapping)
    )
    return dict(sorted(found.items()))


def jsonable(value: Any) -> Any:
    """Plain JSON types, so stats survive a manifest and a `--json` reader."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((jsonable(v) for v in value), key=repr)
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def render(result: RunResult) -> str:
    stats = result.stats
    graph = stats.get("graph", {})
    lines = ["odke run — dry run, nothing written" if result.dry_run else "odke run"]
    if isinstance(stopped := stats.get("stopped"), Mapping):
        lines.append(_row("stopped", stopped_summary(stopped)))
    if isinstance(failed := stats.get("failed"), Mapping) and failed:
        lines.append(_row("failed", failed_summary(failed, of=int(stats.get("documents", 0)))))
    batches = stats.get("batches")
    streamed = f", in {batches} micro-batch{'' if batches == 1 else 'es'}" if batches else ""
    lines.append(
        _row(
            "documents",
            f"{stats.get('documents', 0)} ({stats.get('chunks', 0)} chunks; "
            f"{stats.get('skipped', 0)} skipped, {stats.get('deferred', 0)} deferred, "
            f"{stats.get('empty_extractions', 0)} empty){streamed}",
        )
    )
    for name, report in stats.get("stages", {}).items():
        lines.append(_row(name, _summary(report)))
    if stats.get("derived"):
        lines.append(_row("derived", f"{stats['derived']} inverse and symmetric partners"))
    lines.append(_row("refused", str(stats.get("refused", 0))))
    if isinstance(gaps := stats.get("coverage"), Mapping):
        lines.append(_row("coverage", coverage_summary(gaps)))
    if isinstance(asked := stats.get("reextract"), Mapping):
        lines.append(_row("reextract", reextract_summary(dict(asked))))
    links = graph.get("links", {})
    lines.append(
        _row(
            "graph",
            f"{graph.get('facts', 0)} facts ({graph.get('edges', 0)} edges, "
            f"{graph.get('properties', 0)} properties), {graph.get('entities', 0)} entities, "
            f"{sum(links.values())} links" + (f" ({_summary(links)})" if links else ""),
        )
    )
    lines.append(_row("cost", _cost(stats)))
    if isinstance(limits := stats.get("budget"), Mapping):
        lines.append(_row("budget", budget_summary(limits, stats.get("spent") or {})))
    if isinstance(cache := stats.get("cache"), Mapping):
        failed = cache.get("failed", 0)
        lines.append(
            _row(
                "cache",
                f"{cache.get('hits', 0)} answered from {cache.get('directory')}, "
                f"{cache.get('misses', 0)} asked and stored"
                + (f", {failed} failed and not stored" if failed else ""),
            )
        )
    if result.bootstrap:
        verb = "would apply" if result.dry_run else "applied"
        lines.append(_row("bootstrap", f"{verb} {len(result.bootstrap)} statements"))
        lines.extend(f"  {statement}" for statement in result.bootstrap)
    verb = "would write" if result.dry_run else "wrote"
    if result.written:
        lines.append(_row(verb, result.written[0]))
        lines.extend(f"  {line}" for line in result.written[1:])
    else:
        lines.append(_row(verb, "nothing — no sink configured"))
    if result.manifests:
        lines.append(_row("manifest", result.manifests[0]))
        lines.extend(f"  {path}" for path in result.manifests[1:])
    shown = list(result.graph.facts) or result.sample
    if result.dry_run and shown:
        total = int(graph.get("facts", len(shown)))
        lines.append("")
        lines.append("facts that would be written:")
        for fact in shown[:SAMPLE]:
            lines.append(f"  {_fact_line(fact)}")
        if total > SAMPLE:
            lines.append(f"  … and {total - SAMPLE} more")
    return "\n".join(lines)


def _row(label: str, text: str) -> str:
    return f"{label:<13} {text}"


def _cost(stats: Mapping[str, Any]) -> str:
    """USD (or unknown), calls with the cache's share, and tokens: the line every run prints.

    From the run's ledger; a graph whose stats predate it falls back to the meter's.
    """
    spent = stats.get("spent")
    if isinstance(spent, Mapping):
        calls, cached = spent.get("calls", 0), spent.get("cached_calls", 0)
        tokens = spent.get("input_tokens", 0) + spent.get("output_tokens", 0)
        usd = spent.get("usd")
    else:
        metrics = (stats.get("cost") or {}).get("metrics") or {}
        calls, cached = metrics.get("calls", 0), metrics.get("cached_calls", 0)
        tokens = metrics.get("prompt_tokens", 0) + metrics.get("completion_tokens", 0)
        usd = metrics.get("cost_usd")
    hits = f" ({cached} from the cache)" if cached else ""
    money = f"${usd:.4f}" if isinstance(usd, int | float) else "USD unknown"
    return f"{calls} model calls{hits}, {tokens} tokens, {money}"


def _summary(report: Mapping[str, Any]) -> str:
    parts = []
    for key, value in report.items():
        if isinstance(value, Mapping):
            inner = _summary(value)
            if inner:
                parts.append(f"{key} ({inner})")
        elif isinstance(value, bool) or value in (None, 0, 0.0, [], ""):
            continue
        elif isinstance(value, float):
            parts.append(f"{key} {value:.4f}")
        elif isinstance(value, list):
            parts.append(f"{key} {'+'.join(str(v) for v in value)}")
        else:
            parts.append(f"{key} {value}")
    return ", ".join(parts)


def _fact_line(fact: Fact) -> str:
    subject = fact.subject.label or fact.subject.key
    obj = (
        fact.object_entity.label or fact.object_entity.key
        if fact.object_entity is not None
        else repr(fact.object_value)
    )
    negated = "" if fact.polarity.value == "asserted" else f" ({fact.polarity.value})"
    source = fact.evidence[0].doc_id if fact.evidence else "no evidence"
    return (
        f"{subject} —{fact.predicate}→ {obj}{negated}  "
        f"[{fact.verdict.value}, {fact.confidence:.2f}, {source}]"
    )


__all__ = [
    "RunResult",
    "collect_stats",
    "execute",
    "finish",
    "handed_in",
    "jsonable",
    "manifest_path",
    "prompts_sent",
    "register_documents",
    "render",
    "run_built",
    "run_counts",
    "streamed_inputs",
    "write_manifest",
]
