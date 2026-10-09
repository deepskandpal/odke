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

`validate()` returns the graph and a `ValidationReport` of the job: facts in,
refused, merged, linked and derived; what the pair judge decided; the model
calls, tokens and cost; the registered prompts sent; and the coverage report. A
client held to a budget (`openodke.llm.budget`) that stops the job leaves a
partial graph, written as usual, and the report's `stopped` says where and why.
A dry run asks no model and writes nothing: the free checks, the locator, and
every deterministic stage.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
from openodke.coverage import summary as coverage_summary
from openodke.gate import VerdictGate
from openodke.ground import LLMGrounder
from openodke.ground.checks import REASONS, VERDICTS, CheckedGrounder, refusals
from openodke.interop.triples import TripleRow, TriplesExtractor
from openodke.llm.base import LLMClient
from openodke.llm.budget import stopped_summary
from openodke.llm.roles import ModelRoles
from openodke.ontology import Ontology
from openodke.pipeline import Pipeline
from openodke.reconcile import Reconciler
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
        lines.append(
            _row("in", f"{_n(self.facts_in, 'fact')} from {_n(self.documents, 'text')}{unmatched}")
        )
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

    def validate(
        self,
        facts: Facts,
        documents: Iterable[Document] = (),
        *,
        dry_run: bool = False,
        extractor: str = "triples",
        confidence: float = 0.5,
        update: bool = False,
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
        """
        if isinstance(facts, Fact) or isinstance(documents, Ontology):
            # It has a gate's method name, so code from before 1.0.0 that used
            # `openodke.Validator` as the gate arrives here (DECISIONS #26).
            raise TypeError(
                "Validator.validate takes a batch of facts and their texts: openodke.Validator "
                "is the verification layer since 1.0.0, and the gate is openodke.Gate "
                "(DECISIONS #26)"
            )
        from openodke.run.execute import register_documents

        docs = list(documents)
        source = _source(facts, docs, extractor=extractor, confidence=confidence)
        retracted: dict[str, int] | None = None
        if update and not dry_run:
            retracted = Reconciler(self.sinks).delete([doc.id for doc in docs]).counts()
        stages = self._stages(dry_run)
        # The corroborator reads the texts to count a near-duplicate copy once,
        # and the pair judge reads its contexts from them.
        register_documents(stages["corroborator"], docs)
        register_documents(stages["resolver"], docs)
        # A stage given is kept between calls, so its counts are read as a difference.
        judge = getattr(stages["resolver"], "judge", None)
        spent = _spent(stages["grounder"].grounder)
        judged = _spent(judge)
        judged_before = _counted(judge)
        refused_before = _refused_by(stages["gate"])
        # Stand-ins: the double-stage warning sees each sink's platform, and the
        # real sinks write once the report is in the graph's stats.
        stand_ins = [_StandIn(getattr(sink, "profile", None)) for sink in self.sinks]
        pipeline = Pipeline(
            self.ontology,
            source,
            sinks=stand_ins,
            inverses=self.inverses,
            coverage=self.coverage,
            **stages,
        )
        kg = pipeline.run(docs)
        refused_by = {
            reason: count - refused_before.get(reason, 0)
            for reason, count in _refused_by(stages["gate"]).items()
            if count - refused_before.get(reason, 0)
        }
        spent = _added(_spent(stages["grounder"].grounder, spent), _spent(judge, judged))
        report = self._report(kg, source, stages, dry_run, spent, refused_by, judged_before)
        if retracted is not None:
            report = report.model_copy(update={"retracted": retracted})
        kg = kg.model_copy(update={"stats": _stats(kg, source, stages, report)})
        if not dry_run:
            for sink in self.sinks:
                sink.write(kg)
        return Validated(kg, report)

    def _stages(self, dry_run: bool) -> dict[str, Any]:
        """This job's stages: each one given, or its default, built fresh for the job."""
        ontology = self.ontology
        if dry_run:
            locating = self.locate or getattr(self.grounder, "locator", None) is not None
            grounder = CheckedGrounder(ontology=ontology, locate=locating)
        else:
            inner = self.grounder
            if inner is None:
                inner = LLMGrounder(self.roles, client=self.client, locate=self.locate)
            grounder = CheckedGrounder(inner, ontology=ontology)
        resolver = self.resolver
        if resolver is None:
            resolver = NativeResolver(lookup=self.lookup, judge=self.judge)
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
        kg: KnowledgeGraph,
        source: Any,
        stages: Mapping[str, Any],
        dry_run: bool,
        spent: Mapping[str, Any],
        refused_by: Mapping[str, int],
        judged_before: Mapping[str, int] | None,
    ) -> ValidationReport:
        own: dict[str, Any] = dict(getattr(source, "stats", {}))
        # A triples stage counts rows that found their text; a replay, facts.
        facts_in = int(own.get("rows") or own.get("facts") or 0)
        grounding = stages["grounder"].stats
        derived, refused = int(kg.stats.get("derived", 0)), int(kg.stats.get("refused", 0))
        links = Counter(link.kind.value for link in kg.links)
        coverage = kg.stats.get("coverage")
        resolved = getattr(stages["resolver"], "stats", None)
        store = resolved.get("store") if isinstance(resolved, Mapping) else None
        stopped = kg.stats.get("stopped")
        failed = kg.stats.get("failed")
        corroborated = getattr(stages["corroborator"], "stats", None)
        held = corroborated.get("store") if isinstance(corroborated, Mapping) else None
        judge = _counted(getattr(stages["resolver"], "judge", None), judged_before)
        prompts = dict.fromkeys(str(p) for p in grounding.get("prompts", ()))
        if judge is not None and judge.get("calls") and isinstance(resolved, Mapping):
            prompts.update(dict.fromkeys(str(p) for p in resolved.get("prompts", ())))
        return ValidationReport(
            dry_run=dry_run,
            documents=int(kg.stats.get("documents", 0)),
            facts_in=facts_in,
            unmatched=int(own.get("unmatched_rows", 0)),
            verdicts={v.value: int(grounding["verdicts"].get(v.value, 0)) for v in VERDICTS},
            checked=refusals(grounding),
            refused=refused,
            refused_by=dict(refused_by),
            merged=facts_in + derived - refused - len(kg.facts),
            restated=int(held.get("merged", 0)) if isinstance(held, Mapping) else 0,
            linked=len(kg.links),
            links=dict(sorted(links.items())),
            store=(
                {str(k): int(v) for k, v in store.items()} if isinstance(store, Mapping) else None
            ),
            judge=judge,
            derived=derived,
            facts_out=len(kg.facts),
            edges=len(kg.edges),
            properties=len(kg.properties),
            entities=len(kg.entities),
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
        for fact in facts:
            self.by_doc[fact.evidence[0].doc_id if fact.evidence else None].append(fact)
        self._served: set[str] = set()

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        if chunk.index != 0:
            return []
        self._served.add(chunk.doc_id)
        return list(self.by_doc.get(chunk.doc_id, ()))

    @property
    def stats(self) -> dict[str, int]:
        served = sum(len(f) for doc, f in self.by_doc.items() if doc in self._served)
        total = sum(len(f) for f in self.by_doc.values())
        return {"facts": served, "unmatched_rows": total - served}


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
        return _Replay(item for item in items if isinstance(item, Fact))
    if any(isinstance(item, Fact) for item in items):
        raise ValueError("pass triples rows or Facts, not both")
    rows = [item for item in items if not isinstance(item, Fact)]
    return TriplesExtractor(rows, extractor=extractor, confidence=confidence, documents=docs)


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
)


def _counted(judge: Any, before: Mapping[str, int] | None = None) -> dict[str, int] | None:
    """A judge's counts, since `before` when given; None without a judge."""
    stats = getattr(judge, "stats", None)
    if not isinstance(stats, Mapping):
        return None
    earlier = before or {}
    return {key: int(stats.get(key, 0)) - int(earlier.get(key, 0)) for key in _JUDGED}


def _stats(
    kg: KnowledgeGraph, source: Any, stages: Mapping[str, Any], report: ValidationReport
) -> dict[str, Any]:
    """The pipeline's counts, each stage's own and the report: what a JSONL manifest keeps."""
    from openodke.run.execute import jsonable

    stats: dict[str, Any] = dict(kg.stats)
    stats["graph"] = {
        "facts": len(kg.facts),
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
        )
        if judged.get(k)
    ]
    return line + (f"; {', '.join(extra)}" if extra else "")


def _n(count: int, noun: str) -> str:
    plural = noun[:-1] + "ies" if noun.endswith("y") else noun + "s"
    return f"{count} {noun if count == 1 else plural}"


__all__ = ["ValidationReport", "Validated", "Validator", "stores_of"]
