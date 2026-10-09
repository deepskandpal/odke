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

`validate()` returns the graph and a `ValidationReport` of the job: facts in,
refused, merged, linked and derived; the model calls, tokens and cost; the
registered prompts sent; and the coverage report. A dry run asks no model and
writes nothing: the free checks, the locator, and every deterministic stage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import Field

from openodke.corroborate import (
    EvidenceScorer,
    NativeResolver,
    SignatureCorroborator,
    ValueNormalizer,
)
from openodke.coverage import summary as coverage_summary
from openodke.gate import VerdictGate
from openodke.ground import LLMGrounder
from openodke.ground.checks import REASONS, VERDICTS, CheckedGrounder, refusals
from openodke.interop.triples import TripleRow, TriplesExtractor
from openodke.llm.base import LLMClient
from openodke.llm.roles import ModelRoles
from openodke.ontology import Ontology
from openodke.pipeline import Pipeline
from openodke.stages import (
    Corroborator,
    Gate,
    Grounder,
    Normalizer,
    PlatformProfile,
    Resolver,
    Scorer,
    Sink,
)
from openodke.types import Chunk, Document, Fact, Frozen, KnowledgeGraph

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
    # Links the resolver proposed between entities, by kind.
    linked: int = 0
    links: dict[str, int] = Field(default_factory=dict)
    # Inverse and symmetric partners the ontology implied (DECISIONS #28).
    derived: int = 0
    facts_out: int = 0
    edges: int = 0
    properties: int = 0
    entities: int = 0
    calls: int = 0
    tokens: int = 0
    cost_usd: float | None = None
    prompts: tuple[str, ...] = ()
    # `KnowledgeGraph.stats["coverage"]`: what extraction left behind (#130).
    coverage: dict[str, Any] | None = None

    def render(self) -> str:
        lines = ["odke validate — dry run, no model called" if self.dry_run else "odke validate"]
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
        lines.append(_row("merged", f"{self.merged} into a fact with the same signature"))
        kinds = ", ".join(f"{v} {k}" for k, v in self.links.items())
        lines.append(
            _row("linked", _n(self.linked, "entity link") + (f": {kinds}" if kinds else ""))
        )
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
            lines.append(
                _row("cost", f"{self.calls} model calls, {self.tokens} tokens, {usd}{prompts}")
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
    ) -> None:
        if grounder is not None and locate:
            raise ValueError("with a grounder of your own, it locates: LLMGrounder(locate=True)")
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

    def validate(
        self,
        facts: Facts,
        documents: Iterable[Document] = (),
        *,
        dry_run: bool = False,
        extractor: str = "triples",
        confidence: float = 0.5,
    ) -> Validated:
        """The whole layer over `facts` and the texts they cite, written to every sink.

        Triples rows (or a JSON Lines file of them) become facts as
        `TriplesExtractor` makes them, named `extractor` with `confidence` as
        their prior; `Fact`s are taken as they are, each with the text its
        first evidence cites. A dry run calls no model and writes nothing.
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
        stages = self._stages(dry_run)
        # The corroborator reads the texts to count a near-duplicate copy once.
        register_documents(stages["corroborator"], docs)
        # A stage given is kept between calls, so its counts are read as a difference.
        spent = _spent(stages["grounder"].grounder)
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
        report = self._report(
            kg, source, stages, dry_run, _spent(stages["grounder"].grounder, spent), refused_by
        )
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
        return {
            "grounder": grounder,
            "normalizer": _given(self.normalizer, ValueNormalizer, ontology),
            "resolver": _given(self.resolver, NativeResolver),
            "corroborator": _given(self.corroborator, SignatureCorroborator, ontology),
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
    ) -> ValidationReport:
        own: dict[str, Any] = dict(getattr(source, "stats", {}))
        # A triples stage counts rows that found their text; a replay, facts.
        facts_in = int(own.get("rows") or own.get("facts") or 0)
        grounding = stages["grounder"].stats
        derived, refused = int(kg.stats.get("derived", 0)), int(kg.stats.get("refused", 0))
        links = Counter(link.kind.value for link in kg.links)
        coverage = kg.stats.get("coverage")
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
            linked=len(kg.links),
            links=dict(sorted(links.items())),
            derived=derived,
            facts_out=len(kg.facts),
            edges=len(kg.edges),
            properties=len(kg.properties),
            entities=len(kg.entities),
            calls=int(spent.get("calls", 0)),
            tokens=int(spent.get("prompt_tokens", 0)) + int(spent.get("completion_tokens", 0)),
            cost_usd=spent.get("cost_usd"),
            prompts=tuple(str(p) for p in grounding.get("prompts", ())),
            coverage=dict(coverage) if isinstance(coverage, Mapping) else None,
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


def _given(stage: Any, default: Any, *args: Any) -> Any:
    """The stage given, or a fresh default."""
    return stage if stage is not None else default(*args)


def _refused_by(gate: Any) -> dict[str, int]:
    """A gate's refusals by reason, when it counts them as `VerdictGate` does."""
    stats = getattr(gate, "stats", None)
    refused = stats.get("refused") if isinstance(stats, Mapping) else None
    return {str(k): int(v) for k, v in refused.items()} if isinstance(refused, Mapping) else {}


def _spent(grounder: Any, before: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The model calls, tokens and cost a grounder reports: since `before`, when given."""
    stats = getattr(grounder, "stats", None)
    stats = stats if isinstance(stats, Mapping) else {}
    now: dict[str, Any] = {
        key: int(stats.get(key, 0)) for key in ("calls", "prompt_tokens", "completion_tokens")
    }
    cost = stats.get("cost_usd")
    now["cost_usd"] = float(cost) if isinstance(cost, int | float) else None
    if before is None:
        return now
    out = {
        key: now[key] - int(before.get(key, 0))
        for key in ("calls", "prompt_tokens", "completion_tokens")
    }
    earlier = before.get("cost_usd")
    out["cost_usd"] = None if now["cost_usd"] is None else now["cost_usd"] - (earlier or 0.0)
    return out


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


def _n(count: int, noun: str) -> str:
    plural = noun[:-1] + "ies" if noun.endswith("y") else noun + "s"
    return f"{count} {noun if count == 1 else plural}"


__all__ = ["ValidationReport", "Validated", "Validator"]
