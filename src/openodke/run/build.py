"""From a config to stage objects: the short names, and the checks that make them safe.

Every built-in is a small factory over a class that already exists, so nothing
here has an idea of its own but `DEFAULTS` — the stages `odke run` fills when
the config leaves them out, because for those the library's pass-through is
misleading rather than merely absent. The part worth reading is what a factory
refuses:
an option the class does not take, an option `odke run` sets itself (the
ontology, the model client), a password in a config file, and an inferrer
(DECISIONS #8). Each refusal is a `ConfigError` naming the key.

A user's own stage is `package.module:Name`. A class is constructed with the
options as keyword arguments; any other object is used as it is. Either way it
must satisfy the stage's Protocol, which is checked here rather than discovered
halfway through a run.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import json
import os
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from openodke._renamed import deprecated
from openodke.chunking import SentenceChunker
from openodke.corroborate import (
    EvidenceScorer,
    NativeResolver,
    PairJudge,
    SignatureCorroborator,
    ValueNormalizer,
)
from openodke.eval.cost import CostMeter
from openodke.extract import HybridExtractor, LLMExtractor, PatternExtractor, RecordMapping
from openodke.gate import VerdictGate
from openodke.ground import LLMGrounder, RetryPolicy, SpanGrounder
from openodke.interop import TriplesExtractor
from openodke.llm.base import Completion, LLMClient, Message, ModelSpec, ProviderNotInstalled
from openodke.llm.budget import Ledger
from openodke.llm.cache import CachedClient, DirectoryCache
from openodke.llm.limits import PROVIDER_LIMITS, LimitedClient
from openodke.llm.registry import resolve as resolve_client
from openodke.llm.roles import ModelRoles
from openodke.llm.testing import RecordedClient, ReplayClient
from openodke.loaders import (
    CsvLoader,
    DirectoryLoader,
    DocxLoader,
    HtmlLoader,
    JsonlLoader,
    JsonLoader,
    MarkdownLoader,
    ParquetLoader,
    PdfLoader,
    TextLoader,
    TsvLoader,
)
from openodke.manifest import Served
from openodke.observe import Observer
from openodke.ontology import Ontology, OntologyLoadError
from openodke.pipeline import Pipeline
from openodke.reextract import Reextract
from openodke.run.config import STAGES, ConfigError, InputSpec, RunConfig, StageSpec
from openodke.sinks.bulk import CypherFileSink, Neo4jAdminCsvSink
from openodke.sinks.jsonl import JsonlSink
from openodke.sinks.neo4j import Neo4jConstrainer, Neo4jLookup, Neo4jSink, is_check, plan
from openodke.sinks.networkx import NetworkXSink
from openodke.sinks.rdf import RdfSink
from openodke.stages import (
    Chunker,
    Constrainer,
    Corroborator,
    Delegated,
    Extractor,
    Gate,
    Grounder,
    Inferrer,
    Loader,
    Normalizer,
    PassThroughChunker,
    PassThroughConstrainer,
    PassThroughCorroborator,
    PassThroughGate,
    PassThroughGrounder,
    PassThroughInferrer,
    PassThroughNormalizer,
    PassThroughResolver,
    PassThroughRouter,
    PassThroughScorer,
    PlatformProfile,
    Resolver,
    Router,
    Scorer,
    Sink,
    StoreLookup,
)
from openodke.types import Document, GroundingVerdict, KnowledgeGraph, SourceTier

PROTOCOLS: dict[str, type] = {
    "loader": Loader,
    "chunker": Chunker,
    "router": Router,
    "extractor": Extractor,
    "grounder": Grounder,
    "normalizer": Normalizer,
    "resolver": Resolver,
    "corroborator": Corroborator,
    "scorer": Scorer,
    "gate": Gate,
    "sink": Sink,
    "constrainer": Constrainer,
    "inferrer": Inferrer,
}


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class Context:
    """What a factory may need beyond its options: the ontology, the models, the meter,
    the response cache, and the ledger every model call is counted on."""

    config: RunConfig
    ontology: Ontology
    roles: ModelRoles
    meter: CostMeter | None = None
    cache: DirectoryCache | None = None
    # The run's budget and what it has spent: every client is counted here.
    ledger: Ledger = field(default_factory=Ledger)
    _replays: dict[str, LLMClient] = field(default_factory=dict)
    # Every cached client handed out, for the run's hit and miss counts.
    cached: list[CachedClient] = field(default_factory=list)
    # The model each role asked for and the ids that answered, for the run manifest.
    served: Served = field(default_factory=Served)
    # The run's events and spans (`openodke.observe`): every stage and every model call.
    observer: Observer = field(default_factory=lambda: Observer("run"))

    def client(self, role: str, *, stage: str | None = None) -> LLMClient:
        """The client a model-backed stage should use for `role`.

        Recorded responses win when the config names them; otherwise the
        provider's client, resolved now, so a missing adapter stops the run
        before anything is spent. Behind a cache it may be missing until the
        first miss: a rerun answered from the cache needs no adapter at all.

        Every call that goes out holds a slot of its provider's process-wide
        limit (`openodke.llm.limits`). The ledger counts every call and stops
        the run at its budget, before a call queues for a slot. The response
        cache, when on, answers in front of both, so a
        repeated call reaches neither a provider nor a recording and costs
        nothing against the budget. A meter, when on, wraps the lot, so cost is
        counted without a stage knowing: under `stage` when given, so the
        grounder's widened retries are a cost row of their own, and under the
        role otherwise. Every answer's model, a cached one's too, is noted
        under the role for the run manifest (`served`). Outermost, the run's
        observer sees every call, a cached one too, as an event and a span
        under the same name.
        """
        inner: LLMClient
        replay = self.config.models.replay.get(role)  # type: ignore[call-overload]
        if replay is not None:
            if role not in self._replays:
                self._replays[role] = _replay_client(self.config.resolve(replay), role)
            inner = self._replays[role]
        else:
            spec = getattr(self.roles, role)
            try:
                inner = resolve_client(spec)
            except ProviderNotInstalled:
                if self.cache is None:
                    raise
                inner = OnFirstCall(spec)
        inner = self.ledger.client(LimitedClient(inner))
        if self.cache is not None:
            inner = CachedClient(inner, self.cache)
            self.cached.append(inner)
        inner = self.served.client(role, getattr(self.roles, role), inner)
        if self.meter is not None:
            inner = self.meter.client(inner, stage=stage or role)
        return self.observer.client(inner, stage or role)

    def cache_stats(self) -> dict[str, Any] | None:
        """The response cache's directory and its hits, misses and failed calls, or None."""
        if self.cache is None:
            return None
        totals = {"hits": 0, "misses": 0, "failed": 0}
        for client in self.cached:
            for key, count in client.stats.items():
                totals[key] = totals.get(key, 0) + count
        return {"directory": str(self.cache.directory), **totals}

    def spent(self) -> dict[str, Any]:
        """Calls, tokens and USD for the run: the ledger's, with the cache's hits among the calls.

        `usd` is None when any call's cost went unreported: unknown, never a
        partial sum.
        """
        spent = self.ledger.spent
        cache = self.cache_stats()
        hits = int(cache["hits"]) if cache is not None else 0
        return {
            "calls": spent.calls + hits,
            "cached_calls": hits,
            "input_tokens": spent.input_tokens,
            "output_tokens": spent.output_tokens,
            "usd": spent.usd,
            "priced_usd": spent.priced_usd,
            "unpriced_calls": spent.unpriced_calls,
        }


class OnFirstCall:
    """The client `resolve(spec)` picks, picked on the first call rather than now."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._client: LLMClient | None = None
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        with self._lock:
            if self._client is None:
                self._client = resolve_client(self.spec)
            client = self._client
        return client.complete(messages, spec=spec, schema=schema)


def response_cache(directory: Path, where: str = "models.cache") -> DirectoryCache:
    """The cache in `directory`, created now, so a path that cannot be one is a config error."""
    try:
        return DirectoryCache(directory)
    except OSError as exc:
        raise ConfigError(f"{where}: cannot use {directory} as a cache directory: {exc}") from None


def _replay_client(path: Path, role: str) -> LLMClient:
    where = f"models.replay.{role}"
    if not path.is_file():
        raise ConfigError(f"{where}: {path} does not exist")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # The two recorded formats share `tests/fixtures/llm/`: a list of match
        # entries, or a cassette object.
        if isinstance(data, list):
            return RecordedClient(data)
        return ReplayClient(path)
    except (ValueError, ValidationError) as exc:
        raise ConfigError(f"{where}: {path} is not a recorded-response file: {exc}") from None


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

Factory = Callable[[dict[str, Any], Context, str], Any]


def construct(
    factory: Callable[..., Any],
    options: Mapping[str, Any],
    where: str,
    *,
    injected: Mapping[str, Any] | None = None,
    reserved: Sequence[str] = (),
) -> Any:
    """`factory(**options, **injected)`, with a bad option explained rather than raised.

    `injected` is what `odke run` supplies itself; `reserved` is what it
    refuses to take from a file (a callable, say). Either set in the options is
    an error naming the key.
    """
    injected = dict(injected or {})
    for key in options:
        if key in injected or key in reserved:
            raise ConfigError(f"{where}.{key}: set by odke run, not by the config")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        signature = None
    if signature is not None:
        try:
            signature.bind(**options, **injected)
        except TypeError as exc:
            accepted = [
                name
                for name, p in signature.parameters.items()
                if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
                and name not in injected
                and name not in reserved
            ]
            takes = f"; it takes {', '.join(accepted)}" if accepted else "; it takes no options"
            raise ConfigError(f"{where}: {exc}{takes}") from None
    try:
        return factory(**options, **injected)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from None


class Extra(NamedTuple):
    """An optional dependency a built-in needs: the module to import, the extra, the package."""

    module: str
    extra: str
    package: str


PYARROW = Extra("pyarrow.parquet", "parquet", "pyarrow")
PYPDF = Extra("pypdf", "pdf", "pypdf")
PYTHON_DOCX = Extra("docx", "docx", "python-docx")
RDFLIB = Extra("rdflib", "rdf", "rdflib")
NETWORKX = Extra("networkx", "networkx", "networkx")


def require_extra(needs: Extra, what: str, where: str) -> None:
    """Import what a built-in needs now, or refuse the config with the extra that provides it.

    The built-ins import their library when they first read or write, which is
    right for a library and late for a run: a PDF input without pypdf would fail
    after the sink had opened. Checked while the config is built, a missing extra
    is a config error before anything is loaded, called or written.
    """
    try:
        importlib.import_module(needs.module)
    except ImportError:
        raise ConfigError(
            f"{where}: {what} needs {needs.package}, which is not installed. "
            f'Run: pip install "openodke[{needs.extra}]"'
        ) from None


def _plain(cls: Callable[..., Any]) -> Factory:
    return lambda options, ctx, where: construct(cls, options, where)


def _with_ontology(cls: Callable[..., Any], *reserved: str) -> Factory:
    return lambda options, ctx, where: construct(
        cls, options, where, injected={"ontology": ctx.ontology}, reserved=reserved
    )


def _tier(options: dict[str, Any], where: str) -> dict[str, Any]:
    if "tier" not in options:
        return options
    try:
        return {**options, "tier": SourceTier(options["tier"])}
    except ValueError:
        tiers = ", ".join(t.value for t in SourceTier)
        raise ConfigError(f"{where}.tier: {options['tier']!r} is not one of {tiers}") from None


def _loader(cls: type, needs: Extra | None = None) -> Factory:
    def make(options: dict[str, Any], ctx: Context, where: str) -> Any:
        if needs is not None:
            require_extra(needs, cls.__name__, where)
        return construct(cls, _tier(options, where), where, reserved=("loaders",))

    return make


def _pattern(options: dict[str, Any], ctx: Context, where: str) -> PatternExtractor:
    mappings = options.get("mappings", [])
    if not isinstance(mappings, list):
        raise ConfigError(f"{where}.mappings: a list of record mappings")
    try:
        parsed = [RecordMapping.model_validate(m) for m in mappings]
    except ValidationError as exc:
        raise ConfigError(f"{where}.mappings: {exc.errors()[0]['msg']}") from None
    rest = {k: v for k, v in options.items() if k != "mappings"}
    pattern: PatternExtractor = construct(
        PatternExtractor, {**rest, "mappings": parsed}, where, reserved=("documents",)
    )
    return pattern


def _llm_extractor(options: dict[str, Any], ctx: Context, where: str) -> LLMExtractor:
    injected = {"client": ctx.client("extract"), "spec": ctx.roles.extract}
    extractor: LLMExtractor = construct(
        LLMExtractor,
        _with_retry(options, where),
        where,
        injected=injected,
        reserved=("roles", "documents"),
    )
    return extractor


def _triples(options: dict[str, Any], ctx: Context, where: str) -> TriplesExtractor:
    if "triples" in options:
        raise ConfigError(f"{where}.triples: name the file with path")
    path = options.get("path")
    if not isinstance(path, str):
        raise ConfigError(f"{where}.path: the JSON Lines file of triples to replay")
    rest = {k: v for k, v in options.items() if k != "path"}
    triples: TriplesExtractor = construct(
        TriplesExtractor,
        {**rest, "triples": ctx.config.resolve(path)},
        where,
        reserved=("documents",),
    )
    return triples


def _hybrid(options: dict[str, Any], ctx: Context, where: str) -> HybridExtractor:
    unknown = sorted(set(options) - {"llm", "pattern"})
    if unknown:
        raise ConfigError(f"{where}.{unknown[0]}: unknown option; hybrid takes llm and pattern")
    llm_options = options.get("llm", {})
    llm = None
    if llm_options is not False:
        if not isinstance(llm_options, Mapping):
            raise ConfigError(f"{where}.llm: options for the model path, or false for none")
        llm = _llm_extractor(dict(llm_options), ctx, f"{where}.llm")
    pattern_options = options.get("pattern", {})
    if not isinstance(pattern_options, Mapping):
        raise ConfigError(f"{where}.pattern: options for the pattern path")
    return HybridExtractor(llm, pattern=_pattern(dict(pattern_options), ctx, f"{where}.pattern"))


def _llm_grounder(options: dict[str, Any], ctx: Context, where: str) -> LLMGrounder:
    injected = {"roles": ctx.roles, "client": ctx.client("ground")}
    if options.get("widen"):
        # The retries' calls, metered as their own stage (#102).
        injected["widen_client"] = ctx.client("ground", stage="ground.widen")
    grounder: LLMGrounder = construct(
        LLMGrounder,
        _with_retry(options, where),
        where,
        injected=injected,
        reserved=("sleep", "widen_client"),
    )
    return grounder


def _native_resolver(options: dict[str, Any], ctx: Context, where: str) -> NativeResolver:
    """`resolver: {use: native, judge: ...}`: the pair judge is off unless named (DECISIONS #34).

    `judge: true` takes its defaults; a mapping sets `low`, `queue` and
    `reviewed` (paths relative to the config), `max_workers` and `retry`. It
    asks the `ground` role's model, metered as the stage `judge`.
    `normalize_batch: true` normalises the batch (DECISIONS #43), with the
    run's ontology for its types' aliases; `embed` is a function, so Python
    only.
    """
    opts = dict(options)
    judge = opts.pop("judge", None)
    if judge is not None and judge is not False:
        if judge is True:
            judge = {}
        if not isinstance(judge, Mapping):
            raise ConfigError(f"{where}.judge: true, or the judge's options")
        opts["judge"] = _pair_judge(dict(judge), ctx, f"{where}.judge")
    resolver: NativeResolver = construct(
        NativeResolver, opts, where, injected={"ontology": ctx.ontology}, reserved=("embed",)
    )
    return resolver


def _pair_judge(options: dict[str, Any], ctx: Context, where: str) -> PairJudge:
    for key in ("queue", "reviewed"):
        if isinstance(options.get(key), str):
            options[key] = ctx.config.resolve(options[key])
    if isinstance(options.get("reviewed"), Path) and not options["reviewed"].is_file():
        raise ConfigError(f"{where}.reviewed: {options['reviewed']} does not exist")
    judge: PairJudge = construct(
        PairJudge,
        _with_retry(options, where),
        where,
        injected={"roles": ctx.roles, "client": ctx.client("ground", stage="judge")},
        reserved=("sleep", "store_context"),
    )
    return judge


def _with_retry(options: dict[str, Any], where: str) -> dict[str, Any]:
    """The options with `retry` read as a `RetryPolicy`, so a bad one names its key."""
    opts = dict(options)
    if "retry" in opts:
        try:
            opts["retry"] = RetryPolicy.model_validate(opts["retry"])
        except ValidationError as exc:
            raise ConfigError(f"{where}.retry: {exc.errors()[0]['msg']}") from None
    return opts


def _scorer(options: dict[str, Any], ctx: Context, where: str) -> EvidenceScorer:
    opts = dict(options)
    if "verdict_weights" in opts:
        weights = opts["verdict_weights"]
        try:
            opts["verdict_weights"] = {GroundingVerdict(k): float(v) for k, v in weights.items()}
        except (AttributeError, ValueError, TypeError):
            raise ConfigError(
                f"{where}.verdict_weights: a weight for each of "
                f"{', '.join(v.value for v in GroundingVerdict)}"
            ) from None
    scorer: EvidenceScorer = construct(EvidenceScorer, opts, where, reserved=("source",))
    return scorer


def _delegated(options: dict[str, Any], ctx: Context, where: str) -> Delegated:
    delegated: Delegated = construct(Delegated, options, where)
    return delegated


_PASSTHROUGH: dict[str, Callable[[], Any]] = {
    "chunker": PassThroughChunker,
    "router": PassThroughRouter,
    "grounder": PassThroughGrounder,
    "normalizer": PassThroughNormalizer,
    "resolver": PassThroughResolver,
    "corroborator": PassThroughCorroborator,
    "scorer": PassThroughScorer,
    "gate": PassThroughGate,
    "constrainer": PassThroughConstrainer,
    "inferrer": PassThroughInferrer,
}

BUILTINS: dict[str, dict[str, Factory]] = {
    "loader": {
        # Every suffix below, and each missing extra a warning that skips its files.
        "directory": _loader(DirectoryLoader),
        "text": _loader(TextLoader),
        "markdown": _loader(MarkdownLoader),
        "html": _loader(HtmlLoader),
        "pdf": _loader(PdfLoader, PYPDF),
        "docx": _loader(DocxLoader, PYTHON_DOCX),
        "csv": _loader(CsvLoader),
        "tsv": _loader(TsvLoader),
        "json": _loader(JsonLoader),
        "jsonl": _loader(JsonlLoader),
        "parquet": _loader(ParquetLoader, PYARROW),
    },
    "chunker": {"sentence": _plain(SentenceChunker)},
    "router": {},
    "extractor": {
        "pattern": _pattern,
        "llm": _llm_extractor,
        "hybrid": _hybrid,
        "triples": _triples,
    },
    "grounder": {"span": _plain(SpanGrounder), "llm": _llm_grounder},
    "normalizer": {"value": _with_ontology(ValueNormalizer)},
    "resolver": {"native": _native_resolver},
    "corroborator": {"signature": _with_ontology(SignatureCorroborator, "source", "documents")},
    "scorer": {"evidence": _scorer},
    "gate": {"verdict": _plain(VerdictGate)},
    "constrainer": {"neo4j": _plain(Neo4jConstrainer)},
    "inferrer": {},
}
for _stage, _cls in _PASSTHROUGH.items():
    BUILTINS[_stage]["passthrough"] = _plain(_cls)
    # A stage the platform does (DECISIONS #21). Not for a chunker: a platform
    # never chunks, and `Delegated` would stand in as a no-split chunker.
    if _stage not in {"chunker", "inferrer"}:
        BUILTINS[_stage]["delegated"] = _delegated

# What `odke run` fills a stage with when the config leaves it out, where the
# pass-through would be a worse answer than an opinion. The library keeps its
# pass-through default (DECISIONS #20); this is the CLI's, and a config that
# names the stage — `passthrough` included — always wins.
#
# `scorer`: without one, `Fact.confidence` arrives at the sink as whatever the
# extractor put there, one constant for every fact whatever the grounder found,
# and a threshold set against it silently accepts everything (#81).
# `EvidenceScorer` is the formula the package already documents.
DEFAULTS: dict[str, StageSpec] = {"scorer": StageSpec(use="evidence")}


def build_stage(stage: str, spec: StageSpec, ctx: Context, where: str) -> Any:
    """One stage from its spec, checked against the stage's Protocol."""
    if stage == "inferrer" and spec.use != "passthrough":
        raise ConfigError(
            f"{where}: odke run never infers an ontology. Inference is a bootstrap, not a "
            "mode (DECISIONS #8): run it once, review the result, and name the file under "
            "`ontology`"
        )
    if spec.is_custom:
        built = _custom(spec, where)
    else:
        table = BUILTINS[stage]
        factory = table.get(spec.use)
        if factory is None:
            names = sorted(table)
            close = difflib.get_close_matches(spec.use, names, n=1)
            hint = f" — did you mean {close[0]!r}?" if close else ""
            known = f"built-ins: {', '.join(names)}; " if names else ""
            raise ConfigError(
                f"{where}: unknown {stage} {spec.use!r}{hint} ({known}or name your own as "
                "package.module:Name)"
            )
        built = factory(dict(spec.options), ctx, where)
    protocol = PROTOCOLS[stage]
    if not isinstance(built, protocol):
        methods = ", ".join(
            name
            for name, value in vars(protocol).items()
            if callable(value) and not name.startswith("_")
        )
        raise ConfigError(
            f"{where}: {spec.use} is not a {protocol.__name__}; it needs a method {methods}()"
        )
    return built


def _custom(spec: StageSpec, where: str) -> Any:
    module_name, _, attribute = spec.use.partition(":")
    if not module_name or not attribute:
        raise ConfigError(f"{where}: a stage of your own is package.module:Name, got {spec.use!r}")
    try:
        found = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(
            f"{where}: cannot load {spec.use!r}: {exc}. Is its directory under `pythonpath`?"
        ) from None
    if isinstance(found, type):
        return construct(found, spec.options, where)
    if spec.options:
        raise ConfigError(f"{where}: {spec.use} is an instance, so it takes no options")
    return found


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #


class SinkPlan:
    """A sink as configured: described without connecting, opened only to write.

    A dry run describes what would be written and never opens a sink, so it
    needs neither a database nor its password. `streams` says whether a run in
    micro-batches can write to it (#158): a sink that writes a file of the
    whole graph would keep the last one.
    """

    name: str = "sink"
    profile: PlatformProfile | None = None
    can_bootstrap: bool = False
    streams: bool = True

    def open(self) -> Sink:  # pragma: no cover - every plan overrides it
        raise NotImplementedError

    def describe(self, kg: KnowledgeGraph) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def streamed(self, graph: Mapping[str, Any], batches: int) -> list[str]:
        """What a streamed run wrote, from the run's counts: `describe` reads one graph."""
        return [f"{self.name}: {_counted(graph)}, in {_batches(batches)}"]

    def ddl(self, ontology: Ontology, constrainer: Constrainer | None) -> list[str]:
        return []


class _JsonlPlan(SinkPlan):
    name = "jsonl"

    def __init__(self, options: dict[str, Any], ctx: Context, where: str) -> None:
        unknown = sorted(set(options) - {"directory", "merge"})
        if unknown:
            raise ConfigError(
                f"{where}.{unknown[0]}: unknown option; jsonl takes directory and merge"
            )
        if not isinstance(options.get("directory"), str):
            raise ConfigError(f"{where}.directory: the directory to write into")
        if not isinstance(options.get("merge", False), bool):
            raise ConfigError(f"{where}.merge: true or false")
        self.directory = ctx.config.resolve(options["directory"])
        self.merge = bool(options.get("merge", False))

    def open(self) -> Sink:
        return JsonlSink(self.directory, merge=self.merge)

    def describe(self, kg: KnowledgeGraph) -> list[str]:
        return [
            f"jsonl → {self.directory}: entities.jsonl {len(kg.entities)}, facts.jsonl "
            f"{len(kg.facts)}, links.jsonl {len(kg.links)}, manifest.json"
        ]

    def streamed(self, graph: Mapping[str, Any], batches: int) -> list[str]:
        how = "merged" if self.merge else "appended"
        links = sum(graph.get("links", {}).values())
        return [
            f"jsonl → {self.directory}: entities.jsonl {graph.get('entities', 0)}, facts.jsonl "
            f"{graph.get('facts', 0)}, links.jsonl {links}, manifest.json; "
            f"{how} in {_batches(batches)}"
        ]


class _Neo4jOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str | None = None
    uri_env: str | None = None
    user: str | None = None
    user_env: str | None = None
    password_env: str = "NEO4J_PASSWORD"
    database: str | None = None
    batch_size: int = 500


class _Neo4jPlan(SinkPlan):
    name = "neo4j"
    profile = Neo4jSink.profile
    can_bootstrap = True

    def __init__(self, options: dict[str, Any], ctx: Context, where: str) -> None:
        if "password" in options:
            raise ConfigError(
                f"{where}.password: a password never goes in a config file. Put it in an "
                "environment variable and name that with password_env"
            )
        try:
            self.options = _Neo4jOptions.model_validate(options)
        except ValidationError as exc:
            first = exc.errors()[0]
            key = ".".join(str(p) for p in first["loc"])
            raise ConfigError(f"{where}.{key}: {first['msg']}") from None
        if not self.options.uri and not self.options.uri_env:
            raise ConfigError(f"{where}: neo4j needs uri or uri_env")
        self.ontology = ctx.ontology
        self.where = where

    @property
    def target(self) -> str:
        return self.options.uri or f"${self.options.uri_env}"

    def open(self) -> Sink:
        opts = self.options
        uri = opts.uri or os.environ.get(opts.uri_env or "")
        if not uri:
            raise ConfigError(f"{self.where}.uri_env: {opts.uri_env} is not set")
        user = opts.user or (os.environ.get(opts.user_env) if opts.user_env else None) or "neo4j"
        password = os.environ.get(opts.password_env)
        if password is None:
            raise ConfigError(f"{self.where}.password_env: {opts.password_env} is not set")
        return Neo4jSink(
            uri,
            (user, password),
            database=opts.database,
            batch_size=opts.batch_size,
            ontology=self.ontology,
        )

    def describe(self, kg: KnowledgeGraph) -> list[str]:
        statements = plan(kg, ontology=self.ontology)
        rows = sum(len(s.rows) for s in statements)
        lines = [f"neo4j → {self.target}: {len(statements)} statements, {rows} rows"]
        for statement in statements:
            merge = next(
                (
                    ln.strip()
                    for ln in statement.cypher.splitlines()
                    if ln.lstrip().startswith(("MERGE", "SET s."))
                ),
                statement.cypher.splitlines()[-1],
            )
            lines.append(f"  {len(statement.rows):>4} × {merge}")
        return lines

    def ddl(self, ontology: Ontology, constrainer: Constrainer | None) -> list[str]:
        compiled = (constrainer or Neo4jConstrainer()).constrain(ontology)
        return [s for s in compiled if not is_check(s)]

    def streamed(self, graph: Mapping[str, Any], batches: int) -> list[str]:
        return [f"neo4j → {self.target}: {_counted(graph)}, in {_batches(batches)}"]


class _WriterPlan(SinkPlan):
    """A built-in sink that connects to nothing: it writes a file, a directory, or memory.

    Constructed once while the config is built, so a bad option is a config
    error before anything runs, and again by `open()`, so a run writes with a
    fresh sink. `odke run` supplies the ontology, which decides whether a
    projected value is one value or a list and, where the format has room,
    declares the schema.
    """

    # The option that names where the sink writes, resolved against the config.
    target_key = "path"
    needs: Extra | None = None
    what = ""

    def __init__(self, options: dict[str, Any], ctx: Context, where: str) -> None:
        if self.needs is not None:
            require_extra(self.needs, self.what, where)
        self.where = where
        self.ontology = ctx.ontology
        self.options = dict(options)
        self.target = self._target(ctx)
        self._make()

    def _target(self, ctx: Context) -> Path | None:
        value = self.options.get(self.target_key)
        if not isinstance(value, str):
            noun = "directory" if self.target_key == "directory" else "file"
            raise ConfigError(f"{self.where}.{self.target_key}: the {noun} to write into")
        path = ctx.config.resolve(value)
        self.options[self.target_key] = path
        return path

    def _construct(
        self, cls: Callable[..., Any], options: Mapping[str, Any], reserved: Sequence[str] = ()
    ) -> Any:
        injected = {"ontology": self.ontology}
        return construct(cls, options, self.where, injected=injected, reserved=reserved)

    def _make(self) -> Any:  # pragma: no cover - every plan overrides it
        raise NotImplementedError

    def open(self) -> Sink:
        sink: Sink = self._make()
        return sink


class _CypherFilePlan(_WriterPlan):
    name = "cypher_file"
    streams = False
    profile = CypherFileSink.profile
    what = "CypherFileSink"

    def _make(self) -> CypherFileSink:
        sink: CypherFileSink = self._construct(CypherFileSink, self.options)
        return sink

    def describe(self, kg: KnowledgeGraph) -> list[str]:
        statements = self._make().statements(kg)
        schema = len(Neo4jConstrainer().schema(self.ontology))
        rows = sum(len(s.rows) for s in plan(kg, ontology=self.ontology))
        return [
            f"cypher_file → {self.target}: {len(statements)} statements ({schema} schema, "
            f"{len(statements) - schema} UNWIND … MERGE), {rows} rows"
        ]


class _Neo4jAdminCsvPlan(_WriterPlan):
    name = "neo4j_admin_csv"
    streams = False
    profile = Neo4jAdminCsvSink.profile
    target_key = "directory"
    what = "Neo4jAdminCsvSink"

    def _make(self) -> Neo4jAdminCsvSink:
        sink: Neo4jAdminCsvSink = self._construct(Neo4jAdminCsvSink, self.options)
        return sink

    def describe(self, kg: KnowledgeGraph) -> list[str]:
        tables = self._make().tables(kg)
        nodes = sum(table.kind == "nodes" for table in tables)
        lines = [
            f"neo4j_admin_csv → {self.target}: {nodes} node files, "
            f"{len(tables) - nodes} relationship files, import.args"
        ]
        lines.extend(f"  {len(table.rows):>4} × {table.file}" for table in tables)
        return lines


class _RdfPlan(_WriterPlan):
    name = "rdf"
    streams = False
    needs = RDFLIB
    what = "RdfSink"

    def _make(self) -> RdfSink:
        sink: RdfSink = self._construct(RdfSink, self.options)
        return sink

    def describe(self, kg: KnowledgeGraph) -> list[str]:
        sink = self._make()
        return [f"rdf → {self.target} ({sink.format}): {len(sink.graph(kg))} triples"]


class _NetworkXPlan(_WriterPlan):
    """A `MultiDiGraph`, kept for the run, and written as node-link JSON when `path` is set."""

    name = "networkx"
    needs = NETWORKX
    what = "NetworkXSink"

    def _target(self, ctx: Context) -> Path | None:
        unknown = sorted(set(self.options) - {"path", "graph"})
        if unknown:
            raise ConfigError(f"{self.where}.{unknown[0]}: unknown option; networkx takes path")
        return super()._target(ctx) if "path" in self.options else None

    def _make(self) -> Sink:
        options = {k: v for k, v in self.options.items() if k != "path"}
        sink: NetworkXSink = self._construct(NetworkXSink, options, reserved=("graph",))
        return sink if self.target is None else NodeLinkFile(sink, self.target)

    def describe(self, kg: KnowledgeGraph) -> list[str]:
        graph = NetworkXSink(ontology=self.ontology).to_graph(kg)
        where = (
            f"{self.target} (node-link JSON)" if self.target else "a MultiDiGraph, kept for the run"
        )
        return [
            f"networkx → {where}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        ]

    def streamed(self, graph: Mapping[str, Any], batches: int) -> list[str]:
        where = f"{self.target} (node-link JSON)" if self.target else "a MultiDiGraph"
        return [f"networkx → {where}: {_counted(graph)}, in {_batches(batches)}"]


class NodeLinkFile:
    """A `NetworkXSink` whose graph is also written as node-link JSON after every write.

    The graph a NetworkX sink fills lives in memory and ends with the process
    that filled it, so from `odke run` it would be written nowhere. This keeps
    it. `networkx.node_link_graph(data, edges="edges")` reads the file back
    (`link="edges"` before networkx 3.4). Dates and times are ISO strings.
    """

    def __init__(self, sink: NetworkXSink, path: Path) -> None:
        self.sink = sink
        self.path = path

    @property
    def graph(self) -> Any:
        return self.sink.graph

    def write(self, kg: KnowledgeGraph) -> None:
        import networkx  # type: ignore[import-untyped]

        self.sink.write(kg)
        edges = (
            "edges" if "edges" in inspect.signature(networkx.node_link_data).parameters else "link"
        )
        data = networkx.node_link_data(self.sink.graph, **{edges: "edges"})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, default=_json_value, ensure_ascii=False)
        self.path.write_text(text + "\n", encoding="utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return str(value)


class _CustomPlan(SinkPlan):
    def __init__(self, spec: StageSpec, where: str) -> None:
        self.spec = spec
        self.where = where
        self.name = spec.use
        module_name, _, attribute = spec.use.partition(":")
        try:
            self.target = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(
                f"{where}: cannot load {spec.use!r}: {exc}. Is its directory under `pythonpath`?"
            ) from None
        profile = getattr(self.target, "profile", None)
        self.profile = profile if isinstance(profile, PlatformProfile) else None
        self.can_bootstrap = callable(getattr(self.target, "bootstrap", None))

    def open(self) -> Sink:
        built = _custom(self.spec, self.where)
        if not isinstance(built, Sink):
            raise ConfigError(f"{self.where}: {self.spec.use} is not a Sink; it needs write()")
        return built

    def describe(self, kg: KnowledgeGraph) -> list[str]:
        return [f"{self.spec.use} → write() with {len(kg.facts)} facts, {len(kg.links)} links"]

    def streamed(self, graph: Mapping[str, Any], batches: int) -> list[str]:
        links = sum(graph.get("links", {}).values())
        return [
            f"{self.spec.use} → write() once a micro-batch, {batches} in all: "
            f"{graph.get('facts', 0)} facts, {links} links"
        ]


def _counted(graph: Mapping[str, Any]) -> str:
    links = sum(graph.get("links", {}).values())
    return (
        f"{graph.get('facts', 0)} facts ({graph.get('edges', 0)} edges, "
        f"{graph.get('properties', 0)} properties), {graph.get('entities', 0)} entities, "
        f"{links} links"
    )


def _batches(count: int) -> str:
    return f"{count} micro-batch{'' if count == 1 else 'es'}"


def check_streams(plans: Sequence[SinkPlan], where: str = "batch_size") -> None:
    """Refuse a streamed run a sink that writes the whole graph on every write (#158)."""
    whole = [plan.name for plan in plans if not plan.streams]
    if whole:
        raise ConfigError(
            f"{where}: {', '.join(whole)} writes a file of the whole graph on every write, so "
            "a run in micro-batches would keep only the last; stream to jsonl or neo4j, or "
            "leave batch_size out"
        )


SINKS: dict[str, Callable[[dict[str, Any], Context, str], SinkPlan]] = {
    "jsonl": _JsonlPlan,
    "neo4j": _Neo4jPlan,
    "cypher_file": _CypherFilePlan,
    "neo4j_admin_csv": _Neo4jAdminCsvPlan,
    "rdf": _RdfPlan,
    "networkx": _NetworkXPlan,
}


def sink_plan(spec: StageSpec, ctx: Context, where: str) -> SinkPlan:
    if spec.is_custom:
        return _CustomPlan(spec, where)
    make = SINKS.get(spec.use)
    if make is not None:
        return make(dict(spec.options), ctx, where)
    names = sorted(SINKS)
    close = difflib.get_close_matches(spec.use, names, n=1)
    hint = f" — did you mean {close[0]!r}?" if close else ""
    raise ConfigError(
        f"{where}: unknown sink {spec.use!r}{hint} (built-ins: {', '.join(names)}; or name "
        "your own as package.module:Name)"
    )


# --------------------------------------------------------------------------- #
# The store lookup
# --------------------------------------------------------------------------- #


class _LookupOptions(BaseModel):
    """`store_lookup: {use: neo4j, ...}`: where to read, and how widely.

    The connection keys are the neo4j sink's. Left out, the lookup reads the
    store the run's one neo4j sink writes to, on that sink's connection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str | None = None
    uri_env: str | None = None
    user: str | None = None
    user_env: str | None = None
    password_env: str = "NEO4J_PASSWORD"
    database: str | None = None
    tenant: str | None = None
    tenant_property: str = "tenant"
    limit: int = 100


class LookupPlan:
    """The store lookup as configured: checked while the config is built, opened only to run.

    A dry run opens no sink and so no store: it resolves each batch within
    itself, and says so. `neo4j` reads through the run's Neo4j sink unless it
    names a `uri` of its own; `package.module:Name` is a `StoreLookup` of yours,
    constructed with its options.
    """

    def __init__(
        self, spec: StageSpec, ctx: Context, sinks: Sequence[SinkPlan], where: str
    ) -> None:
        self.where = where
        self.name = spec.use
        self.ontology = ctx.ontology
        self.custom: StoreLookup | None = None
        self.sink: int | None = None
        self._opened: Any = None
        if spec.is_custom:
            built = _custom(spec, where)
            if not isinstance(built, StoreLookup):
                raise ConfigError(
                    f"{where}: {spec.use} is not a StoreLookup; it needs a method candidates()"
                )
            self.custom = built
            return
        if spec.use != "neo4j":
            close = difflib.get_close_matches(spec.use, ["neo4j"], n=1)
            hint = f" — did you mean {close[0]!r}?" if close else ""
            raise ConfigError(
                f"{where}: unknown store lookup {spec.use!r}{hint} (built-in: neo4j; or name "
                "your own as package.module:Name)"
            )
        if "password" in spec.options:
            raise ConfigError(
                f"{where}.password: a password never goes in a config file. Put it in an "
                "environment variable and name that with password_env"
            )
        try:
            self.options = _LookupOptions.model_validate(spec.options)
        except ValidationError as exc:
            first = exc.errors()[0]
            key = ".".join(str(part) for part in first["loc"])
            raise ConfigError(f"{where}.{key}: {first['msg']}") from None
        if not self.options.uri and not self.options.uri_env:
            stores = [i for i, plan in enumerate(sinks) if plan.name == "neo4j"]
            if len(stores) != 1:
                raise ConfigError(
                    f"{where}: neo4j reads the store the run's neo4j sink writes to, and "
                    f"{'there is none' if not stores else 'there are several'}; name one "
                    "with uri or uri_env"
                )
            self.sink = stores[0]

    def open(self, sinks: Sequence[Sink]) -> StoreLookup:
        """The lookup to resolve with: on the opened sink's connection, or its own."""
        if self.custom is not None:
            return self.custom
        opts = self.options
        scope: dict[str, Any] = {
            "tenant": opts.tenant,
            "tenant_property": opts.tenant_property,
            "limit": opts.limit,
        }
        if self.sink is not None:
            sink = sinks[self.sink]
            if not isinstance(sink, Neo4jSink):  # pragma: no cover - the plan said neo4j
                raise ConfigError(f"{self.where}: the neo4j sink did not open as one")
            lookup: StoreLookup = sink.lookup(**scope)
            return lookup
        uri = opts.uri or os.environ.get(opts.uri_env or "")
        if not uri:
            raise ConfigError(f"{self.where}.uri_env: {opts.uri_env} is not set")
        user = opts.user or (os.environ.get(opts.user_env) if opts.user_env else None) or "neo4j"
        password = os.environ.get(opts.password_env)
        if password is None:
            raise ConfigError(f"{self.where}.password_env: {opts.password_env} is not set")
        self._opened = Neo4jLookup(
            uri, (user, password), database=opts.database, ontology=self.ontology, **scope
        )
        return self._opened

    def close(self) -> None:
        """Close a connection this plan opened; one shared with a sink closes with the sink."""
        if self._opened is not None:
            self._opened.close()
            self._opened = None

    @property
    def dry_run_note(self) -> str:
        return (
            f"{self.where}: a dry run opens no store, so each batch was resolved within "
            "itself; the run looks the store up"
        )


# --------------------------------------------------------------------------- #
# The whole configuration
# --------------------------------------------------------------------------- #


@dataclass
class Built:
    """A config turned into objects: the ontology, one object per stage, and the sink plans.

    `stages` holds `None` for a stage the config left out, so `pipeline()` gives
    it the pass-through and the stats say nothing about it — except for the
    stages in `DEFAULTS`, which `odke run` fills itself.
    """

    config: RunConfig
    ontology: Ontology
    context: Context
    stages: dict[str, Any]
    sinks: list[SinkPlan]
    # `store_lookup`, opened with the sinks and handed to the resolver.
    lookup: LookupPlan | None = None

    def open_lookup(self, sinks: Sequence[Sink]) -> StoreLookup | None:
        """Hand the resolver its store lookup, on the opened sinks; None when none is set."""
        if self.lookup is None:
            return None
        opened = self.lookup.open(sinks)
        self.stages["resolver"].lookup = opened
        return opened

    def pipeline(self, *, sinks: Sequence[Sink] = (), **overrides: Any) -> Pipeline:
        if "validator" in overrides:
            deprecated("Built.pipeline(validator=...)", "Built.pipeline(gate=...)")
            overrides["gate"] = overrides.pop("validator")
        s = {**self.stages, **overrides}
        return Pipeline(
            self.ontology,
            s["extractor"],
            chunker=s["chunker"],
            router=s["router"],
            grounder=s["grounder"],
            normalizer=s["normalizer"],
            resolver=s["resolver"],
            corroborator=s["corroborator"],
            scorer=s["scorer"],
            gate=s["gate"],
            constrainer=s["constrainer"],
            sinks=sinks,
            inverses=self.config.inverses,
            coverage=self.config.coverage,
            # An extractor passed in replays one that already ran (the ablation's),
            # and asking it again would be a second extraction pass.
            reextract=None
            if "extractor" in overrides
            else reextract_policy(self.config, s["extractor"]),
            observer=self.context.observer,
        )

    def iter_documents(self) -> Iterator[Document]:
        """`documents()`, one at a time as they are asked for: what a streamed run reads (#158).

        Every input is checked before the first is read, so a missing one
        stops the run before anything is written. A loader that reads lazily,
        as the directory and JSON Lines loaders do, holds one file at a time.
        """
        sources: list[tuple[Any, Path]] = []
        for i, item in enumerate(self.config.inputs):
            spec, where = input_loader(self.config, i, item)
            loader = build_stage("loader", spec, self.context, where)
            path = self.config.resolve(item.path)
            if not path.exists():
                raise ConfigError(f"inputs[{i}].path: {path} does not exist")
            sources.append((loader, path))
        return iter_path_ids((loader.load(path) for loader, path in sources), self.config.base_dir)

    def documents(self) -> list[Document]:
        """Every input through its loader, with ids you can label and query by.

        A document read from a file gets the file's path relative to the config
        as its id, plus `#L<line>` for a record with a source line or
        `#<row>` for one without: `corpus/register.csv#L2`. A labelled set can
        name them, and "every fact from this source" is a query on a string
        you already know rather than on a UUID minted by the run.
        """
        docs: list[Document] = []
        for i, item in enumerate(self.config.inputs):
            spec, where = input_loader(self.config, i, item)
            loader = build_stage("loader", spec, self.context, where)
            path = self.config.resolve(item.path)
            if not path.exists():
                raise ConfigError(f"inputs[{i}].path: {path} does not exist")
            docs.extend(loader.load(path))
        return with_path_ids(docs, self.config.base_dir)


def reextract_policy(config: RunConfig, extractor: Any) -> Reextract | None:
    """The config's `reextract:` as a policy, or None. An extractor that cannot is refused."""
    if config.reextract is None:
        return None
    if not callable(getattr(extractor, "reextract", None)):
        raise ConfigError(
            f"reextract: the extractor ({type(extractor).__name__}) has no "
            "reextract(window, relations, already, ontology); llm and hybrid have one"
        )
    return Reextract(windows=config.reextract.windows)


def input_loader(config: RunConfig, index: int, item: InputSpec) -> tuple[StageSpec, str]:
    """The loader an input is read with, and the key a problem with it is reported under."""
    if item.loader is not None:
        return item.loader, f"inputs[{index}].loader"
    return config.stages.loader or StageSpec(use="directory"), "stages.loader"


def with_path_ids(docs: Sequence[Document], base: Path) -> list[Document]:
    """The documents re-keyed by source path, as `Built.documents` describes."""
    seen: dict[str, int] = {}
    out: list[Document] = []
    for doc in docs:
        name = _path_id(doc, base)
        if name is None:
            out.append(doc)
            continue
        count = seen[name] = seen.get(name, 0) + 1
        out.append(doc.model_copy(update={"id": name if count == 1 else f"{name}~{count}"}))
    return out


def iter_path_ids(inputs: Iterable[Iterable[Document]], base: Path) -> Iterator[Document]:
    """`with_path_ids` over each input's documents in turn, one at a time.

    It counts per file rather than per document, so a streamed run holds a
    number for each file, not an id for each record. A record's line or row
    already sets it apart within its file, so within one input only a
    whole-file name is counted; across inputs, a file read again counts on
    from the highest count an earlier input gave it, which is the id
    `with_path_ids` gives it.
    """
    top: dict[str, int] = {}
    for docs in inputs:
        whole: dict[str, int] = {}
        reached: dict[str, int] = {}
        for doc in docs:
            parts = _path_parts(doc, base)
            if parts is None:
                yield doc
                continue
            file, suffix = parts
            name = file + suffix
            if suffix:
                count = top.get(file, 0) + 1
            else:
                whole[name] = whole.get(name, 0) + 1
                count = top.get(file, 0) + whole[name]
            reached[file] = max(reached.get(file, 0), count)
            yield doc.model_copy(update={"id": name if count == 1 else f"{name}~{count}"})
        for file, count in reached.items():
            top[file] = max(top.get(file, 0), count)


def _path_id(doc: Document, base: Path) -> str | None:
    parts = _path_parts(doc, base)
    return None if parts is None else "".join(parts)


def _path_parts(doc: Document, base: Path) -> tuple[str, str] | None:
    """A document's file, relative to `base`, and the `#L<line>` or `#<row>` of its record."""
    if not doc.uri or not doc.uri.startswith("file:"):
        return None
    path = Path(unquote(urlsplit(doc.uri).path))
    try:
        name = path.relative_to(base).as_posix()
    except ValueError:
        name = path.as_posix()
    line, row = doc.metadata.get("line"), doc.metadata.get("row_index")
    if isinstance(line, int):
        return name, f"#L{line}"
    if isinstance(row, int):
        return name, f"#{row}"
    return name, ""


def build(config: RunConfig) -> Built:
    """Every stage the config names, constructed and checked; nothing is loaded or called."""
    for entry in config.pythonpath:
        directory = str(config.resolve(entry))
        if directory not in sys.path:
            sys.path.insert(0, directory)
    # Process-wide, as a provider's limit is: every stage, and every run here.
    PROVIDER_LIMITS.update(config.models.limits)
    ontology = _ontology(config)
    context = Context(
        config=config,
        ontology=ontology,
        roles=config.models.roles(),
        meter=CostMeter() if config.models.meter else None,
        ledger=Ledger(config.models.budget),
        cache=(
            response_cache(config.resolve(config.models.cache))
            if config.models.cache is not None
            else None
        ),
    )
    # Each input's loader is built here too and discarded, so an unknown loader or
    # a missing extra stops the config before a sink is opened.
    for i, item in enumerate(config.inputs):
        spec, where = input_loader(config, i, item)
        build_stage("loader", spec, context, where)
    stages: dict[str, Any] = {}
    for stage in STAGES:
        if stage in {"loader", "sink"}:
            continue
        filled = getattr(config.stages, stage) or DEFAULTS.get(stage)
        stages[stage] = (
            None if filled is None else build_stage(stage, filled, context, f"stages.{stage}")
        )
    sinks = [
        sink_plan(
            spec, context, f"stages.sink[{i}]" if len(config.stages.sink) > 1 else "stages.sink"
        )
        for i, spec in enumerate(config.stages.sink)
    ]
    reextract_policy(config, stages["extractor"])
    if config.batch_size is not None:
        check_streams(sinks)
    if config.bootstrap and not any(p.can_bootstrap for p in sinks):
        raise ConfigError(
            "bootstrap: true needs a sink that applies constraints (neo4j); "
            f"configured: {', '.join(p.name for p in sinks) or 'no sink'}"
        )
    lookup = None
    if config.store_lookup is not None:
        # The store is looked up by the resolver: the native one, unless one is named.
        if stages["resolver"] is None:
            native = StageSpec(use="native")
            stages["resolver"] = build_stage("resolver", native, context, "stages.resolver")
        if not hasattr(stages["resolver"], "lookup"):
            raise ConfigError(
                f"store_lookup: the resolver looks the store up, and "
                f"{type(stages['resolver']).__name__} takes no lookup; use stages.resolver: "
                "native, or a resolver of your own with a `lookup` attribute"
            )
        lookup = LookupPlan(config.store_lookup, context, sinks, "store_lookup")
    return Built(
        config=config,
        ontology=ontology,
        context=context,
        stages=stages,
        sinks=sinks,
        lookup=lookup,
    )


def _ontology(config: RunConfig) -> Ontology:
    path = config.resolve(config.ontology)
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return Ontology.from_yaml(path)
        return Ontology.from_json(path)
    except OntologyLoadError as exc:
        raise ConfigError(f"ontology: {exc}") from None


__all__ = [
    "BUILTINS",
    "DEFAULTS",
    "PROTOCOLS",
    "SINKS",
    "Built",
    "Context",
    "Extra",
    "LookupPlan",
    "NodeLinkFile",
    "OnFirstCall",
    "SinkPlan",
    "build",
    "build_stage",
    "check_streams",
    "construct",
    "input_loader",
    "require_extra",
    "response_cache",
    "iter_path_ids",
    "sink_plan",
    "with_path_ids",
]
