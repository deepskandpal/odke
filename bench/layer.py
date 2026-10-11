"""Every extractor with and without the openodke layer (#104), from saved extractions.

    python bench/layer.py prepare runs/cmp/t2k/ont_* runs/cmp/redocred runs/trex/set80 --out OUT
    python bench/layer.py ground OUT --estimate                 # the price, from token counts
    python bench/layer.py ground OUT --budget-usd 1.60          # the one step that calls a model
    python bench/layer.py table OUT                             # the tables; no model

`prepare` extracts nothing. For each prepared set (`odke bench prepare`) it
reads what each extractor already wrote there and writes, under
`OUT/<set>/<system>/`, the raw triples (`triples.jsonl`), a run config that
validates them (`odke.json`) and what the extraction cost (`extraction.json`):

- `openodke`, the reference extractor: its candidates from `facts/grounded.jsonl`
  where the run kept them (T-REx). Otherwise they are read back from
  `predictions/extraction-alone.jsonl`, which keeps labels and no types, so
  each subject takes its predicate's first domain type.
- `lgt`, `neo4j` and `langextract`: `competitors/<system>/facts.jsonl` (`competitors.py`).
- `pattern`: `PatternExtractor`, run here over the set's documents, free.

The run config is the set's own, so the grounder is the set's (`--paper`: the
whole document, True or False), the ground model is `GROUND_MODEL`, and every
stage the set leaves out takes the Validator's default, with three
exceptions, each said where it is set: the gate also checks the schema, as the
Validator's default gate does; the pair judge is off, as it is by default; and
the value normaliser is off, because the gold is in the source's words
(`--value-normalizer` keeps it).

`ground` is the Validator's ground job (`openodke.interop.ground_graph`, what
`odke ground` runs) over each system's triples. The answers go through the
response cache (`--cache`), so a rerun costs nothing, under one USD budget for
the whole command (`openodke.llm.budget`): exit 3 when the budget stopped it,
as `odke ground`. Each system gets `ground/facts.jsonl` (every fact with its
verdict), `ground/summary.json` and `ground/cost.json`. The cost is both the
calls made now and the list price of every question asked, answered from the
cache or not, which is what the layer costs without one. `--estimate` prices
the job first and calls nothing: questions the cache holds are free, and the
rest are priced from their length (`estimate_tokens`) at LiteLLM's rate.

`table` makes three rows per system and set, and calls no model:

1. raw: the triples as read;
2. + grounding: the ground job's facts that the config's gate keeps;
3. + resolution and corroboration: `openodke.Validator` over the same triples,
   built from the config, its grounder answered from the cache. A question the
   cache does not hold stops the table rather than calling a model.

Each row is scored with the dataset's own scorer. Text2KGBench's ontologies are
pooled: every sentence is a unit, which is the mean of the per-ontology means
when each has 20. Every precision, recall and F1 carries a 95% percentile
bootstrap over documents, and so does each change against the raw row, drawn
on the same documents for every row (paired). `removed.jsonl` beside each
system lists what the gate removed, each triple judged on its own against the
gold: right, wrong, or (Text2KGBench) unscored because its relation is not one
the sentence's gold uses. `OUT/layer.json` keeps every number the Markdown has.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from openodke import Fact, Ontology, Validator
from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED, draws, percentile_range
from openodke.eval.cost import CostMeter, StageCost
from openodke.eval.datasets._common import (
    doc_names,
    read_jsonl,
    triples_by_doc,
    write_json,
    write_jsonl,
)
from openodke.extract import PatternExtractor
from openodke.ground import LLMGrounder
from openodke.interop import ground_graph
from openodke.interop.triples import read_triples
from openodke.llm.base import Completion, LLMClient, Message, ModelSpec
from openodke.llm.budget import Budget, Ledger, estimate_tokens
from openodke.llm.cache import FORMAT, CachedClient, DirectoryCache, cache_key
from openodke.loaders import DirectoryLoader
from openodke.pipeline import Pipeline
from openodke.run.build import build
from openodke.run.config import RunConfig, load_config
from openodke.types import Document, SpanOrigin

GROUND_MODEL = "anthropic/claude-haiku-4-5-20251001"
SYSTEMS = {
    "openodke": "openodke",
    "lgt": "LLMGraphTransformer",
    "neo4j": "neo4j-graphrag",
    "langextract": "LangExtract",
    "pattern": "PatternExtractor",
}
DATASETS = {"text2kgbench": "Text2KGBench", "redocred": "Re-DocRED", "trex": "T-REx"}
ROWS = ("raw", "+ grounding", "+ resolution and corroboration")
METRICS = ("precision", "recall", "f1")
# A True/False answer: 8 output tokens a call on every saved paper-mode run.
OUTPUT_TOKENS = 8


# --------------------------------------------------------------------------- #
# prepare: the saved extractions as triples, and a config that validates them
# --------------------------------------------------------------------------- #


def set_name(meta: Mapping[str, Any]) -> str:
    """`t2k/ont_1_movie`, `redocred` or `trex`: where a set's systems go under OUT."""
    if meta["dataset"] == "text2kgbench":
        return f"t2k/{meta['ontology_id']}"
    return str(meta["dataset"])


def fact_row(fact: Fact, system: str, names: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A fact as a triples row: what any extractor hands the layer.

    A row names its text by file name: `names` maps a document's id to it,
    else the evidence's file path gives it.
    """
    evidence = fact.evidence[0]
    entity = fact.object_entity
    path = urlparse(evidence.uri).path if evidence.uri else evidence.doc_id
    row: dict[str, Any] = {
        "doc": (names or {}).get(evidence.doc_id) or Path(unquote(path)).stem,
        "subject": fact.subject.label or fact.subject.key,
        "subject_type": fact.subject.type,
        "predicate": fact.predicate,
        "object": (entity.label or entity.key) if entity is not None else fact.object_value,
        "object_type": entity.type if entity is not None else None,
        "polarity": fact.polarity.value,
        # `odke.*` qualifiers are stamps a stage wrote (`odke.ontology`), not the claim's.
        "qualifiers": {k: v for k, v in fact.qualifiers.items() if not k.startswith("odke.")},
        "confidence": fact.confidence,
        "extractor": system,
    }
    span = evidence.span
    if span is not None and evidence.span_origin is SpanOrigin.CITED:
        row |= {"start": span.start, "end": span.end, "quote": span.quote}
    return {k: v for k, v in row.items() if v is not None}


def from_predictions(
    path: Path, labels: Mapping[str, str], ontology: Ontology
) -> list[dict[str, Any]]:
    """openodke's raw triples read back from a saved `predictions/` file.

    Predictions keep the dataset's relation labels and no types: each relation
    goes back to the predicate it labels, and each subject takes its
    predicate's first domain type. Types decide only the entity key here.
    """
    names = {label: name for name, label in labels.items()}
    rows = []
    for line in read_jsonl(path):
        for subject, relation, obj in line["triples"]:
            if not str(subject).strip() or not str(obj).strip():
                continue
            predicate = names.get(relation, relation)
            declared = ontology.predicates.get(predicate)
            row = {"doc": line["id"], "subject": subject, "predicate": predicate, "object": obj}
            if declared is not None and declared.domain:
                row["subject_type"] = declared.domain[0]
            rows.append(row | {"extractor": "openodke"})
    return rows


def pattern_rows(folder: Path, ontology: Ontology) -> list[dict[str, Any]]:
    """`PatternExtractor` over the set's documents: free, and it cites every value."""
    docs = list(DirectoryLoader().load(folder / "docs"))
    graph = Pipeline(ontology, PatternExtractor(), coverage=False, inverses=False).run(docs)
    return [fact_row(fact, "pattern", doc_names(docs)) for fact in graph.facts]


def price(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """List price of the tokens, from LiteLLM's table; None when it has no price."""
    if not (input_tokens or output_tokens):
        return 0.0
    try:
        import litellm

        cost_in, cost_out = litellm.cost_per_token(
            model=model or "", prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
    except Exception:
        return None
    return float(cost_in + cost_out)


def spend(
    model: str | None, calls: int, input_tokens: int, output_tokens: int, usd: float | None
) -> dict[str, Any]:
    if usd is None:
        usd = price(model, input_tokens, output_tokens)
    return {
        "model": model,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": usd,
    }


def cached_extraction(folder: Path, model: str | None) -> dict[str, Any] | None:
    """What a run's extraction cost, from the answers in its own response cache.

    A run read from the cache meters nothing, but each entry keeps the usage it
    was first answered with. Every extraction entry in the set's cache counts,
    a repair turn's included.
    """
    config = json.loads((folder / "odke.json").read_text(encoding="utf-8"))
    directory = config.get("models", {}).get("cache")
    if not directory or not (folder / directory).is_dir():
        return None
    calls = tokens_in = tokens_out = 0
    usd = 0.0
    for path in sorted((folder / directory).glob("*/*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if not any(p.startswith("extract") for p in entry.get("prompts", ())):
            continue
        done = entry.get("completion", {})
        calls += 1
        tokens_in += int(done.get("prompt_tokens") or 0)
        tokens_out += int(done.get("completion_tokens") or 0)
        usd += float(done.get("cost_usd") or 0.0)
    return spend(model, calls, tokens_in, tokens_out, usd) if calls else None


def openodke_cost(folder: Path, model: str | None) -> dict[str, Any]:
    """The reference extractor's metered cost: the report's first row, else the cache's."""
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    stage = report["stages"][0] if "stages" in report else report
    first = stage["breakdown"]["extraction alone"]
    metered = spend(
        model,
        int(first.get("model_calls") or 0),
        int(first.get("prompt_tokens") or 0),
        int(first.get("completion_tokens") or 0),
        first.get("cost_usd") or None,
    )
    if metered["input_tokens"]:
        return metered | {"from": "report.json"}
    cached = cached_extraction(folder, model)
    return (cached or metered) | {"from": "the response cache" if cached else "report.json"}


def sources(
    folder: Path, meta: Mapping[str, Any], ontology: Ontology
) -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """`(system, raw triples, extraction cost)` for every extractor saved on the set."""
    config = json.loads((folder / "odke.json").read_text(encoding="utf-8"))
    extract = config.get("models", {}).get("extract")
    model = extract.get("model") if isinstance(extract, dict) else extract
    documents = len(list((folder / "docs").glob("*.txt")))
    out = []
    if (folder / "facts" / "grounded.jsonl").is_file():
        facts = [Fact.model_validate(r) for r in read_jsonl(folder / "facts" / "grounded.jsonl")]
        out.append(
            ("openodke", [fact_row(f, "openodke") for f in facts], openodke_cost(folder, model))
        )
    elif (folder / "predictions" / "extraction-alone.jsonl").is_file():
        predictions = folder / "predictions" / "extraction-alone.jsonl"
        rows = from_predictions(predictions, meta["relation_labels"], ontology)
        out.append(("openodke", rows, openodke_cost(folder, model)))
    for system in ("lgt", "neo4j", "langextract"):
        saved = folder / "competitors" / system
        if (saved / "facts.jsonl").is_file():
            usage = json.loads((saved / "usage.json").read_text(encoding="utf-8"))
            if usage.get("documents", documents) < documents:  # a --limit run
                print(f"{folder}: {system} ran on {usage['documents']} documents; left out")
                continue
            cost = spend(
                usage.get("model") or model,
                int(usage.get("calls") or 0),
                int(usage.get("input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
                usage.get("usd"),
            )
            rows = [dict(r) | {"extractor": system} for r in read_jsonl(saved / "facts.jsonl")]
            out.append((system, rows, cost | {"from": "usage.json"}))
    out.append(("pattern", pattern_rows(folder, ontology), spend(None, 0, 0, 0, 0.0)))
    return out


def run_config(
    folder: Path, triples: Path, system: str, *, value_normalizer: bool = False
) -> dict[str, Any]:
    """The set's run config, with the system's triples as the extract stage."""
    config = json.loads((folder / "odke.json").read_text(encoding="utf-8"))
    stages = dict(config["stages"])
    gate = stages.pop("validator", None) or stages.pop("gate", None) or {"use": "verdict"}
    gate = {"use": gate} if isinstance(gate, str) else dict(gate)
    # The Validator's default gate refuses what the ontology has no room for;
    # the set's gate predates it.
    stages["gate"] = gate | {"schema": True}
    stages["extractor"] = {"use": "triples", "path": str(triples.resolve()), "extractor": system}
    resolver = stages.get("resolver")
    if isinstance(resolver, dict):
        # The pair judge is off unless asked for (DECISIONS #34).
        stages["resolver"] = {k: v for k, v in resolver.items() if k != "judge"}
    if not value_normalizer:
        # The gold is in the source's words; the value normaliser would rewrite
        # "13 March 1963" as 1963-03-13 and score it wrong (`_common.run_config`).
        stages["normalizer"] = "passthrough"
    models = {k: v for k, v in config.get("models", {}).items() if k == "extract"}
    return {
        "ontology": str((folder / "ontology.json").resolve()),
        "inputs": [{"path": str((folder / "docs").resolve()), "loader": "directory"}],
        "stages": stages,
        "models": models | {"ground": GROUND_MODEL},
    }


def prepare(sets: Sequence[Path], out: Path, *, value_normalizer: bool = False) -> list[Path]:
    """Each set's systems under `out`, ready to ground. Extracts nothing.

    `value_normalizer` keeps the Validator's default value normaliser on.
    """
    made = []
    for folder in sets:
        meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        ontology = Ontology.from_json(folder / "ontology.json")
        name = set_name(meta)
        write_json(out / name / "set.json", {"path": str(folder.resolve()), **meta})
        for system, rows, cost in sources(folder, meta, ontology):
            target = out / name / system
            write_jsonl(target / "triples.jsonl", rows)
            config = run_config(
                folder, target / "triples.jsonl", system, value_normalizer=value_normalizer
            )
            write_json(target / "odke.json", config)
            documents = len(list((folder / "docs").glob("*.txt")))
            write_json(target / "extraction.json", cost | {"documents": documents})
            print(f"{name}/{system}: {len(rows)} triples, extraction {_usd(cost['usd'])}")
            made.append(target)
    return made


# --------------------------------------------------------------------------- #
# ground: the Validator's ground job, through the cache, under one budget
# --------------------------------------------------------------------------- #


class Tally(DirectoryCache):
    """A response cache that adds up what each answer it gives cost when first asked."""

    def __init__(self, directory: str | Path) -> None:
        super().__init__(directory)
        self._lock = threading.Lock()
        self.used = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

    def get(self, key: str) -> dict[str, Any] | None:
        entry = super().get(key)
        if entry is not None and entry.get("format") == FORMAT:
            done = entry.get("completion") or {}
            with self._lock:
                self.used["calls"] += 1
                self.used["input_tokens"] += int(done.get("prompt_tokens") or 0)
                self.used["output_tokens"] += int(done.get("completion_tokens") or 0)
                self.used["usd"] += float(done.get("cost_usd") or 0.0)
        return entry


class Probe:
    """Prices each question without asking it: free when the cache holds it, else by length."""

    def __init__(self, store: DirectoryCache) -> None:
        self.store = store
        self.seen: set[str] = set()
        self.cached = self.repeated = self.asked = self.input_tokens = 0
        # On the questions the cache holds: the length estimate, and the real count.
        self.estimated_on_cached = self.real_on_cached = 0
        self._lock = threading.Lock()

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        key = cache_key(messages, spec=spec, schema=schema)
        tokens = estimate_tokens(messages, schema)
        entry = self.store.get(key)
        with self._lock:
            if entry is not None:
                self.cached += 1
                self.estimated_on_cached += tokens
                self.real_on_cached += int(entry["completion"].get("prompt_tokens") or 0)
            elif key in self.seen:
                self.repeated += 1
            else:
                self.asked += 1
                self.input_tokens += tokens
            self.seen.add(key)
        verdict = True if schema and "boolean" in json.dumps(schema) else "supported"
        return Completion(text=json.dumps({"verdict": verdict}), parsed={"verdict": verdict})


def targets(out: Path) -> list[Path]:
    """Every `OUT/<set>/<system>/` that `prepare` wrote, in a stable order."""
    return sorted(p.parent for p in out.glob("**/triples.jsonl"))


def _grounder_options(config: RunConfig) -> dict[str, Any]:
    spec = config.stages.grounder
    if spec is None or spec.use != "llm":
        raise SystemExit(f"{config.source}: the bench grounds with stages.grounder: llm")
    return dict(spec.options)


def _client(
    spec: ModelSpec, ledger: Ledger, store: DirectoryCache, meter: CostMeter, inner: Any
) -> LLMClient:
    """The provider, held to the ledger, behind the cache, metered: as `odke ground` builds it."""
    if inner is None:
        from openodke.llm.limits import LimitedClient
        from openodke.llm.registry import resolve

        inner = LimitedClient(resolve(spec))
    return meter.client(CachedClient(ledger.client(inner), store), "ground")


def _documents(config: RunConfig) -> tuple[Any, list[Document]]:
    built = build(config)
    return built, built.documents()


def ground(
    out: Path,
    *,
    cache: Path,
    budget_usd: float | None = None,
    estimate: bool = False,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Ground every prepared system, or price it with `estimate`; returns the totals."""
    ledger = Ledger(Budget(usd=budget_usd) if budget_usd is not None else None)
    meter = CostMeter()
    store = Tally(cache)
    totals: dict[str, Any] = defaultdict(float)
    stopped = []
    for target in targets(out):
        config = load_config(target / "odke.json")
        built, docs = _documents(config)
        rows = list(read_triples(target / "triples.jsonl"))
        system = target.name
        label = f"{target.parent.relative_to(out)}/{system}"
        options = _grounder_options(config)
        roles = built.context.roles
        if estimate:
            probe = Probe(DirectoryCache(cache))
            grounder = LLMGrounder(roles, client=probe, **options)
            ground_graph(rows, docs, ontology=built.ontology, grounder=grounder, extractor=system)
            usd = price(roles.ground.model, probe.input_tokens, probe.asked * OUTPUT_TOKENS) or 0.0
            for key in ("cached", "repeated", "asked", "input_tokens"):
                totals[key] += getattr(probe, key)
            totals["estimated_on_cached"] += probe.estimated_on_cached
            totals["real_on_cached"] += probe.real_on_cached
            totals["usd"] += usd
            print(
                f"{label}: {len(rows)} triples; {probe.cached} questions in the cache, "
                f"{probe.repeated} repeated, {probe.asked} to ask "
                f"(~{probe.input_tokens:,} input tokens): ~{_usd(usd)}"
            )
            continue
        before_calls, before_cache = len(meter.records), dict(store.used)
        grounder = LLMGrounder(
            roles, client=_client(roles.ground, ledger, store, meter, client), **options
        )
        grounded = ground_graph(
            rows, docs, ontology=built.ontology, grounder=grounder, extractor=system
        )
        grounded.write(target / "ground")
        fresh = StageCost.of("ground", [r for r in meter.records[before_calls:] if not r.cached])
        answered = {k: store.used[k] - before_cache[k] for k in store.used}
        cost = {
            "model": roles.ground.model,
            "prompts": sorted(grounder.stats.get("prompts", {})),
            "documents": len(docs),
            "now": {
                "calls": fresh.calls,
                "input_tokens": fresh.prompt_tokens,
                "output_tokens": fresh.completion_tokens,
                "usd": fresh.priced_usd,
            },
            "from_cache": answered,
            "list": {
                "calls": fresh.calls + answered["calls"],
                "input_tokens": fresh.prompt_tokens + answered["input_tokens"],
                "output_tokens": fresh.completion_tokens + answered["output_tokens"],
                "usd": fresh.priced_usd + answered["usd"],
            },
            "stopped": grounded.summary.stopped,
        }
        write_json(target / "ground" / "cost.json", cost)
        for key, value in cost["now"].items():
            totals[key] += value
        print(
            f"{label}: {len(grounded.facts)} grounded; {fresh.calls} calls now "
            f"({_usd(fresh.priced_usd)}), {answered['calls']} from the cache"
        )
        if grounded.summary.stopped:
            stopped.append(label)
            break
    totals["stopped"] = stopped
    return dict(totals)


# --------------------------------------------------------------------------- #
# table: three rows each, scored, no model
# --------------------------------------------------------------------------- #


def scoring_for(folder: Path, meta: Mapping[str, Any]) -> Any:
    kind = meta["dataset"]
    gold = read_jsonl(folder / "gold.jsonl")
    module = importlib.import_module(f"openodke.eval.datasets.{kind}")
    return module.scoring(gold, meta)


def validated(
    config: RunConfig,
    built: Any,
    rows: list[Any],
    docs: list[Document],
    cache: Path,
    system: str,
) -> list[Fact]:
    """`openodke.Validator` over the rows, built from the config, its answers the cache's only.

    Built as `odke validate --config` builds it. The ledger allows no call, so a
    question the cache does not hold stops the job, and that stops the table.
    """
    stage = built.stages
    meter = CostMeter()
    client = _client(
        built.context.roles.ground,
        Ledger(Budget(calls=0)),
        DirectoryCache(cache),
        meter,
        _Refuse(),
    )
    validator = Validator(
        built.ontology,
        grounder=LLMGrounder(built.context.roles, client=client, **_grounder_options(config)),
        normalizer=stage["normalizer"],
        resolver=stage["resolver"],
        corroborator=stage["corroborator"],
        scorer=stage["scorer"],
        gate=stage["gate"],
        inverses=config.inverses,
        coverage=False,
    )
    graph, report = validator.validate(rows, docs, extractor=system)
    if report.stopped:
        raise SystemExit(f"{config.source}: a question the cache does not hold; run `ground` first")
    return list(graph.facts)


class _Refuse:
    """The provider behind the table's cache: never reached, since the ledger allows no call."""

    def complete(self, messages: Any, *, spec: ModelSpec, schema: Any = None) -> Completion:
        raise RuntimeError("the table asks no model")


def units_of(scoring: Any, facts: Sequence[Fact], docs: Sequence[Document]) -> tuple[list, dict]:
    """The dataset's per-document units for `facts`, and the triples they were made from."""
    labels: Mapping[str, str] = scoring.meta["relation_labels"]
    predicted = triples_by_doc(facts, doc_names(docs), lambda p: labels.get(p, p))
    return scoring.units(predicted), predicted


def judge(scoring: Any, doc: str, triple: tuple[str, str, str]) -> str:
    """One triple against the gold, on its own: `right`, `wrong` or `unscored`."""
    unit = next(u for u in scoring.units({doc: [triple]}) if u["id"] == doc)
    if unit.get("hits", unit.get("tp", 0)):
        return "right"
    return "unscored" if unit.get("unscored") else "wrong"


def removed(scoring: Any, raw: Mapping[str, Sequence], kept: Mapping[str, Sequence]) -> list[dict]:
    """Every raw triple the gate removed, judged against the gold."""
    out = []
    for doc, triples in raw.items():
        left = set(kept.get(doc, ()))
        for triple in triples:
            if tuple(triple) not in left:
                out.append({"doc": doc, "triple": list(triple), "is": judge(scoring, doc, triple)})
    return out


def score_system(target: Path, folder: Path, meta: Mapping[str, Any], cache: Path) -> dict:
    """One system on one set: each row's units, and what grounding removed."""
    if not (target / "ground" / "facts.jsonl").is_file():
        raise SystemExit(f"{target}: not grounded; run `ground` first")
    cost = json.loads((target / "ground" / "cost.json").read_text(encoding="utf-8"))
    if cost.get("stopped"):
        raise SystemExit(f"{target}: its grounding was stopped by the budget; run `ground` again")
    config = load_config(target / "odke.json")
    built, docs = _documents(config)
    scoring = scoring_for(folder, meta)
    grounded = [Fact.model_validate(r) for r in read_jsonl(target / "ground" / "facts.jsonl")]
    gate = built.stages["gate"]
    gated = [f for f in grounded if gate.validate(f, built.ontology).action != "refuse"]
    rows = list(read_triples(target / "triples.jsonl"))
    full = validated(config, built, rows, docs, cache, target.name) if rows else []
    units: dict[str, list] = {}
    predicted: dict[str, dict] = {}
    for name, facts in zip(ROWS, (grounded, gated, full), strict=True):
        units[name], predicted[name] = units_of(scoring, facts, docs)
    gone = removed(scoring, predicted[ROWS[0]], predicted[ROWS[1]])
    write_jsonl(target / "removed.jsonl", gone)
    extraction = json.loads((target / "extraction.json").read_text(encoding="utf-8"))
    return {
        "units": units,
        "facts": {ROWS[0]: len(grounded), ROWS[1]: len(gated), ROWS[2]: len(full)},
        "removed": gone,
        "extraction": extraction,
        "layer": cost,
        "documents": len(docs),
        "aggregate": scoring.aggregate,
    }


def intervals(
    units: Mapping[str, Sequence[Any]],
    aggregate: Callable[[Sequence[Any]], Mapping[str, Any]],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """Each row's metrics with 95% ranges, and its change from the raw row, paired.

    One set of draws over the documents for every row: a row and the raw row
    are recomputed on the same documents, so a change's range is the paired one.
    """
    base = units[ROWS[0]]
    found: dict[str, dict[str, Any]] = {}
    for row, rows in units.items():
        found[row] = {k: aggregate(rows)[k] for k in (*METRICS, "triples")}
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for picked in draws(len(base), resamples=resamples, seed=seed):
        first = aggregate([base[i] for i in picked])
        for row, rows in units.items():
            drawn = aggregate([rows[i] for i in picked])
            for k in METRICS:
                values[(row, k, "range")].append(float(drawn[k]))
                values[(row, k, "change")].append(float(drawn[k]) - float(first[k]))
    for (row, k, kind), found_values in values.items():
        found[row][f"{k}_{kind}"] = list(percentile_range(found_values, LEVEL))
        if kind == "change":
            found[row][f"{k}_change_value"] = float(found[row][k]) - float(found[ROWS[0]][k])
    return found


def table(out: Path, cache: Path, *, resamples: int = RESAMPLES) -> dict[str, Any]:
    """Every dataset's rows, removals and costs, pooled over its sets; written to layer.json."""
    pooled: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for target in targets(out):
        meta = json.loads((target.parent / "set.json").read_text(encoding="utf-8"))
        folder = Path(meta["path"])
        one = score_system(target, folder, meta, cache)
        kind, system = meta["dataset"], target.name
        into = pooled[kind].setdefault(
            system,
            {
                "units": {r: [] for r in ROWS},
                "facts": dict.fromkeys(ROWS, 0),
                "removed": [],
                "documents": 0,
                "extraction": defaultdict(float),
                "layer": defaultdict(float),
                "aggregate": one["aggregate"],
                "sets": 0,
                "prompts": set(),
            },
        )
        for row in ROWS:
            into["units"][row] += one["units"][row]
            into["facts"][row] += one["facts"][row]
        into["removed"] += one["removed"]
        into["documents"] += one["documents"]
        into["sets"] += 1
        into["prompts"].update(one["layer"].get("prompts", ()))
        for key in ("calls", "input_tokens", "output_tokens", "usd"):
            into["extraction"][key] += one["extraction"].get(key) or 0
            into["layer"][key] += one["layer"]["list"].get(key) or 0
        into["layer"]["now_usd"] += one["layer"]["now"]["usd"]
    result: dict[str, Any] = {}
    for kind in sorted(pooled, key=lambda k: list(DATASETS).index(k) if k in DATASETS else 99):
        systems = pooled[kind]
        result[kind] = {}
        for system in sorted(systems, key=lambda s: list(SYSTEMS).index(s) if s in SYSTEMS else 99):
            into = systems[system]
            counts = defaultdict(int)
            for item in into["removed"]:
                counts[item["is"]] += 1
            raw_units = into["units"][ROWS[0]]
            hits = sum(u.get("hits", u.get("tp", 0)) for u in raw_units)
            result[kind][system] = {
                "sets": into["sets"],
                "documents": into["documents"],
                "facts": into["facts"],
                "rows": intervals(into["units"], into["aggregate"], resamples=resamples),
                "removed": dict(counts),
                "raw_right": hits,
                "raw_wrong": _wrong(raw_units),
                "extraction": dict(into["extraction"]),
                "layer": dict(into["layer"]),
                "prompts": sorted(into["prompts"]),
            }
    write_json(out / "layer.json", result)
    return result


def _wrong(units: Sequence[Mapping[str, Any]]) -> int:
    """Scored triples that are not in the gold: Text2KGBench's `over`, else predicted - tp."""
    if units and "over" in units[0]:
        return sum(u["over"] for u in units)
    return sum(u["predicted"] - u["tp"] for u in units)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def _pct(value: Any) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def _ranged(row: Mapping[str, Any], key: str) -> str:
    low, high = row[f"{key}_range"]
    return f"{_pct(row[key])} [{_pct(low)}, {_pct(high)}]"


def _change(row: Mapping[str, Any], key: str) -> str:
    low, high = row[f"{key}_change"]
    value = row[f"{key}_change_value"]
    return f"{100 * value:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"


def _named(system: str, found: Mapping[str, Any], most: int, kind: str) -> str:
    """The extractor's name, and how many of the dataset's documents it ran on when not all."""
    name = SYSTEMS.get(system, system)
    unit = "sentences" if kind == "text2kgbench" else "documents"
    return name if found["documents"] == most else f"{name} ({found['documents']} of {most} {unit})"


def markdown(result: Mapping[str, Any]) -> str:
    """The three tables `docs/benchmarks.md` publishes."""
    most = {kind: max(f["documents"] for f in systems.values()) for kind, systems in result.items()}
    lines: list[str] = []
    for kind, systems in result.items():
        lines += [
            f"### On {DATASETS.get(kind, kind)}",
            "",
            "| Extractor | Row | Facts | Triples | P | R | F1 | ΔP | ΔF1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for system, found in systems.items():
            name = _named(system, found, most[kind], kind)
            if not found["facts"][ROWS[0]]:
                lines.append(f"| {name} | every row | 0 | 0 | – | 0.0 | 0.0 | – | – |")
                continue
            for row in ROWS:
                scored = found["rows"][row]
                first = row == ROWS[0]
                change = (
                    ("–", "–") if first else (_change(scored, "precision"), _change(scored, "f1"))
                )
                lines.append(
                    f"| {name if first else ''} | {row} | {found['facts'][row]} "
                    f"| {scored['triples']} | {_ranged(scored, 'precision')} "
                    f"| {_ranged(scored, 'recall')} | {_ranged(scored, 'f1')} "
                    f"| {change[0]} | {change[1]} |"
                )
        lines.append("")
    lines += [
        "### What grounding removed",
        "",
        "| Dataset | Extractor | Raw wrong | Raw right | Removed | Wrong | Right | Unscored "
        "| Share of wrong removed | Share of right removed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind, systems in result.items():
        for system, found in systems.items():
            gone = found["removed"]
            if found["facts"][ROWS[0]]:
                lines.append(
                    f"| {DATASETS.get(kind, kind)} | {_named(system, found, most[kind], kind)} "
                    f"| {found['raw_wrong']} | {found['raw_right']} | {sum(gone.values())} "
                    f"| {gone.get('wrong', 0)} | {gone.get('right', 0)} "
                    f"| {gone.get('unscored', 0)} "
                    f"| {_share(gone.get('wrong', 0), found['raw_wrong'])} "
                    f"| {_share(gone.get('right', 0), found['raw_right'])} |"
                )
    lines += [
        "",
        "### Cost per 1,000 documents",
        "",
        "| Dataset | Extractor | Extraction | The layer | Layer calls | Layer / extraction |",
        "|---|---|---|---|---|---|",
    ]
    for kind, systems in result.items():
        for system, found in systems.items():
            per = 1000 / found["documents"]
            extraction = found["extraction"].get("usd") or 0.0
            layer = found["layer"].get("usd") or 0.0
            lines.append(
                f"| {DATASETS.get(kind, kind)} | {_named(system, found, most[kind], kind)} "
                f"| {_usd(extraction * per)} | {_usd(layer * per)} "
                f"| {round(found['layer'].get('calls', 0) * per):,} "
                f"| {f'{layer / extraction:.0%}' if extraction else '–'} |"
            )
    return "\n".join(lines)


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "–"


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    made = sub.add_parser("prepare", help="the saved extractions as triples; free")
    made.add_argument("sets", type=Path, nargs="+", help="prepared sets (odke bench prepare)")
    made.add_argument("--out", type=Path, required=True)
    made.add_argument(
        "--value-normalizer", action="store_true", help="keep the Validator's value normaliser"
    )
    go = sub.add_parser("ground", help="the Validator's ground job; the one step that calls")
    go.add_argument("out", type=Path)
    go.add_argument("--cache", type=Path, help="the response cache; default OUT/cache")
    go.add_argument("--budget-usd", type=float, help="stop calling once this much is spent")
    go.add_argument("--estimate", action="store_true", help="price it; call nothing")
    go.add_argument("--spend-log", type=Path, help="append the run's spend as one JSON line")
    go.add_argument("--who", default="ground", help="the spend line's name for this run")
    shown = sub.add_parser("table", help="three rows each, scored; no model")
    shown.add_argument("out", type=Path)
    shown.add_argument("--cache", type=Path, help="the response cache; default OUT/cache")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.sets, args.out, value_normalizer=args.value_normalizer)
        return 0
    cache = args.cache or args.out / "cache"
    if args.command == "table":
        print(markdown(table(args.out, cache)))
        return 0
    totals = ground(args.out, cache=cache, budget_usd=args.budget_usd, estimate=args.estimate)
    if args.estimate:
        asked, tokens = int(totals.get("asked", 0)), int(totals.get("input_tokens", 0))
        print(f"\n{asked} questions to ask, ~{tokens:,} input tokens: ~{_usd(totals.get('usd'))}")
        if totals.get("estimated_on_cached"):
            # The length rule against the real counts of the questions the cache holds.
            scale = totals["real_on_cached"] / totals["estimated_on_cached"]
            scaled = price(GROUND_MODEL, round(tokens * scale), asked * OUTPUT_TOKENS)
            print(
                f"on the questions the cache holds, the real count is {scale:.2f}x the length "
                f"estimate; scaled by it: ~{round(tokens * scale):,} input tokens, ~{_usd(scaled)}"
            )
        return 0
    line = {
        "who": f"numbers #104: {args.who}",
        "model": GROUND_MODEL,
        "calls": int(totals.get("calls", 0)),
        "input_tokens": int(totals.get("input_tokens", 0)),
        "output_tokens": int(totals.get("output_tokens", 0)),
        "usd": round(float(totals.get("usd", 0.0)), 6),
    }
    print(json.dumps(line))
    if args.spend_log is not None:
        with args.spend_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    if totals["stopped"]:
        print(f"stopped by the budget at {totals['stopped'][0]}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
