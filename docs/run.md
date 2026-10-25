# `odke run`

`odke run` runs the pipeline from one config: the inputs, the ontology, the
models, each of the thirteen stages, and where the graph goes.

```bash
odke run examples/triples/odke.yaml                          # run it and write the graph
odke run examples/triples/odke.yaml --dry-run                # load, extract and ground; print what would be written
odke run examples/triples/odke.yaml --model ollama/llama3.1  # every role on one model, whatever the config says
odke run examples/triples/odke.yaml --budget-usd 1.50        # stop cleanly before spending more, keeping what is done
odke run --from-manifest examples/triples/out                # run again what a run's manifest recorded
odke run examples/triples/odke.yaml --log-format json        # every stage, document and model call as a JSON line
odke run examples/triples/odke.yaml --batch-size 500         # stream: 500 documents at a time, in bounded memory
```

| Exit status | Means |
|---|---|
| 0 | the run finished (a dry run included) |
| 2 | the config cannot run: a bad key, missing file or missing extra |
| 1 | the run failed: a provider error outside any one document, every document failing, or a file error |
| 3 | the run stopped at its [budget](#budgets), and wrote what it kept |

## The config

The smallest config replays another extractor's triples:

```yaml
# examples/triples/odke.yaml
ontology: ontology.json            # paths are relative to this file
inputs:
  - path: texts                    # texts/halden.txt is the text the rows call "halden"
    loader: {use: directory}
models:
  ground: anthropic/claude-haiku-4-5-20251001
  replay: {ground: recorded/ground.json}
stages:
  extractor: {use: triples, path: triples.jsonl, extractor: hand-written}
  grounder: llm
  gate: verdict
  sink: {use: jsonl, directory: out}
```

YAML needs the `yaml` extra; JSON needs nothing. Paths resolve against the
config's directory.
[`examples/run.yaml`](https://github.com/deepskandpal/odke/blob/main/examples/run.yaml)
comments every key.

| Key | Default | What it is |
|---|---|---|
| `ontology` | required | A JSON or YAML ontology, loaded strictly |
| `inputs` | required | Paths, or `{path, loader}` ([below](#inputs)) |
| `stages` | required | Each stage's implementation; only `extractor` is required ([below](#stages)) |
| `models` | | The models and their calls ([below](#models)) |
| `pythonpath` | | Put on `sys.path` before your own stages are imported |
| `bootstrap` | `false` | [Apply the constraints](#bootstrap-and-the-constrainer) first |
| `inverses` | when declared | `false` turns off [inverse partners](concepts.md#inverse-and-symmetric-partners) |
| `coverage` | `true` | [The coverage report](grounding.md#what-extraction-left-behind-the-coverage-report) |
| `reextract` | off | `true` or `{windows: N}`: [the re-extract hook](grounding.md#handing-a-gap-back-the-re-extract-hook) |
| `store_lookup` | off | Resolve against the store ([below](#store_lookup)) |
| `tenant` | off | The store's [tenant](stores.md#tenants); `--tenant` overrides it |
| `manifest` | | Where else to write the [run manifest](#the-run-manifest) |
| `batch_size` | off | [Stream](#streaming) N documents at a time; `--batch-size` overrides it |

### `inputs`

An input's `loader` is a [loader](loading.md#loaders)'s short name, with
options as extra keys. Without one, `stages.loader` reads it, else the
[`directory`](loading.md#directories) loader. A document's id is its file's
path relative to the config, plus `#L<line>` (CSV, TSV, JSONL) or `#<row>`
(JSON, Parquet) for a record, and `~2`, `~3` for a repeat:
`corpus/register.csv#L2`.

### `models`

| Key | What it is |
|---|---|
| `extract`, `ground`, `infer` | A model string or a `ModelSpec` (`model`, `temperature`, `max_tokens`, `timeout`, `base_url`, `api_key_env`, `extra`); left out, the [role's default](models.md#roles-and-defaults). Unset, `temperature` is left out of the request |
| `replay` | A role's recorded responses: a cassette (`ReplayClient`) or match entries (`RecordedClient`) |
| `meter` | `true` puts calls, tokens, USD and latency per stage in the stats |
| `cache` | A directory of [model answers](models.md#the-response-cache); `--cache` overrides it |
| `budget` | `usd`, `calls`, `input_tokens`, `output_tokens` ([Budgets](#budgets)) |
| `limits` | Calls in flight per provider: `{anthropic: 8}` ([Concurrency](models.md#concurrency-per-provider)) |

Keys come from the environment ([Keys](models.md#keys)). `--model` puts every
role on one model string, keeping each role's `max_tokens`; `--model-provider`
supplies a missing prefix.

### `stages`

A stage is a built-in's short name or your own `package.module:Name`, with
options as extra keys: `grounder: {use: llm, max_workers: 4}`. An unknown
option is an error listing the ones taken. Your own class is constructed with
the options. Every stage is checked against its Protocol when the config is
built.

- **A stage left out** is the pass-through ([DECISIONS #20](decisions.md#20)),
  except the scorer, which is `evidence` so confidence reflects the verdict;
  `scorer: passthrough` opts out.
- **`delegated`**, with `to:`, marks a stage the store does
  ([DECISIONS #21](decisions.md#21)):
  `resolver: {use: delegated, to: neo4j-graphrag:FuzzyMatchResolver}`.
- **What `odke run` supplies** (the ontology, the model client, the documents)
  and options that must be Python objects, such as `source`, are refused.

| Stage | Short names | Options |
|---|---|---|
| `loader` | `directory`, `text`, `markdown`, `html`, `pdf`, `docx`, `csv`, `tsv`, `json`, `jsonl`, `parquet` | each loader's own ([Loading documents](loading.md#loaders)) |
| `chunker` | `sentence`, `passthrough` | `max_words`, `overlap` |
| `router` | `passthrough`, `delegated` | — |
| `extractor` | `pattern`, `llm`, `hybrid`, `triples` | `pattern`: `mappings`, `subject_type`, `confidence`; `llm`: `types`, `snippet_limit`, `confidence`, `repairs`, `structured`, `retry`, `max_workers`; `hybrid`: `llm` (options, or `false`), `pattern`; `triples`: `path` (required), `extractor`, `confidence` ([Inputs](inputs.md#from-odke-run)) |
| `grounder` | `span`, `llm`, `passthrough`, `delegated` | `llm`: `max_workers`, [`retry`](grounding.md#retries), `context`, `verdicts`, `locate`, `widen` ([Grounding](grounding.md#the-second-model-llmgrounder)); `--widen` sets `widen` |
| `normalizer` | `value`, `passthrough`, `delegated` | `day_first`, `person_types` |
| `resolver` | `native`, `passthrough`, `delegated` | `threshold`, `nudge_up`, `nudge_down`, `max_block`, `judge` ([below](#the-pair-judge)), `normalize_batch`, `context_floor` ([Resolve](resolution-and-corroboration.md#resolve)) |
| `corroborator` | `signature`, `passthrough`, `delegated` | `half_life_days`, `freshness_floor`, `intervals`, `near_duplicates` |
| `scorer` | `evidence` (the default), `passthrough`, `delegated` | `prior`, `verdict_weights` |
| `gate` | `verdict`, `passthrough`, `delegated` | `refuse_not_found`, `schema` ([the gate](grounding.md#the-gate-verdictgate)) |
| `sink` | `jsonl`, `neo4j`, `cypher_file`, `neo4j_admin_csv`, `rdf`, `networkx` | [below](#sinks) |
| `constrainer` | `neo4j`, `passthrough`, `delegated` | — |
| `inferrer` | `passthrough` only | [below](#the-inferrer) |

### Sinks

`sink` takes one sink or a list; none writes nothing. Every sink but `jsonl`
gets the ontology. A bad option or a missing extra fails while the config is
built.

| `use` | Options |
|---|---|
| [`jsonl`](stores.md#jsonl) | `directory` (required); `merge` (default `false`): keep what the files hold, replacing what the run restates |
| [`neo4j`](stores.md#neo4j) | `uri` or `uri_env` (one required); `user` or `user_env` (default `neo4j`); `password_env` (default `NEO4J_PASSWORD`); `database`; `batch_size`, rows a write transaction (default 500) |
| [`cypher_file`](stores.md#cypher-file) | `path` (required); `batch_size` (default 500) |
| [`neo4j_admin_csv`](stores.md#neo4j-admin-csv) | `directory` (required); `delimiter` (default `,`); `array_delimiter` (default `;`) |
| [`rdf`](stores.md#rdf) | `path` (required); `format` (`turtle`, `nt`, `json-ld`; default from the suffix); `base`; `schema` |
| [`networkx`](stores.md#networkx) | `path`: where to write the graph as node-link JSON, read back by `networkx.node_link_graph(data, edges="edges")` |

`password:` is refused; `password_env` names the variable, read when the sink
opens, which a dry run never does.

### `bootstrap` and the constrainer

`bootstrap: true` applies the constrainer's DDL (`stages.constrainer`, else
`Neo4jConstrainer`) through the sink before the first write and before any model
call. Every statement is `IF NOT EXISTS`. A config with no sink that applies
constraints (`neo4j`) is refused.

### `store_lookup`

`store_lookup` hands the resolver a
[store lookup](resolution-and-corroboration.md#resolving-against-the-store)
(bringing `resolver: native` if none is named). `neo4j` reads the run's neo4j
sink's store, on its connection, in the run's tenant; its options are `limit`
(default 100) and the sink's connection keys, for another store.
`package.module:Name` is your own `StoreLookup`. A dry run opens no store and
warns. `odke validate --config` reads the same key.

### The pair judge

`resolver: {use: native, judge: true}` turns on the
[pair judge](resolution-and-corroboration.md#the-pair-judge). A mapping sets
`low` (default 0.7), `queue`, `reviewed` (which must exist), `max_workers` and
`retry`. It asks the `ground` role's model and replay, metered as `judge`. A dry run asks it but writes no queue.

```yaml
stages:
  resolver:
    use: native
    judge: {queue: review/pairs.jsonl, reviewed: review/reviewed.jsonl}
```

### The inferrer

`inferrer` accepts only `passthrough`: `odke run never infers an ontology`.
Run `odke ontology infer` once, review and freeze the result, and name it under
`ontology` ([DECISIONS #8](decisions.md#8), [Ontology](ontology.md#drafting-one-parked)).

## Checked before anything runs

Loading a config checks every key, by its dotted path, with a suggestion for a
misspelling. Building it constructs every stage, loader and sink, loading no
document and calling no model.

```python
from openodke.run import ConfigError, build, parse_config

try:
    parse_config(
        {
            "ontology": "o.json",
            "inputs": ["corpus"],
            "stages": {"extractor": "hybrid"},
            "bootstrp": True,
        }
    )
except ConfigError as exc:
    print(exc)
# bootstrp: unknown key — did you mean 'bootstrap'?

config = parse_config(
    {
        "ontology": "examples/e2e/ontology.json",
        "inputs": ["examples/e2e/corpus"],
        "stages": {"extractor": "hybrid", "grounder": "lm"},
    }
)
try:
    build(config)
except ConfigError as exc:
    print(exc)
# stages.grounder: unknown grounder 'lm' — did you mean 'llm'? (built-ins: delegated, llm, passthrough, span; or name your own as package.module:Name)
```

## The dry run

`--dry-run` runs every stage, then prints what it would write: the report, the
`bootstrap` DDL, each sink's statements or files, and the first twenty facts.
It opens no sink. It still calls the models, unless `models.replay` answers.

## What a run reports

The report prints `KnowledgeGraph.stats` (here, `examples/e2e/`):

```text
odke run
documents     8 (8 chunks; 0 skipped, 0 deferred, 0 empty)
extractor     paths (paths llm+pattern, chunks 8, pattern_facts 20, llm_facts 18, merged 2, model_calls 3), rejections (quote not in the passage 1), prompts extract@1
grounder      facts 36, calls 36, prompt_tokens 4764, completion_tokens 216, supported 20, contradicted 1, not_found 15, prompts ground.span@1, claim_format inline, span (facts 36, located 36)
corroborator  store (read 26), conflicts (lost 2, won 2)
gate          accepted 26, marked (contradicted 1)
refused       0
marked        1 written for review, not refused
coverage      0 of 5 sentences naming two known entities uncovered, 0 entities in no fact, 0 relations never offered, 0 unused
graph         26 facts (8 edges, 18 properties), 7 entities, 4 links (different 1, similar 3)
cost          39 model calls, 4980 tokens, USD unknown
calibration   ground.span@1 on anthropic/claude-haiku-4-5-20251001 (inline): κ 0.12 [-0.09, 0.38] (n 104), below its bar (κ ≥ 0.6); fail-recall 0.20, false refusals 0.09, planted caught 0.80 (label set G gate, n 104, 2026-10-25); refusals marked for review, not dropped (its card is below the bar, DECISIONS #54)
wrote         jsonl → …/examples/e2e/out: entities.jsonl 7, facts.jsonl 26, links.jsonl 4, manifest.json
writes        0:JsonlSink: entities 7 written; facts 26 written; links 4 written
manifest      …/examples/e2e/out/manifest.json
```

| Key in `stats` | Holds |
|---|---|
| `schema_version` | the stats' format version, `"1.0"` ([Stability](stability.md#formats)) |
| `documents`, `chunks`, `skipped`, `deferred`, `empty_extractions`, `derived`, `refused` | the [pipeline's counts](concepts.md#the-order-the-stages-run-in) |
| `marked` | facts the gate wrote marked for review instead of refusing, below the grounder's [calibration bar](calibration.md#below-the-bar-nothing-is-dropped-on-a-judges-word); absent when none |
| `calibration` | each model judge's card line, or UNCALIBRATED, with what its word did ([Calibration cards](calibration.md)) |
| `candidates` | the facts that reached resolution |
| `graph` | `facts`, `edges`, `properties`, `entities`, and `links` by kind |
| `stages.<name>` | each stage's `stats`, such as the [grounder's](grounding.md#the-second-model-llmgrounder), the gate's (as `gate`, and as `validator`, its 0.2 key, until 2.0.0), the [resolver's](resolution-and-corroboration.md#resolve) `store`, `judge` and `batch`, and the corroborator's `conflicts` |
| `cost` | with `meter: true`: calls, tokens, USD and latency, per role, and `cached_calls` |
| `cache` | with `models.cache`: `directory`, `hits`, `misses`, `failed` |
| `spent`, `budget` | `calls`, tokens and `usd` (`None` if any call went unpriced); the limits set |
| `failed` | the documents [left out](#when-one-document-fails), with reasons |
| `stopped` | a budget stop's `limit`, `spent`, `stage`, and the chunks `unextracted` and facts `unchecked` |
| `coverage`, `reextract` | with those options on |
| `batches` | with `batch_size`, the micro-batches run; every count is the run's sum |
| `writes` | per sink, what it wrote ([the write report](stores.md#the-write-report)) |

A `DoubleStageWarning` prints as a `warning:` line.

## The run manifest

`odke run`, `odke validate`, `odke ground` and `Validator.validate` each write a
manifest ([DECISIONS #40](decisions.md#40)):

| Field | Holds |
|---|---|
| `manifest_version` | `1` |
| `command`, `dry_run` | `run`, `validate` or `ground`; a dry run writes none |
| `config`, `config_hash` | the config resolved with every default (for `validate` and `ground`, their options; for the Validator, its stages), and its SHA-256; a chunker, router or inferrer left out and its `passthrough` named hash alike, as do the scorer left out and `evidence` named; any other stage's `passthrough` is a run of its own |
| `config_file`, `base_dir` | the config's file and directory |
| `models` | per role, the `model` asked for and the ids `served` |
| `prompts` | each [registered prompt](models.md#prompts) sent, with its SHA-256 |
| `ontology_version`, `ontology_hash` | the label and [`fingerprint`](ontology.md#freezing) |
| `package` | `openodke`, `python`, `platform` |
| `inputs` | each document's id and text SHA-256; the facts' `rows` and `hash`; a `hash` of both |
| `cache`, `budget`, `started_at`, `ended_at` | the cache directory, the limits, UTC times |
| `counts`, `spent`, `stopped`, `failed` | the run's counts, spend, budget stop and failed documents |
| `run`, `job` | the [log events'](#logs-and-traces) run id, and `job.end`'s counts |

It goes into each JSONL sink's `manifest.json`, to `manifest:` if set, else
beside the config as `<name>.manifest.json`; `odke validate` writes it into
`-o`, `odke ground` beside its summary. Secrets (`password`, `token`, `api_key`,
`secret`, `authorization`, a URL's password) are `<redacted>` before hashing.

`odke run --from-manifest PATH` runs one again as resolved, refusing (exit 2)
when the ontology or a document changed, naming each, and taking no `--model`,
`--widen`, `--cache`, `--batch-size`, `--tenant` or budget flag; another
openodke version only warns, and a field a newer version added to the manifest is
read past; a config option this version does not know is still refused, since it
cannot run it. To rerun an `odke validate` or `odke ground`
manifest, pass its `config.models` as a `--config` file (`{"models": ...}`) and
the rest as flags.

```python
from openodke.manifest import digest
from openodke.run import execute, load_config

manifest = execute(load_config("examples/e2e/odke.yaml"), dry_run=True).manifest
assert manifest is not None and manifest.dry_run
print(manifest.models["ground"])
# model='anthropic/claude-haiku-4-5-20251001' served=('anthropic/claude-haiku-4-5-20251001',)
assert list(manifest.prompts) == ["extract@1", "ground.span@1"]
assert manifest.config_hash == digest(manifest.config)
assert manifest.counts["facts"] == 26 and len(manifest.inputs.documents) == 8
```

`read_manifest(path)` reads one back; `from_manifest(path)` gives the
`RunConfig` for `execute(config, replaying=manifest)`.

## Logs and traces

`--log-format json` on `odke run`, `validate` and `ground` writes each event as
a JSON line on standard error ([DECISIONS #41](decisions.md#41)), with `ts`,
`level`, `event`, `run`, `command`, `stage`, `document`, `counts`, `latency_s`
and `cost`, `null` where they do not apply.

| `event` | When | Its own |
|---|---|---|
| `job.start` | first | `dry_run` |
| `stage` | per stage: `chunk`, `extract`, `ground`, `reextract`, `normalize`, `resolve`, `derive`, `corroborate`, `gate`, `write` | `counts`, `cost` |
| `document` | per document grounded | `counts`: `chunks`, `facts`, verdicts |
| `document.failed` | per document left out | `reason` |
| `model.call` | per call | `model`, `served`, `cost`, `error` (its type) |
| `job.end` | last | `counts`, `cost`, `documents`, `failed`, `stopped` |
| `log` | any other record | `logger`, `template` |

```text
{"ts":"2026-10-10T11:14:00.808Z","level":"info","event":"job.end","run":"259bdbe59782","command":"run","stage":null,"document":null,"counts":{"facts_in":36,"facts_out":25,"refused":1,"merged":10,"linked":4,"review":0},"latency_s":0.132401,"cost":{"calls":39,"cached_calls":0,"input_tokens":4764,"output_tokens":216,"usd":null},"documents":8,"failed":0,"stopped":null}
```

`job.end`'s `counts` (`JobCounts`: `RunResult.job`, `ValidationReport.job`,
`GroundSummary.job`) are the report's:

| | `odke run` | `odke validate` | `odke ground` |
|---|---|---|---|
| `facts_in` | `candidates` | `facts_in` | `rows` |
| `facts_out` | `graph.facts` | `facts_out` | `facts` |
| `refused` | `refused` | `refused` | the free checks' refusals |
| `merged` | in, plus `derived`, less refused and out | `merged` | 0 |
| `linked` | the links, of every kind | `linked` | 0 |
| `review` | the pairs the judge queued | `judge.queued` | 0 |

An event never holds a passage, quote, entity name, prompt, reply or config. A
stage's warning keeps its `template` but drops its arguments unless
`--log-text`. A stage with its own client, such as a judge handed to the
Validator, sends no `model.call` events. In Python, `configure_logs("json",
stream=...)` installs the handler on the `openodke` logger and
`configure_logs("text")` removes it.

```python
import io
import json

from openodke.observe import configure_logs
from openodke.run import execute, load_config

stream = io.StringIO()
configure_logs("json", stream=stream)
result = execute(load_config("examples/e2e/odke.yaml"), dry_run=True)
configure_logs("text")
events = [json.loads(line) for line in stream.getvalue().splitlines()]
end = next(event for event in events if event["event"] == "job.end")
print(end["counts"])
# {'facts_in': 36, 'facts_out': 25, 'refused': 1, 'merged': 10, 'linked': 4, 'review': 0}
assert end["counts"] == result.job.model_dump()
assert [e["stage"] for e in events if e["event"] == "stage"][:3] == ["chunk", "extract", "ground"]
```

**OpenTelemetry** (`pip install "openodke[otel]"`; names
[provisional](stability.md#provisional)): each job is a span (`odke run`), with
a child per stage (`stage ground`) and model call (`model ground`), carrying
counts and costs as `odke.*` attributes, sent to the tracer provider your
application configures. `opentelemetry-instrument odke run ...` configures one
from the environment; `openodke.observe.TRACER_PROVIDER` overrides it.

## When one document fails

A document whose chunking, extraction, grounding or normalising raises is left
out whole, with its reason in `stats["failed"]`, and the batch goes on. Your
own stage's batch that raises unattributed is retried a document at a time.
With [the cache](models.md#the-response-cache), a rerun pays only for the
failures. A configuration error or a budget stop still stops the run, and when
every document fails the command exits 1.

## Budgets

`models.budget` (`usd`, `calls`, `input_tokens`, `output_tokens`), or
`--budget-usd` and `--budget-calls` on `odke run`, `validate` and `ground`,
caps a run's spend. One `Ledger` counts every call and stops the run cleanly
at the first limit ([DECISIONS #32](decisions.md#32)).

- **Each call is estimated first**: input tokens as characters over four, plus
  four a message; output as `max_tokens`; USD at the run's price per token so
  far. A call that fits only once calls in flight settle waits; one that does
  not fit stops the run.
- **USD before the first priced call** is checked after each call. A provider
  that prices nothing never reaches a USD limit, so set `calls` or tokens too.
- **A failed call or a cache hit** costs nothing.
- **A stop keeps what the run has**: facts it reached first stay `unchecked`,
  the deterministic stages and sinks still run, the report opens with
  `stopped`, and the command exits 3.

```text
odke run
stopped       at budget, calls 3 of 3, during ground: 4 facts left unchecked
documents     2 (2 chunks; 0 skipped, 0 deferred, 0 empty)
…
grounder      facts 6, calls 2, prompt_tokens 254, completion_tokens 12, supported 2, unasked 4, …
…
cost          3 model calls, 886 tokens, USD unknown
budget        calls 3 of 3
wrote         jsonl → …/out: entities.jsonl 3, facts.jsonl 6, links.jsonl 0, manifest.json
manifest      …/out/manifest.json
```

`Ledger(budget).client(inner)` holds any client to a budget; `Pipeline.run` and
`Validator.validate` return the partial graph with `stats["stopped"]`:

```python
from openodke.llm import Budget, BudgetExceeded, Ledger, Message, ModelSpec, RecordedClient

ledger = Ledger(Budget(calls=2, usd=1.50))
client = ledger.client(RecordedClient([{"match": "Claim", "response": {"verdict": "supported"}}]))
ask = [Message(content="Claim: … Passage: …")]
for _ in range(2):
    client.complete(ask, spec=ModelSpec(model="ollama/llama3.1"))
try:
    client.complete(ask, spec=ModelSpec(model="ollama/llama3.1"))
except BudgetExceeded as exc:
    print(exc)
    assert exc.limit == "calls"
# stopped at budget: calls 2 of 2; no further model call is made
assert ledger.spent.calls == 2 and ledger.stopped is not None
```

## Streaming

`batch_size: N` (or `--batch-size N`) runs N documents at a time through every
stage and writes them ([DECISIONS #45](decisions.md#45)). `odke validate` and
`Validator.validate(..., batch_size=N)` read N triples rows at a time, closing
a micro-batch where the cited text changes. `odke run` holds a `triples`
extractor's rows; stream those with `odke validate`.

- **Inputs** are iterators: `read_triples`, the JSON Lines loader, a directory
  walked a file at a time.
- **Each micro-batch is written as made**: JSONL appends, Neo4j merges, your
  own sink is written each time, and RDF, `cypher_file` and `neo4j_admin_csv`
  are refused.
- **One run**: one ledger and cache across micro-batches, summed counts, a
  manifest that hashes the inputs as one batch, and a `--from-manifest` that
  streams. `execute` returns the stats and no facts.
- **The store joins micro-batches** through a sink that
  [says what it holds](stores.md#merge-with-the-store), in `odke validate`.
  Entities resolve across them only with [`store_lookup`](#store_lookup).

A micro-batch does not see the others: a rival value in another is not
contested (`check()` finds it), near-duplicates and coverage stay within it,
and an entity two mention is counted twice.

## From Python

`execute(config, *, dry_run=False)` returns a `RunResult`: the `graph`,
`dry_run`, the `written` lines, the `bootstrap` DDL, the `warnings`, and
`render()`, what the command prints. `load_config(path)` reads a file and
`parse_config(mapping, base_dir=...)` a parsed mapping.

```python
from openodke.run import execute, load_config

result = execute(load_config("examples/e2e/odke.yaml"), dry_run=True)

print(result.stats["graph"])
# {'facts': 25, 'edges': 8, 'properties': 17, 'entities': 7, 'links': {'different': 1, 'similar': 3}}
print(result.stats["stages"]["gate"])
# {'accepted': 25, 'refused': {'contradicted': 1}}
assert result.dry_run and result.written[0].startswith("jsonl → ")
assert result.render().startswith("odke run — dry run, nothing written")
```

[`examples/e2e/`](https://github.com/deepskandpal/odke/blob/main/examples/e2e/README.md)
runs every stage on an invented corpus with recorded responses, into JSONL
(`odke.yaml`) or Neo4j (`odke.neo4j.yaml`):

```bash
pip install "openodke[yaml]"
odke run examples/e2e/odke.yaml
```

