# `odke run`

`odke run` runs the whole pipeline from one file. The file names the inputs and
how to read them, the ontology, which model does which job, which implementation
fills each of the thirteen stages, and where the graph goes. It is data, for the
same reason a `ModelSpec` is: it can be diffed, logged next to the graph it
produced, and read by someone who has never seen the code. Every key is checked,
so a misspelt stage name is an error with a suggestion rather than a stage silently
left as the pass-through.

```bash
odke run examples/e2e/odke.yaml                          # run it and write the graph
odke run examples/e2e/odke.yaml --dry-run                # load, extract and ground; print what would be written
odke run examples/e2e/odke.yaml --model ollama/llama3.1  # every role on one model, whatever the config says
odke run examples/e2e/odke.yaml --budget-usd 1.50        # stop cleanly before spending more, keeping what is done
odke run --from-manifest examples/e2e/out                # run again what a run's manifest recorded
odke run examples/e2e/odke.yaml --log-format json        # every stage, document and model call as a JSON line
odke run examples/e2e/odke.yaml --batch-size 500         # stream: 500 documents at a time, in bounded memory
```

| Exit status | Means |
|---|---|
| 0 | the run finished (a dry run included) |
| 2 | the config cannot run: a bad key, a missing file, a missing extra |
| 1 | the run failed: a provider error outside any one document, every document failing, or a file that could not be read or written |
| 3 | the run stopped at its [budget](#budgets): what it kept was written, and the report says where it stopped |

A YAML config (`.yaml`, `.yml`) needs the `yaml` extra; the same keys as JSON need
nothing. [`examples/run.yaml`](https://github.com/deepskandpal/odke/blob/main/examples/run.yaml)
comments every key. Everything the command does is reachable from Python as
`openodke.run.execute(load_config(path))`.

## The config

```yaml
ontology: ontology.json            # paths are relative to this file
inputs:
  - path: corpus/register.csv
    loader: {use: csv, tier: curated}
  - corpus/notes                   # a bare path: stages.loader, else the directory loader
pythonpath: [.]                    # where a stage of your own is imported from
models:
  extract: anthropic/claude-sonnet-5
  ground: {model: anthropic/claude-haiku-4-5-20251001, max_tokens: 256}
  replay: {extract: recorded/extract.json, ground: recorded/ground.json}
  meter: true
  cache: .odke-cache               # answer a call asked before from here
  budget: {usd: 1.50, calls: 2000} # stop cleanly here, keeping what is done
  limits: {anthropic: 8}           # calls in flight per provider, every stage
stages:
  chunker: {use: sentence, max_words: 120}
  extractor: hybrid                # the one required stage
  grounder: llm
  normalizer: {use: value, person_types: [Person]}
  resolver: native
  corroborator: signature
  scorer: evidence
  gate: verdict
  sink: {use: jsonl, directory: out}
bootstrap: false
coverage: true
reextract: {windows: 3}            # off unless named
store_lookup: neo4j                # off unless named
tenant: acme                       # key the store by tenant; off unless named
manifest: runs/last.json           # the run manifest here too
batch_size: 500                    # stream; off unless named
```

| Key | Required | What it is |
|---|---|---|
| `ontology` | yes | A JSON or YAML ontology (`.yaml`/`.yml` is read as YAML). Loaded strictly, so a schema with validation errors stops the run before anything is spent. |
| `inputs` | yes, at least one | Files or directories. A string is a path read by the default loader; a mapping is `{path, loader}`. |
| `pythonpath` | no | Directories put on `sys.path` before a `package.module:Name` stage is imported. |
| `models` | no | Which model does which job, recorded responses, the cost meter, the response cache and the budget. |
| `stages` | yes | Which implementation fills each of the thirteen stages. Only `extractor` is required. |
| `bootstrap` | no, default `false` | Apply the ontology's constraints through the sink before the first write. |
| `coverage` | no, default `true` | Count what extraction left behind in each document, with no model ([the coverage report](grounding.md#what-extraction-left-behind-the-coverage-report)). |
| `reextract` | no, off by default | Hand those gaps back to the extractor and ground what returns: `true`, or `{windows: N}`, the most windows per document (default 3). The extractor must have a `reextract` method (`llm` and `hybrid` do). Not part of `odke eval ablation` ([the re-extract hook](grounding.md#handing-a-gap-back-the-re-extract-hook)). |
| `store_lookup` | no, off by default | Resolve each batch against what the store already holds, without loading it: `neo4j`, or `package.module:Name` for a `StoreLookup` of your own ([below](#store_lookup)). |
| `manifest` | no | Where the [run manifest](#the-run-manifest) is written, besides each JSONL sink's `manifest.json`. |
| `batch_size` | no, off by default | Stream the run: load, run and write that many documents at a time ([Streaming](#streaming)). `--batch-size` overrides it. Not a Neo4j sink's own `batch_size`, which is rows per write transaction. |
| `tenant` | no, off by default | Key everything the run writes to, and reads from, its store by this tenant, so two tenants' identical facts never merge ([Tenants](stores.md#tenants)). `--tenant` overrides it. |

Every relative path (the ontology, each input, `pythonpath`, replay files, a sink's
output) resolves against the directory the config file is in, so a config runs the
same from any working directory.

### `inputs`

An input's `loader` is a stage spec like any other: a short name, with options as
extra keys. An input without one is read by `stages.loader`, and when that is left
out too, by the `directory` loader, which reads every suffix it knows and warns
about and skips a file whose extra is missing
([Loading documents](loading.md#directories)).

**Document ids are source paths.** A document read from a file gets the file's path
relative to the config as its id, plus `#L<line>` for a record with a source line
(CSV, TSV, JSONL) or `#<row>` for one without (JSON, Parquet): `corpus/register.csv#L2`.
A second document from the same path is `~2`, `~3`. A labelled set and a provenance
query can both name a document by a string you already know, instead of a UUID
minted by the run.

### `models`

`extract`, `ground` and `infer` are each a model string or a full `ModelSpec`
(`model`, `temperature`, `max_tokens`, `timeout`, `base_url`, `api_key_env`, `extra`); left out,
a role takes its `ModelRoles` default. Once `extract` is named, that default is the
extraction model, for grounding too (at `max_tokens: 256`), so passages go nowhere
the file did not name. Keys never go in the file: a provider reads
its own environment variable, or the one `api_key_env` names. Which providers can
be named, and which variable each reads, is
[Models and providers](models.md) — or `odke models`, which also says whether the
variable is set.

**`--model` overrides the block for one run.** `odke run odke.yaml --model
openai/gpt-5.5` puts every role on that model and prints which; `--model-provider
openai` supplies the prefix when the string has none. Only the model string
changes, so grounding keeps its smaller `max_tokens`. A per-role choice stays a
config decision.

**`temperature` is unset by default**, and then left out of the request entirely,
so the provider's own applies. Which temperatures a model accepts is a per-model
fact — some accept only one — and openodke does not keep a table of it. Set the key
and it is sent exactly as written, `0` included.

- **`replay`** maps a role to a file of recorded responses, either a cassette
  object (`ReplayClient`) or a list of match entries (`RecordedClient`), so a run
  needs no key and no network. Delete the lines to call the models.
- **`meter: true`** wraps every model client in a `CostMeter` and puts calls,
  tokens, USD and latency per stage into the graph's stats. A cost no provider
  reported stays unknown rather than being counted as zero.
- **`cache`** names a directory of model answers, so a call asked before is
  answered from it for nothing ([the response cache](models.md#the-response-cache)).
  `--cache DIR` overrides it for one run.
- **`budget`** is the most the run may spend: `usd`, `calls`, `input_tokens`,
  `output_tokens`, each optional ([Budgets](#budgets)). `--budget-usd` and
  `--budget-calls` override it for one run.
- **`limits`** caps the calls in flight to each provider it names, across every
  stage: `{anthropic: 8}` ([Concurrency per provider](models.md#concurrency-per-provider)).

### `stages`

A stage is a built-in's short name or `package.module:Name` for your own, and
either may take options. `grounder: llm` and `grounder: {use: llm, max_workers: 4}`
are both accepted: every key except `use` is passed to the implementation as a
keyword argument, and an option it does not take is an error listing the ones it
does.

**A stage left out is the pass-through** from `openodke.stages`
([DECISIONS #20](decisions.md#20)), and the stats say nothing about it.

**The scorer is the exception.** Left out, `odke run` fills it with `evidence`
(`EvidenceScorer`), because the pass-through leaves `Fact.confidence` at whatever
the extractor put there — one constant for every fact, whatever the grounder
found — and a threshold set against that silently accepts everything. The library
keeps its pass-through: `Pipeline` is for a caller who scores themselves, and this
is the CLI's opinion, on a graph somebody is about to filter. `scorer: passthrough`
opts out and hands the extractor's number to the sink unchanged.

**A stage of your own** is `package.module:Name`. A class is constructed with the
options; any other object is used as it is and takes no options. Either way it must
satisfy the stage's Protocol, which is checked when the config is built rather than
discovered halfway through a run.

**`delegated`**, with `to:`, marks a stage the store does itself
([DECISIONS #21](decisions.md#21)): `resolver: {use: delegated, to: neo4j-graphrag:FuzzyMatchResolver}`.
Every stage but the chunker and the inferrer accepts it.

**What `odke run` sets itself** is refused from the file with
`set by odke run, not by the config`: the ontology (for the normaliser, the
corroborator and the sinks), the model client and spec (for the model-backed
stages), and the loaded documents (for the extractors). So are the options that
would have to be a Python object, such as a corroborator's `source` callable.

| Stage | Short names | Built from | Options |
|---|---|---|---|
| `loader` | `directory`, `text`, `markdown`, `html`, `pdf`, `docx`, `csv`, `tsv`, `json`, `jsonl`, `parquet` | the `openodke.loaders` classes | `tier`, and each class's own: `encoding`, `modality`, `records`, `delimiter`, `columns`, `pattern`; `html`: `strip_boilerplate`; `pdf`: `per_page`, `page_separator` |
| `chunker` | `sentence`, `passthrough` | `SentenceChunker` | `max_words`, `overlap` |
| `router` | `passthrough`, `delegated` | — | your own is `package.module:Name` |
| `extractor` | `pattern`, `llm`, `hybrid`, `triples` | `PatternExtractor`, `LLMExtractor`, `HybridExtractor`, `TriplesExtractor` | `pattern`: `mappings`, `subject_type`, `confidence`; `llm`: `types`, `snippet_limit`, `confidence`, `repairs`, `retry` and `max_workers` (as the grounder's); `hybrid`: `llm` (options, or `false`), `pattern` (options); `triples`: `path` (required), `extractor`, `confidence` — another extractor's output, [in the triples format](inputs.md) |
| `grounder` | `span`, `llm`, `passthrough`, `delegated` | `SpanGrounder`, `LLMGrounder` | `llm`: `max_workers`, `retry` (`attempts`, `base_delay`, `multiplier`, `max_delay`, `jitter`), `context`, `verdicts`, `locate` ([the span locator](grounding.md#locating-spans)), `widen` ([widen and retry](grounding.md#widen-and-retry); `odke run --widen` sets it) |
| `normalizer` | `value`, `passthrough`, `delegated` | `ValueNormalizer` | `day_first`, `person_types` |
| `resolver` | `native`, `passthrough`, `delegated` | `NativeResolver` | `threshold`, `nudge_up`, `nudge_down`, `max_block`, `judge` ([the pair judge](#the-pair-judge)), `normalize_batch` (default `false`; `true` merges [the batch's look-alikes](resolution-and-corroboration.md#normalising-mentions-in-a-batch)), `context_floor`; `odke run` hands it the ontology, for its types' aliases |
| `corroborator` | `signature`, `passthrough`, `delegated` | `SignatureCorroborator` | `half_life_days`, `freshness_floor`, `intervals`, `near_duplicates` |
| `scorer` | `evidence` (the default), `passthrough`, `delegated` | `EvidenceScorer` | `prior`, `verdict_weights` |
| `gate` | `verdict`, `passthrough`, `delegated` | `VerdictGate` | `refuse_not_found` |
| `sink` | `jsonl`, `neo4j`, `cypher_file`, `neo4j_admin_csv`, `rdf`, `networkx` | the [sinks](stores.md) | see [below](#sinks) |
| `constrainer` | `neo4j`, `passthrough`, `delegated` | `Neo4jConstrainer` | — |
| `inferrer` | `passthrough` only | — | `odke run` never infers ([below](#the-inferrer)) |

### Sinks

`sink` takes one sink or a list of them; none writes nothing.

| `use` | Writes | Options |
|---|---|---|
| `jsonl` | [`JsonlSink`](stores.md#jsonl) | `directory` (required); `merge` (default `false`): keep what the files hold and [merge with it](stores.md#merge-with-the-store) |
| `neo4j` | [`Neo4jSink`](stores.md#neo4j) | `uri` or `uri_env` (one required); `user` or `user_env` (default `neo4j`); `password_env` (default `NEO4J_PASSWORD`); `database`; `batch_size`, the most rows a write transaction holds (default 500); the run's `tenant` |
| `cypher_file` | [`CypherFileSink`](stores.md#cypher-file) | `path` (required); `batch_size` (default 500) |
| `neo4j_admin_csv` | [`Neo4jAdminCsvSink`](stores.md#neo4j-admin-csv) | `directory` (required); `delimiter` (default `,`); `array_delimiter` (default `;`) |
| `rdf` | [`RdfSink`](stores.md#rdf), needs `rdf` | `path` (required); `format` (`turtle`, `nt`, `json-ld`; default from the suffix); `base`; `schema` |
| `networkx` | [`NetworkXSink`](stores.md#networkx), needs `networkx` | `path`: where to write the filled graph as node-link JSON. Without it the graph is filled in memory and written nowhere, so give it a `path`. |

`odke run` supplies the ontology to every sink but `jsonl`, so projections of
multi-valued predicates are lists, the Cypher script opens with the constraint DDL,
and the RDF file declares its schema. Paths resolve against the config. A sink is
constructed while the config is built, so a bad option fails before anything
runs, and a missing extra is a config error naming it:
`stages.sink: RdfSink needs rdflib, which is not installed. Run: pip install "openodke[rdf]"`.
The same holds for the `pdf`, `docx` and `parquet` loaders. A node-link file reads
back with `networkx.node_link_graph(data, edges="edges")` (`link="edges"` before
networkx 3.4), with dates and times as ISO strings.

**A password never goes in a config file.** `password:` is refused with an error
that says to name an environment variable with `password_env` instead. The
variables are read when the sink is opened, which a dry run never does.

### `bootstrap` and the constrainer

`bootstrap: true` applies the constrainer's DDL through the sink before the first
write, and before any model is called, so a database that refuses the DDL fails
the run while it is still free. It needs a sink that applies constraints (`neo4j`);
with none, the config is refused. The DDL comes from `stages.constrainer`, or from
`Neo4jConstrainer` when that is left out. Every statement is `IF NOT EXISTS`, so
bootstrapping on every run is safe.

### `inverses`

`inverses: false` turns off the inverse and symmetric partners
([Concepts](concepts.md#inverse-and-symmetric-partners)). Left out, they are
on exactly when the ontology declares an `inverse_of` or a `symmetric`
predicate, and the run report prints how many were added.

### `store_lookup`

`store_lookup` hands the resolver a [store lookup](resolution-and-corroboration.md#resolving-against-the-store)
([DECISIONS #31](decisions.md#31)). With no `stages.resolver`, it brings `native`;
a resolver that cannot take a lookup is refused. `neo4j` reads the store the
run's one neo4j sink writes to, on that sink's connection, after `bootstrap`
has created the indexes it reads through. It reads the run's `tenant`, and a
`tenant` of its own that differs is refused: a run looks up the tenant it
writes. Its options are `limit` (hits per name token, default 100) and the
sink's connection keys (`uri`, `uri_env`, `user`, `user_env`, `password_env`,
`database`) to read a store the run does not write to. A lookup of your own is
scoped to the run's tenant when it has `scoped(tenant)`.
`package.module:Name` is constructed with its options, like a stage. A dry run
opens no store, so it resolves each batch within itself and prints a warning
saying so. `odke validate --config` reads the same key. The resolver's counts
are `stages.resolver.store`: entities `looked_up`, store `candidates`, keys
`rekeyed`, and links by kind.

### The pair judge

`resolver: {use: native, judge: true}` puts the pairs the resolver's rules leave
open, scored from 0.7 up to its `threshold`, to a model in both orders
([The pair judge](resolution-and-corroboration.md#the-pair-judge),
[DECISIONS #34](decisions.md#34)). It is off unless named. A mapping sets its
options: `low` (the band's start, default 0.7), `queue` (the JSONL file unsure
pairs are appended to, for `odke label make pair`), `reviewed` (a person's
labels, as `odke label read` writes them, which must exist), `max_workers` and
`retry`, as the grounder's. Paths are relative to the config. It asks the
`ground` role's model, through `models.replay.ground` when that is set, and
`meter: true` counts its calls as their own row, `judge`. The texts it reads
contexts from are the run's inputs. A dry run asks it, as it grounds, but
writes no queue, and says so.

```yaml
stages:
  resolver:
    use: native
    judge: {queue: review/pairs.jsonl, reviewed: review/reviewed.jsonl}
```

### The inferrer

`inferrer` accepts only `passthrough`. Anything else is refused:
`odke run never infers an ontology`. Inference is a bootstrap, not a mode
([DECISIONS #8](decisions.md#8)): run `odke ontology infer` once, review and freeze
the result, and name the file under `ontology`. See
[Ontology](ontology.md#drafting-one-parked).

## Checked before anything runs

Loading a config checks every key's type and name, reporting each problem on its
own line by its dotted path, with a suggestion for a misspelling. Building it then
loads the ontology, constructs every stage and each input's loader, opens the
replay files and plans the sinks, and still loads no document and calls no model.

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

`--dry-run` loads, chunks, routes, extracts and grounds, then prints what it would
have written instead of writing it: the per-stage report, the DDL `bootstrap` would
apply, the statements or files each sink would produce, and the first twenty facts.
**It opens no sink**, so it needs neither a database nor its password. It still
calls the models, because what would be written depends on their answers; with
`models.replay` it calls nothing.

## What a run reports

Every stage's own counts end up in `KnowledgeGraph.stats`, where the JSONL
manifest and `odke eval` can read them, and the command prints them one line per
stage:

| Key in `stats` | Holds |
|---|---|
| `documents`, `chunks`, `skipped`, `deferred`, `refused` | the pipeline's own counts |
| `candidates` | the facts that reached resolution: extracted, grounded and normalised, from the documents that did not fail. The job's facts in ([Logs and traces](#logs-and-traces)) |
| `graph` | `facts`, `edges`, `properties`, `entities`, and `links` by kind |
| `stages.extractor` | `paths` (the hybrid's `PathReport` totals), `rejections` by reason, or `model_calls`; `prompts`, the keys of the [registered prompts](models.md#prompts) the model sent; for `triples`, rows by how their evidence was made (`cited`, `quoted`, `quote_not_found`, `context`), `unmatched_rows` and `ambiguous_rows` |
| `stages.grounder` | calls, retries, failures, a count per verdict, tokens, `cost_usd`, `prompts`, `unasked` (facts a budget stop reached before their call), the span check's own counts, with `locate` the locator's, and with `widen`, under `widen`: `retried`, `recovered` and the retries' own calls, tokens and cost, which `cost` meters as their own row, `ground.widen` |
| `stages.corroborator` | `conflicts`: how many facts `won`, `lost` or `tied` a contest |
| `stages.resolver` | with `store_lookup`: under `store`, entities `looked_up`, store `candidates`, incoming keys `rekeyed` onto a stored one, and links to the store by kind; under `lookup`, the lookup's own counts. With `judge`: under `judge`, the pairs handed in and `asked`, `calls`, `swapped` calls, pairs whose orders `disagreed`, each decision, `person`, `queued`, `no_context`, `failed`, tokens and cost, `cached` calls, and after a budget stop `unasked` calls and `stopped`, which is the run's `stopped` when the judge reached it first; and `prompts`. With the batch normalised: under `batch`, the look-alike pairs (`alike`), those `merged`, the `groups` and `mentions` they made, the pairs kept apart by `numbers`, `forms`, `context`, as `ambiguous` or `refused`, and the sentences `embedded` |
| `stages.validator` | the gate's `accepted`, and `refused` by reason. The key keeps its 0.2 name ([DECISIONS #26](decisions.md#26)) |
| `stages.<name>` | anything else a stage reports by carrying a `stats` mapping, your own stages included |
| `cost` | with `meter: true`: calls, tokens, USD and latency, in total and per role, and `cached_calls`, the calls the response cache answered |
| `cache` | with `models.cache`: the `directory`, and its `hits`, `misses` and `failed` calls |
| `spent` | always: `calls` (the cache's `cached_calls` among them), `input_tokens`, `output_tokens`, and `usd`, `None` when any call went unpriced. The `cost` line |
| `budget` | with `models.budget`: the limits set. The `budget` line, each against what was spent |
| `failed` | the documents left out because something failed for them alone, each with its reason: `extract: ProviderError: …` ([below](#when-one-document-fails)). The `failed` line |
| `stopped` | when the budget stopped the run: the `limit`, the `budget`, what was `spent`, the `stage` it stopped in, the chunks left `unextracted` and the facts left `unchecked`. The `stopped` line, first in the report |
| `coverage` | with `coverage: true`, the default: totals, the relations never offered and never used, and each document's uncovered sentences and missed entities ([the coverage report](grounding.md#what-extraction-left-behind-the-coverage-report)) |
| `reextract` | with `reextract`: `windows` asked, facts `returned`, `duplicates`, `kept`, `refused` by grounding, and their `verdicts` |
| `batches` | with `batch_size`: the micro-batches run. Every count above is then the run's, summed over them ([Streaming](#streaming)) |
| `writes` | once every sink has written: per sink (`0:Neo4jSink`), per kind, the rows `written` new, `merged` into what the store held, and `skipped`, and Neo4j's `transactions` ([the write report](stores.md#the-write-report)). One `writes` line per sink, after the `wrote` lines |

A `DoubleStageWarning` raised while the pipeline is built is printed as a
`warning:` line on standard error.

## The run manifest

Every run writes one manifest: `odke run`, `odke validate`, `odke ground` and
`Validator.validate` ([DECISIONS #40](decisions.md#40)). It says what the run
was asked, what answered, and what it made, so a graph can be traced to the
run that wrote it and the run made again.

| Field | Holds |
|---|---|
| `manifest_version` | `1`, the format of the fields below |
| `command`, `dry_run` | `run`, `validate` or `ground`; a dry run has a manifest and writes none |
| `config`, `config_hash` | the config resolved, and the SHA-256 of it as canonical JSON. `odke run`'s is the run config with every key filled in and each model role as `ModelRoles` resolves it, so leaving a default out and naming it hash alike. `odke validate` and `odke ground` record their options and the `models` block; the Validator, its stages by class. No secret is in either |
| `config_file`, `base_dir` | the config file, and the directory its relative paths resolve against |
| `models` | per role: `model`, the provider-qualified id asked for, and `served`, the ids the provider said answered, provider-qualified. When the config names an alias, `served` is the pinned id |
| `prompts` | each [registered prompt](models.md#prompts) key sent, with its SHA-256 |
| `ontology_version`, `ontology_hash` | the ontology's label and its [`fingerprint`](ontology.md#freezing); `None` with no ontology |
| `package` | `openodke`'s version, `python`'s and the `platform` |
| `inputs` | `documents`, each id with the SHA-256 of its text; `facts`, the triples handed in (`rows`, `hash`); `hash`, one over both |
| `cache`, `budget` | the response cache's directory and the budget's limits, or `None` |
| `started_at`, `ended_at` | UTC; the end is after the sinks wrote |
| `counts` | the command's own: the pipeline's and the graph's for `odke run`, the report's for `odke validate`, the summary's for `odke ground` |
| `spent` | `calls`, `cached_calls`, `input_tokens`, `output_tokens`, and `usd`, `None` when any call went unpriced |
| `stopped`, `failed` | where a [budget](#budgets) stopped the run, and the documents left out with why |
| `run`, `job` | the id every [log event](#logs-and-traces) of the run carries, and the counts its `job.end` logs: a manifest and its log stream join on `run`, and say the same |

**Where it goes.** Into each JSONL sink's `manifest.json`, beside the keys the
sink has always written there: `ontology`, `created_at`, the counts and
`stats`. An older reader finds what it found, and in a store that merges, those
counts stay the files'. With `manifest:` in the config, there too; with neither,
beside the config as `<name>.manifest.json`, so every run writes one.
`odke validate` writes it into `-o`, and `odke ground` beside its summary. The
report's `manifest` line names each file.

**Secrets.** A value under a key that names one (`password`, `token`,
`api_key`, `secret`, `authorization`) and the password in a URL are
`<redacted>` before anything is hashed or written. A key that names where a
secret is read, `password_env` or `api_key_env`, is kept.

**Two runs of one config write the same manifest**, but for `started_at`,
`ended_at`, the graph's `created_at`, the meter's latencies and the `run` id.
`odke run --from-manifest PATH` runs one again: the config as it was resolved,
from where it was read. It is refused, exit 2 and nothing written, when the
ontology's fingerprint or any document's hash is not what the manifest
recorded, naming each document that changed, and it takes none of `--model`,
`--widen`, `--cache` or the budget flags, which would make it another run.
A different openodke is a warning. With `models.cache`, the replay answers from
the cache and calls nothing it called before.

A manifest of `odke validate` or `odke ground` is run again the same way by
hand: its `config` holds every option, so write `config.models` to a file as
`{"models": ...}`, pass it as `--config`, and pass the rest as flags.

```python
from openodke.manifest import digest
from openodke.run import execute, load_config

manifest = execute(load_config("examples/e2e/odke.yaml"), dry_run=True).manifest
assert manifest is not None and manifest.dry_run
print(manifest.models["ground"])
# model='anthropic/claude-haiku-4-5-20251001' served=('anthropic/claude-haiku-4-5-20251001',)
assert list(manifest.prompts) == ["extract@1", "ground.span@1"]
assert manifest.config_hash == digest(manifest.config)
assert manifest.counts["facts"] == 25 and len(manifest.inputs.documents) == 8
```

`read_manifest(path)` reads one back, from the file or the directory holding
it, and `from_manifest(path)` gives the `RunConfig` to run again with
`execute(config, replaying=manifest)`.

## Logs and traces

`--log-format json` on `odke run`, `odke validate` and `odke ground` writes
every event as one JSON object a line on standard error, while the report
still prints to standard output ([DECISIONS #41](decisions.md#41)). Every
event has the same keys, `null` where they do not apply: `ts` (UTC), `level`,
`event`, `run` (one id for all of a job's events), `command`, `stage`,
`document`, `counts`, `latency_s` and `cost`.

| `event` | When | Its own |
|---|---|---|
| `job.start` | once, first | `dry_run` |
| `stage` | once per stage: `chunk`, `extract`, `ground`, `reextract`, `normalize`, `resolve`, `derive`, `corroborate`, `gate`, `write` | `counts`: the stage's, such as `facts`, a count per verdict, `failed`; `cost`: its model calls' |
| `document` | once per document grounded | `counts`: its `chunks`, `facts` and a count per verdict |
| `document.failed` | once per document left out (a warning) | `reason`, as the report gives it |
| `model.call` | once per model call; a failed one is a warning | `model`, asked for, and `served`, both provider-qualified; `cost`: `input_tokens`, `output_tokens`, `usd`, `cached`; `error`, by type alone |
| `job.end` | once, last | `counts`, the job's; `cost`: `calls`, `cached_calls`, `input_tokens`, `output_tokens`, `usd`; `documents`, `failed`, `stopped` (the limit, or `null`) |
| `log` | any other record a stage logs | `logger`, and `template`, the message before its arguments |

```text
{"ts":"2026-10-10T11:14:00.808Z","level":"info","event":"job.end","run":"259bdbe59782","command":"run","stage":null,"document":null,"counts":{"facts_in":36,"facts_out":25,"refused":1,"merged":10,"linked":4,"review":0},"latency_s":0.132401,"cost":{"calls":39,"cached_calls":0,"input_tokens":4764,"output_tokens":216,"usd":null},"documents":8,"failed":0,"stopped":null}
```

**A job's counts are its report's.** `job.end` counts facts `in` and `out`,
`refused`, `merged`, `linked` and sent to `review`, read from the same numbers
the report prints (`JobCounts`; `RunResult.job`, `ValidationReport.job`,
`GroundSummary.job`):

| | `odke run` | `odke validate` | `odke ground` |
|---|---|---|---|
| `facts_in` | `candidates` | `facts_in` | `rows` |
| `facts_out` | `graph.facts` | `facts_out` | `facts` |
| `refused` | `refused` | `refused` | the free checks' refusals: nothing is dropped |
| `merged` | in, plus `derived`, less refused and out | `merged` | 0 |
| `linked` | the links, of every kind | `linked` | 0 |
| `review` | the pairs the [pair judge](resolution-and-corroboration.md) queued | `judge.queued` | 0 |

**No text, no secret.** An event holds ids, names, counts, times and costs:
never a passage, a quote, an entity's name, a prompt or a reply, and no
config. A stage's own warning can quote a model's reply, so as JSON it keeps
its `template` and drops its arguments; `--log-text` keeps a `message` too. A
model call is named by its stage, not its document: a batch's calls run
concurrently. A stage with a client of its own, a pair judge handed to the
Validator say, sends no `model.call` events, though its spend is in `job.end`.

**From Python**, `configure_logs("json", stream=...)` installs the handler on
the `openodke` logger, and `configure_logs("text")` takes it off. Without a
handler, an event goes nowhere.

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

**OpenTelemetry**, with `pip install "openodke[otel]"`: each job is a span,
`odke run`, each stage a child of it, `stage ground`, and each model call a
child of its stage, `model ground`, with the events' counts and costs as
`odke.*` attributes. The extra is the API alone, imported on first use: the
spans go to the tracer provider your application configures, with the SDK and
the exporter of its choosing, and with none they go nowhere and cost next to
nothing. From the command line, `opentelemetry-instrument odke run ...` with
`opentelemetry-distro` and an exporter configures one from the environment.
`openodke.observe.TRACER_PROVIDER` overrides the global provider.

## When one document fails

A document whose chunking, extraction, grounding or normalising raises is left
out of the graph whole, named with its reason in `stats["failed"]` and the
report's `failed` line, and the rest of the batch goes on: a provider error
that outlasted its retries, or a reply nothing could parse, costs one document,
not the run. The built-in stages say which document failed, so nothing is
asked twice; a stage of your own whose batch raises without saying is asked
again one document at a time. With [the cache](models.md#the-response-cache),
a rerun pays only for the documents that failed. A configuration error is nobody's document and still stops the
run: a missing key, a missing provider adapter, a missing extra. So does a
budget. When every document fails, the cause is almost certainly not in the
documents, and the command exits 1.

## Budgets

`models: {budget: {usd: 1.50, calls: 2000, input_tokens: …, output_tokens: …}}`,
or `--budget-usd` and `--budget-calls`, caps what one run may spend; a limit left
out is no limit. `odke validate` and `odke ground` take the same block from
`--config` and the same flags. One `Ledger` counts every call the run makes, from
every stage and every thread, and stops the run cleanly at the first limit
([DECISIONS #32](decisions.md#32)).

- **Checked before each call**, against an estimate: one call; input tokens as
  the characters of the messages and the schema over four, rounded up, plus four
  a message; output tokens as the spec's `max_tokens`; USD as the run's own USD
  per token so far, over the calls a provider priced, times both. A call that
  fits beside the calls in flight goes out. One that fits only once they settle
  waits for them. One that does not fit beside what is already spent stops the
  run, and neither it nor any call after it is made.
- **USD with nothing to estimate from**, before the first priced call, is
  checked after each call instead: the call after the one that reached the
  limit is refused. A provider that prices nothing, a local server say, never
  reaches a USD limit, so give such a run `calls` or tokens too.
- **A call that raised is not counted**, and a cache hit is free: neither
  touches the budget. A budget stop is never retried.
- **A stop keeps what the run has.** Every chunk extracted and every fact
  grounded before it stays; a fact the stop reached first stays `unchecked`
  and is counted, as is a chunk never extracted. The free checks and the cache
  still answer after a stop, the deterministic stages run, the sinks write, the
  report opens with a `stopped` line, and the command exits 3.

Every run report has a `cost` line, metered or not: the calls (and how many the
cache answered), the tokens, and the USD, or `USD unknown` when any call went
unpriced. With a budget, a `budget` line sets each limit against what was spent.

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

From Python, `Ledger(budget).client(inner)` holds any client to a budget, and
`Pipeline.run` and `Validator.validate` return the partial graph with
`stats["stopped"]` rather than raising:

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

A batch is held whole: every chunk is extracted, then every document grounded,
then resolution, corroboration and the gate run over all of it. `batch_size: N`
(or `--batch-size N`) cuts the run into micro-batches instead
([DECISIONS #45](decisions.md#45)): N documents are loaded, run through every
stage and written, then the next, so memory follows the micro-batch, not the
run. `odke validate` and `Validator.validate(..., batch_size=N)` read N triples
rows at a time.

- **Inputs are read as they are needed.** `read_triples` and the JSON Lines
  loader are iterators, a directory is walked a file at a time, and every input
  is checked before the first is read. Document ids are the ones one batch
  gives.
- **Each micro-batch is written as it is made.** A JSONL sink is written by the
  first and appended to by the rest; its manifest's counts are its lines, and
  its `stats` the run's so far. Neo4j merges each one. RDF, `cypher_file` and
  `neo4j_admin_csv` rewrite a file of the whole graph on every write, so a
  streamed run refuses them before anything runs. A sink of your own is
  written once a micro-batch.
- **One run, whatever its slices.** One ledger counts every call against the
  [budget](#budgets), the [response cache](models.md#the-response-cache)
  answers across micro-batches, one document's failure is its own, and the
  report sums every micro-batch (`stats["batches"]` says how many). What
  `execute` returns holds those stats and no facts, and a dry run prints the
  first facts it would have written.
- **The store joins micro-batches, not memory.** A fact two micro-batches both
  state merges when the second is written, through a sink that says what it
  holds ([merge with the store](stores.md#merge-with-the-store)): `odke
  validate` does this, `odke run` does not yet. Without one, a JSONL file holds a
  line for each, and a reader keeps the last. Entities resolve within a
  micro-batch, and across them with [`store_lookup`](#store_lookup).
- **The manifest is the run's.** `batch_size` is in the config the
  [run manifest](#the-run-manifest) records and hashes. Its `counts`, `job` and
  `spent` are the micro-batches' summed, with `batches`, and the `job.end` event
  logs the same counts. Its inputs hash covers every micro-batch's documents
  and rows, in the digest one batch takes of them, so the same inputs hash
  alike streamed or not. `--from-manifest` replays a streamed run streamed:
  it reads the inputs through once and refuses a change before the first
  micro-batch is written. It takes no `--batch-size`, which would be another run.
- **Rows stay with their text.** `odke validate` closes a micro-batch only where
  the next row cites another text, so one text's rows, sorted together, are
  grounded and measured once; a text with more than twice N rows is split
  there. The texts are held, because a row may cite any of them. `odke run`
  with a `triples` extractor holds its rows; stream them with `odke validate`.

What a micro-batch cannot see: a rival value stated in another micro-batch is
not contested (`check()` finds the pair in a Neo4j store), near-duplicate texts
are compared within one, the coverage report knows its micro-batch's names and
keeps the records of the first 100 documents with a gap, and a count summed
over micro-batches counts an entity once in each. Measured on 25,000 and
50,000 synthetic triples rows, every default stage and a scripted client, the
peak RSS is the same, about 66 MB; 5,000 rows in one batch peak at 141 MB.

## The default gate: `VerdictGate`

The grounder stamps a verdict and drops nothing ([DECISIONS #20](decisions.md#20)), so
something has to refuse. `gate: verdict` is that something:

- **`contradicted` is always refused.** The cited passage was read, and it says
  something else.
- **`not_found` is accepted by default.** A passage that does not settle a claim is
  not evidence against it, and a structured fact is grounded against its own cell,
  which never names the subject, so a careful grounder answers `not_found` for most
  of a CSV. `refuse_not_found: true` refuses those too, trading recall for
  precision.
- **`unchecked` is accepted.** Nothing was asked.

It checks the verdict only, not domain or range. Its `stats` count what it accepted
and why it refused, because the pipeline's own `refused` count cannot say why.
Whether to refuse `not_found` on your corpus is a measurement:
[`odke eval ablation`](evaluation.md#ablation) reports what refusing it would have
kept and lost.

```python
from openodke import Entity, Fact, GroundingVerdict, Ontology, VerdictGate

gate = VerdictGate()
ada = Entity(key="p:ada", type="Person")
verdicts = [GroundingVerdict.SUPPORTED, GroundingVerdict.NOT_FOUND, GroundingVerdict.CONTRADICTED]
actions = [
    gate.validate(
        Fact(subject=ada, predicate="born", object_value=1815, verdict=v), Ontology()
    ).action
    for v in verdicts
]
assert actions == ["accept", "accept", "refuse"]
print(gate.stats)
# {'accepted': 2, 'refused': {'contradicted': 1}}
assert VerdictGate(refuse_not_found=True).refused == {
    GroundingVerdict.CONTRADICTED,
    GroundingVerdict.NOT_FOUND,
}
```

## From Python

`execute(config, *, dry_run=False)` returns a `RunResult`: the `graph`, whether it
was a `dry_run`, the lines describing what was `written` (or would have been), the
`bootstrap` DDL applied, the `warnings`, and `render()`, which is exactly what the
command prints. `load_config(path)` reads a file; `parse_config(mapping, base_dir=...)`
takes an already-parsed mapping.

```python
from openodke.run import execute, load_config

result = execute(load_config("examples/e2e/odke.yaml"), dry_run=True)

print(result.stats["graph"])
# {'facts': 25, 'edges': 8, 'properties': 17, 'entities': 7, 'links': {'different': 1, 'similar': 3}}
print(result.stats["stages"]["validator"])
# {'accepted': 25, 'refused': {'contradicted': 1}}
assert result.dry_run and result.written[0].startswith("jsonl → ")
assert result.render().startswith("odke run — dry run, nothing written")
```

## Walking through `examples/e2e`

[`examples/e2e/`](https://github.com/deepskandpal/odke/blob/main/examples/e2e/README.md)
is a small invented corpus run through all thirteen stages, on recorded model
responses, so it needs no key, no network and no database.

| Path | What it is |
|---|---|
| `corpus/register.csv` | A company-register extract: name, number, incorporation date, head office. Tier `curated` |
| `corpus/staff.csv` | A staff directory. Tier `authoritative` |
| `corpus/factsheet.md` | `Key: value` lines, loaded as `semi_structured` so both extraction paths read it |
| `corpus/notes/*.md` | Two trade-press notes, prose. Tier `community` |
| `ontology.json` | Two types and seven predicates; `headquarters` and `chief_executive` are single-valued |
| `e2e_stages.py` | One stage of the example's own: registration numbers become external ids |
| `recorded/` | The hand-authored model responses |
| `odke.yaml`, `odke.neo4j.yaml` | The run into JSON Lines, and the same run into Neo4j |
| `gold.jsonl` | 36 labelled facts, for the [ablation](evaluation.md#ablation) |

```bash
pip install "openodke[yaml]"
odke run examples/e2e/odke.yaml
```

```text
odke run
documents     8 (8 chunks; 0 skipped, 0 deferred, 0 empty)
extractor     paths (paths llm+pattern, chunks 8, pattern_facts 20, llm_facts 18, merged 2, model_calls 3), rejections (quote not in the passage 1), prompts extract@1
grounder      facts 36, calls 36, prompt_tokens 4764, completion_tokens 216, supported 20, contradicted 1, not_found 15, prompts ground.span@1, span (facts 36, located 36)
corroborator  conflicts (lost 1, won 2)
validator     accepted 25, refused (contradicted 1)
refused       1
coverage      0 of 5 sentences naming two known entities uncovered, 0 entities in no fact, 0 relations never offered, 0 unused
graph         25 facts (8 edges, 17 properties), 7 entities, 4 links (different 1, similar 3)
cost          39 model calls, 4980 tokens, USD unknown
wrote         jsonl → …/examples/e2e/out: entities.jsonl 7, facts.jsonl 25, links.jsonl 4, manifest.json
manifest      …/examples/e2e/out/manifest.json
```

Line by line:

- **documents.** One per register row and per staff row (each record is rendered to
  text, so a span can point into it), plus the factsheet and the two notes. `empty`
  is the chunks that were extracted from and yielded nothing: zero here, and a
  number to watch on a batch run, where a dropped extraction otherwise looks
  exactly like a passage with no facts in it.
- **extractor.** The pattern path read 20 facts from the CSV rows and the
  factsheet's `Key: value` lines, exactly and without a model call. The model read
  the three chunks with prose in them, once each, and two of its facts merged into
  the factsheet's pattern facts. One quote the model gave is not in its passage, and
  was dropped before anything else looked at it.
- **grounder.** One call per fact, each shown the fact and its own cited span. The
  one `contradicted` is a head office the model invented. Most of the fifteen
  `not_found` are register and staff cells, grounded against a cell that cannot say
  whose value it is.
- **corroborator.** Normalisation made a note's "10 March 2014" and the register's
  `2014-03-10` one claim. The register (curated) and a note (community) disagree on
  one company's head office; the register won, and the note's value is kept at
  reduced confidence with the reason attached.
- **validator.** The default gate refused the `contradicted` fact and kept
  `not_found`; refusing `not_found` here would throw away most of the register.
- **coverage.** Every sentence in the prose that names two known entities is
  cited by some fact, every known name the text mentions is in a fact of its
  document, and the model was shown every predicate. A run on real prose is
  rarely this clean ([the coverage report](grounding.md#what-extraction-left-behind-the-coverage-report)).
- **graph.** 25 facts on 7 entities, and 4 links: one `DIFFERENT` between two
  companies that share a name and not a registration number, and three `SIMILAR`.
  Nothing was merged away.
- **cost.** The recorded responses carry no price, so the cost is unknown rather
  than zero.
- **manifest.** The [run manifest](#the-run-manifest), added to the JSONL sink's
  `manifest.json`: the config and its hash, the two models, the two prompts,
  the ontology's fingerprint and each document's hash.

`odke.neo4j.yaml` is the same run with a Neo4j sink, `constrainer: neo4j` and
`bootstrap: true`. Its dry run connects to nothing and prints the 29 statements the
bootstrap would apply and the 16 `UNWIND … MERGE` statements the sink would send;
the example's README goes on to the Cypher that shows where each fact came from.
The recorded responses were written by hand to exercise every path of the
pipeline, so none of these numbers says anything about any model.
