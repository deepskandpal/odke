# `odke run`

`odke run` runs the whole pipeline from one file. The file names the inputs and
how to read them, the ontology, which model does which job, which implementation
fills each of the thirteen stages, and where the graph goes. It is data, for the
same reason a `ModelSpec` is: it can be diffed, logged next to the graph it
produced, and read by someone who has never seen the code. Every key is checked,
so a misspelt stage name is an error with a suggestion rather than a stage silently
left as the pass-through.

```bash
odke run examples/e2e/odke.yaml              # run it and write the graph
odke run examples/e2e/odke.yaml --dry-run    # load, extract and ground; print what would be written
```

| Exit status | Means |
|---|---|
| 0 | the run finished (a dry run included) |
| 2 | the config cannot run: a bad key, a missing file, a missing extra |
| 1 | the run failed: a provider error, or a file that could not be read or written |

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
stages:
  chunker: {use: sentence, max_words: 120}
  extractor: hybrid                # the one required stage
  grounder: llm
  normalizer: {use: value, person_types: [Person]}
  resolver: native
  corroborator: signature
  scorer: evidence
  validator: verdict
  sink: {use: jsonl, directory: out}
bootstrap: false
```

| Key | Required | What it is |
|---|---|---|
| `ontology` | yes | A JSON or YAML ontology (`.yaml`/`.yml` is read as YAML). Loaded strictly, so a schema with validation errors stops the run before anything is spent. |
| `inputs` | yes, at least one | Files or directories. A string is a path read by the default loader; a mapping is `{path, loader}`. |
| `pythonpath` | no | Directories put on `sys.path` before a `package.module:Name` stage is imported. |
| `models` | no | Which model does which job, recorded responses, and the cost meter. |
| `stages` | yes | Which implementation fills each of the thirteen stages. Only `extractor` is required. |
| `bootstrap` | no, default `false` | Apply the ontology's constraints through the sink before the first write. |

Every relative path (the ontology, each input, `pythonpath`, replay files, a sink's
output) resolves against the directory the config file is in, so a config runs the
same from any working directory.

### `inputs`

An input's `loader` is a stage spec like any other: a short name, with options as
extra keys. An input without one is read by `stages.loader`, and when that is left
out too, by the `directory` loader, which reads every suffix it knows and warns
about and skips a file whose extra is missing
([Loaders](loaders-and-extraction.md#directories)).

**Document ids are source paths.** A document read from a file gets the file's path
relative to the config as its id, plus `#L<line>` for a record with a source line
(CSV, TSV, JSONL) or `#<row>` for one without (JSON, Parquet): `corpus/register.csv#L2`.
A second document from the same path is `~2`, `~3`. A labelled set and a provenance
query can both name a document by a string you already know, instead of a UUID
minted by the run.

### `models`

`extract`, `ground` and `infer` are each a model string or a full `ModelSpec`
(`model`, `temperature`, `max_tokens`, `timeout`, `base_url`, `api_key_env`, `extra`); left out,
a role takes its `ModelRoles` default. Keys never go in the file: a provider reads
its own environment variable, or the one `api_key_env` names.

- **`replay`** maps a role to a file of recorded responses, either a cassette
  object (`ReplayClient`) or a list of match entries (`RecordedClient`), so a run
  needs no key and no network. Delete the lines to call the models.
- **`meter: true`** wraps every model client in a `CostMeter` and puts calls,
  tokens, USD and latency per stage into the graph's stats. A cost no provider
  reported stays unknown rather than being counted as zero.

### `stages`

A stage is a built-in's short name or `package.module:Name` for your own, and
either may take options. `grounder: llm` and `grounder: {use: llm, max_workers: 4}`
are both accepted: every key except `use` is passed to the implementation as a
keyword argument, and an option it does not take is an error listing the ones it
does.

**A stage left out is the pass-through** from `openodke.stages`
([DECISIONS #20](decisions.md)), and the stats say nothing about it.

**A stage of your own** is `package.module:Name`. A class is constructed with the
options; any other object is used as it is and takes no options. Either way it must
satisfy the stage's Protocol, which is checked when the config is built rather than
discovered halfway through a run.

**`delegated`**, with `to:`, marks a stage the store does itself
([DECISIONS #21](decisions.md)): `resolver: {use: delegated, to: neo4j-graphrag:FuzzyMatchResolver}`.
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
| `extractor` | `pattern`, `llm`, `hybrid` | `PatternExtractor`, `LLMExtractor`, `HybridExtractor` | `pattern`: `mappings`, `subject_type`, `confidence`; `llm`: `types`, `snippet_limit`, `confidence`, `repairs`; `hybrid`: `llm` (options, or `false`), `pattern` (options) |
| `grounder` | `span`, `llm`, `passthrough`, `delegated` | `SpanGrounder`, `LLMGrounder` | `llm`: `max_workers`, `retry` (`attempts`, `base_delay`, `multiplier`, `max_delay`, `jitter`) |
| `normalizer` | `value`, `passthrough`, `delegated` | `ValueNormalizer` | `day_first`, `person_types` |
| `resolver` | `native`, `passthrough`, `delegated` | `NativeResolver` | `threshold`, `nudge_up`, `nudge_down`, `max_block` |
| `corroborator` | `signature`, `passthrough`, `delegated` | `SignatureCorroborator` | `half_life_days`, `freshness_floor`, `intervals` |
| `scorer` | `evidence`, `passthrough`, `delegated` | `EvidenceScorer` | `prior`, `verdict_weights` |
| `validator` | `verdict`, `passthrough`, `delegated` | `VerdictValidator` | `refuse_not_found` |
| `sink` | `jsonl`, `neo4j`, `cypher_file`, `neo4j_admin_csv`, `rdf`, `networkx` | the [sinks](sinks.md) | see [below](#sinks) |
| `constrainer` | `neo4j`, `passthrough`, `delegated` | `Neo4jConstrainer` | — |
| `inferrer` | `passthrough` only | — | `odke run` never infers ([below](#the-inferrer)) |

### Sinks

`sink` takes one sink or a list of them; none writes nothing.

| `use` | Writes | Options |
|---|---|---|
| `jsonl` | [`JsonlSink`](sinks.md#jsonl) | `directory` (required) |
| `neo4j` | [`Neo4jSink`](neo4j.md) | `uri` or `uri_env` (one required); `user` or `user_env` (default `neo4j`); `password_env` (default `NEO4J_PASSWORD`); `database`; `batch_size` (default 500) |
| `cypher_file` | [`CypherFileSink`](sinks.md#cypher-file) | `path` (required); `batch_size` (default 500) |
| `neo4j_admin_csv` | [`Neo4jAdminCsvSink`](sinks.md#neo4j-admin-csv) | `directory` (required); `delimiter` (default `,`); `array_delimiter` (default `;`) |
| `rdf` | [`RdfSink`](sinks.md#rdf), needs `rdf` | `path` (required); `format` (`turtle`, `nt`, `json-ld`; default from the suffix); `base`; `schema` |
| `networkx` | [`NetworkXSink`](sinks.md#networkx), needs `networkx` | `path`: where to write the filled graph as node-link JSON. Without it the graph is filled in memory and written nowhere, so give it a `path`. |

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

### The inferrer

`inferrer` accepts only `passthrough`. Anything else is refused:
`odke run never infers an ontology`. Inference is a bootstrap, not a mode
([DECISIONS #8](decisions.md)): run `odke ontology infer` once, review and freeze
the result, and name the file under `ontology`. See
[Ontology inference](inference.md).

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
| `graph` | `facts`, `edges`, `properties`, `entities`, and `links` by kind |
| `stages.extractor` | `paths` (the hybrid's `PathReport` totals), `rejections` by reason, or `model_calls` |
| `stages.grounder` | calls, retries, failures, a count per verdict, tokens, `cost_usd`, and the span check's own counts |
| `stages.corroborator` | `conflicts`: how many facts `won`, `lost` or `tied` a contest |
| `stages.validator` | `accepted`, and `refused` by reason |
| `stages.<name>` | anything else a stage reports by carrying a `stats` mapping, your own stages included |
| `cost` | with `meter: true`: calls, tokens, USD and latency, in total and per role |

A `DoubleStageWarning` raised while the pipeline is built is printed as a
`warning:` line on standard error.

## The default gate: `VerdictValidator`

The grounder stamps a verdict and drops nothing ([DECISIONS #20](decisions.md)), so
something has to refuse. `validator: verdict` is that something:

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
from openodke import Entity, Fact, GroundingVerdict, Ontology, VerdictValidator

gate = VerdictValidator()
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
assert VerdictValidator(refuse_not_found=True).refused == {
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
pip install "openodke[yaml] @ git+https://github.com/deepskandpal/odke"
odke run examples/e2e/odke.yaml
```

```text
odke run
documents     8 (8 chunks; 0 skipped, 0 deferred)
extractor     paths (paths llm+pattern, chunks 8, pattern_facts 20, llm_facts 18, merged 2, model_calls 3), rejections (quote not in the passage 1)
grounder      facts 36, calls 36, prompt_tokens 4764, completion_tokens 216, supported 20, contradicted 1, not_found 15, span (facts 36, located 36)
corroborator  conflicts (lost 1, won 2)
validator     accepted 25, refused (contradicted 1)
refused       1
graph         25 facts (8 edges, 17 properties), 7 entities, 4 links (different 1, similar 3)
cost          39 model calls, 4980 tokens, USD unknown
wrote         jsonl → …/examples/e2e/out: entities.jsonl 7, facts.jsonl 25, links.jsonl 4, manifest.json
```

Line by line:

- **documents.** One per register row and per staff row (each record is rendered to
  text, so a span can point into it), plus the factsheet and the two notes.
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
- **graph.** 25 facts on 7 entities, and 4 links: one `DIFFERENT` between two
  companies that share a name and not a registration number, and three `SIMILAR`.
  Nothing was merged away.
- **cost.** The recorded responses carry no price, so the cost is unknown rather
  than zero.

`odke.neo4j.yaml` is the same run with a Neo4j sink, `constrainer: neo4j` and
`bootstrap: true`. Its dry run connects to nothing and prints the 15 statements the
bootstrap would apply and the 16 `UNWIND … MERGE` statements the sink would send;
the example's README goes on to the Cypher that shows where each fact came from.
The recorded responses were written by hand to exercise every path of the
pipeline, so none of these numbers says anything about any model.
