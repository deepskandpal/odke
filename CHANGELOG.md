# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


### Changed

- The documentation site moved to <https://openodke.dev>. The old
  `deepskandpal.github.io/odke` URLs redirect.
- **The RDF vocabulary namespace moved with it**, from
  `https://deepskandpal.github.io/odke/vocab#` to `https://openodke.dev/vocab#`.
  An IRI is an identifier, so this changes the terms in RDF output: graphs
  written by an earlier version use the old namespace, and a store holding both
  sees two sets of terms. Nothing rewrites them for you. Moving it now, while
  the sink is new, is cheaper than moving it later.
## [0.2.1] — 2026-09-26

### Fixed

- The author's name is Deepanshu Kandpal. It was misspelt in the packaging
  metadata, so PyPI showed the wrong author from 0.1.0 to 0.2.0, and in the
  LICENSE and NOTICE copyright lines. A `.mailmap` corrects what git displays
  for the commits, which keep their original author.

## [0.2.0] — 2026-09-26

Provider selection and discovery, and design notes argued from first principles
rather than from measurements taken elsewhere.

### Added
- **`odke models`**, and `--model` / `--model-provider` on `odke run` and
  `odke ontology infer`. The command lists every provider openodke can address,
  the form its model string takes, the environment variable it reads, whether
  that variable is set, and which client serves it — the standard-library
  OpenAI-compatible one, litellm, or one registered with `openodke.llm.register`.
  It runs on the base install, with no key and no network, and never prints a
  value. The provider table (`openodke.llm.providers`) grows from five providers
  to twenty-one on litellm's own conventions, Azure's `AZURE_API_KEY` /
  `AZURE_API_BASE` / `AZURE_API_VERSION` included, and a missing key is now
  `MissingAPIKey` naming the variable, raised before the request instead of
  arriving as a provider's 401. Keys are read from the environment and stored
  nowhere: there is no command that writes one, and no model catalogue is
  shipped. New page: [Models and providers](docs/models.md). (#83)

## [0.1.1] — 2026-09-26

Five defects and one addition from the first use of the published package by
someone who did not write it.

### Added

- `odke eval spans` and `openodke.eval.evaluate_spans`: cited span width split by
  grounding verdict — count, median, quartiles, min and max per verdict, the
  share of facts citing no span, and a summary line naming the gap when the
  distributions separate. The one evaluator that needs no labelled data: where
  the `not_found` widths sit below the `supported` ones, the `not_found` rate is
  a proxy for citation quality with no gold set at all. (#80)

### Fixed

- **The model extractor cites the claim-bearing clause (#77).** It used to cite
  the narrowest text that told one fact from its siblings — `Ireland` out of a
  clause listing three regions — and the grounder, shown that span and nothing
  else, correctly answered `not_found`, so every enumerated fact was lost and
  the loss looked like a refusal. `Evidence` gains an optional `mention`: the
  prompt and the structured-output contract now ask for the clause as the quote
  and the distinguishing words as the mention, both checked with
  `Span.is_faithful`, and a mention outside its clause is dropped rather than
  the fact ([DECISIONS #23](DECISIONS.md)). Additive: facts serialised by 0.1.0
  still load.
- **An empty extraction is visible (#78).** `Pipeline` stats gain
  `empty_extractions`, printed on the `documents` line of `odke run`, and
  `LLMExtractor` counts the same thing per chunk, logs a WARNING naming the
  chunk and the reason it came back empty, and keeps the raw text of a repair
  that failed in `malformed`. Nothing changes when an extraction succeeds.
- `ModelSpec.temperature` may be `None`, and both clients — the standard-library
  OpenAI-compatible one and the litellm adapter — then leave the parameter out of
  the request rather than sending a value. Which temperatures a model accepts is a
  per-model fact, and openodke does not track it: `litellm.drop_params` was not used,
  because it discards a temperature the caller did mean. (#79)
- `odke run` fills the `scorer` stage with `evidence` (`EvidenceScorer`) when the
  config leaves it out, so a written fact's confidence reflects the grounding
  verdict instead of arriving at the sink as the extractor's constant, which no
  threshold can use. `Pipeline`'s pass-through default is unchanged
  (DECISIONS #20) and `scorer: passthrough` opts out. (#81)

### Changed
- **`ModelRoles()` no longer sets `temperature=0.0` on any role**; it sets nothing,
  so each provider's own default applies. The shipped default configuration could
  not make a call: its extract model accepts only `temperature=1`. A caller who
  relied on the implicit `0.0` now gets the provider's default and should set
  `ModelSpec(temperature=0.0)` to keep it — an explicit value is still sent
  unchanged. (#79)

## [0.1.0] — 2026-09-14

The v0.1 milestones now on `main`: the data model (M0), ontology I/O (M1),
loaders and extraction (M2), grounding and corroboration (M3), the Neo4j sink and
`odke run` (M4), and evaluation against your own labels (M6). Published to PyPI on 14 September 2026.

### Added

#### M0 — Data model (#60)

Every change here is to a frozen type, which is why it landed before anything
was serialised. Two of them fix live correctness bugs.

- `Fact.polarity` (`Polarity`: asserted / denied / partial), and it is part of
  `Fact.signature`. A denial no longer merges with its own contradiction. (#47)
- Qualifier identity semantics. `Predicate.qualifiers` maps each key to a
  `Qualifier(identity=...)`; `Ontology.identity_keys()` and `Fact.identity_keys`
  carry the identity-bearing keys onto the fact, and `signature` includes them,
  sorted. Reconcilable qualifiers stay out, as before. (#48)
- `EntityLink` (`LinkKind`: SAME_AS / SIMILAR / DIFFERENT, with `score`,
  `evidence` and a `reason` naming the identifier that disagreed) and
  `KnowledgeGraph.links`, so resolution is never destructive. (#49)
- `Fact.valid_from` / `valid_to` — the valid clock, alongside `retrieved_at`'s
  transaction clock. Neither is in the signature. (#50)
- `Resolution` and `Entity.resolution` — how the key was decided: by the caller,
  an external id, or a named linker with a score. (#51)
- `Chunk`, `RouteVerdict` and the `Router` protocol; the default passes every
  chunk. (#52)
- `openodke.stages`: thirteen stage protocols — `Loader`, `Chunker`, `Router`,
  `Extractor`, `Grounder`, `Normalizer`, `Resolver`, `Corroborator`, `Scorer`,
  `Validator`, `Sink`, `Constrainer`, `Inferrer` — each with a pass-through
  default, plus `ValidationVerdict`. (#53)
- `PlatformProfile` (declared on a sink), `Delegated(to=...)` (a pass-through
  that satisfies every stage protocol and stamps provenance), and
  `DoubleStageWarning`, raised once when a stage is configured in openodke and the
  sink's platform does it too. Warned, never refused. (#59)

#### M1 — Ontology I/O & validation (#63)

- `Ontology.from_dict`, `from_json` and `from_yaml`. A failed load raises
  `OntologyLoadError` with one line per problem: the dotted path to the key,
  what was found there, a did-you-mean for a misspelt key, and a line and column
  for a syntax error. `strict=True` also validates. PyYAML is imported lazily,
  behind the new `[yaml]` extra.
- `Ontology.from_pydantic(*models)`: models become entity types, fields become
  predicates, a field typed as another model is an edge. Adds
  `Predicate.required`.
- `Ontology.validate()`, returning `Diagnostic`s and never raising: unknown
  ranges, parents and keys, unreachable predicates, duplicate aliases, name
  mismatches and inheritance cycles, each with the exact path.
- `Ontology.diff()`, marking every change breaking or compatible, and the
  `odke ontology validate` and `odke ontology diff [--fail-on-breaking]` commands.
- `Predicate.cardinality_scope` and `Predicate.scope_keys`: what a single-valued
  predicate is single *within*, shared by the corroborator and the Neo4j check.

#### M2 — Loaders & extraction (#66)

- `SentenceChunker(max_words, overlap)`: whole sentences, paragraph breaks
  preferred, `doc.text[start:end] == chunk.text` always. (#6)
- `TextLoader`, `MarkdownLoader` (the raw file as text, the heading outline with
  offsets in metadata) and `DirectoryLoader`, reading bytes so CRLF offsets are
  file offsets. (#7)
- `CsvLoader`, `TsvLoader`, `JsonLoader`, `JsonlLoader`, `RecordsLoader` and
  `record_document`: one structured document per record, rendered so a span can
  point into it. `ParquetLoader` behind the new `[parquet]` extra. (#10)
- `PatternExtractor` and `RecordMapping`: records, pipe tables and `Key: value`
  blocks to facts, with no model call. (#11)
- `LLMExtractor`: ontology snippets in, facts with checked evidence spans out;
  every drop in `rejections`, every call's usage in `calls`. (#12)
- `HybridExtractor` and `PathReport`: routed by modality, merged by signature. (#13)
- `ReplayClient`, `Cassette` and `RecordingClient`: model paths in CI with no key
  and no network. (#14)

#### M3 — Grounding (#61)

- `check_span`, `SpanStatus` and `SpanGrounder`: an offset that does not resolve
  to its quote is rejected before any model is asked. (#15)
- `LLMGrounder`, `render_claim` and `parse_verdict`: one fact, one span, one
  verdict from the `ground` role; `RecordedClient` for thread-safe replay. (#16)
- `LLMGrounder.ground_many`, `RetryPolicy` and `is_transient`: a document's facts
  grounded concurrently, transient errors retried, a failed call left
  `unchecked` rather than failing the run. (#17)

#### M3 — Normalise, resolve, corroborate, score (#65)

- `ValueNormalizer`: dates, numbers, quantities and name keys to one form each,
  refusing when unsure and keeping the source spelling. (#18)
- `NativeResolver`: blocking, strong identifiers, a scored name match, and the
  disagreement rule — a `DIFFERENT` link naming both identifiers. (#19)
- `SignatureCorroborator`: merge by signature, `support` as independent sources,
  contested values ranked on trust × freshness × volume-discounted agreement;
  losers kept with the reason. (#20)
- `EvidenceScorer`: confidence from the extractor, the verdict, support and any
  lost contest, with its inputs kept on the fact. (#21)

#### M4 — Neo4j (#62)

- `Neo4jSink`: batched, idempotent `UNWIND … MERGE` on entity keys and fact
  signatures, provenance on every fact relationship, literal facts as `:Claim`
  nodes, `EntityLink`s as relationships. The driver is imported lazily behind
  `[neo4j]`. (#22)
- `Neo4jConstrainer`, `Neo4jSink.bootstrap()` and `Neo4jSink.check()`: the
  ontology compiled into uniqueness constraints and indexes, and a check query
  per single-valued predicate for what Neo4j cannot enforce. (#23)

#### M6 — Evaluation, bring your own labelled dataset (#64)

- `StageReport`, a JSONL format per stage, fixtures that are not a benchmark, and
  `odke eval <stage> --labels … [--predictions … | --run …] [--describe]`. (#36)
- Evaluators for routing (#54), extraction with the four-way error split (#37),
  grounding and `grounding_ablation` (#55), resolution with B-cubed (#56),
  calibration with Brier, reliability and ECE (#57), and validation with sink
  idempotency (`check_idempotency`, `assert_idempotent`) (#58).
- `CostMeter`, `CostReport` and `compare_costs`: tokens, USD and latency per
  stage, with unknown cost kept unknown. (#39)

#### M4 / M6 — The run command, the example, the ablation

- `odke run config.yaml [--dry-run]` and `openodke.run`: the whole pipeline from one
  YAML or JSON file — inputs and loaders, ontology, model roles with recorded
  responses and a cost meter, the implementation of each of the thirteen stages
  by short name or `package.module:Name`, sinks, and `bootstrap`. Every stage's
  counts are copied into `KnowledgeGraph.stats`, and a document's id is its
  source path. `examples/run.yaml` comments every key. (#29)
- `VerdictValidator`: the default gate, refusing `contradicted` and, on request,
  `not_found`. (#29)
- `examples/e2e/`: an invented corpus, ontology, recorded responses and configs,
  run into JSON Lines or Neo4j, with the queries that show provenance, a
  `DIFFERENT` link and a cardinality check. (#30)
- `odke eval ablation --config … --labels …` and `openodke.eval.run_ablation`:
  extraction alone, + grounding, + corroboration over your own labels. (#38)
- `odke run` short names for the loaders and sinks that landed after its
  builder: loaders `html`, `pdf` and `docx`; sinks `cypher_file`,
  `neo4j_admin_csv`, `rdf` and `networkx` (which also writes node-link JSON
  when given `path`). Options pass through as extra keys. A missing extra is a
  config error naming it, raised while the config is built and before any sink
  is opened, and each input's loader is now built then too.

### Changed
- `odke ontology infer` with no readable documents lists every suffix
  `DirectoryLoader` reads, taken from its own table, so `.html`, `.pdf` and
  `.docx` are no longer missing from the hint. The `docs` extra is pypdf and
  python-docx: beautifulsoup4 and lxml were in it and nothing imported them.
- The distribution and the import package are now `openodke` (`pip install
  "openodke[neo4j]"`, `import openodke`); the command is still `odke`, and an
  `openodke` command runs the same app. The reserved `odke.*` keys, the RDF
  `odke:` vocabulary and the Neo4j `odke_*` schema names are unchanged. Nothing
  had been published under the old name, so there is no compatibility shim.
  (DECISIONS #22)
- The stage protocols live in `openodke.stages`; `openodke.pipeline` re-exports them.
  `Extractor` takes one `Chunk` and the ontology; `Grounder` takes one fact and
  its document and sets the verdict rather than dropping; `Corroborator` returns
  facts and the pipeline assembles the graph. (#60)
- `Pipeline.run` chunks and routes before extracting, resolves before
  corroborating, validates before writing, and reports counts in
  `KnowledgeGraph.stats`. `Pipeline.constraints()` exposes the constrainer's
  output for a sink to apply. (#60)
- `Pipeline.run` grounds a document's candidates together, through
  `ground_many` when the grounder has it. (#61)
- No `DoubleStageWarning` for a constrainer whose `platform` matches the sink's
  profile: it is the store's other half, not a second pass. (#62)
- `Predicate.qualifiers` is a mapping; a bare list of names is still accepted
  and every name in it is reconcilable. (#60)
- `JsonlSink` writes `links.jsonl` and counts links in the manifest. (#60)
- README rewritten around what runs today, with install from GitHub until the
  PyPI release. ROADMAP marks M0–M4 and M6 done. (#43)

## [0.0.1] — 2026-09-01

The scaffold. Everything here is the contract later milestones are written
against, not a preview of the finished library.

### Added
- Core data model: `Document`, `Span`, `Evidence`, `Entity`, `Fact`,
  `KnowledgeGraph`, with frozen semantics and character-offset provenance.
- Ontology compiler: `Ontology`, `EntityType`, `Predicate`, inheritance-aware
  `predicates_for()`, and cycle-safe `lineage()`.
- `OntologySnippet` — the ranked, per-type schema fragment from the ODKE+ paper,
  rendered either as prose or as JSON Schema from one object.
- Provider-neutral model layer: `LLMClient` protocol, `ModelSpec`, `ModelRoles`,
  a standard-library OpenAI-compatible client (Ollama, vLLM, LM Studio,
  llama.cpp, OpenRouter, Groq, Together, DeepSeek, gateways), a litellm adapter
  for everything else, a `register()` escape hatch, and `ScriptedClient` for
  offline tests.
- `Pipeline` and the five stage protocols: `Initiator`, `Retriever`, `Extractor`,
  `Grounder`, `Corroborator`, plus `Sink`.
- `JsonlSink`, and the `odke` CLI with `ontology snippet` and `ontology types`.
- `scripts/verify.sh` — ten checks, run identically in CI and locally.
- CI on Python 3.11/3.12/3.13; PyPI release via Trusted Publishing.

[Unreleased]: https://github.com/deepskandpal/odke/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/deepskandpal/odke/releases/tag/v0.0.1
