# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The v0.1 milestones now on `main`: the data model (M0), ontology I/O (M1),
loaders and extraction (M2), grounding and corroboration (M3), the Neo4j sink and
`odke run` (M4), and evaluation against your own labels (M6). Not yet on PyPI:
the release (M7) waits on Trusted Publishing.

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
- `odke.stages`: thirteen stage protocols — `Loader`, `Chunker`, `Router`,
  `Extractor`, `Grounder`, `Normalizer`, `Resolver`, `Corroborator`, `Scorer`,
  `Validator`, `Sink`, `Constrainer`, `Inferrer` — each with a pass-through
  default, plus `ValidationVerdict`. (#53)
- `PlatformProfile` (declared on a sink), `Delegated(to=...)` (a pass-through
  that satisfies every stage protocol and stamps provenance), and
  `DoubleStageWarning`, raised once when a stage is configured in odke and the
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

- `odke run config.yaml [--dry-run]` and `odke.run`: the whole pipeline from one
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
- `odke eval ablation --config … --labels …` and `odke.eval.run_ablation`:
  extraction alone, + grounding, + corroboration over your own labels. (#38)
- `odke run` short names for the loaders and sinks that landed after its
  builder: loaders `html`, `pdf` and `docx`; sinks `cypher_file`,
  `neo4j_admin_csv`, `rdf` and `networkx` (which also writes node-link JSON
  when given `path`). Options pass through as extra keys. A missing extra is a
  config error naming it, raised while the config is built and before any sink
  is opened, and each input's loader is now built then too.

### Changed
- The stage protocols live in `odke.stages`; `odke.pipeline` re-exports them.
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
