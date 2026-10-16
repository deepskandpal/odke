# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

openodke 1.0 is the verification layer between any extractor and the graph
store (DECISIONS #24), as two products in one package (DECISIONS #26). The
Validator (`openodke.Validator`, `odke validate`) grounds, resolves,
corroborates, gates and writes any extractor's facts. The Evaluator
(`openodke.eval`, `odke eval`, `odke bench`) runs a pipeline on benchmarks or
your labels, and says where it loses facts and whether the Validator helps.

### Breaking changes against 0.2.1

- `openodke.Validator` names the whole layer; a `Validator` passed as the gate
  raises a `TypeError` naming `openodke.Gate` (#185, DECISIONS #26).
- `JsonlLoader.load` returns an iterator; `list(...)` gives the old value
  (#210, DECISIONS #45).
- `openodke.run.execute.register_documents` names its first parameter `stage`;
  positional calls are unchanged (#193).
- In Neo4j, an entity attribute or qualifier named `tenant` is stored as
  `attribute_tenant` or `qualifier_tenant` (#214, DECISIONS #44).
- RDF output uses the vocabulary `https://openodke.dev/vocab#` in place of
  `https://deepskandpal.github.io/odke/vocab#`; stored graphs keep theirs (#101).
- Private, outside semver: `openodke.cli.main`'s functions,
  `extract._common.best_type` and `run.build.Context._replays` changed shape.

Changed during the 1.0 cycle, never released (DECISIONS #48):

- `read_triples` returns an iterator, and `TriplesExtractor.rows` is a property
  fed one micro-batch at a time (#210).
- With `batch_size`, `Validator.validate` returns a graph of stats with no
  facts (#210).
- `Neo4jLookup(tenant_property=)` and `store_lookup.tenant_property` are gone;
  an untenanted lookup or retraction reads only untenanted data (#214).
- Every Neo4j write statement ends in `RETURN count(*)` (#214).
- `openodke.Validator` named the gate for a while, then the layer (#171, #185).

### Deprecated

Each old name still works, warns with a `DeprecationWarning` naming its
replacement, and is removed in 2.0.0 (#171, #217, DECISIONS #48).

| Deprecated | Use |
|---|---|
| `openodke.VerdictValidator`, `openodke.validators.VerdictValidator` | `openodke.VerdictGate` |
| `openodke.stages.Validator`, `openodke.pipeline.Validator` | `openodke.Gate` |
| `openodke.stages.PassThroughValidator`, `openodke.pipeline.PassThroughValidator` | `openodke.stages.PassThroughGate` |
| `Pipeline(validator=...)`, `Pipeline.validator` | `Pipeline(gate=...)`, `Pipeline.gate` |
| `Built.pipeline(validator=...)`, `StagesConfig.validator` | `Built.pipeline(gate=...)`, `StagesConfig.gate` |
| `validator:` under `stages` in a run config | `gate:` |
| `openodke.sinks.neo4j.cardinality_scope(predicate)` | `predicate.scope_keys` (#125) |

### Added

#### Validator: ground (0.3.0)

- `LLMGrounder(claim_format="labelled")` writes the claim line of the
  `ground.span` message with its parts named and no types
  (`Claim (subject | relation | object): A | rel | B`). Opt-in: the default stays
  `"inline"` until the grounding calibration on label set G decides it, and
  `stats["claim_format"]` records which was sent. `bench/labels/claim_format.py`
  compares the two on G against dataset gold and planted facts.
- The triples input format: any extractor's output as one `TripleRow` per JSON
  Lines row, read by `read_triples` and replayed by `TriplesExtractor` (#121).
- `Evidence.span_origin` (`cited`, `located` or `context`) says who chose a
  span, and every sink writes it (#121, #169, DECISIONS #25).
- `odke eval spans` counts `context` and `located` spans apart from citations
  (#121, #181).
- `from_graph_documents` reads LangChain `GraphDocument`s and `from_langextract`
  reads LangExtract output as triples rows (provisional) (#170, #172).
- `from_graphrag` and `read_graphrag` read neo4j-graphrag graphs, with each
  relationship's chunk as its evidence (provisional) (#176).
- `read_neo4j` reads a Neo4j graph back as triples rows; `write_verdicts` and
  `odke ground --write-verdicts` (off by default) write them back (#177, #184).
- `odke ground` grounds a graph openodke did not build, from any adapter, and
  summarises its verdicts and failure shapes; `--dry-run` calls no model (#184).
- `ground_graph` is `odke ground` in Python, returning a `GroundedGraph` of
  facts and summary (#184).
- `CheckedGrounder` runs the ontology's free checks ahead of any grounder and
  stamps an off-schema fact with `odke.check` (#184).
- `openodke.Validator` and `odke validate` ground, resolve, corroborate, score,
  gate and write any extractor's facts, reported in a `ValidationReport` (#185).
- `odke validate --dry-run` calls no model, opens no store and writes nothing
  (#185).
- `VerdictGate(schema=True)` refuses a fact the ontology has no room for; the
  Validator's default gate sets it, and a bare `VerdictGate` does not (#185).
- The gate stage is `Gate`, with `VerdictGate`, `PassThroughGate`,
  `Pipeline(gate=...)` and `gate:` in a run config (#171, DECISIONS #26).
- Paper mode: `LLMGrounder(context="document", verdicts="binary")` asks the
  ODKE+ paper's own prompt about the whole document; `--paper` (#105, #184).
- `LLMExtractor(structured=False)` sends no response schema; the prompt, the
  parser and the repair loop hold the model to the JSON shape (#105).
- `LLMGrounder(locate=True)` and `--locate` (off by default) find the window
  naming both ends of a fact that cited nothing, with `SpanLocator` (#181).
- `LLMGrounder(widen=True)` and `odke run --widen` (off by default) ground a
  narrow `not_found` once more against its sentence, under `odke.widen` (#180).
- The coverage report counts what extraction left behind, with no model:
  `Pipeline(coverage=True)`, on in `odke run` and the Validator (#179).
- `chunking.sentences()` gives the chunker's own sentence boundaries (#179).
- The re-extract hook, `openodke.reextract` (provisional, off by default), hands
  coverage gaps back to the extractor and grounds what returns (#183).
- `Predicate.inverse_of` and `.symmetric`: the pipeline adds each edge's
  partner, marked `odke.derived`, unless `inverses: false` (#178, DECISIONS #28).
- Every prompt is registered in `openodke.prompts` as a hashed `id@version`,
  named by each call; which version is sent is provisional (#174, DECISIONS #27).
- `odke label make` (`grounding`, `pair`, `fact`) writes markdown sheets to tick
  by hand, and `odke label read` reads them back as label rows (#175, #206).
- Label set G: 600 grounding items for hand labels, in `bench/labels/G/` (#182).
- `ModelRoles()` extracts with `anthropic/claude-sonnet-5-5` and grounds with
  `anthropic/claude-haiku-4-5-20251001` (#173).

#### Evaluator (0.4.0)

- `odke bench fetch`, `prepare` and `run` for Text2KGBench and Re-DocRED: the
  ablation scored with each dataset's own metrics (#105).
- `openodke.eval.datasets` (provisional) is `odke bench`'s Python side; `score`
  takes triples from any system, and the `bench` extra adds NLTK (#105).
- `odke bench prepare --max-tokens` sets the extractor's output room, 16,000 by
  default (#105).
- Prepared bench configs show the extractor the whole ontology, run without the
  value normaliser, and rank Re-DocRED's predicates by dev-split use (#105).
- `odke bench run` saves the predicted triples in `predictions/` and the
  extractor's rejections in `rejections.jsonl` (#105, #212).
- `openodke.eval.ablation.ablate` returns the three configurations' facts
  unscored; `run_ablation` scores them against your labels (#105).
- `EvalReport`: one versioned JSON report with 95% ranges for any scored run,
  from `odke eval <stage> --report` and `odke bench run`'s `report.json` (#186).
- The eval report's JSON Schema ships in the package, and `check_report` holds
  a report to it with no dependency (#186).
- `odke eval pipeline` scores your pipeline through `--cmd`, `--run` or
  `--predictions`; `--validator` adds the Validator's row (#190).
- `odke eval compare A B` pairs two runs' `--items` files and says better, worse
  or inconclusive, with the detection limit (#189, DECISIONS #29).
- `odke eval compare --fail-under` and `--fail-on-inconclusive` make it a CI
  gate; a worse verdict exits 1 (#189).
- `openodke.eval` has `paired_bootstrap`, `mcnemar` and `detection_limit`, on
  the standard library alone (#189).
- The eval report's `diagnosis` puts each missed gold fact in one cause bucket:
  `--trace`, `--examples` (#204, DECISIONS #38).
- The eval report's `fixes` ranks one fix per bucket by its expected recall
  gain, with its ceiling and a `--reference` run's rate (#204).
- A report written with `--report` appends its predicted fixes to
  `track-record.jsonl`; `odke eval compare --applied` adds what B measured (#204).
- `FactJudge` (provisional) asks in both orders whether a near miss is the gold
  fact; `odke eval pipeline --lenient` adds a lenient score (#206, DECISIONS #39).
- Label set F: 300 fact pairs for hand labels, in `bench/labels/F/` (#206).
- `odke eval precision` corrects the grounder's precision by a labelled sample;
  `--make-sheet` draws the sample (#201, DECISIONS #36).
- `odke eval pipeline --adjudicate` lists predictions the gold lacks that the
  grounder supports in 2 of 3 runs, each run a `ModelSpec.repeat` (#201).
- `odke eval pool` gives each grounded run its recall relative to the pool of
  runs, with the caveat in the report (#201).
- `odke eval refusals` samples what a grounder refused as a sheet, and reads it
  back as refusal precision (#211).
- `odke validate -o` writes what the gate refused to `refused.jsonl` (#211).

#### Validator: resolve (0.5.0)

- `NativeResolver(lookup=...)` resolves against the store without loading it,
  through a `StoreLookup` (#194, DECISIONS #31).
- `MemoryLookup` and `Neo4jLookup` (or `Neo4jSink.lookup()`) answer it, through
  `Validator(lookup=...)` or `store_lookup:` (off by default) (#194).
- `PairJudge` (provisional, off by default) asks in both orders about the name
  pairs the rules leave open: `NativeResolver(judge=...)` (#197, DECISIONS #34).
- `PairJudge(queue=...)` writes unsure pairs for `odke label make pair`, and
  `reviewed=` applies a person's labels with no call (#197).
- `NativeResolver(normalize_batch=True)` (provisional, off by default) merges
  a batch's look-alikes unless something keeps them apart (#207, DECISIONS #43).
- `NativeResolver(embed=...)` (provisional) keeps apart look-alikes whose
  sentences' cosine is below `context_floor` (#207).
- Label set R: 600 entity pairs from Re-DocRED's clusters, in `bench/labels/R/`
  (#199).
- Value normalisation reads units, one canonical unit per dimension (`5 km` is
  `5000 m`); `$`, `¥` and `Rs` stay as written (#191).
- The `quantity` literal range, for numbers with units (#191).

#### Validator: corroborate (0.6.0)

- `Fact.supported_by` names each independent source as a `Support`, and every
  sink writes the list (#196, DECISIONS #33).
- `SignatureCorroborator(near_duplicates=0.9)` counts near-duplicate documents
  as one source, recorded under `odke.near_duplicates` (#193).
- `SignatureCorroborator(store=...)` merges a batch with what a `FactLookup`
  sink holds, so a restated fact gains support (#200, DECISIONS #35).
- The Validator merges with its sinks by default, `odke run` with a `signature`
  corroborator; `JsonlSink(merge=True)` and `odke validate --merge` (#200, #221).
- `odke reconcile --delete` and `odke validate --update` retract a source, and
  retire a fact left with none (`Fact.retired_at`) (#203, DECISIONS #37).
- `Reconciler`, `Retractable` and `Validator.validate(update=True)`;
  `--hard-delete` deletes a fact left with no source (#203).
- `odke bench fetch`, `prepare` and `run trex` measure corroboration on facts
  several T-REx abstracts state (#208, DECISIONS #46).
- `bench/conflicts.py` runs EnterpriseRAG-Bench's 20 conflict cases through the
  corroborator's conflict order (#213).

#### Production (0.7.0)

- The pipeline extracts every chunk, then grounds every document, concurrently;
  `LLMExtractor(max_workers=8)` needs a thread-safe client (#126).
- `CachedClient` answers a request it has seen at no cost: `models.cache` and
  `--cache` (off by default) (#192, DECISIONS #30).
- `models.budget`, `--budget-usd` and `--budget-calls`: a run stops at its
  limit, keeps what it has, and exits 3 (#195, DECISIONS #32).
- Every `odke run` report prints a `cost` line (#195).
- `batch_size` and `--batch-size` (off by default) stream a run in
  micro-batches; `read_triples` reads a line at a time (#210, DECISIONS #45).
- With `batch_size`, `Validator.validate` returns a graph of stats and no
  facts; `JsonlSink.append` adds each micro-batch (#210).
- `tenant:`, `--tenant` and `Validator(tenant=...)` scope keys, signatures,
  lookups, merges and retractions to one tenant (#214, DECISIONS #44).
- Runs report each sink's rows `written`, `merged` and `skipped` (`WriteReport`);
  `Neo4jSink` packs rows into transactions of at most `batch_size` (#214).
- Every run writes a `RunManifest`, and `odke run --from-manifest` replays one
  (#205, DECISIONS #40).
- `--log-format json` writes each job, stage, document and model call as one
  JSON event with no text unless `--log-text` (#209, DECISIONS #41).
- OpenTelemetry spans (provisional) for each job, stage and model call, through
  the `otel` extra and the application's tracer provider (#209).
- `models.limits` and `set_limit` cap concurrent calls per provider across
  stages, and a `Retry-After` pauses that provider (#198).
- A failing document is left out and named in `stats["failed"]`; a run where
  every document failed exits 1 (#198).
- Every fact the gate passes carries `odke.ontology`, the `Ontology.fingerprint`
  of the ontology that checked it (#215, DECISIONS #42).
- A fact the model extracted carries `odke.schema_slice`, the fingerprint of
  the schema slice it was shown (#215).
- `Neo4jSink.bootstrap()` adds a full-text index on `evidence_doc_ids` and a
  range index on `odke.ontology` per predicate (#203, #215).
- `checked_under`, `Neo4jSink.checked_under` and `odke ontology diff --store`
  find the stored facts an ontology checked (#215).
- `bench/volume.py` measures cost, throughput, memory and the reconciler on an
  EnterpriseRAG-Bench slice (#216).

#### Numbers, docs, API freeze (0.9.0)

- The public API is pinned under semver, and `docs/stability.md` says what is
  provisional (#217, DECISIONS #48).
- The triples input format and the run report carry `schema_version` 1.0, and
  their JSON Schemas ship in the package (#217).
- `bench/layer.py` scores every extractor raw, with grounding, and with the
  whole layer; the table is in `docs/benchmarks.md` (#218).
- `bench/competitors.py` runs LangChain, neo4j-graphrag and LangExtract through
  LiteLLM, so one `--extract-model` string runs every system (#105, #218).
- The docs, trimmed, with a Quickstart and the pages The Validator, Labels and
  judges, and Stability (#187, #188, #219, #220, #222).
- `examples/run.yaml` comments every run-config key on the triples example,
  through every Validator stage, with no key (#222).
- The documentation is at <https://openodke.dev>; the old
  `deepskandpal.github.io/odke` URLs redirect (#101).
- Python 3.14 is supported, and CI tests 3.11 to 3.14 (#122).

### Fixed

- Anthropic accepts the extraction schema of an ontology past a few predicates;
  much past a dozen types, use `structured: false` (#105).
- The grounder's module docstring says the paper's grounder judges against the
  context (#105).
- A missing API key or an uninstalled provider stops the run with an error; a
  failed grounding call leaves only its fact unchecked (#126).
- `LLMExtractor(retry=...)` and `retry:` under `stages.extractor` retry
  transient provider errors (429, 5xx) (#126).
- A call to a `base_url` sends only the key `api_key_env` names (#126).
- Cost on the LiteLLM path is read from the response's `_hidden_params` (#126).
- The verdict parser reads `NotFound` and `notfound` as `not_found` (#126).
- Naming only the extraction model grounds on it too, at 256 output tokens;
  `ModelRoles()` alone keeps two models (#126, DECISIONS #7a).
- A value the corroborator voted down is never projected onto its node or
  written as a plain RDF triple (#125).
- Evidence mixing naive and timezone-aware `retrieved_at` writes, with naive
  times as UTC (#125).
- A rewritten fact replaces its properties in Neo4j, NetworkX and CSV, leaving
  no stale stamp from an earlier run (#125).
- RDF writes a literal with its predicate's datatype (`xsd:date`,
  `xsd:integer`, ...) when the value parses (#125).
- `Ontology.from_neo4j` skips openodke's own `odke.*` properties (#125).
- A URL with a path (`linkedin.com/in/...`) is no proof that two entities are
  one (#125).
- Interval qualifiers (`start_time`, `end_time`, ...) are normalised as dates,
  and a merge keeps the earlier start (#125).
- The resolver's `max_block` defaults to 100, bounding the comparisons in a
  name block (#125).
- `odke bench run text2kgbench` checks for NLTK before any model call (#124).
- `odke bench prepare` refuses an `--out` it did not write (#124).
- `odke eval extract` scores a fact merged from several documents in each one
  it cites (#124).
- `grounding_ablation` scores grounding off and on over the same facts (#124).
- `bench/`: a competitor run that lost documents exits non-zero and leaves the
  tables; `--env-file` strips quotes and `export` (#124).
- `bench/`: "rejected by grounder" counts what the gate removed in either mode,
  and `usage.json` records the competitor libraries' versions (#124).
- Text2KGBench is fetched without its unread `test` file (#124).
- Triples rows naming a file name two documents share go to the first only,
  with a warning, and the stats count `ambiguous_rows` (#123).
- The hybrid extractor merges a table cell's text with the model's typed value
  (`"1999"` and `1999`) (#123).
- The pattern extractor counts each type's predicates once, and the HTML loader
  opens no inline tag past depth 256 (#123).
- `__version__` is read from the installed distribution (#122).
- Inferred type names: "prizes", "sizes" and "caches" are "Prize", "Size" and
  "Cache" (#122).
- The release workflow's actions are pinned to commit SHAs (#122).
- `odke eval --help` points to `odke bench` for public benchmarks, and
  `odke label read --help` names `RefusalLabel` (#222).

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
