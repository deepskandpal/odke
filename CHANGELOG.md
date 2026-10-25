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

1.0.0rc1 is its release candidate. It holds the fixes of the 1.0 release audit
(11 Oct 2026; #225, #227, #228, #230, #231, #PR5), the first calibration
cards and the default that follows them (#226, #232, DECISIONS #54), and the
API trimmed before the freeze (DECISIONS #48).

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
- `ModelRoles()` extracts with `anthropic/claude-sonnet-5-5`, where 0.2.1
  extracted with `anthropic/claude-sonnet-5`; it grounds with
  `anthropic/claude-haiku-4-5-20251001` as before (#173).
- One document's error no longer fails a run: a document whose chunking,
  extraction, grounding or normalising raises is left out and named in
  `stats["failed"]`, and the run goes on; a run where every document failed
  exits 1 (#198).
- A gate left to its default follows its grounder's calibration card
  (`DEFAULT_POLICY = "card"`, DECISIONS #54). The default grounder's card is
  below its bar, so a fact it contradicts is written marked for review
  (`qualifiers["odke.review"]`, `marked` in the report) instead of refused, and
  a pair judge left to its default sends the pairs it decides to review.
  `VerdictGate(on_refusal="drop")` and `PairJudge(on_decision="link")` keep the
  old behaviour (#232).
- The default `RetryPolicy` is 6 tries from 1 s (waits of 31 s at most, about
  15 s on average with jitter) in place of 4 from 0.5 s (3.5 s at most), so a
  run waits out a rate limit of a few seconds instead of dropping documents. It
  governs the extractor, the grounder and the pair and equivalence judges
  (#227, release audit 8).
- A text that contradicts a claim another source supports no longer counts as
  its support: it stays out of the evidence, `supported_by`, `support` and the
  confidence, and `odke.conflict` records it as `contradicted_in` (#228,
  release audit 2).
- `DirectoryLoader` gives each document its path as its id (relative to the
  current directory, or `relative_to=`), where it gave a random one, so a rerun
  from Python is the same sources (#228, release audit 7).
- `odke eval ablation` exits 1, and scores nothing, when a document failed in
  the run (#231, rule R1).
- `LLMExtractor`: a reply that no repair makes readable fails its chunk with
  `UnreadableReply` (a `ProviderError`), so its document is left out and named
  in `stats["failed"]`; it was a "malformed reply" rejection and an empty
  extraction, which an ablation scored as misses. The raw text is still in
  `malformed` (#PR5, rule R1).
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
- 112 import paths left `__all__` before the freeze, each new in 1.0 or
  deprecated, and stay importable: the Neo4j sink's Cypher and row plumbing,
  the manifest's recorder and hashing helpers, the stream's internals, the
  pipeline harness's steps and the runner's helpers (`evaluate_pipeline` is
  the Evaluator's one entry point), the 12 names new in `run.build` and
  `run.execute`, `ground.llm.PAPER_PROMPT`, and all but 11 of
  `openodke.calibration`'s. Every name in a 0.2.1 `__all__` stays (#PR5,
  release audit call B, DECISIONS #48).
- `openodke.infer` is provisional, and so are `Initiator`, `Retriever` and
  `Router` (#PR5, calls B and C).
- A run manifest's `config_hash` is of the config with a left-out scorer
  filled as `odke run` fills it, and a named chunker, router or inferrer
  pass-through left out, so such configs hash differently than under earlier
  1.0 builds (#227).
- A Neo4j store merge into a store never bootstrapped, or missing the
  signature index of a predicate the ontology declares, raises (exit 2 from
  `odke validate`) instead of warning and replacing the stored support lists
  (#228, DECISIONS #53).
- `odke validate --update` exits 1 when a text fails after it was retracted
  (#228, release audit 13).
- `odke eval compare`: for pass/fail items, `better` and `worse` also need
  McNemar's exact p below α; a few one-way flips are `inconclusive` (#231,
  DECISIONS #29).
- `odke eval refusals --labels`: labels with no stratum sizes that span
  several strata are reported per stratum, not pooled; give the runs they were
  drawn from to weight them (#231).
- `odke eval pipeline --validator` exits 1, and scores nothing, when a
  document failed; `--adjudicate` and `--lenient` exit 1 when every model call
  behind them failed (#231, rule R1).
- `odke bench run` exits 1, and writes no report, when a document failed or a
  budget stopped the run (#PR5, rule R1).
- The eval report is schema 1.4: the `linked_other` recall bucket and the
  `failed` counts (#231). The run report is 1.1 and the eval report was 1.3
  for a while: `marked`, `calibration` and `run.claim_formats` (#226).
- `VerdictGate.refused` is empty while the gate marks; the ablation's notes say
  what refusing would have done (#232).
- `openodke.observe.spend` is gone; `openodke.manifest.spent_of` is the one
  helper (#227).

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
| `openodke.ValidationVerdict`, `openodke.types.ValidationVerdict` | `openodke.GateVerdict` (#PR5) |
| `openodke.eval.ValidationLabel`, `openodke.eval.formats.ValidationLabel` | `openodke.eval.GateLabel` (#PR5) |
| `openodke.eval.evaluate_validation`, `openodke.eval.validation.evaluate_validation` | `openodke.eval.evaluate_gate` (#PR5) |
| `odke eval validate` (also on stderr) | `odke eval gate`; the stage's labels, report and items keep the name `validate` (#PR5) |
| `openodke.ground.llm.SYSTEM_PROMPT` | `openodke.prompts.get("ground.span").text` (#PR5) |

No deprecated name is in an `__all__`, so `from openodke import *` imports none
of them and warns about none; import one by name (#PR5, release audit call C).

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

#### Calibration, the release audit, the API freeze (1.0.0rc1)

- The maintainer's hand labels for label sets G (200 of 600 items), F (100 of
  300 pairs) and R (the 100-pair audit), and all 100 of the refusal audit, in
  `bench/labels/`, under the sources' licences (#223).
- `LLMGrounder(prompt=...)`: the grounder's two challengers from
  EnterpriseRAG-Bench (Onyx, MIT), `ground.onyx@1` (verbatim) and
  `ground.statement@1` (three-way), reading the fact as `render_sentence`
  writes it (#226, #134).
- Calibration cards, `openodke.calibration` (provisional): one per prompt
  version and dated model, against a versioned label set, with the bar each
  judge must clear to be a default; `odke calibrate` makes one from your own
  grounding labels, and `ODKE_CARDS` names where yours are (#226, #135).
- The first 20 cards ship: Claude Haiku 4.5, Claude Haiku 5.5, Claude Sonnet
  5.5 and gpt-5.4-mini on the grounder's prompts, the pair judge and the
  fact-equivalence judge. Every grounder and pair-judge card is below its bar;
  every fact-equivalence card meets it. "Judge calibration" tables on the
  Benchmarks page and in `docs/calibration.md` are made from them (#232).
- Every run report prints each model judge's card line or UNCALIBRATED:
  `odke validate` (run report 1.1, `calibration`), `odke run`, and the eval
  report's `calibration` section (#226).
- `VerdictGate(on_refusal="drop" | "mark")` and `PairJudge(on_decision="link" |
  "review")`: below its bar, a judge's refusals are written marked for review
  and its pair decisions go to review (#226, DECISIONS #54).
- `bench/calibrate.py`, the grounder contest and the card runner, and
  `bench/spend.py`, the spend guard over `bench/spend.jsonl` (#226).
- `tenant=` and `scoped()` on `CypherFileSink`, `Neo4jAdminCsvSink` and
  `RdfSink`, and `odke ontology diff --store --tenant` (#228, release audit 3,
  20).
- `DirectoryLoader(relative_to=)` names each document by its path relative to
  a folder (#228).
- Refusal precision weighted by stratum: `Refusal.stratum_size` and
  `report_refusals(sizes=)`; `odke eval refusals RUN... --labels` reads the
  sizes from the runs (#231, release audit 5).
- The recall bucket `linked_other` ("pair linked by another true relation")
  and the fix `every-relation` (#231, release audit 11).
- Gate spellings of the eval APIs: `odke eval gate`, `openodke.eval.evaluate_gate`,
  `openodke.eval.GateLabel` and `openodke.GateVerdict` (#PR5, release audit
  call C).
- `odke run`'s and the Validator's stats put the gate's counts under
  `stages.gate`, and keep them under `stages.validator` until 2.0.0 (#PR5).
- `odke run`'s stats and `GroundSummary` name their format version,
  `schema_version` 1.0 (`run.execute.STATS_VERSION`,
  `interop.ground.SUMMARY_VERSION`) (#PR5).
- The run manifest ships its JSON Schema, `openodke/manifest.schema.json`
  (`manifest.SCHEMA_PATH`) (#PR5).
- The triples input and the eval report read a document of a later minor of
  their major for the fields this version knows, at every level; in a document
  of their own version or an older one, an unknown field is still refused
  (#PR5, DECISIONS #52).
- `tests/rules/`: six rules as tests over the whole codebase, with `AGENTS.md`
  as the map, `HUMAN.md` as the escalation list, and terminal recordings and
  an evidence bundle for every pull request (`scripts/record.py`,
  `scripts/evidence.py`) (#225, DECISIONS #50).

### Changed

- The version is 1.0.0rc1, "Development Status :: 4 - Beta" (#PR5, release
  audit S30).
- The source archive holds the package and its notices alone (`src`,
  `LICENSE`, `NOTICE`, `README.md`, `CHANGELOG.md`): 0.2.1's held the label
  sets, CC BY-SA text, under an Apache-2.0 licence expression (#PR5, release
  audit 17).
- Every Neo4j fact relationship carries `odke.version`, a token each write
  renews (#228, DECISIONS #51).
- A streamed run into a merging JSONL store warns once that its time grows
  with the square of the store (#228, S10).
- `odke` prints an `error:` prefix on each line of a multi-line error, and the
  span locator and coverage fold ASCII words without the Unicode tables, about
  a fifth faster (#227, S6, S7).
- Prepared bench sets name `normalizer: passthrough` and `resolver:
  passthrough`, so `odke eval pipeline --bench --validator` runs what `odke
  bench run` runs (#231, release audit 16); their gate says
  `on_refusal: drop` (#232), and a T-REx set's pair judge
  `on_decision: link` (#PR5), so the published numbers measure what each
  judge decides.
- The bench's Wilson intervals are the package's, with the exact z: one
  published figure moves, the T-REx support table's 3+ row, [43.8, 100.0] to
  [43.9, 100.0] (#PR5, release audit S29).
- `bench/layer.py` leaves out a competitor's set whose extraction lost a
  document, and says so: neo4j-graphrag's Text2KGBench rows are on 180 of 200
  sentences (#PR5, release audit 12).
- The public-API pin also holds every public callable's parameter names and
  kinds and every model's fields (#230, release audit S23); CI re-makes
  `bench/trex.py`'s published T-REx tables and runs `bench/test_competitors.py`
  in a job of its own, with hash-pinned bench requirements (#230, S26).
- Dependencies are taken only once they have been on PyPI for 7 days
  (`[tool.uv] exclude-newer`) (#225).

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
- A recorded answer (`models.replay`) is never cached as the live model's:
  recordings are cached under `<cache>/replay/<hash of the file>` (#227,
  release audit 1).
- A URL's password never reaches an error message, the manifest's `failed`
  list, the logs, stderr or a dry run's Neo4j target (#227, release audit 9).
- A streamed run's cost per 1,000 documents divides by every document, not the
  last micro-batch's (#227, release audit 14).
- `odke run --from-manifest` and `read_manifest` read a manifest a newer
  release wrote, ignoring the fields they do not know (#227, release audit 18).
- Warning events no longer print their bare names on stderr with text logs;
  `odke eval refusals --n 150` draws 150; one pipeline written three ways
  hashes alike; the manifest is written atomically (#227, S1, S2, S4, S8).
- Two jobs writing one Neo4j at once no longer lose a source or bring back a
  deleted one: a version on each fact, locked and compared on write (#228,
  release audit 4, DECISIONS #51).
- The Cypher-file, neo4j-admin CSV and RDF sinks key a tenant's writes, so two
  tenants' files never merge, and `Neo4jSink.checked_under` reads one tenant's
  facts (#228, release audit 3, 20).
- A large retraction goes 500 documents a query; a grounder passed in reports
  each job's free checks; both Neo4j readers decode the same stamps; retracting
  a parent retires its derived partner in a tenant's store; an untenanted key
  reading `<tenant>/...` never lands on that tenant's node; two writers into one
  JSONL directory never read a torn file (#228, S9, S11-S15).
- The standard-library OpenAI client sends `max_completion_tokens` to OpenAI's
  own API, so GPT-5 and later can be called; compatible servers keep
  `max_tokens` (#229).
- `odke label read` refuses a line that looks like an item's heading but is not
  one, naming its file and line (#230, S25).
- The refusal precision weights each stratum by its size, with a stratified
  Wilson range; PPI's corrected range no longer shrinks to zero width; `odke
  eval compare` gives no verdict on a handful of flips (#231, release audit 5,
  6, 10).
- The miss diagnosis no longer counts a correct prediction as a wrong
  relation; a triple written twice in one document is one prediction; labels
  join facts read back from Neo4j by id and document (#231, release audit 11,
  15, 19).
- The track record credits a fix only when its knobs alone changed; an
  in-process `--run` report names its prompt and model; the blind PPI sample
  carries no `odke.*` stamps; the diagnosis and the lenient score share one
  pre-filter; `paired_bootstrap` keeps fractional scores; sheets escape `&`
  (#231, S17-S22).
- `bench/conflicts.py`: a case whose document or grounding call failed is a
  `failed` outcome, counted apart, and the run exits 1; a refused key read as
  "no conflict found". `bench/volume.py` leaves a failed document out of the
  cost per 1,000 (#PR5, release audit 12).
- The docs home page and the README status match the benchmarks, the
  calibration cards and 1.0 (#PR5, S27); `run.md`, `benchmarks.md` and
  `evaluation.md` match the code (#PR5, S28).

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
