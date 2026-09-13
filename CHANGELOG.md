# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

M0 — Data model. Every change here is to a frozen type, which is why it lands
before anything is serialised. Two of them fix live correctness bugs.

### Added
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

### Changed
- The stage protocols live in `odke.stages`; `odke.pipeline` re-exports them.
  `Extractor` now takes one `Chunk` and the ontology; `Grounder` takes one fact
  and its document and sets the verdict rather than dropping; `Corroborator`
  returns facts and the pipeline assembles the graph.
- `Pipeline.run` chunks and routes before extracting, resolves before
  corroborating, validates before writing, and reports counts in
  `KnowledgeGraph.stats`. `Pipeline.constraints()` exposes the constrainer's
  output for a sink to apply.
- `Predicate.qualifiers` is a mapping; a bare list of names is still accepted
  and every name in it is reconcilable.
- `JsonlSink` writes `links.jsonl` and counts links in the manifest.
- `ROADMAP.md`: M0 before M1, M6 reshaped as Evaluation (BYOLD) inside v0.1,
  M5 in v0.5, a v0.2 breadth section, and no dates.

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
