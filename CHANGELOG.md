# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
