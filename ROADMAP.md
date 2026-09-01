# Roadmap

Eight milestones. Each one ends at a state where the package still installs,
`./scripts/verify.sh` is green, and something new is usable from Python — no
milestone leaves the tree half-wired.

Estimates are in **focused working days** (a day of real, uninterrupted work).
The calendar column assumes ~12 hours a week, which is what this actually gets.

| # | Milestone | Delivers | Days | Calendar |
|---|---|---|---|---|
| M0 | Scaffold & release plumbing | ✅ done | 1 | — |
| M1 | Ontology I/O & validation | Load, validate, diff schemas | 3 | ~1.5 wk |
| M2 | Loaders & extraction | Any input → candidate facts | 9 | ~4.5 wk |
| M3 | Grounding & corroboration | The precision stages | 8 | ~4 wk |
| M4 | Sinks & the `run` command | End-to-end into Neo4j | 5 | ~2.5 wk |
| M5 | Ontology inference | No schema required | 5 | ~2.5 wk |
| M6 | Evaluation harness | Numbers you can defend | 5 | ~2.5 wk |
| M7 | v0.1.0 release | On PyPI, documented | 3 | ~1.5 wk |
|  | **Total** | | **39** | **~19 wk** |

**The useful shortcut:** M0→M4 is the first genuinely installable product — text
in, a grounded graph in Neo4j out, with an ontology you supply. That is **26 days
(~13 weeks at this pace, or ~5 weeks full-time)**. M5–M6 make it better; M7 makes
it public. If the goal is "something people can use", ship 0.1.0 at the end of M4
and treat M5–M6 as 0.2.

---

## M0 — Scaffold & release plumbing ✅

Repository, packaging, the core data model, the ontology compiler and snippet
generator, the CLI skeleton, the JSONL sink, 24 tests, a 10-step verification
gate, CI on three interpreters, and a PyPI release workflow using Trusted
Publishing.

The data model and snippet generator are real, not stubs: they are the contract
every later milestone is written against, so they had to exist before the tickets
could be written honestly.

## M1 — Ontology I/O & validation

Getting a schema *in* is the first thing every user does, and the error messages
here determine whether they get to a second run.

- `Ontology.from_json` / `from_yaml` / `from_dict`
- `Ontology.from_pydantic(*models)` — the ergonomic path for Python-first users
- `validate()`: dangling ranges, unreachable parents, orphaned domains, duplicate
  aliases, predicates that can never be extracted
- `odke ontology validate` and `odke ontology diff` — schema drift is a real
  operational problem once a graph is live
- Round-trip property test: ontology → JSON → ontology is the identity

## M2 — Loaders & extraction

The largest milestone, and the one the whole promise rests on: *"pass text,
structured or unstructured, in whatever form possible."*

- `Loader` protocol; readers for text, Markdown, HTML, PDF, DOCX, JSON, JSONL,
  CSV/TSV, Parquet
- **Span-preserving chunking.** Chunks must carry offsets back into the original
  document, or the grounder has nothing to check and provenance is decorative.
  This is the subtle part of the milestone and it is worth doing first.
- `PatternExtractor` — tables, key/value blocks, infobox-shaped HTML, JSON paths.
  Exact, free, deterministic, and it handles the structured half of the corpus
  without a single model call.
- `LLMExtractor` — snippet → structured-output call → facts with evidence spans
- `HybridExtractor` — routes each document by modality and merges the results
- The litellm shim, and a recorded-response harness so CI exercises the model path
  without a key or a network

## M3 — Grounding & corroboration

Where precision comes from. The paper's headline number is 98.8%, and it is these
two stages that produce it.

- `LLMGrounder` — per fact, one cheap model call against its own evidence span;
  `SUPPORTED` / `CONTRADICTED` / `NOT_FOUND`; batching and concurrency
- Span verification before the model is even asked: an offset that does not
  resolve to the claimed quote is rejected for free
- Normalisation — dates, numbers, units, person and organisation name forms
- Entity resolution with blocking, so it does not go quadratic on a real corpus
- Conflict resolution on freshness × source tier × agreement count
- Confidence calibration, and `support` counts on every merged fact

## M4 — Sinks & the `run` command

- `Neo4jSink` — batched, idempotent `MERGE`, provenance written onto every edge,
  and a schema-constraint bootstrap
- `CypherFileSink` and `Neo4jAdminCsvSink` for bulk loads that are too big for
  the driver
- `RdfSink` (Turtle / N-Triples / JSON-LD), `NetworkXSink`
- `Ontology.from_owl` and `Ontology.from_neo4j` — reflect a schema you already have
- `odke run` — the whole pipeline from a config file
- A worked end-to-end example against a Neo4j container

## M5 — Ontology inference

The "I don't have a schema" path. Sample the corpus, propose types and predicates,
cluster and merge near-duplicates, set `importance` from corpus support, hand back
a reviewable `Ontology` marked `inferred=True`.

Framed as a bootstrap: infer once, review, freeze, then run guided. An inferred
schema that silently drifts between runs produces a graph nobody can query.

## M6 — Evaluation harness

Precision and recall claims need a harness or they are marketing.

- A gold-standard slice with per-predicate precision / recall / F1
- Ablations that justify the architecture: extraction alone vs. + grounding vs.
  + corroboration. If grounding does not move precision, the paper's central
  claim does not reproduce and the README should say so.
- Cost and latency per thousand documents
- Runs in CI on recorded responses; runs live on demand

## M7 — v0.1.0 release

Documentation site, examples, TestPyPI rehearsal, PyPI publish, announcement.

---

## Explicitly out of scope

- **A graph database.** This produces graphs; it does not store or query them.
- **The Wikipedia refresh loop.** ODKE+'s Initiator and Retriever are specific to
  its deployment. They stay optional protocols here.
- **Reproducing the paper's benchmark numbers.** Those came from Apple's internal
  KG against private evaluation sets. Neither is available, and claiming to match
  a number nobody can check would be dishonest.
- **A UI.** The graph goes into a store that already has one.
