# Roadmap

**Board:** [odke — roadmap to v0.1.0](https://github.com/users/deepskandpal/projects/6) ·
**Issues:** [by milestone](https://github.com/deepskandpal/odke/milestones)

Every item below is a ticket on that board, with an estimate and an area. New
issues go on the board with `gh project item-add` (DECISIONS #13).

Milestones, not dates. Each one ends at a state where the package still
installs, `./scripts/verify.sh` is green, and something new is usable from
Python — no milestone leaves the tree half-wired. Estimates are in **focused
working days** (a day of real, uninterrupted work); a milestone lands when its
days have been spent, and not on a calendar.

The scaffold — repository, packaging, the first data model, the ontology
compiler and snippet generator, the CLI skeleton, the JSONL sink, the ten-step
verification gate, CI on three interpreters and the PyPI release workflow —
shipped as 0.0.1.

---

## v0.1 — the grounded graph · released as 0.1.0, 14 Sep 2026

*"The only pipeline that asks a second model whether the cited span supports
the claim — and lets you measure every stage against your own labelled data."*

| # | Milestone | Delivers | Days | Status |
|---|---|---|---|---|
| M0 | Data model | The types that cannot change later | 3 | done (#60) |
| M1 | Ontology I/O & validation | Load, validate, diff schemas | 3 | done (#63) |
| M2 | Loaders & extraction | Text and structured input → candidate facts | 4 | done (#66) |
| M3 | Grounding & corroboration | The precision stages, and the ablation | 9 | done (#61, #65; ablation #38) |
| M4 | Sinks & the `run` command | End-to-end into Neo4j | 3 | done (#62; `odke run` #29, example #30) |
| M6 | Evaluation (BYOLD) | An evaluator per stage, for your own labels | 5 | done (#64) |
| M7 | v0.1.0 release | On PyPI, documented | 3 | done — [0.1.0 on PyPI](https://pypi.org/project/openodke/0.1.0/) (#41, #42) |
|  | **Total** | | **30** | |

M0 blocks M1–M4: every issue in it changes a frozen type, and each would be a
migration once a fact is serialised. After M0 the milestones touch different
packages and can proceed in parallel. M5 was planned for v0.5 and shipped in 0.1.0
as well — see below; the number is the board's.

### M0 — Data model

The type work that cannot be deferred. `Fact`, `Entity` and `KnowledgeGraph`
are frozen pydantic with `extra="forbid"`; every field added after people have
serialised facts is a breaking change and a migration. Two of these were live
correctness bugs, not gaps.

- `Fact.polarity` — asserted / denied / partial, **in `signature`**. A denial
  and its own contradiction were merging and raising each other's support (#47)
- Qualifier identity semantics — the ontology declares which keys bear identity
  (`percentile`, `tier`) and which reconcile (`start_time`); `signature`
  includes only the former, and DECISIONS #11 holds for the rest (#48)
- `EntityLink` with `SAME_AS | SIMILAR | DIFFERENT` and a `reason`, carried on
  `KnowledgeGraph.links`, so resolution is never destructive (#49)
- `valid_from` / `valid_to` on `Fact` — the valid clock, next to
  `retrieved_at`'s transaction clock (#50)
- `Entity.resolution` — how the key was decided, and by what (#51)
- `Chunk`, `RouteVerdict`, and a `Router` protocol that defaults to
  pass-everything (#52)
- The thirteen-Protocol surface in `openodke.stages`, each with a pass-through
  default, so a caller takes the subset they need (#53)
- `PlatformProfile` on a sink and `Delegated(to=...)`, so a stage the platform
  already does is warned about, never silently done twice (#59)

### M1 — Ontology I/O & validation

Getting a schema *in* is the first thing every user does, and the error messages
here determine whether they get to a second run.

- `Ontology.from_json` / `from_yaml` / `from_dict`
- `Ontology.from_pydantic(*models)` — the ergonomic path for Python-first users
- `validate()`: dangling ranges, unreachable parents, orphaned domains, duplicate
  aliases, predicates that can never be extracted
- `odke ontology validate` and `odke ontology diff` — schema drift is a real
  operational problem once a graph is live
- Round-trip property test: ontology → JSON → ontology is the identity

### M2 — Loaders & extraction

The milestone the promise rests on — *"pass text, structured or unstructured,
in whatever form possible"* — cut to what the pitch depends on. HTML, PDF and
DOCX readers are v0.2: a caller with a PDF can hand over text, and nobody
adopts a graph tool for its reader.

- `Loader` protocol; readers for text, Markdown, JSON, JSONL, CSV/TSV, Parquet
- **Span-preserving chunking.** Chunks must carry offsets back into the original
  document, or the grounder has nothing to check and provenance is decorative.
  This is the subtle part of the milestone and it is worth doing first — and
  worth wrapping LangExtract for rather than hand-rolling
- `PatternExtractor` — tables, key/value blocks, JSON paths. Exact, free,
  deterministic, and it handles the structured half of the corpus without a
  single model call
- `LLMExtractor` — snippet → structured-output call → facts with evidence spans
- `HybridExtractor` — routes each document by modality and merges the results
- A recorded-response harness so CI exercises the model path without a key or a
  network

### M3 — Grounding & corroboration

Where precision comes from. The paper's headline number is 98.8%, and it is
these two stages that produce it.

- Span verification before the model is even asked: an offset that does not
  resolve to the claimed quote is rejected for free. The first thing built
  after M0 — it is the cheapest possible demonstration of the differentiator
- `LLMGrounder` — per fact, one cheap model call against its own evidence span;
  `SUPPORTED` / `CONTRADICTED` / `NOT_FOUND`; batching and concurrency
- Normalisation — dates, numbers, units, person and organisation name forms
- Entity resolution — wrap splink for blocking and scoring; openodke's own part is
  `EntityLink` and the disagreement rule, on top
- Conflict resolution on freshness × source tier × agreement count
- Confidence calibration, and `support` counts on every merged fact
- **The ablation** — extraction alone vs. + grounding vs. + corroboration. It
  is the evidence for the release's sentence; if grounding does not move
  precision, the paper's central claim does not reproduce and the README says so

### M4 — Sinks & the `run` command

One sink, not five. The bulk, RDF and NetworkX sinks and the OWL and Neo4j
ontology importers are v0.2.

- `Neo4jSink` — batched, idempotent `MERGE`, provenance written onto every edge
- The constraint bootstrap — the ontology compiled into uniqueness and scoped
  cardinality constraints, which is what makes the `MERGE` safe
- `odke run` — the whole pipeline from a config file
- A worked end-to-end example against a Neo4j container

### M6 — Evaluation (BYOLD)

*Bring your own labelled dataset.* Precision and recall claims need a harness
or they are marketing — but a number computed against the pipeline's own output
measures nothing, and the package cannot label a corpus for you. So `openodke.eval`
ships, for every stage, a documented **dataset format** (what to label), an
**evaluator** (labelled set in, `StageReport` out) and a tiny **fixture** for our
own tests. It never ships a corpus, and the README says plainly that the
fixtures are not a benchmark. No extra dependencies: Brier and P/R/F1 are
arithmetic.

| Stage | Evaluator | Labels it needs | |
|---|---|---|---|
| — | dataset formats, `StageReport`, fixtures, `odke eval <stage>` | — | #36 |
| route | P/R/F1 per label, plus the skip/extract confusion — a router that skips facts is the expensive failure | chunk → action, label | #54 |
| extract | per-predicate P/R/F1, and the confusion: wrong value, wrong entity, missing, spurious | gold facts per document | #37 |
| ground | verdict accuracy and confusion; the ablation is one call to this with grounding on and off | fact + span → verdict | #55 |
| resolve | pairwise P/R and B-cubed over `EntityLink`s — including links read back from a platform's resolver, so a delegated stage is measured the same way | pairs → same / different | #56 |
| score | Brier, reliability curve, ECE — *of facts scored 0.9, X% were true* | fact → true / false | #57 |
| validate, sink | agreement with labelled verdicts; write a graph twice and count once | fact → verdict; none | #58 |
| cost | tokens, USD and latency per stage, per 1k documents; hybrid vs. model-only, cheap grounder vs. same-model | none | #39 |

### M7 — v0.1.0 release

Documentation site, examples, TestPyPI rehearsal, PyPI publish. **Done:**
0.1.0 was rehearsed on TestPyPI and published to PyPI on 14 September 2026
through trusted publishing, with a GitHub release carrying the wheel and sdist.

---

## v0.2 — breadth · shipped in 0.1.0

Planned after v0.1, because breadth is where this loses and depth after
extraction is where it wins. All additive; none touches a type. It landed
before the first release (#68, #69), so 0.1.0 includes all of it.

- HTML loader with offset preservation (#8)
- PDF and DOCX loaders (#9)
- Bulk sinks: `CypherFileSink` and `Neo4jAdminCsvSink` (#24)
- `RdfSink` — Turtle, N-Triples, JSON-LD (#25)
- `NetworkXSink` (#26)
- `Ontology.from_owl` — OWL, RDFS, SKOS (#27)
- `Ontology.from_neo4j` — reflect a live graph's schema (#28)

Between v0.2 and v0.5 sit the store's half (v0.3: constraints in both halves,
SHACL export, structure-based resolution proposals) and time (v0.4: the valid
clock in use, a staleness queue, per-predicate refresh). Neither has tickets
yet; the types they need landed in M0. Also unticketed: a Neptune
`PlatformProfile`, and the ablation run against a real model rather than
recorded responses. These are ideas, not commitments.

## v0.5 — no schema · shipped in 0.1.0

### M5 — Ontology inference

Shipped in 0.1.0 (#71). The "I don't have a schema" path. Sample the corpus, propose types and predicates,
cluster and merge near-duplicates, set `importance` from corpus support, hand back
a reviewable `Ontology` marked `inferred=True`. Deterministic routes first —
emergent schema, Hearst patterns, axiom mining — with the model naming and
ranking what those find rather than discovering.

Framed as a bootstrap: infer once, review, freeze, then run guided. An inferred
schema that silently drifts between runs produces a graph nobody can query.

---

## Explicitly out of scope

- **A graph database.** This produces graphs; it does not store or query them.
- **The Wikipedia refresh loop.** ODKE+'s Initiator and Retriever are specific to
  its deployment. They stay optional protocols here.
- **Reproducing the paper's benchmark numbers.** Those came from Apple's internal
  KG against private evaluation sets. Neither is available, and claiming to match
  a number nobody can check would be dishonest.
- **A labelled corpus.** `openodke.eval` scores yours; shipping one would be a
  benchmark nobody asked for and a claim nobody could check.
- **A UI.** The graph goes into a store that already has one.
