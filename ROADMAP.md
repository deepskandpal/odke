# Roadmap

**Board:** [odke](https://github.com/users/deepskandpal/projects/6) ·
**Issues:** [by milestone](https://github.com/deepskandpal/odke/milestones)

openodke is the verification layer between any extractor and the graph store:
it grounds, resolves, corroborates, reconciles and evaluates facts, whoever
extracted them ([DECISIONS #24](DECISIONS.md)). Everything below is ordered by
that.

Milestones, not dates. Each one ends with the package installing,
`./scripts/verify.sh` green, and something new usable from Python. No milestone
leaves the tree half-wired. Every item is a ticket on the board. New issues go
on the board with `gh project item-add` (DECISIONS #13).

---

## Where it stands: 0.2.1

The whole pipeline shipped in 0.1.0 on 14 September 2026:
- the data model and ontology I/O;
- loaders and extraction;
- grounding, resolution and corroboration;
- six sinks, `odke run` and the evaluation harness.

0.1.1 fixed what the first outside use found, and 0.2.0 added provider
discovery. Since then, on `main` and unreleased, there is:
- a paper mode for the grounder: the whole document, binary verdicts, and only
  affirmative facts kept;
- `odke bench`: Text2KGBench and Re-DocRED, scored with each dataset's own
  metrics.

The first run of that bench decided the direction. It compared openodke with
LangChain's LLMGraphTransformer and neo4j-graphrag, using one model and one
schema. openodke's extractor trailed on recall, and its grounder improved all
three systems' precision on documents. History is in
[CHANGELOG.md](CHANGELOG.md).

## 0.3.0: the layer

**Done when** someone with output from LangChain, LangExtract, neo4j-graphrag
or a pattern extractor can run `odke ground` and `odke eval` on it without
touching openodke's extractor, and the README, docs and PyPI description say
what openodke now is.

| Item | Issue |
|---|---|
| DECISIONS #24, and this roadmap | #110 |
| A documented input format: triples plus source text, optional spans. A pattern extractor's JSONL is this format | #95 |
| Adapters: LangChain `GraphDocument`, LangExtract, a Neo4j graph read back | #97, #96, #98 |
| Adapter: neo4j-graphrag, with its lexical graph's chunks as evidence | #111 |
| `odke ground` over any graph: free deterministic checks, then the span locator, then the model grounder (span or whole-document mode) | #99, #112 |
| `odke eval` on any system's triples: rejections by reason, and a sample of refusals for a person to judge | #109, #113 |
| Inverse and symmetric relations, added without a model | #106 |
| The repositioning: README, docs home, PyPI description, the tagline | #100 |

#95 goes first, because every adapter and `odke ground` maps onto it.

## 0.4.0: corroboration and resolution that hold up

Corroboration needs canonical keys and repetition. The 0.2 benchmarks had
neither, and across six runs it changed one triple.

**Done when:**
- corroboration measurably changes precision on a public multi-source set;
- the resolver's proposed links have an audited precision;
- a delete retracts support and retires the facts left with none.

| Item | Issue |
|---|---|
| Resolver against the store, through a lookup rather than a loaded copy; within type; links, never merges (#16) | #114 |
| Support lists: which independent sources back each fact | #115 |
| Reconciler: retract a source on update or delete; retire unsupported facts | #116 |
| A multi-source public benchmark. The candidate is T-REx, which finally tests the paper's 91% to 98.8% claim fairly | #117 |
| A hand audit of 100 grounder refusals from the existing runs | #118 |

## 0.5.0: numbers

**Done when** a published table shows each extractor with and without the
openodke layer, losses included, and the README quotes it. It comes with one
written argument: *the grounder is a cheap precision gate you can put after any
extractor; here is what it costs and what it buys.*

| Item | Issue |
|---|---|
| Five extractors (LangChain, neo4j-graphrag, LangExtract, a pattern extractor, openodke's reference extractor), each with and without the layer. Datasets: Text2KGBench, Re-DocRED and the 0.4.0 multi-source set. At least two model providers | #104 |

---

## Parked: the reference extractor

`LLMExtractor`, `PatternExtractor` and `HybridExtractor` stay and keep getting
bug fixes, so someone with no extractor still has text in and a verified graph
out. Work that would extend them waits:
- enumeration recall (#103);
- a coverage prompt (#107);
- evidence across sentences (#108).

If the evidence model changes for #108, the grounder has to learn to check
multi-span claims, so that one gets revisited first. Grounded retry (#102) comes
back as a hook that hands a refusal to whichever extractor produced the fact.

Also parked: more work on ontology inference. It shipped in 0.1.0 and is kept
as it is.

## Not building

- **Retrieval or question answering.** These happen at question time, for
  different users, with a different measure of success. The layer's part is
  provenance: every fact keeps the evidence it was checked against, so any
  retriever can cite it.
- **A graph database.** This checks graphs and writes them; it does not store
  or query them.
- **More sinks.** Neo4j, RDF, NetworkX and JSON Lines, plus the bulk formats,
  are enough to prove the "any store" claim.
- **An industry-specific vertical.** Nothing domain-specific enters the code
  (DECISIONS #20).
- **The Wikipedia refresh loop.** ODKE+'s Initiator and Retriever are specific
  to its deployment, so they stay optional protocols here.
- **Matching the paper's production numbers.** Those came from Apple's internal
  graph and private evaluation sets. The bench tests the two claims that open
  data can test, and reports what it finds.
- **A labelled corpus.** `openodke.eval` scores yours, and the bench fetches
  public ones. The package ships neither.
- **A UI.** The graph goes into a store that already has one.
