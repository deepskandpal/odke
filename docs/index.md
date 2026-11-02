# openodke

openodke is the verification layer between any extractor and the graph store.
It takes the triples an extractor wrote, checks each one against the text it
came from, and writes the ones that hold up, with their evidence.

## Verify

Hand any extractor's triples to `openodke.Validator`, or to `odke validate` on
the command line. It runs free checks on each citation and against the
ontology, asks a second model whether the text supports the claim, resolves entity keys, counts the sources
behind each claim, gates what is written, and writes to Neo4j, RDF, JSONL, a
Cypher file, neo4j-admin CSV or NetworkX. Each stage can be left out or
replaced.

[Quickstart](quickstart.md) · [The Validator](validator.md) · [Inputs](inputs.md)

## Evaluate

This half is the Evaluator. `odke eval` scores each stage, or your whole
pipeline, against labels you supply, and says where it loses facts and what to
change. `odke bench` runs Text2KGBench, Re-DocRED and T-REx three ways:
extraction alone, with grounding, and with corroboration.

[Evaluation](evaluation.md) · [Benchmarks](benchmarks.md)

## In short

1. **Any extractor.** LangChain's `LLMGraphTransformer`, LangExtract,
   neo4j-graphrag, a graph already in Neo4j, patterns, a script of your own, or
   openodke's reference extractor.
2. **One input format.** JSON Lines triples plus the source text. Citations are
   optional.
3. **Evidence on every fact.** Each fact keeps its document, span, source tier
   and grounding verdict.
4. **Measured.** With the same model and schema for every system, on the
   documents of Re-DocRED and T-REx, grounding removed 9–20% of each system's
   wrong triples and 0–8% of its right ones, raising precision by 1.0 to 4.3
   points with F1 unchanged, for 16–37% of what extraction cost (75% after
   LangExtract, the cheapest extractor). On Text2KGBench's single sentences it
   lowered precision by 0.7 to 2.2 points. On T-REx, where a fact can have
   several sources, an edge two or more abstracts state is right about twice
   as often as one stated once ([Benchmarks](benchmarks.md)). These grounder
   numbers are uncalibrated, and no grounder has met its calibration bar yet,
   so by default the gate marks what the grounder would refuse rather than
   dropping it ([Calibration cards](calibration.md)).
5. **Runs offline.** `pip install openodke` on Python 3.11–3.14. The base
   install needs no key, and neither does the quickstart.

[Install](installation.md) · [Quickstart](quickstart.md) · [Concepts](concepts.md)

---

An independent implementation of the ODKE+ architecture
([arXiv:2509.04696](https://arxiv.org/abs/2509.04696)); not affiliated with or
endorsed by Apple Inc. See
[NOTICE](https://github.com/deepskandpal/odke/blob/main/NOTICE).
