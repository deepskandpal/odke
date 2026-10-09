# openodke

openodke is the verification layer between any extractor and the graph store.
It takes the triples an extractor wrote, checks each one against the text it
came from, and writes the ones that hold up, with their evidence.

## Verify

Hand it any extractor's triples. It checks each cited span for free, asks a
second model whether the text supports the claim, resolves entity keys, counts
the sources behind each claim, gates what is written, and writes to Neo4j, RDF,
JSONL, a Cypher file, neo4j-admin CSV or NetworkX. Each stage can be left out
or replaced.

[Quickstart](quickstart.md) · [Inputs](inputs.md) · [Grounding](grounding.md)

## Evaluate

`odke eval` scores each stage against labels you supply. `odke bench` runs
Text2KGBench and Re-DocRED three ways: extraction alone, with grounding, and
with corroboration.

[Evaluation](evaluation.md) · [Benchmarks](benchmarks.md)

At 1.0 these two halves become the Validator and the Evaluator.

## In short

1. **Any extractor.** LangChain's `LLMGraphTransformer`, LangExtract,
   neo4j-graphrag, a graph already in Neo4j, patterns, a script of your own, or
   openodke's reference extractor.
2. **One input format.** JSON Lines triples plus the source text. Citations are
   optional.
3. **Evidence on every fact.** Each fact keeps its document, span, source tier
   and grounding verdict.
4. **Measured.** On Text2KGBench and Re-DocRED, with the same model and schema
   for every system: on documents, grounding removed 13–19% of every system's
   wrong triples, at a tenth to a fifth of what extraction cost. On single
   sentences it removed more facts the gold called right than wrong.
   Corroboration has not been measured at all, because every fact in both
   datasets has one source ([DECISIONS #24](decisions.md#24),
   [Benchmarks](benchmarks.md)).
5. **Runs offline.** `pip install openodke` on Python 3.11–3.14. The base
   install needs no key, and neither does the quickstart.

[Install](installation.md) · [Quickstart](quickstart.md) · [Concepts](concepts.md)

---

An independent implementation of the ODKE+ architecture
([arXiv:2509.04696](https://arxiv.org/abs/2509.04696)); not affiliated with or
endorsed by Apple Inc. See
[NOTICE](https://github.com/deepskandpal/odke/blob/main/NOTICE).
