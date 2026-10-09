# Quickstart: verify any extractor's triples

This page takes three triples another extractor wrote, checks each against its
source text, and keeps the ones the text supports. It needs no key and no
network: recorded answers stand in for the model.

## 1. The text and the triples

Each triple is a row: the text it came from (`doc`), a subject, a predicate and
an object. A row may cite its support with offsets or a `quote`, or cite
nothing. [Inputs](inputs.md#the-row) has every field.

```python
from openodke import Document, Ontology, Pipeline, VerdictGate
from openodke.ground import LLMGrounder
from openodke.interop import TriplesExtractor
from openodke.llm.testing import RecordedClient

text = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. "
    "It opened a second office in Lyon in 2019."
)
doc = Document(id="halden", text=text)
company = {"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in"}
rows = [
    {**company, "object": "Lyon", "quote": "opened a second office in Lyon"},
    {**company, "object": "Berlin"},
    {**company, "object": "Paris", "quote": "an office in Paris"},
]
ontology = Ontology.from_dict(
    {
        "name": "companies",
        "types": {"Company": {}, "City": {}},
        "predicates": {"office_in": {"domain": ["Company"], "range": "City"}},
    }
)
```

The first row quotes its support, the second cites nothing, and the third
quotes text that is not in the document. On a real run, `rows` is the path to a
JSON Lines file with one row per line.

## 2. Ground them

`TriplesExtractor` is the extract stage: it replays the rows as facts.
`LLMGrounder` checks each cited span against the document for free, then asks a
model whether the text supports the claim.

```python
# Recorded answers, so this runs with no key. Drop `client=` to call a model.
client = RecordedClient(
    [
        {"match": "— Lyon (City)", "response": {"verdict": "supported"}},
        {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
    ]
)
grounder = LLMGrounder(client=client)
stage = TriplesExtractor(rows, documents=[doc])
kg = Pipeline(ontology, stage, grounder=grounder).run([doc])

for fact in kg.facts:
    print(fact.object_entity.label, fact.verdict.value, fact.evidence[0].span_origin.value)
# Lyon supported cited
# Berlin not_found context
# Paris not_found cited
assert grounder.stats["calls"] == 2
```

Three facts cost two model calls: the free check refused the quote that is not
in the text. The row with no citation was checked against the whole text, its
`context`.

## 3. Keep what holds up

The grounder only stamps verdicts; a gate decides what is written.
`VerdictGate` refuses `contradicted` facts, and with `refuse_not_found=True`
refuses `not_found` ones too.

```python
gate = VerdictGate(refuse_not_found=True)
stage = TriplesExtractor(rows, documents=[doc])
kg = Pipeline(ontology, stage, grounder=grounder, gate=gate).run([doc])
print([fact.object_entity.label for fact in kg.facts], gate.stats)
# ['Lyon'] {'accepted': 1, 'refused': {'not_found': 2}}
```

`openodke.Validator` does all of this in one call, with resolution and
corroboration added and a default for every stage
([The Validator](validator.md)):

```python
from openodke import Validator

kg, report = Validator(ontology, client=client).validate(rows, [doc])
print(report.facts_in, report.verdicts, report.calls)
# 3 {'supported': 1, 'contradicted': 0, 'not_found': 2, 'unchecked': 0} 2
```

Pass `sinks=` to either to write what is kept:
[Write to a store](stores.md#choose-a-sink) lists the sinks.

To call a real model, drop `client=` and set the provider's key:
[Models and providers](models.md).

## Your extractor

Each adapter below turns one library's output into rows and texts, without
importing the library. Hand both to `TriplesExtractor` or `Validator.validate`
as above.

**LangChain.** `LLMGraphTransformer` returns `GraphDocument`s.
`from_graph_documents` reads them, or a JSON file of them. They cite nothing, so
each fact is checked against its whole document.
[Inputs: LangChain](inputs.md#langchain-graphdocument).

**LangExtract.** `from_langextract` reads the file
`lx.io.save_annotated_documents` writes, or the documents `lx.extract` returns.
Each extraction's character interval becomes a citation.
[Inputs: LangExtract](inputs.md#langextract).

**neo4j-graphrag.** `from_graphrag` reads the graph its extractor returns, and
`read_graphrag` reads a store it wrote. Each relationship is checked against the
chunk its two ends came from.
[Inputs: neo4j-graphrag](inputs.md#neo4j-graphrag).

**A Neo4j graph.** `read_neo4j` reads a graph back so it can be checked where it
stands, and `write_verdicts` writes the verdicts onto its relationships.
[Inputs: a Neo4j graph](inputs.md#a-neo4j-graph).

**Anything else.** Write the rows yourself, one JSON object per line:
[the row](inputs.md#the-row).

## From the command line

`odke validate` is the Validator as a command. On the five rows in
`examples/triples/`, with recorded answers and no key:

```bash
odke validate --facts examples/triples/triples.jsonl --texts examples/triples/texts \
  --ontology examples/triples/ontology.json --config examples/triples/odke.yaml -o out
```

It prints what each step did and writes the graph to `out/` as JSONL.

| Flag | Does |
|---|---|
| `--dry-run` | Runs the free checks and the deterministic stages only: no model, no write |
| `--adapter` | Reads `langchain`, `langextract`, `graphrag` or `neo4j` output instead of `triples` |
| `--config` | With `--facts`, supplies only the `models` block: here, the recorded answers |

`odke ground` runs the grounding step alone
([Grounding](grounding.md#a-graph-from-somewhere-else-odke-ground)), and
`odke run` runs a whole config ([Inputs](inputs.md#from-odke-run)).

## Next

- [Concepts](concepts.md): facts, evidence, verdicts and the order the stages run in.
- [Grounding](grounding.md): the span check, the model check, paper mode, the
  span locator, and the gate.
- [Evaluation](evaluation.md): whether grounding helps on your data.
