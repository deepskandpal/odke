# Triples from any extractor

openodke checks facts whoever extracted them. LangChain's
`LLMGraphTransformer`, neo4j-graphrag, LangExtract, regular expressions over a
table, a script of your own: if it writes triples, write them in this format,
and grounding, resolution, corroboration and evaluation run on them unchanged.

The format is one triple per line of a JSON Lines file, next to the texts the
triples came from. It is the lowest common denominator of what extractors emit,
not a second `Fact`: four fields are required, and the rest are there when your
extractor has them.

## The row

```json
{"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in", "object": "Lyon"}
```

| Field | Required | Meaning |
|---|---|---|
| `doc` | yes | The text the triple came from: a `Document.id`, or a loaded file's name without its suffix (`texts/halden.txt` is `halden`) |
| `subject` | yes | The subject's name, as the text gives it |
| `predicate` | yes | The relation. Matched to the ontology's predicate names, ignoring case |
| `object` | yes | The object: a name, or a literal value (a string, number or boolean) |
| `subject_type`, `object_type` | no | Types, as the extractor labelled them. Matched to the ontology's types, ignoring case |
| `start`, `end` | no | Character offsets of the supporting text, half-open. Both or neither |
| `quote` | no | The supporting text. With offsets, it must be what sits at them; without, it is found in the text |
| `polarity` | no | `asserted` (the default), `denied` or `partial` |
| `qualifiers` | no | An object of qualifiers: `start_time`, a percentile, anything the ontology declares |
| `confidence` | no | Between 0 and 1. Unset, the stage's prior (0.5); the scorer calibrates it |
| `extractor` | no | Which extractor wrote the row. Unset, the stage's name for it |
| `id` | no | The fact's id, to join it to labels or to another run |

Any other field is an error, so a misspelt `objet` is caught rather than
dropped. A row that cannot be read is an error naming its line, never a row
skipped in silence.

## Offsets, a quote, or neither

What a row carries decides what the grounder is shown.

| The row has | Its evidence | What happens |
|---|---|---|
| `start` and `end` | A citation | The free span check confirms the offsets resolve (and match `quote`, if given), then the model is asked about that span |
| `quote` only | A citation, found by exact match | Found: as above. Not in the text: no span, and the free check refuses the fact as `not_found` without a model call |
| neither | The whole text, marked `context` | The model is asked about the whole text, as the paper's grounder does it |

Most extractors cite nothing, which is the third row. Their facts can still be
grounded, but the whole text is not a citation, so its span is marked
`SpanOrigin.CONTEXT` and `odke eval spans` counts it as *no span of its own*
rather than as one very wide citation. Grounding a fact against a whole
document costs more than against a sentence; the [span locator](https://github.com/deepskandpal/odke/issues/112) is what
will find the sentence for you.

## Types

With an ontology, the ontology decides:

- a predicate whose range is an entity type makes an edge, and one whose range
  is a literal makes a property, whatever `object_type` says. Graph extractors
  give literals node labels of their own (`Date`, `Number`), and the schema
  overrides them;
- a subject with no type takes the predicate's domain, when it has exactly one;
- a predicate the ontology does not have is kept as written, for the gate
  or a person to judge. It is not dropped here.

Without an ontology, a string object is an entity unless `object_type` names a
literal type (`string`, `text`, `integer`, `number`, `float`, `boolean`,
`date`, `datetime`). A number or boolean is always a literal, and anything
untyped is a `Thing`.

Entities are keyed the way openodke's own extractors key them: type and
case-folded name, so `Company:halden robotics`. Deciding that two names are one
entity is the resolver's job.

## From Python

`TriplesExtractor` is the extract stage. Everything after it runs unchanged:

```python
from openodke import Document, Ontology, Pipeline
from openodke.ground import LLMGrounder
from openodke.interop import TriplesExtractor
from openodke.llm.testing import RecordedClient

text = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. "
    "It opened a second office in Lyon in 2019."
)
doc = Document(id="halden", text=text)
rows = [
    {
        "doc": "halden",
        "subject": "Halden Robotics",
        "predicate": "office_in",
        "object": "Lyon",
        "quote": "opened a second office in Lyon",
    },
    {"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in", "object": "Berlin"},
]
ontology = Ontology.from_dict(
    {
        "name": "companies",
        "types": {"Company": {}, "City": {}},
        "predicates": {"office_in": {"domain": ["Company"], "range": "City"}},
    }
)
# Recorded answers, so this runs with no key. Drop `client=` to call a model.
client = RecordedClient(
    [
        {"match": "— Lyon (City)", "response": {"verdict": "supported"}},
        {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
    ]
)
kg = Pipeline(
    ontology,
    TriplesExtractor(rows, extractor="my-extractor", documents=[doc]),
    grounder=LLMGrounder(client=client),
).run([doc])
for fact in kg.facts:
    print(fact.object_entity.label, fact.verdict.value, fact.evidence[0].span_origin.value)
```

Pass `documents=` the same documents the pipeline runs on, so each fact's
evidence carries the document's URI and source tier. `rows` can be a path to a
JSON Lines file instead of a list. To build facts without a pipeline,
`to_fact(row, doc, ontology)` makes one, and `read_triples(path)` reads a file
into `TripleRow`s.

Triples are not chunked: a document's triples all arrive with its first chunk,
so leave the chunker out. `TriplesExtractor.stats` counts the rows by how their
evidence was made (`cited`, `quoted`, `quote_not_found`, `context`), and
`unmatched_rows` counts rows whose `doc` matched no document. Check it first
when a graph comes back smaller than the file.

A file name is not unique: `2023/report.txt` and `2024/report.txt` are both
`report`. Rows that name it go to the first of the two to arrive, the other
gets none and a warning names both, and `ambiguous_rows` counts them. Give such
texts ids and name them by id.

## From `odke run`

```yaml
ontology: ontology.json
inputs:
  - path: texts                     # texts/halden.txt is the text the rows call "halden"
    loader: {use: directory}
models:
  ground: anthropic/claude-haiku-4-5-20251001
stages:
  extractor: {use: triples, path: triples.jsonl, extractor: my-extractor}
  grounder: llm
  sink: {use: jsonl, directory: out}
```

`examples/triples/` is this config with five hand-written triples, one of each
kind: offsets, a quote, a quote that is not in the text, and two with no
citation, one of them wrong. It runs on recorded responses with no key:

```bash
odke run examples/triples/odke.yaml
odke eval spans --facts examples/triples/out
```

The run makes four model calls for five facts, because the invented quote is
refused for free. Three facts come back `supported` and two `not_found`, and
`odke eval spans` reports the two uncited facts as having no span of their own.

## From other libraries

Each adapter turns one library's output into rows and the texts they cite, and
imports nothing from the library. Hand both to `TriplesExtractor` as above.

### LangChain `GraphDocument`

`from_graph_documents` reads `LLMGraphTransformer`'s output: the objects, their
`model_dump()` dicts, or a JSON file of either (an array, or one per line).

| `GraphDocument` | Row |
|---|---|
| `source.page_content` | The text. Its id is `source.id`, or a hash of the text when that is unset; `source.metadata["source"]` is its URI |
| `relationships[].source.id`, `.target.id` | `subject`, `object` |
| `relationships[].source.type`, `.target.type` | `subject_type`, `object_type`. The untyped `Node` is left out, for the ontology to decide |
| `relationships[].type` | `predicate` |
| `relationships[].properties` | `qualifiers` |
| `nodes[].properties` | One literal row per property: the node is the `subject`, the key the `predicate` |
| no offsets, no quote | The whole text, marked `context` |

```python
from openodke.interop import from_graph_documents

company, city = {"id": "Halden Robotics", "type": "Company"}, {"id": "Lyon", "type": "City"}
office = {"source": company, "target": city, "type": "OFFICE_IN"}
# GraphDocument.model_dump(); the objects read the same way.
graph_document = {"nodes": [], "relationships": [office], "source": {"page_content": text}}
rows, texts = from_graph_documents([graph_document])
stage = TriplesExtractor(rows, extractor="langchain", documents=texts)
kg = Pipeline(ontology, stage, grounder=LLMGrounder(client=client)).run(texts)
print(kg.facts[0].verdict.value, kg.facts[0].evidence[0].span_origin.value)  # supported context
```

Every fact is grounded against its whole document. `supported` then says the
document, read whole, states the triple. It does not say where: there is no
citation for a reader to check, and a long document lets the grounder join two
passages that never state the claim together. `odke eval spans` counts these
facts as having no span of their own.

### LangExtract

`from_langextract` reads what `lx.io.save_annotated_documents` writes, or the
`AnnotatedDocument`s `lx.extract` returns. An extraction is an entity with
attributes, so the default reading makes each attribute one triple about it.

| LangExtract | Row |
|---|---|
| `document_id`, `text` | The text and its id |
| `extraction_text` | `subject`; also `quote`, unless `alignment_status` says the match was fuzzy or partial |
| `extraction_class` | `subject_type` |
| `attributes` | One row per key and value: the key is the `predicate`, the value the `object` (a list gives a row per item). A literal, unless the ontology's range for the predicate is a type |
| `char_interval.start_pos`, `.end_pos` | `start`, `end`: a citation |
| `char_interval: null` | The whole text, marked `context` |
| no `attributes` | No row: an entity states no claim. A warning counts them |
| `alignment_status`, `extraction_index`, `group_index`, `description` | Passed to `triples=`, otherwise unread |

```python
from openodke.interop import from_langextract

company = {
    "extraction_class": "company",
    "extraction_text": "Halden Robotics",
    "char_interval": {"start_pos": 0, "end_pos": 15},
    "attributes": {"office_in": "Lyon"},
}
# One line of the file, as LangExtract wrote it.
annotated = {"document_id": "halden", "text": text, "extractions": [company]}
rows, texts = from_langextract([annotated])
grounder = LLMGrounder(client=client, context="document")
kg = Pipeline(ontology, TriplesExtractor(rows, documents=texts), grounder=grounder).run(texts)
print(kg.facts[0].verdict.value, kg.facts[0].evidence[0].span.quote)  # supported Halden Robotics
```

The cited span is LangExtract's own, often just a name, and an attribute is
usually stated around it rather than in it (DECISIONS #23). So the example asks
the grounder about the whole text with `context="document"`, which still checks
the offsets for free first.

Any other reading is a function from one extraction, in LangExtract's JSON form,
to the rows it states, passed as `triples=`. The extraction's offsets and quote
are added to each row. Beside the LangExtract call that wrote the file:

<!-- docs: no-run -->
```python
def offices(extraction):  # an "office" extraction names its company and its city
    attributes = extraction["attributes"] or {}
    if extraction["extraction_class"] == "office":
        yield {
            "subject": attributes["company"],
            "predicate": "office_in",
            "object": attributes["city"],
        }


lx.io.save_annotated_documents([lx.extract(text, prompt_description=prompt, examples=examples)])
# test_output/data.jsonl is where LangExtract writes by default.
rows, texts = from_langextract("test_output/data.jsonl", triples=offices)

### neo4j-graphrag

neo4j-graphrag's extractor reads one chunk at a time, and its lexical graph
records which: `FROM_CHUNK` links each entity to the `Chunk` it came from. So a
relationship is grounded against the chunk both its ends came from, which is
what the extractor read. `from_graphrag` reads the `Neo4jGraph` the extractor
returns, or its JSON form; `read_graphrag(driver)` reads a store neo4j-graphrag
has written, in one read transaction. The example is the graph in
`Neo4jGraph.model_dump()`'s form, a chunk and two entities; the objects read the
same way.

| neo4j-graphrag | Row |
|---|---|
| entity `properties.name`, or the node's id | `subject`, `object` |
| entity `label` (in a store, its label other than `__Entity__` and `__KGBuilder__`) | `subject_type`, `object_type` |
| relationship `type` | `predicate` |
| relationship `properties` | `qualifiers` |
| other entity properties | One literal row each: the entity is the `subject`, the key the `predicate` |
| `Chunk.text` of the one chunk both ends link to | The text, marked `context`; its `Document`'s `path` is the URI |
| ends sharing no single chunk of a document | That document's text: `document=`, or its chunks in order |
| no chunk at all | Left out, and a warning counts them; in memory, `document=` instead |
| `LexicalGraphConfig` (`config=`) | The labels, relationship types and properties read |

```python
from openodke.interop import from_graphrag

chunk = {"id": "c0", "label": "Chunk", "properties": {"text": text, "index": 0}}
company = {"id": "c0:0", "label": "Company", "properties": {"name": "Halden Robotics"}}
city = {"id": "c0:1", "label": "City", "properties": {"name": "Lyon"}}
office = {"start_node_id": "c0:0", "end_node_id": "c0:1", "type": "OFFICE_IN"}
links = [{"start_node_id": n, "end_node_id": "c0", "type": "FROM_CHUNK"} for n in ("c0:0", "c0:1")]
rows, texts = from_graphrag({"nodes": [chunk, company, city], "relationships": [office, *links]})
stage = TriplesExtractor(rows, documents=texts)
kg = Pipeline(ontology, stage, grounder=LLMGrounder(client=client)).run(texts)
print(kg.facts[0].verdict.value, kg.facts[0].evidence[0].doc_id)  # supported c0
```

From a store, with the driver and the `LexicalGraphConfig` the pipeline used:

<!-- docs: no-run -->
```python
rows, texts = read_graphrag(GraphDatabase.driver(uri, auth=auth), config=LexicalGraphConfig())

### A Neo4j graph

`read_neo4j(driver)` reads a graph back so it can be grounded where it stands.
A graph `Neo4jSink` wrote carries every fact's evidence on its relationship, so
each row cites exactly what the run cited. The texts are not in the graph:
pass the documents it was built from. Any other graph names the relationship
property that holds each relationship's source text. Reading never writes.
`write_verdicts` writes the verdicts back, and only when called.

| Neo4j | Row |
|---|---|
| relationship `type` | `predicate` |
| each end: its `label` on a node the sink wrote, else `name_property` (`name`), else its `key` or element id | `subject`, `object` |
| node labels other than the sink's `Entity` | `subject_type`, `object_type` |
| a `Claim` end's `value` | `object`, a literal |
| the relationship's element id | `id`, which `write_verdicts` writes back by |
| the sink's `polarity`, `extractor`, `confidence` and qualifiers | The same fields. `qualifier_` prefixes are undone, and `odke.` stamps are read as mappings |
| each entry of `evidence_doc_ids` | One row each. Its text is the document with that id, else the one with that URI |
| `evidence_starts`, `evidence_ends` of a `cited` span | `start`, `end`. A `context` or `located` span is the whole text again |
| `text_property=` on another graph's relationship | The text, grounded whole and marked `context`; the other properties are `qualifiers` |
| none of these | Left out, and a warning counts them |

Not read back: valid times, support and verdicts, which grounding and
corroboration recompute, and a key a resolver changed. Keys are made again from
type and label, as openodke's extractors make them.

<!-- docs: no-run -->
```python
from neo4j import GraphDatabase

from openodke.interop import read_neo4j, write_verdicts

driver = GraphDatabase.driver(uri, auth=auth)
rows, texts = read_neo4j(driver, documents=corpus)  # or text_property="sentence"
stage = TriplesExtractor(rows, documents=texts)
kg = Pipeline(ontology, stage, grounder=LLMGrounder()).run(texts)
write_verdicts(driver, kg.facts)  # sets odke_verdict on each relationship read
```
