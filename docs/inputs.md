# Inputs

openodke reads any extractor's output in one format: one triple per JSON Lines
row, next to the texts the triples came from. The [quickstart](quickstart.md)
runs it end to end, and four [adapters](#from-other-libraries) write it from
other libraries' output.

## The row

```json
{"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in", "object": "Lyon"}
```

| Field | Required | Meaning |
|---|---|---|
| `doc` | yes | The source text: a `Document.id`, or a loaded file's name without its suffix (`texts/halden.txt` is `halden`) |
| `subject` | yes | The subject's name, as the text gives it |
| `predicate` | yes | Matched to the ontology's predicates, ignoring case |
| `object` | yes | A name, or a literal: string, number or boolean |
| `subject_type`, `object_type` | no | Matched to the ontology's types, ignoring case |
| `start`, `end` | no | Half-open character offsets of the support. Both or neither |
| `quote` | no | The supporting text. With offsets, it must be the text at them |
| `polarity` | no | `asserted` (default), `denied` or `partial` |
| `qualifiers` | no | An object, such as `{"start_time": "2019"}` |
| `confidence` | no | 0 to 1. Unset: the stage's `confidence` |
| `extractor` | no | Unset: the stage's `extractor` |
| `id` | no | The fact's id, to join it to labels or another run |

Any other field is an error, so a misspelt `objet` is caught. A row that cannot
be read is an error naming its line.

## Offsets, a quote, or neither

| The row has | Its evidence | What the grounder does |
|---|---|---|
| `start` and `end` | A citation | The free span check confirms the offsets resolve and match any `quote`; the model reads that span |
| `quote` only | A citation, found by exact match | As above. A quote not in the text is refused as `not_found`, with no model call |
| neither | The whole text, with `span_origin` `context` | The model reads the whole text |

A `context` span is not a citation, and `odke eval spans` counts it as none
([DECISIONS #25](decisions.md#25)). `LLMGrounder(locate=True)` finds the sentence
naming both subject and object, with no model call, for the model to read
instead ([Grounding](grounding.md#locating-spans)).

## Types and keys { #types }

With an ontology, the predicate's range decides edge or property, whatever
`object_type` says. A subject with no type takes the predicate's domain when
there is exactly one. A predicate the ontology lacks is kept, for the gate to
judge.

Without one, a string object is an entity unless `object_type` names a literal
type (`string`, `text`, `integer`, `number`, `float`, `boolean`, `date`,
`datetime`). Numbers and booleans are literals, and an untyped entity is a
`Thing`.

An entity's key is its type and case-folded name. The
[resolver](resolution-and-corroboration.md) decides when two names are one
entity.

## From Python

`TriplesExtractor` is the extract stage:

| Argument | Default | Meaning |
|---|---|---|
| `triples` | required | A JSON Lines path, or rows (dicts or `TripleRow`s) |
| `extractor` | `"triples"` | Stamped on facts whose row names none |
| `confidence` | `0.5` | For rows that give none |
| `documents` | `()` | The pipeline's documents, so evidence carries each one's URI and tier |

- A row finds its text by `Document.id`, then by file name. When two documents
  share a file name (`2023/report.txt`, `2024/report.txt`), the first gets the
  rows and the other a warning. Give such texts ids.
- A document's triples all arrive with its first chunk, so leave the chunker
  out.
- `read_triples(path)` reads a file into `TripleRow`s;
  `to_fact(row, doc, ontology)` makes one `Fact`.

When a graph comes back smaller than the file, check `TriplesExtractor.stats`.
It counts `rows` by evidence (`cited`, `quoted`, `quote_not_found`, `context`),
`unmatched_rows` whose `doc` matched no document, and `ambiguous_rows` whose
file name matched two; `unmatched_docs` and `ambiguous_docs` name up to 20.

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
  gate: verdict
  sink: {use: jsonl, directory: out}
```

The `triples` extractor takes `path` (required), `extractor` and `confidence`;
the run's inputs are the documents ([`odke run`](run.md)). `examples/triples/`
runs this config on five rows, one of each kind, with recorded answers and no
key:

```bash
odke run examples/triples/odke.yaml
odke eval spans --facts examples/triples/out
```

`odke eval spans` then counts three of the five facts as citing no span: the two
rows with neither, and the quote that is not in the text.

To check rows without writing a config, `odke ground` takes the file, the texts
and any adapter: [Grounding](grounding.md#a-graph-from-somewhere-else-odke-ground).

## From other libraries

Each adapter returns `(rows, texts)`, the rows and the `Document`s they cite,
without importing the library it reads. The examples share this setup:

```python
from openodke import Ontology, Pipeline
from openodke.ground import LLMGrounder
from openodke.interop import TriplesExtractor
from openodke.llm.testing import RecordedClient

text = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. "
    "It opened a second office in Lyon in 2019."
)
ontology = Ontology.from_dict(
    {
        "name": "companies",
        "types": {"Company": {}, "City": {}},
        "predicates": {"office_in": {"domain": ["Company"], "range": "City"}},
    }
)
# Recorded answers, so the examples run with no key.
client = RecordedClient([{"match": "— Lyon (City)", "response": {"verdict": "supported"}}])
grounder = LLMGrounder(client=client)
```

### LangChain `GraphDocument`

`from_graph_documents` reads `LLMGraphTransformer`'s output: `GraphDocument`s,
their `model_dump()` dicts, or a JSON file of either (an array, or one per
line). It cites nothing, so every row is `context`.

| `GraphDocument` | Row |
|---|---|
| `source.page_content` | The text; its id is `source.id` or a hash of the text, and `metadata["source"]` its URI |
| `relationships[].source.id`, `.target.id` | `subject`, `object` |
| `relationships[].source.type`, `.target.type` | `subject_type`, `object_type`; the untyped `Node` is left out |
| `relationships[].type` | `predicate` |
| `relationships[].properties` | `qualifiers` |
| `nodes[].properties` | One literal row per property |

```python
from openodke.interop import from_graph_documents

company, city = {"id": "Halden Robotics", "type": "Company"}, {"id": "Lyon", "type": "City"}
office = {"source": company, "target": city, "type": "OFFICE_IN"}
graph_document = {"nodes": [], "relationships": [office], "source": {"page_content": text}}
rows, texts = from_graph_documents([graph_document])
kg = Pipeline(ontology, TriplesExtractor(rows, documents=texts), grounder=grounder).run(texts)
print(kg.facts[0].verdict.value, kg.facts[0].evidence[0].span_origin.value)  # supported context
```

### LangExtract

`from_langextract` reads the file `lx.io.save_annotated_documents` writes, the
`AnnotatedDocument`s `lx.extract` returns, or their dicts.

| LangExtract | Row |
|---|---|
| `document_id`, `text` | The text and its id |
| `extraction_text` | `subject`; also `quote`, unless `alignment_status` says the match was not exact |
| `extraction_class` | `subject_type` |
| `attributes` | One row per key and value (per item of a list): key as `predicate`, value as `object` |
| `char_interval.start_pos`, `.end_pos` | `start`, `end` |
| `char_interval: null` | The whole text, `context` |
| no `attributes` | No row; a warning counts them |

For another reading, pass `triples=` a function from one extraction, in its
JSON form, to the rows it states; the extraction's offsets and quote fill in
what it leaves out. A LangExtract span is usually just a name, so the example
grounds against the whole document:

```python
from openodke.interop import from_langextract

company = {"extraction_class": "company", "extraction_text": "Halden Robotics"}
company |= {"char_interval": {"start_pos": 0, "end_pos": 15}, "attributes": {"office_in": "Lyon"}}
rows, texts = from_langextract([{"document_id": "halden", "text": text, "extractions": [company]}])
whole = LLMGrounder(client=client, context="document")
kg = Pipeline(ontology, TriplesExtractor(rows, documents=texts), grounder=whole).run(texts)
print(kg.facts[0].verdict.value, kg.facts[0].evidence[0].span.quote)  # supported Halden Robotics
```

### neo4j-graphrag

`from_graphrag` reads the `Neo4jGraph` its extractor returns, or its JSON form.
`read_graphrag(driver, config=...)` reads a store neo4j-graphrag wrote, in one
read transaction. Each relationship is grounded against the chunk the
extractor read.

| neo4j-graphrag | Row |
|---|---|
| entity `properties.name`, else the node's id | `subject`, `object` |
| entity `label`, other than `__Entity__` and `__KGBuilder__` | `subject_type`, `object_type` |
| relationship `type` | `predicate` |
| relationship `properties` | `qualifiers` |
| other entity properties | One literal row each |
| `Chunk.text` of the chunk both ends link to by `FROM_CHUNK` | The text, `context`; its `Document`'s `path` is the URI |
| ends that share no chunk | Their document's text: `document=`, or its chunks in order |
| no chunk at all | Left out, with a warning, unless `document=` is given |
| `config=` (a `LexicalGraphConfig`) | The label and property names read |

```python
from openodke.interop import from_graphrag

chunk = {"id": "c0", "label": "Chunk", "properties": {"text": text, "index": 0}}
company = {"id": "c0:0", "label": "Company", "properties": {"name": "Halden Robotics"}}
city = {"id": "c0:1", "label": "City", "properties": {"name": "Lyon"}}
office = {"start_node_id": "c0:0", "end_node_id": "c0:1", "type": "OFFICE_IN"}
links = [{"start_node_id": n, "end_node_id": "c0", "type": "FROM_CHUNK"} for n in ("c0:0", "c0:1")]
rows, texts = from_graphrag({"nodes": [chunk, company, city], "relationships": [office, *links]})
kg = Pipeline(ontology, TriplesExtractor(rows, documents=texts), grounder=grounder).run(texts)
print(kg.facts[0].verdict.value, kg.facts[0].evidence[0].doc_id)  # supported c0
```

### A Neo4j graph

`read_neo4j(driver)` reads a graph back to ground it where it stands. On a
graph `Neo4jSink` wrote, each row cites what the run cited; pass `documents=`
the texts it was built from. On any other graph, `text_property=` names the
relationship property holding the source text. Reading never writes;
`write_verdicts` does, when called.

| Neo4j | Row |
|---|---|
| relationship `type` | `predicate` |
| each end's sink `label`, else `name_property` (`name`), else `key` or element id | `subject`, `object` |
| node labels other than the sink's `Entity` | `subject_type`, `object_type` |
| a `Claim` end's `value` | `object`, a literal |
| the relationship's element id | `id`, which `write_verdicts` writes back by |
| the sink's `polarity`, `extractor`, `confidence` and qualifiers | The same fields |
| each `evidence_doc_ids` entry | One row, on the document with that id or URI |
| `evidence_starts`, `evidence_ends` of a `cited` span | `start`, `end`; other spans are the whole text again |
| `text_property=` | The text, `context`; other properties are `qualifiers` |
| none of these | Left out, with a warning |

Valid times, support, verdicts and resolved keys are not read back.

<!-- docs: no-run -->
```python
from neo4j import GraphDatabase

from openodke.interop import read_neo4j, write_verdicts

driver = GraphDatabase.driver(uri, auth=auth)
rows, texts = read_neo4j(driver, documents=corpus)  # or text_property="sentence"
kg = Pipeline(ontology, TriplesExtractor(rows, documents=texts), grounder=LLMGrounder()).run(texts)
write_verdicts(driver, kg.facts)  # sets odke_verdict on each relationship read
```
