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
- a predicate the ontology does not have is kept as written, for a validator
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
