# Reference extractor

!!! note "Reference only: bug fixes, no new features"
    These extractors get bug fixes only; work that would extend them is parked
    ([DECISIONS #24](decisions.md#24)). To check another extractor's output,
    start from [Triples from any extractor](triples.md).

An extractor is anything with `extract(chunk, ontology) -> Iterable[Fact]`. The
three in `openodke.extract` key an entity by its type and folded name
(`Person:ada lovelace`), stamp each fact with its predicate's identity keys
([DECISIONS #15](decisions.md#15)), and check every span with
`Span.is_faithful`. `documents=` gives them each chunk's document; `odke run`
passes it for you.

## `PatternExtractor`

Facts from a record, a pipe table or `Key: value` lines, with no model call
and `confidence` 1.0. Field names match predicate names, labels and aliases,
ignoring case and punctuation. A row with no nameable subject yields nothing.
`mappings=[RecordMapping(...)]` maps paths to predicates by hand.

```python
from openodke import Chunk, Document, Ontology
from openodke.extract import PatternExtractor
from openodke.loaders import CsvLoader

people = Ontology.from_dict(
    {
        "name": "people",
        "types": {"Person": {"keys": ["full_name"]}, "Company": {"keys": ["legal_name"]}},
        "predicates": {
            "full_name": {"domain": ["Person"], "aliases": ["name"]},
            "born": {"domain": ["Person"], "range": "date"},
            "employer": {"domain": ["Person"], "range": "Company"},
            "legal_name": {"domain": ["Company"]},
        },
    }
)
(row,) = CsvLoader(tier="curated").load(
    b"name,born,employer\nAda Lovelace,1815-12-10,Analytical Engines Ltd\n"
)
whole = Chunk(doc_id=row.id, start=0, end=len(row.text), text=row.text, index=0)

for fact in PatternExtractor(documents=[row]).extract(whole, people):
    obj = fact.object_entity.key if fact.object_entity else fact.object_value
    print(fact.subject.key, "|", fact.predicate, "|", obj)
# Person:ada lovelace | full_name | Ada Lovelace
# Person:ada lovelace | born | 1815-12-10
# Person:ada lovelace | employer | Company:analytical engines ltd
```

## `LLMExtractor`

One model call per chunk, through the `extract` [model role](models.md). Each
fact comes back with a `quote` and the offset where the model thinks it starts.

- A quote found at that offset stays there.
- A quote found elsewhere in the passage moves to the occurrence nearest the
  claim.
- A quote not in the passage is dropped. So is a predicate not in the snippet,
  a type not in the prompt, a missing value or an unknown polarity. Each drop is
  recorded in `rejections`.

The quote asked for is the whole clause that states the fact, because the
default grounder judges a claim against its cited span. `mention` is the
narrower span inside it that tells one fact from its siblings
([DECISIONS #23](decisions.md#23)).

| Option | Default |
|---|---|
| `client`, `spec`, `roles` | the `extract` role's client |
| `types` | every entity type gets a snippet |
| `snippet_limit` | 25 predicates per snippet |
| `confidence` | 0.5 |
| `repairs` | 1 more attempt after a malformed reply |
| `structured` | `True`; `False` sends no response schema |
| `retry` | `RetryPolicy()`, for transient provider errors |
| `max_workers` | 8 calls in flight, so the client must be thread-safe |

`calls`, `rejections`, `empty_extractions` and `malformed` count what happened.

```python
from openodke.extract import LLMExtractor
from openodke.llm import ScriptedClient

note = Document(id="note", text="Grace Hopper was born in 1906. She joined the US Navy in 1943.")
passage = Chunk(doc_id="note", start=0, end=len(note.text), text=note.text, index=0)
reply = {
    "entities": [
        {
            "type": "Person",
            "name": "Grace Hopper",
            "facts": [
                {"predicate": "born", "value": "1906", "quote": "born in 1906", "start": 3},
                {"predicate": "employer", "value": "US Navy", "quote": "joined the US Navy"},
                {"predicate": "born", "value": "1907", "quote": "born in 1907"},
            ],
        }
    ]
}
extractor = LLMExtractor(client=ScriptedClient([reply]), documents=[note])
facts = extractor.extract(passage, people)

assert [(f.predicate, f.evidence[0].span.start) for f in facts] == [("born", 17), ("employer", 35)]
assert [r.reason for r in extractor.rejections] == ["quote not in the passage"]
```

### What the model is shown

One `OntologySnippet` per entity type: its predicates, ranked by `importance`
then name, cut at `snippet_limit`, rendered as prose and as JSON Schema from one
object ([DECISIONS #6](decisions.md#6)). `odke ontology snippet` prints it.

```python
print(people.snippet("Person").render())
# Entity type: Person
# Properties you may extract:
# - born (date, single)
# - employer (Company, single)
# - full_name (string, single)
```

## `HybridExtractor`

`HybridExtractor(llm, *, pattern=None, documents)` routes each chunk by its
document's modality and merges both paths' facts by signature, keeping the more
confident one. A chunk whose document it was not given raises `LookupError`.

| `modality` | Pattern | Model |
|---|---|---|
| `structured` | yes | no |
| `semi_structured` | yes | yes |
| `unstructured` | no | yes |

```python
from openodke import HybridExtractor, Pipeline

answer = {
    "entities": [
        {
            "type": "Person",
            "name": "Grace Hopper",
            "facts": [{"predicate": "born", "value": "1906", "quote": "born in 1906"}],
        }
    ]
}
documents = [row, note]
hybrid = HybridExtractor(LLMExtractor(client=ScriptedClient([answer])), documents=documents)
graph = Pipeline(people, hybrid).run(documents)

totals = hybrid.totals()
assert (totals.pattern_facts, totals.llm_facts, totals.model_calls) == (3, 1, 1)
```

## In a run config

The extractors are `pattern`, `llm` and `hybrid` in an [`odke run`](run.md)
config.

```yaml
stages:
  extractor:
    use: hybrid
    llm: {snippet_limit: 25, repairs: 1}
    pattern: {}
```
