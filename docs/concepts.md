# Concepts

Everything on this page is importable from `openodke` and runs offline. The
data types are frozen pydantic models: a stage returns new objects rather than
editing one ([DECISIONS #4](decisions.md#4)).

## `Fact`: an edge or a property

A `Fact` is one (subject, predicate, object) claim with its evidence.
`object_entity`, another `Entity`, makes it an **edge**; `object_value`, a
literal, makes it a **property**. Set exactly one
([DECISIONS #2](decisions.md#2)).

| Field | Holds |
|---|---|
| `subject`, `predicate`, `object_entity` / `object_value` | The claim |
| `polarity` | `asserted`, `denied` or `partial` |
| `qualifiers`, `identity_keys` | Modifiers, and which of them bear identity |
| `valid_from`, `valid_to` | When the claim was true in the world |
| `evidence` | A tuple of `Evidence` |
| `extractor`, `confidence` | Who extracted it; the scorer's number |
| `verdict`, `support` | The grounder's verdict; how many sources agree |
| `id` | A fresh id per extraction. Identity is `signature` |

## `Evidence` { #spans-evidence-and-trust }

| Field | Holds |
|---|---|
| `doc_id`, `uri`, `tier` | The document, its URI and its `SourceTier` |
| `span` | A `Span`: half-open offsets `[start, end)` into the text, and an optional `quote` |
| `mention` | Optional: a narrower span inside `span`, for highlighting. No stage reads it |
| `span_origin` | Who chose `span`: `cited`, the extractor (the default); `located`, the span locator; `context`, nobody: it is the whole text |
| `retrieved_at` | When the text was read |

`Span.resolve(doc)` returns the text at the offsets, and `Span.is_faithful(doc)`
checks the `quote` against it, for free ([DECISIONS #3](decisions.md#3)). By
default `LLMGrounder` reads the span alone, so a `span` should be the clause
that states the claim ([DECISIONS #23](decisions.md#23)). A `context` span is not
a citation ([DECISIONS #25](decisions.md#25)).

```python
from openodke import Document, Evidence, Span, SpanOrigin

doc = Document(id="d1", text="Acme was founded in 1999. It sells anvils.")
start = doc.text.index("anvils")
span = Span(doc_id="d1", start=start, end=start + 6, quote="anvils")

assert span.resolve(doc) == "anvils" and span.is_faithful(doc)
assert not Span(doc_id="d1", start=0, end=4, quote="Acme Corp").is_faithful(doc)
assert Evidence(doc_id="d1", span=span).span_origin is SpanOrigin.CITED
```

| `SourceTier` | `weight` |
|---|---|
| `curated` | 1.0 |
| `authoritative` | 0.8 |
| `community` | 0.5 |
| `unverified` (the default) | 0.2 |

## Verdicts

The grounder stamps a `GroundingVerdict` on each fact and never drops one
([DECISIONS #20](decisions.md#20)):

| Verdict | Means |
|---|---|
| `unchecked` | No model answered |
| `supported` | The text states or entails the claim |
| `contradicted` | The text says otherwise |
| `not_found` | The text does not settle it, or the cited span does not resolve |

The gate decides what is written. Its `ValidationVerdict` is `accept` or
`conflict`, both written, or `refuse`, dropped and counted. `VerdictGate`
refuses `contradicted`, and `not_found` too with `refuse_not_found=True`
([Grounding](grounding.md)).

## The signature { #the-signature-and-what-is-in-it }

`Fact.signature` is what two extractions must share to be one claim. The
corroborator merges on it, the Neo4j sink `MERGE`s on it, and the extraction
evaluator matches on it.

| In the signature | Not in it |
|---|---|
| The subject's key and type | `id`, `evidence`, `extractor`, `confidence`, `verdict`, `support` |
| The predicate | `valid_from`, `valid_to` ([#17](decisions.md#17)) |
| The object's key, or the literal | Reconcilable qualifiers, such as `start_time` ([#11](decisions.md#11)) |
| `polarity` ([#14](decisions.md#14)) | |
| Identity-bearing qualifiers, sorted ([#15](decisions.md#15)) | |

The ontology marks a qualifier with `Qualifier(identity=True)`, and the
extractor stamps its name onto `Fact.identity_keys`
([Ontology](ontology.md#cardinality-and-its-scope)).

```python
from openodke import Entity, Fact, Polarity

acme = Entity(key="c:acme", type="Company", label="Acme")
p50 = Fact(
    subject=acme,
    predicate="uptime",
    object_value="99.9%",
    qualifiers={"percentile": "p50", "as_of": "2026-08"},
    identity_keys=("percentile",),
)
p95 = p50.model_copy(update={"qualifiers": {"percentile": "p95", "as_of": "2026-08"}})
restated = p50.model_copy(update={"qualifiers": {"percentile": "p50", "as_of": "2026-09"}})
denied = p50.model_copy(update={"polarity": Polarity.DENIED})

assert p50.signature != p95.signature  # identity-bearing: two measurements
assert p50.signature == restated.signature  # reconcilable: one claim, told twice
assert p50.signature != denied.signature  # a denial is the opposite claim

print(p50.signature)
# ('c:acme', 'Company', 'uptime', "'99.9%'", 'asserted', (('percentile', "'p50'"),))
```

## Entities and links

An `Entity` is a node: a `key`, which facts merge on, a `type`, a `label`,
`aliases`, `external_id`, `attributes`, and a `resolution` saying how the key
was decided: `caller`, `external_id`, or `linker` with a `score`
([DECISIONS #18](decisions.md#18)).

A resolver proposes `EntityLink`s between keys. It never merges or replaces
nodes ([DECISIONS #16](decisions.md#16)).

| `LinkKind` | Means |
|---|---|
| `SAME_AS` | One entity |
| `SIMILAR` | Alike, not proven |
| `DIFFERENT` | A strong identifier disagrees, however alike the names; `reason` names it |

```python
from openodke import EntityLink, KnowledgeGraph, LinkKind

acme_gmbh = Entity(key="c:acme-gmbh", type="Company", label="Acme GmbH", external_id="DE-114322")
different = EntityLink(
    source_key="c:acme",
    target_key="c:acme-gmbh",
    kind=LinkKind.DIFFERENT,
    reason="external_id mismatch: DE-114322 vs GB-889401",
)
graph = KnowledgeGraph(entities=(acme, acme_gmbh), facts=(p50,), links=(different,))
assert (len(graph.edges), len(graph.properties), len(graph.links)) == (0, 1, 1)
```

`KnowledgeGraph` is the pipeline's output, and all a sink reads: entities,
facts, links and `stats`.

## The order the stages run in

```text
Pipeline(ontology, extractor, *, retriever=, chunker=, router=, grounder=,
         normalizer=, resolver=, corroborator=, scorer=, gate=, constrainer=,
         inverses=, sinks=, coverage=, reextract=)
```

A stage left as `None` is its pass-through. `openodke.Validator` runs this
pipeline over triples, with a default for every stage after extraction
([The Validator](validator.md)). `run(docs)`:

1. **Chunk, route, extract**, per chunk; or triples in, through
   [`TriplesExtractor`](inputs.md).
2. **Ground**, per document: the free span check, then the model. The span
   locator (`locate=True`) and widen-and-retry (`widen=True`) are optional.
3. **Coverage** (`coverage=True`) and **re-extract**
   (`reextract=Reextract()`), both optional: what extraction left behind, handed
   back to the extractor.
4. **Normalise.**
5. **Resolve**, over the batch. It comes before corroboration because the
   signature merges on the subject's key.
6. **Inverse partners**, when the ontology declares them.
7. **Corroborate**, then **score**.
8. **Gate.**
9. **Write** to every sink.

A stage with `extract_many`, `ground_documents` or `ground_many` gets the whole
run at once and may call its model concurrently, with the same result.
`Pipeline.constraints()` returns the DDL for a sink to apply; `run()` does not.
Chunking and routing are on [Loading documents](loading.md#chunking).

| `kg.stats` | Counts |
|---|---|
| `documents`, `chunks` | What went in |
| `skipped`, `deferred` | Chunks the router held back |
| `empty_extractions` | Chunks that yielded no fact |
| `derived` | Inverse partners added |
| `refused` | Facts the gate refused |
| `coverage`, `reextract` | With those options on |

### Inverse and symmetric partners

When the ontology declares an
[inverse or a symmetric predicate](ontology.md#inverse-and-symmetric-predicates),
each edge on it gains its partner, the same claim the other way round
([DECISIONS #28](decisions.md#28)). `inverses=False` turns it off.

- The partner cites its source's evidence and keeps its verdict. It is never
  grounded again, and a gate that refuses the source on its verdict refuses
  the partner too.
- It carries `qualifiers["odke.derived"]`: the rule, and the source's
  signature, which `openodke.corroborate.derived_from(fact)` reads back.
- A partner the batch already states is not derived, and a derived fact gets
  no partner.

```python
from openodke import Chunk, Ontology, Pipeline

geo = Ontology.from_dict(
    {
        "types": {"Place": {}},
        "predicates": {
            "located_in": {"domain": ["Place"], "range": "Place", "inverse_of": "contains"},
            "contains": {"domain": ["Place"], "range": "Place"},
        },
    }
)


class SaysLocated:
    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        brittany = Entity(key="brittany", type="Place", label="Brittany")
        france = Entity(key="france", type="Place", label="France")
        receipt = (Evidence(doc_id=chunk.doc_id),)
        return [
            Fact(subject=brittany, predicate="located_in", object_entity=france, evidence=receipt)
        ]


kg = Pipeline(geo, SaysLocated()).run([Document(id="geo", text="Brittany is in France.")])
print([(f.subject.label, f.predicate, f.object_entity.label) for f in kg.facts])
# [('Brittany', 'located_in', 'France'), ('France', 'contains', 'Brittany')]
print(kg.stats)
# {'documents': 1, 'chunks': 1, 'skipped': 0, 'deferred': 0, 'empty_extractions': 0, 'derived': 1, 'refused': 0}
```

## The Protocols { #the-thirteen-protocols }

Every stage is a `typing.Protocol` in `openodke.stages`: any object with the
right method is a stage ([DECISIONS #5](decisions.md#5)). A pass-through returns
its input unchanged unless the table says otherwise.

| # | Protocol and method | Pass-through default |
|---|---|---|
| 1 | `Loader.load(source) -> Iterable[Document]` | `PassThroughLoader`: documents as given |
| 2 | `Chunker.chunk(doc) -> Iterable[Chunk]` | `PassThroughChunker`: the whole document |
| 3 | `Router.route(chunk) -> RouteVerdict` | `PassThroughRouter`: `extract` |
| 4 | `Extractor.extract(chunk, ontology) -> Iterable[Fact]` | None: required |
| 5 | `Grounder.ground(fact, doc) -> Fact` | `PassThroughGrounder`: `unchecked` |
| 6 | `Normalizer.normalize(fact) -> Fact` | `PassThroughNormalizer` |
| 7 | `Resolver.resolve(facts, index) -> (facts, links)` | `PassThroughResolver`: keys as given |
| 8 | `Corroborator.corroborate(facts) -> Iterable[Fact]` | `PassThroughCorroborator`: `support` 1 |
| 9 | `Scorer.score(fact) -> Fact` | `PassThroughScorer` |
| 10 | `Gate.validate(fact, ontology) -> ValidationVerdict` | `PassThroughGate`: `accept` |
| 11 | `Sink.write(kg) -> None` | None: `sinks=()` writes nowhere |
| 12 | `Constrainer.constrain(ontology) -> DDL` | `PassThroughConstrainer`: no statements |
| 13 | `Inferrer.infer(corpus) -> Ontology` | `PassThroughInferrer`: an empty `Ontology()` |

`Initiator` and `Retriever`, the paper's two front stages, are declared too,
and optional ([DECISIONS #9](decisions.md#9)).

## `PlatformProfile` and `Delegated`

Some stores do a stage themselves: neo4j-graphrag resolves after the write. A
sink says so with `profile = PlatformProfile(name, resolves=, constrains=,
prunes=)`. If openodke also runs that stage, `Pipeline` warns once with a
`DoubleStageWarning` and runs both. `Delegated(to="...")` hands the stage to
the platform: a pass-through that records `to` in `Entity.resolution.linker`
for a resolver, or in the verdict's `reason` for a router or gate
([DECISIONS #21](decisions.md#21)).

```python
import warnings

from openodke import Delegated, DoubleStageWarning, PlatformProfile, Resolution


class GraphRagSink:
    profile = PlatformProfile(name="neo4j-graphrag", resolves=True)

    def write(self, kg):
        self.written = kg


class ExactKeyResolver:
    def resolve(self, facts, index):
        return facts, ()


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Pipeline(geo, SaysLocated(), resolver=ExactKeyResolver(), sinks=[GraphRagSink()])
assert [w.category for w in caught] == [DoubleStageWarning]

resolver = Delegated(to="neo4j-graphrag:FuzzyMatchResolver")
kg = Pipeline(geo, SaysLocated(), resolver=resolver, sinks=[GraphRagSink()]).run(
    [Document(id="geo", text="Brittany is in France.")]
)
assert kg.facts[0].subject.resolution == Resolution(
    method="linker", linker="neo4j-graphrag:FuzzyMatchResolver"
)
```
