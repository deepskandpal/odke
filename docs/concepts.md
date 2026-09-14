# Concepts

Everything in this page lives in `odke.types`, `odke.stages` and
`odke.pipeline`, and is importable from `odke`. All of it is deterministic and
provider-free, so `import odke` works with no model provider, no database driver
and no network.

## Everything is frozen

`Document`, `Span`, `Evidence`, `Entity`, `Fact`, `EntityLink` and
`KnowledgeGraph` are frozen pydantic models with `extra="forbid"`. Facts pass
through several stages and get merged across sources, so a stage that edited one
in place would make its own provenance wrong. Stages return new objects
(`model_copy(update=...)`) instead ([DECISIONS #4](decisions.md)).

## `Fact`: one class for edges and properties

A `Fact` is one (subject, predicate, object) claim with its receipts. The object
is either `object_entity` (another `Entity`, which makes the fact an **edge**) or
`object_value` (a literal, which makes it a **property**). Set one or the other,
never both. They share one class because both kinds need identical provenance,
grounding and corroboration, and when they were split every stage had two code
paths that drifted ([DECISIONS #2](decisions.md)).

| Field | What it holds |
|---|---|
| `id` | A fresh hex id per extraction. It is not identity; `signature` is. |
| `subject`, `predicate`, `object_entity` / `object_value` | The claim. |
| `polarity` | `asserted`, `denied` or `partial`. |
| `qualifiers`, `identity_keys` | Predicate-scoped modifiers, and which of them bear identity. |
| `valid_from`, `valid_to` | The valid clock: when the claim was true in the world. |
| `evidence` | `Evidence` tuples: document, optional `Span`, uri, `SourceTier`, `retrieved_at`. |
| `extractor`, `confidence`, `verdict`, `support` | Who produced it, the scorer's number, the grounder's verdict, and how many independent sources agreed. |

### The signature, and what is in it

`Fact.signature` is the identity two extractions must share to count as the same
claim: subject key, subject type, predicate, object, polarity, and the
identity-bearing qualifiers, sorted. The corroborator merges on it, the Neo4j
sink `MERGE`s on it, and the extraction evaluator matches on it.

```python
from datetime import UTC, datetime

from odke import Entity, Evidence, Fact, Polarity, SourceTier

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

assert p50.signature != p95.signature  # identity-bearing qualifier: two measurements
assert p50.signature == restated.signature  # reconcilable qualifier: one claim, told twice
assert p50.signature != denied.signature  # a denial is the opposite claim

print(p50.signature)
# ('c:acme', 'Company', 'uptime', "'99.9%'", 'asserted', (('percentile', "'p50'"),))
```

**Polarity is in the signature.** Before it was a field, "X sells customer data"
and "X does not sell customer data" shared a signature, so the corroborator merged
them and each raised the other's `support`. A denial is not the same claim as an
assertion; it is the opposite one ([DECISIONS #14](decisions.md)).

**Reconcilable qualifiers are not.** "CEO since 2019" and "CEO 2019–2024" are one
claim told two ways, and if qualifiers were identity both would land in the graph
as separate edges ([DECISIONS #11](decisions.md)).

**Identity-bearing qualifiers are.** "Uptime 99.9% at p50" and "at p95" are two
measurements, and merging them inflates `support`. The ontology declares which
keys bear identity with `Qualifier(identity=True)`. The extractor stamps
`Ontology.identity_keys(predicate)` onto `Fact.identity_keys`, and the signature
includes those keys. The fact carries the key names, not a reference to the
ontology, so that a serialised fact still means the same thing after its schema
has been edited. An extractor that forgets to stamp them gets the reconcilable
behaviour, which is the safe direction to fail in
([DECISIONS #15](decisions.md)). Declaring qualifiers is covered in
[Ontology](ontology.md#cardinality-and-its-scope).

```python
try:
    p50.confidence = 1.0
except Exception as exc:  # frozen: stages return new objects instead
    print(type(exc).__name__)
# ValidationError
```

## Two clocks

`Fact.valid_from` / `valid_to` is the **valid** clock: when the claim was true in
the world. `Evidence.retrieved_at` and `Document.retrieved_at` are the
**transaction** clock: when we came to believe it. With only one clock, a CEO who
changed and two sources that disagree look like the same event, and they need
opposite handling: one is a fact that expired correctly, the other is a conflict
to resolve ([DECISIONS #17](decisions.md)).

Neither clock is in the signature, so an interval is something the corroborator
reconciles and never a reason for two edges.

```python
jane = Entity(key="p:jane", type="Person", label="Jane Doe")
since_2019 = Fact(
    subject=jane,
    predicate="ceo_of",
    object_entity=acme,
    valid_from=datetime(2019, 1, 1, tzinfo=UTC),
    evidence=(
        Evidence(
            doc_id="annual-report-2020",
            tier=SourceTier.CURATED,
            retrieved_at=datetime(2020, 3, 1, tzinfo=UTC),
        ),
    ),
)
ended_2024 = since_2019.model_copy(update={"valid_to": datetime(2024, 6, 30, tzinfo=UTC)})

assert since_2019.is_edge
assert since_2019.signature == ended_2024.signature
```

## Spans, evidence and trust

A `Span` is a half-open character range, `[start, end)`, into `Document.text`,
with an optional `quote`. Spans are offsets, not quoted strings. A model can
produce a quote that reads perfectly and appears nowhere in the source, but an
offset either resolves to the claimed text or it does not, and checking costs
nothing ([DECISIONS #3](decisions.md)). `Span.resolve(doc)` returns the text at
the offsets; `Span.is_faithful(doc)` is true when the recorded quote is what sits
there.

`Evidence` is a document id plus an optional span, uri, `SourceTier` and
`retrieved_at`. Tiers are four named levels with fixed weights, because callers
reason about "curated vs. scraped", not about 0.8
([DECISIONS #10](decisions.md)):

| `SourceTier` | `weight` |
|---|---|
| `curated` | 1.0 |
| `authoritative` | 0.8 |
| `community` | 0.5 |
| `unverified` (the default) | 0.2 |

## `Chunk`, and routing

The chunk is the unit of the pipeline ([DECISIONS #19](decisions.md)). A `Chunk`
is `doc_id`, `start`, `end`, `text` and `index`, and `text` is exactly
`doc.text[start:end]`. An extractor that finds evidence at a local offset
therefore cites `chunk.start + offset` in the document, and provenance survives
chunking without a lookup.

```python
from odke import Chunk, Document, Span

doc = Document(id="d1", text="Acme was founded in 1999. It sells anvils.")
chunk = Chunk(doc_id="d1", start=26, end=42, text=doc.text[26:42], index=1)

local = chunk.text.index("anvils")
span = Span(doc_id="d1", start=chunk.start + local, end=chunk.start + local + 6, quote="anvils")

assert span.resolve(doc) == "anvils" and span.is_faithful(doc)
assert not Span(doc_id="d1", start=0, end=4, quote="Acme Corp").is_faithful(doc)
```

A `Router` sees each chunk before extraction and returns a `RouteVerdict`:

- `action`: `extract`, `skip` or `defer`.
- `label`: your own word for the chunk's kind. It is a free string, and the
  package ships no taxonomy.
- `scope`: `chunk` or `document`. A document-scoped verdict stops that document:
  a router that recognises a marketing page from its first chunk can skip the
  rest of it without a second method.
- `reason`: optional text saying why.

```python
import re

from odke import Chunker, Ontology, Pipeline, RouteVerdict


class Sentences:
    """A chunker: one chunk per sentence, offsets into the document kept."""

    def chunk(self, doc):
        for index, m in enumerate(re.finditer(r"\S[^.]*\.", doc.text)):
            yield Chunk(doc_id=doc.id, start=m.start(), end=m.end(), text=m[0], index=index)


class SkipMarketing:
    def route(self, chunk):
        if "award-winning" in chunk.text:
            return RouteVerdict(action="skip", label="marketing", scope="document")
        return RouteVerdict(action="extract", label="fact")


class Seen:
    """Stands in for an extractor, and remembers what reached it."""

    def __init__(self):
        self.chunks = []

    def extract(self, chunk, ontology):
        self.chunks.append(chunk.text)
        return ()


seen = Seen()
docs = [
    Document(id="about", text="Acme was founded in 1999. It sells anvils."),
    Document(id="promo", text="Our award-winning anvils delight. Buy now. Call us."),
]
kg = Pipeline(Ontology(), seen, chunker=Sentences(), router=SkipMarketing()).run(docs)

assert isinstance(Sentences(), Chunker)  # a Protocol: no base class, no registration
assert seen.chunks == ["Acme was founded in 1999.", "It sells anvils."]
print(kg.stats)
# {'documents': 2, 'chunks': 3, 'skipped': 1, 'deferred': 0, 'refused': 0}
```

## Entities, resolution and `EntityLink`

An `Entity` is a node: `key` (the identity facts merge on), `type`, `label`,
`aliases`, `external_id`, `attributes`, and `resolution`. That last field is a
`Resolution(method, score, linker)` recording how the key was decided
([DECISIONS #18](decisions.md)):

- `caller`: the key arrived with the input.
- `external_id`: a global identifier settled it, exactly, with no score.
- `linker`: something compared and decided. `score` says how sure it was, and
  `linker` names it, including a platform's resolver when the pass was delegated.

Without that field a wrong link is invisible. With it, a wrong link is a query:
every entity whose identity came from a match below 0.9.

**Resolution proposes links; it never replaces nodes.** A resolver returns
`EntityLink(source_key, target_key, kind, score, evidence, reason)` with `kind`
one of `SAME_AS`, `SIMILAR` or `DIFFERENT`. A merge that replaces nodes cannot be
undone when the threshold turns out to be wrong. A link can be lowered, raised or
ignored, and the evidence it was made on is still there. `DIFFERENT` is the kind
other tools do not record. It is where the disagreement rule lives: a strong
identifier that disagrees kills a match however similar the names are, and
`reason` names that identifier, so the rejection can be queried
([DECISIONS #16](decisions.md)).

```python
from odke import EntityLink, KnowledgeGraph, LinkKind, Resolution

acme_gmbh = Entity(
    key="c:acme-gmbh",
    type="Company",
    label="Acme GmbH",
    external_id="DE-114322",
    resolution=Resolution(method="external_id"),
)
links = (
    EntityLink(source_key="c:acme", target_key="c:acme-gmbh", kind=LinkKind.SIMILAR, score=0.91),
    EntityLink(
        source_key="c:acme",
        target_key="c:acme-gmbh",
        kind=LinkKind.DIFFERENT,
        reason="external_id mismatch: DE-114322 vs GB-889401",
    ),
)
graph = KnowledgeGraph(entities=(acme, acme_gmbh), facts=(p50, since_2019), links=links)

assert (len(graph), len(graph.edges), len(graph.properties), len(graph.links)) == (2, 1, 1, 2)
```

`KnowledgeGraph` is the pipeline's output and the only thing a sink has to
understand: `entities`, `facts`, `links`, `ontology_name`, `created_at` and
`stats`. It is storage-neutral. A sink with no use for links ignores them.

## The thirteen Protocols

Every stage is a `typing.Protocol` in `odke.stages`. Any object with the right
method is a stage: nothing to import from odke, no registration, no inheritance
([DECISIONS #5](decisions.md)). Each stage has a pass-through default, and those
defaults are not stubs. A pipeline built from them is a working pipeline that
routes nothing out, grounds nothing, resolves nothing and accepts everything,
which is the right shape for a caller who wants recall and will filter later.

| # | Stage | Protocol and method | Pass-through default |
|---|---|---|---|
| 1 | load | `Loader.load(source) -> Iterable[Document]` | `PassThroughLoader`: hands documents on; a bare string becomes one. Reads no files. |
| 2 | chunk | `Chunker.chunk(doc) -> Iterable[Chunk]` | `PassThroughChunker`: one chunk, the whole document. |
| 3 | route | `Router.route(chunk) -> RouteVerdict` | `PassThroughRouter`: `extract`, for every chunk. |
| 4 | extract | `Extractor.extract(chunk, ontology) -> Iterable[Fact]` | **None.** The one stage with no identity function. |
| 5 | ground | `Grounder.ground(fact, doc) -> Fact` | `PassThroughGrounder`: the verdict stays `UNCHECKED`. |
| 6 | normalise | `Normalizer.normalize(fact) -> Fact` | `PassThroughNormalizer`: unchanged. |
| 7 | resolve | `Resolver.resolve(facts, index) -> (facts, links)` | `PassThroughResolver`: keys as given, no links. |
| 8 | corroborate | `Corroborator.corroborate(facts) -> Iterable[Fact]` | `PassThroughCorroborator`: every fact its own claim; `support` stays 1. |
| 9 | score | `Scorer.score(fact) -> Fact` | `PassThroughScorer`: unchanged. |
| 10 | validate | `Validator.validate(fact, ontology) -> ValidationVerdict` | `PassThroughValidator`: `accept`. |
| 11 | sink | `Sink.write(kg) -> None` | None needed: `sinks=()` writes nowhere. |
| 12 | constrain | `Constrainer.constrain(ontology) -> DDL` | `PassThroughConstrainer`: no statements. |
| 13 | infer | `Inferrer.infer(corpus) -> Ontology` | `PassThroughInferrer`: an empty `Ontology()`. |

The paper's two front stages, `Initiator` (what needs refreshing) and
`Retriever` (fetch it), are declared too, but they are optional and not among the
thirteen. An SDK is usually handed its documents ([DECISIONS #9](decisions.md)).

### How `Pipeline` runs them

`Pipeline(ontology, extractor, *, chunker=, router=, grounder=, normalizer=,
resolver=, corroborator=, scorer=, validator=, constrainer=, sinks=)` takes the
extractor and the stages you have opinions about. Every stage you leave as
`None` is its pass-through. `run(docs)` takes `Document`s (a `Loader` is how you
get them) and does this, in order:

1. **Per chunk:** route, then extract. A chunk routed `skip` or `defer` is counted
   in `stats` and never extracted.
2. **Per document:** ground, then normalise. If the grounder also has
   `ground_many(facts, doc)`, it gets the whole document's facts at once and may
   run its model calls concurrently. Either way one fact must come back for each
   fact that went in: a grounder stamps a verdict, it never drops a fact.
3. **Over the batch:** resolve, then corroborate, score and validate. Resolution
   runs before corroboration on purpose: `Fact.signature` merges on
   `subject.key`, so corroboration cannot repair a resolution failure. A fact the
   validator `refuse`s is dropped and counted; `accept` and `conflict` are
   written.
4. Build the `KnowledgeGraph` and hand it to every sink.

`Pipeline.constraints()` returns the constrainer's DDL for a sink to apply before
its first write. `run()` does not apply it, because which sink applies it is the
caller's business.

## `PlatformProfile` and `Delegated`

Some stores already do a stage themselves: neo4j-graphrag resolves after the
write, GraphPruner prunes, and an RDF store refuses what breaks SHACL. Running
that stage twice is waste at best. Forbidding it would also be wrong, because the
two passes differ: odke's exact match on strong identifiers before the write is
free and never wrong, while the platform's fuzzy pass on names after the write is
neither ([DECISIONS #21](decisions.md)).

So the mechanism is small:

- A sink may declare `profile = PlatformProfile(name, resolves=, constrains=, prunes=)`
  saying what its store covers.
- If a real stage is configured on both sides (a resolver while the store
  `resolves`, a validator while it `prunes`, or a constrainer while it
  `constrains`), `Pipeline` emits **one** `DoubleStageWarning` at construction and
  runs what it was given. The pipeline warns and never refuses. A constrainer whose
  `platform` matches the profile's `name` is that store's other half, not a
  rerun, so it does not warn.
- `Delegated(to="...")` satisfies every stage Protocol as a pass-through and
  stamps `to` wherever the data model has a provenance slot. A delegated resolver
  sets `Entity.resolution = Resolution(method="linker", linker=to)` on every
  entity that arrives unresolved. A delegated router or validator names `to` in
  its verdict's `reason`. The platform's work can then be read back and scored the
  same way as work done here.

```python
import warnings

from odke import Delegated, DoubleStageWarning, PlatformProfile


class GraphRagSink:
    profile = PlatformProfile(name="neo4j-graphrag", resolves=True)

    def write(self, kg):
        self.written = kg


class ExactKeyResolver:
    def resolve(self, facts, index):
        return facts, ()


class OneFact:
    def extract(self, chunk, ontology):
        yield Fact(
            subject=Entity(key="c:acme", type="Company"), predicate="sells", object_value="anvils"
        )


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Pipeline(Ontology(), OneFact(), resolver=ExactKeyResolver(), sinks=[GraphRagSink()])
assert [w.category for w in caught] == [DoubleStageWarning]

sink = GraphRagSink()
pipeline = Pipeline(
    Ontology(),
    OneFact(),
    resolver=Delegated(to="neo4j-graphrag:FuzzyMatchResolver"),
    sinks=[sink],
)
kg = pipeline.run([Document(text="Acme sells anvils.")])
assert kg.facts[0].subject.resolution == Resolution(
    method="linker", linker="neo4j-graphrag:FuzzyMatchResolver"
)
```

Delegating to a store that *replaces* nodes loses the evidence a pre-write link
would have kept. That is a fact about the platform, not a reason to refuse it.
When a caller keeps both passes, the [evaluator](evaluation.md), not the
pipeline, is where the result shows up as a number.
