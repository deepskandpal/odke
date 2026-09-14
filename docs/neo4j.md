# Neo4j sink

`odke.sinks.neo4j` is the Neo4j platform. It has two halves that form one design:

- **`Neo4jSink`** writes a `KnowledgeGraph` with batched, idempotent `UNWIND … MERGE`.
- **`Neo4jConstrainer`** compiles the ontology into the uniqueness constraints
  that make those `MERGE`s correct under concurrent writers, and into check
  queries for the rule Neo4j cannot enforce.

The ontology that shaped the prompt also shapes the store, so nobody writes the
rules twice.

```bash
pip install "odke[neo4j]"
```

The module imports on the base install, and printing the DDL or the write plan
needs no driver. Only connecting needs the extra; without it you get
`the neo4j driver is not installed; run: pip install 'odke[neo4j]'`.

**Neo4j 5.7 or later.** Relationship uniqueness constraints arrived in 5.7.
Community Edition is enough, because nothing Enterprise-only is emitted. CI runs
the live tests against `neo4j:5.26-community`.

## The shape, and why

| odke | In Neo4j |
|---|---|
| `Entity` | `(:Type:Entity {key})`, `MERGE`d per type on `key`, with `label`, `aliases`, `external_id`, `resolution_method`, `resolution_score`, `resolution_linker` and the entity's attributes as properties |
| edge `Fact` | `(s)-[:predicate {signature, …provenance}]->(o)`, `MERGE`d on the fact's signature |
| literal `Fact` | `(s)-[:predicate {signature, …provenance}]->(:Claim {signature, predicate, subject_key, subject_type, value})` |
| `EntityLink` | `(a)-[:SAME_AS \| SIMILAR \| DIFFERENT {score, reason, created_at, evidence_doc_ids, evidence_uris}]->(b)` |

**Every node also carries `:Entity`,** so a link, which knows only keys, can find
both ends without their types. A link `MATCH`es its ends and never creates them.
Nodes are never merged into one another ([DECISIONS #16](decisions.md)).

**A fact's relationship is `MERGE`d on `signature`,** which is a SHA-256 of
`Fact.signature`: subject, predicate, object, polarity and the identity-bearing
qualifiers ([Concepts](concepts.md#the-signature-and-what-is-in-it)). A second run
mints new fact ids and new clocks, finds the relationship it wrote, and updates it
rather than adding another. Two facts that differ only in a reconcilable
qualifier are one relationship; uptime at p50 and at p95 are two; a denial is its
own relationship with `polarity: 'denied'`.

**A literal fact is the `:Claim`'s relationship, not a property.** A property
cannot carry provenance, nor two contested values, nor a denial. So the `:Claim`
is the record, and provenance lives on its relationship exactly as it does on an
edge. That gives one query shape for both kinds of fact.

**The value is also projected onto the subject** as `s.predicate`, because that is
what a Cypher query reaches for first. The projection is derived and lossy on
purpose:

- Only asserted, unscoped claims project. A denial is not a value, and a value
  with an identity-bearing qualifier means nothing without its scope.
- A single-valued predicate carries its best-supported claim (highest `support`,
  then `confidence`).
- A multi-valued predicate carries the list, when the sink knows the ontology.

**A single-valued predicate with two objects is written as two relationships, not
replaced.** The store holds the conflict and a [check query](#what-neo4j-cannot-enforce)
reports it. Picking a winner at write time would destroy the evidence for the
loser.

### Provenance on every fact relationship

| Property | From |
|---|---|
| `fact_id`, `signature`, `polarity`, `identity_keys` | the fact |
| `extractor`, `verdict`, `confidence`, `support` | the stages that touched it |
| `valid_from`, `valid_to` | the valid clock |
| `retrieved_at` | the freshest evidence: the last time a source confirmed it |
| `extracted_at` | `KnowledgeGraph.created_at` |
| `evidence_doc_ids`, `evidence_uris`, `evidence_starts`, `evidence_ends`, `evidence_tiers`, `evidence_retrieved_at` | parallel lists, one position per piece of evidence |

Neo4j lists hold no nulls and no maps. A missing uri is therefore `""` and a
missing span is `-1`, and position *i* in every list is the same piece of
evidence. Reconcilable qualifiers become relationship properties. Anything Neo4j
cannot store (a map, a mixed or nested list) is written as JSON text rather than
dropped.

Names are protected in both directions. A qualifier named like a provenance
property lands as `qualifier_<name>`. An attribute or projected predicate named
like an entity field lands as `attribute_<name>` or `property_<name>`. Every
label, relationship type and property name from the ontology is backtick-quoted,
so an inferred schema cannot inject Cypher.

## Using it

```python
from odke import Entity, Evidence, Fact, KnowledgeGraph, Ontology, Polarity, Span
from odke.sinks.neo4j import Neo4jConstrainer, Neo4jSink

ontology = Ontology.from_dict(
    {
        "name": "companies",
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company"},
            "hq": {"domain": ["Company"]},
            "sells": {"domain": ["Company"], "cardinality": "multi"},
        },
    }
)
ada = Entity(key="p:ada", type="Person", label="Ada Lovelace")
acme = Entity(key="c:acme", type="Company", label="Acme")
report = Evidence(
    doc_id="annual-report",
    uri="https://example.com/ar",
    span=Span(doc_id="annual-report", start=0, end=64),
)
kg = KnowledgeGraph(
    facts=(
        Fact(subject=ada, predicate="employer", object_entity=acme, evidence=(report,)),
        Fact(subject=acme, predicate="hq", object_value="Munich", support=3, evidence=(report,)),
        Fact(subject=acme, predicate="hq", object_value="Berlin", evidence=(report,)),
        Fact(
            subject=acme,
            predicate="sells",
            object_value="customer data",
            polarity=Polarity.DENIED,
            evidence=(report,),
        ),
    )
)

# Creating the sink does not connect, and neither call below sends anything.
with Neo4jSink("bolt://localhost:7687", auth=("neo4j", "change-me"), ontology=ontology) as sink:
    ddl = sink.bootstrap(ontology, dry_run=True)  # the DDL, for a DBA to apply by hand
    planned = sink.statements(kg)  # exactly what write() would run, in order

projections = [s for s in planned if "SET s." in s.cypher]
assert [s.rows for s in projections] == [[{"subject_key": "c:acme", "value": "Munich"}]]
print(planned[-1].cypher)
# UNWIND $rows AS row
# MATCH (s:`Company` {key: row.subject_key})
# SET s.`hq` = row.value
```

`Neo4jSink(uri, auth, *, database=None, batch_size=500, driver=None, ontology=None)`
also accepts a ready-made `driver`. `ontology` decides whether a projected value
is one value or a list, and `bootstrap()` sets it for you. Against a real server:

<!-- docs: no-run -->
```python
import os

uri, auth = os.environ["NEO4J_URI"], (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
with Neo4jSink(uri, auth, database="neo4j") as sink:
    sink.bootstrap(ontology)  # every statement IF NOT EXISTS: safe on every start
    sink.write(kg)
    sink.write(kg)  # a re-run: the same nodes and relationships, updated in place
    print(sink.check(ontology))  # hq is single-valued and Acme holds two: {'hq': [...]}
```

**Batches and transactions.** Rows go in batches of `batch_size`, one managed
transaction each, so a failed batch rolls back whole and never half-writes.
Earlier batches stay committed. Because every statement is a `MERGE`, recovering
from a failure means running the write again. Write order is nodes, then the
relationships that `MATCH` them, then links, and one graph always compiles to the
same statements.

**In a pipeline,** pass the constrainer next to the sink. A constrainer compiled
for the sink's own platform is its other half, so it does not raise a
`DoubleStageWarning`. `run()` does not apply the DDL; call `sink.bootstrap()`
before the first write.

```python
import warnings

from odke import Pipeline


class NoFacts:
    def extract(self, chunk, ontology):
        return ()


with warnings.catch_warnings():
    warnings.simplefilter("error")
    pipeline = Pipeline(
        ontology,
        NoFacts(),
        constrainer=Neo4jConstrainer(),
        sinks=[Neo4jSink("bolt://localhost:7687", auth=("neo4j", "change-me"))],
    )
assert list(pipeline.constraints()) == list(Neo4jConstrainer().constrain(ontology))
```

## What Neo4j enforces

Once `bootstrap()` has run, Neo4j enforces the following in every edition:

- **One node per (type, key):** a uniqueness constraint per entity type. It is
  what makes the sink's `MERGE` correct under concurrent writers, since without it
  two writers can both miss and both create. It is also the index the `MERGE`
  looks up.
- **One relationship per (predicate, signature), and one `:Claim` per signature:**
  the same guarantee for facts.

It also creates indexes, which enforce nothing:

- `external_id`, per type.
- A full-text index over `label` and `aliases`, per type. Full-text is the index
  kind that reaches inside a `LIST<STRING>`.
- `key` on `:Entity`, which is how a link finds a node without knowing its type.

```python
for statement in ddl:
    print(statement)
```

```cypher
CREATE INDEX odke_entity_key IF NOT EXISTS FOR (n:`Entity`) ON (n.key)
CREATE CONSTRAINT odke_claim_signature IF NOT EXISTS FOR (c:`Claim`) REQUIRE c.signature IS UNIQUE
CREATE CONSTRAINT odke_key_Company IF NOT EXISTS FOR (n:`Company`) REQUIRE n.key IS UNIQUE
CREATE INDEX odke_external_id_Company IF NOT EXISTS FOR (n:`Company`) ON (n.external_id)
CREATE FULLTEXT INDEX odke_names_Company IF NOT EXISTS FOR (n:`Company`) ON EACH [n.label, n.aliases]
CREATE CONSTRAINT odke_key_Person IF NOT EXISTS FOR (n:`Person`) REQUIRE n.key IS UNIQUE
CREATE INDEX odke_external_id_Person IF NOT EXISTS FOR (n:`Person`) ON (n.external_id)
CREATE FULLTEXT INDEX odke_names_Person IF NOT EXISTS FOR (n:`Person`) ON EACH [n.label, n.aliases]
CREATE CONSTRAINT odke_signature_employer IF NOT EXISTS FOR ()-[r:`employer`]-() REQUIRE r.signature IS UNIQUE
CREATE CONSTRAINT odke_signature_hq IF NOT EXISTS FOR ()-[r:`hq`]-() REQUIRE r.signature IS UNIQUE
CREATE CONSTRAINT odke_signature_sells IF NOT EXISTS FOR ()-[r:`sells`]-() REQUIRE r.signature IS UNIQUE
```

Only what the ontology names is compiled. A type or predicate that turns up in a
graph but not in the schema gets no constraint: the store enforces nothing it was
not told about.

## What Neo4j cannot enforce

- **Relationship cardinality.** No Neo4j constraint says "a Person has at most one
  employer". Every `single` predicate therefore gets a *check query* instead,
  marked `// odke:check <predicate>`, which returns the violators and changes
  nothing. `sink.check(ontology)` runs them in read transactions and returns
  `{predicate: [violators]}`. It checks and never repairs. The rule is used as a
  check, not as inference: a reasoner told "at most one" would conclude the two
  employers are one company. Only asserted, open-ended relationships count,
  because a denial is not a value and a relationship with `valid_to` set expired
  correctly. The check groups by the predicate's
  [`scope_keys`](ontology.md#cardinality-and-its-scope), so uptime is single per
  percentile.
- **Existence, property-type and node-key constraints.** Neo4j has these only in
  Enterprise Edition, so none is emitted, and `EntityType.keys` is not compiled.
- **Domain and range**, such as "an employer is a Company". A `Validator` is the
  gate for those.

```python
(check,) = [c for c in Neo4jConstrainer().checks(ontology) if "odke:check hq" in c]
print(check)
# // odke:check hq
# MATCH (s)-[r:`hq`]->(o)
# WHERE r.polarity = 'asserted' AND r.valid_to IS NULL
# WITH s, collect(DISTINCT coalesce(o.key, o.value)) AS objects
# WHERE size(objects) > 1
# RETURN labels(s) AS labels, s.key AS subject, objects
```

## Provenance queries

Every fact relationship carries its receipts, so the questions below are plain
Cypher reads:

```cypher
// Why is this edge here?
MATCH (:Person {key: $person})-[r:employer]->(c:Company)
RETURN c.key, r.evidence_doc_ids, r.evidence_starts, r.evidence_ends, r.verdict, r.support;

// Every fact from one source, edges and claims alike.
MATCH (s:Entity)-[r]->(o) WHERE $doc IN r.evidence_doc_ids AND r.signature IS NOT NULL
RETURN s.key, type(r) AS predicate, coalesce(o.key, o.value) AS object, r.polarity;

// What changed since last week: facts written, or reconfirmed by a source, since then.
MATCH (s:Entity)-[r]->(o) WHERE r.extracted_at >= datetime() - duration('P7D')
RETURN s.key, type(r) AS predicate, coalesce(o.key, o.value) AS object, r.retrieved_at;

// Every conflict the cardinality checks would report, straight from the sink:
// sink.check(ontology)

// Forget a source: every relationship it is the only evidence for.
MATCH ()-[r]->() WHERE r.evidence_doc_ids = [$doc] DELETE r;
```
