# Write to a store

A sink takes the finished `KnowledgeGraph` and has one method, `write(kg)`.
Every graph sink writes the same shape, from one write plan
(`openodke.sinks.neo4j.plan()`), so a graph in Neo4j, in a Cypher script, in CSV,
in RDF or in NetworkX is one graph.

## The shape

| openodke | Written as | Matched on a rerun by |
|---|---|---|
| `Entity` | a node `(:Type:Entity {key})` with `label`, `aliases`, `external_id`, `resolution_*` and its attributes | type and key |
| a fact whose object is an entity | a relationship `(s)-[:predicate]->(o)` carrying the [provenance](#provenance-on-every-fact) | the fact's [signature](concepts.md#the-signature-and-what-is-in-it) |
| a fact whose object is a value | the same relationship to a `(:Claim {signature, value})` node, and the value [projected](#projection-and-reruns) onto the subject as `s.predicate` | the fact's signature |
| `EntityLink` | a `SAME_AS`, `SIMILAR` or `DIFFERENT` relationship between two nodes already written; nodes are never merged ([DECISIONS #16](decisions.md#16)) | its two keys and its kind |

The examples on this page write this graph: a single-valued `hq` with two
values, the loser of which has more sources; a denial; a value scoped by an
identity-bearing qualifier; and a `DIFFERENT` link.

```python
import tempfile
from pathlib import Path

from openodke import (
    Entity,
    EntityLink,
    Evidence,
    Fact,
    KnowledgeGraph,
    LinkKind,
    Ontology,
    Polarity,
    Span,
)

ontology = Ontology.from_dict(
    {
        "name": "companies",
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company"},
            "hq": {"domain": ["Company"]},
            "sells": {"domain": ["Company"], "cardinality": "multi"},
            "uptime": {
                "domain": ["Company"],
                "range": "number",
                "qualifiers": {"percentile": {"identity": True}},
            },
        },
    }
)
ada = Entity(key="p:ada", type="Person", label="Ada Lovelace")
acme = Entity(key="c:acme", type="Company", label="Acme")
acme_ltd = Entity(key="c:acme-ltd", type="Company", label="Acme Ltd")
report = Evidence(
    doc_id="annual-report",
    uri="https://example.com/ar",
    span=Span(doc_id="annual-report", start=0, end=18, quote="Acme is in Munich."),
)
lost = {"status": "lost", "reason": "Munich is backed by a more trusted source"}
kg = KnowledgeGraph(
    entities=(ada, acme, acme_ltd),
    facts=(
        Fact(subject=ada, predicate="employer", object_entity=acme, evidence=(report,)),
        Fact(subject=acme, predicate="hq", object_value="Munich", support=3, evidence=(report,)),
        Fact(
            subject=acme,
            predicate="hq",
            object_value="Berlin",
            support=4,
            qualifiers={"odke.conflict": lost},
            evidence=(report,),
        ),
        Fact(
            subject=acme,
            predicate="sells",
            object_value="customer data",
            polarity=Polarity.DENIED,
            evidence=(report,),
        ),
        Fact(
            subject=acme,
            predicate="uptime",
            object_value=99.9,
            qualifiers={"percentile": "p95"},
            identity_keys=("percentile",),
            evidence=(report,),
        ),
    ),
    links=(
        EntityLink(
            source_key="c:acme",
            target_key="c:acme-ltd",
            kind=LinkKind.DIFFERENT,
            reason="external_id mismatch: DE-114322 vs GB-889401",
        ),
    ),
)
out = Path(tempfile.mkdtemp())
```

## Projection and reruns

These rules hold for every graph sink.

- **Projection.** A value is set on its subject as a plain property only when
  the fact is asserted, has no identity-bearing qualifier, and did not lose a
  conflict. A single-valued predicate carries its best-supported claim
  (highest `support`, then `confidence`). A multi-valued one carries the list
  when the sink is given the ontology.
- **Lost conflicts.** A fact the corroborator voted down
  (`odke.conflict` status `lost`) is never projected, however many sources back
  it. Its claim and relationship are still written. `odke.conflict` is stored as
  JSON text, not a map.
- **Two values for a single predicate** are two relationships;
  [`check()`](#check) reports them.
- **A rerun into Neo4j or NetworkX** finds each node, relationship and claim by
  its key and updates it. `SET r =` replaces a relationship's properties, so a
  stale qualifier does not survive. Node attributes and projected values
  accumulate (`SET n +=`), and nothing is ever deleted. Retracting a source is the reconciler's job, which
  is not built yet ([#116](https://github.com/deepskandpal/odke/issues/116)).

## Choose a sink

| Class | `odke run` name | Extra | Writes | On a second `write` |
|---|---|---|---|---|
| `openodke.sinks.JsonlSink` | `jsonl` | — | `entities.jsonl`, `facts.jsonl`, `links.jsonl`, `manifest.json` | rewrites the files |
| `openodke.sinks.neo4j.Neo4jSink` | `neo4j` | `neo4j` | a live Neo4j | updates in place |
| `openodke.sinks.bulk.CypherFileSink` | `cypher_file` | — | a `.cypher` script | rewrites the file; replaying it updates in place |
| `openodke.sinks.bulk.Neo4jAdminCsvSink` | `neo4j_admin_csv` | — | CSV files for `neo4j-admin database import` | rewrites the files |
| `openodke.sinks.rdf.RdfSink` | `rdf` | `rdf` | Turtle, N-Triples or JSON-LD | rewrites the file |
| `openodke.sinks.networkx.NetworkXSink` | `networkx` | `networkx` | a `networkx.MultiDiGraph` | adds and updates, never removes |

Every module imports on the base install. A missing extra fails with the
command that installs it. Options per sink are in
[`odke run`](run.md#sinks).

## Neo4j

Needs Neo4j 5.7 or later (relationship uniqueness constraints arrived in 5.7);
Community Edition is enough. `pip install "openodke[neo4j]"`.

`Neo4jSink(uri, auth, *, database=None, batch_size=500, driver=None, ontology=None)`
writes batched `UNWIND … MERGE`, one transaction per batch. Call
`sink.bootstrap(ontology)` before the first write: it applies the constraints
and indexes, each `IF NOT EXISTS`, and `dry_run=True` returns them without
running them. Creating the sink does not connect.

```python
from openodke.sinks.neo4j import Neo4jSink

with Neo4jSink("bolt://localhost:7687", auth=("neo4j", "change-me"), ontology=ontology) as sink:
    ddl = sink.bootstrap(ontology, dry_run=True)  # the DDL, for a DBA to apply
    planned = sink.statements(kg)  # what write() would run, in order

projected = [s.rows for s in planned if s.kind == "projection"]
assert projected == [[{"subject_key": "c:acme", "value": "Munich"}]]
berlin = next(r for s in planned if s.kind == "claim" for r in s.rows if r["value"] == "Berlin")
assert '"status": "lost"' in berlin["props"]["odke.conflict"]  # JSON text
print(ddl[2])
# CREATE CONSTRAINT odke_key_Company IF NOT EXISTS FOR (n:`Company`) REQUIRE n.key IS UNIQUE
```

Against a server:

<!-- docs: no-run -->
```python
import os

uri, auth = os.environ["NEO4J_URI"], (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
with Neo4jSink(uri, auth, database="neo4j") as sink:
    sink.bootstrap(ontology)
    sink.write(kg)
    print(sink.check(ontology))  # {'hq': [{'subject': 'c:acme', 'objects': [...], ...}]}
```

In a `Pipeline`, pass `constrainer=Neo4jConstrainer()` beside the sink; it does
not raise a `DoubleStageWarning`.

### What Neo4j enforces, and what it cannot

| Rule | How |
|---|---|
| one node per (type, key) | a uniqueness constraint per entity type |
| one relationship per (predicate, signature), one `:Claim` per signature | uniqueness constraints |
| one value for a `single` predicate | cannot be enforced; a check query instead |
| existence, property-type and node-key constraints | Enterprise Edition only, so not emitted |
| domain and range | not enforced by Neo4j; `VerdictGate(schema=True)` refuses a fact outside them |

Bootstrap also creates indexes on `external_id` per type, a full-text index over
`label` and `aliases` per type, and an index on `key` for `:Entity`. A type or
predicate the ontology does not name gets no constraint or index.

### `check()`

`sink.check(ontology)` runs one check query per `single` predicate, marked
`// odke:check <predicate>`, and returns `{predicate: [violators]}`. It
changes nothing. Only asserted relationships with no `valid_to` count, grouped
by the predicate's
[`scope_keys`](ontology.md#cardinality-and-its-scope).

```python
from openodke.sinks.neo4j import Neo4jConstrainer

(check,) = [c for c in Neo4jConstrainer().checks(ontology) if "odke:check hq" in c]
print(check)
# // odke:check hq
# MATCH (s)-[r:`hq`]->(o)
# WHERE r.polarity = 'asserted' AND r.valid_to IS NULL
# WITH s, collect(DISTINCT coalesce(o.key, o.value)) AS objects
# WHERE size(objects) > 1
# RETURN labels(s) AS labels, s.key AS subject, objects
```

### Provenance on every fact

| Property | Holds |
|---|---|
| `fact_id`, `signature`, `polarity`, `identity_keys` | the fact's own fields; `signature` is a SHA-256 of `Fact.signature` |
| `extractor`, `verdict`, `confidence`, `support` | what the stages decided |
| `valid_from`, `valid_to` | when the claim was true |
| `retrieved_at` | the newest evidence's retrieval time |
| `extracted_at` | `KnowledgeGraph.created_at` |
| `evidence_doc_ids`, `evidence_uris`, `evidence_starts`, `evidence_ends`, `evidence_tiers`, `evidence_retrieved_at` | parallel lists, one position per piece of evidence; a missing uri is `""` and a missing span `-1` |
| `evidence_span_origins` | who chose each span: `cited`, `located` or `context` ([DECISIONS #25](decisions.md#25)) |
| each qualifier | under its own name, or `qualifier_<name>` if a provenance property has that name; a map or mixed list as JSON text |

```cypher
// Why is this edge here?
MATCH (:Person {key: $person})-[r:employer]->(c:Company)
RETURN c.key, r.evidence_doc_ids, r.evidence_starts, r.evidence_ends, r.verdict, r.support;

// Every fact from one source, edges and claims alike.
MATCH (s:Entity)-[r]->(o) WHERE $doc IN r.evidence_doc_ids AND r.signature IS NOT NULL
RETURN s.key, type(r) AS predicate, coalesce(o.key, o.value) AS object, r.polarity;

// Facts the corroborator voted down. odke.conflict is JSON text.
MATCH (s:Entity)-[r]->(o) WHERE r.`odke.conflict` CONTAINS '"status": "lost"'
RETURN s.key, type(r) AS predicate, coalesce(o.key, o.value) AS object, r.`odke.conflict`;
```

## Cypher file

`CypherFileSink(path, *, ontology=None, batch_size=500)` writes `Neo4jSink`'s
statements as a script, with each batch's rows inlined. With an ontology it
opens with the bootstrap DDL. Replay it with
`cypher-shell -a <uri> -u <user> -f graph.cypher`; replaying twice changes
nothing.

```python
from openodke.sinks.bulk import CypherFileSink

CypherFileSink(out / "graph.cypher", ontology=ontology).write(kg)
```

## neo4j-admin CSV

`Neo4jAdminCsvSink(directory, *, ontology=None, delimiter=",", array_delimiter=";")`
writes one CSV file per node or relationship group, typed headers, and an
`import.args` file. An import builds a new database; to extend a live one, use
`Neo4jSink` or the Cypher file.

```python
from openodke.sinks.bulk import Neo4jAdminCsvSink

Neo4jAdminCsvSink(out / "import", ontology=ontology).write(kg)
assert (out / "import" / "import.args").is_file()
# then: cd import && neo4j-admin database import full @import.args neo4j
```

## RDF

`RdfSink(path, *, format=None, base="https://example.org/odke/", schema=None, ontology=None)`
writes each fact as a reified `rdf:Statement` carrying its provenance. The plain
triple `<s> <predicate> <o>` is written only for an asserted fact with no
identity-bearing qualifier that did not lose a conflict, so a denial or a scoped
value asserts nothing. openodke's own terms
are in `odke:`, `https://openodke.dev/vocab#`. With an ontology, the file also
holds the schema as OWL, and a value is typed with its predicate's range, such
as `xsd:date`, when it parses as one. The suffix picks the format: `.ttl`, `.nt`,
`.jsonld` or `.json`.

```python
from openodke.sinks.rdf import RdfSink

graph = RdfSink(out / "graph.ttl", ontology=ontology).graph(kg)  # what write() serialises
ASK = "PREFIX ont: <https://example.org/odke/schema/> ASK { ?company ont:%s ?value }"
assert graph.query(ASK % "hq").askAnswer
assert not graph.query(ASK % "sells").askAnswer  # a denial
assert not graph.query(ASK % "uptime").askAnswer  # a value without its percentile
```

## NetworkX

`NetworkXSink(graph=None, *, ontology=None)` fills a `networkx.MultiDiGraph`. A
node's id is the entity's key, a claim is the node `claim:<signature>`, and
each edge has a `kind`: `fact` or `link`.

```python
from openodke.sinks.networkx import NetworkXSink

inspector = NetworkXSink(ontology=ontology)
inspector.write(kg)
inspector.write(kg)  # a rerun adds nothing
assert (inspector.graph.number_of_nodes(), inspector.graph.number_of_edges()) == (7, 6)
assert inspector.graph.nodes["c:acme"]["hq"] == "Munich"
```

## JSONL

`JsonlSink(directory)` writes every entity, fact and link as JSON Lines, with
the stage counts in `manifest.json`. `facts.jsonl` is a predictions file for
[`odke eval`](evaluation.md).

## Your own sink

Anything with `write(kg)` is a sink ([DECISIONS #5](decisions.md#5)).
`plan(kg, ontology=...)` and `provenance_of(fact, extracted_at)` in
`openodke.sinks.neo4j` give you the shape above. A store that does a stage
itself declares `profile = PlatformProfile(...)`
([Concepts](concepts.md#platformprofile-and-delegated)).
`openodke.eval.assert_idempotent(sink, kg, counts)` writes twice and checks that
nothing moved.
