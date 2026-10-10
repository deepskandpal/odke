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
  accumulate (`SET n +=`), and nothing is ever deleted. Through the Validator,
  a fact the store already holds is first [merged with the store](#merge-with-the-store),
  so what replaces the properties carries every source. Retracting a source is
  the [reconciler's](#reconcile) job.

## Merge with the store

A claim the store already holds, stated again by another source, gains that
source. It does not gain a second edge, and it does not lose the sources it had
([DECISIONS #35](decisions.md#35)). Rewriting the relationship alone would
replace its support list with the batch's.

- **The sink reads; the corroborator merges.** A sink that can say what it
  already holds is a `FactLookup`: `stored(facts)` returns, by signature, the
  stored fact for each one the store holds. `SignatureCorroborator(store=...)`
  merges each stored fact into its claim as one more member, after the batch
  merges and before the scorer and the gate. The support list grows by the
  batch's new sources, and the score counts them all.
- **`Neo4jSink.stored`** is one read transaction per write: for each
  predicate, an `UNWIND` of the batch's signature keys through the index its
  relationship uniqueness constraint brings, which `bootstrap()` creates. A
  predicate with no index is not read, and it warns rather than scan, so a
  store never bootstrapped is written as before, without merging. The
  evidence comes back from the relationship's lists, without quotes or
  mentions, which it never held.
- **`JsonlSink(directory, merge=True)`** makes the files a store. A write keeps
  what they hold, replaces each entity, fact and link the graph restates, and
  adds the rest; `stored` reads `facts.jsonl`. Without `merge`, each write
  replaces the files and there is nothing to merge with.
- **A statement wins** across the store as within a batch
  ([DECISIONS #28](decisions.md#28)). A derived fact merges only with a derived
  one. A derived twin of a stored statement is dropped, and a statement
  replaces a stored derived twin. A derived fact then shares its parent's
  merged list.
- **The Validator merges by default.** It hands every sink that is a
  `FactLookup` to its default corroborator, and `report.restated` counts the
  facts merged with the store. A corroborator you pass in merges with the
  `store` you gave it. A dry run reads no sink. `odke validate` merges with a
  config's sinks, with a `jsonl` sink that has `merge: true`, and with `-o`
  given `--merge`. `odke run` does not merge yet.

```python
from openodke import Document, Validator
from openodke.sinks import JsonlSink
from openodke.stages import PassThroughGate, PassThroughGrounder

store = JsonlSink(out / "store", merge=True)
validator = Validator(
    ontology, grounder=PassThroughGrounder(), gate=PassThroughGate(), sinks=[store]
)


def employed(doc_id):
    cited = Evidence(doc_id=doc_id, span=Span(doc_id=doc_id, start=0, end=14))
    return Fact(subject=ada, predicate="employer", object_entity=acme, evidence=(cited,))


validator.validate([employed("ar-2025")], [Document(id="ar-2025", text="Ada joined Acme.")])
_, report = validator.validate(
    [employed("ar-2026")], [Document(id="ar-2026", text="Ada is at Acme.")]
)
(held,) = store.stored([employed("any")]).values()
print(report.restated, held.support, [s.source for s in held.supported_by])
# 1 2 ['doc:ar-2025', 'doc:ar-2026']
```

## Reconcile

Sources change, and a store that only adds keeps facts whose text no longer
exists. The reconciler retracts a source from the store
([DECISIONS #37](decisions.md#37)):

- **Delete.** Every fact a document backed loses that document: its evidence,
  and its place in the [support list](resolution-and-corroboration.md#support-lists).
  A source entry left with no document goes, and `support` is what is left.
- **Update.** The same, then the new version validated, so the facts it still
  states [merge with the store](#merge-with-the-store) again and regain it.
- **Retire.** A fact left with no source is kept, with `Fact.retired_at` set
  to the transaction time it lost its last one, no evidence and `support` 0.
  It was not proven false, so its valid clock is left alone. It is never the
  value: not projected, not counted by [`check()`](#check), not asserted as a
  plain RDF triple. A source that states it again brings it back.
  `hard_delete=True` (`--hard-delete`) deletes it instead.
- **Derived facts** cite their parent's evidence, so the same retraction
  reaches them, and one whose parent is retired is retired with it.
- **Idempotent.** A fact that no longer cites a document is not touched, and a
  retired fact keeps the time it was first retired.

| | Python | Command line |
|---|---|---|
| a document is gone | `Reconciler(sinks).delete(doc_ids)` | `odke reconcile --sink <dir or bolt://…> --delete <doc-id>` |
| a document changed | `Validator(..., sinks=...).validate(facts, [new], update=True)` | `odke validate ... --update` |

A store that can retract is `Retractable`. `JsonlSink.retract` rewrites
`facts.jsonl`. `Neo4jSink.retract` runs in one write transaction:
- it finds the facts through the per-predicate full-text index over
  `evidence_doc_ids` that `bootstrap()` creates, with the keyword analyzer,
  never by a scan; a relationship type without one is not read, and warns;
- it rewrites each fact's evidence and support lists, `support` and
  `retired_at` by element id;
- it projects each value again from the claims left, removing it when none
  is.

Neither rescores, so a fact keeps its confidence until it is next validated.
An update retracts first. A run that fails before it writes leaves the old
version retracted; running it again finishes it.

```python
from datetime import UTC, datetime

from openodke import Reconciler

report = Reconciler([store]).delete("ar-2025", at=datetime(2026, 10, 9, tzinfo=UTC))
print(report.render())
# odke reconcile
# retracted     ar-2025
# cited         1 fact
# lost support  1 fact, still backed
# retired       0 facts left with no source, kept
(held,) = store.stored([employed("any")]).values()
assert [s.source for s in held.supported_by] == ["doc:ar-2026"]

Reconciler([store]).delete("ar-2026")
(gone,) = store.stored([employed("any")]).values()
assert (gone.support, gone.retired_at is not None) == (0, True)
assert Reconciler([store]).delete("ar-2026").cited == 0  # again: nothing moves
```

`odke reconcile` reads a JSONL directory odke wrote, or a Neo4j URI with
`--user`, `--password-env` and `--database` as `odke validate` takes them, and
`--ontology` so a multi-valued value is projected again as a list. To update
a document, give the new version to `odke validate --update`: it retracts
every text it is given from the config's sinks, or from `-o`, which it then
merges into, before it writes.

## Choose a sink

| Class | `odke run` name | Extra | Writes | On a second `write` |
|---|---|---|---|---|
| `openodke.sinks.JsonlSink` | `jsonl` | — | `entities.jsonl`, `facts.jsonl`, `links.jsonl`, `manifest.json` | rewrites the files; with `merge=True`, merges into them |
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
`label` and `aliases` per type, an index on `key` for `:Entity`, and a
full-text index over `evidence_doc_ids` per predicate, which the
[reconciler](#reconcile) finds a document's facts through. A type or
predicate the ontology does not name gets no constraint or index. The key
constraint and these indexes are what `Neo4jSink.lookup()` reads through to
resolve a new batch against the graph without loading it
([Resolving against the store](resolution-and-corroboration.md#resolving-against-the-store)).

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
# WHERE r.polarity = 'asserted' AND r.valid_to IS NULL AND r.retired_at IS NULL
# WITH s, collect(DISTINCT coalesce(o.key, o.value)) AS objects
# WHERE size(objects) > 1
# RETURN labels(s) AS labels, s.key AS subject, objects
```

### Provenance on every fact

| Property | Holds |
|---|---|
| `fact_id`, `signature`, `polarity`, `identity_keys` | the fact's own fields; `signature` is a SHA-256 of `Fact.signature` |
| `extractor`, `verdict`, `confidence`, `support` | what the stages decided |
| `support_sources`, `support_tiers`, `support_retrieved_at` | the [support list](resolution-and-corroboration.md#support-lists) as parallel lists, one position per independent source; `support` is their length when they are filled |
| `support_doc_ids`, `support_doc_sources` | each source's documents, flattened: a document, and the source it belongs to. `support_from(props)` reads the list back ([DECISIONS #33](decisions.md#33)) |
| `valid_from`, `valid_to` | when the claim was true |
| `retrieved_at` | the newest evidence's retrieval time |
| `retired_at` | when the [reconciler](#reconcile) took the fact's last source away; absent on a live fact |
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
writes each fact as a reified `rdf:Statement` carrying its provenance, with an
`odke:evidence` node per piece of evidence and an `odke:supported_by` node per
independent source (`odke:source`, `odke:doc_id`, `odke:tier`,
`odke:retrieved_at`). The plain
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

`JsonlSink(directory, *, merge=False)` writes every entity, fact and link as
JSON Lines, with the stage counts in `manifest.json`, to which a run adds its
[run manifest](run.md#the-run-manifest). `facts.jsonl` is a
predictions file for [`odke eval`](evaluation.md). With `merge=True` the files
are a store, and a write [merges with it](#merge-with-the-store).

## Your own sink

Anything with `write(kg)` is a sink ([DECISIONS #5](decisions.md#5)).
`plan(kg, ontology=...)` and `provenance_of(fact, extracted_at)` in
`openodke.sinks.neo4j` give you the shape above. A store that does a stage
itself declares `profile = PlatformProfile(...)`
([Concepts](concepts.md#platformprofile-and-delegated)).
`openodke.eval.assert_idempotent(sink, kg, counts)` writes twice and checks that
nothing moved.
