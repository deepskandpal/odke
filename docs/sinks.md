# Sinks

A sink is where a finished `KnowledgeGraph` goes, and it is one method:
`write(kg)`. Six ship with odke. Every graph-shaped sink writes the **same shape**
Neo4j gets: entities as nodes, each fact as its own edge carrying its provenance,
a literal fact's value on a claim node, and `EntityLink`s as edges that never merge
nodes. They share one write plan (`openodke.sinks.neo4j.plan()`) and one rule for
which literal values are projected onto a subject. So a graph written to Neo4j,
replayed from a Cypher script, imported from CSV, loaded into a triple store or
inspected in NetworkX is one graph.

| Sink | Module | Writes | Extra | A second `write` |
|---|---|---|---|---|
| `JsonlSink` | `openodke.sinks` | `entities.jsonl`, `facts.jsonl`, `links.jsonl`, `manifest.json` | — | rewrites the files |
| `Neo4jSink` | `openodke.sinks.neo4j` | a live Neo4j, through the driver | `neo4j` | updates in place (`MERGE`) |
| `CypherFileSink` | `openodke.sinks.bulk` | a `.cypher` script for `cypher-shell -f` | — | rewrites the file, which replays idempotently |
| `Neo4jAdminCsvSink` | `openodke.sinks.bulk` | node and relationship CSVs for `neo4j-admin database import`, plus `import.args` | — | rewrites the files |
| `RdfSink` | `openodke.sinks.rdf` | Turtle, N-Triples or JSON-LD | `rdf` | rewrites the file |
| `NetworkXSink` | `openodke.sinks.networkx` | a `networkx.MultiDiGraph` in memory | `networkx` | adds and updates, never removes |

Every module imports on the base install ([DECISIONS #1](decisions.md)). A driver
or library is imported only when a sink needs it, and a missing extra fails with
the command that fixes it. `JsonlSink` is the only sink re-exported from
`openodke.sinks`; the others are imported from their own module. In an
[`odke run`](run.md#sinks) config they are `jsonl`, `neo4j`, `cypher_file`,
`neo4j_admin_csv`, `rdf` and `networkx`.

Every example on this page writes this small graph: an asserted value, a denial, a
value scoped by an identity-bearing qualifier, and a `DIFFERENT` link.

```python
import json
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
        "types": {"Company": {}},
        "predicates": {
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
acme = Entity(key="c:acme", type="Company", label="Acme")
acme_ltd = Entity(key="c:acme-ltd", type="Company", label="Acme Ltd")
report = Evidence(
    doc_id="annual-report",
    uri="https://example.com/ar",
    span=Span(doc_id="annual-report", start=0, end=18, quote="Acme is in Munich."),
)
kg = KnowledgeGraph(
    entities=(acme, acme_ltd),
    facts=(
        Fact(subject=acme, predicate="hq", object_value="Munich", evidence=(report,)),
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

## The shared shape, in one paragraph

An entity is a node keyed by `Entity.key`, carrying its label, aliases,
`external_id`, how its key was decided (`resolution_*`) and its attributes. A fact
is an edge identified by its **signature** (subject, predicate, object, polarity
and identity-bearing qualifiers; see [Concepts](concepts.md#the-signature-and-what-is-in-it)),
so a re-run with new fact ids and clocks addresses the same edge. The edge carries
the provenance: `fact_id`, `signature`, `polarity`, `extractor`, `verdict`,
`confidence`, `support`, both clocks, and the evidence (documents, uris, span
offsets, tiers and retrieval times). A literal fact points at a claim node holding
its value, because a plain property cannot carry provenance, two contested values,
or a denial. The value is also **projected** onto the subject as a plain property,
but only for an asserted fact with no identity-bearing qualifier, and for a
`single` predicate only the best-supported claim. The [Neo4j page](neo4j.md#the-shape-and-why)
has the reasoning in full.

## JSONL

`JsonlSink(directory)` writes the graph as three JSON Lines streams and a manifest,
and needs nothing. `facts.jsonl` holds every `Fact` as `model_dump_json`, evidence
and all, so it is already a predictions file for [`odke eval`](evaluation.md), and
`links.jsonl` is already a resolution predictions file. `manifest.json` holds the
ontology name, `created_at`, the counts, and `KnowledgeGraph.stats`, which is where
`odke run` puts every stage's own counts.

```python
from openodke.sinks import JsonlSink

JsonlSink(out / "jsonl").write(kg)

print(sorted(path.name for path in (out / "jsonl").iterdir()))
# ['entities.jsonl', 'facts.jsonl', 'links.jsonl', 'manifest.json']
manifest = json.loads((out / "jsonl" / "manifest.json").read_text(encoding="utf-8"))
assert (manifest["facts"], manifest["properties"], manifest["links"]) == (3, 3, 1)
```

## Neo4j

`Neo4jSink(uri, auth, *, database=None, batch_size=500, driver=None, ontology=None)`
writes with batched, idempotent `UNWIND … MERGE`, and `Neo4jConstrainer` compiles
the ontology into the uniqueness constraints that make those `MERGE`s correct and
into check queries for the cardinality Neo4j cannot enforce. It has its own page:
[Neo4j sink](neo4j.md).

## Cypher file

`CypherFileSink(path, *, ontology=None, batch_size=500)` is `Neo4jSink`'s write plan
as a script. The statements are the driver's own, with each batch's rows inlined as
a list literal where the driver would bind `$rows`. Inlining, rather than
`:param`, keeps the file runnable by any Cypher client and makes it one
self-contained artefact to review or commit.

- With an `ontology`, the script opens with the `Neo4jConstrainer` DDL, and the
  ontology decides multi-valued projections exactly as it does for `Neo4jSink`.
  The check queries are left out: they return violators, and a script run for its
  side effects would discard the answer.
- Every statement is `CREATE … IF NOT EXISTS` or `UNWIND … MERGE`, so replaying the
  script twice, or over a graph `Neo4jSink` wrote, updates rather than duplicates.
- A value is written as the Cypher literal the Python driver would have sent: an
  aware datetime is a `datetime(...)`, a naive one a `localdatetime(...)`. A string
  never contains a raw newline, which is what keeps statements separable.
- One graph always compiles to the same bytes.

```python
from openodke.sinks.bulk import CypherFileSink

script = CypherFileSink(out / "graph.cypher", ontology=ontology)
script.write(kg)

statements = script.statements(kg)
assert len(statements) == 14  # 8 DDL statements, then 6 UNWIND … MERGE
print(statements[0])
# CREATE INDEX odke_entity_key IF NOT EXISTS FOR (n:`Entity`) ON (n.key)
print(statements[8].splitlines()[0])
# UNWIND [
```

```bash
cypher-shell -a neo4j://localhost:7687 -u neo4j -f graph.cypher
```

## neo4j-admin CSV

`Neo4jAdminCsvSink(directory, *, ontology=None, delimiter=",", array_delimiter=";")`
writes the node and relationship CSV files `neo4j-admin database import` reads. A
driver write is the wrong tool for the first import of a large corpus: a round
trip, a transaction and an index lookup per batch. The layout is the write plan
again, one file per statement group:

- **An entity type is a node file**, `key:ID(<space>)` plus a `:LABEL` column
  holding the type and `Entity`, then the entity fields, its attributes and the
  values projected onto it. Every type gets its own ID space, because Neo4j keys a
  node on (type, key) and two types may share a key.
- **Claims** are a node file per (subject type, predicate), in one `Claim` ID space
  keyed on the signature, with a relationship file from the subjects to them.
- **An edge group and a link kind** are relationship files with `:START_ID`,
  `:END_ID`, `:TYPE` and every provenance property the driver would have set, the
  evidence lists as arrays and both clocks as datetimes.

Every property column is typed in its header, inferred from the values it holds;
a column whose values disagree on a type is written as text, because a CSV column
has one type. Rows `Neo4jSink` would `MERGE` into one (two facts with one
signature, a link written twice) are one row here. Newlines in values are kept, so
the import needs `--multiline-fields=true`, and `import.args` beside the files
carries that and the delimiters:

```bash
cd import && neo4j-admin database import full @import.args neo4j
```

An import builds a new database rather than updating one. To extend a live graph,
use `Neo4jSink` or `CypherFileSink`.

```python
from openodke.sinks.bulk import Neo4jAdminCsvSink

importer = Neo4jAdminCsvSink(out / "import", ontology=ontology)
importer.write(kg)

tables = importer.tables(kg)  # what write() lays out, without writing it
for table in tables:
    print(f"{table.kind:<13} {len(table.rows)}  {table.file}")
# nodes         2  odke_nodes_Company.csv
# nodes         1  odke_claims_Company_hq_ab083334.csv
# relationships 1  odke_claim_edges_Company_hq_ab083334.csv
# nodes         1  odke_claims_Company_sells_c82c6a13.csv
# relationships 1  odke_claim_edges_Company_sells_c82c6a13.csv
# nodes         1  odke_claims_Company_uptime_cd86096d.csv
# relationships 1  odke_claim_edges_Company_uptime_cd86096d.csv
# relationships 1  odke_links_DIFFERENT_Company_Company_dd57a2e2.csv
print(importer.render(tables[0]))
# key:ID(odke_key_Company),:LABEL,label:string,aliases:string[],external_id:string,resolution_method:string,resolution_score:string,resolution_linker:string,hq:string
# "c:acme","Company;Entity","Acme","",,,,,"Munich"
# "c:acme-ltd","Company;Entity","Acme Ltd","",,,,,
assert (out / "import" / "import.args").is_file()
```

## RDF

`RdfSink(path, *, format=None, base="https://example.org/odke/", schema=None,
ontology=None)` writes Turtle, N-Triples or JSON-LD through rdflib. `format` is
`turtle`, `nt` or `json-ld` (with `ttl`, `ntriples`, `n-triples` and `jsonld`
accepted), and when it is left out the file suffix decides (`.ttl`, `.nt`,
`.jsonld`, `.json`), defaulting to Turtle. The default `base` is a documentation
domain on purpose: data minted under it is visibly unplaced. Pass your own.

| odke | RDF |
|---|---|
| `Entity` | `<base>entity/<key>`, typed with its class `<schema><Type>` and `odke:Entity`, with `odke:key`, `rdfs:label`, `skos:altLabel` per alias, `odke:external_id`, `odke:resolution_*`, and its attributes under `<schema>attribute/` |
| `Fact` | a **reified statement** `<base>fact/<signature>`: an `rdf:Statement` and `odke:Fact` with `rdf:subject`, `rdf:predicate` and `rdf:object`, carrying `odke:polarity`, `odke:confidence`, `odke:support`, both clocks, `odke:identity_keys`, its qualifiers under `<schema>qualifier/`, and one `odke:evidence` node per source with `odke:doc_id`, `odke:uri`, `odke:start`, `odke:end`, `odke:quote`, `odke:tier` and `odke:retrieved_at` |
| the plain triple `<s> <schema><predicate> <o>` | written **only** for an asserted fact with no identity-bearing qualifier: the same rule that decides the Neo4j projection |
| `EntityLink` | `owl:sameAs`, `odke:similar_to` or `odke:different_from` between the two entity IRIs, reified the same way so `odke:score`, `odke:reason` and `odke:created_at` can be read |
| the ontology, when given | `owl:Class` with `rdfs:subClassOf` and `owl:hasKey`; `owl:ObjectProperty` or `owl:DatatypeProperty` with domain, range and `owl:FunctionalProperty` for `single` |

`odke:` is `https://deepskandpal.github.io/odke/vocab#`. The statement node plays
the part the relationship plays in Neo4j and carries the same property names, so one
concept has one name in Cypher and in SPARQL.

### How polarity survives

The plain triple is what a SPARQL query reaches for first, and it is a claim of
truth. A denial written that way would say the opposite of its source, and "uptime
99.9" without its percentile says something no source said. So the only asserted
triples are the ones that should be: a denial is a statement node with
`odke:polarity "denied"` and nothing else, and a scoped value is a statement node
with its qualifier and nothing else.

That works because reification is non-asserting in RDF's semantics: describing a
statement does not entail it. The other two candidates for per-fact provenance
lose it:

- **RDF-star** reads cleanest, but rdflib 7 has no quoted-triple term, and neither
  N-Triples 1.1 nor JSON-LD 1.1 can carry one. A quoted triple also has no identity
  beyond its three terms, so uptime at p50 and at p95 with the same value would be
  one quoted triple with two provenance records mixed together, which is the merge
  [DECISIONS #15](decisions.md) exists to prevent.
- **Named graphs** need TriG or N-Quads, not the three formats here, and a denial
  placed in its own graph is still an asserted triple in any store that queries the
  union of its graphs.

The cost is bulk, six or more triples per fact, which is the price of provenance
that can be queried rather than merely stored.

```python
from openodke.sinks.rdf import RdfSink

rdf = RdfSink(out / "graph.ttl", ontology=ontology)
rdf.write(kg)
graph = rdf.graph(kg)  # the rdflib Graph that write() serialises

PREFIXES = """
PREFIX odke: <https://deepskandpal.github.io/odke/vocab#>
PREFIX ont: <https://example.org/odke/schema/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""
statements = graph.query(
    PREFIXES
    + """
    SELECT ?predicate ?object ?polarity WHERE {
      ?fact a odke:Fact ; rdf:predicate ?predicate ; rdf:object ?object ; odke:polarity ?polarity
    } ORDER BY ?predicate"""
)
for predicate, obj, polarity in statements:
    print(predicate.rsplit("/", 1)[-1], obj, polarity)
# hq Munich asserted
# sells customer data denied
# uptime 99.9 asserted

assert graph.query(PREFIXES + 'ASK { ?company ont:hq "Munich" }').askAnswer
# A denial asserts nothing, and neither does a value without its scope.
assert not graph.query(PREFIXES + "ASK { ?company ont:sells ?what }").askAnswer
assert not graph.query(PREFIXES + "ASK { ?company ont:uptime ?value }").askAnswer
```

A nodes-are-never-merged note applies here too ([DECISIONS #16](decisions.md)): a
store that reasons over `owl:sameAs` will merge the two resources itself, which is
that platform's call ([DECISIONS #21](decisions.md)). With an `ontology`, the file
describes its own schema, and [`Ontology.from_owl`](ontology.md#importing-a-schema-you-already-have)
reads it back:

```python
schema_only = RdfSink(out / "schema.ttl", ontology=ontology).serialize(KnowledgeGraph())
assert set(Ontology.from_owl(schema_only).predicates) == {"hq", "sells", "uptime"}
```

## NetworkX

`NetworkXSink(graph=None, *, ontology=None)` fills a `networkx.MultiDiGraph`, to
look at a graph without a database. It keeps the Neo4j shape: an entity is a node
whose id is its key, with `kind="entity"`; an edge fact is an edge keyed by its
signature with `kind="fact"`, `predicate` and the provenance attributes; a literal
fact is an edge to a claim node `claim:<signature>` holding `kind="claim"` and the
`value`, and the value is projected onto the subject node by the same rule as
Neo4j; a link is an edge keyed `SAME_AS`, `SIMILAR` or `DIFFERENT` with
`kind="link"`, `score` and `reason`, drawn only between nodes the graph holds.

Writing adds and updates and never removes, as `MERGE … SET +=` does, so writing one
graph twice, or re-running a pipeline into the same `graph`, leaves the counts where
they were. The node id is the key alone, as in RDF, so a key two types share is one
node. `to_graph(kg)` returns a new graph and leaves `sink.graph` untouched.

```python
from openodke.sinks.networkx import NetworkXSink

inspector = NetworkXSink(ontology=ontology)
inspector.write(kg)
inspector.write(kg)  # a re-run: nothing added
nx_graph = inspector.graph

assert (nx_graph.number_of_nodes(), nx_graph.number_of_edges()) == (5, 4)
# The value is projected onto the node; the denial and the scoped value are not.
assert nx_graph.nodes["c:acme"]["hq"] == "Munich"
for _, _, attrs in nx_graph.edges(data=True):
    print(attrs["kind"], attrs.get("predicate", attrs.get("link")), attrs.get("polarity"))
# fact hq asserted
# fact sells denied
# fact uptime asserted
# link DIFFERENT None
```

## Writing your own

Anything with `write(kg)` is a sink: no base class, no registration
([DECISIONS #5](decisions.md)). Two things are worth borrowing:

- `openodke.sinks.neo4j.plan(kg, ontology=...)` and `provenance_of(fact, extracted_at)`
  give you the shared shape without reimplementing it, which is how the bulk, RDF
  and NetworkX sinks are built.
- A sink whose store does a stage itself declares `profile = PlatformProfile(...)`,
  and the pipeline warns once when a stage is configured on both sides
  ([Concepts](concepts.md#platformprofile-and-delegated)).

Whether a sink is idempotent is checkable without labels:
`openodke.eval.assert_idempotent(sink, kg, counts)` writes twice and compares whatever
`counts` returns ([Evaluation](evaluation.md#sink)).
