# Write to a store

A sink has one method, `write(kg)`. Every graph sink writes one shape, from
one write plan (`openodke.sinks.neo4j.plan()`).

## The shape

| openodke | Written as | Matched on a rerun by |
|---|---|---|
| `Entity` | a node `(:Type:Entity {key})` with `label`, `aliases`, `external_id`, `resolution_*` and its attributes | type and key |
| an edge fact | a relationship `(s)-[:predicate]->(o)` with its [provenance](#provenance-on-every-fact) | its [signature](concepts.md#the-signature-and-what-is-in-it) |
| a property fact | the same, to a `(:Claim {signature, value})`, and the value [projected](#projection-and-reruns) as `s.predicate` | its signature |
| `EntityLink` | a `SAME_AS`, `SIMILAR` or `DIFFERENT` relationship ([links](concepts.md#entities-and-links)) | its keys and kind |

The examples write this graph:

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

- **Projection.** A value is set on its subject only when the fact is asserted,
  has no identity-bearing qualifier, and did not lose a conflict (`odke.conflict`
  status `lost`, stored as JSON text). A single-valued predicate carries its
  best-supported claim (highest `support`, then `confidence`); a multi-valued
  one a list, when the sink has the ontology.
- **A rerun** into Neo4j or NetworkX updates each node, relationship and claim
  found by its key: a relationship's properties are replaced (`SET r =`), node
  attributes and projected values accumulate (`SET n +=`), and nothing is
  deleted; that is the [reconciler's](#reconcile) job.

## Merge with the store

A claim the store holds, stated by another source, gains that source and keeps
its own ([DECISIONS #35](decisions.md#35)). A sink that can say what it holds is
a `FactLookup`, whose `stored(facts)` returns the stored fact for each
signature; `SignatureCorroborator(store=...)` merges those in before the scorer
and the gate, a statement replacing a derived twin
([DECISIONS #28](decisions.md#28)). `Neo4jSink.stored` reads through each
predicate's uniqueness index, warning on a predicate without one; `JsonlSink(directory, merge=True)` keeps what the files hold.

**The Validator merges by default** with every `FactLookup` sink
(`report.restated` counts them), as does `odke validate` with a config's sinks
and `-o --merge`; a dry run reads none. `odke run` does not merge; its `jsonl`
sink with `merge: true` replaces a restated fact.

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

The reconciler retracts a source ([DECISIONS #37](decisions.md#37)):

- **Delete.** Each fact a document backed loses it from its evidence and
  [support list](resolution-and-corroboration.md#support-lists), derived facts
  included.
- **Update.** The same, then the new version is validated and
  [merges](#merge-with-the-store) back.
- **Retire.** A fact left with no source keeps `retired_at`, no evidence and
  `support` 0, and is not projected, [checked](#check) or asserted in RDF until
  a source states it again. `hard_delete=True` (`--hard-delete`) deletes it.

Reconciling twice changes nothing.

| | Python | Command line |
|---|---|---|
| a document is gone | `Reconciler(sinks).delete(doc_ids)` | `odke reconcile --sink <dir or bolt://…> --delete <doc-id>` |
| a document changed | `Validator(..., sinks=...).validate(facts, [new], update=True)` | `odke validate ... --update` |

`JsonlSink` and `Neo4jSink` are `Retractable`; neither rescores. An update
retracts first, so rerun one that fails before writing.

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

`odke reconcile --sink` takes a JSONL directory, or a Neo4j URI with `--user`,
`--password-env`, `--database` and `--ontology`.

## Tenants

Two tenants who state the same claim own two facts in one store
([DECISIONS #44](decisions.md#44)). Give the tenant once:
`Validator(..., tenant="acme")`, `tenant: acme` in a run config, or `--tenant
acme` on `odke run`, `validate` and `reconcile`; every sink and lookup with
`scoped(tenant)` is scoped.

Its keys are `acme/<key>`, its signatures are hashed with `acme`, and its nodes
and relationships carry `tenant: "acme"` (an attribute or qualifier named
`tenant` becomes `attribute_tenant` or `qualifier_tenant`). Lookups, merges,
the reconciler, `check()` and `read_neo4j(driver, tenant=...)` stay in the
tenant and give its own keys back; with no tenant they see only what no tenant
wrote, and one scoped to a tenant refuses another. A JSONL directory holds one
tenant. A tenant is letters, digits, `_`, `.` and `-`, starting with a letter or
digit.

```python
from openodke.sinks.neo4j import plan


def rows(tenant, kind):
    return [row for s in plan(kg, tenant=tenant) if s.kind == kind for row in s.rows]


assert {row["key"] for row in rows("acme", "entity")} == {
    "acme/p:ada",
    "acme/c:acme",
    "acme/c:acme-ltd",
}
assert all(row["props"]["tenant"] == "acme" for row in rows("acme", "edge"))
# The same claim for another tenant is another relationship.
assert rows("acme", "edge")[0]["signature"] != rows("beta", "edge")[0]["signature"]
```

## The write report

Each sink reports, per kind, the rows `written` new, `merged` into one it held,
and `skipped` (a link with no node to `MATCH`), and Neo4j's `transactions`; a
sink without a report lists what it was `handed`. It covers the whole job, and
is `ValidationReport.writes`, `stats["writes"]` (keyed `0:Neo4jSink`), and the
[run manifest's](run.md#the-run-manifest) and `job.end`'s `writes`:

```text
writes        0:Neo4jSink: entities 3 written; facts 2 written, 1 merged; links none; 1 transaction
```

## Choose a sink

| Class | `odke run` name | Extra | Writes | On a second `write` |
|---|---|---|---|---|
| `openodke.sinks.JsonlSink` | `jsonl` | — | JSON Lines and `manifest.json` | rewrites the files, or merges |
| `openodke.sinks.neo4j.Neo4jSink` | `neo4j` | `neo4j` | a live Neo4j | updates in place |
| `openodke.sinks.bulk.CypherFileSink` | `cypher_file` | — | a `.cypher` script | rewrites the file |
| `openodke.sinks.bulk.Neo4jAdminCsvSink` | `neo4j_admin_csv` | — | CSV for `neo4j-admin database import` | rewrites the files |
| `openodke.sinks.rdf.RdfSink` | `rdf` | `rdf` | Turtle, N-Triples or JSON-LD | rewrites the file |
| `openodke.sinks.networkx.NetworkXSink` | `networkx` | `networkx` | a `networkx.MultiDiGraph` | adds and updates, never removes |

A missing extra fails naming its install command. Options in `odke run`:
[Sinks](run.md#sinks).

## Neo4j

Neo4j 5.7 or later, Community Edition included: `pip install "openodke[neo4j]"`.

`Neo4jSink(uri, auth, *, database=None, batch_size=500, driver=None, ontology=None, tenant=None)`
writes batched `UNWIND … MERGE` in transactions of at most `batch_size` rows.
Creating it does not connect. `sink.bootstrap(ontology)` applies the
constraints and indexes (`IF NOT EXISTS`) before the first write;
`dry_run=True` returns them.

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

<!-- docs: no-run -->
```python
import os

uri, auth = os.environ["NEO4J_URI"], (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
with Neo4jSink(uri, auth, database="neo4j") as sink:
    sink.bootstrap(ontology)
    sink.write(kg)
    print(sink.check(ontology))  # {'hq': [{'subject': 'c:acme', 'objects': [...], ...}]}
```

In a `Pipeline`, pass `constrainer=Neo4jConstrainer()` beside the sink; it
raises no `DoubleStageWarning`.

### What Neo4j enforces, and what it cannot

| Rule | How |
|---|---|
| one node per (type, key) | a uniqueness constraint per entity type |
| one relationship per (predicate, signature), one `:Claim` per signature | uniqueness constraints |
| one value for a `single` predicate | not enforceable: [`check()`](#check) |
| existence, property-type, node-key | Enterprise only: not emitted |
| domain and range | not enforced: [`VerdictGate(schema=True)`](grounding.md#the-gate-verdictgate) |

`bootstrap()` also creates indexes: per type on `external_id`, and full-text on
`label` and `aliases`; on `key` for `:Entity`; per predicate full-text on
`evidence_doc_ids`, and on `odke.ontology`. An undeclared type or predicate
gets none. The [store lookup](resolution-and-corroboration.md#resolving-against-the-store)
and the reconciler read through them.

### `check()`

`sink.check(ontology)` returns `{predicate: [violators]}` for each `single`
predicate: asserted, live relationships with no `valid_to`, grouped by
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
| `support_sources`, `support_tiers`, `support_retrieved_at`, `support_doc_ids`, `support_doc_sources` | the [support list](resolution-and-corroboration.md#support-lists) as parallel lists; `support_from(props)` reads it back ([DECISIONS #33](decisions.md#33)) |
| `valid_from`, `valid_to`, `retrieved_at`, `retired_at`, `extracted_at` | the valid clock; the newest evidence; when [retired](#reconcile); `KnowledgeGraph.created_at` |
| `evidence_doc_ids`, `evidence_uris`, `evidence_starts`, `evidence_ends`, `evidence_tiers`, `evidence_retrieved_at` | parallel lists; a missing uri is `""`, a missing span `-1` |
| `evidence_span_origins` | `cited`, `located` or `context` ([DECISIONS #25](decisions.md#25)) |
| each qualifier | under its own name, or `qualifier_<name>` if a property above has it; a map or mixed list as JSON text |

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

### Which ontology checked a fact

Every fact the gate passes carries the qualifier `odke.ontology`, the
[`fingerprint`](ontology.md#freezing) of the last ontology that checked it
([DECISIONS #42](decisions.md#42)); a model's fact also carries
`odke.schema_slice`, its snippet's. Neither is in the signature. Find an old
ontology's facts through the index:

```cypher
// The employer facts one ontology checked, by its fingerprint.
MATCH (s)-[r:employer]->(o) USING INDEX r:employer(`odke.ontology`)
WHERE r.`odke.ontology` = $fingerprint
RETURN s.key, coalesce(o.key, o.value) AS object, r.fact_id;
```

`Neo4jSink.checked_under(ontology)` runs that per predicate in one transaction.
`checked_under(facts, ontology)` and `checked_by(fact)` in
`openodke.corroborate` do it for facts in hand. Each takes an ontology or a
fingerprint.

```python
from openodke import Document, Entity, Evidence, Fact, Ontology, Pipeline, Span
from openodke.corroborate import ONTOLOGY, checked_by, checked_under


class OneClaim:
    def extract(self, chunk, ontology):
        acme = Entity(key="c:acme", type="Company")
        cited = Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=21))
        return [Fact(subject=acme, predicate="hq", object_value="London", evidence=(cited,))]


companies = Ontology.from_dict(
    {"types": {"Company": {}}, "predicates": {"hq": {"domain": ["Company"]}}}
)
checked = Pipeline(companies, OneClaim()).run([Document(id="d1", text="Acme is in London now.")])
fact = checked.facts[0]
assert checked_by(fact) == fact.qualifiers[ONTOLOGY] == companies.fingerprint
assert checked_under(checked.facts, companies) == [fact]
```

[`odke ontology diff --store`](ontology.md#diff) counts them for a schema change.

## Cypher file

`CypherFileSink(path, *, ontology=None, batch_size=500)` writes `Neo4jSink`'s
statements as a script, opening with the bootstrap DDL when given an ontology.
Replay it with `cypher-shell -a <uri> -u <user> -f graph.cypher`; replaying
twice changes nothing.

```python
from openodke.sinks.bulk import CypherFileSink

CypherFileSink(out / "graph.cypher", ontology=ontology).write(kg)
```

## neo4j-admin CSV

`Neo4jAdminCsvSink(directory, *, ontology=None, delimiter=",", array_delimiter=";")`
writes one CSV file per node or relationship group, with typed headers, and an
`import.args` file. An import builds a new database.

```python
from openodke.sinks.bulk import Neo4jAdminCsvSink

Neo4jAdminCsvSink(out / "import", ontology=ontology).write(kg)
assert (out / "import" / "import.args").is_file()
# then: cd import && neo4j-admin database import full @import.args neo4j
```

## RDF

`RdfSink(path, *, format=None, base="https://example.org/odke/", schema=None, ontology=None)`
writes each fact as a reified `rdf:Statement` with its provenance, evidence
(`odke:evidence`) and sources (`odke:supported_by`), in `odke:`
(`https://openodke.dev/vocab#`). The plain triple `<s> <predicate> <o>`
follows the [projection](#projection-and-reruns) rule. An ontology adds the
schema as OWL and types values by range. The suffix picks the format: `.ttl`,
`.nt`, `.jsonld` or `.json`.

```python
from openodke.sinks.rdf import RdfSink

graph = RdfSink(out / "graph.ttl", ontology=ontology).graph(kg)  # what write() serialises
ASK = "PREFIX ont: <https://example.org/odke/schema/> ASK { ?company ont:%s ?value }"
assert graph.query(ASK % "hq").askAnswer
assert not graph.query(ASK % "sells").askAnswer  # a denial
assert not graph.query(ASK % "uptime").askAnswer  # a value without its percentile
```

## NetworkX

`NetworkXSink(graph=None, *, ontology=None)` fills a `networkx.MultiDiGraph`: a
node per entity key, a node `claim:<signature>` per claim, and edges whose
`kind` is `fact` or `link`.

```python
from openodke.sinks.networkx import NetworkXSink

inspector = NetworkXSink(ontology=ontology)
inspector.write(kg)
inspector.write(kg)  # a rerun adds nothing
assert (inspector.graph.number_of_nodes(), inspector.graph.number_of_edges()) == (7, 6)
assert inspector.graph.nodes["c:acme"]["hq"] == "Munich"
```

## JSONL

`JsonlSink(directory, *, merge=False, tenant=None)` writes JSON Lines, and the
stage counts and [run manifest](run.md#the-run-manifest) to `manifest.json`.
`facts.jsonl` is a predictions file for [`odke eval`](evaluation.md).
`append(kg)` adds a graph without reading the files (or merges, with
`merge=True`): a [streamed run](run.md#streaming) appends each micro-batch, and
a reader keeps a fact's last line. RDF, `cypher_file` and `neo4j_admin_csv`
rewrite the whole graph (`streams = False`), so a streamed run refuses them.

## Your own sink

Anything with `write(kg)` is a sink ([DECISIONS #5](decisions.md#5)).
`plan(kg, ontology=...)` and `provenance_of(fact, extracted_at)` in
`openodke.sinks.neo4j` give the shape above. A store that does a stage itself
declares `profile = PlatformProfile(...)`
([Concepts](concepts.md#platformprofile-and-delegated)).
`openodke.eval.assert_idempotent(sink, kg, counts)` writes twice and checks
that nothing moved.
