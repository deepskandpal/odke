# Ontology

An `Ontology` is the schema the extractor is held to: named `types` (each an
`EntityType`) and `predicates` (each a `Predicate`). You supply it as a dict, JSON,
YAML or pydantic models; import it from OWL, RDFS, SKOS or a live Neo4j graph; or
draft it from a corpus by [inference](inference.md) and freeze it once reviewed.
Either way every downstream stage sees this one shape. Loading, validating and
diffing are deterministic and need no model.

| Model | Fields |
|---|---|
| `Ontology` | `name`, `version`, `types`, `predicates`, `inferred`, `frozen_at`, `frozen_by` |
| `EntityType` | `name`, `description`, `parents`, `keys` (the predicates that together name the thing), `aliases` |
| `Predicate` | `name`, `label`, `description`, `domain`, `range` (default `"string"`), `cardinality` (`single` or `multi`), `cardinality_scope`, `required`, `qualifiers`, `aliases`, `importance` (default 0.5), `examples` |
| `Qualifier` | `identity` (default `False`), `description` |

A predicate is an **edge** when its `range` names an entity type, and a
**property** when its range is a literal type: `string`, `integer`, `number`,
`float`, `boolean`, `date` or `datetime`. An empty `domain` is open: the
predicate applies to every type. Inside `types` and `predicates`, an entry's
`name` defaults to its key, so you never have to write it twice.

## Loading

```python
from odke import Ontology

ontology = Ontology.from_dict(
    {
        "name": "saas-vendors",
        "version": "2",
        "types": {
            "Vendor": {"description": "A company that sells software.", "keys": ["legal_name"]},
            "Service": {"description": "A product a vendor runs."},
        },
        "predicates": {
            "legal_name": {"domain": ["Vendor"], "required": True},
            "operates": {"domain": ["Vendor"], "range": "Service", "cardinality": "multi"},
            "price": {
                "description": "List price per seat per month.",
                "domain": ["Service"],
                "range": "number",
                "qualifiers": {"tier": {"identity": True}, "as_of": {}},
            },
        },
    }
)

assert ontology.predicates["operates"].is_edge_in(ontology)
assert ontology.identity_keys("price") == ("tier",)
```

`Ontology.from_json(source)` and `Ontology.from_yaml(source)` take a file path or
the document itself. A `str` counts as the document when it contains a newline or
starts the way JSON or YAML starts (`{`, `[`, `---`, `#`, `%`); any other string
is a path. YAML needs the `yaml` extra and nothing else does.

```python
vendors = Ontology.from_yaml("""
name: saas-vendors
types:
  Service: {description: A product a vendor runs.}
predicates:
  uptime:
    domain: [Service]
    range: number
    qualifiers:
      percentile: {identity: true}
      region: {identity: true}
      as_of: {}
    cardinality_scope: [percentile, region]
""")

assert vendors.predicates["uptime"].scope_keys == ("percentile", "region")
```

A bare list of qualifier names, such as `qualifiers: [start_date, end_date]`, is
also accepted, and every name in it is reconcilable.

### When loading fails

Load errors are `OntologyLoadError`, which is a `ValueError`. Each problem is one
line naming the offending key by its dotted path and saying what was found there.
A misspelt key gets a suggestion, and a syntax error carries a line and column.
The same lines are available individually as `exc.problems`.

```python
from odke import OntologyLoadError

try:
    Ontology.from_dict({"predicates": {"employer": {"rnage": "Company"}}})
except OntologyLoadError as exc:
    print(exc)
# predicates.employer.rnage: unknown key — did you mean 'range'?
```

Loading is `strict` by default: it also runs `validate()` and refuses a schema
with **errors**. Warnings never block. `strict=False` loads anything well-formed,
which is what a tool that wants to *show* the diagnostics needs.

```python
schema = {
    "types": {"Person": {}, "Company": {}},
    "predicates": {"employer": {"domain": ["Person"], "range": "Compnay"}},
}
try:
    Ontology.from_dict(schema)
except OntologyLoadError as exc:
    print(exc.problems[0])
# predicates.employer.range: 'Compnay' is neither an entity type nor a literal type
# (boolean, date, datetime, float, integer, number, string) — did you mean 'Company'? [unknown-range]

loose = Ontology.from_dict(schema, strict=False)
assert [(d.severity, d.code, d.path) for d in loose.validate()] == [
    ("error", "unknown-range", "predicates.employer.range")
]
```

## From pydantic models

If your entities are already pydantic models, `Ontology.from_pydantic(*models)`
reads them rather than making you write the schema a second time:

- Each model is an entity type, and its docstring is the description.
- Each field is a predicate. A field typed as another model you passed is an
  edge to it.
- `list[X]`, `tuple[X, ...]` and `set[X]` are `cardinality="multi"`.
- `X | None`, or a field with a default, is not `required`.
- A field's description becomes the predicate's description, and its title
  becomes the label.
- A model that subclasses another model you passed names it as a parent, and
  does not redeclare the fields it inherits.
- `str`, `int`, `float`, `Decimal`, `bool`, `date`, `datetime` and `UUID` map to
  literal ranges, as do enums and `Literal`s of one of those.

```python
from datetime import date

from pydantic import BaseModel, Field


class Company(BaseModel):
    """An incorporated organisation."""

    legal_name: str


class Person(BaseModel):
    """A human being."""

    name: str = Field(description="Full name as written in the source.")
    born: date | None = None
    employer: list[Company] = []


people = Ontology.from_pydantic(Person, Company, name="people")
employer = people.predicates["employer"]
assert (employer.domain, employer.range, employer.cardinality) == (("Person",), "Company", "multi")
assert people.predicates["name"].required and not people.predicates["born"].required
```

A field that cannot be mapped is reported as `Model.field`, and the whole
conversion raises. A silently dropped field would be a predicate the extractor is
never asked about.

```python
class Ambiguous(BaseModel):
    identifier: int | str


try:
    Ontology.from_pydantic(Ambiguous)
except OntologyLoadError as exc:
    print(exc)
# Ambiguous.identifier: int | str is a union — an ontology range is one type; split it into
# two fields or pick one
```

## Validate

`ontology.validate()` returns a list of `Diagnostic(code, path, message,
severity)` and never raises. It returns a list rather than a bool because
"invalid" is not something you can act on, and a message naming the key and
suggesting a fix is. The line between the two severities is whether extraction
goes wrong. An **error** produces wrong or missing facts: strict loading refuses
it, and `odke ontology validate` exits non-zero. A **warning** is a schema that
works but probably is not what was meant.

| Code | Severity | Found when |
|---|---|---|
| `name-mismatch` | error | an entry's `name` differs from its key |
| `unknown-parent` | error | a type's parent is not a type |
| `unknown-key` | error | a type's `keys` names a predicate that does not exist |
| `key-outside-domain` | error | a type's key predicate never applies to that type |
| `unknown-range` | error | a range is neither a type nor a literal type |
| `unreachable-predicate` | error | no entry in a predicate's domain is a type, so no snippet includes it |
| `scope-undeclared-qualifier` | error | `cardinality_scope` names a key that is not a qualifier |
| `scope-not-identity` | error | `cardinality_scope` names a reconcilable qualifier |
| `duplicate-alias` | error | one alias points at two types, or two predicates |
| `unknown-domain` | warning | part of a domain is not a type; the rest still works |
| `inheritance-cycle` | warning | types inherit from each other; `lineage()` tolerates it |
| `unreviewed` | warning | the schema is still marked `inferred`: nobody has reviewed and [frozen](inference.md#freezing) it |

## Diff

Once a graph is live, an edited ontology is a migration, and the question to ask
of it is whether existing data and queries still fit. `old.diff(new)` returns
`SchemaChange(kind, path, breaking, detail)` for every difference, each marked
breaking or compatible.

Breaking changes are: a removed type, predicate or qualifier; a narrowed range or
domain; a changed cardinality; a cardinality scope that loses a key; a predicate
that becomes required; changed entity keys; and a qualifier whose `identity` flips
or that is added as identity-bearing, because either one re-partitions
`Fact.signature`. Widening a range (to an ancestor type, or `integer` to
`number`) and every documentation change are listed as compatible.

```python
old = Ontology.from_dict(
    {
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company", "cardinality": "multi"},
            "uptime": {"range": "number", "qualifiers": {"percentile": {"identity": True}}},
        },
    }
)
new = Ontology.from_dict(
    {
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company", "description": "Who pays them."},
            "uptime": {"range": "number", "qualifiers": {"percentile": {"identity": False}}},
        },
    }
)
for change in old.diff(new):
    print(change)
# breaking   changed predicates.employer.cardinality: 'multi' → 'single' (subjects already holding several values now conflict)
# compatible changed predicates.employer.description: None → 'Who pays them.'
# breaking   changed predicates.uptime.cardinality_scope: (percentile) → () (no longer unique per percentile, so existing values may conflict)
# breaking   changed predicates.uptime.qualifiers.percentile.identity: True → False (fact signatures change, so existing claims split or merge)

assert sum(c.breaking for c in old.diff(new)) == 3
```

## Cardinality and its scope

`cardinality` says how many values a subject may hold; `cardinality_scope` says
*within what*. "One price per subject" and "one price per subject per tier" are
both `single`. A flat count would treat the second as a stream of contradictions,
and two in five facts in a real corpus were qualifier-scoped.

The scope is a list of qualifier keys that is always unioned with the
identity-bearing ones. A fact that differs on an identity-bearing qualifier is a
different claim, so two such facts can never conflict. The default, `()`,
therefore already means "one value per subject per identity key", and there is no
flat setting to get wrong. Declaring keys makes the scope visible in the schema.
`validate()` requires every declared key to be an identity-bearing qualifier,
because a reconcilable one cannot separate one value from another.
`Predicate.scope_keys` is the resolved, sorted tuple. It is what the
[Neo4j cardinality check](neo4j.md#what-neo4j-cannot-enforce) groups by, so the
schema and the store cannot disagree about what counts as a conflict.

```python
price = ontology.predicates["price"]
assert (price.cardinality, price.cardinality_scope, price.scope_keys) == ("single", (), ("tier",))

reconcilable_scope = {
    "types": {"Service": {}},
    "predicates": {
        "price": {
            "domain": ["Service"],
            "qualifiers": {"as_of": {}},
            "cardinality_scope": ["as_of"],
        }
    },
}
codes = [d.code for d in Ontology.from_dict(reconcilable_scope, strict=False).validate()]
assert codes == ["scope-not-identity"]
```

## Snippets: what the model is shown

The model is never shown the whole schema. For each entity type it sees an
`OntologySnippet`: the predicates whose domain covers that type (inherited ones
included), ranked by `importance` and then by name, and cut at `limit` (25 by
default). That cut is how the prompt stays a fixed size as the ontology grows. A
snippet is data, rendered two ways from one object: `render()` produces prose for
a prompt and `json_schema()` produces a schema for structured output, so the two
cannot drift apart ([DECISIONS #6](decisions.md)).

```python
snippet = ontology.snippet("Service")
print(snippet.render())
# Entity type: Service
# Description: A product a vendor runs.
# Properties you may extract:
# - price (number, single): List price per seat per month. [qualifiers: tier, as_of]

assert snippet.json_schema()["properties"]["price"]["type"] == "number"
```

## Importing a schema you already have

Plenty of schemas already exist, as OWL files or as the graph a database already
holds, and writing them again as JSON is how two copies drift. Two importers read
them directly. Both report everything they could not carry over by where it was
found, and both share one rule for what happens next:

- **`strict=True`** (the default) raises `OntologyLoadError` listing every problem,
  as `from_pydantic` does, and refuses a result `validate()` finds errors in. A
  dropped axiom is a rule the extractor is silently never held to.
- **`strict=False`** loads whatever maps, and the same problems arrive as one
  `OntologyImportWarning`, whose `.problems` holds the lines `OntologyLoadError`
  would have raised with. Choosing to load a large public ontology anyway never
  means choosing not to be told.

### From OWL, RDFS and SKOS

`Ontology.from_owl(source, *, format=None, name=None, version=None, language="en",
strict=True)` reads a file path, a document, or an rdflib `Graph`, and needs the
`rdf` extra. `format` is any rdflib parser name; left out, it is guessed from the
file suffix or from the document (RDF/XML, JSON-LD, else Turtle, which also reads
N-Triples).

| RDF | Ontology |
|---|---|
| `owl:Class`, `rdfs:Class`, `skos:Concept`, and anything used as a class by `rdfs:subClassOf`, `skos:broader`, a domain or an object range | an `EntityType`, named by the IRI's local name |
| `rdfs:subClassOf`, `skos:broader`, and `skos:narrower` read backwards | `parents` |
| `owl:hasKey` | `keys` |
| `owl:ObjectProperty` | an edge predicate to its `rdfs:range` |
| `owl:DatatypeProperty` | a literal predicate, its XSD range mapped to a literal type |
| a bare `rdf:Property` | an edge when its range is a class, and a literal otherwise |
| `rdfs:domain`, or a domain that is an `owl:unionOf` | `domain`, which is already a union |
| `owl:FunctionalProperty` | `cardinality="single"`. Every other property is `multi`, because OWL's open world lets a property hold any number of values unless it says otherwise. |
| `rdfs:label` / `skos:prefLabel` | a predicate's `label` |
| `rdfs:comment` / `skos:definition` | `description`; a type with no comment takes its label when the label says more than the name |
| `skos:altLabel`, `skos:hiddenLabel` | `aliases` |
| the `owl:Ontology`'s `rdfs:label` and `owl:versionInfo` | `name` and `version`, unless you pass them |

Text tagged with `language` wins, then untagged text. `importance` stays at its
default: nothing in an OWL file says how often a predicate is used.

What the model cannot hold is reported by the subject it was found on:
restrictions, property characteristics other than functional, inverse and
sub-properties, equivalence and disjointness, union ranges, unmapped datatypes,
individuals, and imports. `owl:imports` is not followed; parse the imported
ontology into the same rdflib `Graph` and pass the `Graph`. Annotations outside
the OWL, RDF, RDFS and SKOS vocabularies, such as Dublin Core or `rdfs:seeAlso`, are
documentation and are not reported.

```python
import warnings

from odke import OntologyImportWarning

hr_owl = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <https://example.org/hr#> .

:Person a owl:Class ; rdfs:comment "A human being." .
:Company a owl:Class ; owl:hasKey ( :legal_name ) .
:legal_name a owl:DatatypeProperty ; rdfs:domain :Company ; rdfs:range xsd:string .
:employer a owl:ObjectProperty, owl:FunctionalProperty ;
    rdfs:label "Employer" ; rdfs:domain :Person ; rdfs:range :Company .
:born a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:date .
:manages a owl:ObjectProperty ; owl:inverseOf :managedBy ; rdfs:domain :Person ; rdfs:range :Person .
"""
try:
    Ontology.from_owl(hr_owl)
except OntologyLoadError as exc:
    print(exc.problems)
# (':manages: owl:inverseOf is not supported',)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    hr = Ontology.from_owl(hr_owl, name="hr", strict=False)
(warning,) = caught
assert warning.category is OntologyImportWarning
assert warning.message.problems == (":manages: owl:inverseOf is not supported",)

employer = hr.predicates["employer"]
assert (employer.domain, employer.range, employer.cardinality, employer.label) == (
    ("Person",),
    "Company",
    "single",
    "Employer",
)
assert (hr.predicates["born"].range, hr.predicates["born"].cardinality) == ("date", "multi")
assert hr.types["Company"].keys == ("legal_name",)
```

`RdfSink` writes this same vocabulary when it is given an ontology, so a schema it
wrote reads back ([Sinks](sinks.md#rdf)).

### From a live Neo4j graph

`Ontology.from_neo4j(source, *, auth=None, database=None, name=None, version="0",
strict=True)` reads the schema of a graph you already have. `source` is a driver,
or a URI to connect to with `auth`, which needs the `neo4j` extra. It calls three
procedures present and not deprecated in Neo4j 5, `db.schema.nodeTypeProperties()`,
`db.schema.relTypeProperties()` and `db.schema.visualization()`, plus counts.
Nothing is written. The two property procedures read the store, and are not free on
a large graph.

- A **label** is an entity type.
- A **relationship type** is an edge predicate to the label it ends at, or a literal
  predicate when it ends at `:Claim` nodes. Its properties that are not provenance
  become qualifiers.
- A **node property** is a literal predicate on the labels that hold it. Its range
  is read from the value types Neo4j reports; it is `multi` when it holds lists,
  and `required` when every node of those labels has it.
- **`importance` comes from counts**: relationships per type and nodes holding each
  property, log-scaled against the most-used predicate. That is the frequency
  signal the paper ranks snippets by, so a snippet puts the graph's most-used
  predicates first.

A graph `Neo4jSink` wrote reads back as the shape it wrote: `:Entity` and `:Claim`
are the sink's labels, not types; the entity fields and provenance properties are
not predicates or qualifiers; a projected property and the claims behind it are one
predicate; and the `SAME_AS`, `SIMILAR` and `DIFFERENT` links are not predicates.

What the store cannot say is not invented. Labels have no hierarchy, so there are
no parents. Edges are `multi`, because how many one node holds is not in the
schema. Qualifiers are reconcilable, because whether one bears identity is a
decision ([DECISIONS #15](decisions.md)) that no count reveals. A value type with no
literal range, a relationship that ends at several labels (read as the most-used
one), and one name used by both a relationship and a property are reported.

<!-- docs: no-run -->
```python
import os

live = Ontology.from_neo4j(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    database="neo4j",
    strict=False,
)
print(live.snippet("Company").render())  # the graph's most-used predicates first
```

## Inferring a draft

A corpus with no schema at all can get a draft: `Ontology.infer(documents)`, or
`odke ontology infer corpus/ --out draft.yaml`. Deterministic proposers find
candidate types and predicates with the spans that produced them, a model
optionally names and ranks them, and the result is marked `inferred=True` for a
person to review, edit and `freeze()`. It is a bootstrap, never a mode
([DECISIONS #8](decisions.md)). [Ontology inference](inference.md) covers it end to
end.

## From the shell

The CLI loads `.yaml` / `.yml` files as YAML and everything else as JSON. These
commands never load strictly, because they exist to show you what is wrong.

```bash
odke ontology validate schema.json other.yaml    # every diagnostic; exit 1 if any file has an error
odke ontology diff old.json new.json             # breaking changes first
odke ontology diff old.json new.json --fail-on-breaking   # ...and exit 1 if there are any
odke ontology types schema.json                  # each type and how many predicates it can carry
odke ontology snippet schema.json Person         # the exact prompt fragment
odke ontology snippet schema.json Person --json-schema --limit 10
odke ontology infer corpus/ --out draft.yaml --no-llm   # a draft for review: see Ontology inference
odke ontology freeze draft.yaml --by "Ada Lovelace"      # after review; refuses while there are errors
```

`validate` takes several files because that is how pre-commit calls it, and
warnings print without failing.
