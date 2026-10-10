# Ontology

An `Ontology` is the schema facts are held to: named `types` and `predicates`.
Load it from a dict, JSON, YAML, pydantic models, OWL or a live Neo4j graph;
every stage sees the same shape. Loading, validating and diffing need no model.

## Fields

| Model | Field | Default | Meaning |
|---|---|---|---|
| `Ontology` | `name`, `version` | `"untitled"`, `"0"` | |
| | `types`, `predicates` | empty | keyed by name; an entry's `name` defaults to its key |
| | `inferred`, `frozen_at`, `frozen_by` | `False`, `None`, `None` | set by [inference](#drafting-one-parked) and [`freeze()`](#freezing) |
| `EntityType` | `description`, `parents`, `aliases` | | |
| | `keys` | `()` | the predicates that together name the thing |
| `Predicate` | `domain` | `()`, meaning every type | the types it applies to |
| | `range` | `"string"` | an entity type makes an **edge**; `string`, `integer`, `number`, `float`, `boolean`, `date`, `datetime` or `quantity` (a number with its unit, normalised to [one unit per dimension](resolution-and-corroboration.md#normalise)) makes a **property** |
| | `cardinality` | `"single"` | or `"multi"` |
| | `cardinality_scope` | `()` | see [Cardinality and its scope](#cardinality-and-its-scope) |
| | `qualifiers` | `{}` | name to `Qualifier`; a bare list of names is all reconcilable |
| | `required` | `False` | every entity in the domain should hold a value |
| | `importance` | 0.5 | ranks the predicate in [snippets](reference-extractor.md#what-the-model-is-shown) |
| | `label`, `description`, `aliases`, `examples` | | |
| | `inverse_of`, `symmetric` | `None`, `False` | see [Inverse and symmetric predicates](#inverse-and-symmetric-predicates) |
| `Qualifier` | `identity` | `False` | `True` puts the qualifier in `Fact.signature` ([DECISIONS #15](decisions.md#15)) |

## Loading

```python
from openodke import Ontology

ontology = Ontology.from_dict(
    {
        "name": "saas-vendors",
        "types": {
            "Vendor": {"description": "A company that sells software.", "keys": ["legal_name"]},
            "Service": {"description": "A product a vendor runs."},
        },
        "predicates": {
            "legal_name": {"domain": ["Vendor"], "required": True},
            "operates": {"domain": ["Vendor"], "range": "Service", "cardinality": "multi"},
            "price": {
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

| Source | Call | Extra |
|---|---|---|
| a dict | `Ontology.from_dict(data)` | — |
| JSON | `Ontology.from_json(source)` | — |
| YAML | `Ontology.from_yaml(source)` | `yaml` |
| pydantic models | `Ontology.from_pydantic(*models, name=...)` | — |
| OWL, RDFS or SKOS | `Ontology.from_owl(source, *, format=None, language="en")` | `rdf` |
| a live Neo4j graph | `Ontology.from_neo4j(driver_or_uri, *, auth=None, database=None)` | `neo4j` |

`from_json` and `from_yaml` read a `str` as the document itself when it contains
a newline or starts with `{`, `[`, `---`, `#` or `%`, and as a path otherwise.

### Strict loading

Every loader but `from_pydantic`, which is always strict, takes `strict=True` by
default: it runs `validate()` and raises `OntologyLoadError`, a `ValueError`, on
any **error**. Each problem is one line naming the key by its dotted path, and
`exc.problems` lists them. With `strict=False`, anything well-formed loads; the
two importers then warn once with `OntologyImportWarning`, whose `.problems`
holds the same lines.

```python
from openodke import OntologyLoadError

try:
    Ontology.from_dict({"predicates": {"employer": {"rnage": "Company"}}})
except OntologyLoadError as exc:
    print(exc)
# predicates.employer.rnage: unknown key — did you mean 'range'?
```

### From pydantic models

Each model is a type and each field a predicate. A field typed as another model
you passed is an edge; `list[X]`, `tuple[X, ...]` and `set[X]` are `multi`;
`X | None` or a field with a default is not `required`. A field that cannot be
mapped, such as `int | str`, raises.

```python
from datetime import date

from pydantic import BaseModel


class Company(BaseModel):
    legal_name: str


class Person(BaseModel):
    name: str
    born: date | None = None
    employer: list[Company] = []


people = Ontology.from_pydantic(Person, Company, name="people")
employer = people.predicates["employer"]
assert (employer.domain, employer.range, employer.cardinality) == (("Person",), "Company", "multi")
```

### Importing a schema you already have

`from_owl` reads a path, a document or an rdflib `Graph`. What it cannot carry
over, such as restrictions, sub-properties, equivalence or union ranges, is
reported by the subject it was found on. `owl:imports` is not followed.

| RDF | Ontology |
|---|---|
| `owl:Class`, `rdfs:Class`, `skos:Concept` | a type, named by the IRI's local name |
| `rdfs:subClassOf`, `skos:broader` | `parents` |
| `owl:hasKey` | `keys` |
| `owl:ObjectProperty`, `owl:DatatypeProperty` | an edge, a property |
| `rdfs:domain`, `rdfs:range` | `domain`, `range` |
| `owl:FunctionalProperty` | `single`; every other property is `multi` |
| `owl:inverseOf`, `owl:SymmetricProperty` | `inverse_of`, `symmetric` |
| `rdfs:label`, `rdfs:comment`, `skos:altLabel` | `label`, `description`, `aliases` |

```python
import warnings

from openodke import OntologyImportWarning

hr_owl = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix : <https://example.org/hr#> .

:Person a owl:Class .
:Company a owl:Class .
:employer a owl:ObjectProperty, owl:FunctionalProperty ; rdfs:domain :Person ; rdfs:range :Company .
:manages a owl:ObjectProperty, owl:TransitiveProperty ; rdfs:domain :Person ; rdfs:range :Person .
"""
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    hr = Ontology.from_owl(hr_owl, name="hr", strict=False)
(warning,) = caught
assert warning.category is OntologyImportWarning
print(warning.message.problems[0])
# :manages: is a owl:TransitiveProperty, which the ontology model cannot express
assert hr.predicates["employer"].cardinality == "single"
```

`from_neo4j` reads a graph's own schema and writes nothing. A label is a type, a
relationship type is an edge (or a property when it ends at `:Claim` nodes),
and a node property is a property. `importance` comes from how often each is
used. A graph [`Neo4jSink`](stores.md#neo4j) wrote reads back as the shape it
wrote. Labels have no hierarchy, so there are no `parents`; every edge is
`multi`, and every qualifier reconcilable.

## Validate

`ontology.validate()` returns a list of `Diagnostic(code, path, message,
severity)` and never raises. An **error** produces wrong or missing facts, and
strict loading refuses it. A **warning** marks a schema that works but is
probably not what was meant.

| Code | Severity | Found when |
|---|---|---|
| `name-mismatch` | error | an entry's `name` differs from its key |
| `unknown-parent` | error | a type's parent is not a type |
| `unknown-key` | error | a type's `keys` names a predicate that does not exist |
| `key-outside-domain` | error | a type's key predicate never applies to that type |
| `unknown-range` | error | a range is neither a type nor a literal type |
| `unreachable-predicate` | error | nothing in a predicate's domain is a type, so no snippet includes it |
| `scope-undeclared-qualifier` | error | `cardinality_scope` names a key that is not a qualifier |
| `scope-not-identity` | error | `cardinality_scope` names a reconcilable qualifier |
| `duplicate-alias` | error | one alias points at two types, or two predicates |
| `unknown-inverse` | error | `inverse_of` names a predicate that does not exist |
| `inverse-not-edge` | error | an end of an inverse, or a symmetric predicate, is a property |
| `inverse-not-mutual` | error | an inverse does not name its partner back |
| `inverse-domain-range` | error | an inverse's ends do not swap |
| `unknown-domain` | warning | part of a domain is not a type |
| `inheritance-cycle` | warning | types inherit from each other |
| `unreviewed` | warning | the schema is still marked `inferred` |

## Inverse and symmetric predicates

`inverse_of` names the edge that states a predicate the other way round, and
`symmetric: true` marks an edge that is its own inverse. The pipeline adds each
fact's partner without a model call
([Concepts](concepts.md#inverse-and-symmetric-partners),
[DECISIONS #28](decisions.md#28)). Declare an inverse on one side; loading fills
in the other. A predicate's range must sit inside its inverse's domain.

```python
geo = Ontology.from_yaml("""
types: {Place: {}, Person: {}}
predicates:
  located_in: {domain: [Place], range: Place, inverse_of: contains}
  contains: {domain: [Place], range: Place, cardinality: multi}
  spouse: {domain: [Person], range: Person, symmetric: true}
""")
assert geo.inverses == {"located_in": "contains", "contains": "located_in", "spouse": "spouse"}
```

## Diff

`old.diff(new)` returns a `SchemaChange(kind, path, breaking, detail)` for
every difference.

- **Breaking:** a removed type, predicate or qualifier; a narrowed range or
  domain; a changed cardinality; a scope that loses a key; a predicate that
  becomes required; changed entity keys; a qualifier whose `identity` flips, or
  that is added as identity-bearing; an `inverse_of` or `symmetric` removed or
  changed.
- **Compatible:** a widened range (to an ancestor type, or `integer` to
  `number`), a new inverse, and every documentation change.

```python
employs = {"domain": ["Person"], "range": "Company"}
types = {"Person": {}, "Company": {}}
multi = {**employs, "cardinality": "multi"}
old = Ontology.from_dict({"types": types, "predicates": {"employer": multi}})
new = Ontology.from_dict({"types": types, "predicates": {"employer": employs}})
for change in old.diff(new):
    print(change)
# breaking   changed predicates.employer.cardinality: 'multi' → 'single' (subjects already holding several values now conflict)
```

## Cardinality and its scope

`cardinality` says how many values a subject may hold; `cardinality_scope` says
within what. The scope always includes the identity-bearing qualifiers, so with
an identity-bearing `tier`, a `single` price means one price per subject per
tier. Declared scope keys must be identity-bearing qualifiers.
`Predicate.scope_keys` is the resolved tuple, and the
[Neo4j cardinality check](stores.md#check) groups by it.

```python
price = ontology.predicates["price"]
assert (price.cardinality, price.cardinality_scope, price.scope_keys) == ("single", (), ("tier",))
```

## Freezing

`ontology.freeze(by=...)` returns a copy with `inferred` cleared and
`frozen_at` and `frozen_by` set. It raises `OntologyFreezeError` while
`validate()` reports an error, and `ValueError` for an empty `by`. After that,
`diff` reports an edit like any other.

```python
frozen = ontology.freeze(by="Ada Lovelace")
assert (frozen.inferred, frozen.frozen_by) == (False, "Ada Lovelace")
```

`ontology.fingerprint` is the SHA-256 of the schema's content: its types and
predicates as canonical JSON. `name`, `version` and the review fields are not in
it, so a frozen or relabelled copy keeps it, and any change to a type or a
predicate gives a new one. Two ontologies with one fingerprint check every fact
alike.

```python
assert frozen.fingerprint == ontology.fingerprint and len(ontology.fingerprint) == 64
```

## Drafting one (parked)

For a corpus with no schema, `odke ontology infer` drafts one. Record shapes,
Hearst patterns and co-occurrence propose types and predicates, each with the
spans behind it. Unless `--no-llm` is given, one call through the `infer` model
role then names and merges them. The draft is marked `inferred=True` and
carries its evidence as YAML comments. Until it is frozen, `validate()` warns
`unreviewed` and `Neo4jSink` warns `UnreviewedOntologyWarning`. Nothing infers
on its own: `odke run` refuses an inferrer ([DECISIONS #8](decisions.md#8)).
Inference is parked: it works, and is not being extended.

```bash
odke ontology infer corpus/ --out draft.yaml --no-llm   # no model, no key
odke ontology validate draft.yaml                        # warns: unreviewed
# review and edit draft.yaml
odke ontology freeze draft.yaml --by "Ada Lovelace"      # refuses while there is an error
```

## From the shell

These commands load non-strictly, so they can show what is wrong. `.yaml` and
`.yml` files are read as YAML, anything else as JSON.

```bash
odke ontology validate schema.json other.yaml    # exit 1 if any file has an error
odke ontology diff old.json new.json --fail-on-breaking
odke ontology types schema.json                  # each type and how many predicates it can carry
odke ontology snippet schema.json Person         # the prompt fragment for one type
```
