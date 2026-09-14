# Ontology

An `Ontology` is the schema the extractor is held to: named `types` (each an
`EntityType`) and `predicates` (each a `Predicate`). It is supplied by you as a
dict, JSON, YAML or pydantic models, or later inferred from a corpus. Either way
every downstream stage sees this one shape. Everything in `odke.ontology` is
deterministic and needs no model.

| Model | Fields |
|---|---|
| `Ontology` | `name`, `version`, `types`, `predicates`, `inferred` |
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
```

`validate` takes several files because that is how pre-commit calls it, and
warnings print without failing.
