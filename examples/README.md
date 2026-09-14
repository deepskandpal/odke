# Examples

## Inspect a snippet without spending a token

The single most useful debugging step: see exactly what the extractor will be
prompted with for a given entity type.

```bash
odke ontology types examples/people.ontology.json
odke ontology snippet examples/people.ontology.json Scientist
odke ontology snippet examples/people.ontology.json Scientist --json-schema
```

`Scientist` inherits every `Person` predicate, so its snippet is larger than its
own two lines of schema suggest — which is exactly the sort of thing worth seeing
before wondering why an extraction came back with a field you did not define.

## End-to-end: text and tables to a queryable graph

[`e2e/`](e2e/README.md) runs a small invented corpus through all thirteen stages
into JSON Lines or Neo4j, on recorded model responses, and ends with the Cypher
that shows provenance, a `DIFFERENT` link and a cardinality check.

```bash
odke run examples/e2e/odke.yaml
```

## Every key of a run config

[`run.yaml`](run.yaml) is the same run with every key commented: inputs and
loaders, model roles, recorded responses, the thirteen stages and their options,
sinks and `bootstrap`.
