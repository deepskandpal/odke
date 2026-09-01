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

## End-to-end extraction

Arrives with M2. See [ROADMAP.md](../ROADMAP.md).
