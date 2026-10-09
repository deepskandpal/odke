# The Validator

openodke is the verification layer between any extractor and the graph store
([DECISIONS #24](decisions.md)). The Validator is that layer as one class,
`openodke.Validator`, and one command, `odke validate`. You give it another
extractor's triples and the texts they came from. It grounds, normalises,
resolves, corroborates, gates and writes them, and reports what each step did.
`odke ground` is the grounding step on its own ([Grounding](grounding.md#a-graph-from-somewhere-else-odke-ground)).

## One command

The command takes the same facts `odke ground` takes. Here it is on the five
triples in `examples/triples/`, with recorded answers and no key:

```bash
odke validate --facts examples/triples/triples.jsonl --texts examples/triples/texts \
  --ontology examples/triples/ontology.json --config examples/triples/odke.yaml -o out
```

```text
odke validate
in            5 facts from 1 text
grounded      supported 3, contradicted 0, not_found 2, unchecked 0
free checks   1 refused: 1 not in the text
refused       0 by the gate
merged        0 into a fact with the same signature
linked        0 entity links
derived       0 inverse and symmetric partners
out           5 facts (4 edges, 1 property), 5 entities
cost          4 model calls, 657 tokens, USD unknown (ground.span@1)
coverage      0 of 1 sentences naming two known entities uncovered, 0 entities in no fact, relations offered: unknown, 0 unused
wrote         jsonl → out: entities.jsonl 5, facts.jsonl 5, links.jsonl 0, manifest.json
```

The two `not_found` facts are written, because the default gate refuses only
what the text contradicts (see the stage table below). To keep them out, pass
`VerdictGate(refuse_not_found=True, schema=True)`.

`odke validate --config run.yaml` runs a run config instead, provided its
extractor is `triples`. The config gives the inputs, the ontology, the models,
the stages and the sinks, and a stage it leaves out gets the Validator's
default, not the pass-through. With `--facts`, `--config` is read only for its
`models` block, as `odke ground` reads it. `-o` adds a JSONL sink to whatever
sinks the config names.

`--dry-run` calls no model and writes nothing. It runs the free checks, the
locator with `--locate`, and every deterministic stage, then prints what
would be written. Exit 2 is input or a config that can't be read; exit 1 is a
run that failed.

## In Python

```python
from openodke import Document, Ontology, Validator
from openodke.llm import RecordedClient

places = Ontology.from_dict(
    {
        "types": {"Region": {}, "Country": {}, "Company": {}},
        "predicates": {
            "located_in": {"domain": ["Region"], "range": "Country", "inverse_of": "contains"},
            "contains": {"domain": ["Country"], "range": "Region"},
            "founded": {"domain": ["Company"], "range": "integer"},
        },
    }
)
a = Document(id="a", text="Brittany is in France. Halden Robotics was founded in 2014.")
b = Document(id="b", text="Brittany lies in France. Halden Robotic opened in Lyon.")
halden = {"doc": "a", "subject": "Halden Robotics", "predicate": "founded"}
rows = [
    {"doc": "a", "subject": "Brittany", "predicate": "located_in", "object": "France"},
    {"doc": "b", "subject": "Brittany", "predicate": "located_in", "object": "France"},
    {**halden, "object": 2012, "quote": "Halden Robotics was founded in 2014."},
    {**halden, "object": 2014, "quote": "Halden Robotics was founded in 2014."},
    {"doc": "b", "subject": "Halden Robotic", "subject_type": "Company",
     "predicate": "opened_in", "object": "Lyon"},
]  # fmt: skip
# Recorded answers, so this runs with no key. Drop `client=` to call a model.
client = RecordedClient(
    [
        {"match": "— 2012.", "response": {"verdict": "contradicted"}},
        {"match": "Claim:", "response": {"verdict": "supported"}},
    ]
)
kg, report = Validator(places, client=client).validate(rows, [a, b])
print(report.facts_in, report.refused, report.merged, report.linked, report.derived)
# 5 2 2 1 2
print(report.refused_by, report.calls)
# {'contradicted': 1, 'predicate': 1} 4
```

The run above did five things:

- **Refused two facts.** The text contradicts the 2012 founding year, and
  `opened_in` isn't in the ontology. The free checks refused the second, so it
  never cost a call: four calls for five facts.
- **Merged two.** Brittany's `located_in` is stated in both texts, so it is one
  fact with `support` 2.
- **Derived two.** The ontology says `contains` is the inverse of
  `located_in`, so France contains Brittany, once from each text, and the two
  copies merge.
- **Linked one.** The misspelt `Halden Robotic` is a `SIMILAR` link to
  `Halden Robotics`, not a merge ([DECISIONS #16](decisions.md)).
- **Returned the graph and a report.** `validate()` returns the
  `KnowledgeGraph` and a `ValidationReport`, and writes the graph to every
  sink in `sinks=`.

`validate(rows, documents)` takes the following inputs:

- any adapter's `(rows, documents)` ([Triples](triples.md#from-other-libraries));
- a triples file and the texts it cites;
- `Fact`s and their texts.

`dry_run=True` is the command's dry run. A `Validator` keeps its stages between
calls, and each report counts its own call.

## The stages

Each stage defaults to what `odke run` builds under its usual name, with that
name's default options:

| Stage | Default |
|---|---|
| grounder | `LLMGrounder(roles, client=client, locate=locate)`, behind the free checks (`CheckedGrounder`) |
| normalizer | `ValueNormalizer(ontology)` |
| resolver | `NativeResolver()` |
| corroborator | `SignatureCorroborator(ontology)` |
| scorer | `EvidenceScorer()` |
| gate | `VerdictGate(schema=True)`: refuses what the text contradicts, and what the free checks refused |
| inverses | on when the ontology declares a pair ([DECISIONS #28](decisions.md)); `inverses=False` turns it off |
| coverage | on: what extraction left behind ([Grounding](grounding.md#what-extraction-left-behind-the-coverage-report)); `coverage=False` turns it off |
| sinks | none |

Passing a stage replaces its default, and a pass-through from `openodke.stages`
turns that stage off. A grounder you pass in still runs behind the free checks.
The default gate has `schema=True` because the free checks don't change a
fact's verdict, so a gate that read only verdicts would write a fact the
ontology has no room for. The ontology is optional; without one, the checks on
the relation and the types have nothing to check.

## The report

`ValidationReport` covers one `validate()` call:

- **Volumes.** `facts_in` (and `unmatched`, rows whose text wasn't given),
  `refused` (and `refused_by`, the gate's reasons), `merged`, `linked` (and
  `links` by kind), `derived` and `facts_out`. The arithmetic holds:
  in + derived − refused − merged = out.
- **Grounding.** `verdicts` counts every fact in by the verdict grounding left
  it with. `checked` is what the free checks refused.
- **Cost.** `calls`, `tokens` and `cost_usd` (`None` until a provider reports
  one), plus `prompts`, the [registered prompts](models.md#prompts) sent.
- **Coverage.** `coverage` holds what extraction left behind, as
  `KnowledgeGraph.stats["coverage"]` keeps it.

The graph's `stats` carry the same report under `"validation"`, beside each
stage's own counts under `"stages"`, so a JSONL sink's `manifest.json` keeps
it too.

The name `openodke.Validator` meant the gate until 1.0.0. That alias is gone.
`openodke.Gate` is the gate, and `openodke.stages.Validator` still names it,
with a warning ([DECISIONS #26](decisions.md)).
