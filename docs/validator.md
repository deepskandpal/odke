# The Validator

openodke is the verification layer between any extractor and the graph store
([DECISIONS #24](decisions.md#24)). The Validator is that layer as one class,
`openodke.Validator`, and one command, `odke validate`. Give it another
extractor's triples and their texts: it grounds, normalises, resolves,
corroborates, gates and writes them, and reports what each step did.
`odke ground` is the [grounding step alone](grounding.md#a-graph-from-somewhere-else-odke-ground).

## One command

On the five triples in `examples/triples/`, with recorded answers and no key:

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
calibration   ground.span@1 on anthropic/claude-haiku-4-5-20251001 (inline): κ 0.12 [-0.09, 0.38] (n 104), below its bar (κ ≥ 0.6); fail-recall 0.20, false refusals 0.09, planted caught 0.80 (label set G gate, n 104, 2026-10-25); refusals marked for review, not dropped (its card is below the bar, DECISIONS #54)
wrote         jsonl → out: entities.jsonl 5, facts.jsonl 5, links.jsonl 0, manifest.json, refused.jsonl 0
writes        0:JsonlSink: entities 5 written; facts 5 written; links none
manifest      out/manifest.json
```

The two `not_found` facts are written: the default gate refuses only what the
text contradicts and what the ontology has no room for. To keep them out too,
give the gate `refuse_not_found` ([the gate](grounding.md#the-gate-verdictgate)).
Beside the graph, `-o` writes `refused.jsonl`, each refused fact with the gate's
reason, which [`odke eval refusals`](evaluation.md#what-the-grounder-refused)
samples for a person to judge.

The `calibration` line is the grounder's [calibration card](calibration.md), or
UNCALIBRATED where there is none, and what its refusals did in this job. The
default grounder's card is below its bar, so a fact it contradicts is written
marked for review (`qualifiers["odke.review"]`, counted under `marked`) instead
of refused; `on_refusal="drop"` refuses it ([DECISIONS #54](calibration.md#below-the-bar-nothing-is-dropped-on-a-judges-word)).

| Flag | |
|---|---|
| `--facts`, `--texts`, `--adapter`, `--ontology` | The facts and their texts, as [`odke ground`](grounding.md#a-graph-from-somewhere-else-odke-ground) reads them |
| `--config` | A run config whose extractor is `triples`: its inputs, ontology, models, stages and sinks, a stage it leaves out taking the Validator's default. With `--facts`, only its `models` block is read |
| `-o` | Also write the graph here as JSONL |
| `--merge` | [Merge](stores.md#merge-with-the-store) into the JSONL already at `-o` |
| `--update` | The texts are new versions of ones the store cites: [retract](stores.md#reconcile) each first. Needs a sink that can retract; implies `--merge` |
| `--locate` | The [span locator](grounding.md#locating-spans) |
| `--dry-run` | The free checks, the locator and every deterministic stage: no model, no write |
| `--model`, `--model-provider`, `--cache`, `--budget-usd`, `--budget-calls`, `--log-format`, `--log-text` | As [`odke run`](run.md) takes them |
| `--batch-size` | [Stream](run.md#streaming) this many rows at a time |
| `--tenant` | Key the store by this [tenant](stores.md#tenants) |
| `--text-property`, `--user`, `--password-env`, `--database` | Neo4j, as `odke ground` takes them |

Exit 2 is input or a config that cannot be read, exit 1 a failed run, exit 3 a
run its budget stopped, which still writes what it kept.

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

The two refusals are the contradicted 2012, and `opened_in`, which is not in
the ontology and so cost no call. Brittany's `located_in`, stated in both
texts, merged into one fact with `support` 2, and its inverse partner `contains`
was derived from each and merged too. `Halden Robotic` became a `SIMILAR` link
to `Halden Robotics`.

`validate(facts, documents=(), *, dry_run=False, extractor="triples",
confidence=0.5, update=False, batch_size=None)` takes any adapter's
`(rows, documents)` ([Inputs](inputs.md#from-other-libraries)), a triples file
and its texts, or `Fact`s and their texts. Rows become facts named `extractor`,
with `confidence` as their prior. It returns the `KnowledgeGraph` and a
`ValidationReport`, writing the graph to every sink in `sinks=`. A `Validator` keeps its stages between calls, and each report counts
its own call. `batch_size=N` [streams](run.md#streaming) the rows; the graph
returned then holds the stats and no facts.

```python
import tempfile
from pathlib import Path

from openodke.sinks import JsonlSink

store = JsonlSink(Path(tempfile.mkdtemp()), merge=True)
_, streamed = Validator(places, client=client, sinks=[store]).validate(rows, [a, b], batch_size=1)
# a, b, a a, b: each micro-batch ends where its rows' text changes.
print(streamed.batches, streamed.restated)
# 4 2
```

## The stages

`Validator(ontology=None, *, ...)` builds each stage left out as `odke run`
would, with that name's defaults. Passing a stage replaces its default, and a
pass-through from `openodke.stages` turns it off.

| Argument | Default |
|---|---|
| `grounder` | `LLMGrounder(roles, client=client, locate=locate)`. Any grounder runs behind the free checks (`CheckedGrounder`) |
| `roles`, `client`, `locate` | `ModelRoles()`, resolved from the model, `False` |
| `normalizer` | `ValueNormalizer(ontology)` |
| `resolver` | `NativeResolver(ontology=ontology)`, given `lookup`, `judge`, `normalize_batch` and `embed`; with a resolver of your own, give it those instead |
| `lookup` | none; a `StoreLookup` [resolves against the store](resolution-and-corroboration.md#resolving-against-the-store) |
| `judge` | none; a `PairJudge` is [the pair judge](resolution-and-corroboration.md#the-pair-judge), never asked in a dry run |
| `normalize_batch`, `embed` | `False`, none: [normalise the batch](resolution-and-corroboration.md#normalising-mentions-in-a-batch) |
| `corroborator` | `SignatureCorroborator(ontology, store=...)`, given every sink that is a `FactLookup`, so a write [merges with the store](stores.md#merge-with-the-store) |
| `scorer` | `EvidenceScorer()` |
| `gate` | `VerdictGate(schema=True)`: refuses what the text contradicts and what the free checks on the ontology failed, since they leave the verdict alone |
| `inverses` | on when the ontology declares a pair ([DECISIONS #28](decisions.md#28)) |
| `coverage` | `True`: [what extraction left behind](grounding.md#what-extraction-left-behind-the-coverage-report) |
| `sinks` | none |
| `manifest` | none; a path to write the [run manifest](run.md#the-run-manifest) to as well |
| `tenant` | none; scopes every sink and lookup to a [tenant](stores.md#tenants) |

Without an ontology, the checks on the relation and the types have nothing to
check.

## The report

`ValidationReport` covers one `validate()` call. In + derived − refused −
merged = out.

| Field | Holds |
|---|---|
| `documents`, `facts_in`, `unmatched` | texts, facts in, and rows whose text was not given |
| `verdicts`, `checked` | every fact in by its verdict; what the free checks refused |
| `refused`, `refused_by` | the gate's refusals, by reason |
| `merged`, `restated` | facts merged by signature; of those, merged with the store |
| `linked`, `links`, `derived` | resolver links, by kind; inverse partners added |
| `facts_out`, `edges`, `properties`, `entities` | the graph |
| `calls`, `cached`, `tokens`, `cost_usd`, `prompts` | the grounder's and judge's calls, `cost_usd` `None` until priced, and the [prompts](models.md#prompts) sent |
| `coverage` | as `KnowledgeGraph.stats["coverage"]` |
| `store`, `judge`, `batch` | with a lookup, a judge or batch normalisation: [their counts](resolution-and-corroboration.md#resolve) |
| `retracted` | with `update`, what retracting the old versions did |
| `stopped`, `failed`, `batches` | a budget stop, the texts left out, the micro-batches |
| `tenant`, `writes`, `dry_run` | the tenant, [the write report](stores.md#the-write-report), and whether it was a dry run |

The graph's `stats` carry the report under `"validation"`, beside each stage's
counts under `"stages"`, so a JSONL sink's `manifest.json` keeps it. Its JSON
is versioned by `schema_version` (`"1.0"`), with
`openodke/validation_report.schema.json` as its schema
([Stability](stability.md#formats)). `report.manifest` is the job's
[run manifest](run.md#the-run-manifest), kept out of the JSON and of equality.
