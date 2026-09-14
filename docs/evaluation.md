# Evaluation (BYOLD)

*Bring your own labelled dataset.*

Precision and recall claims need a harness, or they are marketing. A number
computed against the pipeline's own output measures nothing, though, and the
package cannot label your corpus for you. So for every stage, `odke.eval` ships
three things, and never a fourth:

- a **dataset format**: a JSONL row model whose docstring says exactly what one
  row is and what "correct" means.
- an **evaluator**: labels and predictions in, a `StageReport` out, plus a
  `run_*` helper that scores a stage in-process where that is cheap.
- a **fixture**: a handful of rows under `tests/fixtures/eval/`, so the package's
  own tests run.

!!! warning "The fixtures are not a benchmark"
    The rows under `tests/fixtures/eval/` exist to check the arithmetic. Every
    expected number in the tests was computed by hand, and no result on them says
    anything about any extractor, grounder or resolver, yours or ours. There is no
    corpus, no gold slice and no leaderboard. The numbers that mean something are
    the ones you compute on labels you made from your own documents.

Brier, B-cubed and P/R/F1 are arithmetic, so `odke.eval` adds no dependency to
the base install.

## `StageReport`

Every evaluator returns the same type:

- `stage` and `n` (the rows scored).
- `metrics`: a flat dict of headline numbers.
- `breakdown`: a per-key table (per label, per predicate, per verdict or per
  confidence bin).
- `confusion`: labelled class → predicted class → count, where the stage has
  classes.
- `notes`: what a number cannot say.

`report.render()` is the plain-text form `odke eval` prints; `--json` prints
`model_dump_json`.

A class nothing was predicted for has an **undefined** precision, and the report
keeps that as `None` (shown as `—`). Zero would read as "every prediction was
wrong", which is a different and worse finding.

## Formats and evaluators, per stage

| Stage | Label row | Prediction row | Evaluator | Headline metrics |
|---|---|---|---|---|
| route | `RouteLabel(id, text, action, label)` | `RoutePrediction(id, action, label)` | `evaluate_routing`, `run_route` | per-label P/R/F1, the skip/extract confusion, `facts_skipped` |
| extract | `GoldFact(doc_id, fact)` | `Fact` | `evaluate_extraction`, `run_extract` | per-predicate P/R/F1; wrong value, wrong entity, missing, spurious |
| ground | `GroundingLabel(text, fact, verdict, doc_id)` | `Fact` (optional) | `evaluate_grounding`, `grounding_ablation`, `run_ground` | accuracy, confusion, `false_support`, `lost_support`; precision off vs on |
| resolve | `PairLabel(a, b, same)` | `LinkRow(a, b, kind)` or an `EntityLink` | `evaluate_resolution`, `links_from_clusters` | pairwise P/R/F1 and B-cubed |
| score | `CalibrationLabel(fact, true)` | `Fact` (optional) | `evaluate_calibration`, `run_score` | Brier (with its baseline), ten-bin reliability curve, ECE |
| validate | `ValidationLabel(fact, action)` | `ValidationPrediction(id, action, reason)` | `evaluate_validation`, `run_validate` | agreement, Cohen's kappa, `wrongly_refused`, `wrongly_written`, `conflicts_missed` |
| sink | none | none | `check_idempotency`, `assert_idempotent`, `jsonl_counts` | every count unchanged across writes |
| cost | none | none | `CostMeter`, `CostReport`, `compare_costs` | calls, tokens, USD and latency per stage, per 1k documents |

Fact predictions are joined on `Fact.id`. The `facts.jsonl` a `JsonlSink` writes
is therefore already a predictions file, and a `links.jsonl` is already a
resolution predictions file. A hand-written fact with no `"id"` gets a fresh one
on every load, so give it one, or use a `run_*` helper, which needs no join.

```json
{"id": "c1", "text": "Ada Lovelace was born in London in 1815.", "action": "extract", "label": "fact"}
{"doc_id": "d1", "fact": {"subject": {"key": "ada", "type": "Person"}, "predicate": "born", "object_value": 1815}}
{"a": "a", "b": "b", "same": true}
```

`odke eval <stage> --describe` prints the full contract for a stage.

## extract

Scoring is never only an aggregate. A corpus-wide F1 of 0.8 is what twelve
predicates that work and three that never do look like, so the breakdown is per
predicate, and every miss is sorted into one of four kinds:

- **wrong value**: right subject and predicate, a different literal.
- **wrong entity**: the right claim pinned to the wrong node.
- **missing**: a gold fact no prediction came near.
- **spurious**: a prediction no gold fact came near.

A wrong value or wrong entity counts as a false positive *and* a false negative.
Matching is by `Fact.signature`, with literal values normalised first (whitespace
collapsed, case folded, numbers in one spelling), so `1815` and `"1815"` are one
value. A flipped polarity is never a near miss.

```python
from odke import Entity, Evidence, Fact, GroundingVerdict, Span
from odke.eval import GoldFact, evaluate_extraction

ada = Entity(key="ada", type="Person")
gold = [
    GoldFact(doc_id="d1", fact=Fact(subject=ada, predicate="born", object_value=1815)),
    GoldFact(doc_id="d1", fact=Fact(subject=ada, predicate="birthplace", object_value="London")),
]
cites_d1 = (Evidence(doc_id="d1"),)
extracted = [
    Fact(subject=ada, predicate="born", object_value="1815", evidence=cites_d1),  # correct
    # wrong value:
    Fact(subject=ada, predicate="birthplace", object_value="Paris", evidence=cites_d1),
    # spurious:
    Fact(subject=ada, predicate="died", object_value=1852, evidence=cites_d1),
]
report = evaluate_extraction(gold, extracted)

assert report.metrics["tp"] == report.metrics["wrong_value"] == report.metrics["spurious"] == 1
assert report.breakdown["born"]["f1"] == 1.0 and report.breakdown["birthplace"]["f1"] == 0.0
```

## ground

The grounder's two errors cost differently. `false_support` means a fact the
passage does not support was stamped `supported`, so a false fact got through.
`lost_support` means a true fact was stamped something else, so it was thrown
away. Both are counted separately from accuracy, which would blur them. A fact
that comes back `unchecked` is scored wrong, not skipped.

`grounding_ablation` is the release's sentence as a measurement: the precision of
what survives with grounding off (everything is kept) and on (a fact is kept
unless its verdict is `contradicted` or `not_found`), and the recall that
precision cost.

```python
from odke.eval import GroundingLabel, evaluate_grounding, grounding_ablation

text = "Ada Lovelace was born in London in 1815."
passage = (Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=len(text))),)


def labelled(id, predicate, value, verdict):
    fact = Fact(id=id, subject=ada, predicate=predicate, object_value=value, evidence=passage)
    return GroundingLabel(text=text, fact=fact, verdict=verdict)


labels = [
    labelled("g1", "born", 1815, "supported"),
    labelled("g2", "birthplace", "London", "supported"),
    labelled("g3", "born", 1816, "contradicted"),
    labelled("g4", "died", 1852, "not_found"),
]
# What a grounder stamped on each: right three times, and it lost g2.
said = {"g1": "supported", "g2": "not_found", "g3": "contradicted", "g4": "not_found"}
grounded = [
    row.fact.model_copy(update={"verdict": GroundingVerdict(said[row.fact.id])}) for row in labels
]

report = evaluate_grounding(labels, grounded)
assert (report.metrics["false_support"], report.metrics["lost_support"]) == (0, 1)

ablation = grounding_ablation(labels, grounded)
print(ablation.notes[0])
# grounding moved precision from 0.500 to 1.000, keeping 1 of 2 true facts
```

## resolve

This stage reports two numbers, because they fail differently. Pairwise precision
and recall ask, for each pair you labelled, whether the links put the two keys
together. They need no exhaustive labels, and they punish a wrong merge of two big
clusters exactly once. B-cubed asks, for each key, how much of its predicted
cluster is really its cluster. It sees that same wrong merge from every member's
side, which is what a user of the graph sees too.

Links are `(a, b, kind)` triples from any source: an `EntityLink` proposed here, a
line of `links.jsonl`, or a platform's merges read back out of the store.
`same_as` chains are followed. A store that *replaced* nodes leaves groups of
original keys behind; `links_from_clusters` turns each group into `same_as` links
so it scores exactly like a resolver that linked. `similar` counts as no decision
unless you pass `similar_as_same=True`.

```python
from odke.eval import PairLabel, evaluate_resolution, links_from_clusters

pairs = [
    PairLabel(a="acme", b="acme-inc", same=True),
    PairLabel(a="acme", b="acme-gmbh", same=False),
]
merged_by_platform = links_from_clusters([["acme", "acme-inc", "acme-gmbh"]])
report = evaluate_resolution(pairs, merged_by_platform)

assert (report.metrics["pairwise_recall"], report.metrics["pairwise_precision"]) == (1.0, 0.5)
```

## score

The test for `Fact.confidence` is whether the facts scored 0.9 were true about 90%
of the time, and the report states that first. Normalising scores into [0, 1]
makes them look like probabilities without making them behave like ones.

- **Brier** is mean((confidence − outcome)²). It is printed beside
  `brier_baseline`, the Brier score of always answering the base rate. A scorer
  that does not beat that baseline is worse than a constant.
- **Reliability curve**: ten equal-width bins, each showing mean confidence beside
  the fraction of its facts that were true.
- **ECE** is the per-bin gap weighted by the bin's share. It can be zero for a
  useless scorer, which is why Brier sits next to it.

```python
from odke.eval import CalibrationLabel, evaluate_calibration

rows = [
    CalibrationLabel(
        fact=Fact(subject=ada, predicate=f"claim_{i}", object_value=i, confidence=0.9),
        true=i < 6,
    )
    for i in range(10)
]
report = evaluate_calibration(rows)

print(report.notes[0])
# of facts scored ≥ 0.9, 60% were true (6 of 10)
assert round(report.metrics["ece"], 3) == 0.3
assert report.metrics["brier"] > report.metrics["brier_baseline"]  # worse than a constant
```

## validate

The validator is the gate, so its costly disagreements point in opposite
directions. `wrongly_refused` is silent data loss. `wrongly_written` puts a fact in
the graph that should not be there. `conflicts_missed` is a contradiction nobody
will be asked to look at. Agreement is reported as accuracy and as Cohen's kappa:
a validator that accepts everything scores a high accuracy on a mostly acceptable
slice, and a kappa of zero.

## sink

No labels are needed. Write a graph twice and count once. What "count" means is
the store's business, so you pass it as a callable. `jsonl_counts` is that
callable for `JsonlSink`, and a Neo4j test passes one that runs `count(n)`.

```python
import tempfile

from odke import KnowledgeGraph
from odke.eval import assert_idempotent, jsonl_counts
from odke.sinks import JsonlSink

graph = KnowledgeGraph(facts=tuple(row.fact for row in labels))
with tempfile.TemporaryDirectory() as out:
    report = assert_idempotent(JsonlSink(out), graph, jsonl_counts(out))

print(report.notes)
# ('idempotent: every count unchanged over 2 writes',)
```

## cost

Cost needs no labels, and measuring it changes no stage. Every model call goes
through an `LLMClient`, and a model-backed stage takes its client as a constructor
argument. `CostMeter.client(inner, stage=...)` wraps a client, records every
call's tokens, `cost_usd` and wall-clock latency under that stage's name, and
forwards the call. An unknown cost stays unknown. A stage with any unpriced call
has `cost_usd = None` rather than a partial sum that would understate the bill.
The partial sum is still available, as `priced_usd`.

```python
from odke.eval import CostMeter
from odke.ground import LLMGrounder
from odke.llm import ModelRoles, ScriptedClient

meter = CostMeter()
client = meter.client(ScriptedClient([{"verdict": "supported"}] * len(labels)), stage="ground")
grounder = LLMGrounder(ModelRoles.single("ollama/qwen2.5:3b"), client=client)
for row in labels:
    grounder.ground(row.fact, row.as_document())

cost = meter.report(documents=1)
assert cost.stages["ground"].calls == 4
assert cost.total.cost_usd is None  # a scripted client reports no price, and none is invented
print(cost.per_1k_documents()["total"]["calls"])
# 4000.0
```

`compare_costs({"hybrid": a, "model-only": b})` sets several metered runs side by
side per thousand documents.

## From the shell

```bash
odke eval ground --describe                        # what to label, and what to predict
odke eval ground --labels ground.jsonl --predictions facts.jsonl
odke eval resolve --labels pairs.jsonl --predictions links.jsonl --json
odke eval route --labels route.jsonl --run mypackage.routers:MarketingRouter
odke eval extract --labels gold.jsonl --run mypackage.extract:MyExtractor \
    --documents documents.jsonl --ontology schema.json
odke eval validate --labels verdicts.jsonl --run mypackage.gate:MyValidator --ontology schema.json
```

- The CLI covers route, extract, ground, resolve, score and validate.
- `--run package.module:Name` imports a stage and runs it over the labels. A class
  is instantiated with no arguments. Otherwise, point it at a module-level
  instance.
- Resolution cannot run from labels, because a resolver takes facts, not pairs.
  Its links always come from a file, which is also how a platform's merges arrive.
- Errors exit with status 2.
