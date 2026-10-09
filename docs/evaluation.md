# Evaluation (BYOLD)

*Bring your own labelled dataset.*

Precision and recall claims need a harness, or they are marketing. A number
computed against the pipeline's own output measures nothing, though, and the
package cannot label your corpus for you. So for every stage, `openodke.eval` ships
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

Brier, B-cubed and P/R/F1 are arithmetic, so `openodke.eval` adds no dependency to
the base install.

One evaluator is not BYOLD: [`odke eval spans`](#spans) needs **no labelled data
at all**. It reads a run's own facts and reports cited span width split by the
grounder's verdict, which scores the too-narrow-citation error directly.

## `StageReport`

Every evaluator returns the same type:

- `stage` and `n` (the rows scored).
- `metrics`: a flat dict of headline numbers.
- `breakdown`: a per-key table (per label, per predicate, per verdict or per
  confidence bin).
- `confusion`: labelled class → predicted class → count, where the stage has
  classes.
- `notes`: what a number cannot say.

`report.render()` is the plain-text form `odke eval` prints, below the eval
report's rows; `--json` prints `model_dump_json`.

A class nothing was predicted for has an **undefined** precision, and the report
keeps that as `None` (shown as `—`). Zero would read as "every prediction was
wrong", which is a different and worse finding.

## The eval report

A `StageReport` is shaped by its stage. A CI job, a dashboard or a second run to
compare against needs precision in the same place every time, so `odke eval`
and `odke bench run` also write an `EvalReport`: one JSON document, versioned by
`schema_version`. A minor version adds fields and a major one changes them, so
check the major version first.

- `odke eval <stage> --report report.json` writes it. `--json` still prints the
  stage's `StageReport`, as before.
- `odke bench run` writes it as `report.json` beside the predictions; `--report`
  puts it elsewhere.
- The schema ships in the package, as `openodke/eval/eval_report.schema.json`.
  `check_report(data)` holds a document to it in plain Python and lists every
  mismatch. The package checks each report before writing it, and
  `read_report(path)` checks before loading.

| Key | What it holds |
|---|---|
| `schema_version` | `"1.0"` |
| `title`, `n` | what was scored (`extract`, `ablation`, `text2kgbench:ont_1_movie`) and how many rows |
| `run` | `openodke` (the version), `models` (role → model), `prompts` (the registered keys sent), `dataset` (name, path, documents, labels, details) |
| `bootstrap` | how the ranges were drawn: `unit`, `units`, `resamples`, `seed`, `level`, `method`; `null` with no rows |
| `rows[]` | one per configuration scored against gold facts: one for `extract`, three for the ablation and a benchmark, none for the other stages |
| `rows[].performance` | `precision`, `recall`, `f1`, each `{value, low, high}`; `average` is `micro` (facts pooled) or `macro` (the mean of documents, as Text2KGBench averages) |
| `rows[].counts` | `hits`; `over_extraction` (predicted, not in gold); `under_extraction` (gold, not predicted); `predicted`, `gold`; `uncited` (spurious predictions citing no document); `unscored`; `documents` |
| `rows[].conformance` | `rate`, `conformant`, `facts`, `checks`: the share of predicted facts whose relation and types fit the ontology; `null` with no ontology |
| `rows[].hallucination` | `definition`, `hallucinated`, `facts`, `rate`, and per-part rates where the dataset gives them; `null` where it defines none |
| `rows[].cost`, `rows[].latency` | calls, tokens, USD (`null` when any call was unpriced) and seconds in calls; `null` when the run was not metered |
| `stages[]` | the `StageReport`s, unchanged |
| `notes[]` | what a number cannot say about the whole run |
| `comparison` | `odke eval compare`'s block, unchanged: `a`, `b`, `unit`, `metric`, the `primary` paired result, `guardrails`, `mcnemar`, `notes`; `null` when nothing was compared. `odke eval compare A B --report PATH` writes a report that carries it |
| `diagnosis`, `fixes`, `calibration` | reserved, empty and typed until #140, #141 and #135 fill them |

- **The ranges are 95% bootstrap ranges over documents.** Facts from one
  document share one reading of it, so they are not independent; resampling
  facts would print a range too narrow. 2,000 draws, seed 0, so the same labels
  give the same range on any machine. A draw on which a number is undefined is
  left out.
- **One bootstrap.** `openodke.eval.bootstrap` draws the documents and reads
  the percentiles, for a report's ranges and for [`compare`](#was-the-change-real)
  alike, so the same documents and seed give the same range in both. A
  document's counts are the row `--items` writes for it.
- **A spurious prediction that cites no document** is in no document's counts,
  and so in no `--items` row. The report still counts it, as
  `evaluate_extraction` does, in its numbers and in every draw, and
  `counts.uncited` says how many.
- **A wrong value or a wrong entity** counts once in `over_extraction` and once
  in `under_extraction`, as it does in `fp` and `fn`.
- **The numbers are the evaluator's own.** The rows hold the same precision,
  recall and F1 as the `StageReport` beside them, to the last digit, and
  `render()` prints them from the same fields.

```python
from openodke import Entity, Evidence, Fact
from openodke.eval import GoldFact, evaluate_extraction
from openodke.eval.eval_report import check_report, extraction_rows, from_stage

ada, babbage = Entity(key="ada", type="Person"), Entity(key="babbage", type="Person")
gold = [
    GoldFact(doc_id="d1", fact=Fact(subject=ada, predicate="born", object_value=1815)),
    GoldFact(doc_id="d2", fact=Fact(subject=babbage, predicate="born", object_value=1791)),
]
said = [
    Fact(subject=ada, predicate="born", object_value="1815", evidence=(Evidence(doc_id="d1"),)),
    Fact(subject=babbage, predicate="born", object_value=1792, evidence=(Evidence(doc_id="d2"),)),
]
rows, how = extraction_rows([("extract", said, None)], gold)
report = from_stage(evaluate_extraction(gold, said), rows=rows, bootstrap=how)

# One right document and one wrong one: the range is everything from 0 to 1.
precision = report.rows[0].performance.precision
assert (precision.value, precision.low, precision.high) == (0.5, 0.0, 1.0)
assert check_report(report.model_dump(mode="json")) == []
```

## Formats and evaluators, per stage

| Stage | Label row | Prediction row | Evaluator | Headline metrics |
|---|---|---|---|---|
| route | `RouteLabel(id, text, action, label)` | `RoutePrediction(id, action, label)` | `evaluate_routing`, `run_route` | per-label P/R/F1, the skip/extract confusion, `facts_skipped` |
| extract | `GoldFact(doc_id, fact)` | `Fact` | `evaluate_extraction`, `run_extract` | per-predicate P/R/F1; wrong value, wrong entity, missing, spurious |
| ground | `GroundingLabel(text, fact, verdict, doc_id)` | `Fact` (optional) | `evaluate_grounding`, `grounding_ablation`, `run_ground` | accuracy, confusion, `false_support`, `lost_support`; precision off vs on |
| resolve | `PairLabel(a, b, same)` | `LinkRow(a, b, kind)` or an `EntityLink` | `evaluate_resolution`, `links_from_clusters` | pairwise P/R/F1 and B-cubed |
| score | `CalibrationLabel(fact, true)` | `Fact` (optional) | `evaluate_calibration`, `run_score` | Brier (with its baseline), ten-bin reliability curve, ECE |
| validate | `ValidationLabel(fact, action)` | `ValidationPrediction(id, action, reason)` | `evaluate_validation`, `run_validate` | agreement, Cohen's kappa, `wrongly_refused`, `wrongly_written`, `conflicts_missed` |
| spans | **none** | a run's `facts.jsonl` | `evaluate_spans`, `odke eval spans` | span width per verdict — count, median, quartiles, min/max; `not_found_rate`, `median_gap` |
| sink | none | none | `check_idempotency`, `assert_idempotent`, `jsonl_counts` | every count unchanged across writes |
| cost | none | none | `CostMeter`, `CostReport`, `compare_costs` | calls, tokens, USD and latency per stage, per 1k documents |
| ablation | `GoldFact(doc_id, fact)` | a run config, run three ways | `run_ablation`, `odke eval ablation` | precision, recall and F1, facts written and model calls for extraction alone, + grounding, + corroboration |

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
from openodke import Entity, Evidence, Fact, GroundingVerdict, Span
from openodke.eval import GoldFact, evaluate_extraction

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
from openodke.eval import GroundingLabel, evaluate_grounding, grounding_ablation

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

## spans

**This is the one evaluator that needs no labelled data at all.** Everything else
on this page is BYOLD; `odke eval spans` reads a run's facts and nothing else.

It exists because a citation too narrow to carry its claim is an extraction bug
the grounder reports for free. Shown `Ireland` and asked whether
`Acme operates_in Ireland` follows from it, a correct grounder answers
`not_found` — the fact was true, the extraction was right, and the citation threw
it away. Where an extractor makes this mistake the two distributions separate:
narrow citations cluster in `not_found`, clause-width ones in `supported`.

So where the widths separate like that, **the `not_found` rate is a usable proxy
for citation quality with no gold set**: it scores the too-narrow-citation error
directly, which is the error a span gold set would be built to find. The report
is the width distribution per verdict — count, median, quartiles, min and max —
plus the share of facts citing no span at all, and one summary line. The widths
are of `Evidence.span`, the span the grounder is shown. A span the
[locator](grounding.md#locating-spans) found is not a citation, so it is counted
apart, in the `located` column, and left out of the widths.

```python
from openodke.eval import evaluate_spans


def cited(predicate, value, width, verdict):
    """One fact whose citation is `width` characters wide."""
    evidence = (Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=width)),)
    return Fact(
        subject=ada,
        predicate=predicate,
        object_value=value,
        evidence=evidence,
        verdict=GroundingVerdict(verdict),
    )


facts = [
    cited("supports_language", "German", 6, "not_found"),  # cited 'German'
    cited("operates_in", "Ireland", 7, "not_found"),  # cited 'Ireland'
    cited("operates_in", "Virginia", 8, "not_found"),
    cited("operates_in", "Singapore", 9, "not_found"),
    cited("uptime_commitment", 99.9, 12, "not_found"),
    cited("performs", "on-call rotation", 46, "supported"),  # cited the clause
    cited("performs", "data residency", 81, "supported"),
]
report = evaluate_spans(facts)

print(report.notes[0])
# not_found median 8 chars vs supported 63.5 — citations are too narrow
assert report.metrics["not_found_rate"] == 5 / 7
assert report.breakdown["not_found"]["max"] < report.breakdown["supported"]["min"]
```

The summary line names the gap only when the distributions really separate — the
upper quartile of the `not_found` widths below the lower quartile of the
`supported` ones. When they overlap it says so and nothing more: that is a
finding about your corpus, not an alarm. And the numbers are the grounder's own
verdicts rather than labels, so the gap is a diagnostic of the citations, not a
score of the facts.

```bash
odke eval spans --facts out/facts.jsonl   # or --facts out/, the directory a sink wrote
odke eval spans --describe
```

In process, `evaluate_spans(kg.facts)` takes a `KnowledgeGraph`'s facts, and
`load_facts(path)` reads the JSONL.

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
from openodke.eval import PairLabel, evaluate_resolution, links_from_clusters

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
from openodke.eval import CalibrationLabel, evaluate_calibration

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

This scores a gate. What it refuses is not written, so its costly disagreements
point in opposite directions. `wrongly_refused` is silent data loss.
`wrongly_written` puts a fact in the graph that should not be there.
`conflicts_missed` is a contradiction nobody will be asked to look at. Agreement
is reported as accuracy and as Cohen's kappa: a gate that accepts everything
scores a high accuracy on a mostly acceptable slice, and a kappa of zero.

## sink

No labels are needed. Write a graph twice and count once. What "count" means is
the store's business, so you pass it as a callable. `jsonl_counts` is that
callable for `JsonlSink`, and a Neo4j test passes one that runs `count(n)`.

```python
import tempfile

from openodke import KnowledgeGraph
from openodke.eval import assert_idempotent, jsonl_counts
from openodke.sinks import JsonlSink

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
from openodke.eval import CostMeter
from openodke.ground import LLMGrounder
from openodke.llm import ModelRoles, ScriptedClient

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

## ablation

The architecture's claim is that grounding and corroboration each earn their place.
`run_ablation(config, gold)` measures that claim on your labels, or fails to. It
runs one [run config](run.md) three ways against one labelled extraction set:

1. **extraction alone**: the config's loaders, chunker, router and extractor, with
   every candidate kept.
2. **+ grounding**: the same candidates through the config's grounder, then its
   gate, which is the configured one or `VerdictGate` when it names none.
3. **+ corroboration**: the whole configured pipeline (normalise, resolve,
   corroborate and score), then the same gate.

Extraction runs once and grounding runs once. The later configurations replay the
facts the earlier ones produced through the real `Pipeline`, so the rows differ only
by the stages they add, and a model is never asked the same question twice, which
would double the bill and could get a different answer the second time. Nothing is
written to the config's sinks, and the cost meter is on whatever the config says,
because calls and cost are columns of the table.

Each row is `evaluate_extraction` against the gold facts, per document. A gold fact
names its document by the id [`odke run`](run.md#inputs) gives it: the input's path
relative to the config, plus `#L<line>` for a record. A fact that corroboration
merged across documents cites each of them and is scored once in each, because the
graph now claims every one of those documents states it. The notes add the view from
inside the extracted set: how many true facts the gate kept, how many false ones it
let through, and what refusing `not_found` as well would have changed. If grounding
does not move precision on your labels, the first note says so. That is a finding
about your corpus, not a failure of the command.

```bash
odke eval ablation --config odke.yaml --labels gold.jsonl
odke eval ablation --describe
```

### On the end-to-end example

!!! warning "A demonstration on recorded responses, not a benchmark"
    The table below is `examples/e2e`, run on **hand-authored** recorded model
    responses against 36 labels written for the same example. The responses and
    the labels were both written for it, so the numbers measure how that fixture
    was written and nothing about any model. They show what the command reports
    and how to read it. Run it on your own labels.

| Configuration | Precision | Recall | F1 | TP | FP | FN | Facts written | Model calls |
|---|---|---|---|---|---|---|---|---|
| extraction alone | 0.889 | 0.889 | 0.889 | 32 | 4 | 4 | 36 | 3 |
| + grounding (gate refuses `contradicted`) | 0.914 | 0.889 | 0.901 | 32 | 3 | 4 | 35 | 39 |
| + corroboration (normalise, resolve, corroborate, score) | 0.971 | 0.944 | 0.958 | 34 | 1 | 2 | 25 | 39 |

Cost is unknown in all three rows: the recorded responses carry no price.

```python
from openodke.eval import GoldFact, load_jsonl, run_ablation
from openodke.run import load_config

ablation = run_ablation(
    load_config("examples/e2e/odke.yaml"), load_jsonl("examples/e2e/gold.jsonl", GoldFact)
)
rows = ablation.breakdown
assert [rows[name]["precision"] for name in rows] == [8 / 9, 32 / 35, 34 / 35]
assert [rows[name]["model_calls"] for name in rows] == [3, 39, 39]
for note in ablation.notes[:4]:
    print(note)
# grounding moved precision from 0.889 to 0.914; recall 0.889 → 0.889
# normalising, resolving and corroborating moved precision from 0.914 to 0.971; recall 0.889 → 0.944
# of the 36 extracted facts, 32 are true: the gate kept 32 of them and 3 of the 4 false ones
# refusing not_found as well would keep 19 of 32 true facts and 1 of 4 false ones (precision 0.950)
```

What it says about that fixture, and only that fixture:

- **Grounding moved precision from 0.889 to 0.914, and not recall.** Of the four
  wrong candidates it caught one, a head office the model invented. A second
  invented fact came back `not_found`, which the default gate keeps.
- **Refusing `not_found` as well would have been worse there.** It would keep 19
  of the 32 true facts: a register cell is grounded against itself, which cannot
  say whose value it is, so a careful grounder answers `not_found` for most
  structured facts.
- **Most of the gain came after grounding, from normalisation.** Two dates the
  model copied as written became the ISO dates the labels use. Merging alone moves
  neither number, because a merged fact is scored once in each document it cites.

## Was the change real?

An eval can mislead three ways: in what it sampled, in how its judges are
calibrated, and in its statistics. `odke eval compare` handles the third. Two
runs scored on the same items are not two independent samples. Most items pass
or fail in both, and only the ones that flip carry information, so the
comparison is paired. A **paired bootstrap** (Koehn 2004) resamples items with
replacement, recomputes the metric for both runs on the same resample, and reads
the 95% interval of the difference. For corpus precision, recall and F1 the item
is the document: its counts are resampled, and the corpus metric is recomputed
from their sums.

**Three verdicts, never two** ([DECISIONS #29](decisions.md#29)).

| Verdict | When | Exit |
|---|---|---|
| better | the whole 95% interval is above zero | 0 |
| worse | the whole 95% interval is below zero | 1 |
| inconclusive | the interval contains zero | 0, or 1 with `--fail-on-inconclusive` |

Inconclusive never means "no regression". It is printed with the **detection
limit**, the smallest change the set detects at 95% confidence and 80% power:
about Z·√(d/n), with Z = 1.959964 + 0.841621 and d the share of items that flip.

| Items | Detection limit at 10% flips |
|---|---|
| 300 | 5.1 points |
| 1,500 | 2.3 points |
| 5,000 | 1.3 points |

For pass/fail items the limit is that formula. For a corpus metric, where one
document moves F1 by its own amount rather than by 1/n, it is Z times the
bootstrap's own standard error. When no item changed at all, the limit is
reported as unknown rather than as zero.

**One primary metric.** F1 decides for documents and accuracy for pass/fail
items; `--metric` picks another. The rest are guardrails: printed with their
intervals, never deciding, so three metrics are not three chances of a false
alarm. For pass/fail items, McNemar's exact test on the flips is printed as a
cross-check, and a note says when it disagrees with the bootstrap.

```bash
odke eval ground --labels gate.jsonl --predictions p1/facts.jsonl --items p1.items.jsonl
odke eval ground --labels gate.jsonl --predictions p2/facts.jsonl --items p2.items.jsonl
odke eval compare p1.items.jsonl p2.items.jsonl          # A, then B
odke eval compare p1.items.jsonl p2.items.jsonl --json   # one comparison block
odke eval compare --describe
```

```text
compare  accuracy over 300 items  (A: p1.items.jsonl, B: p2.items.jsonl)
  verdict   inconclusive: with this set the smallest change it can detect is 5.1 points
  accuracy  A 0.853  B 0.847  difference -0.7 points, 95% interval -4.3 to +3.0
  limit     5.1 points: the smallest change these 300 items detect (95% confidence, 80% power)
  flips     30 of 300 items changed (10.0%): 14 gained, 16 lost, McNemar exact p = 0.856
  B range   0.807 to 0.887 (95%)
```

- `--items` writes one row per item, for extract, ground, validate and route:
  `{"id": "d1", "tp": 3, "fp": 1, "fn": 0}` per labelled document, and
  `{"id": "g7", "correct": true}` per labelled fact or chunk. The rows sum to the
  report's own counts, less any spurious prediction that cites no document,
  which is in no document's row and is named in a warning.
- Both files must hold the same ids, and items are paired by id. Different items
  are refused, naming the ones only one side has: a comparison over different
  items measures the difference between the sets.
- `--resamples` (2,000) and `--seed` (0): the same items and seed give the same
  interval on every interpreter. `--json` prints one `Comparison` block, and
  `--report PATH` writes it as the [eval report](#the-eval-report)'s
  `comparison` section.

**In CI**, keep the base run's items and compare the head's against them:

```bash
odke eval extract --labels gold.jsonl --predictions out/facts.jsonl --items head.items.jsonl
odke eval compare base.items.jsonl head.items.jsonl --fail-under 0.80
```

- The step exits 1 when the verdict is worse, with a `gate: fail:` line on
  stderr for each reason.
- `--fail-under 0.80` also exits 1 when the low end of B's own 95% range for the
  primary metric is under 0.80. B has to clear the floor with its uncertainty, not
  with its point estimate.
- `--fail-on-inconclusive` also exits 1 on inconclusive. It is off by default: a
  change too small for the set to see is not a regression it saw, and the limit
  says how small. Turn it on once the set is big enough that "inconclusive" means
  "too small to matter".
- Exit 2 means files that cannot be compared.

The comparison is tested on itself. `tests/test_eval_aa.py` runs 300 seeded A/A
comparisons, the same simulated system twice, and 96.7% come back inconclusive,
within three binomial standard deviations of 95%. Offline, at 1,500 comparisons
each, the rate was 95.2% to 95.3% on pass/fail items and 94.7% to 94.9% for corpus
F1. A planted drop well above the limit comes back worse; one well below it comes
back inconclusive, with the limit printed.

```python
from openodke.eval import detection_limit, mcnemar, paired_bootstrap

print([round(detection_limit(n, 0.10) * 100, 1) for n in (300, 1_500, 5_000)])
# [5.1, 2.3, 1.3]

# 300 items: 240 pass in both, 3 fixed, 27 broken, 30 fail in both.
before = [True] * 240 + [False] * 3 + [True] * 27 + [False] * 30
after = [True] * 240 + [True] * 3 + [False] * 27 + [False] * 30
result = paired_bootstrap(before, after)
print(result.verdict, round(result.difference * 100, 1), round(result.detection_limit * 100, 1))
# worse -8.0 5.1
assert mcnemar(before, after).p_value < 0.001
```

`compare_items` does the same over two runs' `ItemRow`s, and
`bootstrap_interval` gives one run's 95% range from the same resampler.

## Labelling by hand

`odke label` writes rows out as markdown sheets that a person ticks wherever a
markdown file opens, Obsidian on a tablet included. It then reads the ticks
back as `GroundingLabel` or `PairLabel` rows.

```bash
odke label make grounding to-check.jsonl -o sheets/ --per-sheet 50
odke label make pair pairs.jsonl -o pair-sheets/
odke label read sheets/ -o ground.jsonl        # or one sheet: sheets/sheet-03.md
odke eval ground --labels ground.jsonl --predictions out/facts.jsonl
```

- **A grounding row** is a `GroundingLabel` without its `verdict`: `text`,
  `fact` and an optional `doc_id`. Give every fact an `id`, because predictions
  join on it; `make` warns about any fact that has none.
- **A pair row** is `{"a": {...}, "b": {...}}`. Each side has a `key` and a
  `type`, plus an optional `label` (the name as written) and `context`.
- **Sheets** are `sheet-01.md`, `sheet-02.md` and so on, and one sheet is one
  sitting. Item ids run across the sheets (`G-0001`, `P-0001`). Beside each
  sheet, `sheet-NN.items.jsonl` keeps its rows, so the markdown only has to
  carry the ticks and the notes. The same rows give byte-identical sheets.
  `make` refuses a directory that already holds sheets.

A grounding item, as written:

```markdown
### G-0007

Acme Cloud (Company) — operates in — "Ireland".

> **Acme Cloud runs data centres in Ireland, Virginia and Singapore.** Its uptime target is 99.9\% and it costs \$5 a seat.

- [ ] supported
- [ ] contradicted
- [ ] not found

note:
```

The claim line is `render_claim(fact)`, the sentence the grounder is asked
about, and the cited span is bold inside the passage. A `context` span, the
whole text of a fact that cited nothing, is no citation and stays plain.
Passage text is escaped, so markdown cannot hide or restyle it: `$` would
otherwise open a formula and `%%` a comment. A pair item shows `A:` and `B:`,
each a name and its type with its context quoted, then the boxes `same`,
`different` and `unsure`. Each sheet opens with two lines that explain the
answers and asks for exactly one tick.

Reading back:

- **Exactly one tick** (`[x]` or `[X]`) is a label. **No tick** leaves the item
  unlabelled; it is counted in the summary, not refused.
- **Two or more ticks**, or a heading the sidecar does not know, exit 2 with
  the file and the line of the item's heading. A mark other than `x` names the
  box's own line. Every problem in every sheet is listed at once, and nothing is
  written.
- **Labels** go to `-o` in input order: the row as given plus `verdict`, or
  `PairLabel(a, b, same)` on the two keys. Pairs ticked `unsure` never become a
  `PairLabel`. They go to `<out>.unsure.jsonl` in the input shape, ready to be
  made into a sheet again. Text after `note:` goes to `<out>.notes.jsonl` with
  the item's id, sheet and answer.
- **The summary** gives the sheets and items, how many are labelled and
  unlabelled, and the count for each answer.

## From the shell

```bash
odke eval ground --describe                        # what to label, and what to predict
odke eval ground --labels ground.jsonl --predictions facts.jsonl
odke eval resolve --labels pairs.jsonl --predictions links.jsonl --json
odke eval route --labels route.jsonl --run mypackage.routers:MarketingRouter
odke eval extract --labels gold.jsonl --run mypackage.extract:MyExtractor \
    --documents documents.jsonl --ontology schema.json
odke eval validate --labels verdicts.jsonl --run mypackage.gate:MyGate --ontology schema.json
odke eval ablation --config examples/e2e/odke.yaml --labels examples/e2e/gold.jsonl
odke eval spans --facts out/                        # no labels: width by verdict
odke eval extract --labels gold.jsonl --predictions facts.jsonl --items b.items.jsonl
odke eval compare a.items.jsonl b.items.jsonl       # better, worse or inconclusive
odke eval extract --labels gold.jsonl --predictions facts.jsonl --report report.json
```

- The CLI covers route, extract, ground, resolve, score and validate, and the
  ablation, which takes `--config` and `--labels` and nothing else.
- `spans` takes `--facts` and no labels: a run's `facts.jsonl`, or the directory
  a sink wrote it into.
- `--run package.module:Name` imports a stage and runs it over the labels. A class
  is instantiated with no arguments. Otherwise, point it at a module-level
  instance.
- Resolution cannot run from labels, because a resolver takes facts, not pairs.
  Its links always come from a file, which is also how a platform's merges arrive.
- `compare` takes two runs' `--items` files and exits 1 when B is worse
  ([Was the change real?](#was-the-change-real)).
- `--report PATH` writes the [eval report](#the-eval-report) for any stage.
- Errors exit with status 2.
