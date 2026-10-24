# Evaluate a pipeline

The Evaluator (`openodke.eval`, `odke eval`, [`odke bench`](benchmarks.md))
scores a pipeline against your labels or a benchmark's. For each stage it ships
a label format and an evaluator, on the base install
([DECISIONS #26](decisions.md#26)). With no gold, see
[Labels and judges](labels.md).

## Point it at your pipeline

`odke eval pipeline` runs any extraction pipeline and scores its triples:

| Mode | Flag | What runs |
|---|---|---|
| command | `--cmd "python my_extract.py {in} {out}"` | the command; `{in}` is a folder with one `<name>.txt` per document, `{out}` the file it must write |
| callable | `--run my_pkg.extract:run`, or `--run path/to/file.py:run` | the function, called with the `Document`s; it returns triples rows, `TripleRow`s or `Fact`s |
| files | `--predictions out.jsonl` | nothing: the output is already written |

Score it against `--labels gold.jsonl --documents texts/`, or a `--bench` set
from `odke bench prepare`, by that benchmark's metrics.

```bash
odke eval pipeline --labels examples/triples/gold.jsonl \
    --documents examples/triples/texts --ontology examples/triples/ontology.json \
    --cmd "python examples/triples/pipeline.py {in} {out}" --report report.json
odke eval pipeline --bench runs/movie --run my_pkg.extract:run --validator
```

| Flag | |
|---|---|
| `--adapter` | `langchain`, `langextract` or `graphrag` output instead of [triples](inputs.md); a file of `Fact` rows is read as facts |
| `--documents` | a folder (a file's id is its path in it, `halden.txt`) or `Document` JSONL |
| `--timeout` | seconds `--cmd` may run (600); it runs with no shell |
| `--validator` | add a `+ validator` row: what the [Validator](validator.md) writes from the same output, its stages from `--config` (a bench set's `odke.json` by default) |
| `--adjudicate`, `--lenient` | an [adjudicated](#is-the-gold-complete) or [lenient](#surface-forms-the-lenient-score) score beside the strict one |
| `--trace`, `--reference`, `--examples` | inputs to the [diagnosis](#where-does-it-lose-facts) of the last row |
| `--items`, `--report`, `--json` | per-document counts for [`compare`](#was-the-change-real); the [eval report](#the-eval-report) written or printed |

Exit 2 is unreadable input; exit 1 a provider error, or a command that fails,
times out or writes nothing at `{out}`.

## `StageReport`

Every evaluator returns one: `stage`, `n`, `metrics`, `breakdown` (per label,
predicate, verdict or bin), `confusion` and `notes`. `render()` is what
`odke eval` prints. An undefined precision is `None` (shown `—`).

## The eval report

`odke eval --report PATH` and `odke bench run` (as `report.json`) write an
`EvalReport`: one JSON document, versioned by `schema_version`
([Stability](stability.md#formats)). Its schema,
`openodke/eval/eval_report.schema.json`, lists every field; `check_report(data)`
lists every mismatch, and each report is checked when written and read.

| Key | What it holds |
|---|---|
| Key | What it holds |
|---|---|
| `schema_version` | `"1.4"` |
| `title`, `n`, `run` | what was scored and how many rows; the version, models, prompts, each grounder prompt's claim format (`claim_formats`, 1.3) and dataset |
| `bootstrap` | how the ranges were drawn |
| `rows[]` | one per configuration scored against gold: `performance` (`precision`, `recall`, `f1`, each `{value, low, high}`, `micro` or `macro`), `counts` (`hits`, `over_extraction`, `under_extraction`, `uncited`, …), `conformance`, `hallucination`, `cost`, `latency` |
| `stages[]`, `notes[]` | the `StageReport`s; what a number cannot say |
| `comparison` | [`compare`](#was-the-change-real)'s result |
| `diagnosis[]`, `fixes[]` | [where it loses facts](#where-does-it-lose-facts) and [what should change](#what-should-change); the `linked_other` bucket is 1.4 |
| `calibration` | each model judge the run called, with its [calibration card](calibration.md)'s line or UNCALIBRATED (1.3) |
| `judged_precision`, `pooled_recall` | [precision](labels.md#precision-the-judge-corrected) and [recall](labels.md#recall-a-pool-of-pipelines) without gold, when asked for |
| `adjudication`, `lenient` | the [adjudicated](#is-the-gold-complete) and [lenient](#surface-forms-the-lenient-score) scores, when asked for |

Ranges are 95% bootstrap ranges over documents, since a document's facts share
one reading of it: 2,000 draws, seed 0, as for [`compare`](#was-the-change-real).
A spurious prediction citing no document counts in every draw and in
`uncited`, but in no document's `--items` row.

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

## Where does it lose facts?

`odke eval pipeline` and `extract` put each gold fact their last row missed in
the first bucket its evidence fits ([DECISIONS #38](decisions.md#38)), with
`--examples` (3) of each.

| # | Bucket | The miss, when | Read from |
|---|---|---|---|
| 1 | relation never offered | the extractor was not shown the relation for the subject's type | `--trace` |
| 2 | refused by the Validator | the gate refused a matching prediction | `--validator` |
| 3 | wrong relation | a prediction links the pair with another relation, and is not itself a hit | predictions |
| 4 | inverse direction | a prediction links the pair the other way round | predictions, ontology |
| 5 | surface form | a prediction with the relation and one end is the gold fact in other words | predictions, `--lenient` |
| 6 | same triple, scored apart | a type, polarity or qualifier differs, or it paired with another gold fact | predictions |
| 7 | cross-sentence | nothing came near, and the evidence spans sentences | the dataset, else the text |
| 8 | output saturation | nothing came near, output is flat against length, and the document is long | the texts |
| 9 | pair linked by another true relation | a prediction that is a hit links the pair with another relation | predictions |
| 10 | entity never extracted | an end is in no prediction for the document | predictions |
| 11 | both seen but not linked | both ends are, never in one prediction | predictions |

- **`--trace`** is the run's `manifest.json`, or the run config that made the
  predictions (its extractor's snippets are read, with no call).
- **Surface form** uses the [lenient](#surface-forms-the-lenient-score)
  pre-filter and, with `--lenient`, its decisions. Without a judge it prints
  *(unconfirmed)*: the other ends' name keys are equal, one contains the other
  as whole words, or they are 85% alike. `FactJudge.equivalent` is the hook.
- **Cross-sentence** reads the dataset's evidence sentences, else whether one
  sentence names both ends, by the [locator's](grounding.md#locating-spans)
  rules.
- **Output saturation** regresses extractions and gold per document on length,
  log on log; under half the gold's rate, long documents' misses land here.
- **A bucket the run cannot measure is `—`** (`null`).

False positives split into those the gate refused, those supported by their
passage (perhaps missing from the gold; the `Adjudicator` hook), and the rest.

```bash
odke eval pipeline --bench runs/cmp/redocred --predictions snippet25.jsonl \
    --trace runs/cmp/redocred/odke-snippet25.json --report s25/report.json \
    --items s25/items.jsonl
```

```text
where it loses facts  (pipeline: 1,574 misses)
                              misses  share
  relation never offered      363     23.1%
  refused by the Validator    —       —
  wrong relation              82      5.2%
  inverse direction           0       0.0%
  surface form (unconfirmed)  41      2.6%
  same triple, scored apart   6       0.4%
  cross-sentence              680     43.2%
  output saturation           219     13.9%
  entity never extracted      144     9.1%
  both seen but not linked    39      2.5%

  e.g. relation never offered: test_0001: Orlov · educated at · St Petersburg University (not offered for Person)
```

## What should change?

Each bucket maps to a fix and its knob, ranked by expected recall gain
([DECISIONS #38](decisions.md#38)). Gains overlap, so they never add.

| Fix | Buckets | Knob or command | Expected gain |
|---|---|---|---|
| `offer-relations` | never offered | `stages.extractor.snippet_limit`, `types` | misses × recall on the offered relations / gold |
| `audit-refusals` | refused | `odke eval refusals`, `stages.gate.refuse_not_found` | exact: accept every refusal and rescore; precision too |
| `relation-descriptions` | wrong relation | each predicate's `description` and `examples` | misses × the row's recall / gold |
| `inverses` | inverse direction | `inverse_of`, `symmetric`, `inverses: true` | exact: the inverse step replayed; precision too |
| `normalise-names` | surface form | `stages.normalizer`, `Entity.aliases` | 0 until a judge confirms; confirmed / gold after |
| `check-details` | scored apart | the ontology's types and qualifiers | misses × the row's recall / gold |
| `wider-context` | cross-sentence | `stages.chunker` | misses × same-sentence recall / gold |
| `smaller-chunks` | output saturation | `stages.chunker`, `models.extract.max_tokens` | misses × the short documents' recall / gold |
| `every-relation` | pair linked by another true relation | the extraction prompt; each predicate's `description` | misses × the row's recall / gold |
| `reextract` | entity never extracted, not linked | `reextract:` | misses × the row's recall / gold |

`expected` is exact where the fix replays, else assumes the bucket is found as
often as the facts it does not touch (`basis` shows how). `ceiling` is every
miss found; `reference` is the gain at another system's rate
(`--reference FILE`).

```text
what should change  (pipeline: ranked by expected recall gain; gains overlap)
     fix                    expected  ceiling  reference  knob
  1  wider-context          +6.4      +38.9    +3.7       stages.chunker
  2  offer-relations        +2.6      +20.8    +7.3       stages.extractor.snippet_limit
  3  smaller-chunks         +1.5      +12.5    +1.9       stages.chunker; models.extract.max_tokens

  2. Offer every relation the subject's type can take: raise the snippet limit, or name the types. 363 misses × 12.4% (its recall on the relations it was offered: 172 of 1,383) / 1,747 gold
     - Misc · producer: 35
```

**The track record.** `--report` appends the fixes, the run's `--items` hash
and its config to `track-record.jsonl` beside it (`--track-record`). When
`compare A B` finds B applied a fix A predicted (`--applied <fix-id>`, or
configs differing at that fix's knobs alone), it appends what B measured.

```bash
odke eval pipeline ... --trace odke-snippet25.json --report s25/report.json --items s25/items.jsonl --track-record record.jsonl
odke eval pipeline ... --trace odke.json --report s96/report.json --items s96/items.jsonl --track-record record.jsonl
odke eval compare s25/items.jsonl s96/items.jsonl --track-record record.jsonl
```

```text
track record: offer-relations expected +2.6 (ceiling +20.8), measured +4.5 [+2.0, +7.4] better (config: stages.extractor.snippet_limit (unset) → 96)
```

In Python: `gold_view` (or `diagnosis_view`), `diagnose`, `fixes`.

```python
from openodke import Entity, Evidence, Fact
from openodke.eval import GoldFact
from openodke.eval.diagnosis import Offered, diagnose, gold_view
from openodke.eval.fixes import fixes


def said(subject, predicate, obj, doc="d1"):
    return Fact(
        subject=Entity(key=subject, type="Person"),
        predicate=predicate,
        object_entity=Entity(key=obj, type="Thing"),
        evidence=(Evidence(doc_id=doc),),
    )


gold = [
    GoldFact(doc_id="d1", fact=said("ada", p, o))
    for p, o in [
        ("born_in", "london"),
        ("employer", "acme"),
        ("award", "medal"),
        ("award", "prize"),
    ]
]
view = gold_view(gold, [said("ada", "born_in", "london"), said("ada", "founded", "acme")])
found = diagnose(view, offered=Offered(by_type={"Person": ("born_in", "employer")}))
print({b.bucket: b.count for b in found.buckets if b.count})
# {'never_offered': 2, 'wrong_relation': 1, 'written_other': 1}
best = fixes(found)[0]
print(best.id, round(best.recall.expected, 3), best.recall.basis)
# offer-relations 0.25 2 misses × 50.0% (its recall on the relations it was offered: 1 of 2) / 4 gold
```

## Formats and evaluators, per stage

| Stage | Label row | Prediction row | Evaluator | Headline metrics |
|---|---|---|---|---|
| route | `RouteLabel(id, text, action, label)` | `RoutePrediction(id, action, label)` | `evaluate_routing`, `run_route` | per-label P/R/F1, confusion, `facts_skipped` |
| extract | `GoldFact(doc_id, fact)` | `Fact` | `evaluate_extraction`, `run_extract` | per-predicate P/R/F1 |
| ground | `GroundingLabel(text, fact, verdict, doc_id)` | `Fact` (optional) | `evaluate_grounding`, `grounding_ablation`, `run_ground` | accuracy, `false_support`, `lost_support` |
| resolve | `PairLabel(a, b, same)` | `LinkRow(a, b, kind)` or `EntityLink` | `evaluate_resolution`, `links_from_clusters` | pairwise P/R/F1, B-cubed |
| score | `CalibrationLabel(fact, true)` | `Fact` (optional) | `evaluate_calibration`, `run_score` | Brier, reliability curve, ECE |
| validate | `ValidationLabel(fact, action)` | `ValidationPrediction(id, action, reason)` | `evaluate_validation`, `run_validate` | agreement, kappa |
| spans | none | a run's `facts.jsonl` | `evaluate_spans` | span width per verdict |
| sink | none | none | `check_idempotency`, `assert_idempotent`, `jsonl_counts` | counts unchanged across writes |
| cost | none | none | `CostMeter`, `CostReport`, `compare_costs` | calls, tokens, USD, latency |
| ablation | `GoldFact(doc_id, fact)` | a run config | `run_ablation` | P/R/F1 three ways |

Predictions join labels on `Fact.id`, so a sink's `facts.jsonl` and
`links.jsonl` are predictions files; give a hand-written fact an `"id"`, or use
a `run_*` helper. `odke eval <stage> --describe` prints a stage's contract.

```json
{"id": "c1", "text": "Ada Lovelace was born in London in 1815.", "action": "extract", "label": "fact"}
{"doc_id": "d1", "fact": {"subject": {"key": "ada", "type": "Person"}, "predicate": "born", "object_value": 1815}}
{"a": "a", "b": "b", "same": true}
```

## extract

Each miss is a **wrong value**, a **wrong entity**, **missing** or
**spurious**; a wrong value or entity is a false positive and a false negative.
Matching is by `Fact.signature` with literals normalised (whitespace, case,
numbers), so `1815` and `"1815"` match; a flipped polarity never does.

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

`false_support` counts unsupported facts stamped `supported`; `lost_support`,
true facts stamped anything else; `unchecked` is scored wrong.
`grounding_ablation(labels, grounded, *, drop=...)` gives the precision kept
with grounding off and on (dropping `contradicted` and `not_found` by default),
and the recall it cost, both over the same set: the labelled facts that were
grounded. A label with no grounded fact is left out and noted.

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

`odke eval spans` needs no labels: it reports a run's `Evidence.span` widths
per verdict and the share citing no span. A citation too narrow for its claim
comes back `not_found` ([Evidence](concepts.md#spans-evidence-and-trust)), so
when narrow citations cluster there, the `not_found` rate measures citation
quality. The summary names the gap only when the `not_found` upper quartile is
below the `supported` lower one. [Located](grounding.md#locating-spans) spans
are counted apart.

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

```bash
odke eval spans --facts out/facts.jsonl   # or --facts out/, the directory a sink wrote
odke eval spans --describe
```

`evaluate_spans(kg.facts)` takes a graph's facts; `load_facts(path)` reads the
JSONL.

## resolve

Pairwise P/R asks, per labelled pair, whether the links join the two keys;
B-cubed sees a wrong merge from every member's side. Links are `(a, b, kind)`
from any source, `same_as` chains followed; `links_from_clusters` turns a
store's merged groups into links. `similar` is no decision unless
`similar_as_same=True`.

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

**Brier** is mean((confidence − outcome)²), beside `brier_baseline`, always
answering the base rate. The **reliability curve** has ten equal bins of mean
confidence against the share true; **ECE** is their weighted gap.

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

Scores a gate: `wrongly_refused`, `wrongly_written`, `conflicts_missed`, and
agreement as accuracy and Cohen's kappa.

## sink

Write a graph twice and count once; the count is a callable you pass.
`jsonl_counts` is the one for `JsonlSink`.

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

`CostMeter.client(inner, stage=...)` records each call's tokens, `cost_usd` and
latency by stage. Any unpriced call makes `cost_usd` `None`; `priced_usd` is the
partial sum.

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

`compare_costs({"hybrid": a, "model-only": b})` sets runs side by side.

## ablation

`run_ablation(config, gold)` runs one [run config](run.md) three ways against
one labelled extraction set:

1. **extraction alone**: the config's loaders, chunker, router and extractor,
   every candidate kept;
2. **+ grounding**: the same candidates through its grounder and gate (or
   `VerdictGate`);
3. **+ corroboration**: the whole pipeline (normalise, resolve, corroborate,
   score), then the same gate.

Each stage runs once, later rows replaying earlier facts, and nothing is
written. Rows are `evaluate_extraction` per document, by the
[ids `odke run` gives](run.md#inputs); a fact merged across documents scores in
each. The notes say what the gate kept and what refusing `not_found` would
change.

```bash
odke eval ablation --config odke.yaml --labels gold.jsonl
odke eval ablation --describe
```

On [`examples/e2e`](https://github.com/deepskandpal/odke/blob/main/examples/e2e/README.md)
(recorded answers and labels written for each other):

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

## Was the change real?

`odke eval compare A B` compares two runs on the same items with a **paired
bootstrap** (Koehn 2004): the 95% interval of the difference over resampled
items, documents for corpus P/R/F1.

| Verdict | When | Exit |
|---|---|---|
| better | the whole 95% interval is above zero | 0 |
| worse | the whole 95% interval is below zero | 1 |
| inconclusive | the interval contains zero, or, for right-or-wrong items, McNemar's exact p on the flips is 0.05 or more | 0, or 1 with `--fail-on-inconclusive` |

Inconclusive comes with the **detection limit**, the smallest change the set
detects (95% confidence, 80% power): Z·√(d/n), with Z = 1.959964 + 0.841621 and
d the share of items that flip, or Z times the bootstrap's standard error for a
corpus metric ([DECISIONS #29](decisions.md#29)). One metric decides (F1 for
documents, accuracy for items, or `--metric`); the rest are guardrails. Four
items that all flip one way leave the interval clear of zero one time in eight
by chance, so a right-or-wrong verdict needs McNemar's exact test to agree.

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

`--items` (extract, ground, validate, route, pipeline) writes
`{"id": "d1", "tp": 3, "fp": 1, "fn": 0}` per document or
`{"id": "g7", "correct": true}` per item; both files must hold the same ids.
`--resamples` (2,000) and `--seed` (0) fix the interval.

**In CI**, compare the head's items against the base run's:

```bash
odke eval extract --labels gold.jsonl --predictions out/facts.jsonl --items head.items.jsonl
odke eval compare base.items.jsonl head.items.jsonl --fail-under 0.80
```

It exits 1 on worse, with a `gate: fail:` line per reason, and with
`--fail-under 0.80` when the low end of B's range is under 0.80. Exit 2 is
files that cannot be compared. `tests/test_eval_aa.py` checks the false-alarm
rate on seeded A/A runs.

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

`compare_items` takes two runs' `ItemRow`s; `bootstrap_interval` gives one
run's range.

## Is the gold complete?

Text2KGBench and Re-DocRED leave out facts their texts state, so precision
against them is understated. `--adjudicate` (with `--labels`) finds the
predictions that may be missing from the gold:

```bash
odke eval pipeline --labels gold.jsonl --documents texts/ --predictions out.jsonl \
    --config odke.yaml --adjudicate adjudicated.jsonl
```

```text
gold adjudication  (each prediction the gold lacks grounded 3 times; supported in 2 is possibly missing from gold)
            strict                adjudicated           not in gold  asked  possibly missing
  pipeline  0.600 [0.600, 0.600]  0.800 [0.800, 0.800]  2            2      1
  the strict precision never changes; the list, with every verdict: adjudicated.jsonl
```

- Each prediction the gold lacks is grounded three times against its document
  (`--config`'s grounder, else the default; `ModelSpec.repeat` 0–2, keyed by
  the [cache](models.md#the-response-cache)). Supported twice, it is
  **possibly missing from gold**.
- The adjudicated precision counts those as hits, beside the strict one, which
  never changes.
- The list has a line per prediction: `doc_id`, `claim`, `kind`, `rows`,
  `verdicts`, `supported`, `possibly_missing_from_gold`, `fact`.

## Surface forms: the lenient score

"Gabby Logan" for "Gabrielle Nicole Logan" is a miss and a false positive to a
string matcher. `--lenient PAIRS` (with `--labels`) asks a model whether such a
near miss is the gold fact in other words, and prints a lenient score beside
the strict one ([DECISIONS #39](decisions.md#39)). Provisional
([Stability](stability.md#provisional)).

```bash
odke eval pipeline --labels gold.jsonl --documents texts/ --predictions out.jsonl \
    --config odke.yaml --lenient pairs.jsonl
```

- **The pre-filter** (`candidate`, `surface_pair`), with no model, pairs a
  missed gold fact and an unmatched prediction in one document with the same
  relation and polarity and one end equal after normalisation.
- **The judge** (`FactJudge`) reads the relation, both facts and the gold
  fact's evidence sentences, sending `fact_equiv@1` to the `ground` role's
  model. "same" counts only when both orders say so.
- **The lenient score** makes each such pair a hit, once per gold fact and
  prediction. The pairs go to the path given, with both answers and `why`.

```python
from openodke import Document, Entity, Evidence, Fact
from openodke.eval import GoldFact
from openodke.eval.equivalence import FactJudge, lenient
from openodke.llm import RecordedClient

ada = Entity(key="ada", type="Person", label="Ada Lovelace")
doc = Document(id="a1", text="Ada Lovelace was born in London on 10 December 1815.")
cited = (Evidence(doc_id="a1"),)
gold = [GoldFact(doc_id="a1", fact=Fact(subject=ada, predicate="born", object_value="1815"))]
said = [Fact(subject=ada, predicate="born", object_value="10 December 1815", evidence=cited)]

# Recorded answers: "same" in both orders.
judge = FactJudge(client=RecordedClient([{"match": "Fact 1", "response": {"decision": "same"}}]))
section, pairs = lenient([("extract", said)], gold, [doc], judge)
(row,) = section.rows
assert (row.candidates, row.same, row.counted) == (1, 1, 1)
assert (row.strict.recall.value, row.lenient.recall.value) == (0.0, 1.0)
assert pairs[0].passage == "Ada Lovelace was born in London on 10 December 1815."
```

Until its calibration card on label set F exists, the lenient score is printed,
not trusted.

## What the grounder refused

`odke eval refusals` samples what a grounder refused for a person to judge.

```bash
odke eval refusals out/ --documents texts/ --make-sheet refusal-sheets/ --n 100
odke label read refusal-sheets/ -o refusal-labels.jsonl
odke eval refusals --labels refusal-labels.jsonl
```

- **It reads** `odke ground` output (facts the free checks stamped, or
  `contradicted` or `not_found`), an `odke validate -o` directory's
  `refused.jsonl`, or bench sets (triples the gated row lacks). Ground and
  validate output need `--documents`.
- **The sample** shares `--n` (100) equally across strata (dataset, extractor,
  verdict), predicates taking turns, seeded by `--seed` (0); `refusals.jsonl`
  beside the sheets keeps the draw.
- **An item** shows the claim, the verdict and reason, any gold, and the text,
  with boxes `refusal correct`, `refusal wrong`, `gold wrong`, `unsure`.
- **The refusal precision** is right refusals (`refusal correct`, `gold wrong`)
  over every judgement but unsure, overall, per dataset, extractor and
  verdict, and per stratum. Equal shares draw a small stratum far more often
  than its size says, so every drawn row records its stratum's size, and a row
  that spans strata weights each stratum by it, with a Wilson 95% interval on
  the sample's effective size (stratified variance). Pooled as one random
  sample instead, 1,000 refusals 90% right and 50 that are 30% right read as
  62% against a true 87%. Labels with no sizes are reported per stratum only,
  unless the runs they were drawn from come too:
  `odke eval refusals RUN... --labels LABELS`.

```python
from openodke.eval import RefusalLabel, report_refusals


def label(judgement: str, gold: bool) -> RefusalLabel:
    return RefusalLabel(
        id=judgement,
        claim="Loud — performer — Rihanna.",
        text="…",
        verdict="not_found",
        reason="the passage does not settle it",
        predicate="performer",
        gold=gold,
        judgement=judgement,
    )


labels = [label("refusal_correct", False)] * 6 + [label("gold_wrong", True)] * 2
labels += [label("refusal_wrong", True)] * 2 + [label("unsure", False)]
report = report_refusals(labels)
assert "refusal precision 0.800 [0.490, 0.943] (Wilson 95%), 10 judged, 1 unsure" in report.render()
```

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
odke eval precision --facts out/facts.jsonl --labels labels.jsonl   # no gold: judge + sample
odke eval pool runs/a runs/b --documents texts/     # no gold: recall relative to a pool
odke eval compare a.items.jsonl b.items.jsonl --applied offer-relations   # record what B measured
odke eval pipeline --labels gold.jsonl --documents texts/ --predictions out.jsonl \
    --lenient pairs.jsonl                           # near misses judged; strict stays

odke eval refusals runs/cmp --make-sheet sheets/ --n 100   # what the grounder refused, to tick
odke eval refusals --labels refusal-labels.jsonl    # refusal precision, strata weighted by size
```

`--run package.module:Name` runs a stage over the labels (a class is built
with no arguments); resolution reads its links from a file. Exit 2 is
unreadable input; exit 1 a provider error, a failed pipeline, or a failed
`compare` gate.
