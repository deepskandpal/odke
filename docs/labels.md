# Labels and judges

On your own documents there is usually no gold. Two measurements still hold: a
judge's precision corrected by a small labelled sample, and each pipeline's
recall relative to what several found together. Both need labels made by hand,
on [sheets](#labelling-by-hand). Scoring against gold is
[Evaluate a pipeline](evaluation.md).

## Precision: the judge, corrected

The grounder, or any judge, grades every fact a pipeline wrote, and its
supported share is a precision biased whichever way the judge leans. A person
labels a random sample, and **prediction-powered inference** corrects the judge
by the mean gap between the labels and the judge on it (Angelopoulos et al.,
*Science* 382, 2023; [DECISIONS #36](decisions.md#36)).

```bash
odke ground --facts triples.jsonl --texts texts/ -o out/   # or any run whose grounder ran
odke eval precision --facts out/facts.jsonl --documents texts/ --make-sheet sheets/ --n 150
odke label read sheets/ -o labels.jsonl                 # after ticking the sheets
odke eval precision --facts out/facts.jsonl --labels labels.jsonl --documents texts/ \
    --report report.json
```

On the example below, 1,000 judged facts with 120 labelled:

```text
precision without gold  (1000 facts judged, 120 labels)
  judge only   0.850                 the judge's verdicts, uncorrected
  corrected    0.783 [0.717, 0.850]  prediction-powered (PPI), 95%
  labels only  0.750 [0.672, 0.828]  the labels alone, 95%
  the judge on the sample: 12 false support, 4 lost support
```

- **Judge only** is the share of `--facts` whose verdict is `supported`; any
  other, `unchecked` included, counts as wrong. It has no range.
- **Corrected** is `(1/N) Σ fᵢ + (1/n) Σ (yⱼ − fⱼ)` over N judged and n labelled
  facts, with PPI's closed-form normal interval, variance
  `Var(y)/N + Var(y − f)·(1/n − 1/N)` (the labelled facts are among the judged
  ones), clipped to [0, 1].
- **Labels only** is `ȳ ± 1.96·s/√n` on the same labels.
- **Under 100 labels** the corrected number is an *uncalibrated estimate*
  (`calibrated` false).
- **The judge on the sample** is `odke eval ground` over the labels.
- **Recall is never claimed.** With `--documents`, the
  [coverage report](grounding.md#what-extraction-left-behind-the-coverage-report)
  stands in for it.
- **Facts count as independent**, so a corpus of a few long documents gets too
  narrow an interval.

`--make-sheet DIR` draws `--n` facts (150) with `--seed` (0), `--by-predicate`
giving each predicate its share. The sample goes to `DIR/sample.jsonl` with
verdicts cleared, and onto grounding sheets. Every judged fact must cite one of
`--documents`. Labels join on `Fact.id`; one naming no judged fact, or repeated,
is left out and counted. `tests/test_eval_ppi.py` checks the interval's
coverage on simulated biased judges.

```python
from openodke.eval.ppi import prediction_powered

# 1,000 facts, 850 judged supported. 120 labelled at random: the judge let 12
# false facts through and lost 4 true ones.
judged = [True] * 850 + [False] * 150
labelled = [(True, True)] * 86 + [(True, False)] * 12 + [(False, True)] * 4 + [(False, False)] * 18
judge, corrected, labels = prediction_powered(judged, labelled)

assert judge.value == 0.85
assert (round(corrected.value, 3), round(corrected.low, 3), round(corrected.high, 3)) == (
    0.783,
    0.717,
    0.85,
)
assert (round(labels.low, 3), round(labels.high, 3)) == (0.672, 0.828)
```

In Python, `judged_precision(facts, labels, documents=...)` returns the
section, `report_precision` the eval report, and `write_sample` draws the
sheets.

## Recall: a pool of pipelines

Two or more pipelines on the same documents can be compared on recall without
gold, as TREC does: pool what they found, and score each against the pool.

```bash
odke eval pool runs/lgt runs/neo4j runs/reference --documents texts/
```

- **Each run** is its facts after grounding (a `facts.jsonl` or a sink's
  directory). Only `supported` facts are pooled.
- **The pool** is every supported fact any run wrote, per document, once per
  signature after normalisation, with each entity by its name key, so
  `ACME, Inc.` and `Acme Inc.` state one fact.
- **Relative recall** is a run's supported facts over the pool's, with a 95%
  bootstrap range over documents; `found by no other` counts the facts only it
  has.
- **It overstates true recall**: the pool misses what every run missed (Zobel,
  SIGIR 1998). The caveat is printed beside the numbers.

## Labelling by hand

`odke label make` writes rows as markdown sheets to tick in any markdown
editor; `odke label read` reads the ticks back as `GroundingLabel`,
`PairLabel`, `FactPairLabel` or `RefusalLabel` rows.

```bash
odke label make grounding to-check.jsonl -o sheets/ --per-sheet 50
odke label make pair pairs.jsonl -o pair-sheets/
odke label make fact fact-pairs.jsonl -o fact-sheets/
odke label read sheets/ -o ground.jsonl        # or one sheet: sheets/sheet-03.md
odke eval ground --labels ground.jsonl --predictions out/facts.jsonl
```

| Kind | A row | Item ids |
|---|---|---|
| `grounding` | a `GroundingLabel` without its `verdict`: `text`, `fact` (with an `id`) and an optional `doc_id` | `G-0001` |
| `pair` | `{"a": {...}, "b": {...}}`, each a `key` and `type`, with an optional `label`, `context` and `aliases`; the [pair judge's](resolution-and-corroboration.md#the-pair-judge) queue is one | `P-0001` |
| `fact` | a `FactPair`: `id`, `relation`, `description`, facts `first` and `second`, `passage`; nothing says which is gold | `F-0001` |
| refusal | a `Refusal`, made by [`odke eval refusals --make-sheet`](evaluation.md#what-the-grounder-refused) | `X-0001` |

Sheets are `sheet-01.md`, `sheet-02.md` and so on, one a sitting
(`--per-sheet`, 50), each with `sheet-NN.items.jsonl` beside it. The same rows
write the same sheets; `make` refuses a directory that holds some. A grounding
item:

```markdown
### G-0007

Acme Cloud (Company) — operates in — "Ireland".

> **Acme Cloud runs data centres in Ireland, Virginia and Singapore.** Its uptime target is 99.9\% and it costs \$5 a seat.

- [ ] supported
- [ ] contradicted
- [ ] not found

note:
```

The claim is `render_claim(fact)`, the cited span is bold, and a `context` span
stays plain. Passage text is escaped, so markdown cannot hide it.

Reading back:

- **Exactly one tick** (`[x]` or `[X]`) is a label; no tick leaves the item
  unlabelled, counted in the summary.
- **Two ticks**, an unknown heading or another mark exit 2, naming the file and
  line; every problem is listed and nothing is written.
- **Labels** go to `-o` in input order: the row plus `verdict`, a
  `PairLabel(a, b, same)`, a fact row plus `same`, or a refusal row plus
  `judgement`. Pairs ticked `unsure` go to `<out>.unsure.jsonl`, ready for
  another sheet; text after `note:` goes to `<out>.notes.jsonl`.
