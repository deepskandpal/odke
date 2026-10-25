# Calibration cards

openodke asks a model three kinds of question: does this passage support this
fact (the grounder), do these two mentions name one entity (the pair judge),
and is this prediction the gold fact in other words (the fact-equivalence
judge). Each answer is a measurement, and a measurement is only as good as its
agreement with a person. A **calibration card** is that agreement, for one
registered prompt on one dated model, against one version of one hand-labelled
set. Every run report prints each judge's card line, or **UNCALIBRATED** where
there is none.

The first cards were measured on 25 October 2026 and ship with the package:
Claude Haiku 4.5 (the default grounder model), Claude Haiku 5.5, Claude
Sonnet 5.5 and gpt-5.4-mini, on the maintainer's hand labels. Every grounder
and pair-judge card is below its bar; every fact-equivalence card meets it.
[The results](#the-results) are at the end of this page.

## What a card holds

A card names what it measured, and changes with none of it:

- the prompt key and each registered prompt's SHA-256 ([DECISIONS #27](decisions.md#27));
- the dated model id, `anthropic/claude-haiku-4-5-20251001`, never an alias;
- for a grounder, how the claim was written: `inline` or `labelled` for
  `ground.span`, `sentence` for its challengers, `triple` for paper mode;
- the label set's name, version, content hash, split and size;
- the date, the openodke version, and the cost per 1,000 items.

A change to the prompt version, the model id or the label set expires the card
until it is measured again. Raw agreement is never reported alone: 88%
agreement on a skewed set can be κ 0.27 (Cohen's κ corrects for the agreement
two raters reach by chance).

| Judge | Primary metric, and the bar to be a default | Reported beside it |
|---|---|---|
| Grounder (`ground.span`, `ground.onyx`, `ground.statement`, `ground.paper`) | Cohen's κ on keep against refuse, **≥ 0.6** | three-way κ; fail-recall and false refusals with Wilson intervals; the planted-error catch rate; the abstain rate; the rerun flip rate; results by passage length; the confusion matrix |
| Pair judge (`pair`) | the precision of "same", its **lower 95% bound ≥ 0.9** | the recall of "same"; the unsure rate; the rate the two orders disagree; the label noise of the audit |
| Fact-equivalence judge (`fact_equiv`) | Cohen's κ on same against not, **≥ 0.6** | the unsure rate; the order-disagreement rate; how often the judge and the person call a pair the same |

For a grounder, **keep** is `supported` and **refuse** is `contradicted` or
`not_found`. **Fail-recall** is the share of facts the person did not mark
supported that the judge refuses; **false refusals** the share of facts the
person marked supported that it refuses. An answer the judge did not give
(`unchecked`) counts as kept there, as the gate keeps it, and is left out of κ;
the abstain rate counts it. The planted-error catch rate is on facts made false
by construction (an object swapped, a relation reversed, a denial, a changed
date), reported apart, because they are easier than natural errors. The flip
rate is the share of facts whose verdict changed between runs of the same
question.

The grounder's bar has a second half: κ within 0.1 of the labeller's agreement
with themselves, the ceiling a judge is held to. Until that is measured, a card
says `self_agreement: null`, and its bar is judged on κ alone, which its
`bar.note` says.

## Reading a card's line

```text
calibration   PROMPT on MODEL (CLAIM FORMAT): κ K [LOW, HIGH] (n N), below its bar (κ ≥ 0.6); fail-recall F, false refusals R, planted caught P (label set G gate, n N, DATE); refusals marked for review, not dropped (its card is below the bar, DECISIONS #54)
```

The judge, its card's primary metric with its 95% interval, whether it meets
its bar, the numbers beside it, the labels it was measured on, and what its
answers did in this run, and why. With no card:

```text
calibration   ground.span@1 on ollama/qwen2.5:3b (inline): UNCALIBRATED (no card); refusals marked for review, not dropped (it has no card, DECISIONS #54)
```

`odke validate`, `odke run` and every eval report print one line per model
judge the run called. In JSON they are the run report's and the eval report's
`calibration` entries, each with `prompt`, `model`, `card`, `meets_bar`,
`line`, `action` and `why`.

## Below the bar, nothing is dropped on a judge's word

A judge below its bar, or with no card, still runs, and its answers are kept
and reported ([DECISIONS #54](decisions.md#54)). What changes is what they do:

- **The gate marks instead of refusing.** A fact the grounder refused is
  written, with `qualifiers["odke.review"]` holding the refusal and why it was
  not made. The run report counts them under `marked`. A refusal for the
  schema is code, not a judge, and is always made. A fact a later run grounds as
  supported loses its mark.
- **The pair judge sends every pair it decided to review.** The decision is
  kept, but links and merges nothing; the pair goes to the
  [review queue](resolution-and-corroboration.md#the-pair-judge). A person's
  decision always links.
- **The fact-equivalence judge drops nothing anyway**: its lenient score sits
  beside the strict one, and its line says how far to read the gap.

At or above its bar, a judge acts: the gate drops, the pair judge links. An
option given always wins:

```python
from openodke import Entity, Fact, GroundingVerdict, Ontology, VerdictGate

gate = VerdictGate(on_refusal="mark")  # or "drop"; left out, the grounder's card decides
fact = Fact(
    subject=Entity(key="Person:ada", type="Person", label="Ada"),
    predicate="born",
    object_value="1816",
    verdict=GroundingVerdict.CONTRADICTED,
)
verdict = gate.validate(fact, Ontology())
assert (verdict.action, verdict.review) == ("accept", "on_refusal='mark' given")
assert gate.stats == {"accepted": 1, "refused": {}, "marked": {"contradicted": 1}}
```

`PairJudge(on_decision="link" | "review")` is the pair judge's. Left out, the
default comes from the judge's card through one switch,
`openodke.calibration.policy.DEFAULT_POLICY`: `"card"` reads each card as
above, `"act"` lets every judge act as before #54, `"hold"` lets none. It reads
`"card"`: the default grounder's card (Haiku 4.5, `ground.span@1`) is below its
bar, so a default run marks the facts its grounder refuses, and a pair judge
left to its default sends its pairs to review. A grounder of your own has no card, so under `"card"` its
refusals are marked until you give it one or pass `on_refusal="drop"`.

## A card from your own labels: `odke calibrate`

Label your facts on [sheets](labels.md#labelling-by-hand), read them back, and
ground them with the prompt and model you mean to run:

```bash
odke label read sheets/ -o labels.jsonl
odke calibrate labels.jsonl --config models.yaml --runs 3 -o cards/
odke calibrate labels.jsonl --config models.yaml --prompt ground.onyx -o cards/
ODKE_CARDS=cards/ odke validate --facts triples.jsonl --texts texts/ --config models.yaml
```

`--planted` adds facts false by construction, for the catch rate;
`--label-set`, `--labels-version` and `--split` name the labels on the card.
The card is printed and written as JSON. `ODKE_CARDS` names directories or
files of cards, separated as `PATH` is; a run reads them before the package's
own, so your card for a judge stands in front of a shipped one.

In Python, a card is made from per-item outcomes, so any harness can make one:

```python
from openodke.calibration import JudgedFact, LabelSet, Made, grounding_card

facts = [JudgedFact(id=f"s{n}", label="supported", runs=("supported",)) for n in range(18)]
facts += [JudgedFact(id=f"r{n}", label="not_found", runs=("not_found",)) for n in range(6)]
facts += [JudgedFact(id="x", label="supported", runs=("not_found",))]
card = grounding_card(
    facts,
    Made(
        prompt="ground.span@1",
        prompts={"ground.span@1": "dbc23eec"},
        model="anthropic/claude-haiku-4-5-20251001",
        labels=LabelSet(name="mine", version="1", sha256="-", split="all", n=len(facts)),
        claim_format="inline",
    ),
)
assert card.bar.met and card.grounding.false_refusals.k == 1
print(card.render())
```

## How the package's cards are made

`bench/calibrate.py` makes them, on public data only, from the label sets in
`bench/labels/` ([labels README](https://github.com/deepskandpal/odke/tree/main/bench/labels)):

- **G**, grounding: facts three extractors wrote from Text2KGBench and
  Re-DocRED, labelled by hand, and 100 planted false facts. Split into dev and
  gate by document.
- **R**, entity pairs: Re-DocRED's own mention clusters, with a 100-pair
  audit by hand that overrides the dataset's label where it disagrees; about 1
  in 6 of the free labels is wrong, which the card reports.
- **F**, fact pairs: misses that pass the fact-equivalence pre-filter,
  labelled by hand.

The grounder's default is a contest. `ground.span@1` (inline and labelled),
and two challengers from EnterpriseRAG-Bench (Onyx, arXiv 2605.05253, MIT):
`ground.onyx@1`, its fact validator word for word, and `ground.statement@1`,
the same question answered three ways. Each is compared with `ground.span@1` on
the same gate items, paired, by the Evaluator's own comparison
([Was the change real?](evaluation.md#was-the-change-real)): an item passes when
the judge's keep or refuse matches the person's. A challenger becomes the
default only when it is better by more than the detection limit; an
inconclusive result keeps `ground.span@1`. Dev numbers are reported apart and
never pick a winner. Each prompt runs three times, so the cards carry a flip
rate; above 5%, "refuse only if 2 of 3 runs refuse" is measured as a card of
its own. The runner also measures how often gold adjudication's "supported in
2 of 3 runs" agrees with a person, and, when a non-Claude model runs too,
whether a Claude judge is more lenient with Claude-extracted facts.

Every live run is held to a spend cap: it prices itself from the messages it
would send, less what the response cache holds, and refuses to start past its
provider's cap (`bench/spend.py`, `bench/spend.jsonl`).

## The results

The tables below are the same ones the Benchmarks page shows.

<!-- judge calibration: generated by `bench/calibrate.py --tables`; do not edit -->

### Judge calibration

Each model judge, measured on 2026-10-25 against the maintainer's hand labels: label sets G (grounding), R (entity pairs) and F (fact pairs), public data only (`bench/labels/`). How a card is made and read, and the bars: [Calibration cards](calibration.md). These tables are generated from the cards the package ships (`src/openodke/calibration/cards/`) and `bench/calibration/results.json` by `bench/calibrate.py --tables`, and a test holds them to both.

#### The grounder, `ground.span@1`, claim inline

Label set G's gate split: the maintainer's hand labels on facts three extractors wrote, and 100 planted false facts for the catch rate. Keep is `supported`; refuse is `contradicted` or `not_found`. Three runs of each question.

| Model | n | κ keep vs refuse [95%] | Meets the bar (κ ≥ 0.6) | Flips over 3 runs | Planted caught | False refusals | Adjudication agrees | κ, ≤ 40 words / > 150 words | $ per 1,000 facts |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| `anthropic/claude-haiku-4-5-20251001` | 104 | 0.12 [-0.09, 0.38] | no | 0.02 | 0.80 | 0.09 | 0.82 (28/34) | 0.18 / 0.15 | $0.52 |
| `anthropic/claude-haiku-5-5` | 104 | 0.38 [0.07, 0.66] | no | 0.12 | 0.73 | 0.08 | 0.82 (28/34) | 0.37 / -0.06 | $0.11 |
| `openai/gpt-5.4-mini-2026-03-17` | 104 | 0.39 [0.14, 0.62] | no | 0.11 | 0.82 | 0.11 | 0.82 (28/34) | 0.36 / 0.37 | $0.31 |
| `anthropic/claude-sonnet-5-5` | 104 | 0.40 [0.09, 0.65] | no | 0.05 | 0.75 | 0.07 | 0.85 (29/34) | 0.33 / 0.26 | $1.90 |

Answered with reasoning on, the model's default: `anthropic/claude-haiku-5-5`, `anthropic/claude-sonnet-5-5`. Reasoning counts against the 256-token max_tokens every judge here has, so an answer can stop before its verdict (an abstention), and its cost includes the reasoning: not like for like with a model that answers without it.

Not measured: the maintainer's agreement with themselves, the bar's other half (within 0.1 of it), so each bar here is κ ≥ 0.6 alone. Adjudication agrees: on G's gate facts the dataset's gold lacks, how often "supported in 2 of 3 runs" matches the maintainer's "supported". Cost is measured token use at list price.

#### The pair judge, `pair@1`, both orders

Label set R's gate split: Re-DocRED's own mention clusters, with the maintainer's 100-pair audit replacing a dataset label where it decided one. "Same" counts only when both orders say same.

| Model | n | Precision of "same" [95%] | Meets the bar (lower bound ≥ 0.9) | Recall of "same" | Unsure | Orders disagree | $ per 1,000 pairs |
|---|---:|---|---|---:|---:|---:|---:|
| `anthropic/claude-haiku-4-5-20251001` | 400 | 0.84 [0.79, 0.89] | no | 0.84 | 0.06 | 0.06 | $0.78 |
| `anthropic/claude-haiku-5-5` | 400 | 0.86 [0.80, 0.90] | no | 0.74 | 0.17 | 0.13 | $0.11 |
| `openai/gpt-5.4-mini-2026-03-17` | 400 | 0.83 [0.78, 0.88] | no | 0.85 | 0.09 | 0.09 | $0.45 |
| `anthropic/claude-sonnet-5-5` | – | not run: budget, the $16 Anthropic stop line | | | | | |

Not measured: the maintainer's agreement with themselves. The audit disagreed with the dataset's label on 17 of 96 pairs it decided (0.18): the labels the judge is scored on are noisy.

#### The fact-equivalence judge, `fact_equiv@1`, both orders

Label set F's gate split: the pairs the maintainer decided same or different (the unsure ones are asked and not scored). "Same" counts only when both orders say same.

| Model | n | κ same vs not [95%] | Meets the bar (κ ≥ 0.6) | Unsure | Orders disagree | Judge "same" / person "same" | $ per 1,000 pairs |
|---|---:|---|---|---:|---:|---|---:|
| `anthropic/claude-haiku-4-5-20251001` | 62 | 0.71 [0.51, 0.89] | yes | 0.08 | 0.08 | 0.32 / 0.35 | $0.48 |
| `anthropic/claude-haiku-5-5` | 62 | 0.75 [0.55, 0.91] | yes | 0.10 | 0.10 | 0.31 / 0.35 | $0.09 |
| `openai/gpt-5.4-mini-2026-03-17` | 62 | 0.67 [0.46, 0.85] | yes | 0.06 | 0.06 | 0.31 / 0.35 | $0.27 |
| `anthropic/claude-sonnet-5-5` | 62 | 0.71 [0.51, 0.89] | yes | 0.00 | 0.00 | 0.32 / 0.35 | $1.48 |

Not measured: the maintainer's agreement with themselves.

#### The contest, and model against model

On G's gate split, run 1: an item passes when the judge's keep or refuse matches the maintainer's, compared item by item with the Evaluator's own comparison ([Was the change real?](evaluation.md#was-the-change-real)). A challenger replaces `ground.span@1` only when it is better by more than the detection limit.

| Model | Against | Accuracy | Difference [95%] | Detection limit | McNemar p | Verdict |
|---|---|---|---|---:|---:|---|
| `anthropic/claude-haiku-4-5-20251001` | `ground.onyx@1` vs inline | 0.808 → 0.798 | -1.0 pts [-6.7, +4.8] | 8.1 pts | 1 | inconclusive |
| `anthropic/claude-haiku-4-5-20251001` | `ground.span@1` labelled vs inline | 0.808 → 0.788 | -1.9 pts [-7.7, +3.8] | 8.5 pts | 0.754 | inconclusive |
| `anthropic/claude-haiku-4-5-20251001` | `ground.statement@1` vs inline | 0.808 → 0.827 | +1.9 pts [-3.8, +7.7] | 8.5 pts | 0.754 | inconclusive |
| `anthropic/claude-haiku-5-5` | `ground.onyx@1` vs inline | 0.740 → 0.769 | +2.9 pts [-3.8, +9.6] | 9.7 pts | 0.581 | inconclusive |
| `anthropic/claude-haiku-5-5` | `ground.span@1` labelled vs inline | 0.740 → 0.788 | +4.8 pts [+0.0, +9.6] | 7.1 pts | 0.125 | inconclusive |
| `anthropic/claude-haiku-5-5` | `ground.statement@1` vs inline | 0.740 → 0.817 | +7.7 pts [+0.0, +15.4] | 11.4 pts | 0.0963 | inconclusive |
| `openai/gpt-5.4-mini-2026-03-17` | `ground.onyx@1` vs inline | 0.837 → 0.865 | +2.9 pts [-4.8, +10.6] | 10.4 pts | 0.607 | inconclusive |
| `openai/gpt-5.4-mini-2026-03-17` | `ground.span@1` labelled vs inline | 0.837 → 0.817 | -1.9 pts [-8.7, +4.8] | 9.3 pts | 0.774 | inconclusive |
| `openai/gpt-5.4-mini-2026-03-17` | `ground.statement@1` vs inline | 0.837 → 0.875 | +3.8 pts [-3.8, +11.5] | 10.8 pts | 0.454 | inconclusive |
| `anthropic/claude-haiku-5-5` | `anthropic/claude-haiku-4-5-20251001`, `ground.onyx@1` | 0.798 → 0.769 | -2.9 pts [-9.6, +3.8] | 9.7 pts | 0.581 | inconclusive |
| `anthropic/claude-haiku-5-5` | `anthropic/claude-haiku-4-5-20251001`, `ground.span@1` inline | 0.808 → 0.740 | -6.7 pts [-13.5, +0.0] | 10.4 pts | 0.118 | inconclusive |
| `anthropic/claude-haiku-5-5` | `anthropic/claude-haiku-4-5-20251001`, `ground.span@1` labelled | 0.788 → 0.788 | +0.0 pts [-6.8, +6.7] | 10.1 pts | 1 | inconclusive |
| `anthropic/claude-haiku-5-5` | `anthropic/claude-haiku-4-5-20251001`, `ground.statement@1` | 0.827 → 0.817 | -1.0 pts [-7.7, +5.8] | 9.7 pts | 1 | inconclusive |
| `anthropic/claude-sonnet-5-5` | `anthropic/claude-haiku-4-5-20251001`, `ground.span@1` inline | 0.808 → 0.808 | +0.0 pts [-5.8, +5.8] | 7.6 pts | 1 | inconclusive |
| `openai/gpt-5.4-mini-2026-03-17` | `anthropic/claude-haiku-4-5-20251001`, `ground.onyx@1` | 0.798 → 0.865 | +6.7 pts [+2.9, +11.5] | 7.1 pts | 0.0156 | better |
| `openai/gpt-5.4-mini-2026-03-17` | `anthropic/claude-haiku-4-5-20251001`, `ground.span@1` inline | 0.808 → 0.837 | +2.9 pts [-4.8, +10.6] | 10.4 pts | 0.607 | inconclusive |
| `openai/gpt-5.4-mini-2026-03-17` | `anthropic/claude-haiku-4-5-20251001`, `ground.span@1` labelled | 0.788 → 0.817 | +2.9 pts [-1.9, +7.7] | 7.1 pts | 0.453 | inconclusive |
| `openai/gpt-5.4-mini-2026-03-17` | `anthropic/claude-haiku-4-5-20251001`, `ground.statement@1` | 0.827 → 0.875 | +4.8 pts [-2.9, +12.5] | 10.4 pts | 0.302 | inconclusive |

Self-preference: on the 15 gate facts the maintainer did not mark supported, all extracted by Claude models, `ground.span@1` on `anthropic/claude-haiku-4-5-20251001` refused 3 (0.20) and on `openai/gpt-5.4-mini-2026-03-17` 8 (0.53); paired, McNemar exact p = 0.125, so the gap is inconclusive on this set.

<!-- judge calibration: end -->
