# Calibration cards

openodke asks a model three kinds of question: does this passage support this
fact (the grounder), do these two mentions name one entity (the pair judge),
and is this prediction the gold fact in other words (the fact-equivalence
judge). Each answer is a measurement, and a measurement is only as good as its
agreement with a person. A **calibration card** is that agreement, for one
registered prompt on one dated model, against one version of one hand-labelled
set. Every run report prints each judge's card line, or **UNCALIBRATED** where
there is none.

No card ships yet. The first are measured on Anthropic's default grounder
model and published with 1.0 (#134, #135); until then every judge reads
UNCALIBRATED.

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
answers did in this run, and why. With no card, as every judge reads today:

```text
calibration   ground.span@1 on anthropic/claude-haiku-4-5-20251001 (inline): UNCALIBRATED (no card); refusals dropped (the default policy is 'act')
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
`"act"` until the first cards are measured, so no run changes before there is a
card to read. A grounder of your own has no card, so under `"card"` its
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
