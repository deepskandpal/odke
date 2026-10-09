# bench/labels — sets a person labels by hand

Labels for what no public benchmark scores: whether a passage supports a fact
(G), and whether two mentions name one entity (R). Each set keeps the script that drew it, so it can be drawn again, and
quotes public data only, under its sources' licences ([LICENSES.md](LICENSES.md)).

## G: grounding (#133)

600 facts, each with the text a grounder would read, for a person to tick
`supported`, `contradicted` or `not found`. 500 are real; 100 are planted
false facts, labelled by construction. The real ones are what grounding is
calibrated on: 200 to tune on (dev) and 300 to gate the release on (gate).

### Where the facts come from

The published comparison (PR #105, `bench/run_all.sh`): Text2KGBench's
Wikidata-TekGen test sentences (10 ontologies × 20) and 50 Re-DocRED test
documents, each read by three extractors with `anthropic/claude-sonnet-5-5`:
openodke's `LLMExtractor`, LangChain's LLMGraphTransformer (`lgt`) and
neo4j-graphrag (`neo4j`). Each extractor's own triples are used, before any
grounding (`extraction-alone.jsonl`). Re-DocRED's own `test_revised.json` gives
the evidence sentences its gold lists.

```bash
uv run python bench/labels/make_g.py "$CMP" data/redocred/test_revised.json   # seed 133
```

No model is called, and the same inputs and seed write the same bytes.

### How they were drawn

- **Gold status**, by each dataset's own scoring in `openodke.eval.datasets`:
  `match` (the scorer counts the triple as a gold fact), `not_in_gold` (it
  does not), `gold_only` (a gold fact no extractor wrote, in any of the three
  configurations). 100, 100 and 50 per dataset. Text2KGBench only scores a
  triple whose relation the sentence's gold uses; `scored: false` marks the 49
  of its 100 `not_in_gold` items it would not score at all.
- **Extractor**: a third of each dataset's `match` and `not_in_gold` items per
  extractor; the odd item goes to each in turn. A triple two extractors wrote
  is drawn once, for one of them, and `also_written_by` names the others. No
  two items make the same claim about one document, and no gold fact is shown
  twice.
- **Planted**: 50 per dataset, from gold facts no real item shows, ten of each
  corruption: `object_swap` (another entity of the object's type from the same
  document), `subject_object_swap` (on a relation that is not symmetric),
  `relation_swap` (another relation with the same range whose domain takes the
  subject), `polarity` (the gold fact, denied) and `value_change` (a year moved
  by up to four, or else a number changed, to a value the text does not
  contain). A corruption that is itself gold by the dataset's scoring, or that
  any extractor wrote, is dropped. So is a swap between relations one passage
  often states together (country and located-in, director and screenwriter,
  publication date and inception, and the rest of `NEAR`), since it could be
  supported. Text2KGBench allowed only 9 object swaps; a relation swap took the
  tenth place. A planted item may be labelled `contradicted` or `not_found`
  (`expected`). One the owner ticks `supported` is a corruption to review, not
  a mistake to count.
- **Text**: the whole document, since no extractor here cited a span. The
  competitors cite nothing, and openodke's saved predictions keep triples and
  not offsets. So every fact's evidence is its whole text, marked `context`,
  and nothing is bold. A Re-DocRED document over 350 words is cut to the
  sentences gold lists as the fact's evidence, plus one either side. A fact with
  no listed evidence is cut to where both its names occur, or else to where the
  less mentioned one does. Only `test_0043` (442 words) was cut, six times;
  `trimmed` and `sentences` record it.
- **Names and types**: a gold name is spelled as the text spells it
  (`Assassin's Creed`, not `Assassin 's Creed`, and `2010`, not Text2KGBench's
  padded `01 January 2010`), so tokenised spacing never marks an item as gold.
  Types come from the set's ontology. On Re-DocRED they come from the entity the
  name mentions in the document, else from the schema.
- **Splits**: 200 dev and 300 gate, taken in whole documents, so no passage is
  read in both. Each dataset sends exactly 40% of each gold status to dev, and
  within a status each extractor sends 12 to 14 of its 33 or 34. Planted items
  are their own set, in neither split.
- **Blinding**: all 600 are shuffled together. A sheet shows an opaque id
  (`G-0001`), the claim as the grounder reads it and the passage, nothing
  more. The sidecar beside each sheet holds the row as given: text, fact
  (identity fields, an opaque `id`) and `doc_id`. One thing the claim itself
  gives away: one that opens "It is NOT the case that" is planted, since the
  saved predictions carry no polarity and no real item is a denial.
- **Self-agreement**: 100 real items, in proportion to each stratum, are
  labelled a second time from fresh sheets a week after the first labels.

| Dataset | Gold status | openodke | LGT | neo4j-graphrag | none | Total | dev | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text2kgbench | match | 34 | 33 | 33 | 0 | 100 | 40 | 60 |
| text2kgbench | not_in_gold | 33 | 33 | 34 | 0 | 100 | 40 | 60 |
| text2kgbench | gold_only | 0 | 0 | 0 | 50 | 50 | 20 | 30 |
| redocred | match | 33 | 34 | 33 | 0 | 100 | 40 | 60 |
| redocred | not_in_gold | 34 | 33 | 33 | 0 | 100 | 40 | 60 |
| redocred | gold_only | 0 | 0 | 0 | 50 | 50 | 20 | 30 |
| **all** |  | 134 | 133 | 133 | 100 | 500 | 200 | 300 |

| Planted | text2kgbench | redocred | Total |
|---|---:|---:|---:|
| object_swap | 9 | 10 | 19 |
| subject_object_swap | 10 | 10 | 20 |
| relation_swap | 11 | 10 | 21 |
| polarity | 10 | 10 | 20 |
| value_change | 10 | 10 | 20 |
| **all** | 50 | 50 | 100 |

### Files in `G/`

- `items.jsonl`: the 600 rows given to `odke label make grounding`, in sheet
  order, so row n is item `G-n`. Each is a `GroundingLabel` without its verdict.
- `items.private.jsonl`: per item id, its `fact_id`, `dataset`, `set`, `doc`,
  `triple`, `denied`, `extractor`, `also_written_by`, `gold_status`, `scored`,
  `gold` (the gold fact as the dataset writes it), `planted`, `expected`,
  `split`, `self_agreement`, `trimmed` and `sentences`. No sheet shows it.
- `gate.jsonl`: the gate split's rows, so the prompt-leakage test in
  `tests/test_prompts.py` checks every gate passage.
- `selfagreement.ids`: the 100 ids to label again.

### Labelling

The sheets are not in this repository. They live in the owner's vault at
`deepanshu-kandpal/openodke.dev/labels/G/`: twelve sheets of 50, made with

```bash
odke label make grounding bench/labels/G/items.jsonl -o <vault>/labels/G --per-sheet 50
```

Once ticked, they are read back with `odke label read <vault>/labels/G -o
bench/labels/G/labels.jsonl`, which writes each labelled row as given plus its
`verdict`, and notes beside it. A label joins `items.private.jsonl` on
`fact.id` = `fact_id`.

## R: entity pairs (#151)

600 pairs of mentions, each with the text around it, for the
[pair judge](../../docs/resolution-and-corroboration.md#the-pair-judge): one
entity or two? Every label is the dataset's own, so R costs no labelling. The
owner audits 100 of them by hand, which says how far the dataset's labels can
be trusted. 200 are to tune on (dev) and 400 to gate the release on (gate).

### Where the pairs come from

Re-DocRED's own test file, `test_revised.json` (500 documents), which groups
each document's mentions into entity clusters, with a type and a position for
each mention. Within a document that is the gold: two mentions of one cluster
name the same entity, and two clusters name two.

```bash
uv run python bench/labels/make_r.py data/redocred/test_revised.json   # seed 151
```

No model is called, and the same input and seed write the same bytes.

### How they were drawn

- **Names.** A mention's name is its tokens joined as the text joins them
  (`Assassin's Creed`), keyed by `name_key`. TIME and NUM mentions are values
  and are left out. A cluster is typed by its mentions' majority type (Person,
  Organization, Location, Misc). Two names with one key are never a pair: the
  rules settle them, so they never reach a judge, and such a pair across two
  clusters is more often a gold slip than two entities.
- **Strata**, all within one document and one type:
  - `same` (300): two names of one cluster, `Clapton` and `Eric Clapton`.
  - `hard` (200): two clusters whose names blocking would compare (a shared
    first or last token) or whose name score is at least 0.5: `Maryland` and
    `Maryland State House`.
  - `other` (100): two clusters of the type, any other names.

  The pools hold 1,329, 2,625 and 25,397 name pairs. Each pair is one mention
  of each name, drawn per pair. `score` is the resolver's name score with no
  normaliser, and `band` whether it is in the judge's band, [0.7, 0.9): 92 of
  the 600 are.
- **Context.** Each mention's sentence and one either side, from Re-DocRED's
  own sentences, joined as `odke bench` joins them: what the judge reads.
- **Splits**: a third of the documents (167 of 500) are dev and the rest gate,
  shuffled by seed, so no passage is read in both. Each split's share of each
  stratum is drawn one pair a document a pass, so the pairs spread over 394
  documents, at most three from one.
- **Blinding**: all 600 are shuffled together. A mention's key is its place,
  `test_0291:s7:11-13` (document, sentence, tokens), never its cluster, and
  both sides of a pair show one type. A sheet shows two names, a type and two
  passages, nothing more.
- **The audit**: 100 items, 50 `same`, 40 `hard` and 10 `other`, so hard
  negatives are 40% of the audit against 33% of R. Drawn by seed within each
  stratum, then shuffled.
- **Across documents**: none. Re-DocRED has no ids, so nothing says whether a
  cluster in one document is one in another. Identity across documents comes
  from T-REx with Wikidata ids (#117, in 0.6.0); `Plan.across` is its slot and
  `scope` is `within` on every row today.

| Stratum | Label | dev | gate | Total | In the band | Audited |
|---|---|---:|---:|---:|---:|---:|
| same | same | 100 | 200 | 300 | 52 | 50 |
| hard | different | 67 | 133 | 200 | 40 | 40 |
| other | different | 33 | 67 | 100 | 0 | 10 |
| **all** |  | 200 | 400 | 600 | 92 | 100 |

The 600 are 208 Location, 177 Person, 112 Organization and 103 Misc pairs.

### Files in `R/`

- `items.jsonl`: the 600 rows, in id order, so row n is item `R-n`: `a` and
  `b`, each a mention's `key`, `type`, `label` and `context`, the row
  `odke label make pair` reads and the pair judge's `Mention`.
- `labels.jsonl`: each item's label from the dataset, a `PairLabel` on the two
  keys, in the same order: the `--labels` of `odke eval resolve`.
- `items.private.jsonl`: per item id, its `doc`, `split`, `scope`, `stratum`,
  `same`, `clusters` (Re-DocRED's indices), `names`, `score`, `band` and
  `audit` (its sheet id, or null). No sheet shows it.
- `gate.jsonl`: each gate document's whole text, so the prompt-leakage test in
  `tests/test_prompts.py` checks every gate passage.
- `audit.jsonl`: the 100 audited rows, in sheet order, so row n is `P-n`.

### Labelling

The audit sheet is not in this repository. It lives in the owner's vault at
`deepanshu-kandpal/openodke.dev/labels/R/`: one sheet of 100, made with

```bash
odke label make pair bench/labels/R/audit.jsonl -o <vault>/labels/R --per-sheet 100
```

Once ticked, it is read back with `odke label read <vault>/labels/R -o
bench/labels/R/audit.labels.jsonl`, which writes each labelled pair as a
`PairLabel`, the unsure ones beside it. An audit label joins
`items.private.jsonl` on `audit` = its sheet id, and the dataset's label on the
two keys.
