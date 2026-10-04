# Benchmarks

`odke bench` runs openodke on public datasets and scores it with each dataset's
own metrics. It answers a question your own labels cannot: how does this
pipeline do on data anyone can check, next to numbers others have published?

It is also how the ODKE+ paper's claims are tested. The paper reports only
production numbers on a private knowledge graph. Two of them can be checked on
open data (§4, p. 6):

- *"Grounding reduced hallucinated extractions by 35%."*
- *"Corroboration improved factual precision from 91% (raw LLM) to 98.8%
  post-ranking."*

Every run is the [ablation](evaluation.md): extraction alone, + grounding (the
gate), + corroboration. Extraction and grounding are each called once; the later
rows replay them. The report's notes put the paper's two numbers beside yours.

## Three commands

```bash
pip install "openodke[llm,bench]"

odke bench fetch text2kgbench data/t2k --ontology ont_1_movie
odke bench prepare text2kgbench data/t2k --ontology ont_1_movie --out runs/movie \
    --limit 20 --extract-model anthropic/claude-sonnet-5-5 \
    --ground-model anthropic/claude-haiku-4-5 --paper
odke bench run text2kgbench runs/movie
```

- `fetch` downloads the dataset's official files. openodke redistributes none.
- `prepare` writes a directory you can read before spending anything:
  `docs/` (one text file per document), `ontology.json` (the dataset's schema in
  openodke's form), `gold.jsonl`, `dataset.json` and `odke.json`, the run config.
  It calls no model. `--limit` keeps the first N documents for a cheap pilot.
  `--paper` configures [the paper's own grounder and gate](grounding.md#paper-mode-the-odke-grounder-as-written).
- `run` runs the config three ways and prints the table; `--json` for the report.
  Each row carries calls and tokens; the notes give the grounder's verdicts on
  the candidates, because the gate lets an unchecked fact through and a row that
  drops nothing must say whether the grounder confirmed everything or failed.

The prepared config extracts with `structured: false` — no response schema, as
the paper prompts — gives the extractor 16,000 tokens of room for a
reasoning model's thinking, and sets `snippet_limit` to the ontology's size, so
every relation a type can take is in the prompt. The default snippet of 25 would
hide most of Re-DocRED's 96 relations, and another system given the dataset's
schema sees all of them.

From Python: `openodke.eval.datasets.text2kgbench.fetch / prepare / run / score`,
and the same for `redocred`. `score` takes triples from any system, which is how
another tool is compared on the same footing.

## Datasets

### Text2KGBench

Mihindukulasooriya et al., ISWC 2023 —
[github.com/cenguix/Text2KGBench](https://github.com/cenguix/Text2KGBench), data
CC BY-SA 4.0. Given an ontology and a sentence, extract the facts the sentence
states using only the ontology's relations. Two sources: `wikidata_tekgen`
(10 ontologies, 13,474 sentences) and `dbpedia_webnlg` (19 ontologies, 4,860).

The metrics are the benchmark's own `run_eval.py`, checked to agree with it to
four decimal places:

| Metric | Meaning |
|---|---|
| precision, recall, f1 | per sentence on normalised triples (underscores and spaces removed, lowercased), counting only predicted triples whose relation the sentence's gold uses; averaged over sentences |
| onto_conf / rel_halluc | share of predicted triples whose relation is (is not) in the ontology |
| sub_halluc / obj_halluc | share whose subject (object), stemmed, is not in the stemmed sentence plus the ontology's concept labels |

Two differences: every test sentence is scored, an empty answer included (the
original skips a sentence a system wrote no line for), and tokenising uses
NLTK's Treebank tokenizer, which `word_tokenize` wraps, so no tokenizer data is
downloaded. The hallucination metrics need `openodke[bench]`.

### Re-DocRED

Tan et al., EMNLP 2022 —
[github.com/tonytan48/Re-DocRED](https://github.com/tonytan48/Re-DocRED), MIT.
DocRED's Wikipedia passages with 96 Wikidata relations, and with the missing
labels restored that made DocRED's precision unmeasurable. The closest public
stand-in for ODKE+'s own setting: one entity's page at a time.

No entity list is given — the model reads the passage, as in the paper — so the
score is openodke's own: a predicted triple is correct when the relation matches
and the subject and object each name a mention of the gold head and tail
(case and punctuation ignored); each gold fact counts once; micro-averaged over
documents, as DocRED is. An entity whose name is nowhere in the passage is
hallucinated. The ontology's domains and ranges are read off the dev split, so
the test split's labels are never used to build the schema.

## What a run cannot tell you

- **Corroboration across sources.** Neither dataset states one fact in several
  independent sources, which is what the paper's 91% → 98.8% claim is about.
  Re-DocRED gives several mentions inside one document; that is the nearest
  either comes.
- **Gold is incomplete.** A correct fact a dataset did not label counts against
  precision. Audit a sample of "false positives" before reading precision as an
  error rate.
- **Small samples move.** A pilot of 20 sentences is for measuring cost, not for
  conclusions.
