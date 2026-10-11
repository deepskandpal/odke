# bench — openodke against its competitors

Not part of the package: this folder holds what `pip install openodke` must
never pull in — competitor libraries — and the scripts that compare them with
openodke on the public datasets `odke bench` prepares.

## The comparison

Four extractors on the same prepared sets — the same documents, whole; the same
schema, every relation shown to every system; and the same model with the same
output room, whichever LiteLLM model string the sets were prepared with — and
each one run three ways:

| Configuration | What it is |
|---|---|
| extraction alone | the extractor's own triples |
| + grounding | openodke's grounder and gate on those triples |
| + corroboration | then openodke's corroborator, scorer and gate |

The extractors:

- **openodke** — `LLMExtractor(structured=False)`.
- **LLMGraphTransformer** (LangChain, `langchain-experimental`) — `allowed_nodes`
  and `(head type, RELATION, tail type)` patterns from the ontology, strict mode.
- **neo4j-graphrag** — `LLMEntityRelationExtractor` with a `GraphSchema` of the
  same node types and patterns, no lexical graph.
- **LangExtract** (`langextract`) — `lx.extract` with every relation and its types
  in the prompt description, each document whole as one chunk, and one example
  that shows the output's shape and names none of a dataset's relations. Each
  attribute naming a relation is a triple, read by `openodke.interop.from_langextract`.

Every competitor is held to the schema the way LLMGraphTransformer's strict mode
does it. Their triples reach openodke in its [triples format](../docs/inputs.md),
through the `triples` extract stage. LLMGraphTransformer and neo4j-graphrag quote
nothing, and LangExtract cites only the subject's mention, so each is grounded
against its whole document, which is the paper's own mode: the whole context,
True or False, affirmed facts kept (`odke bench prepare --paper`; the set's
ground model grounds).

Microsoft GraphRAG is not here: its extraction writes free-text entity and
relationship descriptions for community summaries, with no ontology, so it cannot
be scored against a dataset's relations without inventing a mapping.

### One model string for all of them

openodke calls its model through LiteLLM, and so do the competitors here:
LLMGraphTransformer gets a runnable that calls LiteLLM, neo4j-graphrag an
`LLMInterface` that does, and LangExtract a `BaseLanguageModel` that does. So any provider LiteLLM supports runs the whole
comparison — `openai/…`, `anthropic/…`, `gemini/…`, `ollama/…` — and a reasoning
model's thinking never reaches either library's parser: LiteLLM returns the
reply's text. Nothing about what the model is asked changes. Each library runs
its prompt-and-parse path, not its structured one, which forces a tool choice
some reasoning models refuse.

## Running it

```bash
uv venv --python 3.12 bench/.venv
uv pip install --python bench/.venv/bin/python -e ".[llm,bench]" \
    langchain-experimental json-repair neo4j-graphrag langextract

# prepare the sets under $CMP/t2k/ont_* and $CMP/redocred, naming the models once:
odke bench prepare text2kgbench data/t2k --ontology ont_1_movie --out "$CMP/t2k/ont_1_movie" \
    --extract-model openai/gpt-5 --ground-model openai/gpt-5-mini --paper
# (a model with a smaller output cap: add --max-tokens 4000)

CMP=runs/cmp bench/run_all.sh          # openodke + both competitors, then verification
python bench/tables.py runs/cmp        # the tables, priced from LiteLLM's table

# a smoke test of one competitor on three documents:
bench/.venv/bin/python bench/competitors.py extract lgt "$CMP/t2k/ont_1_movie" --limit 3

# inverse and symmetric partners (#106) on the saved predictions, no model:
python bench/inverses.py runs/cmp --out /tmp/inverses
```

`inverses.py` re-scores every saved `predictions/<row>.jsonl` with the partner
of each triple added, by the same step `Pipeline` runs, after declaring the
pairs Wikidata declares among Re-DocRED's relations in a copy of its ontology.
It also sorts each partner by what the gold says of it and of the fact it came
from. No Text2KGBench ontology holds both ends of a pair, which it reports.

Keys come from the environment, under the names LiteLLM reads
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …); `ODKE_ENV_FILE`, or `--env-file` on
`competitors.py`, names a file of `KEY=value` lines to load first. `--model` on
`competitors.py` overrides the set's extraction model, for a deliberate
cross-model comparison.

Each competitor directory keeps its triples (`facts.jsonl`), its model, token
usage and library versions (`usage.json`) and its report; every run keeps its
predictions per configuration (`predictions/`), which is what an audit reads,
and what its extractor refused before grounding (`rejections.jsonl`). A
document a competitor fails on (a bad key fails them all) would score as an
empty answer, so `competitors.py` exits non-zero when any did, and `tables.py`
leaves that set out of the averages and says so.

The published comparison (PR #105) ran on `anthropic/claude-sonnet-5-5`
extracting and `anthropic/claude-haiku-4-5` grounding. No other provider has run
it end to end yet.

## Every extractor, with and without the layer

`layer.py` makes the table in
[docs/benchmarks.md](../docs/benchmarks.md#every-extractor-with-and-without-the-layer)
(#104) from the saved extractions, without extracting again: each extractor raw,
+ grounding (the Validator's ground job and the set's gate), and + resolution
and corroboration (the whole `openodke.Validator`), on every set given.

```bash
python bench/layer.py prepare "$CMP"/t2k/ont_* "$CMP/redocred" "$CMP/trex" --out runs/layer
python bench/layer.py ground runs/layer --estimate          # the price; no call
python bench/layer.py ground runs/layer --budget-usd 1.00 --spend-log spend.jsonl
python bench/layer.py table runs/layer                      # no call
```

| Step | Calls a model | Writes, under `OUT/<set>/<system>/` |
|---|---|---|
| `prepare` | no | the raw triples, a run config that validates them (the set's own, with the Validator's defaults for what it leaves out), and the extraction's metered cost |
| `ground --estimate` | no | nothing: prints what grounding would cost, the cache's answers free and the rest priced by length |
| `ground` | yes, through the response cache (`--cache`, default `OUT/cache`), under one `--budget-usd`; exit 3 when it stops | `ground/`: each fact with its verdict, the summary, and the cost, both now and at list price |
| `table` | no: a question the cache lacks stops it | `removed.jsonl`, and `OUT/layer.json` with every number the Markdown prints |

`prepare` reads openodke's triples from the run's `facts/grounded.jsonl`, or
back from `predictions/extraction-alone.jsonl`; each competitor's from
`competitors/<system>/facts.jsonl`, unless it ran on fewer documents than the
set; and it runs `PatternExtractor` itself. The value normaliser is off, since
the gold keeps the source's words; `--value-normalizer` keeps it on.

Once the calibration contest (#134) picks the default grounder prompt, `ground`
reruns these rows with that default, from the saved extractions and the cache:
set each `OUT/<set>/<system>/odke.json`'s grounder to `"llm"`, then run `ground`
and `table`; only questions the cache has not seen are paid for.

## The span locator

`locator.py` measures the span locator (#112) on a published comparison's
Re-DocRED set: how many of each competitor's facts it places, how wide the
windows are, and, with `--live`, the same facts grounded on their located
window against the saved whole-document verdicts, with a whole-document re-run
of a sample as the noise floor. It reads the saved run and writes only under
`--out`; the rule that decides the default is in its docstring.

```bash
python bench/locator.py "$CMP/redocred" --out out/locator          # free: placement
python bench/locator.py "$CMP/redocred" --out out/locator --live   # + grounding, priced first
```

## Resolving against the store

`store_lookup.py` measures `NativeResolver(lookup=MemoryLookup(...))` (#114,
#149) on Re-DocRED's entity clusters, offline: no model, no database. Gold
identity is within a document, so each document's first half of sentences is
the store and its second half the batch, scoped to the document as its tenant.
It reports link precision and recall, pairwise P/R and B-cubed per `SIMILAR`
threshold, and how many links an unscoped store would send to other
documents. Re-DocRED has no ids, so only the weak path is measured; the
multi-source benchmark (#117) measures proofs and identity across documents.

```bash
python bench/store_lookup.py data/redocred/test_revised.json --out out/store-lookup
```

## Normalising mentions in a batch

`batch_normalize.py` measures `NativeResolver(normalize_batch=True)` (#148) on
label set R, offline and free: each R document is one batch, and each labelled
pair is scored by whether it ended in one entity. It reports precision and
recall of merged pairs on R's dev and gate splits for the resolver as it was
(its `SIMILAR` counted as merged), the batch judge-free, the batch with its
name rules off, and both with a perfect judge (R's labels as `reviewed`, never
called). `--redocred` adds Re-DocRED's test documents as they come, one batch
each, every pair labelled.

```bash
python bench/batch_normalize.py --redocred data/redocred/test_revised.json --out out/batch
```

## T-REx: corroboration and identity across documents

`odke bench prepare trex` builds a set of Wikipedia abstracts in which each gold
fact has a known number of independent sources (#117, and
[docs/benchmarks.md](../docs/benchmarks.md#t-rex)). Both extractors run on it
as on any prepared set, and each run writes `corroboration.json` (the graph,
corroboration off and on, and the resolver's links against the Wikidata ids)
and its facts under `facts/`.

```bash
odke bench prepare trex data/trex --out "$CMP/trex" --documents 80 --judge \
    --extract-model anthropic/claude-sonnet-5-5 \
    --ground-model anthropic/claude-haiku-4-5-20251001 --paper
odke bench run trex "$CMP/trex"
bench/.venv/bin/python bench/competitors.py extract lgt "$CMP/trex" --budget-usd 2.5
odke bench run trex "$CMP/trex/competitors/lgt"

python bench/trex.py lookup "$CMP/trex"                               # no model
python bench/trex.py lookup "$CMP/trex" --judge --budget-usd 0.10     # + the pair judge
python bench/trex.py tables "$CMP/trex" "$CMP/trex/competitors/lgt"   # the published tables
```

`trex.py lookup` measures what a run never exercises: resolving against a store.
Every other abstract's entities go in a `MemoryLookup` and the rest are resolved
against it, with Wikidata ids as gold identity across documents, which
`store_lookup.py` could not have on Re-DocRED.

`competitors.py --budget-usd` stops calling the model once the extraction has
spent that much. The spend is known after a call returns, so up to three calls
in flight can pass it; a document refused counts as failed, and the set is not
scored until it is run again.

## Gold adjudication on label set G

`adjudication.py` measures how often `odke eval pipeline --adjudicate` (#145)
is right. Label set G holds 200 real predictions the public gold does not
count and 100 planted false ones, each labelled by hand. The script sets the
list (supported in 2 of 3 grounding runs) beside those labels: its precision
(how often "possibly missing from gold" is true), its recall (how much of the
gold's gap it finds), per dataset, with Wilson intervals, and how many planted
facts it lists. Until `labels/G/labels.jsonl` is read back from the ticked
sheets, it says so and exits 0.

```bash
python bench/adjudication.py --model anthropic/claude-haiku-4-5-20251001 --out out/adjudication
python bench/adjudication.py --verdicts out/adjudication/G.verdicts.jsonl   # no calls
```

## The conflict order on EnterpriseRAG-Bench

`conflicts.py` runs EnterpriseRAG-Bench's 20 conflicting-information cases
(#155) through openodke and asks whether the corroborator's conflict order puts
the newer document's value first. `fetch` downloads only those questions and
the 39 documents they cite, from the repository at a pinned commit, and stores
them with the benchmark's canary. `run` runs each case as its own small run, one
predicate described by the question, each document with its source type's tier
and its own latest timestamp, under one USD budget across the cases and the
response cache; it writes `results.json`. `table` prints the cases. The results
and the tier mapping are in [docs/benchmarks.md](../docs/benchmarks.md#the-conflict-order-on-enterpriserag-bench).

```bash
python bench/conflicts.py fetch data/erb
python bench/conflicts.py run data/erb --out runs/erb --budget-usd 1.00 \
    --extract-model anthropic/claude-sonnet-5-5 \
    --ground-model anthropic/claude-haiku-4-5-20251001 --cache .odke-cache
python bench/conflicts.py table runs/erb
```
