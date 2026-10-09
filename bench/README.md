# bench — openodke against its competitors

Not part of the package: this folder holds what `pip install openodke` must
never pull in — competitor libraries — and the scripts that compare them with
openodke on the public datasets `odke bench` prepares.

## The comparison

Three extractors on the same prepared sets — the same documents, whole; the same
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

Both competitors are held to the schema the way LLMGraphTransformer's strict mode
does it. Their triples reach openodke in its [triples format](../docs/triples.md),
through the `triples` extract stage. They quote nothing, so each is grounded
against its whole document, which is the paper's own mode: the whole context,
True or False, affirmed facts kept (`odke bench prepare --paper`; the set's
ground model grounds).

Microsoft GraphRAG is not here: its extraction writes free-text entity and
relationship descriptions for community summaries, with no ontology, so it cannot
be scored against a dataset's relations without inventing a mapping.

### One model string for all three

openodke calls its model through LiteLLM, and so do the competitors here:
LLMGraphTransformer gets a runnable that calls LiteLLM, neo4j-graphrag an
`LLMInterface` that does. So any provider LiteLLM supports runs the whole
comparison — `openai/…`, `anthropic/…`, `gemini/…`, `ollama/…` — and a reasoning
model's thinking never reaches either library's parser: LiteLLM returns the
reply's text. Nothing about what the model is asked changes. Both libraries run
their prompt-and-parse paths, not their structured ones, which force a tool choice
some reasoning models refuse.

## Running it

```bash
uv venv --python 3.12 bench/.venv
uv pip install --python bench/.venv/bin/python -e ".[llm,bench]" \
    langchain-experimental json-repair neo4j-graphrag

# prepare the sets under $CMP/t2k/ont_* and $CMP/redocred, naming the models once:
odke bench prepare text2kgbench data/t2k --ontology ont_1_movie --out "$CMP/t2k/ont_1_movie" \
    --extract-model openai/gpt-5 --ground-model openai/gpt-5-mini --paper
# (a model with a smaller output cap: add --max-tokens 4000)

CMP=runs/cmp bench/run_all.sh          # openodke + both competitors, then verification
python bench/tables.py runs/cmp        # the tables, priced from LiteLLM's table

# a smoke test of one competitor on three documents:
bench/.venv/bin/python bench/competitors.py extract lgt "$CMP/t2k/ont_1_movie" --limit 3
```

Keys come from the environment, under the names LiteLLM reads
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …); `ODKE_ENV_FILE`, or `--env-file` on
`competitors.py`, names a file of `KEY=value` lines to load first. `--model` on
`competitors.py` overrides the set's extraction model, for a deliberate
cross-model comparison.

Each competitor directory keeps its triples (`facts.jsonl`), its model and token
usage (`usage.json`) and its report; every run keeps its predictions per
configuration (`predictions/`), which is what an audit reads.

The published comparison (PR #105) ran on `anthropic/claude-sonnet-5-5`
extracting and `anthropic/claude-haiku-4-5` grounding. No other provider has run
it end to end yet.
