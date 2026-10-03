# bench — openodke against its competitors

Not part of the package: this folder holds what `pip install openodke` must
never pull in — competitor libraries — and the scripts that compare them with
openodke on the public datasets `odke bench` prepares.

## The comparison

Three extractors on the same prepared sets — the same documents, whole; the same
schema; the same model (`claude-sonnet-5-5`, 16,000 tokens of room); and each one
run three ways:

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
does it. Their triples reach openodke through `replay:Replayed`, an extractor stage
that cites the whole document — they quote nothing — so grounding runs in the
paper's own mode: the whole context, True or False, affirmed facts kept
(`odke bench prepare --paper`; Haiku 4.5 grounds).

Microsoft GraphRAG is not here: its extraction writes free-text entity and
relationship descriptions for community summaries, with no ontology, so it cannot
be scored against a dataset's relations without inventing a mapping.

### Two shims

Claude Sonnet 5.5 answers with a thinking block before its text, and neither
competitor reads that as shipped: neo4j-graphrag's `AnthropicLLM` reads only the
first block and raises; LLMGraphTransformer without tool calling hands the block
list to its JSON parser. `competitors.py` gives each the reply's text — nothing
about what the model is asked changes. Tool calling is not an option on this
model (forced `tool_choice` is refused), and both libraries' structured paths
force it.

## Running it

```bash
uv venv --python 3.12 bench/.venv
uv pip install --python bench/.venv/bin/python -e ".[llm,bench]" \
    langchain-experimental langchain-anthropic "neo4j-graphrag[anthropic]"

# prepare sets with `odke bench prepare ... --paper` under $CMP/t2k/ont_* and $CMP/redocred
CMP=runs/cmp bench/run_all.sh          # openodke + both competitors, then verification
python bench/tables.py runs/cmp        # the tables
```

`ANTHROPIC_API_KEY` must be set. Each competitor directory keeps its triples
(`facts.jsonl`), its token usage (`usage.json`) and its report; every run keeps
its predictions per configuration (`predictions/`), which is what an audit reads.
