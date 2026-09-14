# Installation

odke needs Python 3.11, 3.12 or 3.13.

!!! note "Not on PyPI yet"
    `odke` has no release on PyPI yet; v0.1.0 is the first planned one. Until
    then, install from the repository. These pages describe what is on `main`.

```bash
pip install "odke @ git+https://github.com/deepskandpal/odke"
pip install "odke[neo4j,yaml] @ git+https://github.com/deepskandpal/odke"   # with extras
uv add "odke[neo4j] @ git+https://github.com/deepskandpal/odke"             # or with uv
```

## The base install talks to nothing

The base install pulls in two dependencies, `pydantic` and `typer`: no model
provider, no database driver, no HTTP client. Compiling and inspecting an
ontology is useful offline, and it should not need credentials to exist
([DECISIONS #1](decisions.md)). This is checked rather than promised:
`scripts/verify.sh` step 8 imports the package with the base dependencies alone,
in a throwaway environment, on every push.

On the base install you already have:

- the data model, the thirteen stage Protocols and `Pipeline`;
- ontologies from dicts, JSON and pydantic models, plus `validate`, `diff`,
  snippets and the `odke ontology` commands;
- the loaders in `odke.loaders` (text, Markdown, directories, JSON, JSONL, CSV,
  TSV; Parquet needs an extra) and the extractors in `odke.extract`;
- `SpanGrounder`, and `LLMGrounder` over the standard-library OpenAI-compatible
  client;
- the normalise, resolve, corroborate and score stages in `odke.corroborate`;
- `JsonlSink`, and all of `odke.eval` and `odke eval`;
- `import odke.sinks.neo4j`, including printing its DDL and write plan. Only
  connecting to a server needs the driver.

## Extras

| Extra | Pulls in | What uses it on `main` |
|---|---|---|
| `llm` | `litellm>=1.55,<2` | Model strings the built-in client does not serve (see below) |
| `neo4j` | `neo4j>=5.20,<7` | `Neo4jSink` connecting to a server |
| `yaml` | `pyyaml>=6,<7` | `Ontology.from_yaml`, and the `odke ontology` commands on `.yaml` / `.yml` files |
| `parquet` | `pyarrow>=15` | `ParquetLoader`, which imports pyarrow only when it reads a file |
| `rdf` | `rdflib>=7.0,<8` | Nothing yet: `RdfSink` and `Ontology.from_owl` are planned for v0.2 |
| `networkx` | `networkx>=3.2,<4` | Nothing yet: `NetworkXSink` is planned for v0.2 |
| `docs` | `beautifulsoup4`, `pypdf`, `lxml` | Nothing yet: HTML and PDF readers are planned for v0.2. This extra is not this site's tooling. |
| `all` | every extra above | |

### Which model strings need `llm`

Every model call goes through one `LLMClient` Protocol. A model string such as
`ollama/llama3.1` is served by the first client that matches:

1. an adapter you registered for that provider with `odke.llm.register(provider, factory)`;
2. the standard-library OpenAI-compatible client, when the provider is `ollama`,
   `vllm`, `lmstudio`, `llamacpp`, `openrouter`, `together`, `groq`, `deepseek` or
   `openai`, or when the `ModelSpec` sets a `base_url` (a proxy, a gateway or a
   local server). This needs no extra;
3. litellm, for everything else (Anthropic, Azure, Bedrock, Vertex, Gemini,
   Mistral, Cohere and more). This needs `odke[llm]`.

`ModelRoles.single("ollama/llama3.1")` on the base install, with a local Ollama,
is a complete setup.

## A missing extra says which one

Nothing fails at import. The error comes when the feature is used, and it names
the fix:

| Used without its extra | Raises |
|---|---|
| `Ontology.from_yaml` | `ImportError: PyYAML is not installed. Run: pip install "odke[yaml]"` |
| `Neo4jSink(uri, auth)` | `ImportError: the neo4j driver is not installed; run: pip install 'odke[neo4j]'` |
| `ParquetLoader`, reading a file | `ImportError: reading Parquet needs pyarrow. Run: pip install "odke[parquet]"` |
| a model string that needs litellm | `ProviderNotInstalled`, listing `pip install "odke[llm]"`, a `base_url`, or `odke.llm.register` |

## Working on odke

```bash
git clone https://github.com/deepskandpal/odke && cd odke
./scripts/verify.sh                                  # the whole check: ten steps, same as CI
uv run --group docs mkdocs serve                     # this site, at http://127.0.0.1:8000
uv run --group docs mkdocs build --strict            # what the docs workflow runs
```

`verify.sh` needs [uv](https://docs.astral.sh/uv/). Its first step refuses to run
if a provider key or `NEO4J_URI` / `NEO4J_PASSWORD` is in the environment, because
the suite must never spend money or write to somebody's real graph. The tests
that do need a real Neo4j run in their own CI job, against a throwaway container.
