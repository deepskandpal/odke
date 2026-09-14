# Installation

odke needs Python 3.11, 3.12 or 3.13.

!!! note "Not on PyPI yet"
    `odke` has no release on PyPI yet; v0.1.0 is the first planned one. Until
    then, install from the repository with `git+https://github.com/deepskandpal/odke`.
    These pages describe what is on `main`.

```bash
pip install "odke @ git+https://github.com/deepskandpal/odke"
pip install "odke[neo4j,yaml] @ git+https://github.com/deepskandpal/odke"   # with extras
pip install "odke[all] @ git+https://github.com/deepskandpal/odke"          # every extra
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
  snippets, `freeze` and the `odke ontology` commands;
- ontology inference with the deterministic proposers (`odke ontology infer --no-llm`,
  writing a `.json` draft), and with a model over the standard-library client;
- the loaders for text, Markdown, HTML, directories, JSON, JSONL, CSV and TSV, the
  sentence chunker, and the extractors in `odke.extract`;
- `SpanGrounder`, and `LLMGrounder` over the standard-library OpenAI-compatible
  client;
- the normalise, resolve, corroborate and score stages in `odke.corroborate`;
- `JsonlSink`, `CypherFileSink` and `Neo4jAdminCsvSink`, and
  `import odke.sinks.neo4j` including printing its DDL and write plan (only
  connecting to a server needs the driver);
- `odke run` with a JSON config, and all of `odke.eval` and `odke eval`.

## Extras

These are the extras `pyproject.toml` declares, exactly:

| Extra | Pulls in | What uses it on `main` |
|---|---|---|
| `llm` | `litellm>=1.55,<2` | Model strings the built-in client does not serve (see below) |
| `neo4j` | `neo4j>=5.20,<7` | `Neo4jSink` connecting to a server; `Ontology.from_neo4j` given a URI |
| `rdf` | `rdflib>=7.0,<8` | `RdfSink`; `Ontology.from_owl` |
| `networkx` | `networkx>=3.2,<4` | `NetworkXSink` |
| `docs` | `beautifulsoup4>=4.12,<5`, `pypdf>=5.0,<7`, `lxml>=5.0,<7`, `python-docx>=1.1,<2` | The document readers together, so it covers what `pdf` and `docx` do. Nothing on `main` imports beautifulsoup4 or lxml, because `HtmlLoader` runs on the standard library. This extra is not this site's tooling: that is the `docs` *dependency group* (below). |
| `pdf` | `pypdf>=5.0,<7` | `PdfLoader`, which imports pypdf when it reads a file |
| `docx` | `python-docx>=1.1,<2` | `DocxLoader`, which imports python-docx when it reads a file |
| `yaml` | `pyyaml>=6,<7` | `Ontology.from_yaml`; YAML run configs; the `odke ontology` commands on `.yaml`/`.yml` files, including writing a YAML draft or frozen file |
| `parquet` | `pyarrow>=15` | `ParquetLoader`, which imports pyarrow when it reads a file |
| `all` | `llm`, `neo4j`, `rdf`, `networkx`, `docs`, `pdf`, `docx`, `yaml`, `parquet` | |

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
| a YAML config in `odke run` | `ImportError: reading a YAML config needs PyYAML. Run: pip install "odke[yaml]", or write the same keys as JSON` |
| `Neo4jSink(uri, auth)` | `ImportError: the neo4j driver is not installed; run: pip install 'odke[neo4j]'` |
| `RdfSink(path)` | `ImportError: rdflib is not installed; run: pip install 'odke[rdf]'` |
| `Ontology.from_owl` | `ImportError: rdflib is not installed. Run: pip install "odke[rdf]"` |
| `NetworkXSink()` | `ImportError: networkx is not installed; run: pip install 'odke[networkx]'` |
| `PdfLoader`, reading a file | `MissingExtraError: reading PDF needs pypdf. Run: pip install "odke[pdf]"` |
| `DocxLoader`, reading a file | `MissingExtraError: reading Word documents needs python-docx. Run: pip install "odke[docx]"` |
| `ParquetLoader`, reading a file | `MissingExtraError: reading Parquet needs pyarrow. Run: pip install "odke[parquet]"` |
| `DirectoryLoader`, meeting one of those files | a `MissingExtraWarning` naming the file and the install line; the file is skipped and the walk goes on |
| a model string that needs litellm | `ProviderNotInstalled`, listing `pip install "odke[llm]"`, a `base_url`, or `odke.llm.register` |

`MissingExtraError` is an `ImportError`.

## Working on odke

```bash
git clone https://github.com/deepskandpal/odke && cd odke
./scripts/verify.sh                                  # the whole check: ten steps, same as CI
uv run --group docs mkdocs serve                     # this site, at http://127.0.0.1:8000
uv run --group docs mkdocs build --strict            # what the docs workflow runs
```

The site's tooling (`mkdocs` and `mkdocs-material`) is the `docs` dependency group,
not the `docs` extra: a group never reaches a user's install, and groups and extras
are separate namespaces.

`verify.sh` needs [uv](https://docs.astral.sh/uv/). Its first step refuses to run
if a provider key or `NEO4J_URI` / `NEO4J_PASSWORD` is in the environment, because
the suite must never spend money or write to somebody's real graph. The tests
that do need a real Neo4j run in their own CI job, against a throwaway container.
