# Install

```bash
pip install openodke
pip install "openodke[neo4j,yaml]"   # with extras
pip install "openodke[all]"          # every extra
uv add openodke                      # with uv
```

openodke runs on Python 3.11, 3.12, 3.13 and 3.14. The package is `openodke`;
the command it installs is `odke`.

To install what is on `main` but not yet released:
`pip install "openodke @ git+https://github.com/deepskandpal/odke"`.

## The base install

The base install depends on `pydantic` and `typer` alone, and runs offline: no
model provider, no database driver, no network. Everything on these pages works
on it except what the extras below add.

## Extras

| Extra | Installs | Needed for |
|---|---|---|
| `llm` | `litellm>=1.55,<2` | Model strings the built-in client does not serve |
| `neo4j` | `neo4j>=5.20,<7` | `Neo4jSink` connecting to a server; `Ontology.from_neo4j` given a URI |
| `rdf` | `rdflib>=7.0,<8` | `RdfSink`; `Ontology.from_owl` |
| `networkx` | `networkx>=3.2,<4` | `NetworkXSink` |
| `docs` | `pypdf>=5.0,<7`, `python-docx>=1.1,<2` | `PdfLoader` and `DocxLoader` together |
| `pdf` | `pypdf>=5.0,<7` | `PdfLoader` |
| `docx` | `python-docx>=1.1,<2` | `DocxLoader` |
| `yaml` | `pyyaml>=6,<7` | `Ontology.from_yaml`; YAML run configs |
| `parquet` | `pyarrow>=15` | `ParquetLoader` |
| `bench` | `nltk>=3.8,<4` | `odke bench` on Text2KGBench (its hallucination metrics) |
| `otel` | `opentelemetry-api>=1.24,<2` | [Spans](run.md#logs-and-traces) per job, stage and model call |
| `all` | every extra above | |

Nothing fails at import. A feature used without its extra raises an error that
names the install line; `DirectoryLoader` warns and skips the file instead.

Which model strings need `llm`, and which key each provider reads, is on
[Models and providers](models.md).

To work on openodke itself, see
[CONTRIBUTING](https://github.com/deepskandpal/odke/blob/main/CONTRIBUTING.md).
