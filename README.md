# odke

**Text in, a grounded knowledge graph out.** Point it at documents — prose, HTML,
PDFs, CSVs, JSON, whatever you have — give it an ontology or let it infer one, and
get back entities and relations that load straight into Neo4j, RDF, NetworkX, or
your own store.

```python
from odke import Document, Ontology, Pipeline
from odke.extract import HybridExtractor  # M2
from odke.ground import LLMGrounder  # M3
from odke.corroborate import Corroborator  # M3
from odke.sinks.neo4j import Neo4jSink  # M4

ontology = Ontology.from_json("schema.json")  # or Ontology.infer(docs)  — M5

kg = Pipeline(
    ontology,
    extractor=HybridExtractor(model="claude-sonnet-5"),
    grounder=LLMGrounder(model="claude-haiku-4-5-20251001"),
    corroborator=Corroborator(),
    sinks=[Neo4jSink(uri="bolt://localhost:7687", auth=("neo4j", "…"))],
).run(docs)

print(len(kg.edges), "relations,", len(kg.properties), "attributes")
```

> **Status: pre-alpha.** The data model, the ontology compiler and the snippet
> generator are implemented and tested. The extractor, grounder, corroborator and
> database sinks are the next four milestones — see [ROADMAP.md](ROADMAP.md). The
> API above is the committed shape, not a description of what runs today.

## Why this exists

Apple published [ODKE+ (arXiv:2509.04696)](https://arxiv.org/abs/2509.04696), a
production system that extracts open-domain facts at scale and ingests them into a
knowledge graph at 98.8% precision. It is a genuinely good architecture and there
is no implementation of it — Apple released none, and nothing on PyPI, conda,
GitLab or Hugging Face implements it. This is an independent implementation of
that architecture, generalised from one company's internal graph into an SDK
anyone can install.

The three ideas worth stealing from that paper, and what they buy you:

| Idea | What it does | Why the alternatives lose |
|---|---|---|
| **Ontology snippets** | Prompts the model with a small, ranked, per-type schema fragment rather than the whole ontology | A 200-predicate schema does not fit in a useful prompt, and stuffing it in degrades every extraction. Snippets keep prompt size flat as the schema grows |
| **Grounding verification** | A second, cheap model checks each candidate fact against its own evidence span, and drops what it cannot find | This is where the precision comes from. Extraction alone hallucinates; a yes/no check against a quoted span is cheap and catches it |
| **Corroboration** | Merges the same claim across sources, resolves conflicts on freshness × trust × agreement | Real corpora disagree with themselves. Without this you write both answers into the graph and the graph stops being trustworthy |

## How it differs from what already exists

There are good libraries that turn text into a graph — [iText2KG](https://github.com/AuvaLab/itext2kg),
[neo4j-graphrag](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)'s
`SimpleKGPipeline`, LangChain's `LLMGraphTransformer`, `knowledge-graph-maker`.
They extract, and most can be handed a schema. What none of them do is the part
after extraction: **no per-type snippet ranking, no separate grounding pass, no
cross-source corroboration, no provenance carried to the sink.** Those are
precisely the three stages that take a demo to 98.8% precision, and they are what
this library is for.

Concretely, every fact `odke` emits carries the document, the character span, the
source tier and the grounding verdict that produced it. You can always answer
"why is this edge in my graph?" — and delete every edge that came from a source
you no longer trust.

## Install

```bash
pip install odke                 # data model, ontology compiler, pattern extractor
pip install "odke[llm]"          # + model-backed extraction and grounding
pip install "odke[neo4j]"        # + the Neo4j sink
pip install "odke[all]"          # everything
```

The base install pulls two dependencies and talks to nothing. Model providers and
database drivers are extras on purpose: nobody should have to install a Neo4j
driver to compile an ontology.

Python 3.11–3.13.

## Inputs

Anything you can get into a `Document`. Loaders (M2) handle plain text, Markdown,
HTML, PDF, DOCX, JSON, JSONL, CSV/TSV and Parquet, and the extractor routes each
by modality: structured and semi-structured inputs go through the **pattern
extractor**, which is exact, free and deterministic, and only genuine prose costs
a model call.

## Ontologies: bring one, or infer one

```python
Ontology.from_json("schema.json")  # or .from_yaml
Ontology.from_owl("schema.ttl")  # OWL / RDFS / SKOS, via rdflib      — M4
Ontology.from_pydantic(Person, Company)  # — M2
Ontology.from_neo4j(driver)  # reflect a live graph's schema      — M4
Ontology.infer(docs, sample=200)  # propose one from the corpus        — M5
```

An inferred ontology is marked `inferred=True` and carries corpus support counts
on every predicate, so you can review it, edit it, and freeze it as the schema
for subsequent runs. Inference is a bootstrap, not a permanent mode.

You can inspect exactly what the model will be shown before spending a token:

```bash
odke ontology snippet schema.json Person
odke ontology snippet schema.json Person --json-schema
```

A bad extraction is usually a bad snippet, and this is how you see it.

## Models and providers

Nothing in this library imports a provider SDK or names a vendor. Every model
call goes through one `LLMClient` protocol, and which client serves a model
string is decided by a small, inspectable routing rule you can override.

```python
from odke.llm import ModelRoles, ModelSpec

ModelRoles()  # Claude by default: Sonnet extracts, Haiku grounds
ModelRoles.single("ollama/llama3.1")  # entirely local — no extras, no key, no network
ModelRoles.single("openai/gpt-4.1")
ModelRoles.single("azure/my-deployment", extra={"api_version": "2024-10-21"})
ModelRoles.single("bedrock/anthropic.claude-sonnet-4-20250514-v1:0")

# Mix freely. Frontier extraction, local grounding — the cheap combination
# that makes the paper's precision affordable at volume.
ModelRoles(
    extract=ModelSpec(model="anthropic/claude-sonnet-5"),
    ground=ModelSpec(model="ollama/qwen2.5:3b"),
)
```

**Two models, not one, by default.** Extraction wants a capable model;
grounding asks ten thousand yes/no questions and wants a small one. Making that
asymmetry the default is most of what keeps the precision stage affordable.

**Zero-dependency path.** Anything speaking the OpenAI chat shape — Ollama,
vLLM, LM Studio, llama.cpp, OpenRouter, Groq, Together, DeepSeek, a LiteLLM
proxy, a corporate gateway — is served by a client written against the standard
library. `pip install odke` and a local Ollama is a complete, working setup.

**Everything else via litellm.** Anthropic, OpenAI, Azure, Bedrock, Vertex,
Gemini, Mistral, Cohere and ~100 more, behind `pip install "odke[llm]"`.

**Your own gateway.** One call, and every matching model string routes through
you — no fork, no subclass:

```python
from odke.llm import register

register("mycorp", lambda spec: MyAuditedClient(spec))
```

**Testing without a key.** `ScriptedClient` ships in the package, not just the
test suite, so your extractors are testable offline too:

```python
from odke.llm import ScriptedClient

client = ScriptedClient(['{"employer": "Analytical Engine Co"}'])
```

## Outputs

`KnowledgeGraph` is storage-neutral. Sinks translate:

| Sink | Extra | Status |
|---|---|---|
| `JsonlSink` | — | shipped |
| `Neo4jSink` — batched `MERGE`, provenance on every edge | `[neo4j]` | M4 |
| `CypherFileSink` / `Neo4jAdminCsvSink` — bulk load | — | M4 |
| `RdfSink` — Turtle / N-Triples / JSON-LD | `[rdf]` | M4 |
| `NetworkXSink` | `[networkx]` | M4 |
| Memgraph, Kùzu, ArangoDB, TigerGraph | — | community |

Writing your own is one method. `Sink` is a `Protocol` — no base class, no
registration:

```python
class MySink:
    def write(self, kg: KnowledgeGraph) -> None: ...
```

The same is true of every stage. Swap the grounder, keep the rest.

## Documentation

- [ROADMAP.md](ROADMAP.md) — milestones, what each one delivers, and the estimate
- [DECISIONS.md](DECISIONS.md) — the design calls and why they went that way
- [CONTRIBUTING.md](CONTRIBUTING.md) — `./scripts/verify.sh` is the whole check
- [NOTICE](NOTICE) — the relationship to the ODKE+ paper

## Relationship to the paper

This is an independent implementation of a published architecture. It is not
affiliated with or endorsed by Apple Inc., uses no Apple code, data or models,
and was written from the paper alone. All credit for the architecture belongs to
Khorshidi et al. See [NOTICE](NOTICE).

It also deliberately departs from the paper where the paper is specific to its
deployment: ODKE+ decides *what* to refresh by watching Wikipedia edits and
retrieves its own evidence. Here those two stages are optional protocols —
`Initiator` and `Retriever` — because an SDK is usually handed its documents. The
three stages that carry the ideas are the three this library implements.

## License

Apache-2.0.
