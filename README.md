# openodke

An open implementation of ODKE+ (Apple, [arXiv:2509.04696](https://arxiv.org/abs/2509.04696)) — independent, and not affiliated with Apple.

[PyPI](https://pypi.org/project/openodke/) · [Documentation](https://deepskandpal.github.io/odke/) · [Changelog](CHANGELOG.md) — `pip install openodke`

**The seam between text and any graph store.** openodke turns documents — prose,
tables, records — into a knowledge graph held to your ontology, and writes it to
Neo4j, to JSON Lines, or to a store of your own. It is the only pipeline that
asks a second model whether the cited span supports the claim, and records the
verdict on every fact.

Every fact carries the document, the character span, the source tier and the
grounding verdict that produced it, so *why is this edge in my graph?* is a query,
not an investigation.

> **Status: 0.1.0, released on [PyPI](https://pypi.org/project/openodke/) on
> 14 September 2026.** Every milestone in [ROADMAP.md](ROADMAP.md) has shipped:
> the data model, ontology I/O, import and inference, loaders (text, Markdown,
> records, HTML, PDF, DOCX) and extraction, grounding and corroboration, the
> Neo4j, RDF, NetworkX and bulk sinks, `odke run`, and the evaluation harness.

## Quickstart

From a clone, with no key, no network and no database:

```bash
git clone https://github.com/deepskandpal/odke && cd odke
pip install -e ".[yaml]"
odke run examples/e2e/odke.yaml
```

That runs a small invented corpus through all thirteen stages on recorded model
responses and writes the graph to `examples/e2e/out/`. The same run from Python:

```python
from openodke import HybridExtractor, LLMExtractor, Ontology, Pipeline, VerdictValidator
from openodke.ground import LLMGrounder
from openodke.llm import RecordedClient, ReplayClient
from openodke.loaders import DirectoryLoader
from openodke.sinks import JsonlSink

ontology = Ontology.from_json("examples/e2e/ontology.json")
docs = list(DirectoryLoader().load("examples/e2e/corpus"))
# Recorded responses, so this runs with no key. Drop `client=` to call the models.
extract = ReplayClient("examples/e2e/recorded/extract.json")
ground = RecordedClient.from_fixture("examples/e2e/recorded/ground.json")
kg = Pipeline(
    ontology,
    HybridExtractor(LLMExtractor(client=extract), documents=docs),
    grounder=LLMGrounder(client=ground),  # a second model checks every cited span
    validator=VerdictValidator(),  # refuses a fact its own span contradicts
    sinks=[JsonlSink("out")],
).run(docs)
print(len(kg.facts), "facts,", kg.stats["refused"], "refused")
```

[`examples/e2e/`](examples/e2e/README.md) goes on into Neo4j and ends with the
Cypher that shows provenance, a `DIFFERENT` link between two companies with one
name, and a cardinality check.

## Does it work?

On the example, `odke eval ablation` runs one config three ways against 36
labelled facts:

| Configuration | Precision | Recall | Model calls |
|---|---|---|---|
| extraction alone | 0.889 | 0.889 | 3 |
| + grounding | 0.914 | 0.889 | 39 |
| + corroboration | 0.971 | 0.944 | 39 |

**This is a demonstration on hand-authored recorded responses, not a benchmark.**
The responses and the labels were written for the example, so the table shows
what the harness reports, not how well any model does. On that fixture,
grounding caught one of four wrong candidates, and normalisation did more for
precision than grounding. No number from real model runs exists yet. The paper's
98.8% is neither reproduced nor claimed. [What the example shows, and does not](examples/e2e/README.md#5-the-ablation-on-this-example).

The harness is the point: run the same command on a slice of your own corpus
that you have labelled, and it tells you whether grounding earns its calls there.

## Why this exists

Apple published [ODKE+ (arXiv:2509.04696)](https://arxiv.org/abs/2509.04696), a
production system that extracts open-domain facts at scale and ingests them into a
knowledge graph at 98.8% precision. Apple released no implementation, and nothing
on PyPI, conda, GitLab or Hugging Face implements it. This is an independent
implementation of that architecture, generalised from one company's internal
graph into an SDK anyone can install.

The three ideas worth taking from that paper:

| Idea | What it does | Why the alternatives lose |
|---|---|---|
| **Ontology snippets** | Prompts the model with a small, ranked, per-type schema fragment rather than the whole ontology | A 200-predicate schema does not fit in a useful prompt. Snippets keep prompt size flat as the schema grows |
| **Grounding verification** | A second, cheap model checks each candidate fact against its own evidence span and records `supported`, `contradicted` or `not_found` on the fact; a validator decides what is written | Extraction alone hallucinates. A yes/no check against a quoted span is cheap, and a verdict kept on the fact can be measured and re-gated later |
| **Corroboration** | Merges the same claim across sources, and resolves conflicts on freshness × trust × agreement | Real corpora disagree with themselves. Without it you write both answers and cannot say which to believe |

Other libraries turn text into a graph — [iText2KG](https://github.com/AuvaLab/itext2kg),
[neo4j-graphrag](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)'s
`SimpleKGPipeline`, LangChain's `LLMGraphTransformer`. They extract, and most can
be handed a schema. None of them has a separate grounding pass, cross-source
corroboration, or character-span provenance carried through to the store.

## Thirteen stages, each one a Protocol

The pipeline is a superset: thirteen stages, each a `typing.Protocol` in
`openodke.stages` with a pass-through default (DECISIONS #20). Take the subset your
corpus needs, supply your own for any stage by writing one method, and leave the
rest out. Nothing domain-specific is in the code; it arrives as an ontology, a
config, or a stage of yours.

| Stage | Built-in | Name in `odke run` | Left out |
|---|---|---|---|
| load | `DirectoryLoader`, `TextLoader`, `MarkdownLoader`, `HtmlLoader`, `PdfLoader`, `DocxLoader`, `CsvLoader`, `TsvLoader`, `JsonLoader`, `JsonlLoader`, `ParquetLoader` | `directory` (every suffix), `text`, `markdown`, `html`, `pdf`, `docx`, `csv`, … | `directory` |
| chunk | `SentenceChunker` — whole sentences, offsets intact | `sentence` | one chunk per document |
| route | yours: skip marketing, policy, narrative | — | extract everything |
| extract | `PatternExtractor`, `LLMExtractor`, `HybridExtractor` | `pattern`, `llm`, `hybrid` | required |
| ground | `SpanGrounder` (free), `LLMGrounder` | `span`, `llm` | verdict `unchecked` |
| normalise | `ValueNormalizer` — dates, numbers, units, name keys | `value` | unchanged |
| resolve | `NativeResolver` — blocking, identifiers, links | `native` | keys as given |
| corroborate | `SignatureCorroborator` | `signature` | every fact its own claim |
| score | `EvidenceScorer` | `evidence` | confidence unchanged |
| validate | `VerdictValidator` — refuses `contradicted` | `verdict` | accept everything |
| sink | `JsonlSink`, `Neo4jSink`, `CypherFileSink`, `Neo4jAdminCsvSink`, `RdfSink`, `NetworkXSink` | `jsonl`, `neo4j`, `cypher_file`, `neo4j_admin_csv`, `rdf`, `networkx` | nothing written |
| constrain | `Neo4jConstrainer` — the ontology as DDL | `neo4j` | no constraints |
| infer | `OntologyInferrer`; `odke ontology infer`, then review and `freeze` | — | never run: inference is a bootstrap (DECISIONS #8) |

## Install

From [PyPI](https://pypi.org/project/openodke/):

```bash
pip install openodke                   # the base install
pip install "openodke[yaml]"           # + YAML configs
pip install "openodke[neo4j,yaml]"     # + the Neo4j sink
pip install "openodke[all]"            # everything
```

For unreleased changes on `main`: `pip install "openodke @ git+https://github.com/deepskandpal/odke"`.

The base install is pydantic and typer, and talks to nothing: the data model,
the ontology compiler, the text, Markdown, HTML and record loaders, the pattern
extractor, the OpenAI-compatible model client, corroboration, evaluation, and
the JSONL and Cypher-file sinks. Each extra is imported on first use, and a
missing one is named in the error.

| Extra | Adds | For |
|---|---|---|
| `llm` | litellm | Model providers that do not speak the OpenAI shape |
| `neo4j` | the Neo4j driver | `Neo4jSink`, `Ontology.from_neo4j` |
| `yaml` | PyYAML | YAML ontologies and run configs |
| `parquet` | pyarrow | `ParquetLoader` |
| `pdf`, `docx` | pypdf; python-docx | `PdfLoader`, `DocxLoader` |
| `rdf` | rdflib | `RdfSink`, `Ontology.from_owl` |
| `networkx` | networkx | `NetworkXSink` |
| `docs` | pypdf, python-docx | The document readers together |
| `all` | all of the above | |

Python 3.11–3.13.

## `odke run`

One file names the inputs, the ontology, the models, which implementation fills
each stage, and the sink:

```yaml
ontology: ontology.json
inputs:
  - path: corpus/register.csv
    loader: {use: csv, tier: curated}
  - corpus/notes
models:
  extract: anthropic/claude-sonnet-5
  ground: anthropic/claude-haiku-4-5-20251001
stages:
  chunker: {use: sentence, max_words: 120}
  extractor: hybrid
  grounder: llm
  normalizer: value
  corroborator: signature
  validator: verdict
  sink: {use: neo4j, uri_env: NEO4J_URI, password_env: NEO4J_PASSWORD}
  constrainer: neo4j
bootstrap: true
```

```bash
odke run config.yaml --dry-run     # load, extract and ground; print what would be written
odke run config.yaml
```

A stage of your own is `package.module:Name` with its options alongside. A dry
run opens no sink, so it needs no database. Every stage's own counts — verdicts,
rejections, conflicts, refusals, cost — go into the graph's `stats`, and a
document's id is its path, so a label or a query can name it.
[`examples/run.yaml`](examples/run.yaml) comments every key.

## Evaluate against your own labels

openodke ships the formats and the arithmetic for scoring every stage, and never a
corpus: a number computed against the pipeline's own output measures nothing.
Bring your own labelled dataset.

```bash
odke eval extract --describe                           # what to label
odke eval extract --labels gold.jsonl --predictions out/facts.jsonl
odke eval ablation --config config.yaml --labels gold.jsonl
```

There is an evaluator for routing, extraction, grounding, resolution, scoring
(Brier, reliability, ECE) and validation, and `openodke.eval.CostMeter` measures
tokens and USD per stage without touching a stage. The fixtures in the test
suite exercise the arithmetic and are not a benchmark.

## When the platform does a stage itself

Some stores already resolve, prune or constrain after the write. A sink declares
that with a `PlatformProfile`; a stage configured on both sides raises one
`DoubleStageWarning` and still runs, because the two passes are not the same
pass (DECISIONS #21). To hand a stage to the platform, pass
`Delegated(to="neo4j-graphrag:FuzzyMatchResolver")` — or `{use: delegated, to: ...}`
in a config — and openodke stamps who did it wherever the data model has room, so
the platform's work can be read back and scored with the same evaluator.

## Models and providers

Nothing in this library imports a provider SDK or names a vendor. Every model
call goes through one `LLMClient` protocol, and which client serves a model
string is decided by a small, inspectable routing rule you can override.

```python
from openodke.llm import ModelRoles, ModelSpec

ModelRoles()  # Claude by default: Sonnet extracts, Haiku grounds
ModelRoles.single("ollama/llama3.1")  # entirely local — no extras, no key, no network
ModelRoles(
    extract=ModelSpec(model="anthropic/claude-sonnet-5"),
    ground=ModelSpec(model="ollama/qwen2.5:3b"),
)
```

**Two models, not one, by default.** Extraction wants a capable model; grounding
asks thousands of yes/no questions and wants a small one. Anything speaking the
OpenAI chat shape — Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, Groq, a
gateway — works on the base install; everything else goes through litellm with
`[llm]`; `openodke.llm.register("mycorp", factory)` routes a provider through your
own client. `ReplayClient`, `RecordedClient` and `ScriptedClient` ship in the
package, so your own stages are testable offline too.

## Ontologies

```python
from openodke import Ontology

ontology = Ontology.from_json("examples/e2e/ontology.json")  # or from_owl, from_neo4j
print(ontology.validate())  # diagnostics with dotted paths, never an exception
print(ontology.snippet("Company").render())  # exactly what the model is prompted with
```

```bash
odke ontology validate schema.json
odke ontology diff old.json new.json --fail-on-breaking
odke ontology snippet schema.json Company
```

A bad extraction is usually a bad snippet, and `snippet` shows it before a token
is spent. The ontology also marks which qualifiers bear identity (`percentile`)
and which reconcile (`start_time`), and that decides what counts as one claim.

## Honest limits

- **The fixtures and the example are not benchmarks.** Their model responses are
  hand-authored. There are no real-model numbers yet.
- **Live Neo4j needs 5.7 or later.** The constraint bootstrap uses relationship
  uniqueness constraints; Community edition is enough.
- **Structured facts are grounded against their own cell.** The cell `Leeds`
  cannot say whose head office it is, so a careful grounder answers `not_found`
  and the call is spent for little. The default gate keeps `not_found` for this
  reason.
- **The grounder sees the cited span, never the document.** That is what keeps
  it cheap, and a span that leaves out the subject cannot support the claim.
- **Resolution never merges on names.** Only a shared identifier re-keys an
  entity; a name match is a `SIMILAR` link for someone to act on.
- **Inference drafts; it never decides.** `odke ontology infer` proposes a small
  schema for a person to review and freeze, and nothing infers implicitly
  (DECISIONS #8).

## Documentation

- [deepskandpal.github.io/odke](https://deepskandpal.github.io/odke/) — the documentation site: concepts, ontology and inference, loaders and extraction, grounding, resolution and corroboration, sinks, `odke run`, evaluation
- [examples/](examples/README.md) — the end-to-end example and the commented run config
- [ROADMAP.md](ROADMAP.md) — milestones, what each one delivers, and the estimate
- [CHANGELOG.md](CHANGELOG.md) — what landed, milestone by milestone
- [DECISIONS.md](DECISIONS.md) — the design calls and why they went that way
- [CONTRIBUTING.md](CONTRIBUTING.md) — `./scripts/verify.sh` is the whole check
- [NOTICE](NOTICE) — the relationship to the ODKE+ paper

## Relationship to the paper

This is an independent implementation of a published architecture. It is not
affiliated with or endorsed by Apple Inc., uses no Apple code, data or models,
and was written from the paper alone. All credit for the architecture belongs to
Khorshidi et al. See [NOTICE](NOTICE).

It departs from the paper where the paper is specific to its deployment. ODKE+
decides what to refresh by watching Wikipedia edits and retrieves its own
evidence; here those two stages are optional protocols, `Initiator` and
`Retriever`, because an SDK is usually handed its documents.

## License

Apache-2.0.
