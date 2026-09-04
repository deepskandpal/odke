# Decisions

The calls that shaped the design, and what each one cost. Written down because
the reasoning is the part that gets lost, and a decision without its reason gets
reversed by the next person who finds it inconvenient.

### 1. The base install talks to nothing

`pip install odke` pulls pydantic and typer. No model provider, no database
driver, no HTTP client. Compiling and inspecting an ontology is a genuinely
useful thing to do offline, and it should not require credentials to exist.

*Cost:* the README has to explain extras, and `verify.sh` step 8 exists solely to
keep this honest.

### 2. One `Fact` class for edges and properties

`object_entity` and `object_value` are mutually exclusive fields on one class
rather than two classes. Both kinds need identical provenance, grounding and
corroboration; when they were split, every stage had two nearly identical code
paths and they drifted.

*Cost:* an invalid state is representable — both fields set. Validation catches
it; the alternative was worse.

### 3. Spans are character offsets, not quoted strings

A model can produce a quote that reads perfectly and appears nowhere in the
source. An offset either resolves to the claimed text or it does not, and
checking costs nothing. `Span.is_faithful()` runs before the grounder is asked,
so the cheapest rejection happens first.

*Cost:* chunking has to preserve offsets back to the original document, which is
the hardest part of M2. It is worth it — without this, provenance is decorative.

### 4. Everything is frozen

Facts pass through four stages and get merged across sources. A stage that edited
one in place would make its own provenance wrong. Stages return new objects.

*Cost:* more allocation. Irrelevant next to a model call.

### 5. Stages are Protocols, not base classes

`Extractor`, `Grounder`, `Corroborator`, `Sink` are `typing.Protocol`. Anyone can
supply their own by writing one method — no import of ours, no registration, no
inheritance. This is what makes "or any graph DB" true rather than aspirational.

*Cost:* no shared implementation to inherit. There was not much to share.

### 6. Snippets are data, rendered two ways

`OntologySnippet` holds structure; `.render()` produces prose and
`.json_schema()` produces a schema. One object, so the prompt and the structured
-output contract cannot describe different things — which they will, eventually,
if they are built separately.

### 7. Two model adapters, not one, and not a hundred

litellm covers ~100 providers and is the obvious single answer — but making it a
hard dependency means `pip install odke` drags in a large package before the user
has decided to call a model at all, and it makes the most common serious setup
(a local Ollama) require it too.

So: an OpenAI-compatible client written against the standard library handles
everything that speaks that shape — Ollama, vLLM, LM Studio, llama.cpp,
OpenRouter, Groq, Together, DeepSeek, proxies, gateways — and litellm, behind the
`[llm]` extra, handles the rest. `resolve()` picks between them by a rule short
enough to read.

*Cost:* two adapters to keep behaviourally identical. `test_llm.py` asserts they
normalise to the same `Completion`, including that unknown cost is `None` rather
than `0.0` in both.

### 7a. Model roles are part of the configuration

`ModelRoles` names three jobs — extract, ground, infer — and defaults grounding
to a smaller model than extraction. The paper's precision comes from a second
verification pass; if that pass costs the same per call as extraction, people
turn it off, and then the architecture does not work.

Defaults name Claude models because something must be the default. Nothing in the
library depends on them, and `ModelRoles.single("ollama/…")` is a first-class
configuration rather than a degraded one.

### 7b. A registry, so an internal gateway is not a fork

Many teams can only call models through their own audited proxy.
`register("provider", factory)` routes every matching model string through a
caller-supplied client. Without it, those teams would vendor the library.

### 8. Ontology inference is a bootstrap, not a mode

`Ontology.infer()` returns an ontology marked `inferred=True` for the caller to
review, edit and freeze. It is not wired to run implicitly on every call.

A schema that silently re-infers between runs produces a graph whose edge labels
change underneath existing queries. The one-time cost of review buys a graph that
stays queryable.

### 9. The Initiator and Retriever are optional

The paper's first two stages watch Wikipedia for edits and fetch evidence. An SDK
is normally handed its documents. Both are declared as protocols so the refresh
loop can be built without forking, and neither is required to run a pipeline.

### 10. Trust tiers are an enum with weights, not a free float

Callers reason about "curated vs. scraped", not about 0.8. Four named tiers with
fixed weights make conflict resolution explainable, which matters the first time
someone asks why the graph picked one of two contradictory answers.

### 11. `Fact.signature` excludes qualifiers

"CEO since 2019" and "CEO 2019–2024" are one claim told two ways. If qualifiers
were part of identity, both would land in the graph as separate edges, which is
exactly the failure corroboration exists to prevent.

### 12. Verification is one script

`scripts/verify.sh` is what CI runs and what a contributor runs. There is no
second list of steps to drift out of sync with the first.

### 13. The board is linked to the repo, not auto-populated

Projects v2 boards are owned by a user or an org — `createProjectV2` takes an
`ownerId`, and Repository is not a valid owner. Repo-owned boards were Projects
(classic), retired in 2024. So the board is account-owned and *linked* to the
repo, which is why it appears under the Projects tab.

The consequence is that `GITHUB_TOKEN` cannot write to it, and auto-adding new
issues would need a PAT stored as a repo secret. For a project this size that
trades a rotating credential for a keystroke, so there is no add-to-project
workflow. New issues go on the board with:

    gh project item-add 6 --owner deepskandpal --url <issue-url>
