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

### 11. `Fact.signature` excludes reconcilable qualifiers

"CEO since 2019" and "CEO 2019–2024" are one claim told two ways. If qualifiers
were part of identity, both would land in the graph as separate edges, which is
exactly the failure corroboration exists to prevent.

*Amended in M0 (#48):* this holds for **reconcilable** qualifiers, which is what
every qualifier is unless the ontology says otherwise. A key the ontology
declares `identity: true` is a different matter — see #15.

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

### 14. Polarity is a field, and it is part of the signature

`Fact.polarity` is `asserted` / `denied` / `partial`. Before it existed the only
place a denial could go was `qualifiers`, which #11 keeps out of `signature` —
so "X sells customer data" and "X does not sell customer data" shared a
signature, the corroborator merged them, and each raised the other's `support`.
A denial and its own contradiction strengthened each other.

It joins `signature` because a denial is not the same claim as an assertion; it
is the opposite one. Hand annotation of a real corpus found 6% of facts negative
or partial, which is too many to lose and too many to merge wrongly.

*Cost:* one more field on a frozen model, and a re-partition of any graph built
before it — which is why it lands in M0, before anything is serialised.

### 15. The ontology says which qualifiers bear identity; the fact carries the answer

#11 is right about `start_time` and wrong about `percentile`. "Uptime 99.9% at
p50" and "uptime 99.9% at p95" are two measurements; with qualifiers excluded
from `signature` they merged into one claim with `support = 2`, and where the
values differed a single-valued predicate saw a contradiction that was not one.
Two in five facts in a real corpus carried a qualifier of this kind.

So `Predicate.qualifiers` maps each key to a `Qualifier(identity=...)`,
defaulting to `False` — #11 stays the default and nothing in an existing
ontology changes meaning. The extractor stamps `Ontology.identity_keys(pred)`
onto `Fact.identity_keys`, and `signature` includes those keys, sorted.

The fact carries the key names rather than a reference to the ontology because
`signature` has to stay a pure property of the fact: a serialised fact must mean
the same thing after the schema it came from has been edited, and the
corroborator must never need the schema in hand to merge two facts.

*Cost:* an extractor that forgets to stamp `identity_keys` silently gets #11's
old behaviour. That is the safe direction to fail in.

### 16. Resolution proposes links; it never replaces nodes

`EntityLink(source_key, target_key, kind, score, evidence, reason)` with `kind`
one of `SAME_AS`, `SIMILAR`, `DIFFERENT`. neo4j-graphrag's three resolvers
*replace* the nodes they match, and once merged you cannot lower the threshold
and re-run to see what moved, because there is nothing left to move. A link is
reversible; a merge is not.

`DIFFERENT` is the kind nobody else records. It is where the disagreement rule
lives — a strong identifier that disagrees kills a match however similar the
names — and `reason` names the identifier that disagreed, so the rejection is a
query rather than a mystery. A string, not a second structure.

Links ride on the output object as `KnowledgeGraph.links`, default empty. One
output object is what "any sink" means; an empty tuple costs nothing, and a
sink with no use for links ignores them.

*Cost:* the store holds duplicates until something acts on the links. That is
the point — acting on them is a choice a caller can revisit.

### 17. Two clocks: valid time on the fact, transaction time on the evidence

`Fact.valid_from` / `valid_to` say when the claim was true in the world.
`Evidence.retrieved_at` and `Document.retrieved_at` say when we came to believe
it. They were one clock before, and one clock cannot tell a CEO who changed
from two sources that disagree — one is a fact that expired correctly, the
other is a conflict to resolve, and the corroborator has to do opposite things
with them.

Neither clock is in `signature`. "CEO 2019–2024" and "CEO since 2019" are one
claim with two views of its interval, which is the reconciliation #11 exists
for. Graphiti's bi-temporal model invalidates edges rather than deleting them;
that is the same instinct, and the shape here is chosen so it can be borrowed
when the staleness queue is built.

*Cost:* two nullable fields most extractors will leave empty. A migration if
added later, which is why they are here now.

### 18. An entity records how its key was decided

`Entity.resolution` is `Resolution(method, score, linker)`, with `method` one
of `caller`, `external_id`, `linker`. The paper's two identity paths — a global
identifier links directly, otherwise a linker decides — become a field, plus
the third case an SDK meets and the paper does not: the caller already knew.

Without it a wrong link is invisible. With it, it is a query — every entity
whose identity came from an embedding match below 0.9 — and a `DIFFERENT` link
(#16) is auditable rather than merely recorded. `linker` also names a platform
resolver when the pass was delegated (#21), so a merge the store made can be
read back and scored the same way as one made here.

*Cost:* one nullable field. Left `None` by anything that did not resolve, which
is itself the answer to "who decided this?".

### 19. The chunk is the unit of the pipeline, and the router sees chunks

`Router.route` takes a `Chunk`, not a `Document` and not a union of the two.
A document is a chunk *source*; document-level routing is a chunker configured
not to split, which is exactly what the default chunker does. `RouteVerdict`
carries `scope: chunk | document` so a router that recognises a marketing page
from its first chunk can skip the rest of that document — the one real reason
to route at document level, without a second method.

This fixes the unit of every metric: one row is one chunk, and the chunker's
configuration is what defines it. The chunker's user-facing size is a word cap
and it never splits mid-sentence, because a router asked "fact or narrative?"
about half a sentence is asked nothing.

The package ships no taxonomy. `RouteVerdict.label` is a free string; one
corpus's fact / policy / narrative split is an argument to a router, not an
enum in this code.

*Cost:* a `Chunk` carries `doc_id`, offsets, text and an index, and nothing
else. An extractor that needs the document's modality or tier looks it up.

### 20. Thirteen Protocols, each with a pass-through default

The generic pipeline is thirteen stages — load, chunk, route, extract, ground,
normalise, resolve, corroborate, score, validate, sink, constrain, infer —
and every one is a `Protocol` in `odke.stages` with a concrete default that is
the identity function, or the nearest thing to one. A caller who wants only
extract-and-sink gets no-ops for the other eleven and never notices them.

This is what makes the package a superset rather than a product. Every corpus
that has been looked at needs a different subset of the thirteen, and nothing
domain-specific — labels, ontologies, tenancy keys, qualifier semantics —
enters the code. It all arrives as data through one of these seams.

The extractor is the one stage with no identity function, so it has no
default. `Grounder` stamps a verdict rather than dropping the fact, so an
ablation can count what would have gone; the `Validator` is the gate.

*Cost:* the four original Protocols changed shape — per chunk and per fact
rather than per batch — before anything implemented them. Later would have
been a migration for every caller.

### 21. A stage the platform also does is warned about, never forbidden

The package sits on top of any platform, and some platforms already do a
stage: neo4j-graphrag resolves after the write, GraphPruner prunes, an RDF
store refuses what breaks SHACL. Doing those twice is waste at best and a
second opinion nobody asked for at worst — but forbidding it would be wrong,
because the two passes are not the same pass. odke's exact match on strong
identifiers before the write is free and never wrong; the platform's fuzzy
pass on names after the write is neither. A caller may want both.

So the mechanism is small. A sink may declare a `PlatformProfile` saying what
its store covers (`resolves`, `constrains`, `prunes`). `Delegated(to=...)`
satisfies every stage Protocol as a pass-through and stamps `to` wherever the
model has a provenance slot — `Entity.resolution.linker` for a resolver, the
verdict's `reason` for a router or validator — so the platform's work can be
read back and scored like ours. And when a real stage is configured on both
sides, `Pipeline` emits one `DoubleStageWarning` at construction and runs
what it was given.

The documentation says plainly that *replace* loses the evidence a pre-write
link would have kept. That is a fact about the platform, not a reason to
refuse it.

*Cost:* a user who ignores the warning gets two resolutions. The evaluator,
not the pipeline, is where that shows up as a number.
