# Decisions

The calls that shaped the design, and what each one cost. Written down because
the reasoning is the part that gets lost, and a decision without its reason gets
reversed by the next person who finds it inconvenient.

### 1. The base install talks to nothing

`pip install openodke` pulls pydantic and typer. No model provider, no database
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
hard dependency means `pip install openodke` drags in a large package before the user
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
turn it off, and then the architecture does not work. The two-model default holds
only when no model is named: naming only `extract` grounds on that model too, at
grounding's small `max_tokens`, so a config pointed at a local server never sends
its passages to the default's vendor.

Defaults name Claude models because something must be the default. Nothing in the
library depends on them, and `ModelRoles.single("ollama/…")` is a first-class
configuration rather than a degraded one. Defaults are pinned model ids, never
moving aliases, and a default changes only along with a new calibration card.

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
is the opposite one. Denials and partial claims are ordinary in real answers —
too many to lose, and too many to merge wrongly.

*Cost:* one more field on a frozen model, and a re-partition of any graph built
before it — which is why it lands in M0, before anything is serialised.

### 15. The ontology says which qualifiers bear identity; the fact carries the answer

#11 is right about `start_time` and wrong about `percentile`. "Uptime 99.9% at
p50" and "uptime 99.9% at p95" are two measurements; with qualifiers excluded
from `signature` they merged into one claim with `support = 2`, and where the
values differed a single-valued predicate saw a contradiction that was not one.
Qualifiers of this kind are ordinary: a price per licence tier, an uptime per
plan, a figure per percentile.

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
and every one is a `Protocol` in `openodke.stages` with a concrete default that is
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

*Renamed by #26:* the tenth stage is `Gate` now. "Validator" names the layer.

### 21. A stage the platform also does is warned about, never forbidden

The package sits on top of any platform, and some platforms already do a
stage: neo4j-graphrag resolves after the write, GraphPruner prunes, an RDF
store refuses what breaks SHACL. Doing those twice is waste at best and a
second opinion nobody asked for at worst — but forbidding it would be wrong,
because the two passes are not the same pass. openodke's exact match on strong
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

### 22. The package is `openodke`; the tool and its data keep `odke`

The distribution on PyPI and the import package are `openodke`. Anyone who has
read Apple's ODKE+ paper and searches for it still finds the package, and the
`open` prefix says what it is: an independent, open implementation of a
published architecture, not Apple's system. That is the pattern OpenCLIP
(OpenAI's CLIP), OpenFlamingo and OpenLLaMA already set, and readers know it.
A bare `odke` on PyPI would read as the paper authors' own release, which is
the affiliation confusion PEP 541 lets a name be reassigned over; a name that
states its independence avoids that before the first upload rather than after.

The command stays `odke` — short, and what a person types — and `openodke` is
installed beside it as the same app. The data keeps `odke` too: the reserved
qualifier and attribute keys (`odke.source_form`, `odke.name_key`,
`odke.conflict`, `odke.score`), the resolver's `odke.native` linker, the RDF
`odke:` prefix and its vocabulary IRI, the Neo4j `odke_*` index and constraint
names and the `// odke:check` marker. Those are a format, not a Python name:
they are documented, and they are what graphs written today contain. Renaming
them would be a data migration nobody asked for, bought only for symmetry. The
repository and documentation URLs stay where they are until the repository
itself is renamed, if it ever is.

The rename came before any release, when nobody depends on `import odke`, so
there is no shim: `import odke` fails.

*Cost:* two names to explain — `pip install openodke`, then `odke run`. The
README and the installation page say it once each.

### 23. Evidence cites the claim-bearing clause, and carries the mention beside it

The grounder is shown the cited span and nothing else, which is #3 working as
intended: a span that does not support the claim on its own cannot be argued
into supporting it. The extractor, left to choose, cited the narrowest text that
distinguished one fact from its siblings — `Ireland` out of a clause listing
three regions — and the grounder, asked whether "Acme Cloud operates in Ireland"
follows from the text `Ireland`, correctly answered `not_found`. An extractor
that cites this way loses every enumerated fact — regions, languages,
certifications — while the facts whose object came from its own clause ground
cleanly. The loss is invisible, because a `not_found` fact looks exactly like a
fact that was properly refused.

So `Evidence.span` is the clause that supports the claim, and `Evidence.mention`
is the narrower span inside it that tells this fact from the others the same
clause states. Both are checked with `Span.is_faithful`, and a mention that is
not inside the clause is dropped rather than refused — a highlight is not worth
a fact. The prompt and the structured-output contract ask for both. Nothing in
the pipeline reads `mention`: the grounder still reads `span`, and the narrower
span is there for a reader, a UI and anyone who wants the tightest citation.

The alternative was a grounder that re-grounds against the enclosing sentence
when a narrow span comes back `not_found`. That doubles the calls on exactly the
facts a corpus has most of, and it repairs a citation after the fact instead of
citing correctly. It remains the fallback if the prompt half of this proves
unworkable.

*Cost:* one more optional field on a frozen model, and a widening of what a
citation means. A fact serialised by 0.1.0 still loads and still grounds exactly
as it did, because `mention` defaults to `None` — but its `span` is a bare
mention, and no migration can widen it after the fact.

### 24. openodke is the verification layer between any extractor and the store

Until 0.2.x the package led with extraction (*text in, a grounded graph out*)
and treated other extractors' output as on-ramps into its own pipeline. The
first public benchmark (PR #105) changed that. It ran on Text2KGBench and
Re-DocRED, with the same model and schema for every system. The extraction half
is not where openodke is strongest; the checking half is.

- **openodke's extractor trailed LangChain's LLMGraphTransformer on recall and
  F1:** 14.4% against 25.1% recall on Re-DocRED, and 42.4 against 47.3 F1 on
  Text2KGBench. On documents it was the most precise system, and it found the
  fewest facts.
- **The grounder ran unchanged over two other libraries' triples.** On documents
  it removed 13–19% of every system's wrong triples, at a tenth to a fifth of
  what extraction cost.
- **LLMGraphTransformer followed by openodke's grounder gave the best
  precision–recall balance measured:** 60.2% precision at 24.0% recall.

So the package becomes the piece between an extractor and the graph store. The
extractor can be a model, patterns, both, or openodke's own. The layer grounds,
resolves, corroborates and evaluates. It also reconciles: the corroborator owns
each fact's support, so when a source changes or disappears, it retracts that
source and retires the facts left with none. Grounding and corroboration are
ODKE+'s own stages, so the name keeps its meaning (#22).

Four calls come with it.

**Resolution is in, and limited.** Corroboration counts sources per fact
signature, and a signature is only as good as its entity keys; without
resolution no claim ever repeats. So resolution stays, with limits:
- it works within a batch, and against what the store already holds through a
  lookup rather than a loaded copy;
- it matches within one type only;
- on proof, the incoming fact takes the store's key; anything weaker is a link
  (#16);
- there is no learned linker to external identifiers.

**Retrieval is out.** It happens at question time, for different users, with a
different measure of success, and the space is crowded. The layer's part in it
is provenance: every fact keeps the evidence it was checked against, so any
retriever can cite it.

**The extractor stays, as a reference.** `LLMExtractor`, `PatternExtractor` and
`HybridExtractor` keep working and keep getting bug fixes, so someone with no
extractor still has *text in, verified graph out*. Work that would extend them
(enumeration recall, coverage, cross-sentence evidence) is parked. Grounded
retry comes back as a hook that hands a refusal to whichever extractor produced
the fact.

**Scope is capped.** There are two pushes: the layer (0.3.0), then the numbers
(0.5.0), which show every extractor with and without the layer, losses included.
Resolution, support lists and the reconciler (0.4.0) are built as far as those
numbers need, and no further.

The published 0.2.x API does not change for this; only the emphasis does.
Nothing is removed, so it is not a reason for a major version.

*Cost:* the package now leads with a claim the experiment only partly supports.
- **Grounding helped on documents,** but on single sentences it removed more
  facts the gold called right than wrong.
- **Corroboration has not been measured at all,** because every fact in both
  datasets has one source.

The multi-source benchmark and the refusal audit exist to settle both. The
README says so until they do.

### 25. A span says who chose it

Most extractors outside this package cite nothing. LangChain's
`LLMGraphTransformer` and neo4j-graphrag emit bare triples. The layer still has
to ground them (#24), and the free span check refuses a fact with no located
span. So a triple with no citation is given the whole text it came from as its
span. That is what the ODKE+ grounder reads anyway: the whole context against
one triple.

But that span is not a citation, and nothing on `Evidence` could say so. Left
unmarked, `odke eval spans` would count every uncited fact as one very wide,
well-supported citation, and the width diagnostic (#23) would drift towards
"citations are fine" in exact proportion to how little the extractor cited.

So `Evidence.span_origin` records who chose the span:
- `cited`: the extractor chose it;
- `located`: openodke found it for a fact that cited nothing, which is the span
  locator (issue #112);
- `context`: nobody chose it, and it is the text the fact came from.

The grounder reads every kind the same way. Only the diagnostics, and a reader,
need to tell them apart.

It is a field, not a qualifier, because it describes the evidence and not the
claim. A fact merged from two sources can hold one cited span and one context
span.

*Cost:* one more optional field on a frozen model. It defaults to `cited`,
because every extractor in this package cites, so a fact serialised by 0.2.x
loads as exactly what it was.

### 26. The Validator is the layer; the gate is a stage

1.0 is two products in one package. The Validator sits between any extractor
and the store, and grounds, resolves, corroborates, gates and writes (#24). The
Evaluator runs a pipeline on benchmarks or your labels and says where it loses
facts and whether the Validator helps. "Validator" is the name people will look
for: `openodke.Validator`, `odke validate`.

But since 0.1 that name has meant one stage: the tenth Protocol, the gate that
decides what is written (#20), and `VerdictValidator`, the gate `odke run`
ships. One word for the whole layer and for one of its stages makes every
sentence about either ambiguous, in the docs and in a traceback. So the stage
gives the name up:
- `stages.Validator` is `stages.Gate`, and `PassThroughValidator` is
  `PassThroughGate`;
- `VerdictValidator` is `VerdictGate`, in `openodke.gate`;
- `Pipeline(validator=...)` is `Pipeline(gate=...)`, and the `odke run` key
  `validator:` is `gate:`.

The method keeps its name, `validate(fact, ontology)`. Renaming it would break
every gate already written, and a gate validating a fact still reads correctly.
The inner stages keep theirs too (grounder, normaliser, resolver, corroborator,
scorer), because each names one job and the decisions above use them.

The old names keep working until 1.0.0, because nothing public is removed
before 1.0 (#24). Each warns where it is used:
- an old class name, at every path where it was public (`openodke`,
  `openodke.stages`, `openodke.pipeline`, `openodke.validators`), is the new
  class itself, read through the module's `__getattr__` with a
  `DeprecationWarning`. `isinstance`, subclassing and old pickles keep working.
- `Pipeline(validator=...)` and `Pipeline.validator` warn and pass through to
  the gate.
- a config with `validator:` warns and runs. `odke run` prints that warning
  itself, because Python hides a library's `DeprecationWarning` from the person
  at the terminal. Naming both keys is an error.

Two names do not follow the rest:
- **`openodke.Validator` is the gate only until #129.** Then it becomes the
  layer: one entry point that runs ground, resolve, corroborate, gate and write.
  Its warning says so now, so a caller learns the name is about to change
  meaning, not merely move.
- **The run report keeps `stages.validator`.** The gate's counts sit under that
  key in `KnowledgeGraph.stats`, in every JSONL manifest written so far, and in
  whatever reads those. A report key is a format, and it changes with the
  report's schema version, not with a Python name.

*Cost:* two names for one thing until 1.0.0. `openodke.Validator` changes
meaning instead of disappearing, so code that ignored its warning will not
fail at import; it will fail later, wherever it used the name as the gate. And
until the report's schema moves, a config that says `gate:` produces a report
whose line says `validator`.

*2026-10-09, for 1.0.0 (#129):* the planned break. `openodke.Validator` is the
layer, `openodke.validator.Validator`, and the alias that named the gate there
is gone. It goes without a release that warns in between: the name now
resolves to a working class, so a warning would fire on every correct use.
`stages.Validator`, `pipeline.Validator` and the other old names still name
the gate and still warn. The layer's method is `validate`, the gate's method
name too, so a `Validator` fits the `Gate` Protocol by shape. Code that still
passes it as the gate builds a pipeline. The first `validate(fact, ontology)`
call then raises a `TypeError` naming `openodke.Gate`.

### 27. A prompt is a registered, versioned object

A calibration card says what it measured: a model, a prompt, a dataset. "The
grounder's prompt" names no prompt, because it is whatever the source said on the
day of the run. A one-word edit changes the instrument, and a card measured before
the edit then describes a grounder nobody can run.

So every prompt openodke sends is registered in `openodke.prompts` as
`id@version`, with its text, its SHA-256 and its source, and every model call
records the key it sent. A text is never edited in place. `prompts.lock.json`
holds each key's hash, and the suite fails on a mismatch with "bump the version".
A change is the next version, and the old one stays registered, so an old card
still names a prompt that exists. The stages send the latest.

Only the fixed instruction text is registered. How a claim, a passage or an
ontology snippet is rendered into the message is code, and the package version
names it.

The same suite fails when a prompt shares twelve words in a row with a passage
from a benchmark gate split. A prompt tuned on the test makes the test
meaningless.

*Cost:* fixing a typo is a new version, and the next run names a different
prompt from the last. That is correct: it sent one.

### 28. An inverse comes from the schema, is marked, and is never grounded twice

A passage that says "France contains Brittany" has also said that Brittany is
located in France. An extractor states the claim once, in whichever direction
the sentence ran. On Re-DocRED that cost 25 gold facts (#106), each the
partner of a fact already extracted. The ontology already knows the pairs, so
the partner needs no model: `Predicate.inverse_of` and `Predicate.symmetric`
declare them, and the pipeline adds each edge's partner after resolution and
before corroboration. Declaring an inverse on one side is enough; loading fills
in the other, as `owl:inverseOf` holds both ways, and `validate()` refuses one
that does not hold both ways or whose ends do not swap.

Four calls come with it.

**A step, not a fourteenth stage.** The ontology decides everything the step
does, so there is nothing for a caller to swap in, and a Protocol whose only
sensible implementation is ours would be a seam in name only (#5, #20). It is a
step in `Pipeline`, on exactly when the ontology declares a pair.
`inverses=False`, or `inverses: false` in a run config, turns it off.

**Marked the way the stages mark their work.** The partner carries
`qualifiers["odke.derived"]`: the rule, and the signature of the fact it came
from. It says how the claim got into the graph, which is what the reserved
`odke.*` keys are for, and every sink already writes qualifiers. The link is a
signature, not an id, because a corroborator merge keeps one id of several
and the signature survives it. Hashed, it is also the Neo4j relationship's
MERGE key.

**Never grounded twice, gated with its source.** The partner cites the
source's evidence and keeps its verdict. Asking the grounder again would pay
for a question already answered, and could get a different answer. Because the
verdict is inherited, the gate refuses the partner whenever it refuses the
source on its verdict (#20), so a refused fact has no partner in the graph.
Support is counted from evidence, so the two share it, and a source retracted
by the reconciler (#116) takes its partner's evidence with it.

**A stated fact wins.** When the batch already states the partner, nothing is
derived. The stated fact stands on its own evidence, and no duplicate reaches
the store, whether or not a corroborator runs. The price is that a claim stated
forward in one document and backward in another counts one source in each
direction, not two.

*Cost:* the graph holds facts no source stated in that direction. They are
true whenever their source is, and marked, but a query that wants only stated
facts has to filter on `odke.derived`. Errors are doubled along with facts. On
the published Re-DocRED runs, the six pairs #106 named raise openodke's recall
by 1.2 points and cut its precision by 5.1. Of its 59 partners, 21 are gold
facts, 15 are partners of facts the gold calls right but leaves the other
direction out of, and 23 come from facts the gold calls wrong. And since the
step is on by default, adding an `inverse_of` to a live ontology changes the
next run's graph; `diff` calls that compatible, and calls removing one breaking.

### 29. A comparison has three verdicts, and inconclusive is not a pass

Two runs on the same items are compared item by item: a paired bootstrap over
the items, the 95% interval of the difference, and *better*, *worse* or
*inconclusive*. Folding inconclusive into "pass" is how a regression ships on a
set too small to see it, so inconclusive is never printed as "no regression". It
carries the detection limit, the smallest change the set could have seen. One
metric is primary and the rest are guardrails that print and never decide,
because every metric that decides is one more chance of a false alarm.

The CI gate fails *worse* and passes *inconclusive* unless asked. Failing
inconclusive by default would fail every small change on a small set, and teach
people to delete the step.

The bootstrap is the standard library's, not numpy's, so the base install keeps
its promise (#1). A resample of pass/fail items is a multinomial draw over their
four pairings, not one draw per item, which keeps 2,000 resamples fast without
an array library. LangChef's `compare` was the other candidate. It takes
pass/fail verdicts only, not a corpus F1 resampled by document, and needs Python
3.12 or 3.13 with numpy, scipy and pyarrow.

*Cost:* an inconclusive result passes CI, so a real regression smaller than the
limit can ship. The limit printed beside it says how big that could be, and the
A/A test holds the false-alarm rate near 5%.

### 30. A cached answer is keyed on everything that changes it, and costs 0.0

A batch rerun after a crash, a sink change or a gate change asks the models
the same questions again, and pays for them again. So `CachedClient` answers
a request it has seen from a store (#156). A cache is only as good as its key,
and a key that leaves something out replays an answer to a different
question. So the key holds everything that changes the answer: the model and
its `base_url`, every message, the schema, `temperature`, `max_tokens`,
`extra`, and the registered prompts the messages carry (#27). It leaves out
only `timeout` and `api_key_env`, which decide whether a call succeeds, not
what it says. A miss costs one call. A false hit is a wrong fact that looks
grounded.

An error is never stored, because the next run should ask again. A reply the
caller rejects is stored, because rejecting it is the caller's reading, and the
repair turn that follows has a key of its own.

A hit costs `0.0`, not `None`. `None` means the provider did not say (#7), and
here nobody was asked: nothing was spent. The meter records the hit as a call,
marked `cached`, so a run that cost nothing still says why.

The store is files: one JSON file per key, renamed into place. It needs nothing
beyond the standard library, a person can read it, and threads and processes
can share it without a lock. SQLite was the alternative. It is in the
standard library too, but its write lock makes concurrent writers wait, and
nothing here needs a query.

*Cost:* a cached run is not a fresh sample. At a temperature above zero the
rerun replays the draw it made the first time, and nothing expires. Asking
again means deleting the directory or naming a new one. Two threads asking the
same new question at once both pay for it.

### 31. The store is looked up, never loaded, and a lookup never changes it

#24 put resolution against the store in scope "through a lookup rather than a
loaded copy", and `EntityIndex` could not keep that promise. It is a
`Mapping[str, Entity]`, so resolving a batch against a graph of a million
nodes meant reading a million nodes first. But blocking already says which
entities a batch can match: the same type and a shared key, external id,
domain, or first or last name token. Those are questions a store's indexes
answer.

So `StoreLookup` is a Protocol with one method, `candidates(entities)`: for
each entity, by key, the store's entities sharing a block key with it, within
its type, and within a tenant when the lookup is scoped to one. It is not a
fourteenth stage. It is the resolver's way into the store, and
`NativeResolver(lookup=...)` is how it is used. `MemoryLookup` answers from
today's mapping, in memory. `Neo4jLookup` answers from the indexes
`bootstrap()` creates, each batch in one read transaction, each block key
asked once. A type with no index is not read, and says so: a lookup that
scans is a load by another name.

Four calls come with it.

**A proof re-keys the incoming facts, and the stored entity travels as
stored.** On a shared id or domain the incoming facts take the store's key and
the store's entity as the store holds it: label, aliases, id, resolution and
attributes. The sink's `MERGE` then writes every property back to what it
already is. Nothing the batch knew about the entity lands on the node. Merging
the incoming names into it would change a stored node on one batch's say-so,
which is the replace #16 refuses, one property at a time.

**Anything weaker is a link**, as within a batch (#16): `SIMILAR` at the same
threshold, `DIFFERENT` on a disagreeing id. Store entities are never compared
with each other. Resolving the store against itself is a different job, and a
batch that proposed links between nodes it never mentioned would be doing it
unasked.

**The batch's own statement of a key wins.** A batch that states a stored key
itself, or a key its caller chose (`method="caller"`, #18), keeps its own
entity, as it would with no store. The sink has always updated a node a batch
restates, and a lookup does not take that from the caller.

**An embedding widens what is compared, never what counts as a match.** The
vector lookup is a slot. With `embed`, the caller's function, the nearest
stored labels of the type are candidates too, judged by the same rules. There
is no embedding dependency, no model, and no new kind of evidence in the
resolver.

The tenant is a filter on a property until tenant keys arrive (#159). The
defaults (the resolver's 0.9 for `SIMILAR`, 100 hits a name token, five
nearest vectors) were set before anything was measured, and are documented as
defaults.

*Cost:* the incoming entity's other names reach the store only through its
`SAME_AS` link, and in Neo4j not even there. That link's source key has no
node, so the sink's `MATCH` finds nothing to hang it on, which is also true
of a within-batch merge's losing key; a JSONL sink keeps it. A lookup reads
which indexes exist once, so an index created after its first batch is seen
by the next lookup. And the first measurement is within documents only:
Re-DocRED has no ids, so it measures the `SIMILAR` path alone, and identity
across documents waits for the multi-source benchmark (#117).

### 32. A budget stop keeps what the run has

A batch that runs away costs real money before anyone looks. So a run takes a
budget (#157): USD, input tokens, output tokens and calls. One ledger counts
every call from every stage and thread, and a call that would go past a limit
is refused before it is made. Before the call there is only an estimate:
characters over four for the input, `max_tokens` for the output, and the run's
own USD per token for the price. The output estimate is the most the call can
return, so a run can stop with room left, and never past the limit. The one
exception is USD before the first priced call: there is no price to estimate
from, so that check runs after the call, and the run can overshoot by the
calls already in flight.

The stop is the run's, not the stage's. The first refused call stops the
ledger, and every call after it is refused too, so a stopped extraction does
not hand its room to the grounder. The run then ends as if the remaining calls
had answered nothing. Facts grounded so far keep their verdicts, a fact the
stop reached first stays `UNCHECKED` and is counted, the sinks write, and the
report says where it stopped. `UNCHECKED` already means "not asked" (#20), and
the next run picks those facts up, from the cache for whatever was answered
before (#30).

Raising was the alternative, and it throws away what was paid for. The CLI
exits 3, so a scheduler can tell a budget stop from a failure (1) and a bad
config (2) without parsing the report.

*Cost:* a stopped run writes facts nobody checked, marked `unchecked`, and a
gate that accepts `unchecked` writes them to the store. The output estimate
wastes room: a call that would have returned 50 tokens is refused for its
`max_tokens`. And a provider that reports no cost never reaches a USD limit.
