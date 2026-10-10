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

*2026-10-10, for #145:* the key also holds `ModelSpec.repeat` when it is not
0. Gold adjudication asks one question three times on purpose, and a key
without it replays the first answer as all three (#36). Draw 0 adds nothing to
the key, so every entry written before is still found, and the first draw
shares its entry with an ordinary call of the same question.

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


### 33. A support list names each source, and Neo4j keeps it as parallel lists

`Fact.support` was a count. The corroborator already worked out which
independent sources back a claim, then kept only the number. A reconciler
needs the names (#115, #116): when a source changes or disappears, which facts
lose support, and is anything left? So `Fact.supported_by` names them, one
`Support` per source: the key the corroborator grouped on, the documents, the
best tier and the newest clock. `support` is its length whenever it is filled,
and the suite checks that on every fact it makes. The list is empty on a fact
no corroborator merged, and on a fact serialised by 0.2.x, which still loads
(#4).

Three calls come with it.

**The list is the evidence, counted.** It comes from the merged evidence by
the rules `support` already used. A host is one source, and a group of
near-duplicate documents is one entry under its least key (#154). A derived
fact cites its parent's evidence, so it carries the same list and adds no
entry (#28). A claim with a member that cites nothing keeps its count and
names no source. A list naming some sources and not others would be wrong
about which facts a retraction leaves with none.

**In Neo4j, parallel lists on the relationship, not support nodes.** A fact
is a relationship, and a relationship cannot have relationships. Support
nodes would need every edge reified as a node, which changes the shape of
every query on the graph for one property. Evidence is already parallel lists
on the relationship, for the same reason. A list holds no lists, so the
documents are flattened into `support_doc_ids`, each beside its source in
`support_doc_sources`. The bulk sinks and NetworkX write the same properties
from the same `provenance_of`. RDF has nodes to spare, so it gives each source
an `odke:Support` node in the evidence's vocabulary.

**The scorer keeps a listed count.** `EvidenceScorer` recounts `support` only
on a fact with no list, so a scorer configured with another `source` cannot
leave a count its list disagrees with.

*Cost:* five more properties on every relationship, read back with
`support_from` rather than as one property. And the list can be recounted
from the evidence, yet is kept beside it, so a store can be reconciled without
the `source` function that made it.

### 34. The pair judge asks in both orders, and only where the rules leave a pair open

The resolver's rules settle a pair with an id, a domain or a name score at the
`SIMILAR` bar, and below the bar nothing links. That is where its misses are:
a surname and the full name, a short form, a former name. Names alone cannot
tell whether "Lovelace" is the "Ada Lovelace" two sentences earlier; the
sentences around them can. So a model reads them, and three things about how
are decided here.

**Where it runs.** Only on a pair no rule settled, whose name score is in a
band below the bar: from the judge's `low`, 0.7, up to the resolver's
`threshold`, 0.9. Above the bar the rules stand; a model is not asked to
overturn them. A pair whose ids or domains disagree is never asked: evidence
against beats evidence for (#16), and a model's "same" would be resemblance
arguing with proof. Below the band the names share too little for a call to
be worth it. 0.7 was read off Re-DocRED's dev split (nearly a third of the
533 blocked pairs between 0.7 and 0.9 are one entity), never its test split or
label set R, and is a default until the calibration card on R (#151) says
otherwise. The judge is off unless given.

**The swap rule.** Each pair is asked twice, as (A, B) and as (B, A). A model
judging two items in a row favours a position (Zheng et al. 2023, §3.4), and
on a pair the two orders are the same question, so they must give one answer.
"same" counts only when both orders say same, and "different" only when both
say different. Anything else is unsure: a disagreement, an unsure, or an
answer that could not be read. Each decision keeps both answers and whether
the orders disagreed, and the stats count calls, swapped calls and
disagreements, so position bias is a number a run reports rather than an
assumption about the model.

**What a decision makes.** "same" is a `SIMILAR` link, never a merge. Only a
proof re-keys (#16, #31): a model's reading of two sentences is better
resemblance, not proof, and a wrong merge cannot be undone where a link can be
ignored. "different" is a `DIFFERENT` link, because a pair the judge looked at
and rejected is worth a query, as a disagreeing id is. The reason names the
judge, its prompt key and its model, in the string #16 already has. Unsure is
no link at all, and goes to a review queue when one is given: a JSONL file in
the pair sheet's own row format, so `odke label make pair` reads it as it
stands, and the person's labels come back as `reviewed`, decided in the
judge's place with no call and a reason that starts `person:`. A person's
"same" is a `SIMILAR` too: a re-key is a proof's, and a person who means one
adds the id.

The prompt is LLM entity matching's (Peeters, Steiner & Bizer 2023): two
mentions, their types and their contexts, the knowledge of names allowed
(abbreviations, former names) and a decision on names alone forbidden. It is
`pair@1`, and its user message is `pair.user@1`, so a calibration card names
both (#27). It asks the `ground` role's model, since it is the same size of
question, metered as the stage `judge`.

*Cost:* two calls a pair in the band, about one pair a document on Re-DocRED's
dev split. An unsure pair costs a person's time, and is queued once. A pair
with a side that has no text is not asked, since the prompt forbids deciding on
names, and goes to the queue. A stored entity has text only through
`store_context`, because the store keeps offsets and not passages, so against
a store without it every pair in the band waits for a person. A budget stop
ends the judge's calls and not the resolver: a pair it reached is unasked, no
link and no queue, and the next run asks it.

### 35. A write merges with the store, in the corroborator, before the score

A rerun into Neo4j found each relationship by its signature and replaced its
properties (`SET r =`). So a claim stated again by a second source replaced
the first source's support instead of adding to it, which loses exactly what a
support list (#33) exists to keep. Now a fact the store already holds merges
with an incoming one by signature, and its list grows (#153).

Four calls come with it.

**The sink reads; the corroborator merges.** A `FactLookup`
(`stored(facts)`) says what the store holds under each signature in the
batch. The corroborator merges each stored fact into its claim as one more
member, after the batch merges and before the scorer and the gate. So the
score counts every source, and the receipts written back agree with each
other. A merge in the sink, at write time, would write a support of two
beside a score computed on one.

**One read per write, through the signature index.** `Neo4jSink.stored` is
one read transaction: per predicate, an `UNWIND` of the batch's signature keys
through the index the relationship uniqueness constraint brings. A predicate
with no index is not read and warns, because a read that scans is a load
(#31). `JsonlSink` is a store only when asked (`merge=True`). By default each
write replaces its files, so there is nothing to merge with.

**A statement wins across the store, as within a batch (#28).** A derived
fact merges only with a derived one. A derived twin of a stored statement is
dropped, and a statement replaces a stored derived twin. A derived fact then
takes its parent's merged list, so the two stay shared.

**The Validator merges by default.** It hands every sink that is a
`FactLookup` to its default corroborator. A corroborator passed in merges
with the stores it was given, and a dry run reads no sink. `odke run` does
not merge yet: its corroborator comes from the config, which has no key for a
store.

*Cost:* a read transaction before every write. A stored fact's receipts come
back from the relationship's properties, which hold no quotes or mentions, so
its evidence returns without them. A store never bootstrapped has no
signature index, so nothing merges with it, and the warning is the only sign.

### 36. Without gold, the judge is corrected, the strict number stands, and pooled recall overstates

Three evaluators score a pipeline where gold is missing or incomplete (#144,
#145, #146). Each has a number that is easy to print and wrong to trust alone.

**A judge's precision is corrected by labels.** The grounder can grade every
fact, and the share it supports is a precision. It is the judge's, though,
biased whichever way the judge leans, and agreement on a sample does not say
which way. So a person labels a random sample, and prediction-powered
inference adds the mean gap between the labels and the judge on it to the
judge's share (Angelopoulos et al. 2023; ARES uses it for LLM judges, arXiv
2311.09476). That removes the bias on average, and the interval is narrower
than the labels' own whenever the judge mostly agrees with them. The report
prints all three numbers: judge only, corrected and labels only.

- **The interval is PPI's closed form, not a bootstrap.** It is one formula
  with no resampling. Its variance, Var(y)/N + Var(y − f)·(1/n − 1/N), takes
  out the overlap between the two terms, because the labelled facts are among
  the judged ones, and it is exact when every fact is labelled.
- **It treats facts as independent,** where every other range in the report
  resamples documents. The sample is drawn fact by fact, so few labelled facts
  share a document.
- **Under 100 labels it is an "uncalibrated estimate".** In simulation a judge
  that misses one true fact in ten covers the truth 94.8% of the time at 300
  labels and 90.8% at 50.
- **Recall is never claimed.** The coverage report stands in for it.

**The strict precision never changes.** Adjudication lists a prediction the
gold lacks as possibly missing from gold when the grounder supports it in 2 of
3 runs. The adjudicated precision is printed beside the strict one and never
replaces it, because it trusts the grounder on exactly the facts in question.
Label set G measures how far that trust goes. The three runs are three calls:
each carries its run index as `ModelSpec.repeat`, which no provider sees.

**Pooled recall carries its caveat.** Pooling two or more runs' supported
facts gives each run a recall relative to the pool. That overstates true
recall, because the pool misses whatever every run missed. The caveat is a
field of the report section, not a footnote, and each run's coverage report
sits beside the number.

*Cost:* three numbers where a reader wanted one, and a judge-only number
printed though never trusted, since hiding it would hide the bias it shows. A
calibrated number still needs a person to label 100 facts or more.
Adjudication triples the grounding bill for the predictions the gold lacks.
And with the response cache on, the first of the three draws is not always
fresh: the cache keys `repeat` only when it is not 0, so run 0 is answered by
any earlier grounding of the same question.

### 37. A source that changes is retracted, and a fact left with none is retired, not deleted

A layer that only adds keeps facts whose text no longer exists. Support lists
(#33) make a change mechanical. A deleted document comes out of the evidence
and the support list of every fact it backed. An updated one is retracted the
same way, then its new version is validated, so the facts it still states
merge with the store again (#35) and regain it (#116).

Four calls come with it.

**Retired, not deleted, by default.** A fact left with no source was not
proven false; it lost its evidence. It keeps its claim, its edge and its valid
clock, and `Fact.retired_at` records the transaction time it lost its last
source. Graphiti invalidates an edge rather than delete it (#17), and this is
the same instinct. A retired fact has no support and is never the value:
- it is not projected;
- no cardinality check counts it;
- no plain RDF triple asserts it.

A source that states it again brings it back, and `hard_delete` removes it
instead.

**One change, two commands.** `odke reconcile --delete <doc-id>` retracts a
source that is gone, and needs only the store. An update needs the whole
layer, to ground the new version's facts against its text. So it is
`odke validate --update`, which retracts each text it is given before it
writes. An `--update` on `odke reconcile` would have repeated every option
`odke validate` has.

**Found through an index, rewritten in one transaction.** `bootstrap()` adds a
full-text index over `evidence_doc_ids` per predicate. It uses the keyword
analyzer, so a document id is one exact term. `Neo4jSink.retract` finds every
fact a document backs through those indexes, never by a scan, and a
relationship type with none is not read and warns (#31). In one write
transaction it then:
- rewrites the evidence and support lists, `support` and `retired_at`, by
  element id;
- projects each value again from the claims left.

**Idempotent by construction.** A fact that no longer cites a document is not
touched, so retracting it twice changes nothing, and a retired fact keeps the
time it was first retired. An update run twice ends where it ended once: the
second run retracts what the first wrote, then writes it back.

*Cost:* a full-text index per predicate. An update retracts before it
validates, so a run that fails in between leaves the old version retracted
and the new one unwritten; running it again finishes it. The reconciler does
not rescore, so a fact that lost a source keeps its confidence until it is
next validated.
### 38. A miss goes where its evidence first points, and a fix is priced by arithmetic

A recall number says how much a pipeline lost. The Evaluator also says where
(#140) and what to change first (#141), and both are counts on the run's own
evidence: the trace of what the extractor was offered, the gate's refusals, the
predictions that came near a gold fact, the documents. Nothing is judged by a
model.

**One bucket per miss, the first its evidence fits, in three tiers:**
- what the pipeline did: relation never offered, refused by the Validator, wrong
  relation, inverse direction, surface form, and the same triple scored apart;
- the condition the fact or its document was in: cross-sentence, output
  saturation;
- what is merely missing: entity never extracted, both seen but not linked.

Within each tier the order is the one #140 proposed; the tiers are not. There,
the two symptoms came before cross-sentence, saturation and surface form. But a
miss with no prediction near it always has an end missing or two ends unlinked,
so in that order the last three buckets could never fill, and the planted output
cap of #147 would read as "entity never extracted". A bucket that says what the
pipeline did beats one that describes the fact, and that beats one that names a
symptom. "Same triple, scored apart" is a tenth bucket: the matcher pairs one
prediction with one gold fact, and compares type, polarity and qualifiers, so a
stated triple can still be a miss. A bucket the run cannot measure is `null`,
not zero: with no trace, nothing is known about what was offered.

**Expected gain is arithmetic on the counts, never an opinion.** A fix that can
be replayed on the run's own output is exact: accepting the refusals writes the
refused gold facts back, and the inverse step adds exactly the partners
`Pipeline` would (#28), so recall gained and precision lost are both counts.
Any other fix assumes its bucket is found as often as the run finds the facts
its cause does not touch: never-offered relations at the recall on offered
ones, cross-sentence facts at the same-sentence recall, a saturated run's long
documents at its short ones' recall, and the rest at the run's recall. That is
the expected gain, and fixes are ranked by it; every miss in the bucket found is
the ceiling. A reference run, another system on the same gold, gives a third
figure at its rate. Surface form is worth zero until #143's judge confirms a
candidate.

On the published Re-DocRED runs this reproduces the 4 October estimate for
offering every relation: +2.6 recall points at openodke's own rate, +7.3 at
LangChain's (+7.4 by hand), +20.8 at most. The rerun measured +4.5.

**Each prediction is kept beside its measurement.** A report appends its
predicted fixes to a track record, a JSON Lines file beside it, naming the run
by a hash of its per-document outcomes. `odke eval compare A B` appends the
recall B measured for a fix A predicted, when `--applied` names it or the two
configs differ at that fix's knobs alone. The file only grows.

*Cost:* the rate is an assumption, and the ranking inherits it. On Re-DocRED,
cross-sentence ranks first at +6.4, priced at the same-sentence rate, which no
fix has yet been measured against. The order is a choice too: a cross-sentence
fact whose entity was never extracted counts as cross-sentence. Gains overlap,
so they never add. And the inverse replay matches a partner on the triple,
which is what Re-DocRED scores but not every detail a labelled `Fact` carries.

### 39. The fact-equivalence judge reads only the pre-filter's pairs, in both orders, and its score sits beside the strict one

A scorer compares strings after a normaliser. "Gabby Logan" for "Gabrielle
Nicole Logan", or "track and field athlete" for "athletics competitor", is one
miss and one false positive, and no normaliser can know otherwise without the
passage. A model can (#143). Three things about how are decided here.

**A deterministic pre-filter decides which pairs reach it.** A gold fact the
scorer counted missed and a prediction it left unmatched, in one document, make
a pair when they have the same relation and polarity, one end equal after
normalisation, and the other end not exactly equal as written. A name is
compared by `name_key` against every name the gold end goes by; a value by
`normalise_value`. Nothing else is asked. A pair with only the relation in
common is not a question of wording, and two identical triples are the
scorer's question, not the judge's. So the judge sets one bucket, and its bill
grows with the near misses, not with misses times predictions. The diagnosis's
surface-form bucket (#38) takes its candidates from the same pre-filter, and
with `--lenient` it reads the same decisions instead of asking again, so the
bucket and the lenient score agree. With no judge, only a candidate whose
other end is a near name is counted there, as unconfirmed, so a plain wrong
value is never taken for wording.

**The swap rule, as the pair judge's (#34).** Each pair is asked with the gold
fact first and with the prediction first. "same" counts only when both orders
say same, and anything else counts nothing. The gold fact is the reference, as
in reference-guided judging (Zheng et al. 2023). A quantity, date or number
must have one value, EnterpriseRAG-Bench's correctness rule (arXiv
2605.05253): "1988" is not "27 October 1988", however the facts are worded.

**Lenient beside strict, never instead.** A pair judged the same moves one miss
and one false positive to a hit, at most once per gold fact and once per
prediction. Each row's lenient precision, recall and F1 sit beside its strict
ones in the report's `lenient` section (schema 1.2), resampled on the same
draws. The strict number is the one that compares across runs and papers; the
lenient one says how much of the gap is wording. Like adjudication (#36), it
trusts a model on exactly the facts in question, so label set F measures that
trust: 300 pairs the pre-filter made from the published comparison, labelled
blind, 100 dev and 200 gate by document. A sheet never says which fact is
gold, and each item's order is drawn from its id.

The prompt is `fact_equiv@1` and its user message `fact_equiv.user@1` (#27).
It asks the `ground` role's model, the same size of question, metered as the
stage `judge`.

*Cost:* two calls a pair. Until F's calibration card exists, the lenient score
is an unvalidated number printed beside a validated one. And the pre-filter is
a normaliser's: a pair whose ends `name_key` matches on neither side never
reaches the judge, so the lenient score is a floor on what wording costs, not a
measure of it.

### 43. A batch's own look-alikes are one entity unless something keeps them apart

#16 let only a proof re-key, and #34 made the judge's "same" a link. Issue
#148 asks for more: mentions in one batch that look alike and mean the same
thing in context become one entity, with every name kept. That is consistent
with #16 once it is said where it happens. #16 refused to replace a stored
node, because a merged node cannot be re-run at a new threshold. A batch's
mentions are not nodes yet. Re-keying them onto one key before anything is
written replaces nothing, and running the batch again, with the option off,
undoes it.

**Only what the batch introduces.** A merge joins two entities that facts of
the batch state, by keys the store does not hold. A pair with a store entity
is still a link (#31), so a stored node changes on proof alone. A group of
mentions proven to be a stored entity takes that entity as stored, as one
mention would.

**Evidence against wins, through any chain.** `NativeResolver` blocks as it
always has, by type (an ontology type's aliases being that type) and a shared
token, id or domain. A pair above the `SIMILAR` bar merges unless something
keeps it apart. The judge's "same", or a person's, merges a pair in its band.
What keeps a pair apart:
- a disagreeing id or domain, or the judge's "different";
- names that both carry numbers or legal forms, and differ in them;
- given `embed`, sentences whose cosine is below `context_floor`.

Merges are made strongest first, and one that would put a pair kept apart, two
caller keys or two ids that disagree in one entity is refused. A mention alike
to two entities kept apart could be either, so it merges with neither.

**A merge is a `SAME_AS`, and says what it rests on.** `SAME_AS` now means "one
entity, re-keyed", not only "proven". A proof has score 1.0 and an id or
domain in its reason; a batch merge has the name score and a reason starting
`in-batch merge:`. The entity's `resolution` is `linker` with the weakest score
its group was joined at, so a group the judge joined is a query (#18). The
embedding is the caller's function, as the lookup's is (#31): no dependency
and no model. A pair kept apart keeps its `SIMILAR`, with why in its reason.

**Off by default: the rule passed, and the supplement overrode it.** The rule,
stated before measuring: on only if, on label set R's gate split, the
judge-free batch's precision of merged pairs is at least the resolver's as it
was, at equal or better recall. R has no ids, so the resolver as it was merges
nothing there, and its `SIMILAR` links stood in as its merges. The batch
merged 4 gate pairs, 2 right (50.0%, recall 1.0%), against 7, 2 right (28.6%,
1.0%), so the rule passed. It was not enough. Four pairs against seven is
little to stand on, and R leaves out names with one key, which are most of
what the option merges. Re-DocRED's test documents as they come said more: of
the 103 pairs the batch merged, 52 are one entity by the dataset's clusters
(50.5%). About half the rest are one entity the dataset splits, and the others
are the name score's own mistakes: "South Africa" and "South African", "West
Germany" and "East Germany", "Borderlands" and "Borderlands 2", "José Maria"
and "María José". They score above the judge's band, so no judge sees them. A
wrong merge is the worst error #16 names, and a default that re-keys needs
stronger evidence than a link does. So `normalize_batch` is off in
`NativeResolver`, the Validator and `odke run`, and a caller turns it on.

**What would turn it on.** The calibration pass on R with the real judge's
card (#151), together with `embed`'s context check measured against the
mistakes above the band; or the owner's audit labels on R, read back, showing
the merges hold. Either is a measurement, run by this decision's rule again.
With R's own labels as a perfect judge, the band takes gate precision to 93.8%
at recall 15.0%: the ceiling a judge can reach, not a number it has reached.

*Cost:* a caller who wants a batch's look-alikes as one entity must ask, and a
run that does not ask keeps `SIMILAR` links where an obvious spelling variant
("Acme Corp." and "ACME Corporation") could have been one node. Above the band
only `embed` can catch the name score's mistakes, and its `context_floor` of
0.5 was set before anything was measured. A gold keyed by each text's names, as
the end-to-end example's is, scores a right merge as a miss, so that example
pins the option off.

### 46. Corroboration is scored on the graph, by support, with checked alignments as sources

Corroboration had never been measured (#24): Text2KGBench and Re-DocRED state
each fact once. T-REx aligns Wikidata triples to Wikipedia abstracts, and a fact
can be aligned in more than one, so it has facts with a number of independent
sources (#117). Four calls come with measuring it.

**A source is an abstract a checked aligner placed the fact in.** T-REx's
Simple-Aligner aligns any two linked entities in a sentence to a triple Wikidata
holds between them, and most of its multi-abstract alignments are co-mentions:
Indonesia and the United States in a list of countries, aligned to "diplomatic
relation". With every aligner, 20.4% of the sample's entity facts are in two or
more abstracts; with the two aligners that check the sentence, 4.6%. The gold
and its source counts use those two. A corpus at 4.6% would barely move a
number, so the set is drawn around multi-source facts: 78 of its 513. It
measures what agreement is worth, not how often a corpus has it.

**Precision is factual, and closed-world.** Every T-REx alignment is a Wikidata
triple whatever its sentence says, so a predicted triple is right when T-REx
aligns it anywhere in the sample, by any aligner. A true fact the 2017 slice
never aligned counts as wrong, so the number is a lower bound. The precision
against each abstract's own gold is printed beside it, as Re-DocRED's is.

**On is the edges two or more sources back, scored edge by edge.** The third
row of the ablation merges claims and counts their sources, but the gate
refuses nothing a verdict did not, so its precision barely moves. The paper's
98.8% is after ranking. Once the gate keeps only supported facts and the
extractor gives one confidence, `EvidenceScorer` rises with support alone, so
ranking by the score and keeping support of two or more are the same cut.
Support belongs to an edge, not to a document, so the graph is scored one edge
per distinct claim.

**One entity type.** The resolver matches within a type (#24), and a fact told
by two abstracts must not stay two claims because the extractor typed France as
a Country once and a Location once. The extractor loses the types' hint, which
none of T-REx's gold needs.

*Cost:* the support cut's precision is measured on a set built to contain
agreement, and a corpus with less would lose more recall to the same cut. A
name stands for every id T-REx linked it to anywhere in the sample, so an
ambiguous name matches any of them. The sample is the first 10,000 of about 3
million abstracts, so a fact's sources in it undercount its sources in the full
set, which figshare would not serve to the bench machine. And sources are
counted on name keys: a fact stated under two names is two claims of one source
each, which the run reports beside what ids would have pooled.

### 40. A run writes its manifest after the sinks, and a replay refuses what changed

A graph says what it holds, not how it was made. When a number moves, the
first questions are which model, which prompt, which schema and which texts.
Prompts are versioned (#27) and an ontology has a fingerprint, so a run can
name each one. Every run now writes a manifest (#160): `odke run`,
`odke validate`, `odke ground` and `Validator.validate`.

Four calls come with it.

**The JSONL manifest is the run manifest.** `manifest.json` already held the
graph's counts and stats. The run adds its own fields beside them, after the
sinks have written, and never replaces a key the sink wrote. An older reader
finds what it found. In a store that merges, the counts stay the files', and
the run's own counts sit apart under `counts`. Written after the sinks, the end
time is the real end, and the JSONL sink still knows nothing about runs. A run
with no JSONL sink writes the same document beside its config, so every run
writes one.

**The config hash is of the resolved config, without its secrets.** Every key
is filled in and each model role is the spec `ModelRoles` resolves it to, so a
config that leaves a default out and one that names it hash alike. A value
under a key that names a secret, and the password in a URL, is redacted
before the hash. The hash then identifies the run, not the credential, and a
manifest can be passed around.

**Asked for and served, both.** A config may name an alias. The provider's
answer names the model that served it, so the manifest records both, by role.
A replay runs the config as written. If the alias has moved, the next manifest
says so.

**A replay is the same run or no run.** `odke run --from-manifest` refuses,
before anything is written, when the ontology's fingerprint or any document's
hash differs, and it takes no override. Otherwise it would be another run
under an old name. A different openodke only warns: an upgrade is not a
reason to refuse. `odke validate` and `odke ground` record their options
instead, and are run again by hand.

*Cost:* one hash per document, so the manifest grows with the corpus, and
every text is hashed once more. A secret under a name the pattern
does not know is written as it is. A run with no JSONL sink leaves a file
beside its config.
