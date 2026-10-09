# Grounding

Grounding asks whether the cited span supports the claim. It asks two questions,
in this order, and the order is the point:

1. **Does the span exist, and does it say what the fact claims it says?** This is
   `SpanGrounder`. It is pure and free, and it always runs first. An offset that
   does not resolve to its quote is rejected before a token is spent.
2. **Does that text actually support the claim?** This is `LLMGrounder`, a second,
   small model reading one claim against one passage. Locatability alone cannot
   answer this, because a model can cite a span that genuinely exists and does not
   support the fact it was attached to.

Both **stamp** `Fact.verdict` (`supported`, `contradicted` or `not_found`), and
neither drops a fact. The gate decides what is written, and because nothing is
dropped early, the ablation can count what grounding *would* have removed
([DECISIONS #20](decisions.md)).

## The free check: `SpanGrounder`

`check_span(evidence, doc)` classifies one piece of evidence as a `SpanStatus`:

| Status | Meaning | Rejected? |
|---|---|---|
| `located` | the span lies in this document and resolves to its quote (or has no quote) | no |
| `no_span` | a document-level citation: nothing to check | no |
| `foreign` | a span into another document | no |
| `out_of_range` | offsets outside the text, or `start > end` | yes |
| `empty` | `start == end` | yes |
| `quote_mismatch` | the text at those offsets is not the quote | yes |

Range is checked before the quote. Python slicing truncates silently, so a span
running past the end of the text could otherwise "resolve" to its quote while
citing offsets that do not exist.

For each fact, `SpanGrounder`:

- drops every rejected span, logs the reason at DEBUG and counts it. An offset
  that does not say what it claims is not provenance.
- keeps span-less and foreign citations as they are, since nothing showed them
  wrong. They do not locate the claim in *this* document, though.
- stamps `NOT_FOUND` when no evidence is `located`, and otherwise leaves the fact
  `UNCHECKED` for the model. Only a model can say `SUPPORTED`.
- leaves any verdict already on the fact in place, which is what lets an
  interrupted run resume.

```python
from openodke import Document, Entity, Evidence, Fact, GroundingVerdict, Span
from openodke.ground import SpanGrounder, SpanStatus, check_span

doc = Document(id="d1", text="Ada Lovelace was born in London in 1815.")
ada = Entity(key="p:ada", type="Person", label="Ada Lovelace")


def cites(start, end, quote=None):
    return Evidence(doc_id="d1", span=Span(doc_id="d1", start=start, end=end, quote=quote))


born = Fact(subject=ada, predicate="born", object_value=1815, evidence=(cites(0, 40),))
invented = Fact(subject=ada, predicate="born", object_value=1816, evidence=(cites(35, 39, "1816"),))

assert check_span(cites(35, 39, "1815"), doc) is SpanStatus.LOCATED
assert check_span(cites(35, 39, "1816"), doc) is SpanStatus.QUOTE_MISMATCH
assert check_span(cites(35, 90), doc) is SpanStatus.OUT_OF_RANGE
assert check_span(Evidence(doc_id="d1"), doc) is SpanStatus.NO_SPAN

spans = SpanGrounder()
assert spans.ground(born, doc).verdict is GroundingVerdict.UNCHECKED  # located: a model decides
checked = spans.ground(invented, doc)
assert checked.verdict is GroundingVerdict.NOT_FOUND and checked.evidence == ()
assert spans.stats["located"] == spans.stats["quote_mismatch"] == spans.stats["not_found"] == 1
```

A high rejection rate here points at the prompt, not the model.

## The second model: `LLMGrounder`

```text
LLMGrounder(roles=None, *, client=None, max_workers=8, retry=None)
```

- `roles.ground` names the model. The default `ModelRoles()` grounds with
  `anthropic/claude-haiku-4-5-20251001` at `max_tokens=256`, a smaller model than
  extraction, on purpose. If this pass cost what extraction costs, people would
  turn it off ([DECISIONS #7a](decisions.md)). Which provider that string names,
  which key it reads and who serves it are one page:
  [Models and providers](models.md), or `odke models`.
- `client` overrides how the model is reached. Pass any `LLMClient`: your
  gateway, or `ScriptedClient` / `RecordedClient` in tests.

For each fact the grounder runs the span check first, inside itself. Only a fact
the span check left `UNCHECKED` reaches the model, which is shown the first
located span's text and nothing more. Not the whole document: the question is
whether *this span* supports the claim, and showing more would answer a
different, dearer question.

```python
from openodke.ground import LLMGrounder
from openodke.llm import ModelRoles, ScriptedClient

client = ScriptedClient([{"verdict": "supported"}])
grounder = LLMGrounder(ModelRoles.single("ollama/qwen2.5:3b"), client=client)

assert grounder.ground(invented, doc).verdict is GroundingVerdict.NOT_FOUND
assert client.calls == []  # settled by the free check: no call made

assert grounder.ground(born, doc).verdict is GroundingVerdict.SUPPORTED
messages, spec, schema = client.calls[0]
print(messages[1].content)
# Claim: Ada Lovelace (Person) — born — 1815.
#
# Passage:
# Ada Lovelace was born in London in 1815.
assert schema["properties"]["verdict"]["enum"] == ["supported", "contradicted", "not_found"]
```

The claim is rendered the same way on every run, so provider-side prompt caching
works. Only identity-bearing qualifiers appear in it. Polarity is spelled out
("It is NOT the case that: …"), because "X sells data" and "X does not sell data"
are the same triple and opposite claims.

### Reading the answer

The verdict is read from structured output. A model that ignored the schema and
answered with the bare word (`supported`, `not found`) is still readable. One
that wrote a sentence is not: "not supported" must never be read as `SUPPORTED`,
so prose counts as a failed answer rather than a guess.

### When a call fails

`ground` never raises for a provider failure, because one bad call must not lose
a run of ten thousand. A failed or unreadable call leaves the fact `UNCHECKED`
and logs a warning. The next run picks it up, since a verdict already on a fact
stands and costs no call. A missing key (`MissingAPIKey`) or adapter
(`ProviderNotInstalled`) does raise: it is configuration, it would fail every
call alike, and the run would otherwise write a graph nothing had checked.

```python
chatty = LLMGrounder(
    ModelRoles.single("ollama/qwen2.5:3b"),
    client=ScriptedClient(["I would say this is not supported."]),
)
assert chatty.ground(born, doc).verdict is GroundingVerdict.UNCHECKED
assert chatty.stats["unparseable"] == 1
```

Transient errors are retried under a `RetryPolicy` (`openodke.ground.RetryPolicy`):

- 4 attempts in total by default.
- Exponential backoff from 0.5 s, multiplier 2, capped at 30 s, with full jitter,
  so threads that hit one rate limit together do not all come back together.
- A server's `Retry-After` wins when it sends one, still capped at `max_delay`.
- Transient means an HTTP 408, 409, 425, 429, 500, 502, 503, 504 or 529, a timeout
  or connection error, or a message that says so. A 401 or a missing adapter is
  not transient, and is not retried.

### Stats

`grounder.stats` is the stage's run report:

- `facts`, `skipped`, `calls`, `retries`, `failed`, `unparseable`
- a count for each verdict
- `prompt_tokens` and `completion_tokens`
- `cost_usd`, which stays `None` until a provider reports a cost, rather than a
  misleading `0.0`
- `prompts`, the key of the [registered prompt](models.md#prompts) the calls
  sent, such as `ground.span@1`; empty when no call was made
- the span check's own counts, under `span`

### Batching

`ground_many(facts, doc)` grounds one document's facts with at most `max_workers`
model calls in flight. `ground_documents([(facts, doc), ...])` does the same for
many documents with one ceiling across them all, and `Pipeline` uses it
automatically, so a corpus of one-row documents runs as concurrently as one long
document. Span checks run first, in order; one fact comes back for each fact that
went in, in the same order; and a failed call leaves only its own fact
`UNCHECKED`. The client must be thread-safe. Both built-in clients are.
`ScriptedClient` answers by position and is not, which is what `RecordedClient`
(answers by matching the prompt) is for.

### Paper mode: the ODKE+ grounder as written

openodke's default grounder differs from the paper's on purpose: it is shown the
cited span, answers in three ways, and drops nothing. The paper's (ODKE+ §3.3.1,
App. B) is shown the whole context, answers True or False, and only affirmed
facts are kept. Two options and a gate setting reproduce it, so the paper's
claims can be tested as written (see [Benchmarks](benchmarks.md)):

```yaml
stages:
  grounder: {use: llm, context: document, verdicts: binary}
  gate: {use: verdict, refuse_not_found: true}
```

`verdicts: binary` sends the paper's own prompt and its
`<subject, predicate(qualifier: value), object>` triple. "True" is read as
`supported` and "False" as `not_found`: the paper's "No" merges a context that
contradicts the triple with one that is silent about it, so a binary run cannot
tell them apart. The free span check still runs first, so a fact whose quote is
not in the document is settled before any call — the one step of openodke's
that paper mode keeps.

## Locating spans

Most extractors outside openodke cite nothing, so their facts carry the whole
text they came from as a `context` span ([Triples](triples.md)), and the model
reads all of it. `LLMGrounder(locate=True)` first looks for the support for
free: `SpanLocator` finds the narrowest window of one sentence, or two adjacent
ones in a paragraph, that names both the subject and the object. The window
becomes the fact's span, marked `SpanOrigin.LOCATED`, and the model reads it
instead. With no window nothing changes. A citation is never touched, and a
quote that is not in the text is still refused by the free check.

A wrong window is worse than none: shown a window that names both and does not
state the fact, the model refuses a fact the whole text supports. So the match
is strict:

- a name is the label or an alias, as written or as its name key (`name_key`),
  in whole words, with case, accents and punctuation ignored. `Ann` is not in
  `Annapolis`;
- a capitalised name is found only capitalised, and never inside a longer one:
  `US` is not `us`, and `Africa` is not in `South Africa`;
- a literal object is found by its words or, for a date or a quantity, its
  normalised value: `July 15, 1895` is found as `15 July 1895`;
- the object must be named apart from the subject's own name. One sentence beats
  two, then the narrower window wins, then the earlier.

```python
from openodke import SpanOrigin
from openodke.ground import locate_span
from openodke.llm import RecordedClient

text = "Halden Robotics was founded in Leeds in 2014. It opened a second office in Lyon in 2019."
halden = Document(id="halden", text=text)
company = Entity(key="Company:halden robotics", type="Company", label="Halden Robotics")
whole = Evidence(  # what a triple with no citation carries
    doc_id="halden", span=Span(doc_id="halden", start=0, end=len(text)), span_origin="context"
)


def office_in(city):
    city = Entity(key=f"City:{city.lower()}", type="City", label=city)
    return Fact(subject=company, predicate="office_in", object_entity=city, evidence=(whole,))


lyon, berlin = office_in("Lyon"), office_in("Berlin")
print(locate_span(lyon, halden).resolve(halden))
# Halden Robotics was founded in Leeds in 2014. It opened a second office in Lyon in 2019.
assert locate_span(berlin, halden) is None  # never named: the model reads the whole text

client = RecordedClient([{"match": "— Lyon (City)", "response": {"verdict": "supported"}}])
locating = LLMGrounder(ModelRoles.single("ollama/qwen2.5:3b"), client=client, locate=True)
grounded = locating.ground(lyon, halden)
assert grounded.evidence[0].span_origin is SpanOrigin.LOCATED
assert locating.stats["locate"] == {"facts": 1, "located": 1, "not_located": 0}
```

In `odke run` it is `grounder: {use: llm, locate: true}`. With
`context: document` the model still reads the whole text, and the window is kept
for a reader. `odke eval spans` counts located spans apart from citations.

It is off by default, because its verdicts differ from whole-document grounding
by more than the bench's noise. On the published Re-DocRED runs (PR #105) it
placed 75% of LLMGraphTransformer's facts and 72% of neo4j-graphrag's, in
windows with a median of 169 characters against documents of 932–984, and cut
grounding cost by 38% a fact. Precision against gold and the true facts kept did
not move. But its verdicts matched the whole-document ones on 93% and 92% of
those facts, where a second whole-document run matched on 99% and 97%.
`bench/locator.py` is the measurement, and states the rule.

## Grounding is a stamp; the gate decides

The pipeline drops only what its gate refuses. To keep ungrounded facts out of
the graph, pass a gate that refuses them:

```python
from openodke import Ontology, Pipeline, ValidationVerdict


class RefuseUngrounded:
    def validate(self, fact, ontology):
        if fact.verdict in (GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND):
            return ValidationVerdict(action="refuse", reason=f"grounding: {fact.verdict.value}")
        return ValidationVerdict(action="accept")


class Replay:
    """Stands in for an extractor: the two candidate facts above."""

    def extract(self, chunk, ontology):
        return [born, invented]


pipeline = Pipeline(
    Ontology(),
    Replay(),
    grounder=LLMGrounder(
        ModelRoles.single("ollama/qwen2.5:3b"),
        client=ScriptedClient([{"verdict": "supported"}]),
    ),
    gate=RefuseUngrounded(),
)
kg = pipeline.run([doc])

assert [(f.object_value, f.verdict) for f in kg.facts] == [(1815, GroundingVerdict.SUPPORTED)]
assert kg.stats["refused"] == 1
```

Whether grounding earns its cost on *your* corpus is a measurement, not a
promise: `openodke.eval.grounding_ablation` compares precision with grounding off and
on against your labels. See [Evaluation](evaluation.md#ground).

## When `not_found` means the citation, not the fact

A true fact cited to a bare mention — `Ireland` rather than the clause that says
where the company operates — is correctly refused: the grounder is shown the
span and nothing else, and that span does not support the claim. So a run with
many `not_found` verdicts may have a citation problem rather than a fact
problem, and the two are told apart by width: citations too narrow to carry
their claim cluster in `not_found`, clause-width ones in `supported`.

[`odke eval spans`](evaluation.md#spans) reports that split. It needs no
labelled data, which makes it the cheapest check on a new extractor or prompt.

## What extraction left behind: the coverage report

Grounding judges the facts an extractor returned; it cannot see the ones it
did not. A validator cannot invent those ([DECISIONS #24](decisions.md)), but it
can count where they are missing, with no model. `odke run` does by default
(`coverage: false` turns it off), `Pipeline(coverage=True)` does in Python, and
`openodke.coverage.measure(documents, facts, ontology)` does on any facts. Per
document:

- **Entities in no fact.** A known name is a label or alias of any subject or
  edge object in the batch. One the text mentions that no fact of that document
  names is reported, at its first mention.
- **Sentences no fact covers.** A sentence naming two or more known entities is
  one a fact could have come from. It is covered when any fact's evidence span
  overlaps it; the rest are reported with their text. A sentence naming fewer is
  not counted, because most of those are narrative. A span nobody chose
  (`context`) overlaps every sentence, so it covers only the sentences naming
  both ends of its fact. A chunk the router skipped is not a gap.
- **Relations never offered.** Predicates the extractor was not shown, from its
  `offered(ontology)` (`LLMExtractor`: its snippets, so a type left out of
  `types` or a predicate past `snippet_limit`), and the shown predicates no fact
  used. An extractor without `offered` reports the first as unknown.

Names match as [`name_key`](resolution-and-corroboration.md) token runs, longest
first: case, accents, a dotted initialism, a leading "the" and a legal suffix do
not matter. It is a small matcher until the span locator's (#112) lands. `odke
run` prints one line, and `odke eval ablation` adds the same line to its notes:

```
coverage      1 of 3 sentences naming two known entities uncovered, 1 entities in no fact, 2 relations never offered, 0 unused
```

These are gaps, not errors: a sentence naming two entities may relate them in
no way the ontology has. They are what the re-extract hook (#102) hands back.

```python
from openodke import Document, Entity, Evidence, Fact, Ontology, Span
from openodke.coverage import measure

text = "Ada Lovelace worked with Babbage. She wrote to Babbage for years. Ada Lovelace met Babbage in 1834."
doc = Document(id="d1", text=text)
cited = Span(doc_id="d1", start=0, end=33, quote="Ada Lovelace worked with Babbage.")
fact = Fact(
    subject=Entity(key="p:ada", type="Person", label="Ada Lovelace"),
    predicate="collaborator",
    object_entity=Entity(key="p:babbage", type="Person", label="Babbage"),
    evidence=(Evidence(doc_id="d1", span=cited),),
)
(record,) = measure([doc], [fact], Ontology()).documents
assert record.sentences == 2  # "She wrote to Babbage" names one known entity
assert [s.quote for s in record.uncovered] == ["Ada Lovelace met Babbage in 1834."]
```
