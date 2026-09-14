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
neither drops a fact. The validator is the gate, and because nothing is dropped
early, the ablation can count what grounding *would* have removed
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
from odke import Document, Entity, Evidence, Fact, GroundingVerdict, Span
from odke.ground import SpanGrounder, SpanStatus, check_span

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
  turn it off ([DECISIONS #7a](decisions.md)). `ModelRoles.single("ollama/…")`
  uses one local model for everything.
- `client` overrides how the model is reached. Pass any `LLMClient`: your
  gateway, or `ScriptedClient` / `RecordedClient` in tests.

For each fact the grounder runs the span check first, inside itself. Only a fact
the span check left `UNCHECKED` reaches the model, which is shown the first
located span's text and nothing more. Not the whole document: the question is
whether *this span* supports the claim, and showing more would answer a
different, dearer question.

```python
from odke.ground import LLMGrounder
from odke.llm import ModelRoles, ScriptedClient

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
stands and costs no call.

```python
chatty = LLMGrounder(
    ModelRoles.single("ollama/qwen2.5:3b"),
    client=ScriptedClient(["I would say this is not supported."]),
)
assert chatty.ground(born, doc).verdict is GroundingVerdict.UNCHECKED
assert chatty.stats["unparseable"] == 1
```

Transient errors are retried under a `RetryPolicy` (`odke.ground.RetryPolicy`):

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
- the span check's own counts, under `span`

### Batching

`ground_many(facts, doc)` grounds one document's facts with at most `max_workers`
model calls in flight, and `Pipeline` uses it automatically. Span checks run
first, in order; one fact comes back for each fact that went in, in the same
order; and a failed call leaves only its own fact `UNCHECKED`. The client must be
thread-safe. Both built-in clients are. `ScriptedClient` answers by position and
is not, which is what `RecordedClient` (answers by matching the prompt) is for.

## Grounding is a stamp; the validator is the gate

The pipeline drops only what a validator refuses. To keep ungrounded facts out of
the graph, pass a validator that refuses them:

```python
from odke import Ontology, Pipeline, ValidationVerdict


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
    validator=RefuseUngrounded(),
)
kg = pipeline.run([doc])

assert [(f.object_value, f.verdict) for f in kg.facts] == [(1815, GroundingVerdict.SUPPORTED)]
assert kg.stats["refused"] == 1
```

Whether grounding earns its cost on *your* corpus is a measurement, not a
promise: `odke.eval.grounding_ablation` compares precision with grounding off and
on against your labels. See [Evaluation](evaluation.md#ground).
