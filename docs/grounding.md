# Grounding

Grounding checks each fact against its text: a free span check, then a small
second model. Both stamp a [verdict](concepts.md#verdicts) and drop nothing;
[the gate](#the-gate-verdictgate) decides what is written
([DECISIONS #20](decisions.md#20)). [The Validator](validator.md) runs it with
every other stage.

## A graph from somewhere else: `odke ground`

On `examples/triples/`, five triples with the model's answers recorded:

```bash
odke ground --facts examples/triples/triples.jsonl --texts examples/triples/texts \
  --ontology examples/triples/ontology.json --config examples/triples/odke.yaml -o out
```

```text
odke ground
rows          5 (5 grounded)
verdicts      supported 3, contradicted 0, not_found 2, unchecked 0
free checks   1 refused: 1 not in the text
no span       3 of 5 cite no span of their own
model         4 calls, 657 tokens, USD unknown (ground.span@1)
failures
  the evidence does not support the fact: 1
    Halden Robotics —office_in→ Berlin  [not_found, examples/triples/texts/halden.txt]
  the citation is too narrow for its claim: 0

spans  (n=5)
  …
wrote out/facts.jsonl, out/summary.json, out/manifest.json
```

Each fact passes the [free checks](#the-free-check-spangrounder), the span
locator (`--locate`) and the model (unless `--dry-run`), stopping where it is
settled. `--ontology` adds checks on the predicate and types, recorded under
`qualifiers["odke.check"]`.

The summary splits `not_found` by where the fix belongs. **The evidence does
not support the fact**: Berlin is never mentioned. **The citation is too narrow
for its claim**: the span does not name both ends, so the fact may be true
([Evidence](concepts.md#spans-evidence-and-trust)); `--paper` counts none.

It writes the facts with their verdicts (`facts.jsonl`, for `odke eval spans`),
the summary (`summary.json`) and the [run manifest](run.md#the-run-manifest).

| Option | |
|---|---|
| `--facts` | A triples file, an adapter's output, or a Neo4j URI |
| `--adapter` | `triples` (default), `langchain`, `langextract`, `graphrag` or `neo4j` ([Inputs](inputs.md#from-other-libraries)) |
| `--texts` | The cited texts, a file or a directory: for triples, and a Neo4j graph openodke wrote |
| `--ontology` | Adds the free checks on the relation and the types |
| `--config`, `--model`, `--model-provider` | The model: a file's `models` block, `replay` included, as in `odke run` |
| `--locate`, `--paper`, `--dry-run` | The [span locator](#locating-spans); [paper mode](#paper-mode-the-odke-grounder-as-written); no model |
| `--text-property` | `neo4j`: the relationship property holding each text |
| `--user`, `--password-env`, `--database` | Neo4j; the password's variable defaults to `NEO4J_PASSWORD` |
| `--write-verdicts` | `neo4j`: set `odke_verdict` on each relationship read |
| `--cache`, `--budget-usd`, `--budget-calls`, `--log-format`, `--log-text` | As [`odke run`](run.md) takes them |
| `-o` | The directory to write into |

Exit 2 is unreadable input, exit 1 a failed run, exit 3 a run its budget
stopped, which still writes what it grounded. In Python it is `ground_graph`,
over any adapter's rows and texts:

```python
from openodke import Document
from openodke.ground import LLMGrounder
from openodke.interop import ground_graph
from openodke.llm import RecordedClient

passage = "Halden Robotics opened its second office in Lyon in 2019."
claim = {"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in"}
rows = [{**claim, "object": "Lyon", "quote": "Lyon"}, {**claim, "object": "Berlin"}]
client = RecordedClient([{"match": "Claim:", "response": {"verdict": "not_found"}}])
grounded = ground_graph(
    rows, [Document(id="halden", text=passage)], grounder=LLMGrounder(client=client)
)
print(grounded.summary.too_narrow.examples[0])
# Halden Robotics —office_in→ Lyon  [not_found, halden]
print(grounded.summary.unsupported.examples[0])
# Halden Robotics —office_in→ Berlin  [not_found, halden]
```

`grounder=None` is the dry run. `grounded.write("out")` writes `facts.jsonl` and
`summary.json`.

## The free check: `SpanGrounder`

`check_span(evidence, doc)` classifies one piece of evidence as a `SpanStatus`:

| Status | Meaning | Rejected? |
|---|---|---|
| `located` | resolves to its quote, or has none | no |
| `no_span` | a document-level citation | no |
| `foreign` | a span into another document | no |
| `out_of_range` | outside the text, or `start > end` | yes |
| `empty` | `start == end` | yes |
| `quote_mismatch` | the text there is not the quote | yes |

`SpanGrounder` drops and counts each rejected span, and stamps `not_found` when
no evidence is `located`; otherwise the fact stays `unchecked` for the model. A
verdict already on a fact stands, so an interrupted run resumes.

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

## The second model: `LLMGrounder`

```text
LLMGrounder(roles=None, *, client=None, max_workers=8, retry=None, context="span",
            verdicts="three_way", locate=False, widen=False, widen_client=None)
```

| Argument | |
|---|---|
| `roles` | `roles.ground` is the model ([Roles and defaults](models.md#roles-and-defaults)) |
| `client` | Any `LLMClient`; `ScriptedClient` or `RecordedClient` in tests |
| `max_workers` | Model calls in flight |
| `retry` | A [`RetryPolicy`](#retries) |
| `context` | `"document"` reads the whole document, not the span |
| `verdicts` | `"binary"` is [paper mode](#paper-mode-the-odke-grounder-as-written)'s True/False |
| `locate`, `widen` | [Locating spans](#locating-spans), [widen and retry](#widen-and-retry) |
| `widen_client` | The client widen's retries go through, so a meter counts them apart |

The span check runs first, inside it; only a fact it leaves `unchecked` reaches
the model, which reads the first located span.

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

The verdict is read from structured output or a bare word; a sentence is a
failed answer. A failed or unreadable call leaves the fact `unchecked`, for the
next run to ask. A missing key (`MissingAPIKey`) or adapter
(`ProviderNotInstalled`) raises.

```python
chatty = LLMGrounder(
    ModelRoles.single("ollama/qwen2.5:3b"),
    client=ScriptedClient(["I would say this is not supported."]),
)
assert chatty.ground(born, doc).verdict is GroundingVerdict.UNCHECKED
assert chatty.stats["unparseable"] == 1
```

`grounder.stats` counts `facts`, `skipped`, `calls`, `retries`, `failed`,
`unparseable`, each verdict, tokens, `cached` calls and `unasked` facts, with
`cost_usd` (`None` until priced), the [`prompts`](models.md#prompts) sent, and
the `span`, `locate` and `widen` steps' counts.

`ground_many(facts, doc)` and `ground_documents([(facts, doc), ...])`, which
`Pipeline` uses, keep at most `max_workers` calls in flight and return facts in
order. The client must be thread-safe; `ScriptedClient` is not.

### Retries

`RetryPolicy` (`openodke.ground.RetryPolicy`) retries HTTP 408, 409, 425, 429,
500, 502, 503, 504 and 529, timeouts and connection errors.

| Field | Default |
|---|---|
| `attempts` | `6`, in total |
| `base_delay` | `1` s; with `multiplier`, waits of up to 1, 2, 4, 8 and 16 s: 31 s at most, about 15 s on average with jitter |
| `multiplier` | `2` |
| `max_delay` | `30` s; a server's `Retry-After` wins, capped here |
| `jitter` | `true`: full jitter |

### The challengers: `prompt=`

Two other prompts compete with `ground.span@1` for the default (#134), both from
EnterpriseRAG-Bench (Onyx, arXiv 2605.05253, MIT): `ground.onyx@1`, its fact
validator word for word, and `ground.statement@1`, the same question answered
three ways in JSON. Both read the fact as one plain sentence (`render_sentence`:
`Ada Lovelace born in London.`; a denial reads "It is not the case that …").
Onyx's yes or no is read as Onyx reads it, from the first line: "yes" is
`supported`, "no" is `not_found`, and anything else is unreadable, leaving the
fact `unchecked`. A [calibration card](calibration.md) on label set G decides
which is the default; until then it is `ground.span@1`.

```python
onyx = LLMGrounder(
    ModelRoles.single("ollama/qwen2.5:3b"), client=ScriptedClient(["yes"]), prompt="ground.onyx"
)
assert onyx.ground(born, doc).verdict is GroundingVerdict.SUPPORTED
assert onyx.stats["prompts"] == ["ground.onyx@1"] and onyx.claim_line == "sentence"
```

### Paper mode: the ODKE+ grounder as written

The paper's grounder (ODKE+ §3.3.1, App. B) reads the whole context, answers
True or False, and keeps only affirmed facts ([Benchmarks](benchmarks.md)):

```yaml
stages:
  grounder: {use: llm, context: document, verdicts: binary}
  gate: {use: verdict, refuse_not_found: true}
```

`verdicts: binary` sends the paper's prompt and reads "True" as `supported`,
"False" as `not_found`. The span check still runs first. `odke ground --paper`
sets both grounder options.

### Widen and retry

Off by default ([DECISIONS #23](decisions.md#23)): `widen=True`,
`grounder: {use: llm, widen: true}` or `odke run --widen`. A `not_found` fact
whose span is narrower than its sentence (`openodke.chunking.sentences`) is
asked once more against the sentence, unless the sentence adds only
punctuation. On `supported` the sentence becomes its span and the narrow
citation its `Evidence.mention`; otherwise nothing changes. Either way
`qualifiers["odke.widen"]` records `{"from": [start, end], "to": [start, end],
"verdict": ...}`, the way to tell a widened span from a cited one.
`widen=True` with `context="document"` is refused. `stats["widen"]` counts
`retried`, `recovered` and the retries' own calls, tokens and cost, which `odke
run` meters as `ground.widen`.

```python
from openodke.ground import LLMGrounder
from openodke.llm import ScriptedClient

text = "Acme Cloud runs data centres in Ireland, Singapore and Virginia."
cloud = Document(id="c", text=text)
word = Span(doc_id="c", start=32, end=39, quote="Ireland")
operates = Fact(
    subject=Entity(key="p:acme", type="Provider", label="Acme Cloud"),
    predicate="operates_in",
    object_value="Ireland",
    evidence=(Evidence(doc_id="c", span=word),),
)
answers = ScriptedClient([{"verdict": "not_found"}, {"verdict": "supported"}])
grounder = LLMGrounder(ModelRoles.single("ollama/qwen2.5:3b"), client=answers, widen=True)
recovered = grounder.ground(operates, cloud)

assert recovered.verdict == GroundingVerdict.SUPPORTED
assert recovered.evidence[0].span.quote == text and recovered.evidence[0].mention == word
assert recovered.qualifiers["odke.widen"]["verdict"] == "supported"
assert (grounder.stats["widen"]["retried"], grounder.stats["widen"]["recovered"]) == (1, 1)
```

## Locating spans

A fact with no citation carries its whole text as a `context` span
([Inputs](inputs.md#offsets-a-quote-or-neither)). `LLMGrounder(locate=True)`
first runs `SpanLocator`, free: the narrowest window of one sentence, or two
adjacent ones in a paragraph, naming both the subject and the object becomes
the span, marked `located`, for the model to read. A citation is never touched.
The match is strict:

- a name is a label or alias, as written or as its `name_key`, in whole words,
  ignoring case, accents and punctuation: `Ann` is not in `Annapolis`;
- a capitalised name matches only capitalised, never inside a longer one:
  `Africa` is not in `South Africa`;
- a literal object matches its words or, for a date or quantity, its
  normalised value: `July 15, 1895` matches `15 July 1895`;
- the object must be named apart from the subject. One sentence beats two,
  then the narrower window, then the earlier.

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

In a run config it is `grounder: {use: llm, locate: true}`. With
`context: document` the model still reads the whole text. It is off by default:
its verdicts differed from whole-document grounding by more than the bench's
noise (`bench/locator.py` states the rule and measures it).

## The gate: `VerdictGate`

`VerdictGate` (`gate: verdict`) decides what is written:

| Verdict | Default | `refuse_not_found=True` |
|---|---|---|
| `contradicted` | refused, or marked below the bar | refused, or marked below the bar |
| `not_found` | accepted | refused, or marked below the bar |
| `supported`, `unchecked` | accepted | accepted |

`schema=True`, the Validator's default, also refuses an undeclared predicate
or a type outside the domain or range. `stats` counts accepted and refused
facts, by reason. Whether to refuse `not_found` on your corpus is a
measurement: [`odke eval ablation`](evaluation.md#ablation).

A refusal on a grounding verdict is the grounder's word, and below its
calibration bar the gate **marks** instead of dropping: the fact is written
with `qualifiers["odke.review"]` saying why. `on_refusal="drop"` or `"mark"`
chooses; left out, the grounder's [calibration card](calibration.md#below-the-bar-nothing-is-dropped-on-a-judges-word)
decides ([DECISIONS #54](decisions.md#54)), and every card measured so far is
below its bar. A schema refusal is always made. The example below drops, to
show a refusal.

```python
from openodke import Ontology, Pipeline, VerdictGate

gate = VerdictGate()
verdicts = [GroundingVerdict.SUPPORTED, GroundingVerdict.NOT_FOUND, GroundingVerdict.CONTRADICTED]
stamped = [born.model_copy(update={"verdict": v}) for v in verdicts]
assert [gate.validate(f, Ontology()).action for f in stamped] == ["accept", "accept", "refuse"]
print(gate.stats)
# {'accepted': 2, 'refused': {'contradicted': 1}}


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
    gate=VerdictGate(refuse_not_found=True, on_refusal="drop"),
)
kg = pipeline.run([doc])

assert [(f.object_value, f.verdict) for f in kg.facts] == [(1815, GroundingVerdict.SUPPORTED)]
assert kg.stats["refused"] == 1
```

## What extraction left behind: the coverage report

The coverage report counts, with no model, where an extractor left facts
behind ([DECISIONS #24](decisions.md#24)). `odke run` and the Validator run it
by default (`coverage: false`, `coverage=False`); in Python,
`Pipeline(coverage=True)` or `openodke.coverage.measure(documents, facts,
ontology)`. Per document:

- **Entities in no fact**: a known name (a label or alias of any subject or
  edge object in the batch) the text mentions and no fact of the document
  names, at its first mention.
- **Sentences no fact covers**: a sentence naming two or more known entities
  that no evidence span overlaps. A `context` span covers only its
  [located](#locating-spans) window; a chunk the router skipped is no gap.
- **Relations never offered**: predicates the extractor was not shown, from its
  `offered(ontology)` (`LLMExtractor`: a type left out of `types`, a predicate
  past `snippet_limit`), and shown predicates no fact used. Without `offered`,
  the first is unknown.

Names match by [the locator's rules](#locating-spans). `odke run` and
`odke eval ablation` print one line:

```
coverage      1 of 3 sentences naming two known entities uncovered, 1 entities in no fact, 2 relations never offered, 0 unused
```


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

## Handing a gap back: the re-extract hook

Provisional ([Stability](stability.md#provisional)), off by default:
`Pipeline(reextract=Reextract())`, or `reextract: true` (or `{windows: N}`) in a
run config. It hands the coverage report's gaps back to the extractor.

- **Windows**: uncovered sentences, then ones holding a missed entity, at most
  `windows` per document (default 3), each once per run.
- **The hook**: `reextract(window: Chunk, relations: list[str], already:
  list[Fact], ontology) -> list[Fact]`. `relations` are the predicates that fit
  the entities the window names; `already` are the facts cited in it or naming
  them. `Reextract(hook=...)` names a hook; left out, it is the pipeline's
  extractor. `LLMExtractor` sends [`reextract@1`](models.md#prompts), and
  `HybridExtractor` passes it to its model path.
- **What returns** is dropped if the document has its signature; the rest carry
  `qualifiers["odke.reextract"] = {"window": [start, end]}` and are grounded.
- `stats["reextract"]` counts `windows`, `returned`, `duplicates`, `kept`, and
  `refused` by grounding, with the `verdicts`.

```
reextract     1 windows asked, 1 facts returned (0 already held), 1 kept and 0 refused by grounding
```

```python
from openodke import Pipeline
from openodke.reextract import Reextract


class Gleaner:
    """Finds the first sentence's fact, and the gap's when asked again."""

    def extract(self, chunk, ontology):
        return [fact]

    def reextract(self, window, relations, already, ontology):
        start = text.index(window.text)
        span = Span(doc_id="d1", start=start, end=start + len(window.text), quote=window.text)
        cited = (Evidence(doc_id="d1", span=span),)
        return [fact.model_copy(update={"predicate": "met", "evidence": cited})]


people = Ontology.from_dict(
    {
        "types": {"Person": {}},
        "predicates": {
            name: {"domain": ["Person"], "range": "Person"} for name in ("collaborator", "met")
        },
    }
)
kg = Pipeline(people, Gleaner(), reextract=Reextract()).run([doc])
assert kg.stats["reextract"]["windows"] == kg.stats["reextract"]["returned"] == 1
assert [f.predicate for f in kg.facts] == ["collaborator", "met"]
```
