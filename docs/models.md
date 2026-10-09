# Models and providers

Every model call goes through one `LLMClient` protocol, and the provider prefix
on the model string decides which client makes it
([DECISIONS #7](decisions.md#7)). `odke models` lists every provider, its key
variable, whether that variable is set, and which client serves it. It needs no
key and no network.

## Model strings

A model string is `provider/model`. Only the first segment is the provider, and
a string with no prefix is `openai`.

```python
from openodke.llm import ModelSpec

assert ModelSpec(model="ollama/llama3.1").provider == "ollama"
assert ModelSpec(model="openrouter/anthropic/claude-sonnet-5").provider == "openrouter"
assert ModelSpec(model="gpt-5.5").provider == "openai"
```

## Which client serves a call

The first that matches:

1. a client registered for the provider with [`register`](#your-own-client);
2. the standard-library OpenAI-compatible client, when the provider has a
   default endpoint (`built-in` in `odke models`) or the spec sets `base_url`;
3. litellm, which needs `pip install "openodke[llm]"`.

```python
from openodke.llm import OpenAICompatClient, resolve

assert isinstance(resolve(ModelSpec(model="ollama/llama3.1")), OpenAICompatClient)
```

## Keys

Keys are read from the environment and nowhere else; openodke stores none.

| Provider | Model string | Key variable | Served by |
|---|---|---|---|
| `openai` | `openai/<model>` | `OPENAI_API_KEY` | built-in |
| `anthropic` | `anthropic/<model>` | `ANTHROPIC_API_KEY` | litellm |
| `azure` | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` | litellm |
| `ollama` | `ollama/<model>` | none | built-in |

`odke models` lists the rest.

- `api_key_env` names a different variable for one spec.
- A spec with `base_url` sends a key only when `api_key_env` names one, so an
  exported `OPENAI_API_KEY` never reaches a self-hosted server.
- A missing key raises `MissingAPIKey`, naming the variable, before any request.
  A missing key or provider stops a grounding run instead of leaving every fact
  unchecked.

```python
from openodke.llm import MissingAPIKey, require_key

try:
    require_key(ModelSpec(model="openai/gpt-5.5", api_key_env="OPENAI_API_KEY_EVAL"))
except MissingAPIKey as exc:
    assert "OPENAI_API_KEY_EVAL" in str(exc)
assert require_key(ModelSpec(model="openai/local", base_url="http://localhost:8000/v1")) == ""
```

## Roles and defaults

`ModelRoles` names a model per job. The defaults are pinned model ids, and
change only with a new calibration card ([DECISIONS #7a](decisions.md#7a)).

| Role | Default | Used by |
|---|---|---|
| `extract` | `anthropic/claude-sonnet-5-5` | `LLMExtractor` |
| `ground` | `anthropic/claude-haiku-4-5-20251001`, `max_tokens=256` | `LLMGrounder` |
| `infer` | the `extract` model | `odke ontology infer` |

Naming only `extract` grounds on that model too, at 256 output tokens.
`ModelRoles.single(model)` puts every role on one model. In `odke run`, the
`models` block or `--model` sets them ([`odke run`](run.md#models)).

```python
from openodke.llm import ModelRoles

roles = ModelRoles(extract=ModelSpec(model="ollama/llama3.1"))
assert (roles.ground.model, roles.ground.max_tokens) == ("ollama/llama3.1", 256)
```

## Ollama and other local servers

`ollama/…` calls `http://localhost:11434/v1` with no extra and no key. `vllm`,
`lmstudio` and `llamacpp` default to their usual local ports. Set `base_url` for
a server elsewhere, or for any endpoint that speaks the OpenAI
`/chat/completions` shape.

```yaml
models:
  extract: ollama/llama3.1
  ground: ollama/qwen2.5:3b
```

## Your own client

`register(provider, factory)` routes every `provider/…` string through your
client, ahead of both built-in paths ([DECISIONS #7b](decisions.md#7b)). The
client needs one method, `complete`, and must be thread-safe: the extractor and
the grounder each keep up to 8 calls in flight. `ScriptedClient`,
`RecordedClient` and `ReplayClient` in `openodke.llm.testing` answer without a
network, for tests.

```python
from openodke.llm import LLMClient, register, unregister
from openodke.llm.testing import ScriptedClient


def audited(spec: ModelSpec) -> LLMClient:
    return ScriptedClient(['{"entities": []}'])


register("acme", audited)
assert isinstance(resolve(ModelSpec(model="acme/internal-large")), ScriptedClient)
unregister("acme")
```

## Prompts

Every prompt openodke sends is registered in `openodke.prompts` as
`id@version`, with its text, SHA-256 and source, and every model call records
the key it sent ([DECISIONS #27](decisions.md#27)).

| Key | Sent by | Source |
|---|---|---|
| `ground.span@1` | `LLMGrounder`: one claim against its cited span | openodke |
| `ground.paper@1` | `LLMGrounder(verdicts="binary")`, [paper mode](grounding.md#paper-mode-the-odke-grounder-as-written) | ODKE+ App. B, verbatim |
| `extract@1` | `LLMExtractor`, followed by the ontology snippets | openodke |
| `extract.repair@1` | `LLMExtractor`, after a reply that broke the contract | openodke |
| `infer@1` | `LLMProposer` (`odke ontology infer`) | openodke |
| `infer.repair@1` | `LLMProposer`, after a reply that broke the contract | openodke |
| `reextract@1` | `LLMExtractor.reextract`, [the re-extract hook](grounding.md#handing-a-gap-back-the-re-extract-hook) | openodke, after GraphRAG's gleaning pass (Edge et al. 2024, arXiv 2404.16130), scoped to one window and grounded after |

```python
from openodke import prompts

span = prompts.get("ground.span@1")  # or get("ground.span", 1); no version is the latest
assert (span.id, span.version, span.source) == ("ground.span", 1, "openodke")
assert prompts.read_lock()[span.key] == span.sha256
```

The keys appear as `ModelCall.prompt`, `LLMExtractor.prompts`,
`InferenceCall.prompt` and `LLMGrounder.stats["prompts"]`, and on the extractor
and grounder lines of [`odke run`](run.md#what-a-run-reports).

**To change a prompt,** register the new text as the next version
(`ground.span@2`) and keep the old one, then add the new key's hash to
`src/openodke/prompts.lock.json`. The suite fails on an edited text with
`bump the version: <id>`, and on a prompt that shares twelve words in a row with
a benchmark gate split (`bench/labels/**/*gate*.jsonl`). The stages send the
latest version.

## The response cache

`CachedClient` wraps any client and answers a request it has seen before from a
store, so a rerun of an unchanged batch makes no call, costs nothing and needs
no key ([DECISIONS #30](decisions.md#30)).

- **The key** is the SHA-256 of one canonical JSON object: the
  provider-qualified model and its `base_url`; every message, role and content,
  in order; the response schema; `temperature`, `max_tokens` and `extra`; and
  the `id@version` of each [registered prompt](#prompts) the messages carry.
  `timeout` and `api_key_env` are not in it. Change anything else and the
  request misses.
- **An error is never stored**, so a failed call is made again next time. A reply
  the caller rejects is stored: the repair turn after a malformed extraction has
  its own messages, so its own key, and a rerun replays both.
- **A hit** comes back with `cached=True`, no tokens and a cost of `0.0`. The
  cost meter records it as a call with `cached: true`, and counts them as
  `cached_calls`; the grounder's stats count them under `cached`, the extractor's
  under `cached_calls`.
- **`DirectoryCache(path)`** keeps one JSON file per key, at
  `<path>/<first two characters>/<key>.json`. Each is written to a temporary
  file beside it and renamed into place, so the thread pools, and several
  processes, can share one directory without a torn entry. A file that does
  not parse is a miss. **`MemoryCache`**, the default, lasts one process.
- **Nothing expires.** A sample drawn at a temperature above zero replays as it
  was drawn. Delete the directory, or point at a new one, to ask again.

```python
from openodke.llm import CachedClient, Message

model = ScriptedClient(['{"verdict": "supported"}'])  # answers once, then fails
cached = CachedClient(model)  # CachedClient(model, ".odke-cache") keeps it on disk
ask = [Message(content="Claim: … Passage: …")]
first = cached.complete(ask, spec=ModelSpec(model="ollama/llama3.1"))
again = cached.complete(ask, spec=ModelSpec(model="ollama/llama3.1"))
assert (again.text, again.cached, again.cost_usd) == (first.text, True, 0.0)
assert cached.stats == {"hits": 1, "misses": 1, "failed": 0}
```

In `odke run`, `models: {cache: .odke-cache}` names the directory, relative to
the config. `odke validate` and `odke ground` read the same key from `--config`,
and all three take `--cache DIR`, which overrides it. The cache answers in front
of `models.replay` and the provider alike, and a rerun answered entirely from it
needs no provider adapter and no key. The run report gains a `cache` line, hits
and misses, and `stats["cache"]`.

## Concurrency per provider

The extractor and the grounder each keep up to `max_workers` calls in flight,
often against one provider account. `models: {limits: {anthropic: 8}}` caps
the calls in flight to each provider it names, across every stage and every
run in the process; `odke validate` and `odke ground` read the same key from
`--config`.

- **The key is the provider**, the model string's prefix, not the model: a
  provider counts its rate limit per account, so an `anthropic/…` extractor
  and an `anthropic/…` grounder share one limit. A provider with no limit is
  not held back.
- **The stricter wins.** A stage's own `max_workers` still applies, so a
  grounder with 4 workers under a limit of 8 keeps 4 in flight.
- **A `Retry-After` pauses the provider.** When a call fails with one, every
  call to that provider waits it out, not only the one the retry policy will
  repeat, capped at 30 seconds. The header is read as the
  [retry policy](grounding.md) reads it.

From Python, `set_limit(provider, n)` sets the process's limit (`None` lifts
it), and `LimitedClient(client)` holds a client's calls to it. `odke run` wraps
every model client this way.

```python
from openodke.llm import LimitedClient, ProviderLimits, set_limit

limits = ProviderLimits()  # set_limit("ollama", 2) sets the process's own
limits.set("ollama", 2)
held = LimitedClient(ScriptedClient(['{"verdict": "supported"}']), limits)
assert held.complete(ask, spec=ModelSpec(model="ollama/llama3.1")).text
assert limits.limits == {"ollama": 2}
```
