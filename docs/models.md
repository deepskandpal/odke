# Models and providers

Every model call in openodke goes through one `LLMClient` protocol, and which
client serves a call is decided by the *provider prefix* on the model string.
Nothing else in the library imports a provider SDK or names a vendor
([DECISIONS #7](decisions.md)).

Picking a provider is therefore two decisions and no more: the model string, and
the environment variable that holds the key.

```bash
odke models
```

prints the whole table — every provider, the form its model string takes, the
variable it reads, whether that variable is set *here*, and which client makes
the call. It runs on the base install, with no key and no network, so it is the
right first command when a run cannot reach a model.

Values are never printed, not even masked, and openodke writes no key anywhere.
The environment is the only place it reads one from.

## Provider-qualified names

A model string is `provider/model`:

```python
from openodke.llm import ModelRoles, ModelSpec

ModelSpec(model="openai/gpt-5.5")
ModelSpec(model="anthropic/claude-sonnet-5")
ModelSpec(model="ollama/llama3.1")
ModelSpec(model="azure/my-deployment")
assert ModelSpec(model="groq/llama-3.3-70b").provider == "groq"
```

A string with no prefix is read as `openai`:

```python
assert ModelSpec(model="gpt-5.5").provider == "openai"
```

Some providers put a vendor inside the model name; only the first segment is the
provider:

```python
assert ModelSpec(model="openrouter/anthropic/claude-sonnet-5").provider == "openrouter"
```

### Which client serves it

In order, the first that matches:

1. an adapter registered for that provider with `openodke.llm.register`;
2. the standard-library OpenAI-compatible client, when the provider has a default
   endpoint (the `built-in` rows of `odke models`) or the `ModelSpec` sets a
   `base_url`. **No extra dependency**;
3. litellm, for everything else. Needs `pip install "openodke[llm]"`.

```python
from openodke.llm import ModelSpec, OpenAICompatClient, resolve

assert isinstance(resolve(ModelSpec(model="ollama/llama3.1")), OpenAICompatClient)
assert isinstance(resolve(ModelSpec(model="groq/llama-3.3-70b")), OpenAICompatClient)
```

`odke models` says which of the three would serve each provider, and says plainly
when the `[llm]` extra is missing and which providers that excludes.

## Keys

Each provider reads one variable, and openodke does not invent a scheme of its
own: the names are the ones litellm and the vendors' own SDKs already use.

| Provider | Model string | Key variable | Served by |
|---|---|---|---|
| `openai` | `openai/<model>` | `OPENAI_API_KEY` | built-in |
| `openrouter` | `openrouter/<vendor>/<model>` | `OPENROUTER_API_KEY` | built-in |
| `together` | `together/<vendor>/<model>` | `TOGETHER_API_KEY` | built-in |
| `groq` | `groq/<model>` | `GROQ_API_KEY` | built-in |
| `deepseek` | `deepseek/<model>` | `DEEPSEEK_API_KEY` | built-in |
| `ollama` | `ollama/<model>` | none | built-in |
| `vllm` | `vllm/<model>` | none | built-in |
| `lmstudio` | `lmstudio/<model>` | none | built-in |
| `llamacpp` | `llamacpp/<model>` | none | built-in |
| `anthropic` | `anthropic/<model>` | `ANTHROPIC_API_KEY` | litellm |
| `azure` | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` | litellm |
| `bedrock` | `bedrock/<model>` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` | litellm |
| `vertex_ai` | `vertex_ai/<model>` | `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` | litellm |
| `gemini` | `gemini/<model>` | `GEMINI_API_KEY` | litellm |
| `mistral` | `mistral/<model>` | `MISTRAL_API_KEY` | litellm |
| `cohere` | `cohere/<model>` | `COHERE_API_KEY` | litellm |
| `xai` | `xai/<model>` | `XAI_API_KEY` | litellm |
| `perplexity` | `perplexity/<model>` | `PERPLEXITYAI_API_KEY` | litellm |
| `fireworks_ai` | `fireworks_ai/<model>` | `FIREWORKS_AI_API_KEY` | litellm |
| `cerebras` | `cerebras/<model>` | `CEREBRAS_API_KEY` | litellm |
| `huggingface` | `huggingface/<model>` | `HF_TOKEN` | litellm |

`odke models` prints this from the same table the code routes on, with a "set"
column for the environment you are actually in. The table here will fall behind
one day; the command cannot.

**A key elsewhere.** `api_key_env` names a different variable for one spec — a
second OpenAI account, a gateway's own token:

```python
ModelSpec(model="openai/gpt-5.5", api_key_env="OPENAI_API_KEY_EVAL")
```

**A missing key is an error that names the variable**, raised before the request
rather than arriving as a provider's 401:

```python
from openodke.llm import MissingAPIKey, ModelSpec, require_key

try:
    require_key(ModelSpec(model="openai/gpt-5.5"))
except MissingAPIKey as exc:
    assert "OPENAI_API_KEY" in str(exc)
```

It is not raised where a key is genuinely optional: `base_url` means the caller
owns that endpoint's authentication, and `bedrock` and `vertex_ai` accept their
cloud's own credential chain. A `base_url` call is sent no key unless
`api_key_env` names one, so an exported `OPENAI_API_KEY` never reaches a
self-hosted server or a proxy.

```python
assert require_key(ModelSpec(model="openai/local", base_url="http://localhost:8000/v1")) == ""
assert require_key(ModelSpec(model="ollama/llama3.1")) == ""
```

**There is no `set-apikey`.** openodke reads the environment and nothing else: no
command writes a credential to disk, and no configuration file has a place to put
one. A run config names models, never keys.

## Choosing from the command line

`odke run` and `odke ontology infer` take `--model` and `--model-provider`, which
override the config's `models` block:

```bash
odke run odke.yaml --model openai/gpt-5.5
odke run odke.yaml --model gpt-5.5 --model-provider openai      # the same run
odke run odke.yaml --model llama3.1 --model-provider ollama     # entirely local
odke ontology infer corpus/ -o draft.yaml --model ollama/llama3.1
```

`--model` puts **every role** on one model and prints which, so the run states
what it called:

```text
models: every role on openai/gpt-5.5
```

Only the model string changes: grounding keeps its smaller `max_tokens`, so an
override on the command line cannot quietly make the verification pass as dear as
extraction ([DECISIONS #7a](decisions.md)). A per-role choice stays a config
decision — see [`odke run`](run.md#models).

`--model-provider` supplies the prefix when the string has none. A string that
already names a *different* provider is a contradiction and is refused, rather
than one of the two being picked silently.

## OpenAI-compatible endpoints

Anything that serves `/chat/completions` in the OpenAI shape works on the base
install: point `base_url` at it. vLLM, LM Studio, llama.cpp's server,
text-generation-inference, a LiteLLM proxy, an OpenRouter-compatible relay, or a
corporate gateway.

```python
ModelSpec(model="vllm/meta-llama/Llama-3.1-70B-Instruct", base_url="http://gpu-01:8000/v1")
ModelSpec(model="lmstudio/qwen2.5-7b-instruct", base_url="http://localhost:1234/v1")
ModelSpec(
    model="gateway/gpt-5.5",
    base_url="https://llm.corp.internal/v1",
    api_key_env="CORP_LLM_TOKEN",
)
```

`vllm`, `lmstudio` and `llamacpp` have their usual local ports as defaults, so the
`base_url` is only needed when the server is somewhere else. A `base_url` on any
model string routes to the built-in client, whatever the provider is called — that
is how a gateway keeps its own name in the logs.

## Local models with Ollama

The shortest complete setup, with no extra and no key:

```python
from openodke.llm import ModelRoles

roles = ModelRoles.single("ollama/llama3.1")
assert roles.extract.model == roles.ground.model == "ollama/llama3.1"
```

```yaml
models:
  extract: ollama/llama3.1
  ground: ollama/qwen2.5:3b     # grounding is thousands of small yes/no questions
```

`ollama/…` defaults to `http://localhost:11434/v1`; set `base_url` for a host
elsewhere. Leave `ground` out and it follows `extract`, at its own small
`max_tokens`: naming one local model never sends a passage anywhere else.

Mixing is normal — a capable hosted model to extract, a small local one to ground:

```python
from openodke.llm import ModelSpec

ModelRoles(
    extract=ModelSpec(model="anthropic/claude-sonnet-5"),
    ground=ModelSpec(model="ollama/qwen2.5:3b", max_tokens=256),
)
```

## Azure

Azure takes a deployment name where other providers take a model name, and three
variables rather than one:

```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://my-resource.openai.azure.com
export AZURE_API_VERSION=2024-10-21
```

```python
ModelRoles.single("azure/my-deployment")
```

The API version can also travel with the spec, which is what `extra` is for —
anything provider-specific that has no business in `ModelSpec`:

```python
ModelSpec(model="azure/my-deployment", extra={"api_version": "2024-10-21"})
```

Azure is served by litellm, so it needs `pip install "openodke[llm]"`.

## An internal gateway

Many teams may only call models through their own audited proxy. `register`
routes every model string for a provider through a client they supply — no fork,
no subclass, one callable ([DECISIONS #7b](decisions.md)):

```python
from openodke.llm import LLMClient, ModelSpec, register, resolve, unregister
from openodke.llm.testing import ScriptedClient


def audited(spec: ModelSpec) -> LLMClient:
    """Whatever the team's own client is; it needs one method, `complete`."""
    return ScriptedClient(['{"entities": []}'])


register("acme", audited)
assert isinstance(resolve(ModelSpec(model="acme/internal-large")), ScriptedClient)
unregister("acme")
```

Register at import time — a `sitecustomize`, a package your configs already put on
`pythonpath` — and every `acme/…` string in every config routes through it.
Registration wins over both built-in paths, so a team that must send even Ollama
through the proxy can take that over too.

`odke models` lists what has been registered, and marks a provider the registry
has taken over.

## Testing without a provider

`ReplayClient`, `RecordedClient` and `ScriptedClient` ship in the package, so a
model-backed pipeline is testable with no key, no network and no bill. A run
config's `models.replay` is the same thing from a file — see
[`odke run`](run.md#models) and
[Loaders & extraction](loaders-and-extraction.md).

```python
from openodke.llm import Message, ModelSpec, ScriptedClient

client = ScriptedClient([{"verdict": "supported"}])
answer = client.complete([Message(content="…")], spec=ModelSpec(model="test/model"))
assert answer.parsed == {"verdict": "supported"}
```

## What openodke does not ship

- **No key storage.** No command writes a credential anywhere. The environment is
  the whole mechanism.
- **No model catalogue.** No table of context windows, prices or capabilities:
  they change weekly and a copy shipped in a package is wrong by the next release.
  `odke models` lists providers and how to address them, and
  [`meter: true`](run.md#models) reports what a run actually cost, from what the
  provider reported.
- **No default temperature.** Which temperatures a model accepts is a per-model
  fact openodke does not track, so the parameter is omitted entirely unless a
  caller sets it ([DECISIONS #7](decisions.md)).
