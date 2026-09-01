# Contributing

## The whole check is one script

```bash
git clone https://github.com/deepskandpal/odke && cd odke
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
./scripts/verify.sh
```

Ten steps: credentials, interpreter, lock, lint, format, types, tests, the
base-install guarantee, the build, and a smoke test of the built wheel in a clean
environment. CI runs this same script on three interpreters — there is no second
list of steps to drift out of sync with this one.

## What a good change looks like

- **A test that would fail without it.** Not coverage for its own sake: a test
  that names the behaviour and would catch its loss.
- **The reason, in the code.** Comments here explain *why*, not *what*. If you
  found the reasoning non-obvious, so will the next person.
- **A `DECISIONS.md` entry** for anything a future contributor would reasonably
  try to reverse.

## Things worth knowing before you start

- **The base install talks to nothing.** `pip install odke` must keep working
  with no provider, no driver and no network. `verify.sh` step 8 enforces it, so
  a new top-level import of `litellm`, `neo4j` or `rdflib` will fail the build.
  Import those inside the module that needs them.
- **No vendor names outside `odke/llm/`.** Everything else goes through
  `LLMClient`. A `import anthropic` in the extractor is a bug, not a shortcut.
- **Tests never touch the network.** Use `ScriptedClient` for model calls and the
  injected opener for HTTP. Step 1 refuses to run if a live key is in the
  environment.
- **Facts are frozen.** Stages return new objects. A stage that mutated one in
  place would make its own provenance wrong.

## Adding a sink

`Sink` is a `Protocol` — one method, no base class, no registration:

```python
class MySink:
    def write(self, kg: KnowledgeGraph) -> None: ...
```

If it needs a driver, put it behind an extra in `pyproject.toml` and import the
driver inside the module, not at package level.

## Adding a provider

Most providers need no code: they are either OpenAI-shaped (already covered) or
supported by litellm (already covered). If yours is genuinely neither, implement
`LLMClient` and register it:

```python
from odke.llm import register

register("myprovider", lambda spec: MyClient(spec))
```

A new adapter in this repository needs a test proving it normalises to the same
`Completion` as the others — including `cost_usd is None` when the provider does
not report cost, rather than `0.0`.
