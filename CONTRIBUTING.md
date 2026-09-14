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

## Every change is a pull request

`main` is protected. Nothing is pushed to it directly, and a pull request can
merge only when these checks pass: `verify (python 3.11)`, `verify (python 3.12)`,
`verify (python 3.13)` and `neo4j live tests (python 3.12, neo4j 5.26)`. No
approving review is required. Force-pushes and deleting `main` are blocked.

## Releasing

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/openodke/__init__.py`, and move the CHANGELOG's Unreleased section under
   a dated heading — in a pull request.
2. Optionally rehearse: `gh workflow run release.yml -f target=testpypi`.
3. After it merges, tag and push: `git tag -a vX.Y.Z -m "openodke X.Y.Z" && git push origin vX.Y.Z`.

`release.yml` runs the checks, refuses a tag that does not match the package
version, publishes to PyPI through trusted publishing (the `pypi` environment
deploys only from `main` or a `v*` tag), and creates the GitHub release.

## What a good change looks like

- **A test that would fail without it.** Not coverage for its own sake: a test
  that names the behaviour and would catch its loss.
- **The reason, in the code.** Comments here explain *why*, not *what*. If you
  found the reasoning non-obvious, so will the next person.
- **A `DECISIONS.md` entry** for anything a future contributor would reasonably
  try to reverse.

## Things worth knowing before you start

- **The base install talks to nothing.** `pip install openodke` must keep working
  with no provider, no driver and no network. `verify.sh` step 8 enforces it, so
  a new top-level import of `litellm`, `neo4j` or `rdflib` will fail the build.
  Import those inside the module that needs them.
- **No vendor names outside `openodke/llm/`.** Everything else goes through
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
from openodke.llm import register

register("myprovider", lambda spec: MyClient(spec))
```

A new adapter in this repository needs a test proving it normalises to the same
`Completion` as the others — including `cost_usd is None` when the provider does
not report cost, rather than `0.0`.
