"""What the rule tests share: the repository's paths, a copied example, a live model's stand-in.

The rules in this directory check the whole codebase for one kind of bug each
(DECISIONS #50). They run on recorded or scripted model answers only: no
network, no key, no bill.
"""

from __future__ import annotations

import ast
import importlib
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openodke.llm import Completion, Message, ModelSpec

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "openodke"
BENCH = REPO / "bench"
EXAMPLES = REPO / "examples"


def copy_example(name: str, into: Path) -> Path:
    """`examples/<name>` copied under `into`, so a run writes nowhere in the repository."""
    target = into / name
    shutil.copytree(EXAMPLES / name, target, ignore=shutil.ignore_patterns("out"))
    return target


class LiveModel:
    """Stands in for a provider's model: answers every call with `answer`, and counts them.

    A test puts it where `openodke.llm.registry.resolve` would put a real
    client, so "the live model was asked" is something to count.
    """

    def __init__(self, answer: dict[str, Any] | None = None) -> None:
        self.answer = answer if answer is not None else {"verdict": "contradicted"}
        self.calls = 0
        self._lock = threading.Lock()

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        import json

        with self._lock:
            self.calls += 1
        return Completion(text=json.dumps(self.answer), parsed=dict(self.answer), model=spec.model)


# Every module that binds the registry's `resolve` when it is imported, and the name it binds.
# A module imported for the first time while `resolve` is patched would keep the patch for
# good, so each is imported before it is patched, and patched where it looks the name up.
RESOLVERS = {
    "openodke.llm.registry": "resolve",
    "openodke.llm": "resolve",
    "openodke.llm.roles": "resolve",
    "openodke.run.build": "resolve_client",
    "openodke.extract.llm": "resolve",
    "openodke.infer.llm": "resolve",
    "openodke.corroborate.judge": "resolve_client",
    "openodke.eval.equivalence": "resolve_client",
}


def answer_every_model_with(monkeypatch: Any, client: Any) -> None:
    """`client` wherever a provider's client is resolved, undone when the test ends."""
    # All imported before any is patched: one imported later would bind the patch.
    modules = {name: importlib.import_module(name) for name in RESOLVERS}
    for name, attribute in RESOLVERS.items():
        monkeypatch.setattr(modules[name], attribute, lambda spec, _client=client: _client)


def bindings_of_resolve() -> dict[str, str]:
    """Each module under `src/` that binds the registry's `resolve` on import, and the name.

    Read from the syntax tree, so every spelling counts: absolute or relative, from
    `openodke.llm` or `openodke.llm.registry`, on one line or several. An import inside a
    function looks the name up when it runs, so it is patched already and is left out.
    """
    found = {"openodke.llm.registry": "resolve"}
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        package = parts[:-1]
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        for node in _module_level(ast.parse(path.read_text(encoding="utf-8")).body):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = list(package[: len(package) - node.level + 1]) if node.level else []
            source = ".".join([*base, *(node.module.split(".") if node.module else [])])
            if source not in ("openodke.llm", "openodke.llm.registry"):
                continue
            for alias in node.names:
                if alias.name == "resolve":
                    found[module] = alias.asname or alias.name
    return found


def _module_level(body: list[ast.stmt]) -> list[ast.stmt]:
    """The statements that run on import: the module's, and those under its `if` and `try`."""
    out: list[ast.stmt] = []
    for node in body:
        out.append(node)
        if isinstance(node, ast.If | ast.Try):
            for block in (node.body, node.orelse, getattr(node, "finalbody", [])):
                out += _module_level(block)
            for handler in getattr(node, "handlers", []):
                out += _module_level(handler.body)
    return out
