"""What the rule tests share: the repository's paths, a copied example, a live model's stand-in.

The rules in this directory check the whole codebase for one kind of bug each
(DECISIONS #50). They run on recorded or scripted model answers only: no
network, no key, no bill.
"""

from __future__ import annotations

import importlib
import re
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


def bindings_of_resolve() -> set[str]:
    """The modules under `src/` that bind `resolve` from the registry at import time."""
    found = {"openodke.llm.registry"}
    binding = re.compile(r"^from openodke\.llm\.registry import .*\bresolve\b", re.M)
    for path in sorted(SRC.rglob("*.py")):
        if binding.search(path.read_text(encoding="utf-8")):
            parts = path.relative_to(SRC.parent).with_suffix("").parts
            found.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return found
