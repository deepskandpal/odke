"""What the rule tests share: the repository's paths, a copied example, a live model's stand-in.

The rules in this directory check the whole codebase for one kind of bug each
(DECISIONS #50). They run on recorded or scripted model answers only: no
network, no key, no bill.
"""

from __future__ import annotations

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


def python_files(*roots: Path) -> list[Path]:
    """Every `.py` file under the roots, in a stable order."""
    return sorted(p for root in roots for p in root.rglob("*.py"))
