"""A client that answers from a script.

Ships in the package rather than in `tests/` because anyone building on this
library needs the same thing: a way to test their extractor without a key, a
network, or a bill. It is also what lets CI execute the model-backed code paths
instead of skipping them.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from odke.llm.base import Completion, Message, ModelSpec, ProviderError

Call = tuple[list[Message], ModelSpec, dict[str, Any] | None]


def _completion_for(
    item: str | dict[str, Any], messages: Sequence[Message], spec: ModelSpec
) -> Completion:
    text = json.dumps(item) if isinstance(item, dict) else item
    return Completion(
        text=text,
        parsed=item if isinstance(item, dict) else None,
        model=spec.model,
        prompt_tokens=sum(len(m.content) for m in messages) // 4,
        completion_tokens=len(text) // 4,
    )


class ScriptedClient:
    """Returns queued responses in order and records what it was asked.

    client = ScriptedClient(['{"name": "Ada"}'])
    client.complete([Message(content="…")], spec=ModelSpec(model="test/x"))
    """

    def __init__(
        self, responses: Sequence[str | dict[str, Any]] = (), *, strict: bool = True
    ) -> None:
        self._responses = list(responses)
        self._strict = strict
        self.calls: list[Call] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls.append((list(messages), spec, schema))
        if not self._responses:
            if self._strict:
                raise ProviderError(
                    f"ScriptedClient ran out of responses on call {len(self.calls)}. "
                    "Queue more, or pass strict=False to return empty completions."
                )
            return Completion(text="", model=spec.model)
        return _completion_for(self._responses.pop(0), messages, spec)

    @property
    def exhausted(self) -> bool:
        return not self._responses


class RecordedClient:
    """Answers by matching the prompt rather than by position.

    `ScriptedClient` answers in order, which is right for one call at a time and
    wrong the moment calls run concurrently: which question gets which answer
    becomes whichever thread got there first. This one holds recorded entries,
    each with a `match` string, and answers with the first entry whose `match`
    appears in the prompt — so a fixture of recorded responses replays the same
    way under a thread pool as it does in a loop.

    Each entry is a mapping. `{"match": "…", "response": "…" | {…}}` answers;
    `{"match": "…", "error": "…"}` raises `ProviderError`, which is how a
    recorded failure is replayed. `from_fixture()` loads a JSON list of them.
    """

    def __init__(self, entries: Sequence[Mapping[str, Any]], *, strict: bool = True) -> None:
        self._entries = [dict(e) for e in entries]
        self._strict = strict
        self._lock = threading.Lock()
        self.calls: list[Call] = []

    @classmethod
    def from_fixture(cls, path: str | Path, *, strict: bool = True) -> RecordedClient:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), strict=strict)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        with self._lock:
            self.calls.append((list(messages), spec, schema))
            count = len(self.calls)
        prompt = "\n".join(m.content for m in messages)
        entry = next((e for e in self._entries if e["match"] in prompt), None)
        if entry is None:
            if self._strict:
                raise ProviderError(
                    f"RecordedClient has no entry matching call {count}: {prompt[:200]!r}. "
                    "Record one, or pass strict=False to return empty completions."
                )
            return Completion(text="", model=spec.model)
        if "error" in entry:
            raise ProviderError(str(entry["error"]))
        return _completion_for(entry["response"], messages, spec)


__all__ = ["Call", "RecordedClient", "ScriptedClient"]
