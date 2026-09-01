"""A client that answers from a script.

Ships in the package rather than in `tests/` because anyone building on this
library needs the same thing: a way to test their extractor without a key, a
network, or a bill. It is also what lets CI execute the model-backed code paths
instead of skipping them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from odke.llm.base import Completion, Message, ModelSpec, ProviderError


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
        self.calls: list[tuple[list[Message], ModelSpec, dict[str, Any] | None]] = []

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
        item = self._responses.pop(0)
        text = json.dumps(item) if isinstance(item, dict) else item
        return Completion(
            text=text,
            parsed=item if isinstance(item, dict) else None,
            model=spec.model,
            prompt_tokens=sum(len(m.content) for m in messages) // 4,
            completion_tokens=len(text) // 4,
        )

    @property
    def exhausted(self) -> bool:
        return not self._responses


__all__ = ["ScriptedClient"]
