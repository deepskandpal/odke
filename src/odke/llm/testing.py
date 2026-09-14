"""Clients that answer without a network: from a script, from entries, or from a cassette.

Ships in the package rather than in `tests/` because anyone building on this
library needs the same thing: a way to test their extractor without a key, a
network, or a bill. It is also what lets CI execute the model-backed code paths
instead of skipping them.

Three shapes, for three kinds of test:

- `ScriptedClient` answers queued responses in order: one call at a time.
- `RecordedClient` answers whichever entry's `match` string is in the prompt,
  so it replays the same way under a thread pool as in a loop.
- `ReplayClient` replays a *cassette* — a JSON file of request matchers and
  recorded completions, usage and cost included. Each interaction answers once,
  and a request nothing matches fails loudly, so a changed prompt fails a test
  rather than silently replaying a stale answer.

Re-recording a cassette when a prompt changes
---------------------------------------------
Wrap a real client in `RecordingClient`, run the code once, and save::

    from odke.llm import ModelSpec, resolve
    from odke.llm.testing import RecordingClient

    spec = ModelSpec(model="anthropic/claude-sonnet-5")
    recorder = RecordingClient(resolve(spec))
    LLMExtractor(client=recorder, spec=spec).extract(chunk, ontology)
    recorder.save("tests/fixtures/llm/my_case.json", description="what it covers")

Do it outside `scripts/verify.sh`, which refuses to start with a provider key in
the environment, and review the diff like any other change. A recorded
interaction matches on its request fingerprint — model, messages and schema —
so the next edit to the prompt fails the replay until it is recorded again. A
hand-authored cassette matches on `contains` instead: substrings the request
must include, which survives rewording while still pinning the passage.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from odke.llm.base import Completion, LLMClient, Message, ModelSpec, ProviderError
from odke.llm.openai_compat import _maybe_json

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


# --------------------------------------------------------------------------- #
# Cassettes
# --------------------------------------------------------------------------- #


def fingerprint(
    messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
) -> str:
    """A stable digest of what was asked: the model, the messages and the schema.

    Temperature, token limits and timeouts are left out: changing them does not
    make a recorded answer to the same question stale.
    """
    payload = {
        "model": spec.model,
        "messages": [[m.role, m.content] for m in messages],
        "schema": schema,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class RequestMatch(BaseModel):
    """What a request must look like for an interaction to answer it.

    Every field that is set must hold. `contains` is checked against the text of
    all the messages together.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: str | None = None
    model: str | None = None
    contains: tuple[str, ...] = ()

    def matches(
        self, messages: Sequence[Message], spec: ModelSpec, schema: dict[str, Any] | None
    ) -> bool:
        if self.model is not None and self.model != spec.model:
            return False
        if self.fingerprint is not None and self.fingerprint != fingerprint(
            messages, spec=spec, schema=schema
        ):
            return False
        body = "\n".join(m.content for m in messages)
        return all(needle in body for needle in self.contains)


class RecordedCompletion(BaseModel):
    """The part of a `Completion` worth keeping. `raw` is dropped: it carries ids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # None is "the provider did not say", as everywhere else — never 0.0.
    cost_usd: float | None = None


class Interaction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    match: RequestMatch = Field(default_factory=RequestMatch)
    response: RecordedCompletion


class Cassette(BaseModel):
    """Recorded interactions, as they are stored under `tests/fixtures/llm/`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = ""
    interactions: tuple[Interaction, ...] = ()

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Cassette:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | os.PathLike[str]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = self.model_dump_json(indent=2, exclude_defaults=True)
        target.write_text(body + "\n", encoding="utf-8")


class ReplayClient:
    """Answers from a cassette, with no network and no key.

    Each interaction answers once: the first unused one whose match fits. So a
    test that makes the same call twice needs two interactions, and calls made in
    another order — or from another thread — still replay. A call nothing matches
    is a `ProviderError` naming its fingerprint, which is what a recording needs.
    """

    def __init__(self, cassette: Cassette | str | os.PathLike[str]) -> None:
        self.cassette = cassette if isinstance(cassette, Cassette) else Cassette.load(cassette)
        self._used = [False] * len(self.cassette.interactions)
        self._lock = threading.Lock()
        self.calls: list[Call] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        recorded: RecordedCompletion | None = None
        with self._lock:
            self.calls.append((list(messages), spec, schema))
            count = len(self.calls)
            for i, interaction in enumerate(self.cassette.interactions):
                if not self._used[i] and interaction.match.matches(messages, spec, schema):
                    self._used[i] = True
                    recorded = interaction.response
                    break
            unused = self._used.count(False)
        if recorded is None:
            raise ProviderError(
                f"no recorded interaction matches call {count} "
                f"(fingerprint {fingerprint(messages, spec=spec, schema=schema)}, "
                f"{unused} unused). If the prompt changed on purpose, re-record the "
                "cassette — see odke.llm.testing."
            )
        return Completion(
            text=recorded.text,
            parsed=_maybe_json(recorded.text),
            model=recorded.model or spec.model,
            prompt_tokens=recorded.prompt_tokens,
            completion_tokens=recorded.completion_tokens,
            cost_usd=recorded.cost_usd,
        )

    @property
    def unused(self) -> int:
        return self._used.count(False)

    @property
    def exhausted(self) -> bool:
        return self.unused == 0


class RecordingClient:
    """Wraps a real client and keeps every exchange, for `save()`."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self.interactions: list[Interaction] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        completion = self.inner.complete(messages, spec=spec, schema=schema)
        interaction = Interaction(
            match=RequestMatch(
                fingerprint=fingerprint(messages, spec=spec, schema=schema), model=spec.model
            ),
            response=RecordedCompletion(
                text=completion.text,
                model=completion.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost_usd=completion.cost_usd,
            ),
        )
        with self._lock:
            self.interactions.append(interaction)
        return completion

    def cassette(self, description: str = "") -> Cassette:
        with self._lock:
            return Cassette(description=description, interactions=tuple(self.interactions))

    def save(self, path: str | os.PathLike[str], *, description: str = "") -> Cassette:
        cassette = self.cassette(description)
        cassette.save(path)
        return cassette


__all__ = [
    "Call",
    "Cassette",
    "Interaction",
    "RecordedClient",
    "RecordedCompletion",
    "RecordingClient",
    "ReplayClient",
    "RequestMatch",
    "ScriptedClient",
    "fingerprint",
]
