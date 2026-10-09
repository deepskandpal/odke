"""A response cache, so a rerun of an unchanged batch costs nothing (#156).

`CachedClient` wraps any `LLMClient`. Each request gets a key, and a request
whose key is already in the store is answered from it without a call. Every
other request goes to the client it wraps, and its answer is stored under its
key before it is returned.

The key is the SHA-256 of one canonical JSON object holding everything that
changes the answer:

- the provider-qualified model, and the `base_url` when one redirects the call;
- every message, role and content, in order: the system prompt, the ontology
  snippets, the claim and the passage;
- the response schema;
- the spec's sampling parameters: `temperature`, `max_tokens` and `extra`;
- the key of each registered prompt the messages carry (DECISIONS #27), where
  the registry knows the text;
- `repeat`, when it is not 0: which draw of one question this is. Gold
  adjudication asks each question three times on purpose (#145), and without
  it the second and third would replay the first. Draw 0 adds nothing, so a key
  written before `repeat` existed is still the same key, and the first draw of
  a question shares its entry with an ordinary call of it.

`timeout` and `api_key_env` are left out: they decide whether a call
succeeds, not what it says. Change anything else, the prompt's version
included, and the request misses.

An error is never stored, so a call that failed is made again next time. A
reply the caller rejects is stored like any other: a malformed extraction is
followed by a repair turn, whose messages differ and so whose key does too, and
a rerun replays the same reply, the same rejection and the same repair.

A hit is the stored answer with `cached=True`, no tokens and a cost of `0.0`:
nothing was sent, so nothing was spent, and a cost meter counts it as a call
that cost nothing. The answer's own usage stays in the store.

Two stores. `DirectoryCache` keeps one JSON file per key under a two-character
shard. Each file is written to a temporary name beside it and renamed into
place, so a reader sees a whole entry or none, from any thread or process. A
file that does not parse is a miss. `MemoryCache` is a dictionary, for one
process. Any object with `get(key)` and `put(key, entry)` is a store too.

Two threads asking the same new question at once both call the model; the
second answer overwrites the first. The thread pools in this library ask
different questions, so the cache does not hold one caller back for another.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openodke.llm.base import Completion, LLMClient, Message, ModelSpec

log = logging.getLogger("openodke.llm.cache")

# Part of every key, so an entry written in another layout is never read as this one.
FORMAT = 1


@runtime_checkable
class CacheStore(Protocol):
    """Where answers are kept: an entry by key, or None for a miss."""

    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, entry: dict[str, Any]) -> None: ...


def prompt_keys(messages: Sequence[Message]) -> list[str]:
    """The registered prompts the messages carry, by key, as far as the registry knows.

    A prompt is carried when a message starts with its text: the system message
    that opens with `extract@1`'s instructions, or the repair turn that is
    `extract.repair@1` exactly. A template (`infer@1`) is filled in before it is
    sent, so it is not found, and the messages alone key it.
    """
    from openodke.prompts import registered

    found = [p.key for p in registered() if any(m.content.startswith(p.text) for m in messages)]
    return sorted(found)


def request(
    messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Everything that changes the answer, as the object `cache_key` hashes.

    `repeat` is in it only when it is not 0, so every key from before it
    existed still finds its entry.
    """
    body: dict[str, Any] = {
        "format": FORMAT,
        "model": spec.model,
        "base_url": spec.base_url,
        "messages": [[m.role, m.content] for m in messages],
        "schema": schema,
        "sampling": {
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
            "extra": spec.extra,
        },
        "prompts": prompt_keys(messages),
    }
    if spec.repeat:
        body["repeat"] = spec.repeat
    return body


def cache_key(
    messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
) -> str:
    """The SHA-256 of the request as canonical JSON: sorted keys, no whitespace."""
    return _digest(request(messages, spec=spec, schema=schema))


def _digest(body: dict[str, Any]) -> str:
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MemoryCache:
    """Answers kept in a dictionary, for as long as the process runs."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = entry

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class DirectoryCache:
    """Answers kept on disk: `<directory>/<key[:2]>/<key>.json`, one file per key.

    The directory is created here, so a path that cannot be one fails when the
    cache is made rather than on the first answer.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            text = self.path(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            entry = json.loads(text)
        except json.JSONDecodeError:
            return None
        return entry if isinstance(entry, dict) else None

    def put(self, key: str, entry: dict[str, Any]) -> None:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # A temporary file in the same directory, renamed over the target:
        # `os.replace` is atomic there, so no reader ever sees half an entry.
        handle, temporary = tempfile.mkstemp(prefix=f".{key[:16]}.", dir=target.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*/*.json"))


class CachedClient:
    """An `LLMClient` that answers a request it has seen from the store.

    `cache` is a store, a directory path for a `DirectoryCache`, or None for a
    `MemoryCache`. `stats` counts `hits`, `misses` and `failed` calls, which
    raised and stored nothing. A store that cannot be written logs a warning
    and the answer is still returned: the call was paid for either way.
    """

    def __init__(
        self, inner: LLMClient, cache: CacheStore | str | os.PathLike[str] | None = None
    ) -> None:
        self.inner = inner
        if cache is None:
            self.store: CacheStore = MemoryCache()
        elif isinstance(cache, str | os.PathLike):
            self.store = DirectoryCache(cache)
        else:
            self.store = cache
        self._lock = threading.Lock()
        self._counts = {"hits": 0, "misses": 0, "failed": 0}

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        body = request(messages, spec=spec, schema=schema)
        key = _digest(body)
        stored = _completion(self._read(key))
        if stored is not None:
            self._bump("hits")
            return stored.model_copy(
                update={"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "cached": True}
            )
        try:
            completion = self.inner.complete(messages, spec=spec, schema=schema)
        except BaseException:
            self._bump("failed")
            raise
        self._bump("misses")
        entry = {
            "format": FORMAT,
            "key": key,
            "model": spec.model,
            "prompts": body["prompts"],
            "completion": completion.model_dump(mode="json", include=set(_KEPT)),
        }
        try:
            self.store.put(key, entry)
        except OSError as exc:
            log.warning("could not store a model answer in the cache: %s", exc)
        return completion

    def _read(self, key: str) -> dict[str, Any] | None:
        try:
            return self.store.get(key)
        except OSError as exc:
            # A store that cannot be read is a miss: the call is made, not lost.
            log.warning("could not read the response cache: %s", exc)
            return None

    def _bump(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


# What an entry keeps of a `Completion`. `raw` is left out: it carries ids.
_KEPT = ("text", "parsed", "model", "prompt_tokens", "completion_tokens", "cost_usd")


def _completion(entry: dict[str, Any] | None) -> Completion | None:
    """The stored answer, or None when there is none or it is not one this version wrote."""
    if entry is None or entry.get("format") != FORMAT:
        return None
    body = entry.get("completion")
    if not isinstance(body, dict):
        return None
    try:
        return Completion.model_validate({k: v for k, v in body.items() if k in _KEPT})
    except ValueError:
        return None


__all__ = [
    "FORMAT",
    "CacheStore",
    "CachedClient",
    "DirectoryCache",
    "MemoryCache",
    "cache_key",
    "prompt_keys",
    "request",
]
