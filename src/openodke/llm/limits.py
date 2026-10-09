"""Concurrency limits per provider, shared by every stage in the process (#162).

The extractor and the grounder each keep up to `max_workers` calls in flight,
and both may be calling one provider account. Eight extraction calls and eight
grounding calls is sixteen against a limit the account counts once. So
`ProviderLimits` holds one semaphore per provider for the whole process, and
`LimitedClient` takes a slot before each call and gives it back after.

The key is the provider, the prefix of the model string as `ModelSpec.provider`
reads it, not the model. A provider counts rate limits per account and per
key, so `anthropic/claude-sonnet-5-5` extracting and
`anthropic/claude-haiku-4-5-20251001` grounding share one limit. Two servers
behind one prefix with different `base_url`s share it too; give one its own
prefix through `openodke.llm.register` to separate them.

A stage's own `max_workers` still applies. The stricter of the two wins: a
grounder with four workers under a provider limit of eight keeps four in flight,
and with sixteen workers keeps eight, less whatever the extractor holds.

A provider that answers with a `Retry-After` has said when the account may ask
again, so every call to that provider waits it out, not only the one that will
retry. The header is read by `openodke.ground.retry.retry_after`, the reader
the retry policy already uses, and each wait is capped at `max_pause`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from openodke.llm.base import Completion, LLMClient, Message, ModelSpec


class ProviderLimits:
    """A semaphore per provider, and when each provider may next be asked.

    `PROVIDER_LIMITS` is the process's. A provider with no limit set is not
    held back, apart from a `Retry-After` it sent. `clock` and `sleep` are
    injected so a test can check a pause without waiting it out.
    """

    def __init__(
        self,
        *,
        max_pause: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_pause = max_pause
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._limits: dict[str, int] = {}
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self._resume: dict[str, float] = {}

    def set(self, provider: str, limit: int | None) -> None:
        """At most `limit` calls in flight to `provider`; None lifts the limit.

        Calls already in flight finish under the limit they started with.
        """
        name = provider.lower()
        if limit is not None and limit < 1:
            raise ValueError(f"a provider limit is at least 1, got {provider}: {limit}")
        with self._lock:
            if limit is None:
                self._limits.pop(name, None)
                self._slots.pop(name, None)
            elif self._limits.get(name) != limit:
                self._limits[name] = limit
                self._slots[name] = threading.BoundedSemaphore(limit)

    def update(self, limits: Mapping[str, int | None]) -> None:
        for provider, limit in limits.items():
            self.set(provider, limit)

    @property
    def limits(self) -> dict[str, int]:
        with self._lock:
            return dict(self._limits)

    @contextmanager
    def slot(self, provider: str) -> Iterator[None]:
        """Hold one of `provider`'s slots, once any pause it asked for is over."""
        name = provider.lower()
        with self._lock:
            semaphore = self._slots.get(name)
        if semaphore is None:
            self._wait(name)
            yield
            return
        with semaphore:
            self._wait(name)
            yield

    def pause(self, provider: str, seconds: float) -> None:
        """No call to `provider` goes out for `seconds`, capped at `max_pause`."""
        until = self._clock() + min(max(seconds, 0.0), self.max_pause)
        name = provider.lower()
        with self._lock:
            self._resume[name] = max(self._resume.get(name, 0.0), until)

    def _wait(self, name: str) -> None:
        while True:
            with self._lock:
                left = self._resume.get(name, 0.0) - self._clock()
            if left <= 0:
                return
            self._sleep(left)

    def client(self, inner: LLMClient) -> LimitedClient:
        """`inner`, with every call it makes held to these limits."""
        return LimitedClient(inner, self)


# The process's limits: every `LimitedClient` without limits of its own shares these.
PROVIDER_LIMITS = ProviderLimits()


def set_limit(provider: str, limit: int | None) -> None:
    """At most `limit` calls in flight to `provider` in this process; None lifts it."""
    PROVIDER_LIMITS.set(provider, limit)


class LimitedClient:
    """An `LLMClient` that holds a slot of its spec's provider for every call."""

    def __init__(self, inner: LLMClient, limits: ProviderLimits | None = None) -> None:
        self.inner = inner
        self.limits = limits if limits is not None else PROVIDER_LIMITS

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        provider = spec.provider
        with self.limits.slot(provider):
            try:
                return self.inner.complete(messages, spec=spec, schema=schema)
            except Exception as exc:
                # Here, not at the top: the grounder's package imports this one.
                from openodke.ground.retry import retry_after

                after = retry_after(exc)
                if after is not None:
                    self.limits.pause(provider, after)
                raise


__all__ = ["PROVIDER_LIMITS", "LimitedClient", "ProviderLimits", "set_limit"]
