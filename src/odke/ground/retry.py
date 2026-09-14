"""Retrying the one call worth retrying.

Grounding is the highest-call-volume stage — one call per candidate fact — so it
meets rate limits first. A 429 on fact 6,000 of 10,000 is weather, not a
verdict: waiting and asking again is right. A 401, a missing adapter or a
malformed response is not weather, and retrying it only spends the wait.

Both adapters raise `ProviderError` with the transport's own exception as its
cause, so the classification reads the cause chain first — an HTTP status where
there is one — and falls back to the message only when nothing structured is
there. Standard library only: no retry package is worth a dependency here.
"""

from __future__ import annotations

import random
import re
import time
import urllib.error
from collections.abc import Callable, Iterator
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from odke.llm.base import ProviderNotInstalled

T = TypeVar("T")

# Timed out, conflicted, too early, rate limited, or the server's own trouble
# (529 is the "overloaded" some providers use).
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

_TRANSIENT_WORDS = re.compile(
    r"rate.?limit|too many requests|timed? ?out|overloaded|temporarily unavailable|"
    r"connection (?:reset|refused|aborted)|\b(?:408|429|500|502|503|504|529)\b",
    re.IGNORECASE,
)


class RetryPolicy(BaseModel):
    """How hard to try before a fact is left `UNCHECKED`.

    Data rather than arguments, like `ModelSpec`, so it can come from the same
    config file and be logged next to the run it shaped. Backoff is exponential
    with full jitter by default — many threads that hit one rate limit together
    must not all come back together — and a server's `Retry-After` wins when it
    sends one, capped at `max_delay`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Total tries, the first included: 1 means never retry.
    attempts: int = Field(default=4, ge=1)
    base_delay: float = Field(default=0.5, ge=0)
    multiplier: float = Field(default=2.0, ge=1)
    max_delay: float = Field(default=30.0, ge=0)
    jitter: bool = True

    def delay(self, retry: int, exc: BaseException | None = None) -> float:
        """Seconds to wait before retry number `retry`, counting from 1."""
        after = retry_after(exc) if exc is not None else None
        if after is not None:
            return min(self.max_delay, after)
        backoff = min(self.max_delay, self.base_delay * self.multiplier ** (retry - 1))
        return random.uniform(0.0, backoff) if self.jitter else backoff


def is_transient(exc: BaseException) -> bool:
    """True when the same call, asked again after a wait, could plausibly answer."""
    chain = list(_chain(exc))
    if any(isinstance(e, ProviderNotInstalled) for e in chain):
        return False
    for e in chain:
        status = _status(e)
        if status is not None:
            return status in TRANSIENT_STATUS
        # URLError without a status is "could not reach"; HTTPError has one.
        if isinstance(e, TimeoutError | ConnectionError | urllib.error.URLError):
            return True
    return any(_TRANSIENT_WORDS.search(str(e)) for e in chain)


def retry_after(exc: BaseException) -> float | None:
    """A `Retry-After` in seconds from anywhere in the cause chain, if one was sent.

    The HTTP-date form is ignored rather than parsed; backoff covers it.
    """
    for e in _chain(exc):
        headers = getattr(e, "headers", None) or getattr(
            getattr(e, "response", None), "headers", None
        )
        get = getattr(headers, "get", None)
        if not callable(get):
            continue
        value = get("Retry-After")
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None
    return None


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    transient: Callable[[BaseException], bool] = is_transient,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """`fn()`, retried on transient errors; the last error is raised when tries run out.

    `sleep` is injected so a test can check the schedule without waiting it.
    """
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= policy.attempts or not transient(exc):
                raise
            wait = policy.delay(attempt, exc)
            if on_retry is not None:
                on_retry(attempt, exc, wait)
            sleep(wait)
            attempt += 1


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status(exc: BaseException) -> int | None:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    # litellm's exceptions, and most SDKs', carry the status as an attribute.
    for name in ("status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600:
            return value
    return None


__all__ = ["TRANSIENT_STATUS", "RetryPolicy", "call_with_retry", "is_transient", "retry_after"]
