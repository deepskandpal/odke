"""How a batch says which of its items failed, and hands back the rest (#162).

A stage that runs a batch, `extract_many` or `ground_documents`, is asked
about many documents at once. One document's provider error must not cost the
others theirs, so a batching stage finishes every item it can and then raises
the first failure it met, with two attributes set on it:

- `partial`: one result per item, in order, None where the item failed;
- `failures`: the items that failed, by index, each with its own exception.

The exception keeps its own type, so a caller that catches `ProviderError`
still catches it; the pipeline reads the two attributes, leaves out the
documents that failed, and carries on with the rest. A budget stop
(`BudgetExceeded`) sets the same two attributes, and is the run's, not one
document's.

A configuration error is nobody's document. A missing key, a missing provider
adapter or a missing extra fails every call alike, so it is raised at once and
stops the run, as it always has (DECISIONS #7a).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openodke.llm.base import MissingAPIKey, ProviderNotInstalled

# What stops a run rather than one document: every call would fail the same way.
CONFIG_ERRORS: tuple[type[BaseException], ...] = (MissingAPIKey, ProviderNotInstalled, ImportError)


def is_config_error(exc: BaseException) -> bool:
    """True for an error no document caused: a missing key, adapter or extra."""
    return isinstance(exc, CONFIG_ERRORS) or not isinstance(exc, Exception)


def incomplete(failures: Mapping[int, Exception], partial: Sequence[Any]) -> Exception:
    """The first failure, by index, carrying the batch's `partial` and `failures`, to raise."""
    first = failures[min(failures)]
    setattr(first, "partial", list(partial))  # noqa: B010 - an attribute the class may lack
    setattr(first, "failures", dict(failures))  # noqa: B010
    return first


def told(exc: BaseException, size: int) -> tuple[list[Any], dict[int, Exception]] | None:
    """What a batch of `size` items said about itself when it raised `exc`, or None.

    None when it said nothing usable: no `partial` of the right length, or no
    `failures` by index.
    """
    partial = getattr(exc, "partial", None)
    failures = getattr(exc, "failures", None)
    if not isinstance(partial, list) or len(partial) != size or not isinstance(failures, Mapping):
        return None
    return partial, {int(i): e for i, e in failures.items()}


def reason(stage: str, exc: BaseException) -> str:
    """`extract: ProviderError: 503 overloaded`: why a document failed, in one line."""
    text = " ".join(str(exc).split())
    if len(text) > 200:
        text = text[:197] + "..."
    return f"{stage}: {type(exc).__name__}: {text}" if text else f"{stage}: {type(exc).__name__}"


# How many failed documents a report line names before it says how many more.
SHOWN = 3


def failed_summary(
    failed: Mapping[str, str], *, of: int | None = None, noun: str = "document"
) -> str:
    """`2 of 40 documents left out: a.md (extract: ...), b.md (...)`: the report's line."""
    shown = [f"{name} ({why})" for name, why in list(failed.items())[:SHOWN]]
    more = len(failed) - len(shown)
    tail = f", and {more} more" if more > 0 else ""
    count = f"{len(failed)} of {of}" if of is not None else str(len(failed))
    plural = noun if len(failed) == 1 and of is None else f"{noun}s"
    return f"{count} {plural} left out: {', '.join(shown)}{tail}"


__all__ = [
    "CONFIG_ERRORS",
    "SHOWN",
    "failed_summary",
    "incomplete",
    "is_config_error",
    "reason",
    "told",
]
