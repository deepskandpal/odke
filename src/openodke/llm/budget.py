"""Budgets: a run stops cleanly at its limit and keeps what it has (#157).

A `Budget` is the most a run may spend: USD, input tokens, output tokens and
calls, each optional. A `Ledger` holds one budget and what has been spent
against it, and `Ledger.client(inner)` wraps a client so every call it makes
is counted there. One ledger serves every stage and every thread of a run, so
the extractor's calls and the grounder's come out of the same budget.

Each call is checked before it is made, against an estimate:

- one call;
- its input tokens: the characters of every message and of the schema, divided
  by four and rounded up, plus four per message for the role framing. Four
  characters a token is the usual rule for English under the tokenisers in
  use, and it needs no tokeniser;
- its output tokens: the spec's `max_tokens`, the most it can return;
- its USD: the run's own USD per token so far, over the calls a provider
  priced, times the input and output estimates. Before any priced call there
  is no rate to estimate with, so USD is checked after each call instead, and
  the call after the one that reached the limit is refused.

A call is made when what is spent, what the calls in flight may still spend,
and its estimate all fit. When only the calls in flight stand in the way, it
waits for them to settle and checks again; when what is already spent leaves
no room, the ledger stops, and that call and every call after it raise
`BudgetExceeded` without being made. A call that raised is not counted: most
providers bill nothing for one.

`BudgetExceeded` is not a `ProviderError`, and it is never retried. The
pipeline, the Validator and the CLI turn it into a partial result: everything
grounded so far is kept, a fact the stop reached first stays `UNCHECKED`, the
sinks still write, and the report says where the run stopped and why. A stage
that runs a batch hands back what it finished as `BudgetExceeded.partial`.

A USD budget counts priced calls only. A provider that reports no cost, a
local server say, never reaches one, so give such a run a limit on calls or
tokens as well.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from openodke.llm.base import Completion, LLMClient, Message, ModelSpec

# Each limit, in the order a report names them.
LIMITS = ("usd", "calls", "input_tokens", "output_tokens")
# Characters a token, and tokens a message for its role framing.
CHARS_PER_TOKEN = 4
TOKENS_PER_MESSAGE = 4


class Budget(BaseModel):
    """The most a run may spend. A limit left out is no limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    usd: float | None = Field(default=None, ge=0)
    calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @property
    def limits(self) -> dict[str, float | int]:
        """The limits that are set, by name."""
        return {name: value for name in LIMITS if (value := getattr(self, name)) is not None}


@dataclass(frozen=True)
class Spent:
    """What the calls a ledger counted spent, and whether their cost is known."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # The sum over the calls a provider priced.
    priced_usd: float = 0.0
    # Calls whose cost no provider reported.
    unpriced_calls: int = 0

    @property
    def usd(self) -> float | None:
        """The total, or None when any call's cost is unknown: never a partial sum."""
        return None if self.unpriced_calls else self.priced_usd

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "usd": self.usd}


@dataclass(frozen=True)
class Estimate:
    """What one call may spend, as the ledger reckons it before the call."""

    input_tokens: int
    output_tokens: int
    # None when no call so far was priced, so there is no rate to estimate with.
    usd: float | None


class BudgetExceeded(RuntimeError):
    """A call would take the run past its budget, so the run stops here.

    `limit` names the limit that stopped it, `budget` and `spent` say where it
    stood, and `ledger` is the ledger, whose `spent` keeps counting the calls
    that were already in flight. `partial` is set by a stage that ran a batch:
    what it finished before the stop.
    """

    def __init__(
        self, limit: str, *, budget: Budget, spent: Spent, ledger: Ledger | None = None
    ) -> None:
        self.limit = limit
        self.budget = budget
        self.spent = spent
        self.ledger = ledger
        self.partial: Any = None
        super().__init__(
            f"stopped at budget: {describe(limit, spent, budget)}; no further model call is made"
        )

    def report(self) -> dict[str, Any]:
        """The numbers a run report states: the limit, the budget, and what was spent."""
        spent = self.ledger.spent if self.ledger is not None else self.spent
        return {
            "reason": "budget",
            "limit": self.limit,
            "budget": self.budget.limits,
            "spent": spent.as_dict(),
        }


def describe(limit: str, spent: Spent, budget: Budget) -> str:
    """`usd $1.5012 of $1.50`, `calls 40 of 40`: one limit, as a person reads it."""
    value = budget.limits.get(limit)
    if limit == "usd":
        used = spent.priced_usd
        return f"usd ${used:.4f} of ${value:.2f}" if value is not None else f"usd ${used:.4f}"
    used_count = getattr(spent, limit)
    return f"{limit} {used_count} of {value}" if value is not None else f"{limit} {used_count}"


def stopped_summary(stopped: Mapping[str, Any]) -> str:
    """A run's `stats["stopped"]` as one line of its report."""
    spent = stopped.get("spent") or {}
    budget = Budget(**dict(stopped.get("budget") or {}))
    fields = {k: spent[k] for k in ("calls", "input_tokens", "output_tokens") if k in spent}
    used = Spent(**fields, priced_usd=float(spent.get("priced_usd") or 0.0))
    line = f"at budget, {describe(str(stopped.get('limit')), used, budget)}"
    if stopped.get("stage"):
        line += f", during {stopped['stage']}"
    left = []
    if unextracted := stopped.get("unextracted"):
        left.append(f"{unextracted} chunks not extracted")
    left.append(f"{stopped.get('unchecked', 0)} facts left unchecked")
    return f"{line}: {', '.join(left)}"


def budget_summary(budget: Mapping[str, Any], spent: Mapping[str, Any]) -> str:
    """Each limit set and what was spent against it: `usd $0.0123 of $1.50, calls 12 of 40`."""
    limits = Budget(**dict(budget))
    fields = {k: spent[k] for k in ("input_tokens", "output_tokens") if k in spent}
    # The calls a budget counts are the ones that went out, not the cache's.
    calls = int(spent.get("calls", 0)) - int(spent.get("cached_calls", 0))
    used = Spent(calls=calls, **fields, priced_usd=float(spent.get("priced_usd") or 0.0))
    return ", ".join(describe(name, used, limits) for name in limits.limits)


def estimate_tokens(messages: Sequence[Message], schema: dict[str, Any] | None = None) -> int:
    """Input tokens, reckoned without a tokeniser: characters / 4, plus 4 a message."""
    chars = sum(len(m.content) for m in messages)
    if schema is not None:
        chars += len(json.dumps(schema, separators=(",", ":"), ensure_ascii=False))
    return math.ceil(chars / CHARS_PER_TOKEN) + TOKENS_PER_MESSAGE * len(messages)


class Ledger:
    """A run's budget and what has been spent against it, shared by every client it wraps.

    Thread-safe: the extractor's pool and the grounder's draw on one ledger.
    `stopped` is the first `BudgetExceeded`, or None while the run may still
    call.
    """

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget if budget is not None else Budget()
        self._cond = threading.Condition()
        self._calls = self._input = self._output = self._unpriced = 0
        self._usd = 0.0
        # Tokens and USD in calls that priced their cost, for the USD estimate.
        self._priced_tokens = 0
        self._priced_for_rate = 0.0
        # What the calls in flight may still spend.
        self._held_calls = self._held_input = self._held_output = 0
        self._held_usd = 0.0
        # A limit already reached by a call checked after the fact (USD with no estimate).
        self._reached: str | None = None
        self.stopped: BudgetExceeded | None = None

    def client(self, inner: LLMClient) -> BudgetedClient:
        """`inner`, with every call it makes checked and counted here."""
        return BudgetedClient(inner, self)

    @property
    def spent(self) -> Spent:
        with self._cond:
            return self._snapshot()

    def estimate(
        self, messages: Sequence[Message], spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Estimate:
        input_tokens = estimate_tokens(messages, schema)
        output_tokens = spec.max_tokens
        with self._cond:
            rate = self._priced_for_rate / self._priced_tokens if self._priced_tokens else None
        usd = None if rate is None else rate * (input_tokens + output_tokens)
        return Estimate(input_tokens, output_tokens, usd)

    def reserve(self, estimate: Estimate) -> None:
        """Hold `estimate` for a call about to be made, or raise `BudgetExceeded`."""
        with self._cond:
            while True:
                if self.stopped is not None:
                    raise self._exceeded(self.stopped.limit)
                over = self._reached or self._over(estimate, held=True)
                if over is None:
                    self._hold(estimate, 1)
                    return
                if (
                    self._reached is None
                    and self._held_calls
                    and self._over(estimate, held=False) is None
                ):
                    # Only the calls in flight stand in the way: they may spend
                    # less than they hold, so wait for them and look again.
                    self._cond.wait()
                    continue
                self.stopped = self._exceeded(over)
                raise self._exceeded(over)

    def settle(self, estimate: Estimate, completion: Completion) -> None:
        """Count a call that answered, and release what it held."""
        with self._cond:
            self._hold(estimate, -1)
            if completion.cached:
                self._cond.notify_all()
                return
            self._calls += 1
            self._input += completion.prompt_tokens
            self._output += completion.completion_tokens
            if completion.cost_usd is None:
                self._unpriced += 1
            else:
                self._usd += completion.cost_usd
                tokens = completion.prompt_tokens + completion.completion_tokens
                if tokens:
                    self._priced_tokens += tokens
                    self._priced_for_rate += completion.cost_usd
            limit = self.budget.usd
            if limit is not None and self._usd >= limit:
                # Checked after the call too: before the first priced call there
                # is nothing to estimate with, and an estimate can fall short.
                self._reached = "usd"
            self._cond.notify_all()

    def release(self, estimate: Estimate) -> None:
        """Release what a call held when it raised: it is not counted."""
        with self._cond:
            self._hold(estimate, -1)
            self._cond.notify_all()

    def _hold(self, estimate: Estimate, sign: int) -> None:
        self._held_calls += sign
        self._held_input += sign * estimate.input_tokens
        self._held_output += sign * estimate.output_tokens
        self._held_usd += sign * (estimate.usd or 0.0)

    def _over(self, estimate: Estimate, *, held: bool) -> str | None:
        """The first limit this call could take the run past, or None when it fits.

        `held=False` leaves out what the calls in flight hold: the question is
        then whether the call could ever fit, once they have settled.
        """
        k = 1 if held else 0
        # (limit, spent, held by calls in flight, this call's estimate)
        checks = (
            ("usd", self._usd, self._held_usd, estimate.usd),
            ("calls", self._calls, self._held_calls, 1),
            ("input_tokens", self._input, self._held_input, estimate.input_tokens),
            ("output_tokens", self._output, self._held_output, estimate.output_tokens),
        )
        for name, spent, holding, wanted in checks:
            limit = getattr(self.budget, name)
            # USD with no estimate yet is checked after the call (`settle`).
            if limit is not None and wanted is not None and spent + k * holding + wanted > limit:
                return name
        return None

    def _snapshot(self) -> Spent:
        return Spent(
            calls=self._calls,
            input_tokens=self._input,
            output_tokens=self._output,
            priced_usd=self._usd,
            unpriced_calls=self._unpriced,
        )

    def _exceeded(self, limit: str) -> BudgetExceeded:
        # A fresh exception for each refused call: one instance raised from many
        # threads would share, and garble, one traceback.
        return BudgetExceeded(limit, budget=self.budget, spent=self._snapshot(), ledger=self)


class BudgetedClient:
    """An `LLMClient` that asks its ledger before every call and tells it after."""

    def __init__(self, inner: LLMClient, ledger: Ledger) -> None:
        self.inner = inner
        self.ledger = ledger

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        estimate = self.ledger.estimate(messages, spec, schema)
        self.ledger.reserve(estimate)
        try:
            completion = self.inner.complete(messages, spec=spec, schema=schema)
        except BaseException:
            self.ledger.release(estimate)
            raise
        self.ledger.settle(estimate, completion)
        return completion


__all__ = [
    "CHARS_PER_TOKEN",
    "LIMITS",
    "TOKENS_PER_MESSAGE",
    "Budget",
    "BudgetExceeded",
    "BudgetedClient",
    "Estimate",
    "Ledger",
    "Spent",
    "budget_summary",
    "describe",
    "estimate_tokens",
    "stopped_summary",
]
