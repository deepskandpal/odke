"""Watching a run: one JSON object per event, per-job counts, and OpenTelemetry spans (#161).

A report at the end says what a run did. A batch job in production also has
to say, while it runs and to a machine, which stage it is in, which document
failed, what each model call cost and how long it took. So a job emits events,
and every event is one record on the `openodke.events` logger:

- `job.start` and `job.end`, once each; `job.end` carries the job's counts:
  facts `in` and `out`, `refused`, `merged`, `linked`, and sent to `review`
  (`JobCounts`), the same numbers the report prints;
- `stage`, once per pipeline stage, with that stage's counts, its latency and
  what its model calls cost;
- `document`, once per document grounded, with its chunks, facts and
  verdicts, and `document.failed`, once per document left out, with why;
- `model.call`, once per model call, with its stage, the model asked and the
  one that served, its latency, tokens and cost, and whether a cache answered.

Every event has the same keys: `ts`, `level`, `event`, `run` (an id shared by
one job's events), `command`, `stage`, `document`, `counts`, `latency_s` and
`cost`, null where they do not apply, and a few of its own. Nothing is
written anywhere until a handler is installed: `configure_logs("json")`, or
`--log-format json` on `odke run`, `odke validate` and `odke ground`, writes
one JSON object a line to standard error.

**No text.** An event holds ids, names, counts, times and costs: never a
passage, a quote, an entity's name, a prompt or a reply, and never a secret.
The free-form warnings the stages log can quote a model's reply, so as JSON
they keep their template and drop its arguments, unless `text=True`
(`--log-text`) says otherwise.

**OpenTelemetry**, behind the `otel` extra and imported lazily: with
`opentelemetry-api` installed, each job is a span, each stage a child of it and
each model call a child of its stage, with the events' counts and costs as
attributes. They go to the tracer provider the application configured, so
with none configured, or without the extra, they cost next to nothing and go
nowhere. `TRACER_PROVIDER` overrides the global one.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO, Any, Literal, NamedTuple
from uuid import uuid4

from openodke.llm.base import Completion, LLMClient, Message, ModelSpec
from openodke.types import Frozen

EVENTS = logging.getLogger("openodke.events")
# What `configure_logs` installs its handler on: every logger the package has.
ROOT = logging.getLogger("openodke")
# The tracer provider spans go to; None is OpenTelemetry's global one.
TRACER_PROVIDER: Any = None

# The keys every event has, null where they do not apply, in this order.
KEYS = ("event", "run", "command", "stage", "document", "counts", "latency_s", "cost")


class JobCounts(Frozen):
    """One job's facts: in, out, refused, merged, linked, and sent to a person to review.

    `facts_in` is what the job was handed or extracted, and `facts_out` what
    it wrote. `refused` is what its gate refused; `odke ground` drops nothing,
    and counts what the free checks refused. `merged` is what was folded into
    another fact with the same signature, `linked` the entity links proposed,
    and `review` the pairs the pair judge queued for a person. Each is the
    number the job's report prints.
    """

    facts_in: int = 0
    facts_out: int = 0
    refused: int = 0
    merged: int = 0
    linked: int = 0
    review: int = 0


def spend(stats: Mapping[str, Any]) -> dict[str, Any]:
    """A stage's or a run's model calls, tokens and cost, by the names the events use."""
    usd = stats.get("cost_usd", stats.get("usd"))
    return {
        "calls": int(stats.get("calls", 0)),
        "cached_calls": int(stats.get("cached", stats.get("cached_calls", 0))),
        "input_tokens": int(stats.get("prompt_tokens", stats.get("input_tokens", 0))),
        "output_tokens": int(stats.get("completion_tokens", stats.get("output_tokens", 0))),
        "usd": float(usd) if isinstance(usd, int | float) else None,
    }


def job_counts_of_run(stats: Mapping[str, Any]) -> JobCounts:
    """An `odke run`'s counts, from the graph's stats as its report reads them."""
    graph = stats.get("graph") or {}
    facts_in, facts_out = int(stats.get("candidates", 0)), int(graph.get("facts", 0))
    refused, derived = int(stats.get("refused", 0)), int(stats.get("derived", 0))
    resolver = (stats.get("stages") or {}).get("resolver") or {}
    judge = resolver.get("judge") if isinstance(resolver, Mapping) else None
    return JobCounts(
        facts_in=facts_in,
        facts_out=facts_out,
        refused=refused,
        merged=facts_in + derived - refused - facts_out,
        linked=sum(int(v) for v in (graph.get("links") or {}).values()),
        review=int(judge.get("queued", 0)) if isinstance(judge, Mapping) else 0,
    )


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def event(
    name: str,
    *,
    run: str | None = None,
    command: str | None = None,
    stage: str | None = None,
    document: str | None = None,
    counts: Mapping[str, Any] | None = None,
    latency_s: float | None = None,
    cost: Mapping[str, Any] | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Log one event on `openodke.events`. A handler with `JsonFormatter` writes it as one line."""
    if not EVENTS.isEnabledFor(level):
        return
    data: dict[str, Any] = {
        "event": name,
        "run": run,
        "command": command,
        "stage": stage,
        "document": document,
        "counts": dict(counts) if counts is not None else None,
        "latency_s": round(latency_s, 6) if latency_s is not None else None,
        "cost": dict(cost) if cost is not None else None,
        **fields,
    }
    EVENTS.log(level, name, extra={"odke_event": data})


class JsonFormatter(logging.Formatter):
    """One JSON object a record: an event as its fields, any other record by its template.

    Another record's arguments are left out, because a stage's warning can
    quote a passage or a model's reply; `text=True` puts the formatted
    message back.
    """

    def __init__(self, *, text: bool = False) -> None:
        super().__init__()
        self.text = text

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        head = {"ts": stamp.replace("+00:00", "Z"), "level": record.levelname.lower()}
        found = getattr(record, "odke_event", None)
        if isinstance(found, Mapping):
            return _dumps({**head, **found})
        data: dict[str, Any] = {**head, **dict.fromkeys(KEYS)}
        data.update(event="log", logger=record.name, template=str(record.msg))
        if self.text:
            data["message"] = record.getMessage()
        return _dumps(data)


def _dumps(data: Mapping[str, Any]) -> str:
    return json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":"))


# The handler `configure_logs` installed, and the level it found.
_INSTALLED: list[tuple[logging.Handler, int]] = []


def configure_logs(
    format: Literal["json", "text"] = "json",
    *,
    stream: IO[str] | None = None,
    text: bool = False,
    level: int = logging.INFO,
) -> logging.Handler | None:
    """Write the package's logs as JSON, one object a line, or put them back as they were.

    `json` installs a handler on the `openodke` logger that writes to `stream`
    (standard error when None) and stops the records reaching the root logger,
    so nothing is printed twice. `text` removes it. Installing again replaces
    the one installed before. Returns the handler, or None for `text`.
    """
    if format not in ("json", "text"):
        raise ValueError(f"log format is json or text, not {format!r}")
    for handler, found in _INSTALLED:
        ROOT.removeHandler(handler)
        ROOT.setLevel(found)
    _INSTALLED.clear()
    ROOT.propagate = True
    if format == "text":
        return None
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(text=text))
    _INSTALLED.append((handler, ROOT.level))
    ROOT.addHandler(handler)
    ROOT.setLevel(level)
    ROOT.propagate = False
    return handler


# --------------------------------------------------------------------------- #
# One job
# --------------------------------------------------------------------------- #


class Observer:
    """One job's events and spans: its id, the stage now running, and what the calls cost.

    `start()` and `finish(counts)` open and close the job. `stage(name)` is a
    context manager around one stage; the mapping it yields is the stage's
    counts, filled in as it runs. `client(inner, stage)` wraps a model client
    so each call is an event and a span, under the stage running when it is
    made. Thread-safe: a stage's calls come from its thread pool.
    """

    def __init__(self, command: str = "pipeline", *, run: str | None = None) -> None:
        self.command = command
        self.run = run if run is not None else uuid4().hex[:12]
        self._spent = _Spend()
        self._tracer = _tracer()
        self._job: Any = None
        self._job_context: Any = None
        self._stage_context: Any = None
        self._started = 0.0

    # The job ------------------------------------------------------------- #

    def start(self, **fields: Any) -> None:
        self._started = time.perf_counter()
        self._job = self._span(f"odke {self.command}", None, {"odke.run": self.run, **fields})
        self._job_context = _context(self._job)
        event("job.start", run=self.run, command=self.command, **fields)

    def finish(
        self, counts: JobCounts, *, cost: Mapping[str, Any] | None = None, **fields: Any
    ) -> None:
        """The job ends: its counts, and `cost`, what its report says it spent.

        Left out, the cost is what the calls this observer saw cost; a stage
        with a client of its own, a pair judge say, is in the report's.
        """
        latency = time.perf_counter() - self._started if self._started else None
        cost = dict(cost) if cost is not None else self.spent()
        event(
            "job.end",
            run=self.run,
            command=self.command,
            counts=counts.model_dump(),
            latency_s=latency,
            cost=cost,
            **fields,
        )
        _end(self._job, {**_attributes("counts", counts.model_dump()), **_cost_attributes(cost)})

    # Stages -------------------------------------------------------------- #

    def begin(self, name: str) -> Stage:
        """A stage starts: its span opens, and the model calls from now on are under it."""
        span = self._span(f"stage {name}", self._job_context, {"odke.stage": name})
        stage = Stage(name, span, self._spent.copy(), self._stage_context, time.perf_counter())
        self._stage_context = _context(span)
        return stage

    def end(self, stage: Stage, counts: Mapping[str, Any] | None = None) -> None:
        """A stage ends: its event, with `counts`, its latency and what its calls cost."""
        self._stage_context = stage.outer
        latency = time.perf_counter() - stage.started
        cost = self._spent.since(stage.spent)
        found = dict(counts or {})
        event(
            "stage",
            run=self.run,
            command=self.command,
            stage=stage.name,
            counts=found,
            latency_s=latency,
            cost=cost,
        )
        _end(stage.span, {**_attributes("counts", found), **_cost_attributes(cost)})

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """`begin` and `end` around a block: yields the counts, to fill in as it runs."""
        counts: dict[str, Any] = {}
        begun = self.begin(name)
        try:
            yield counts
        finally:
            self.end(begun, counts)

    def document(self, stage: str, document: str, counts: Mapping[str, Any]) -> None:
        event(
            "document",
            run=self.run,
            command=self.command,
            stage=stage,
            document=document,
            counts=counts,
        )

    def failed(self, stage: str, document: str, reason: str) -> None:
        event(
            "document.failed",
            level=logging.WARNING,
            run=self.run,
            command=self.command,
            stage=stage,
            document=document,
            reason=reason,
        )

    # Model calls --------------------------------------------------------- #

    def client(self, inner: LLMClient, stage: str) -> LLMClient:
        """`inner`, with every call an event and a span under the stage running when it is made."""
        return ObservedClient(inner, self, stage)

    def spent(self) -> dict[str, Any]:
        """What the calls seen so far cost: calls, cached calls, tokens, and USD or None."""
        return self._spent.as_dict()

    def _called(
        self,
        stage: str,
        spec: ModelSpec,
        completion: Completion | None,
        latency: float,
        span: Any,
        error: BaseException | None = None,
    ) -> None:
        model = _qualified(spec.model, spec.provider)
        cost: dict[str, Any] | None = None
        extra: dict[str, Any] = {"model": model, "served": None}
        if completion is not None:
            self._spent.add(completion)
            cost = {
                "input_tokens": completion.prompt_tokens,
                "output_tokens": completion.completion_tokens,
                "usd": completion.cost_usd,
                "cached": completion.cached,
            }
            if completion.model:
                extra["served"] = _qualified(completion.model, spec.provider)
        if error is not None:
            extra["error"] = type(error).__name__
        event(
            "model.call",
            level=logging.INFO if error is None else logging.WARNING,
            run=self.run,
            command=self.command,
            stage=stage,
            latency_s=latency,
            cost=cost,
            **extra,
        )
        attributes = {"odke.stage": stage, "odke.model": model, "odke.latency_s": latency}
        if cost is not None:
            attributes.update(_cost_attributes(cost))
        _end(span, attributes, error=error)

    def _span(self, name: str, parent: Any, attributes: Mapping[str, Any]) -> Any:
        if self._tracer is None:
            return None
        return self._tracer.start_span(name, context=parent, attributes=dict(attributes))


class Stage(NamedTuple):
    """A stage begun: its name and span, the spend and the stage before it, and its clock."""

    name: str
    span: Any
    spent: _Spend
    outer: Any
    started: float


def _qualified(model: str, provider: str) -> str:
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


class ObservedClient:
    """An `LLMClient` that tells its observer what each call was, took and cost."""

    def __init__(self, inner: LLMClient, observer: Observer, stage: str) -> None:
        self.inner = inner
        self.observer = observer
        self.stage = stage

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        observer = self.observer
        parent = observer._stage_context or observer._job_context
        span = observer._span(f"model {self.stage}", parent, {"odke.stage": self.stage})
        started = time.perf_counter()
        try:
            completion = self.inner.complete(messages, spec=spec, schema=schema)
        except BaseException as exc:
            latency = time.perf_counter() - started
            observer._called(self.stage, spec, None, latency, span, error=exc)
            raise
        observer._called(self.stage, spec, completion, time.perf_counter() - started, span)
        return completion


class _Spend:
    """Calls, cached calls, tokens and USD, added up under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = self.cached = self.input = self.output = self.unpriced = 0
        self.usd = 0.0

    def add(self, completion: Completion) -> None:
        with self._lock:
            self.calls += 1
            self.cached += completion.cached
            self.input += completion.prompt_tokens
            self.output += completion.completion_tokens
            if completion.cost_usd is None:
                self.unpriced += 1
            else:
                self.usd += completion.cost_usd

    def copy(self) -> _Spend:
        out = _Spend()
        with self._lock:
            for name in ("calls", "cached", "input", "output", "unpriced", "usd"):
                setattr(out, name, getattr(self, name))
        return out

    def since(self, before: _Spend) -> dict[str, Any]:
        now = self.copy()
        for name in ("calls", "cached", "input", "output", "unpriced", "usd"):
            setattr(now, name, getattr(now, name) - getattr(before, name))
        return now.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached,
            "input_tokens": self.input,
            "output_tokens": self.output,
            "usd": None if self.unpriced else round(self.usd, 10),
        }


# --------------------------------------------------------------------------- #
# OpenTelemetry, imported on first use
# --------------------------------------------------------------------------- #


def _tracer() -> Any:
    """The OpenTelemetry tracer, or None without `opentelemetry-api` (the `otel` extra)."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    from openodke import __version__

    return trace.get_tracer("openodke", __version__, tracer_provider=TRACER_PROVIDER)


def _context(span: Any) -> Any:
    if span is None:
        return None
    from opentelemetry import trace

    return trace.set_span_in_context(span)


def _end(span: Any, attributes: Mapping[str, Any], error: BaseException | None = None) -> None:
    if span is None:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
    if error is not None:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, type(error).__name__))
    span.end()


def _attributes(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Counts as span attributes: `odke.counts.facts`, numbers and strings only."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, bool | int | float | str):
            out[f"odke.{prefix}.{key}"] = value
    return out


def _cost_attributes(cost: Mapping[str, Any]) -> dict[str, Any]:
    return {f"odke.cost.{key}": value for key, value in cost.items() if value is not None}


__all__ = [
    "EVENTS",
    "KEYS",
    "TRACER_PROVIDER",
    "JobCounts",
    "JsonFormatter",
    "ObservedClient",
    "Observer",
    "Stage",
    "configure_logs",
    "event",
    "job_counts_of_run",
    "spend",
]
