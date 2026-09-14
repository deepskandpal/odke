"""The second model's question: does that text support this claim?

Locatability is not verification. A model can cite a span that genuinely exists
and does not support the fact it was attached to, and only a second reading
catches that. This is where the paper's precision comes from — one fact, one
span, one verdict from a small model (DECISIONS #7a: the `ground` role defaults
to a cheaper model than `extract`, because if this pass cost what extraction
costs, people would turn it off).

The span check runs first, inside this grounder, every time. The model is only
shown a span that really resolves in the document, and is never asked about a
fact the free check already settled. It sets `Fact.verdict` and nothing else:
`confidence` belongs to the scorer.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from odke.ground.retry import RetryPolicy, call_with_retry
from odke.ground.span import Counts, SpanGrounder, located
from odke.llm.base import Completion, LLMClient, Message, ModelSpec
from odke.llm.roles import ModelRoles
from odke.types import Document, Entity, Fact, GroundingVerdict, Polarity

log = logging.getLogger("odke.ground")

# The three answers the model may give. UNCHECKED is what a fact has before it
# is asked, or after a call that failed; it is never an answer.
VERDICTS = (GroundingVerdict.SUPPORTED, GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND)

# Short on purpose: this is sent once per fact, and the cost story of the whole
# stage rests on it being a fraction of the extraction prompt.
SYSTEM_PROMPT = (
    "You check one knowledge-graph claim against one passage from its source. "
    "Judge from the passage alone; ignore anything you know from elsewhere.\n"
    'Answer "supported" if the passage states the claim or clearly entails it, '
    '"contradicted" if the passage states the opposite or an incompatible value, '
    'and "not_found" if the passage does not settle it either way.\n'
    'Reply with JSON only: {"verdict": "supported" | "contradicted" | "not_found"}'
)

GROUNDING_SCHEMA: dict[str, Any] = {
    "title": "GroundingVerdict",
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": [v.value for v in VERDICTS]}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def render_claim(fact: Fact) -> str:
    """The fact as one line a small model can judge against a passage.

    Stable across runs — the same fact renders identically — so provider-side
    prompt caching works and a run is repeatable. Only identity-bearing
    qualifiers appear: a reconcilable one (`start_time`) is a view of the same
    claim rather than part of it (DECISIONS #11, #15), and the model is being
    asked about the claim. Polarity is spelled out because "X sells data" and
    "X does not sell data" are opposite claims and the same triple.
    """
    subject = _mention(fact.subject)
    predicate = fact.predicate.replace("_", " ")
    obj = (
        _mention(fact.object_entity)
        if fact.object_entity is not None
        else _literal(fact.object_value)
    )
    claim = f"{subject} — {predicate} — {obj}"
    scoped = [
        f"{key} = {_literal(fact.qualifiers[key])}"
        for key in fact.identity_keys
        if key in fact.qualifiers
    ]
    if scoped:
        claim += f" ({', '.join(scoped)})"
    if fact.polarity is Polarity.DENIED:
        return f"It is NOT the case that: {claim}."
    if fact.polarity is Polarity.PARTIAL:
        return f"Only partially, or under a stated condition: {claim}."
    return f"{claim}."


def build_messages(fact: Fact, passage: str) -> list[Message]:
    """Exactly what the model is sent: the claim and the cited passage, nothing more.

    Not the whole document. The question is whether *this span* supports the
    claim; showing more would answer a different, dearer question.
    """
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=f"Claim: {render_claim(fact)}\n\nPassage:\n{passage}"),
    ]


def parse_verdict(completion: Completion) -> GroundingVerdict | None:
    """The verdict in a completion, or None when there is not exactly one to read.

    Structured output is the normal path. A model that ignored the schema and
    answered with the bare word is still readable; one that wrote a sentence
    is not — "not supported" must never be read as `SUPPORTED`, so prose is a
    failed answer rather than a guess.
    """
    value: object = completion.parsed.get("verdict") if completion.parsed else None
    if not isinstance(value, str):
        match = _BARE_WORD.match(completion.text)
        value = match.group(1) if match else None
    if not isinstance(value, str):
        return None
    normalised = re.sub(r"[\s-]+", "_", value.strip().lower())
    return next((v for v in VERDICTS if v.value == normalised), None)


_BARE_WORD = re.compile(r"^\W*(supported|contradicted|not[\s_-]?found)\W*$", re.IGNORECASE)


def _mention(entity: Entity) -> str:
    return f"{entity.label or entity.key} ({entity.type})"


def _literal(value: object) -> str:
    # JSON spelling: strings arrive quoted, so their boundary is unambiguous,
    # and numbers, booleans and null read the same way in every run.
    return json.dumps(value, default=str, ensure_ascii=False)


class LLMGrounder:
    """One fact, one span, one verdict — after the free check has had its say.

    `roles.ground` names the model; `client` overrides how it is reached, which
    is how the suite replays recorded answers. Everything the stage did is in
    `stats`: calls made and retried, calls that failed, each verdict's count,
    answers that could not be read, tokens, and the span check's own counts
    under `"span"`.

    A call is retried on transient errors under `retry`; when it still fails, or
    fails in a way retrying cannot fix, the fact is left `UNCHECKED` and logged.
    `ground` never raises for a provider failure — one bad call must not lose a
    run of ten thousand — and an `UNCHECKED` fact is exactly what the next run
    picks up, because a verdict already on a fact stands and costs no call.

    `ground_many` is the batched path the pipeline uses when it is there: a
    document's facts at once, with at most `max_workers` model calls in flight.
    Both clients are synchronous and the wait is network I/O, so a thread pool
    is the right tool and needs nothing outside the standard library.
    """

    def __init__(
        self,
        roles: ModelRoles | None = None,
        *,
        client: LLMClient | None = None,
        max_workers: int = 8,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.roles = roles if roles is not None else ModelRoles()
        self.spec: ModelSpec = self.roles.ground
        self.client: LLMClient = client if client is not None else self.roles.client_for("ground")
        self.max_workers = max_workers
        self.retry = retry if retry is not None else RetryPolicy()
        self.span_grounder = SpanGrounder()
        self._sleep = sleep
        # The ceiling lives on the call, not the pool, so it also holds for a
        # caller who runs several `ground_many`s — or plain `ground`s — at once.
        self._slots = threading.BoundedSemaphore(max_workers)
        self._counts = Counts(
            "facts",
            "skipped",
            "calls",
            "retries",
            "failed",
            "unparseable",
            "prompt_tokens",
            "completion_tokens",
            *(v.value for v in VERDICTS),
        )
        self._cost_lock = threading.Lock()
        self._cost: float | None = None

    def ground(self, fact: Fact, doc: Document) -> Fact:
        prepared, passage = self._prepare(fact, doc)
        return prepared if passage is None else self._ask(prepared, passage)

    def ground_many(self, facts: Sequence[Fact], doc: Document) -> list[Fact]:
        """One document's facts, with the model calls run concurrently.

        The span checks run first, here and in order, and only the facts that
        survive them are handed to the pool. One fact comes back for each that
        went in, in the same order: a failed call leaves its own fact
        `UNCHECKED` and the rest of the batch carries on. A client used here
        must be thread-safe — both built-in clients are; `ScriptedClient`
        answers by position and is not, which is what `RecordedClient` is for.
        """
        out: list[Fact] = []
        pending: list[tuple[int, Fact, str]] = []
        for index, fact in enumerate(facts):
            prepared, passage = self._prepare(fact, doc)
            out.append(prepared)
            if passage is not None:
                pending.append((index, prepared, passage))
        if len(pending) == 1:
            index, fact, passage = pending[0]
            out[index] = self._ask(fact, passage)
        elif pending:
            workers = min(self.max_workers, len(pending))
            with ThreadPoolExecutor(workers, thread_name_prefix="odke-ground") as pool:
                futures = [(i, pool.submit(self._ask, f, p)) for i, f, p in pending]
                for index, future in futures:
                    out[index] = future.result()
        return out

    def _prepare(self, fact: Fact, doc: Document) -> tuple[Fact, str | None]:
        """The fact after the free check, and the passage to ask about — or None
        when there is nothing left to ask."""
        if fact.verdict is not GroundingVerdict.UNCHECKED:
            self._counts.bump("skipped")
            return fact, None
        self._counts.bump("facts")
        checked = self.span_grounder.ground(fact, doc)
        if checked.verdict is not GroundingVerdict.UNCHECKED:
            return checked, None
        evidence = located(checked, doc)
        # The span check leaves a fact UNCHECKED only when something located it,
        # so this guard is for the type checker, not a path that runs.
        if evidence is None or evidence.span is None:  # pragma: no cover
            return checked, None
        return checked, evidence.span.resolve(doc)

    def _ask(self, fact: Fact, passage: str) -> Fact:
        messages = build_messages(fact, passage)

        def on_retry(attempt: int, exc: BaseException, wait: float) -> None:
            self._counts.bump("retries")
            log.info(
                "retrying grounding call for fact %s in %.2fs after attempt %d: %s",
                fact.id,
                wait,
                attempt,
                exc,
            )

        try:
            completion = call_with_retry(
                lambda: self._complete(messages),
                self.retry,
                sleep=self._sleep,
                on_retry=on_retry,
            )
        except Exception as exc:  # isolation: one failed call never fails the batch
            self._counts.bump("failed")
            log.warning("grounding call failed for fact %s, left unchecked: %s", fact.id, exc)
            return fact
        self._account(completion)
        verdict = parse_verdict(completion)
        if verdict is None:
            self._counts.bump("unparseable")
            log.warning(
                "unreadable grounding answer for fact %s: %r", fact.id, completion.text[:200]
            )
            return fact
        self._counts.bump(verdict.value)
        return fact.model_copy(update={"verdict": verdict})

    def _complete(self, messages: list[Message]) -> Completion:
        with self._slots:
            self._counts.bump("calls")
            return self.client.complete(messages, spec=self.spec, schema=GROUNDING_SCHEMA)

    def _account(self, completion: Completion) -> None:
        self._counts.bump("prompt_tokens", completion.prompt_tokens)
        self._counts.bump("completion_tokens", completion.completion_tokens)
        if completion.cost_usd is not None:
            with self._cost_lock:
                self._cost = (self._cost or 0.0) + completion.cost_usd

    @property
    def stats(self) -> dict[str, Any]:
        """The stage's run report. `cost_usd` is None until a provider reports one."""
        out: dict[str, Any] = self._counts.snapshot()
        with self._cost_lock:
            out["cost_usd"] = self._cost
        out["span"] = self.span_grounder.stats
        return out


__all__ = [
    "GROUNDING_SCHEMA",
    "SYSTEM_PROMPT",
    "VERDICTS",
    "LLMGrounder",
    "build_messages",
    "parse_verdict",
    "render_claim",
]
