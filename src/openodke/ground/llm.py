"""The second model's question: does that text support this claim?

Locatability is not verification. A model can cite a span that genuinely exists
and does not support the fact it was attached to, and only a second reading
catches that. One fact, one verdict from a small model (DECISIONS #7a: the
`ground` role defaults to a cheaper model than `extract`, because if this pass
cost what extraction costs, people would turn it off).

By default the model sees the cited span and answers in three ways. The paper's
own grounder (ODKE+ §3.3.1, App. B) is different: it sees the whole context and
answers True or False, and only affirmed facts are kept. `context="document"`,
`verdicts="binary"` and `VerdictGate(refuse_not_found=True)` together
reproduce it, so the paper's claims can be tested as written.

The span check runs first, inside this grounder, every time. The model is only
asked about a fact whose citation really resolves in the document, and never
about one the free check already settled. `locate=True` then gives a fact that
cited nothing the window naming its subject and object (`SpanLocator`), so the
model reads that window rather than the whole text. Apart from that span, the
grounder sets `Fact.verdict` and nothing else: `confidence` belongs to the scorer.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from openodke.ground.locate import SpanLocator
from openodke.ground.retry import RetryPolicy, call_with_retry
from openodke.ground.span import Counts, SpanGrounder, located
from openodke.ground.widen import WIDEN, widen
from openodke.llm.base import (
    Completion,
    LLMClient,
    Message,
    MissingAPIKey,
    ModelSpec,
    ProviderNotInstalled,
)
from openodke.llm.roles import ModelRoles
from openodke.prompts import Prompt
from openodke.prompts import get as get_prompt
from openodke.types import Document, Entity, Fact, GroundingVerdict, Polarity

log = logging.getLogger("openodke.ground")

# The three answers the model may give. UNCHECKED is what a fact has before it
# is asked, or after a call that failed; it is never an answer.
VERDICTS = (GroundingVerdict.SUPPORTED, GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND)

# The two prompts this grounder can send, from the registry (DECISIONS #27):
# `ground.span` by default, and the paper's `ground.paper` (ODKE+ App. B) when
# `verdicts="binary"`. The latest version of each is what is sent, and its key
# is what `stats["prompts"]` names.
_SPAN = get_prompt("ground.span")
_PAPER = get_prompt("ground.paper")
# The texts under their old names, for anything that imports them.
SYSTEM_PROMPT = _SPAN.text
PAPER_PROMPT = _PAPER.text

GROUNDING_SCHEMA: dict[str, Any] = {
    "title": "GroundingVerdict",
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": [v.value for v in VERDICTS]}},
    "required": ["verdict"],
    "additionalProperties": False,
}

# The paper's prompt asks for True or False. "False" is read as NOT_FOUND: the
# paper's "No" merges a context that contradicts the triple with one that is
# silent about it, so a binary run cannot tell them apart.
BINARY_SCHEMA: dict[str, Any] = {
    "title": "GroundingJudgement",
    "type": "object",
    "properties": {"verdict": {"type": "boolean"}},
    "required": ["verdict"],
    "additionalProperties": False,
}

Context = Literal["span", "document"]
Verdicts = Literal["three_way", "binary"]


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


def render_triple(fact: Fact) -> str:
    """The fact in the paper's grounder shape: `<subject, predicate(qualifier: value), object>`.

    Labels rather than typed mentions and bare strings rather than JSON, as in the
    paper's worked example (`<Felton Ross, Date of Birth, May 9, 1927>`). The paper
    has no polarity; a denial is spelled into the predicate so it is not asked as
    an assertion.
    """
    subject = fact.subject.label or fact.subject.key
    predicate = fact.predicate.replace("_", " ")
    scoped = [
        f"{key}: {_bare(fact.qualifiers[key])}"
        for key in fact.identity_keys
        if key in fact.qualifiers
    ]
    if scoped:
        predicate += f"({', '.join(scoped)})"
    if fact.polarity is Polarity.DENIED:
        predicate = f"not {predicate}"
    elif fact.polarity is Polarity.PARTIAL:
        predicate = f"partially {predicate}"
    obj = (
        fact.object_entity.label or fact.object_entity.key
        if fact.object_entity is not None
        else _bare(fact.object_value)
    )
    return f"<{subject}, {predicate}, {obj}>"


def build_messages(fact: Fact, passage: str, *, binary: bool = False) -> list[Message]:
    """Exactly what the model is sent: the claim and the passage, nothing more.

    By default the passage is the cited span — the question is whether *this
    span* supports the claim. `binary=True` sends the paper's prompt and triple
    instead; which passage it is asked against is the grounder's `context`.
    """
    if binary:
        return [
            Message(role="system", content=_PAPER.text),
            Message(
                role="user", content=f"**Context:\n{passage}\n**triple:\n{render_triple(fact)}"
            ),
        ]
    return [
        Message(role="system", content=_SPAN.text),
        Message(role="user", content=f"Claim: {render_claim(fact)}\n\nPassage:\n{passage}"),
    ]


def parse_verdict(completion: Completion, *, binary: bool = False) -> GroundingVerdict | None:
    """The verdict in a completion, or None when there is not exactly one to read.

    Structured output is the normal path. A model that ignored the schema and
    answered with the bare word is still readable; one that wrote a sentence
    is not — "not supported" must never be read as `SUPPORTED`, so prose is a
    failed answer rather than a guess.
    """
    value: object = completion.parsed.get("verdict") if completion.parsed else None
    if binary:
        return _parse_binary(value, completion.text)
    if not isinstance(value, str):
        match = _BARE_WORD.match(completion.text)
        value = match.group(1) if match else None
    if not isinstance(value, str):
        return None
    # Separators dropped, not swapped: `NotFound`, `not-found` and `not found` are
    # all spellings `_BARE_WORD` accepts, and must all read as the same verdict.
    normalised = re.sub(r"[\s_-]+", "", value.strip().lower())
    return next((v for v in VERDICTS if v.value.replace("_", "") == normalised), None)


_BARE_WORD = re.compile(r"^\W*(supported|contradicted|not[\s_-]?found)\W*$", re.IGNORECASE)
_BARE_BOOL = re.compile(r"^\W*(true|false|yes|no)\W*$", re.IGNORECASE)


def _parse_binary(value: object, text: str) -> GroundingVerdict | None:
    """True is SUPPORTED, False is NOT_FOUND; anything else is unreadable."""
    if not isinstance(value, bool):
        word = value if isinstance(value, str) else None
        match = _BARE_BOOL.match(word if word is not None else text)
        if match is None:
            return None
        value = match.group(1).lower() in ("true", "yes")
    return GroundingVerdict.SUPPORTED if value else GroundingVerdict.NOT_FOUND


def _mention(entity: Entity) -> str:
    return f"{entity.label or entity.key} ({entity.type})"


def _bare(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)


def _literal(value: object) -> str:
    # JSON spelling: strings arrive quoted, so their boundary is unambiguous,
    # and numbers, booleans and null read the same way in every run.
    return json.dumps(value, default=str, ensure_ascii=False)


class LLMGrounder:
    """One fact, one span, one verdict — after the free check has had its say.

    `roles.ground` names the model; `client` overrides how it is reached, which
    is how the suite replays recorded answers. Everything the stage did is in
    `stats`: calls made and retried, calls that failed, each verdict's count,
    answers that could not be read, tokens, the key of the registered prompt the
    calls sent under `"prompts"`, and the span check's own counts under `"span"`.

    A call is retried on transient errors under `retry`; when it still fails, or
    fails in a way retrying cannot fix, the fact is left `UNCHECKED` and logged.
    `ground` never raises for a provider failure — one bad call must not lose a
    run of ten thousand — and an `UNCHECKED` fact is exactly what the next run
    picks up, because a verdict already on a fact stands and costs no call. A
    missing key or adapter is not a provider failure but configuration: it fails
    every call alike, so it raises instead of leaving the whole run `UNCHECKED`.

    `locate=True` runs `SpanLocator` after the span check: a fact whose span is
    the whole text it came from (`SpanOrigin.CONTEXT`) is asked about the one
    or two sentences naming its subject and object instead, when they exist.
    Its counts are under `"locate"`.

    `widen=True` gives a `not_found` whose span is narrower than its sentence
    one more call, against the sentence (`openodke.ground.widen`, #102). Off by
    default. The extra calls go through `widen_client` when one is given, so a
    cost meter can count them apart; `stats["widen"]` counts them either way.

    `ground_many` is the batched path: a document's facts at once, with at most
    `max_workers` model calls in flight. `ground_documents` is the same across
    many documents, and is what the pipeline uses, so a corpus of one-row
    documents runs as concurrently as one long one. Both clients are
    synchronous and the wait is network I/O, so a thread pool is the right tool
    and needs nothing outside the standard library.
    """

    def __init__(
        self,
        roles: ModelRoles | None = None,
        *,
        client: LLMClient | None = None,
        max_workers: int = 8,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        context: Context = "span",
        verdicts: Verdicts = "three_way",
        locate: bool = False,
        widen: bool = False,
        widen_client: LLMClient | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        if context not in ("span", "document"):
            raise ValueError(f"context must be 'span' or 'document', got {context!r}")
        if verdicts not in ("three_way", "binary"):
            raise ValueError(f"verdicts must be 'three_way' or 'binary', got {verdicts!r}")
        if widen and context == "document":
            raise ValueError(
                "widen=True re-asks against the sentence around a narrow span, and "
                "context='document' already shows the model the whole document"
            )
        self.context: Context = context
        self.binary = verdicts == "binary"
        self.prompt: Prompt = _PAPER if self.binary else _SPAN
        self.roles = roles if roles is not None else ModelRoles()
        self.spec: ModelSpec = self.roles.ground
        self.client: LLMClient = client if client is not None else self.roles.client_for("ground")
        self.widen = widen
        self.widen_client: LLMClient = widen_client if widen_client is not None else self.client
        self.max_workers = max_workers
        self.retry = retry if retry is not None else RetryPolicy()
        self.span_grounder = SpanGrounder()
        self.locator = SpanLocator() if locate else None
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
        # Kept apart from the counts above, so they read the same with widen off.
        self._widened = Counts(
            "retried",
            "recovered",
            "calls",
            "failed",
            "unparseable",
            "prompt_tokens",
            "completion_tokens",
        )
        self._widen_cost: float | None = None

    def ground(self, fact: Fact, doc: Document) -> Fact:
        prepared, passage = self._prepare(fact, doc)
        return prepared if passage is None else self._ask(prepared, passage, doc)

    def ground_many(self, facts: Sequence[Fact], doc: Document) -> list[Fact]:
        """One document's facts, with the model calls run concurrently.

        The span checks run first, here and in order, and only the facts that
        survive them are handed to the pool. One fact comes back for each that
        went in, in the same order: a failed call leaves its own fact
        `UNCHECKED` and the rest of the batch carries on. A client used here
        must be thread-safe — both built-in clients are; `ScriptedClient`
        answers by position and is not, which is what `RecordedClient` is for.
        """
        return self.ground_documents([(facts, doc)])[0]

    def ground_documents(
        self, batches: Sequence[tuple[Sequence[Fact], Document]]
    ) -> list[list[Fact]]:
        """`ground_many` for several documents at once, their calls sharing one pool.

        Each `(facts, doc)` gets back a list of its facts in the same order, as
        `ground_many` would return it; the only difference is that the
        `max_workers` ceiling spans documents rather than stopping at each one.
        """
        out: list[list[Fact]] = []
        pending: list[tuple[int, int, Fact, str, Document]] = []
        for at, (facts, doc) in enumerate(batches):
            row: list[Fact] = []
            for index, fact in enumerate(facts):
                prepared, passage = self._prepare(fact, doc)
                row.append(prepared)
                if passage is not None:
                    pending.append((at, index, prepared, passage, doc))
            out.append(row)
        if len(pending) == 1:
            at, index, fact, passage, doc = pending[0]
            out[at][index] = self._ask(fact, passage, doc)
        elif pending:
            workers = min(self.max_workers, len(pending))
            with ThreadPoolExecutor(workers, thread_name_prefix="openodke-ground") as pool:
                futures = [(a, i, pool.submit(self._ask, f, p, d)) for a, i, f, p, d in pending]
                for at, index, future in futures:
                    out[at][index] = future.result()
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
        if self.locator is not None:
            checked = self.locator.locate(checked, doc)
        evidence = located(checked, doc)
        # The span check leaves a fact UNCHECKED only when something located it,
        # so this guard is for the type checker, not a path that runs.
        if evidence is None or evidence.span is None:  # pragma: no cover
            return checked, None
        if self.context == "document":
            return checked, doc.text
        return checked, evidence.span.resolve(doc)

    def _ask(self, fact: Fact, passage: str, doc: Document) -> Fact:
        verdict = self._judge(fact, passage)
        if verdict is None:
            return fact
        if verdict is GroundingVerdict.NOT_FOUND and self.widen:
            retried = self._retry_wider(fact, doc)
            if retried is not None:
                return retried
        self._counts.bump(verdict.value)
        return fact.model_copy(update={"verdict": verdict})

    def _retry_wider(self, fact: Fact, doc: Document) -> Fact | None:
        """One more call against the sentence, or None when the span is already that wide."""
        cited = located(fact, doc)
        wider = widen(cited, doc) if cited is not None else None
        if cited is None or cited.span is None or wider is None or wider.span is None:
            return None
        self._widened.bump("retried")
        verdict = self._judge(fact, wider.span.resolve(doc), widened=True)
        record = {
            "from": [cited.span.start, cited.span.end],
            "to": [wider.span.start, wider.span.end],
            "verdict": (verdict or GroundingVerdict.UNCHECKED).value,
        }
        qualifiers = {**fact.qualifiers, WIDEN: record}
        if verdict is not GroundingVerdict.SUPPORTED:
            self._counts.bump(GroundingVerdict.NOT_FOUND.value)
            return fact.model_copy(
                update={"verdict": GroundingVerdict.NOT_FOUND, "qualifiers": qualifiers}
            )
        self._widened.bump("recovered")
        self._counts.bump(GroundingVerdict.SUPPORTED.value)
        evidence = tuple(wider if e is cited else e for e in fact.evidence)
        return fact.model_copy(
            update={
                "verdict": GroundingVerdict.SUPPORTED,
                "evidence": evidence,
                "qualifiers": qualifiers,
            }
        )

    def _judge(self, fact: Fact, passage: str, *, widened: bool = False) -> GroundingVerdict | None:
        """The model's verdict on `fact` against `passage`, or None when there is none to read."""
        messages = build_messages(fact, passage, binary=self.binary)
        counts = self._widened if widened else self._counts

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
                lambda: self._complete(messages, widened=widened),
                self.retry,
                sleep=self._sleep,
                on_retry=on_retry,
            )
        except (MissingAPIKey, ProviderNotInstalled):
            # Not one fact's problem: every call would fail the same way, and a
            # gate that accepts UNCHECKED would write the run as if it were checked.
            raise
        except Exception as exc:  # isolation: one failed call never fails the batch
            counts.bump("failed")
            left = "its not_found stands" if widened else "left unchecked"
            log.warning("grounding call failed for fact %s, %s: %s", fact.id, left, exc)
            return None
        self._account(completion, widened=widened)
        verdict = parse_verdict(completion, binary=self.binary)
        if verdict is None:
            counts.bump("unparseable")
            log.warning(
                "unreadable grounding answer for fact %s: %r", fact.id, completion.text[:200]
            )
        return verdict

    def _complete(self, messages: list[Message], *, widened: bool = False) -> Completion:
        with self._slots:
            self._counts.bump("calls")
            if widened:
                self._widened.bump("calls")
            schema = BINARY_SCHEMA if self.binary else GROUNDING_SCHEMA
            client = self.widen_client if widened else self.client
            return client.complete(messages, spec=self.spec, schema=schema)

    def _account(self, completion: Completion, *, widened: bool = False) -> None:
        for counts in (self._counts, self._widened) if widened else (self._counts,):
            counts.bump("prompt_tokens", completion.prompt_tokens)
            counts.bump("completion_tokens", completion.completion_tokens)
        if completion.cost_usd is not None:
            with self._cost_lock:
                self._cost = (self._cost or 0.0) + completion.cost_usd
                if widened:
                    self._widen_cost = (self._widen_cost or 0.0) + completion.cost_usd

    @property
    def stats(self) -> dict[str, Any]:
        """The stage's run report. `cost_usd` is None until a provider reports one."""
        out: dict[str, Any] = self._counts.snapshot()
        with self._cost_lock:
            out["cost_usd"] = self._cost
        # The registered prompt the calls sent, by key: none when no call was made.
        out["prompts"] = [self.prompt.key] if out["calls"] else []
        out["span"] = self.span_grounder.stats
        if self.locator is not None:
            out["locate"] = self.locator.stats
        if self.widen:
            # The retries' own share; their calls, tokens and cost are in the totals too.
            widened: dict[str, Any] = self._widened.snapshot()
            with self._cost_lock:
                widened["cost_usd"] = self._widen_cost
            out["widen"] = widened
        return out


__all__ = [
    "BINARY_SCHEMA",
    "GROUNDING_SCHEMA",
    "PAPER_PROMPT",
    "SYSTEM_PROMPT",
    "VERDICTS",
    "LLMGrounder",
    "build_messages",
    "parse_verdict",
    "render_claim",
    "render_triple",
]
