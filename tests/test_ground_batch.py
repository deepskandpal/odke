"""Grounding at volume: bounded concurrency, retry with backoff, failures kept to one fact."""

from __future__ import annotations

import threading
import time
import urllib.error
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from openodke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    Ontology,
    Pipeline,
    Span,
)
from openodke.ground import LLMGrounder, RetryPolicy, is_transient
from openodke.ground.retry import call_with_retry, retry_after
from openodke.llm import (
    Completion,
    LLMClient,
    Message,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
    RecordedClient,
)

N = 8
DOC = Document(id="d1", text="".join(f"Sentence number {i}. " for i in range(N)))
SUBJECT = Entity(key="s", type="Thing", label="The thing")
NO_WAIT = RetryPolicy(attempts=4, base_delay=0.5, jitter=False)


def _facts(doc: Document = DOC, n: int = N) -> list[Fact]:
    """`n` facts, each citing its own real sentence, each a distinct prompt."""
    out = []
    for i in range(n):
        sentence = f"Sentence number {i}."
        start = doc.text.index(sentence)
        span = Span(doc_id=doc.id, start=start, end=start + len(sentence), quote=sentence)
        out.append(
            Fact(
                subject=SUBJECT,
                predicate="numbered",
                object_value=i,
                evidence=(Evidence(doc_id=doc.id, span=span),),
            )
        )
    return out


def _supported() -> RecordedClient:
    return RecordedClient([{"match": "Claim:", "response": {"verdict": "supported"}}])


def _http(code: int, **headers: str) -> ProviderError:
    """What the stdlib client raises: a ProviderError caused by an HTTPError."""
    cause = urllib.error.HTTPError("https://x/v1", code, "err", headers or None, None)  # type: ignore[arg-type]
    error = ProviderError(f"x returned {code}")
    error.__cause__ = cause
    return error


class _Flaky:
    """Fails each distinct prompt `fail` times with `error()`, then answers."""

    def __init__(
        self, fail: int, error: Callable[[], Exception], answer: LLMClient | None = None
    ) -> None:
        self.fail = fail
        self.error = error
        self.answer = answer or _supported()
        self.seen: Counter[str] = Counter()
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        key = messages[-1].content
        with self._lock:
            self.seen[key] += 1
            count = self.seen[key]
        if count <= self.fail:
            raise self.error()
        return self.answer.complete(messages, spec=spec, schema=schema)

    @property
    def calls(self) -> int:
        return sum(self.seen.values())


class _Poisoned:
    """Answers every prompt except those containing `poison`, which raise `error()`."""

    def __init__(self, poison: Sequence[str], error: Callable[[], Exception]) -> None:
        self.poison = poison
        self.error = error
        self.answer = _supported()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        if any(p in messages[-1].content for p in self.poison):
            raise self.error()
        return self.answer.complete(messages, spec=spec, schema=schema)


class _Gauge:
    """Records the most calls ever in flight at once."""

    def __init__(self, hold: float = 0.01, barrier: threading.Barrier | None = None) -> None:
        self.hold = hold
        self.barrier = barrier
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        self.answer = _supported()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait()
            time.sleep(self.hold)
            return self.answer.complete(messages, spec=spec, schema=schema)
        finally:
            with self._lock:
                self.active -= 1


def _grounder(client: LLMClient, **kwargs: Any) -> tuple[LLMGrounder, list[float]]:
    slept: list[float] = []
    kwargs.setdefault("retry", NO_WAIT)
    return LLMGrounder(client=client, sleep=slept.append, **kwargs), slept


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #


def test_a_transient_failure_is_retried_with_backoff_and_then_answers() -> None:
    client = _Flaky(2, lambda: _http(429))
    grounder, slept = _grounder(client)
    (fact,) = _facts(n=1)
    assert grounder.ground(fact, DOC).verdict is GroundingVerdict.SUPPORTED
    assert slept == [0.5, 1.0]
    stats = grounder.stats
    assert (stats["calls"], stats["retries"], stats["failed"]) == (3, 2, 0)


def test_when_retries_run_out_the_fact_is_left_unchecked_and_nothing_raises() -> None:
    client = _Flaky(99, lambda: _http(503))
    grounder, slept = _grounder(client, retry=RetryPolicy(attempts=3, jitter=False))
    (fact,) = _facts(n=1)
    grounded = grounder.ground(fact, DOC)
    assert grounded.verdict is GroundingVerdict.UNCHECKED
    assert client.calls == 3 and len(slept) == 2
    assert grounder.stats["failed"] == 1


def test_a_permanent_error_is_not_retried() -> None:
    """A 401 is not weather; waiting and asking again only spends the wait."""
    client = _Flaky(99, lambda: _http(401))
    grounder, slept = _grounder(client)
    (fact,) = _facts(n=1)
    assert grounder.ground(fact, DOC).verdict is GroundingVerdict.UNCHECKED
    assert client.calls == 1 and slept == []
    assert grounder.stats["retries"] == 0


def test_backoff_grows_caps_and_jitters_below_the_cap() -> None:
    policy = RetryPolicy(base_delay=0.5, multiplier=2, max_delay=10, jitter=False)
    assert [policy.delay(r) for r in range(1, 7)] == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0]
    jittered = RetryPolicy(base_delay=0.5, multiplier=2, max_delay=10)
    for retry in range(1, 7):
        assert 0.0 <= jittered.delay(retry) <= policy.delay(retry)


def test_a_servers_retry_after_wins_and_is_capped() -> None:
    assert retry_after(_http(429, **{"Retry-After": "3"})) == 3.0
    assert RetryPolicy(max_delay=30, jitter=False).delay(1, _http(429, **{"Retry-After": "3"})) == 3
    assert RetryPolicy(max_delay=2, jitter=False).delay(1, _http(429, **{"Retry-After": "3"})) == 2
    # The HTTP-date form falls back to backoff rather than being parsed.
    dated = _http(429, **{"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert RetryPolicy(base_delay=0.5, jitter=False).delay(1, dated) == 0.5


class _LitellmStyle(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("provider said no")
        self.status_code = status_code


def _caused(cause: BaseException, message: str = "call failed") -> ProviderError:
    error = ProviderError(message)
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (_http(429), True),
        (_http(503), True),
        (_http(529), True),
        (_http(401), False),
        (_http(400), False),
        (_caused(urllib.error.URLError("connection refused")), True),
        (_caused(TimeoutError("read timed out")), True),
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (_caused(_LitellmStyle(429)), True),
        (_caused(_LitellmStyle(403)), False),
        (ProviderError("anthropic/x call failed: RateLimitError: rate limit exceeded"), True),
        (ProviderError("unexpected response shape from ollama: {}"), False),
        (ProviderNotInstalled("litellm is not installed"), False),
        (ValueError("a bug"), False),
    ],
)
def test_is_transient_reads_the_status_first_and_the_message_last(
    exc: BaseException, transient: bool
) -> None:
    assert is_transient(exc) is transient


def test_call_with_retry_raises_the_last_error() -> None:
    attempts: list[int] = []

    def fail() -> None:
        attempts.append(1)
        raise _http(429)

    with pytest.raises(ProviderError, match="429"):
        call_with_retry(fail, RetryPolicy(attempts=3, jitter=False), sleep=lambda s: None)
    assert len(attempts) == 3


def test_retry_policy_and_worker_ceiling_are_validated() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(attempts=0)
    with pytest.raises(ValueError, match="max_workers"):
        LLMGrounder(client=_supported(), max_workers=0)


# --------------------------------------------------------------------------- #
# Batches: concurrency and isolation
# --------------------------------------------------------------------------- #


def test_one_failed_call_never_fails_the_batch() -> None:
    """Order and length hold; the failed fact is UNCHECKED and everything else is decided."""
    client = _Poisoned(["— 3.", "— 5."], lambda: _http(401))
    grounder, _ = _grounder(client, max_workers=4)
    facts = _facts()
    grounded = grounder.ground_many(facts, DOC)
    assert [f.id for f in grounded] == [f.id for f in facts]
    verdicts = {f.object_value: f.verdict for f in grounded}
    assert verdicts[3] is verdicts[5] is GroundingVerdict.UNCHECKED
    assert all(v is GroundingVerdict.SUPPORTED for k, v in verdicts.items() if k not in (3, 5))
    assert grounder.stats["failed"] == 2
    assert grounder.stats["supported"] == N - 2


def test_an_exception_the_client_never_wrapped_is_isolated_too() -> None:
    """A read timeout can escape urllib unwrapped; it is still one fact's problem."""
    client = _Poisoned(["— 2."], lambda: RuntimeError("socket closed"))
    grounder, _ = _grounder(client, max_workers=4)
    grounded = grounder.ground_many(_facts(), DOC)
    assert [f.verdict for f in grounded].count(GroundingVerdict.UNCHECKED) == 1


def test_a_flaky_batch_recovers_every_fact() -> None:
    client = _Flaky(2, lambda: _http(429))
    grounder, slept = _grounder(client, max_workers=4)
    grounded = grounder.ground_many(_facts(), DOC)
    assert all(f.verdict is GroundingVerdict.SUPPORTED for f in grounded)
    assert grounder.stats["retries"] == 2 * N
    assert grounder.stats["calls"] == 3 * N
    assert sorted(slept) == sorted([0.5, 1.0] * N)


def test_calls_really_run_concurrently_up_to_the_ceiling() -> None:
    """Four calls must be in flight at once for the barrier to open. Serial would time out."""
    client = _Gauge(hold=0, barrier=threading.Barrier(4, timeout=5))
    grounder, _ = _grounder(client, max_workers=4, retry=RetryPolicy(attempts=1))
    grounded = grounder.ground_many(_facts(), DOC)
    assert all(f.verdict is GroundingVerdict.SUPPORTED for f in grounded)
    assert client.peak == 4


def test_the_ceiling_holds_across_concurrent_batches() -> None:
    """The limit is on the call, so two batches at once still share three slots."""
    client = _Gauge(hold=0.02)
    grounder, _ = _grounder(client, max_workers=3)
    docs = [DOC, DOC.model_copy(update={"id": "d2"})]
    results: dict[str, list[Fact]] = {}

    def run(doc: Document) -> None:
        results[doc.id] = grounder.ground_many(_facts(doc), doc)

    threads = [threading.Thread(target=run, args=(d,)) for d in docs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert client.peak <= 3
    assert all(f.verdict is GroundingVerdict.SUPPORTED for fs in results.values() for f in fs)


def test_span_rejections_in_a_batch_cost_no_call_and_hold_their_place() -> None:
    facts = _facts(n=4)
    lie = Span(doc_id="d1", start=0, end=19, quote="Something else said")
    facts[1] = facts[1].model_copy(update={"evidence": (Evidence(doc_id="d1", span=lie),)})
    client = _Gauge(hold=0)
    grounder, _ = _grounder(client, max_workers=4)
    grounded = grounder.ground_many(facts, DOC)
    assert [f.verdict for f in grounded] == [
        GroundingVerdict.SUPPORTED,
        GroundingVerdict.NOT_FOUND,
        GroundingVerdict.SUPPORTED,
        GroundingVerdict.SUPPORTED,
    ]
    assert grounder.stats["calls"] == 3


def test_an_interrupted_run_resumes_by_asking_only_what_is_still_unchecked() -> None:
    """The first pass loses two calls; the second pass asks about those two and nothing else."""
    first, _ = _grounder(_Poisoned(["— 1.", "— 6."], lambda: _http(503)), max_workers=4)
    partial = first.ground_many(_facts(), DOC)
    assert [f.verdict for f in partial].count(GroundingVerdict.UNCHECKED) == 2

    client = _Gauge(hold=0)
    second, _ = _grounder(client, max_workers=4)
    resumed = second.ground_many(partial, DOC)
    assert all(f.verdict is GroundingVerdict.SUPPORTED for f in resumed)
    assert second.stats["calls"] == 2
    assert second.stats["skipped"] == N - 2


def test_an_empty_batch_is_free() -> None:
    grounder, _ = _grounder(_Gauge(hold=0))
    assert grounder.ground_many([], DOC) == []
    assert grounder.stats["calls"] == 0


# --------------------------------------------------------------------------- #
# Through the pipeline
# --------------------------------------------------------------------------- #


class _SentenceChunker:
    def chunk(self, doc: Document) -> Iterable[Chunk]:
        start = 0
        for index, part in enumerate(doc.text.split(". ")):
            end = start + len(part) + 2
            yield Chunk(doc_id=doc.id, start=start, end=end, text=doc.text[start:end], index=index)
            start = end


class _NumberedExtractor:
    """One fact per chunk that names a sentence number, citing that sentence."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        doc = DOC.model_copy(update={"id": chunk.doc_id})
        return [f for f in _facts(doc) if f"number {f.object_value}." in chunk.text]


class _BatchRecorder:
    def __init__(self, drop: bool = False) -> None:
        self.batches: list[tuple[str, int]] = []
        self.drop = drop

    def ground(self, fact: Fact, doc: Document) -> Fact:  # pragma: no cover - not used
        raise AssertionError("the pipeline should have batched")

    def ground_many(self, facts: Sequence[Fact], doc: Document) -> list[Fact]:
        self.batches.append((doc.id, len(facts)))
        return list(facts)[1:] if self.drop else list(facts)


def test_the_pipeline_hands_a_batching_grounder_each_document_once() -> None:
    grounder = _BatchRecorder()
    docs = [DOC, DOC.model_copy(update={"id": "d2"})]
    kg = Pipeline(
        Ontology(), _NumberedExtractor(), chunker=_SentenceChunker(), grounder=grounder
    ).run(docs)
    assert grounder.batches == [("d1", N), ("d2", N)]
    assert len(kg) == 2 * N


def test_the_pipeline_refuses_a_batch_that_drops_facts() -> None:
    """A grounder stamps, it never drops; the validator is the gate (DECISIONS #20)."""
    pipeline = Pipeline(
        Ontology(),
        _NumberedExtractor(),
        chunker=_SentenceChunker(),
        grounder=_BatchRecorder(drop=True),
    )
    with pytest.raises(ValueError, match="never drops"):
        pipeline.run([DOC])


def test_through_the_pipeline_a_documents_calls_run_concurrently() -> None:
    client = _Gauge(hold=0, barrier=threading.Barrier(4, timeout=5))
    grounder, _ = _grounder(client, max_workers=4, retry=RetryPolicy(attempts=1))
    kg = Pipeline(
        Ontology(), _NumberedExtractor(), chunker=_SentenceChunker(), grounder=grounder
    ).run([DOC])
    assert len(kg) == N
    assert all(f.verdict is GroundingVerdict.SUPPORTED for f in kg.facts)
    assert client.peak == 4
