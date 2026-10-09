"""Concurrency limits per provider (#162): one limit, shared by every stage in the process.

No model is called. A slow fake client counts the calls in flight across every
client that reaches it, which is what a provider's rate limit counts.
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Chunk, Document, Entity, Evidence, Fact, Ontology, Span
from openodke.cli.main import app
from openodke.extract import LLMExtractor
from openodke.ground import LLMGrounder, RetryPolicy
from openodke.llm import (
    PROVIDER_LIMITS,
    Completion,
    LimitedClient,
    Message,
    ModelRoles,
    ModelSpec,
    ProviderError,
    ProviderLimits,
)
from openodke.run import build, parse_config

runner = CliRunner()
SPEC = ModelSpec(model="acme/large")


class _InFlight:
    """Answers after a short wait, and keeps the most calls it ever had in flight."""

    def __init__(self, wait: float = 0.02) -> None:
        self.wait = wait
        self.now = 0
        self.most = 0
        self.calls = 0
        self._lock = threading.Lock()

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        with self._lock:
            self.now += 1
            self.calls += 1
            self.most = max(self.most, self.now)
        try:
            time.sleep(self.wait)
        finally:
            with self._lock:
                self.now -= 1
        text = (
            '{"verdict": "supported"}' if "Claim:" in messages[-1].content else '{"entities": []}'
        )
        return Completion(text=text, parsed=json.loads(text), model=spec.model)


@pytest.fixture(autouse=True)
def _process_limits() -> Iterator[None]:
    """The process-wide table, put back as it was: a limit set here must not outlive the test."""
    before = PROVIDER_LIMITS.limits
    yield
    for provider in PROVIDER_LIMITS.limits:
        PROVIDER_LIMITS.set(provider, before.get(provider))


def _ask_many(client: Any, n: int, spec: ModelSpec = SPEC) -> None:
    threads = [
        threading.Thread(
            target=client.complete, args=([Message(content="Claim: x")],), kwargs={"spec": spec}
        )
        for _ in range(n)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_a_provider_s_limit_holds_however_many_threads_call() -> None:
    limits = ProviderLimits()
    limits.set("acme", 2)
    inner = _InFlight()
    _ask_many(LimitedClient(inner, limits), 12)
    assert inner.calls == 12 and inner.most == 2


def test_the_limit_is_the_provider_s_not_the_model_s_and_other_providers_are_free() -> None:
    limits = ProviderLimits()
    limits.set("acme", 1)
    inner = _InFlight()
    client = LimitedClient(inner, limits)
    # Two models of one provider share its one slot.
    big, small = ModelSpec(model="acme/large"), ModelSpec(model="acme/small")
    threads = [threading.Thread(target=_ask_many, args=(client, 4, spec)) for spec in (big, small)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert inner.most == 1

    other = _InFlight()
    _ask_many(LimitedClient(other, limits), 6, ModelSpec(model="ollama/llama3.1"))
    assert other.most > 1


def test_the_limit_holds_across_the_extractor_and_the_grounder_at_once(people: Ontology) -> None:
    limits = ProviderLimits()
    limits.set("acme", 3)
    inner = _InFlight()
    client = LimitedClient(inner, limits)
    chunks = [
        Chunk(doc_id=f"c{i}", index=0, text=f"Person {i} was born.", start=0, end=19)
        for i in range(16)
    ]
    docs = [Document(id=f"g{i}", text=f"Person {i} was born in 18{i:02}.") for i in range(16)]
    extractor = LLMExtractor(client=client, spec=SPEC, types=["Person"], max_workers=8)
    grounder = LLMGrounder(ModelRoles.single("acme/large"), client=client, max_workers=8)
    work = [([_fact(doc)], doc) for doc in docs]

    threads = [
        threading.Thread(target=extractor.extract_many, args=(chunks, people)),
        threading.Thread(target=grounder.ground_documents, args=(work,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Sixteen workers between them, three slots: the provider never saw more.
    assert inner.calls == 32
    assert inner.most == 3


def test_the_stricter_of_a_stage_s_workers_and_the_provider_s_limit_wins() -> None:
    limits = ProviderLimits()
    limits.set("acme", 8)
    inner = _InFlight()
    grounder = LLMGrounder(
        ModelRoles.single("acme/large"), client=LimitedClient(inner, limits), max_workers=2
    )
    docs = [Document(id=f"g{i}", text=f"Person {i} was born in 18{i:02}.") for i in range(10)]
    grounder.ground_documents([([_fact(doc)], doc) for doc in docs])
    assert inner.most == 2


def test_a_limit_is_at_least_one_and_none_lifts_it() -> None:
    limits = ProviderLimits()
    with pytest.raises(ValueError, match="at least 1"):
        limits.set("acme", 0)
    limits.set("ACME", 2)
    assert limits.limits == {"acme": 2}
    limits.set("acme", None)
    assert limits.limits == {}


# --------------------------------------------------------------------------- #
# Retry-After
# --------------------------------------------------------------------------- #


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.now += seconds


class _Throttled:
    """Fails its first call as a provider does at its rate limit: 429, Retry-After."""

    def __init__(self, after: str) -> None:
        self.after = after
        self.calls = 0

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        self.calls += 1
        if self.calls == 1:
            headers = {"Retry-After": self.after}
            cause = urllib.error.HTTPError("https://x", 429, "Too Many", headers, io.BytesIO())  # type: ignore[arg-type]
            raise ProviderError("acme returned 429") from cause
        return Completion(text="ok", model=spec.model)


def test_a_retry_after_holds_every_call_to_that_provider_not_only_the_retry() -> None:
    clock = _Clock()
    limits = ProviderLimits(clock=clock, sleep=clock.sleep)
    throttled = LimitedClient(_Throttled("7"), limits)
    with pytest.raises(ProviderError):
        throttled.complete([Message(content="x")], spec=SPEC)

    # Another client, another stage, the same provider: it waits the seven seconds.
    LimitedClient(_InFlight(0), limits).complete([Message(content="x")], spec=SPEC)
    assert clock.slept == [7.0]
    # Another provider does not.
    LimitedClient(_InFlight(0), limits).complete(
        [Message(content="x")], spec=ModelSpec(model="ollama/llama3.1")
    )
    assert clock.slept == [7.0]


def test_a_pause_is_capped_and_the_retry_policy_still_retries() -> None:
    clock = _Clock()
    limits = ProviderLimits(max_pause=30.0, clock=clock, sleep=clock.sleep)
    inner = _Throttled("3600")
    grounder = LLMGrounder(
        ModelRoles.single("acme/large"),
        client=LimitedClient(inner, limits),
        retry=RetryPolicy(attempts=2, max_delay=30.0),
        sleep=clock.sleep,
    )
    doc = Document(id="g0", text="Person 0 was born in 1800.")
    grounder.ground(_fact(doc), doc)
    # The retry policy's own wait, then the provider's pause, both capped at 30 s.
    assert clock.slept == [30.0] and inner.calls == 2


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_models_limits_sets_the_process_wide_limit_per_provider(tmp_path: Path) -> None:
    (tmp_path / "ontology.json").write_text('{"name": "o"}', encoding="utf-8")
    (tmp_path / "corpus").mkdir()
    config = parse_config(
        {
            "ontology": "ontology.json",
            "inputs": ["corpus"],
            "models": {"extract": "acme/large", "limits": {"acme": 4, "ollama": 2}},
            "stages": {"extractor": "pattern"},
        },
        base_dir=tmp_path,
    )
    build(config)
    assert PROVIDER_LIMITS.limits == {"acme": 4, "ollama": 2}


def test_a_limit_below_one_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / "odke.json").write_text(
        json.dumps(
            {
                "ontology": "ontology.json",
                "inputs": ["corpus"],
                "models": {"limits": {"acme": 0}},
                "stages": {"extractor": "pattern"},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", str(tmp_path / "odke.json")])
    assert result.exit_code == 2 and "models.limits.acme" in result.output


def _fact(doc: Document) -> Fact:
    span = Span(doc_id=doc.id, start=0, end=len(doc.text), quote=doc.text)
    return Fact(
        subject=Entity(key=f"p:{doc.id}", type="Person", label=f"Person {doc.id}"),
        predicate="birth_date",
        object_value=doc.id,
        evidence=(Evidence(doc_id=doc.id, span=span),),
    )
