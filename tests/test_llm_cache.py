"""The response cache (#156): a rerun of an unchanged batch costs nothing.

No model is called. The clients wrapped here are scripted, recorded or replayed,
and a counting client says how many requests got past the cache.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from openodke import prompts
from openodke.ground.llm import GROUNDING_SCHEMA, build_messages
from openodke.llm import (
    CachedClient,
    Completion,
    DirectoryCache,
    MemoryCache,
    Message,
    ModelSpec,
    ProviderError,
)
from openodke.llm.cache import FORMAT, cache_key, prompt_keys, request

SPEC = ModelSpec(model="test/model", max_tokens=64)
MESSAGES = [Message(role="system", content="Judge."), Message(content="Claim: x. Passage: y.")]


class _Counting:
    """Answers every request with its own count, and says how many it was asked."""

    def __init__(self, *, fail: int = 0) -> None:
        self.calls = 0
        self._fail = fail
        self._lock = threading.Lock()

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        with self._lock:
            self.calls += 1
            count = self.calls
        if count <= self._fail:
            raise ProviderError("429 too many requests")
        return Completion(
            text=json.dumps({"verdict": "supported", "n": count}),
            parsed={"verdict": "supported", "n": count},
            model=spec.model,
            prompt_tokens=100,
            completion_tokens=5,
            cost_usd=0.0003,
        )


# --------------------------------------------------------------------------- #
# The key
# --------------------------------------------------------------------------- #


def test_the_key_is_the_sha256_of_the_request_as_canonical_json() -> None:
    key = cache_key(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)
    assert len(key) == 64 and int(key, 16) >= 0
    assert key == cache_key(list(MESSAGES), spec=SPEC.model_copy(), schema=dict(GROUNDING_SCHEMA))
    body = request(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)
    assert set(body) == {"format", "model", "base_url", "messages", "schema", "sampling", "prompts"}


@pytest.mark.parametrize(
    ("messages", "spec", "schema"),
    [
        (MESSAGES, SPEC.model_copy(update={"model": "test/other"}), None),
        (MESSAGES, SPEC.model_copy(update={"base_url": "http://gpu-box:8000/v1"}), None),
        ([Message(role="system", content="Judge!"), MESSAGES[1]], SPEC, None),
        ([MESSAGES[0], Message(content="Claim: x. Passage: z.")], SPEC, None),
        ([Message(role="user", content="Judge."), MESSAGES[1]], SPEC, None),
        (MESSAGES[1:], SPEC, None),
        (MESSAGES, SPEC, GROUNDING_SCHEMA),
        (MESSAGES, SPEC.model_copy(update={"temperature": 0.0}), None),
        (MESSAGES, SPEC.model_copy(update={"max_tokens": 65}), None),
        (MESSAGES, SPEC.model_copy(update={"extra": {"reasoning_effort": "low"}}), None),
    ],
    ids=[
        "model",
        "base_url",
        "system",
        "user",
        "role",
        "turns",
        "schema",
        "temperature",
        "max_tokens",
        "extra",
    ],
)
def test_any_component_that_changes_the_answer_misses(
    messages: list[Message], spec: ModelSpec, schema: dict[str, Any] | None
) -> None:
    inner = _Counting()
    client = CachedClient(inner)
    client.complete(MESSAGES, spec=SPEC)
    client.complete(messages, spec=spec, schema=schema)
    assert inner.calls == 2
    assert client.stats == {"hits": 0, "misses": 2, "failed": 0}


def test_what_decides_only_whether_a_call_succeeds_is_not_in_the_key() -> None:
    inner = _Counting()
    client = CachedClient(inner)
    client.complete(MESSAGES, spec=SPEC)
    client.complete(MESSAGES, spec=SPEC.model_copy(update={"timeout": 5.0, "api_key_env": "K"}))
    assert inner.calls == 1


def test_the_registered_prompt_and_its_version_are_in_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = build_messages(_fact(), "Ada Lovelace was born in 1815.")
    assert prompt_keys(messages) == ["ground.span@1"]
    before = cache_key(messages, spec=SPEC)

    # The same text under a second version: what a key records is the version too.
    span = prompts.get("ground.span@1")
    again = prompts.Prompt(span.id, 2, span.text, span.source)
    known = prompts.registered
    monkeypatch.setattr(prompts, "registered", lambda: [*known(), again])
    assert prompt_keys(messages) == ["ground.span@1", "ground.span@2"]
    assert cache_key(messages, spec=SPEC) != before


# --------------------------------------------------------------------------- #
# Hits, misses and errors
# --------------------------------------------------------------------------- #


def test_a_second_identical_request_is_answered_from_the_store_for_nothing() -> None:
    inner = _Counting()
    client = CachedClient(inner)
    first = client.complete(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)
    second = client.complete(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)

    assert inner.calls == 1
    assert (second.text, second.parsed, second.model) == (first.text, first.parsed, first.model)
    assert not first.cached and first.cost_usd == pytest.approx(0.0003)
    # Nothing was sent, so nothing was spent: a measured zero, not an unknown.
    assert second.cached
    assert (second.prompt_tokens, second.completion_tokens, second.cost_usd) == (0, 0, 0.0)
    assert client.stats == {"hits": 1, "misses": 1, "failed": 0}


def test_an_error_is_never_stored() -> None:
    inner = _Counting(fail=1)
    store = MemoryCache()
    client = CachedClient(inner, store)
    with pytest.raises(ProviderError):
        client.complete(MESSAGES, spec=SPEC)
    assert len(store) == 0

    answered = client.complete(MESSAGES, spec=SPEC)
    assert inner.calls == 2 and not answered.cached and len(store) == 1
    assert client.stats == {"hits": 0, "misses": 1, "failed": 1}


# --------------------------------------------------------------------------- #
# The directory store
# --------------------------------------------------------------------------- #


def test_the_directory_store_survives_the_process_and_shards_by_key(tmp_path: Path) -> None:
    inner = _Counting()
    CachedClient(inner, tmp_path / "cache").complete(MESSAGES, spec=SPEC)
    key = cache_key(MESSAGES, spec=SPEC)
    path = tmp_path / "cache" / key[:2] / f"{key}.json"
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert (entry["format"], entry["key"], entry["model"]) == (FORMAT, key, "test/model")
    # The answer's own usage is kept; only a hit reports zero.
    assert entry["completion"]["prompt_tokens"] == 100
    assert "raw" not in entry["completion"]

    later = CachedClient(_Counting(), str(tmp_path / "cache"))
    assert later.complete(MESSAGES, spec=SPEC).cached


def test_an_unreadable_or_foreign_entry_is_a_miss(tmp_path: Path) -> None:
    store = DirectoryCache(tmp_path)
    key = cache_key(MESSAGES, spec=SPEC)
    store.path(key).parent.mkdir(parents=True)
    store.path(key).write_text('{"format": 1, "completion": {"te', encoding="utf-8")
    inner = _Counting()
    assert not CachedClient(inner, store).complete(MESSAGES, spec=SPEC).cached
    store.put(key, {"format": FORMAT + 1, "completion": {"text": "old layout"}})
    assert not CachedClient(inner, store).complete(MESSAGES, spec=SPEC).cached
    assert inner.calls == 2


def test_concurrent_writers_never_leave_a_torn_entry(tmp_path: Path) -> None:
    store = DirectoryCache(tmp_path)
    keys = [f"{i:02x}" * 32 for i in range(8)]
    errors: list[BaseException] = []
    seen: list[dict[str, Any] | None] = []

    def write(worker: int) -> None:
        try:
            for round in range(40):
                key = keys[(worker + round) % len(keys)]
                store.put(key, {"format": FORMAT, "worker": worker, "pad": "x" * 20_000})
                seen.append(store.get(keys[round % len(keys)]))
        except BaseException as exc:  # pragma: no cover - the failure this test exists for
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    # Every read was nothing yet or a whole entry, never half of one.
    assert all(entry is None or len(entry["pad"]) == 20_000 for entry in seen)
    files = sorted(tmp_path.rglob("*"))
    assert [f for f in files if f.is_file() and not f.name.endswith(".json")] == []
    assert len(store) == len(keys)
    assert all(json.loads(store.path(k).read_text())["pad"] for k in keys)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fact(i: int = 0) -> Any:
    from openodke import Entity, Evidence, Fact, Span

    text = f"Person {i} was born in {1800 + i}."
    return Fact(
        subject=Entity(key=f"p{i}", type="Person", label=f"Person {i}"),
        predicate="birth_date",
        object_value=str(1800 + i),
        evidence=(
            Evidence(doc_id=f"d{i}", span=Span(doc_id=f"d{i}", start=0, end=len(text), quote=text)),
        ),
    )
