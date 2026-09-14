"""The recorded-response harness: model paths run in CI with no key and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openodke.llm import (
    Cassette,
    LLMClient,
    Message,
    ModelSpec,
    ProviderError,
    RecordedClient,
    RecordingClient,
    ReplayClient,
    ScriptedClient,
)
from openodke.llm.testing import Interaction, RecordedCompletion, RequestMatch, fingerprint

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
SONNET = ModelSpec(model="anthropic/claude-sonnet-5")
LLAMA = ModelSpec(model="ollama/llama3.1")


def _ask(text: str) -> list[Message]:
    return [Message(role="system", content="Answer briefly."), Message(content=text)]


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_every_fixture_loads_in_the_client_that_reads_it(path: Path) -> None:
    """A hand-authored fixture with a typo fails here, not deep inside another test.

    Two formats share the directory: a JSON list is `RecordedClient` entries, an
    object is a cassette for `ReplayClient`.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        assert data and all("match" in entry for entry in data)
        RecordedClient.from_fixture(path)
        return
    cassette = Cassette.load(path)
    assert cassette.description
    assert cassette.interactions


def test_a_cassette_replays_by_content_and_reports_usage_and_cost() -> None:
    client = ReplayClient(FIXTURES / "replay_basics.json")
    assert isinstance(client, LLMClient)
    # Out of order on purpose: matching is by content, not position.
    rome = client.complete(_ask("What is the capital of Italy?"), spec=LLAMA)
    paris = client.complete(_ask("What is the capital of France?"), spec=SONNET)
    assert (rome.text, rome.parsed, rome.model) == ("Rome.", None, "ollama/llama3.1")
    assert paris.parsed == {"answer": "Paris"}
    assert (paris.prompt_tokens, paris.completion_tokens, paris.cost_usd) == (12, 5, 0.00021)
    # Unknown cost stays unknown, as in both real adapters.
    assert rome.cost_usd is None
    assert client.exhausted and len(client.calls) == 2


def test_each_interaction_answers_once_and_a_miss_names_its_fingerprint() -> None:
    client = ReplayClient(FIXTURES / "replay_basics.json")
    client.complete(_ask("capital of France"), spec=SONNET)
    with pytest.raises(ProviderError, match="re-record") as missed:
        client.complete(_ask("capital of France"), spec=SONNET)
    assert fingerprint(_ask("capital of France"), spec=SONNET) in str(missed.value)
    # The Italy interaction is pinned to a model.
    with pytest.raises(ProviderError, match="1 unused"):
        client.complete(_ask("capital of Italy"), spec=SONNET)


def test_record_then_replay_round_trips_through_a_file(tmp_path: Path) -> None:
    recorder = RecordingClient(ScriptedClient([{"answer": "Paris"}, "Rome."]))
    schema = {"type": "object", "title": "answer"}
    first = recorder.complete(_ask("capital of France"), spec=SONNET, schema=schema)
    recorder.complete(_ask("capital of Italy"), spec=SONNET)
    path = tmp_path / "recorded.json"
    saved = recorder.save(path, description="round trip")
    assert Cassette.load(path) == saved

    replay = ReplayClient(path)
    again = replay.complete(_ask("capital of France"), spec=SONNET, schema=schema)
    assert (again.text, again.parsed) == (first.text, first.parsed)


def test_a_recorded_interaction_goes_stale_when_the_prompt_changes(tmp_path: Path) -> None:
    """The point of fingerprints: an edited prompt fails loudly instead of replaying."""
    recorder = RecordingClient(ScriptedClient(["Paris."]))
    recorder.complete(_ask("capital of France"), spec=SONNET)
    replay = ReplayClient(recorder.cassette("stale"))
    reworded = [
        Message(role="system", content="Answer in one word."),
        Message(content="capital of France"),
    ]
    with pytest.raises(ProviderError, match="no recorded interaction"):
        replay.complete(reworded, spec=SONNET)
    # A different schema is a different question too.
    with pytest.raises(ProviderError):
        replay.complete(_ask("capital of France"), spec=SONNET, schema={"type": "object"})


def test_fingerprints_ignore_sampling_settings() -> None:
    cooler = SONNET.model_copy(update={"temperature": 0.7, "max_tokens": 10})
    assert fingerprint(_ask("x"), spec=SONNET) == fingerprint(_ask("x"), spec=cooler)
    assert fingerprint(_ask("x"), spec=SONNET) != fingerprint(_ask("x"), spec=LLAMA)


def test_an_empty_match_answers_anything() -> None:
    cassette = Cassette(
        description="catch-all",
        interactions=(Interaction(match=RequestMatch(), response=RecordedCompletion(text="ok")),),
    )
    assert ReplayClient(cassette).complete(_ask("anything"), spec=LLAMA).text == "ok"
