"""Provider neutrality, tested as a property rather than asserted in a README."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from odke.llm import (
    Completion,
    LLMClient,
    Message,
    ModelRoles,
    ModelSpec,
    OpenAICompatClient,
    ProviderError,
    ScriptedClient,
    register,
    registered_providers,
    resolve,
    unregister,
)
from odke.llm.litellm_client import LiteLLMClient


def _openai_body(content: str = "hi", model: str = "llama3.1") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


class _FakeHTTP:
    """Stands in for urllib.request.urlopen; records the request it was given."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.request: Any = None

    def __call__(self, request: Any, timeout: float | None = None) -> Any:
        self.request = request
        payload = json.dumps(self.body).encode()
        stream = io.BytesIO(payload)
        stream.__enter__ = lambda: stream  # type: ignore[method-assign]
        stream.__exit__ = lambda *a: None  # type: ignore[method-assign]
        return stream


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model",
    ["ollama/llama3.1", "vllm/mistral", "lmstudio/x", "openrouter/y", "groq/z", "openai/gpt-4.1"],
)
def test_openai_shaped_providers_need_no_extra_dependency(model: str) -> None:
    """The point of the stdlib client: a bare install can drive a local model."""
    assert isinstance(resolve(ModelSpec(model=model)), OpenAICompatClient)


def test_a_custom_base_url_routes_to_the_compatible_client() -> None:
    """A gateway is OpenAI-shaped by convention; honour the caller's intent."""
    spec = ModelSpec(model="corp-gateway/whatever", base_url="https://llm.internal/v1")
    assert isinstance(resolve(spec), OpenAICompatClient)


def test_other_providers_route_to_litellm() -> None:
    """Bedrock, Vertex, Azure and the rest are litellm's problem, by design."""
    assert isinstance(resolve(ModelSpec(model="bedrock/anthropic.claude-sonnet-4")), LiteLLMClient)
    assert isinstance(resolve(ModelSpec(model="anthropic/claude-sonnet-5")), LiteLLMClient)


def test_a_registered_adapter_wins_over_everything() -> None:
    """The escape hatch: one call and every model string routes through you."""
    sentinel = ScriptedClient(["x"])
    register("mycorp", lambda spec: sentinel)
    try:
        assert "mycorp" in registered_providers()
        assert resolve(ModelSpec(model="mycorp/model-a")) is sentinel
    finally:
        unregister("mycorp")
    assert "mycorp" not in registered_providers()


def test_registration_can_override_a_builtin_provider() -> None:
    """Teams that must route Ollama through their own proxy are not blocked."""
    sentinel = ScriptedClient(["x"])
    register("ollama", lambda spec: sentinel)
    try:
        assert resolve(ModelSpec(model="ollama/llama3.1")) is sentinel
    finally:
        unregister("ollama")
    assert isinstance(resolve(ModelSpec(model="ollama/llama3.1")), OpenAICompatClient)


def test_a_bare_model_string_defaults_to_openai() -> None:
    assert ModelSpec(model="gpt-4.1").provider == "openai"


# --------------------------------------------------------------------------- #
# The OpenAI-compatible client
# --------------------------------------------------------------------------- #


def test_local_providers_get_a_default_base_url() -> None:
    """`ollama/llama3.1` must work with no configuration at all."""
    http = _FakeHTTP(_openai_body())
    OpenAICompatClient(opener=http).complete(
        [Message(content="hello")], spec=ModelSpec(model="ollama/llama3.1")
    )
    assert http.request.full_url == "http://localhost:11434/v1/chat/completions"
    assert json.loads(http.request.data)["model"] == "llama3.1"


def test_a_schema_becomes_a_response_format() -> None:
    http = _FakeHTTP(_openai_body('{"name": "Ada"}'))
    completion = OpenAICompatClient(opener=http).complete(
        [Message(content="extract")],
        spec=ModelSpec(model="ollama/llama3.1"),
        schema={"title": "Person", "type": "object"},
    )
    sent = json.loads(http.request.data)
    assert sent["response_format"]["json_schema"]["name"] == "Person"
    assert completion.parsed == {"name": "Ada"}


def test_usage_is_carried_through() -> None:
    http = _FakeHTTP(_openai_body())
    completion = OpenAICompatClient(opener=http).complete(
        [Message(content="hi")], spec=ModelSpec(model="ollama/llama3.1")
    )
    assert (completion.prompt_tokens, completion.completion_tokens) == (11, 3)
    # Unknown rather than zero: these servers do not report cost, and 0.0 is a lie.
    assert completion.cost_usd is None


def test_prose_does_not_become_a_failed_parse() -> None:
    """Plenty of calls ask for prose; that is not an error."""
    http = _FakeHTTP(_openai_body("Ada Lovelace was a mathematician."))
    completion = OpenAICompatClient(opener=http).complete(
        [Message(content="hi")], spec=ModelSpec(model="ollama/llama3.1")
    )
    assert completion.parsed is None
    assert completion.text.startswith("Ada")


def test_an_unknown_provider_without_a_base_url_says_so() -> None:
    with pytest.raises(ProviderError, match="no base_url"):
        OpenAICompatClient().complete(
            [Message(content="hi")], spec=ModelSpec(model="mystery/model")
        )


def test_a_malformed_response_names_the_provider() -> None:
    with pytest.raises(ProviderError, match="unexpected response shape"):
        OpenAICompatClient(opener=_FakeHTTP({"nope": True})).complete(
            [Message(content="hi")], spec=ModelSpec(model="ollama/llama3.1")
        )


# --------------------------------------------------------------------------- #
# litellm
# --------------------------------------------------------------------------- #


def test_litellm_adapter_normalises_to_the_same_completion() -> None:
    """Both adapters must be indistinguishable to everything downstream."""
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "model": "claude-sonnet-5",
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            "_response_cost": 0.0004,
        }

    completion = LiteLLMClient(completion_fn=fake_completion).complete(
        [Message(content="hi")],
        spec=ModelSpec(model="anthropic/claude-sonnet-5", base_url="https://proxy/v1"),
        schema={"title": "Out", "type": "object"},
    )
    assert completion.parsed == {"ok": True}
    assert completion.cost_usd == 0.0004
    assert captured["api_base"] == "https://proxy/v1"
    assert captured["model"] == "anthropic/claude-sonnet-5"


def test_litellm_errors_are_wrapped() -> None:
    """Callers catch ProviderError, not a different exception type per vendor."""

    def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("rate limited")

    with pytest.raises(ProviderError, match="rate limited"):
        LiteLLMClient(completion_fn=boom).complete(
            [Message(content="hi")], spec=ModelSpec(model="anthropic/claude-sonnet-5")
        )


# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #


def test_grounding_defaults_to_a_cheaper_model_than_extraction() -> None:
    """The paper's precision story needs two models; the default must reflect it."""
    roles = ModelRoles()
    assert roles.ground.model != roles.extract.model
    assert roles.ground.max_tokens < roles.extract.max_tokens


def test_infer_defaults_to_the_extraction_model() -> None:
    assert ModelRoles().infer == ModelRoles().extract


def test_single_uses_one_model_everywhere() -> None:
    """The right call for a local setup, and it must not silently keep a default."""
    roles = ModelRoles.single("ollama/llama3.1")
    assert roles.extract == roles.ground == roles.infer
    assert roles.extract.model == "ollama/llama3.1"


def test_roles_can_mix_providers() -> None:
    """Frontier extraction, local grounding — the configuration worth supporting."""
    roles = ModelRoles(
        extract=ModelSpec(model="anthropic/claude-sonnet-5"),
        ground=ModelSpec(model="ollama/qwen2.5:3b"),
    )
    assert isinstance(roles.client_for("ground"), OpenAICompatClient)
    assert isinstance(roles.client_for("extract"), LiteLLMClient)


def test_an_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        ModelRoles().client_for("summarise")


def test_roles_round_trip_through_json() -> None:
    """A whole model configuration has to be loadable from a config file."""
    roles = ModelRoles.single("ollama/llama3.1")
    assert ModelRoles.model_validate_json(roles.model_dump_json()) == roles


# --------------------------------------------------------------------------- #
# The scripted client
# --------------------------------------------------------------------------- #


def test_scripted_client_satisfies_the_protocol_and_records_calls() -> None:
    client = ScriptedClient([{"name": "Ada"}, "plain text"])
    assert isinstance(client, LLMClient)
    spec = ModelSpec(model="test/model")
    first = client.complete([Message(content="a")], spec=spec, schema={"type": "object"})
    second = client.complete([Message(content="b")], spec=spec)
    assert first.parsed == {"name": "Ada"}
    assert second.text == "plain text"
    assert client.exhausted
    assert len(client.calls) == 2
    assert client.calls[0][2] == {"type": "object"}


def test_scripted_client_fails_loudly_when_it_runs_out() -> None:
    """A silent empty response would look like a model that extracted nothing."""
    with pytest.raises(ProviderError, match="ran out of responses"):
        ScriptedClient([]).complete([Message(content="a")], spec=ModelSpec(model="test/m"))
    assert ScriptedClient([], strict=False).complete(
        [Message(content="a")], spec=ModelSpec(model="test/m")
    ) == Completion(text="", model="test/m")
