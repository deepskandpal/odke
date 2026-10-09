"""Provider discovery: what `odke models` says, and what a missing key says.

Two promises are load-bearing here and are asserted rather than documented: the
listing runs on a base install with no key and no network, and no output path
ever carries the value of an environment variable.
"""

from __future__ import annotations

import io
import json
import urllib.request
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact, Span
from openodke.cli.main import app
from openodke.ground import LLMGrounder, is_transient
from openodke.llm import (
    Message,
    MissingAPIKey,
    ModelRoles,
    ModelSpec,
    OpenAICompatClient,
    register,
    unregister,
)
from openodke.llm.litellm_client import LiteLLMClient
from openodke.llm.providers import (
    BY_NAME,
    DEFAULT_BASE_URLS,
    DEFAULT_KEY_ENVS,
    PROVIDERS,
    describe,
    key_env_for,
    qualify,
    require_key,
    rows,
)
from openodke.llm.testing import ScriptedClient

runner = CliRunner()

# A value that must never appear in any output. Not a real key, and never sent.
SECRET = "sk-test-value-that-must-not-be-printed"


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from an environment with none of the table's variables set."""
    for provider in PROVIDERS:
        for name in provider.envs:
            monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def test_every_provider_is_named_once_and_says_how_to_address_it() -> None:
    names = [p.name for p in PROVIDERS]
    assert len(names) == len(set(names))
    assert all(p.model_form.startswith(f"{p.name}/") for p in PROVIDERS)
    # The five the issue started from, plus Azure's trio and litellm's own names.
    assert {"openai", "openrouter", "together", "groq", "deepseek"} <= set(names)
    assert {"anthropic", "azure", "bedrock", "vertex_ai", "gemini", "mistral"} <= set(names)
    assert BY_NAME["azure"].envs == ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION")


def test_the_routing_rule_and_the_listing_read_the_same_table() -> None:
    """`resolve` picks the built-in client by base URL; the listing must agree."""
    built_in = {p.name: p.base_url for p in PROVIDERS if p.served_by == "built-in"}
    assert built_in == DEFAULT_BASE_URLS
    assert set(DEFAULT_KEY_ENVS) == set(DEFAULT_BASE_URLS)
    assert DEFAULT_KEY_ENVS["openai"] == "OPENAI_API_KEY"
    assert DEFAULT_KEY_ENVS["ollama"] == ""


def test_no_model_names_are_shipped() -> None:
    """DECISIONS: a catalogue of models would be a table of facts that rot."""
    listing = describe({}, litellm=False)
    for token in ("gpt-4", "claude-sonnet", "llama3.1", "context window", "$"):
        assert token not in listing


# --------------------------------------------------------------------------- #
# odke models
# --------------------------------------------------------------------------- #


def test_odke_models_runs_with_no_keys_and_names_every_provider() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.output
    for provider in PROVIDERS:
        assert provider.model_form in result.output
        for name in provider.envs:
            assert name in result.output
    assert "built-in" in result.output and "litellm" in result.output


def test_a_variable_is_reported_set_or_unset_and_never_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.output
    assert SECRET not in result.output
    by_env = {row[2]: row for row in rows()}  # the real environment, as the command reads it
    assert by_env["GROQ_API_KEY"][3] == "yes"
    assert by_env["OPENAI_API_KEY"][3] == "no"
    # Nothing to set, so nothing to report: local servers take any key, or none.
    assert by_env["(none)"][3] == "-"


def test_an_empty_variable_counts_as_unset() -> None:
    assert dict((r[2], r[3]) for r in rows({"OPENAI_API_KEY": ""}))["OPENAI_API_KEY"] == "no"


def test_the_listing_says_plainly_when_the_llm_extra_is_absent() -> None:
    absent = describe({}, litellm=False)
    assert "NOT installed" in absent
    for provider in PROVIDERS:
        if provider.served_by == "litellm":
            assert provider.name in absent.split("NOT installed", 1)[1]
    assert 'pip install "openodke[llm]"' in absent
    assert "NOT installed" not in describe({}, litellm=True)


def test_a_registered_client_is_shown_as_the_one_that_serves_its_provider() -> None:
    served = {row[0]: row[4] for row in rows({}, registered=("gateway", "ollama"))}
    assert served["gateway"] == "registered"  # not in the table at all
    assert served["ollama"] == "registered"  # in the table, overridden
    assert served["openai"] == "built-in"


def test_odke_models_shows_what_register_added() -> None:
    register("mycorp", lambda spec: ScriptedClient(["x"]))
    try:
        result = runner.invoke(app, ["models"])
    finally:
        unregister("mycorp")
    assert result.exit_code == 0, result.output
    assert "mycorp/<model>" in result.output
    assert "registered" in result.output


# --------------------------------------------------------------------------- #
# Key resolution
# --------------------------------------------------------------------------- #


def test_the_variable_a_spec_reads_comes_from_the_table_or_the_spec() -> None:
    assert key_env_for(ModelSpec(model="groq/llama-3.3-70b")) == "GROQ_API_KEY"
    assert key_env_for(ModelSpec(model="azure/deploy")) == "AZURE_API_KEY"
    assert key_env_for(ModelSpec(model="ollama/llama3.1")) == ""
    assert key_env_for(ModelSpec(model="whoknows/x")) == ""
    assert key_env_for(ModelSpec(model="groq/x", api_key_env="MY_KEY")) == "MY_KEY"


def test_a_missing_key_names_the_variable_and_says_nothing_else() -> None:
    with pytest.raises(MissingAPIKey) as raised:
        require_key(ModelSpec(model="openai/gpt-5.5"))
    message = str(raised.value)
    assert "OPENAI_API_KEY" in message and "openai/gpt-5.5" in message
    assert "odke models" in message
    # The fix is a variable, not a file: nothing here suggests storing a key.
    assert "stores none" in message


def test_a_present_key_is_returned_and_a_local_provider_needs_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert require_key(ModelSpec(model="groq/llama-3.3-70b")) == SECRET
    assert require_key(ModelSpec(model="ollama/llama3.1")) == ""
    assert require_key(ModelSpec(model="whoknows/x")) == ""


def test_a_redirected_call_and_a_credential_chain_are_not_missing_keys() -> None:
    """`base_url` means the caller owns that endpoint's auth; bedrock has its own chain."""
    assert require_key(ModelSpec(model="openai/x", base_url="http://localhost:8000/v1")) == ""
    assert require_key(ModelSpec(model="bedrock/anthropic.claude-sonnet-4")) == ""
    # Naming the variable is saying the endpoint needs it, so it is enforced.
    with pytest.raises(MissingAPIKey, match="GATEWAY_TOKEN"):
        require_key(
            ModelSpec(model="corp/x", base_url="https://llm.internal", api_key_env="GATEWAY_TOKEN")
        )


def test_a_missing_key_is_not_retried() -> None:
    """Waiting does not set an environment variable."""
    assert is_transient(MissingAPIKey("openai/gpt-5.5 needs an API key in OPENAI_API_KEY")) is False


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a request was made before the key was checked")


def test_the_builtin_client_refuses_before_it_opens_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _never)
    client = OpenAICompatClient()  # the default opener, which is now _never
    with pytest.raises(MissingAPIKey, match="OPENAI_API_KEY"):
        client.complete([Message(content="hi")], spec=ModelSpec(model="openai/gpt-5.5"))


def test_the_litellm_client_refuses_before_it_calls_the_provider() -> None:
    pytest.importorskip("litellm")
    with pytest.raises(MissingAPIKey, match="ANTHROPIC_API_KEY"):
        LiteLLMClient().complete(
            [Message(content="hi")], spec=ModelSpec(model="anthropic/claude-sonnet-5")
        )


def test_a_grounder_with_no_key_raises_rather_than_leaving_every_fact_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallowed per fact, this was a run that verified nothing and exited 0."""
    monkeypatch.setattr(urllib.request, "urlopen", _never)
    doc = Document(text="Ada Lovelace was born in 1815.")
    span = Span(doc_id=doc.id, start=0, end=len(doc.text), quote=doc.text)
    fact = Fact(
        subject=Entity(key="ada", type="Person"),
        predicate="born",
        object_value="1815",
        evidence=(Evidence(doc_id=doc.id, span=span),),
    )
    grounder = LLMGrounder(ModelRoles.single("openai/gpt-5.5"))
    with pytest.raises(MissingAPIKey, match="OPENAI_API_KEY"):
        grounder.ground(fact, doc)


class _FakeHTTP:
    """An injected opener: it authenticates itself, so it is never asked for a key."""

    def __init__(self) -> None:
        self.request: Any = None

    def __call__(self, request: Any, timeout: float | None = None) -> Any:
        self.request = request
        body = {"choices": [{"message": {"content": "ok"}}]}
        stream = io.BytesIO(json.dumps(body).encode())
        stream.__enter__ = lambda: stream  # type: ignore[method-assign]
        stream.__exit__ = lambda *a: None  # type: ignore[method-assign]
        return stream


def test_an_injected_transport_is_not_asked_for_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    http = _FakeHTTP()
    completion = OpenAICompatClient(opener=http).complete(
        [Message(content="hi")], spec=ModelSpec(model="openai/gpt-5.5")
    )
    assert completion.text == "ok"
    assert "Authorization" not in http.request.headers
    # And it still sends the key when there is one.
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    OpenAICompatClient(opener=http).complete(
        [Message(content="hi")], spec=ModelSpec(model="openai/gpt-5.5")
    )
    assert http.request.headers["Authorization"] == f"Bearer {SECRET}"


def test_a_vendor_key_never_follows_a_call_to_another_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With OPENAI_API_KEY exported, a self-hosted server on plain HTTP was sent it."""
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    http = _FakeHTTP()
    monkeypatch.setattr(urllib.request, "urlopen", http)
    gpu = ModelSpec(model="llama3.1", base_url="http://gpu-01:8000/v1")
    assert key_env_for(gpu) == ""
    OpenAICompatClient().complete([Message(content="hi")], spec=gpu)
    assert http.request.full_url == "http://gpu-01:8000/v1/chat/completions"
    assert "Authorization" not in http.request.headers
    # Naming the variable is how a caller says that endpoint wants a key.
    monkeypatch.setenv("GPU_TOKEN", "gpu-token")
    named = gpu.model_copy(update={"api_key_env": "GPU_TOKEN"})
    OpenAICompatClient().complete([Message(content="hi")], spec=named)
    assert http.request.headers["Authorization"] == "Bearer gpu-token"
    # The vendor's own endpoint still gets the vendor's own key.
    OpenAICompatClient().complete([Message(content="hi")], spec=ModelSpec(model="openai/gpt-5.5"))
    assert http.request.headers["Authorization"] == f"Bearer {SECRET}"


def test_litellm_is_not_left_to_find_the_vendor_key_for_a_redirected_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm falls back to ANTHROPIC_API_KEY whenever it is handed no key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.clear()
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    client = LiteLLMClient(completion_fn=fake_completion)
    gpu = ModelSpec(model="anthropic/claude-sonnet-5", base_url="http://gpu-01:8000/v1")
    client.complete([Message(content="hi")], spec=gpu)
    assert captured["api_base"] == "http://gpu-01:8000/v1"
    assert captured["api_key"] and SECRET not in captured["api_key"]
    monkeypatch.setenv("GPU_TOKEN", "gpu-token")
    client.complete(
        [Message(content="hi")], spec=gpu.model_copy(update={"api_key_env": "GPU_TOKEN"})
    )
    assert captured["api_key"] == "gpu-token"
    # Not redirected, litellm reads the provider's own variable as it always has.
    client.complete([Message(content="hi")], spec=ModelSpec(model="anthropic/claude-sonnet-5"))
    assert "api_key" not in captured


# --------------------------------------------------------------------------- #
# --model / --model-provider
# --------------------------------------------------------------------------- #


def test_a_provider_qualifies_a_bare_model_string() -> None:
    assert qualify("gpt-5.5", "openai") == "openai/gpt-5.5"
    assert qualify("openai/gpt-5.5", "openai") == "openai/gpt-5.5"
    assert qualify("openai/gpt-5.5", None) == "openai/gpt-5.5"
    # OpenRouter's own model names carry a vendor, so this nests rather than clashes.
    assert (
        qualify("anthropic/claude-sonnet-5", "openrouter") == "openrouter/anthropic/claude-sonnet-5"
    )
    assert qualify("meta-llama/Llama-3-70b", "together") == "together/meta-llama/Llama-3-70b"


def test_a_provider_that_contradicts_the_string_is_an_error() -> None:
    with pytest.raises(ValueError, match="already names the provider"):
        qualify("anthropic/claude-sonnet-5", "openai")


def test_the_cli_refuses_a_provider_with_no_model(tmp_path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "x.json"), "--model-provider", "openai"])
    assert result.exit_code == 2
    assert "--model-provider needs --model" in result.output
