"""The model seam.

Every model call in this library goes through `LLMClient`. Nothing in the
extractor, the grounder or the inference path imports a provider SDK, names a
vendor, or assumes a wire format — they ask a client for a completion and get one
back. Swapping Claude for a local Llama is a constructor argument.

`LLMClient` is a Protocol, so a caller who already has their own gateway, their
own retry policy or their own audit logging implements one method and passes it
in. No base class to inherit, nothing to register.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One turn. Deliberately the lowest common denominator across providers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = "user"
    content: str


class ModelSpec(BaseModel):
    """Which model to call, and how.

    Held as data rather than as constructor arguments so a whole configuration
    can come from a YAML file, be logged next to the run it produced, and be
    compared against the next run's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # "provider/model" — anthropic/claude-sonnet-5, openai/gpt-4.1,
    # ollama/llama3.1, azure/my-deployment, openrouter/…, bedrock/…, vertex/…
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0
    # For self-hosted and OpenAI-compatible endpoints: Ollama, vLLM, LM Studio,
    # llama.cpp, LiteLLM proxy, or a corporate gateway.
    base_url: str | None = None
    api_key_env: str | None = None
    # Anything provider-specific that has no business in this model: reasoning
    # effort, safety settings, an Azure api_version, a Bedrock region.
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0] if "/" in self.model else "openai"


class Completion(BaseModel):
    """What came back, plus what it cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    # Populated when the call used structured output and the provider returned
    # parsed JSON rather than a string that merely looks like JSON.
    parsed: dict[str, Any] | None = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Providers report cost inconsistently or not at all; None means "unknown",
    # which is honest, where 0.0 would be a lie.
    cost_usd: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """The one method the rest of the library depends on.

    `schema`, when given, is a JSON Schema the response must conform to. Clients
    that have native structured output should use it; clients that do not must
    fall back to instructing the model and parsing, so that callers never have to
    branch on which provider they configured.
    """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion: ...


class ProviderError(RuntimeError):
    """A provider call failed in a way the caller may want to handle."""


class ProviderNotInstalled(ProviderError):
    """The adapter for this provider needs a package that is not installed."""


__all__ = [
    "Completion",
    "LLMClient",
    "Message",
    "ModelSpec",
    "ProviderError",
    "ProviderNotInstalled",
]
