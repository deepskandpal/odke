"""Everything else, via litellm.

litellm already maintains the wire format for a hundred providers — Anthropic,
OpenAI, Azure, Bedrock, Vertex, Gemini, Mistral, Cohere, Groq, Together,
OpenRouter, Ollama and the rest. Reimplementing that is a maintenance tax paid
forever for something nobody chose this library for.

Requires `pip install "openodke[llm]"`. The base install deliberately does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openodke.llm.base import (
    Completion,
    Message,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
)
from openodke.llm.openai_compat import _key, _maybe_json
from openodke.llm.providers import require_key

# What a redirected call is handed when the caller named no key: vLLM's own word
# for "none", and nobody's secret. Handed nothing, litellm reads the vendor's key
# from the environment and sends it wherever `api_base` points.
_NO_KEY = "EMPTY"


class LiteLLMClient:
    """A thin adapter. Anything clever belongs on one side of it or the other."""

    def __init__(self, *, completion_fn: Any = None) -> None:
        self._checks_key = completion_fn is None
        if completion_fn is not None:
            # The test suite injects a recorded transport here, so the litellm
            # path is executed in CI rather than skipped. An injected transport
            # answers for itself, so it is not asked for a key.
            self._completion = completion_fn
            return
        try:
            from litellm import completion
        except ImportError as exc:  # pragma: no cover - exercised by extras test
            raise ProviderNotInstalled(
                'litellm is not installed. Run: pip install "openodke[llm]" — or use '
                "OpenAICompatClient, which needs no extra dependency."
            ) from exc
        self._completion = completion

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        # litellm reads the key from the environment itself; this only makes a
        # missing one say which variable, before the request rather than after.
        key = require_key(spec) if self._checks_key else _key(spec)
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": spec.max_tokens,
            "timeout": spec.timeout,
        }
        # Left out rather than defaulted, and not `drop_params`, which would
        # discard a temperature the caller did mean (#79).
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        kwargs.update(spec.extra)
        if spec.base_url:
            kwargs["api_base"] = spec.base_url
            # Only a key the caller named goes to an endpoint the caller chose.
            kwargs["api_key"] = key or _NO_KEY
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "output"), "schema": schema},
            }

        try:
            response = self._completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - litellm raises provider-specific types
            raise ProviderError(f"{spec.model} call failed: {exc}") from exc

        body = response if isinstance(response, dict) else response.model_dump()
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            parsed=_maybe_json(text),
            model=body.get("model", spec.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=body.get("_response_cost"),
            raw=body,
        )


__all__ = ["LiteLLMClient"]
