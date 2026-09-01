"""Everything else, via litellm.

litellm already maintains the wire format for a hundred providers — Anthropic,
OpenAI, Azure, Bedrock, Vertex, Gemini, Mistral, Cohere, Groq, Together,
OpenRouter, Ollama and the rest. Reimplementing that is a maintenance tax paid
forever for something nobody chose this library for.

Requires `pip install "odke[llm]"`. The base install deliberately does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from odke.llm.base import (
    Completion,
    Message,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
)
from odke.llm.openai_compat import _maybe_json


class LiteLLMClient:
    """A thin adapter. Anything clever belongs on one side of it or the other."""

    def __init__(self, *, completion_fn: Any = None) -> None:
        if completion_fn is not None:
            # The test suite injects a recorded transport here, so the litellm
            # path is executed in CI rather than skipped.
            self._completion = completion_fn
            return
        try:
            from litellm import completion
        except ImportError as exc:  # pragma: no cover - exercised by extras test
            raise ProviderNotInstalled(
                'litellm is not installed. Run: pip install "odke[llm]" — or use '
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
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
            "timeout": spec.timeout,
            **spec.extra,
        }
        if spec.base_url:
            kwargs["api_base"] = spec.base_url
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
