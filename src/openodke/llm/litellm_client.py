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

from openodke._text import without_password
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

# litellm's own names for an endpoint. A spec can pass one through `extra`
# instead of `base_url` (litellm documents `api_base` for proxies), and the call
# goes there all the same.
_ENDPOINTS = ("api_base", "base_url")


def _endpoint(spec: ModelSpec) -> str | None:
    """Where this call goes when it is not the provider's own endpoint: `base_url`, else `extra`."""
    if spec.base_url:
        return spec.base_url
    named = [str(spec.extra[name]) for name in _ENDPOINTS if spec.extra.get(name)]
    return named[0] if named else None


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
        # The key is read for where the call goes: an endpoint in `extra`
        # redirects it as `base_url` does, so only `api_key_env` is read for it
        # (security audit, 5 Nov 2026). Read here, a missing key also says which
        # variable, before the request rather than after.
        redirect = _endpoint(spec)
        keyed = spec
        if redirect != spec.base_url:
            keyed = spec.model_copy(update={"base_url": redirect})
        key = require_key(keyed) if self._checks_key else _key(keyed)
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
        kwargs.update({k: v for k, v in spec.extra.items() if k not in _ENDPOINTS})
        if redirect:
            kwargs["api_base"] = redirect
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
            # litellm's message can quote `api_base`, password and all (release audit 9).
            raise ProviderError(f"{spec.model} call failed: {without_password(str(exc))}") from exc

        if isinstance(response, dict):
            body, cost = response, response.get("_response_cost")
        else:
            body = response.model_dump()
            # litellm keeps its cost beside the response rather than in it, so
            # `model_dump()` leaves it behind.
            cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            parsed=_maybe_json(text),
            model=body.get("model", spec.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=cost,
            raw=body,
        )


__all__ = ["LiteLLMClient"]
