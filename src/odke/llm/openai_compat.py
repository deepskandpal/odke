"""An OpenAI-compatible chat client written against the standard library.

This exists so that the most common self-hosted setups work with `pip install
odke` and nothing else: Ollama, vLLM, LM Studio, llama.cpp's server, text
-generation-inference, an OpenRouter key, a LiteLLM proxy, or a corporate
gateway that speaks the same shape. All of them expose `/chat/completions`, and
none of them are worth a dependency.

For hosted providers with their own wire formats — Bedrock, Vertex, Azure's
older API versions — use `LiteLLMClient` instead.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from odke.llm.base import Completion, Message, ModelSpec, ProviderError

# Endpoints people actually run locally, so `ollama/llama3.1` needs no base_url.
DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "llamacpp": "http://localhost:8080/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}

DEFAULT_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    # Local servers accept any key, or none.
    "ollama": "",
    "vllm": "",
    "lmstudio": "",
    "llamacpp": "",
}


class OpenAICompatClient:
    """Chat completions over HTTP, with no third-party dependency."""

    def __init__(self, *, opener: Any = None) -> None:
        # Injected so the test suite can replay recorded responses without a
        # network, rather than monkeypatching urllib globally.
        self._opener = opener or urllib.request.urlopen

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        provider = spec.provider
        base = spec.base_url or DEFAULT_BASE_URLS.get(provider)
        if not base:
            raise ProviderError(
                f"no base_url for provider {provider!r}; set ModelSpec.base_url "
                f"or use a known provider ({', '.join(sorted(DEFAULT_BASE_URLS))})"
            )

        payload: dict[str, Any] = {
            "model": spec.model.split("/", 1)[-1],
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
            **spec.extra,
        }
        if schema is not None:
            # Servers that do not know this key ignore it; the caller still gets
            # JSON because the prompt asks for it too.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "output"), "schema": schema},
            }

        headers = {"Content-Type": "application/json"}
        key_env = spec.api_key_env or DEFAULT_KEY_ENVS.get(provider, "")
        key = os.environ.get(key_env, "") if key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"

        request = urllib.request.Request(
            f"{base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=spec.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network shape
            raise ProviderError(f"{provider} returned {exc.code}: {exc.read()[:400]!r}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network shape
            raise ProviderError(f"could not reach {provider} at {base}: {exc.reason}") from exc

        return _to_completion(body, spec)


def _to_completion(body: dict[str, Any], spec: ModelSpec) -> Completion:
    try:
        text = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"unexpected response shape from {spec.provider}: {body!r}") from exc
    usage = body.get("usage") or {}
    return Completion(
        text=text,
        parsed=_maybe_json(text),
        model=body.get("model", spec.model),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        raw=body,
    )


def _maybe_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object if that is what came back, else None.

    Not an error when it fails: plenty of calls in this library ask for prose.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


__all__ = ["DEFAULT_BASE_URLS", "DEFAULT_KEY_ENVS", "OpenAICompatClient"]
