"""Turning a model string into a client, and letting anyone override that.

The rule is deliberately simple and inspectable:

  1. an adapter explicitly registered for the provider wins;
  2. otherwise, if the provider speaks the OpenAI chat shape, use the
     dependency-free client — so Ollama and friends work on a bare install;
  3. otherwise use litellm, which knows the rest.

`register()` is the escape hatch. A team with an internal gateway registers one
callable at import time and every model string in their configs routes through it
without touching this library.
"""

from __future__ import annotations

from collections.abc import Callable

from odke.llm.base import LLMClient, ModelSpec, ProviderNotInstalled
from odke.llm.openai_compat import DEFAULT_BASE_URLS, OpenAICompatClient

ClientFactory = Callable[[ModelSpec], LLMClient]

_REGISTRY: dict[str, ClientFactory] = {}


def register(provider: str, factory: ClientFactory) -> None:
    """Route every `provider/...` model string through `factory`."""
    _REGISTRY[provider.lower()] = factory


def unregister(provider: str) -> None:
    _REGISTRY.pop(provider.lower(), None)


def registered_providers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve(spec: ModelSpec) -> LLMClient:
    """The client that should serve this spec."""
    provider = spec.provider.lower()

    factory = _REGISTRY.get(provider)
    if factory is not None:
        return factory(spec)

    # A custom base_url means the caller is pointing at something that speaks the
    # OpenAI shape — a proxy, a gateway, a local server. Honour that over litellm.
    if provider in DEFAULT_BASE_URLS or spec.base_url:
        return OpenAICompatClient()

    from odke.llm.litellm_client import LiteLLMClient

    try:
        return LiteLLMClient()
    except ProviderNotInstalled as exc:
        raise ProviderNotInstalled(
            f"{spec.model!r} needs a provider adapter. Either:\n"
            f'  pip install "odke[llm]"                     (adds ~100 providers)\n'
            f"  ModelSpec(model=..., base_url=...)          (any OpenAI-compatible endpoint)\n"
            f"  odke.llm.register({provider!r}, my_factory)  (your own adapter)"
        ) from exc


__all__ = ["ClientFactory", "register", "registered_providers", "resolve", "unregister"]
