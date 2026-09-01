"""Model access, deliberately provider-neutral.

Nothing elsewhere in this library imports a provider SDK or names a vendor. Every
model call goes through `LLMClient`, and which client serves a given model string
is decided by `resolve()`.

    from odke.llm import ModelRoles, ModelSpec

    ModelRoles()                                        # Claude, two-model default
    ModelRoles.single("ollama/llama3.1")                # entirely local, no extras
    ModelRoles.single("openai/gpt-4.1")
    ModelRoles.single("azure/my-deployment", extra={"api_version": "2024-10-21"})
    ModelRoles(extract=ModelSpec(model="bedrock/anthropic.claude-sonnet-…"),
               ground=ModelSpec(model="ollama/qwen2.5:3b"))   # mix freely

Anything OpenAI-shaped — Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, Groq,
Together, DeepSeek, a LiteLLM proxy, a corporate gateway — works on the base
install with no extra dependency. Everything else routes through litellm, which
`pip install "odke[llm]"` provides. Your own adapter is one `register()` call.
"""

from odke.llm.base import (
    Completion,
    LLMClient,
    Message,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
)
from odke.llm.openai_compat import DEFAULT_BASE_URLS, OpenAICompatClient
from odke.llm.registry import register, registered_providers, resolve, unregister
from odke.llm.roles import ModelRoles
from odke.llm.testing import ScriptedClient

__all__ = [
    "DEFAULT_BASE_URLS",
    "Completion",
    "LLMClient",
    "Message",
    "ModelRoles",
    "ModelSpec",
    "OpenAICompatClient",
    "ProviderError",
    "ProviderNotInstalled",
    "ScriptedClient",
    "register",
    "registered_providers",
    "resolve",
    "unregister",
]
