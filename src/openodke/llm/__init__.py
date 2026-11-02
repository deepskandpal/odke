"""Model access, deliberately provider-neutral.

Nothing elsewhere in this library imports a provider SDK or names a vendor. Every
model call goes through `LLMClient`, and which client serves a given model string
is decided by `resolve()`.

    from openodke.llm import ModelRoles, ModelSpec

    ModelRoles()                                        # Claude, two-model default
    ModelRoles.single("ollama/llama3.1")                # entirely local, no extras
    ModelRoles.single("openai/gpt-4.1")
    ModelRoles.single("azure/my-deployment", extra={"api_version": "2024-10-21"})
    ModelRoles(extract=ModelSpec(model="bedrock/anthropic.claude-sonnet-…"),
               ground=ModelSpec(model="ollama/qwen2.5:3b"))   # mix freely

Anything OpenAI-shaped — Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, Groq,
Together, DeepSeek, a LiteLLM proxy, a corporate gateway — works on the base
install with no extra dependency. Everything else routes through litellm, which
`pip install "openodke[llm]"` provides. Your own adapter is one `register()` call.

`openodke.llm.providers` is the table behind that: every provider openodke can
address, the variable each reads its key from, and who serves it. `odke models`
prints it. Keys come from the environment and are never stored or printed.

`CachedClient` wraps any client so that a request it has answered before is
answered again from the store, for nothing (`openodke.llm.cache`). A `Ledger`
holds a run to a `Budget`, and stops it with `BudgetExceeded` before a call
would go past it (`openodke.llm.budget`). `LimitedClient` holds every call to
a provider to that provider's limit, shared by the whole process
(`openodke.llm.limits`).
"""

from openodke.llm.base import (
    Completion,
    LLMClient,
    Message,
    MissingAPIKey,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
)
from openodke.llm.budget import Budget, BudgetExceeded, Ledger
from openodke.llm.cache import CachedClient, DirectoryCache, MemoryCache
from openodke.llm.limits import PROVIDER_LIMITS, LimitedClient, ProviderLimits, set_limit
from openodke.llm.openai_compat import DEFAULT_BASE_URLS, OpenAICompatClient
from openodke.llm.providers import PROVIDERS, Provider, key_env_for, qualify, require_key
from openodke.llm.registry import register, registered_providers, resolve, unregister
from openodke.llm.roles import ModelRoles
from openodke.llm.testing import (
    Cassette,
    RecordedClient,
    RecordingClient,
    ReplayClient,
    ScriptedClient,
)

__all__ = [
    "DEFAULT_BASE_URLS",
    "PROVIDERS",
    "PROVIDER_LIMITS",
    "Budget",
    "BudgetExceeded",
    "CachedClient",
    "Cassette",
    "Completion",
    "DirectoryCache",
    "LLMClient",
    "Ledger",
    "LimitedClient",
    "MemoryCache",
    "Message",
    "MissingAPIKey",
    "ModelRoles",
    "ModelSpec",
    "OpenAICompatClient",
    "Provider",
    "ProviderError",
    "ProviderLimits",
    "ProviderNotInstalled",
    "RecordedClient",
    "RecordingClient",
    "ReplayClient",
    "ScriptedClient",
    "key_env_for",
    "qualify",
    "register",
    "registered_providers",
    "require_key",
    "resolve",
    "set_limit",
    "unregister",
]
