"""Which providers openodke knows how to address, and where each one reads its key.

A model string is provider-qualified — `openai/gpt-5.5`, `ollama/llama3.1` — and
that prefix decides three things: which client serves the call (DECISIONS #7),
which environment variable holds the key, and whether a missing one is an error.
Keeping all three in one table is what lets `odke models` print the answer and
both adapters enforce it, instead of a user inferring it from a 401.

It is a table of *providers*, not of models. No context windows, no prices, no
capability flags: those change weekly, and a copy shipped in a package is wrong
by the next release. What a provider is called, and which variable it reads, is
stable enough to be worth writing down.

Keys are read from the environment and from nowhere else. Nothing here writes a
key to disk, and nothing here prints one.
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openodke.llm.base import MissingAPIKey, ModelSpec

DOCS_URL = "https://openodke.dev/models/"


@dataclass(frozen=True)
class Provider:
    """One provider: how to name a model for it, what it reads, and who serves it."""

    name: str
    # What the model string looks like, with the parts a user fills in.
    model_form: str
    key_env: str = ""
    # Further variables the provider needs. Reported by `odke models`, never enforced:
    # which of them matter is the provider's business, not ours.
    also: tuple[str, ...] = ()
    # A default endpoint is what makes a provider OpenAI-shaped here, so it is also
    # what decides that the standard-library client serves it (DECISIONS #7).
    base_url: str = ""
    # False where the provider also accepts its cloud's own credential chain, and an
    # unset variable therefore is not an error.
    key_required: bool = True

    @property
    def served_by(self) -> str:
        return "built-in" if self.base_url else "litellm"

    @property
    def nests_vendor(self) -> bool:
        """True where the model name itself carries a vendor: openrouter/anthropic/claude-…"""
        return self.model_form.count("/") > 1

    @property
    def envs(self) -> tuple[str, ...]:
        return ((self.key_env,) if self.key_env else ()) + self.also


# Ordered the way `odke models` prints it: what runs on the base install first.
PROVIDERS: tuple[Provider, ...] = (
    # Served by the standard-library OpenAI-compatible client — no extra dependency.
    Provider("openai", "openai/<model>", "OPENAI_API_KEY", base_url="https://api.openai.com/v1"),
    Provider(
        "openrouter",
        "openrouter/<vendor>/<model>",
        "OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    Provider(
        "together",
        "together/<vendor>/<model>",
        "TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
    ),
    Provider("groq", "groq/<model>", "GROQ_API_KEY", base_url="https://api.groq.com/openai/v1"),
    Provider(
        "deepseek", "deepseek/<model>", "DEEPSEEK_API_KEY", base_url="https://api.deepseek.com/v1"
    ),
    # Local servers: they accept any key, or none.
    Provider("ollama", "ollama/<model>", base_url="http://localhost:11434/v1"),
    Provider("vllm", "vllm/<model>", base_url="http://localhost:8000/v1"),
    Provider("lmstudio", "lmstudio/<model>", base_url="http://localhost:1234/v1"),
    Provider("llamacpp", "llamacpp/<model>", base_url="http://localhost:8080/v1"),
    # Served by litellm, which `pip install "openodke[llm]"` provides.
    Provider("anthropic", "anthropic/<model>", "ANTHROPIC_API_KEY"),
    Provider(
        "azure",
        "azure/<deployment>",
        "AZURE_API_KEY",
        also=("AZURE_API_BASE", "AZURE_API_VERSION"),
    ),
    Provider(
        "bedrock",
        "bedrock/<model>",
        "AWS_ACCESS_KEY_ID",
        also=("AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"),
        key_required=False,
    ),
    Provider(
        "vertex_ai",
        "vertex_ai/<model>",
        "GOOGLE_APPLICATION_CREDENTIALS",
        also=("VERTEXAI_PROJECT", "VERTEXAI_LOCATION"),
        key_required=False,
    ),
    Provider("gemini", "gemini/<model>", "GEMINI_API_KEY"),
    Provider("mistral", "mistral/<model>", "MISTRAL_API_KEY"),
    Provider("cohere", "cohere/<model>", "COHERE_API_KEY"),
    Provider("xai", "xai/<model>", "XAI_API_KEY"),
    Provider("perplexity", "perplexity/<model>", "PERPLEXITYAI_API_KEY"),
    Provider("fireworks_ai", "fireworks_ai/<model>", "FIREWORKS_AI_API_KEY"),
    Provider("cerebras", "cerebras/<model>", "CEREBRAS_API_KEY"),
    Provider("huggingface", "huggingface/<model>", "HF_TOKEN"),
)

BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}

# Both derived from the one table, so the routing rule in `registry.resolve` and the
# listing in `odke models` cannot come to disagree about who serves what.
DEFAULT_BASE_URLS: dict[str, str] = {p.name: p.base_url for p in PROVIDERS if p.base_url}
DEFAULT_KEY_ENVS: dict[str, str] = {p.name: p.key_env for p in PROVIDERS if p.base_url}


def provider_for(name: str) -> Provider | None:
    return BY_NAME.get(name.lower())


def qualify(model: str, provider: str | None) -> str:
    """A provider-qualified model string from `--model` and `--model-provider`.

    `--model-provider` names the provider when the string does not carry one, so
    `gpt-5.5` + `openai` is `openai/gpt-5.5`. A string that already names a
    *different* known provider is a contradiction rather than a nesting, and
    saying so beats picking one silently — except for a provider whose own model
    names carry a vendor, where `openrouter` + `anthropic/claude-…` is exactly
    what a user means.
    """
    if not provider:
        return model
    head, _, _ = model.partition("/")
    if head.lower() == provider.lower():
        return model
    target = provider_for(provider)
    if head.lower() in BY_NAME and not (target and target.nests_vendor):
        raise ValueError(
            f"--model {model!r} already names the provider {head!r}, "
            f"but --model-provider says {provider!r}"
        )
    return f"{provider}/{model}"


def key_env_for(spec: ModelSpec) -> str:
    """The variable this spec's key comes from, or "" when it needs none.

    A `base_url` sends the call somewhere the provider's own key was never meant
    to go — a GPU box, a proxy, often over plain HTTP — so only a variable the
    caller named with `api_key_env` is read for it.
    """
    if spec.api_key_env is not None:
        return spec.api_key_env
    if spec.base_url:
        return ""
    provider = provider_for(spec.provider)
    return provider.key_env if provider else ""


def require_key(spec: ModelSpec) -> str:
    """The key for this spec, or `MissingAPIKey` naming the variable that should hold it.

    Raised before the request goes out, because a provider's own 401 says nothing
    about which of twenty variables was meant to carry the key. Not raised when
    `base_url` redirects the call — the caller owns that endpoint's
    authentication, and no key is read for it (`key_env_for`) — nor for a
    provider that accepts its cloud's credential chain, where an unset variable
    is normal. An `api_key_env` the caller named is always enforced: naming it
    is saying the endpoint needs it.
    """
    env = key_env_for(spec)
    if not env:
        return ""
    key = os.environ.get(env, "")
    if key:
        return key
    provider = provider_for(spec.provider)
    if spec.api_key_env is None and provider and not provider.key_required:
        return ""
    raise MissingAPIKey(
        f"{spec.model} needs an API key in {env}, which is not set.\n"
        f"  export {env}=...   openodke reads keys from the environment and stores none\n"
        f"  odke models        every provider, its variable, and whether it is set"
    )


def litellm_installed() -> bool:
    """Whether `openodke[llm]` is present — found, not imported: litellm is slow to import."""
    try:
        return importlib.util.find_spec("litellm") is not None
    except (ImportError, ValueError):  # pragma: no cover - a broken sys.path
        return False


# --------------------------------------------------------------------------- #
# The listing
# --------------------------------------------------------------------------- #

_HEADINGS = ("provider", "model string", "key variable", "set", "served by")


def rows(
    environ: Mapping[str, str] | None = None, *, registered: Sequence[str] = ()
) -> list[tuple[str, str, str, str, str]]:
    """The table `describe()` prints, as data: one row per variable, never a value.

    A provider with two further variables gets a row each, with the first three
    columns blank, so that "is it set?" is answerable for every one of them.
    """
    env = os.environ if environ is None else environ
    known = {name.lower() for name in registered}
    out: list[tuple[str, str, str, str, str]] = []
    for provider in PROVIDERS:
        served = "registered" if provider.name in known else provider.served_by
        if not provider.envs:
            out.append((provider.name, provider.model_form, "(none)", "-", served))
            continue
        for i, name in enumerate(provider.envs):
            out.append(
                (
                    provider.name if i == 0 else "",
                    provider.model_form if i == 0 else "",
                    name,
                    "yes" if env.get(name) else "no",
                    served if i == 0 else "",
                )
            )
    # A registry entry for a provider this table has never heard of is still a way
    # to address a model, and is the only one whose key handling is not ours.
    for name in sorted(known - set(BY_NAME)):
        out.append((name, f"{name}/<model>", "(your client's)", "-", "registered"))
    return out


def describe(
    environ: Mapping[str, str] | None = None,
    *,
    litellm: bool | None = None,
    registered: Sequence[str] = (),
) -> str:
    """What `odke models` prints: every provider, its variable, and whether it is set.

    Takes the environment and the litellm answer rather than reading them, so the
    listing is a pure function of them and a test can state the situation it
    means. No value from `environ` is ever put in the output.
    """
    env = os.environ if environ is None else environ
    has_litellm = litellm_installed() if litellm is None else litellm

    table = rows(env, registered=registered)
    widths = [max(len(head), *(len(row[i]) for row in table)) for i, head in enumerate(_HEADINGS)]
    lines = [
        "Model strings are provider-qualified: <provider>/<model>.",
        "A string with no provider is read as openai/<model>.",
        "",
        _row(_HEADINGS, widths),
        _row(tuple("-" * w for w in widths), widths),
        *(_row(row, widths) for row in table),
        "",
        'Keys are read from the environment only. "set" never shows a value, and openodke',
        "writes no key to disk.",
        "",
        "built-in    the standard-library OpenAI-compatible client; no extra dependency",
    ]

    litellm_only = [p.name for p in PROVIDERS if p.served_by == "litellm"]
    if has_litellm:
        lines.append("litellm     the [llm] extra, which is installed: every row above works")
    else:
        lines.append("litellm     the [llm] extra, which is NOT installed. Without it these")
        lines.append("            cannot be called, whatever their key says:")
        lines += textwrap.wrap(
            ", ".join(litellm_only), width=78, initial_indent=" " * 14, subsequent_indent=" " * 14
        )
        lines.append('              pip install "openodke[llm]"')
    if any(row[4] == "registered" for row in table):
        lines.append("registered  a client registered with openodke.llm.register()")

    chained = [p.name for p in PROVIDERS if p.envs and not p.key_required]
    lines += [
        "",
        f"{' and '.join(chained)} also accept their cloud's own credential chain, so an",
        "unset variable there is not necessarily an error.",
        "",
        "Not in the table, and needing no entry:",
        "  any OpenAI-compatible endpoint — set base_url on the ModelSpec (vLLM, LM Studio,",
        "    llama.cpp, an OpenRouter-compatible proxy, a corporate gateway)",
        '  your own client — openodke.llm.register("gateway", factory)',
        "",
        DOCS_URL,
    ]
    return "\n".join(lines)


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()


__all__ = [
    "BY_NAME",
    "DEFAULT_BASE_URLS",
    "DEFAULT_KEY_ENVS",
    "DOCS_URL",
    "PROVIDERS",
    "Provider",
    "describe",
    "key_env_for",
    "litellm_installed",
    "provider_for",
    "qualify",
    "require_key",
    "rows",
]
