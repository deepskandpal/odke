"""Which model does which job.

The paper's precision story depends on *two* models, not one: a capable model
extracts, and a small cheap one verifies. Making that a first-class shape rather
than a convention means the asymmetry survives configuration — and that nobody
accidentally pays frontier prices to answer ten thousand yes/no questions.

Every role defaults to the previous one, so `ModelRoles(extract=…)` alone is a
valid, working configuration on that one model, with grounding at its own small
`max_tokens`. Only `ModelRoles()`, naming nothing, gets the two-model default.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from openodke.llm.base import LLMClient, ModelSpec
from openodke.llm.registry import resolve

# Sensible starting points, not a hard-coded vendor. Every one of these is
# overridable, and nothing in the library breaks if they are replaced with
# ollama/… or openai/… or a gateway string.
#
# Each is an exact model id, and changes only with a new calibration card
# (DECISIONS #7a). Anthropic publishes Sonnet 5.5 under `claude-sonnet-5-5`
# alone, with no dated snapshot, and it is the model the published bench ran on.
DEFAULT_EXTRACT = "anthropic/claude-sonnet-5-5"
DEFAULT_GROUND = "anthropic/claude-haiku-4-5-20251001"
# One word comes back, whichever model is asked.
_GROUND_MAX_TOKENS = 256


class ModelRoles(BaseModel):
    """The models a pipeline uses, by job.

    No role sets a temperature. A default one model accepts is one the next model
    rejects, so each provider's own applies until a caller asks for something
    else (`ModelSpec.temperature`, #79).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    extract: ModelSpec = ModelSpec(model=DEFAULT_EXTRACT)
    # Cheap and small on purpose: one fact, one span, one yes/no.
    ground: ModelSpec = ModelSpec(model=DEFAULT_GROUND, max_tokens=_GROUND_MAX_TOKENS)
    # Schema proposal (M5). Defaults to the extraction model.
    infer: ModelSpec | None = None

    @model_validator(mode="after")
    def _default_to_extract(self) -> ModelRoles:
        # A caller who names only the extraction model — a local server, say —
        # has chosen where passages go. The default grounder would quietly send
        # them to another vendor, so grounding follows extraction instead.
        if "extract" in self.model_fields_set and "ground" not in self.model_fields_set:
            ground = self.extract.model_copy(update={"max_tokens": _GROUND_MAX_TOKENS})
            object.__setattr__(self, "ground", ground)
        if self.infer is None:
            object.__setattr__(self, "infer", self.extract)
        return self

    @classmethod
    def single(cls, model: str, **kwargs: object) -> ModelRoles:
        """Use one model everywhere — the right call for a local setup.

        ModelRoles.single("ollama/llama3.1")
        """
        spec = ModelSpec(model=model, **kwargs)  # type: ignore[arg-type]
        return cls(extract=spec, ground=spec, infer=spec)

    def client_for(self, role: str) -> LLMClient:
        spec = getattr(self, role, None)
        if not isinstance(spec, ModelSpec):
            raise ValueError(f"unknown role {role!r}; expected extract, ground or infer")
        return resolve(spec)


__all__ = ["DEFAULT_EXTRACT", "DEFAULT_GROUND", "ModelRoles"]
