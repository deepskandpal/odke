"""Which model does which job.

The paper's precision story depends on *two* models, not one: a capable model
extracts, and a small cheap one verifies. Making that a first-class shape rather
than a convention means the asymmetry survives configuration — and that nobody
accidentally pays frontier prices to answer ten thousand yes/no questions.

Every role defaults to the previous one, so `ModelRoles(extract="…")` alone is a
valid, working configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from odke.llm.base import LLMClient, ModelSpec
from odke.llm.registry import resolve

# Sensible starting points, not a hard-coded vendor. Every one of these is
# overridable, and nothing in the library breaks if they are replaced with
# ollama/… or openai/… or a gateway string.
DEFAULT_EXTRACT = "anthropic/claude-sonnet-5"
DEFAULT_GROUND = "anthropic/claude-haiku-4-5-20251001"


class ModelRoles(BaseModel):
    """The models a pipeline uses, by job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extract: ModelSpec = ModelSpec(model=DEFAULT_EXTRACT)
    # Cheap and small on purpose: one fact, one span, one yes/no.
    ground: ModelSpec = ModelSpec(model=DEFAULT_GROUND, max_tokens=256)
    # Schema proposal (M5). Defaults to the extraction model.
    infer: ModelSpec | None = None

    @model_validator(mode="after")
    def _default_infer_to_extract(self) -> ModelRoles:
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
