"""What `odke run` reads: one file that names every stage, and nothing implicit.

The config is data for the same reason `ModelSpec` is: it can be diffed, logged
next to the graph it produced, and read by someone who has never seen the code.
Every key is checked, so a misspelt stage name is an error with a suggestion
rather than a stage silently left as the pass-through.

    ontology: ontology.json          # paths are relative to this file
    inputs:
      - corpus/                      # the default loader: stages.loader, else directory
      - path: register.csv
        loader: {use: csv, tier: curated}
    models:
      extract: anthropic/claude-sonnet-5
      ground: anthropic/claude-haiku-4-5-20251001
      replay: {extract: recorded/extract.json}   # answer from recorded responses
      meter: true                                # cost, per stage
    stages:
      chunker: {use: sentence, max_words: 120}
      extractor: hybrid                           # the one required stage
      grounder: llm
      gate: verdict
      sink: {use: jsonl, directory: out}
    bootstrap: false

A stage is a built-in's short name, or `package.module:Name` for your own, and
either may take options: `{use: name, option: value, ...}`. A stage left out is
the pass-through from `openodke.stages` (DECISIONS #20) — except the scorer,
which `odke run` fills with `evidence` so that a written fact's confidence
reflects what the grounder found; `scorer: passthrough` opts out.

`gate` was `validator` in 0.2 (DECISIONS #26). The old key still works, with a
warning, until 1.0.0.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from openodke._renamed import deprecated
from openodke.llm.base import ModelSpec
from openodke.llm.roles import ModelRoles

# The thirteen, in pipeline order (DECISIONS #20).
STAGES = (
    "loader",
    "chunker",
    "router",
    "extractor",
    "grounder",
    "normalizer",
    "resolver",
    "corroborator",
    "scorer",
    "gate",
    "sink",
    "constrainer",
    "inferrer",
)

Role = Literal["extract", "ground", "infer"]


class ConfigError(ValueError):
    """A config that cannot run, explained one problem per line."""

    def __init__(self, message: str, *, problems: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.problems = problems or (message,)


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StageSpec(_Strict):
    """Which implementation fills a stage, and its options.

    `"llm"` and `{use: llm, max_workers: 4}` are both accepted; every key but
    `use` is an option passed to the implementation.
    """

    use: str
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"use": value}
        if isinstance(value, Mapping):
            out: dict[str, Any] = {"options": {k: v for k, v in value.items() if k != "use"}}
            if "use" in value:
                out["use"] = value["use"]
            return out
        return value

    @property
    def is_custom(self) -> bool:
        return ":" in self.use


class InputSpec(_Strict):
    """A file or directory, and the loader that reads it when not the default."""

    path: str
    loader: StageSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def _bare_path(cls, value: Any) -> Any:
        return {"path": value} if isinstance(value, str) else value


class ModelsConfig(_Strict):
    """`ModelRoles` by job, plus recorded responses and a cost meter.

    `replay` maps a role to a file of recorded responses — a cassette object
    (`openodke.llm.ReplayClient`) or a list of match entries
    (`openodke.llm.RecordedClient`) — so a run needs no key and no network. `meter`
    wraps every model client in a `CostMeter` and puts the report in the graph's
    stats.
    """

    extract: ModelSpec | None = None
    ground: ModelSpec | None = None
    infer: ModelSpec | None = None
    replay: dict[Role, str] = Field(default_factory=dict)
    meter: bool = False

    @model_validator(mode="before")
    @classmethod
    def _model_strings(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            key: {"model": item} if key in ("extract", "ground", "infer") and isinstance(item, str)
            else item
            for key, item in value.items()
        }  # fmt: skip

    def roles(self) -> ModelRoles:
        given = {
            role: spec for role in ("extract", "ground", "infer") if (spec := getattr(self, role))
        }
        return ModelRoles(**given)

    def with_model(self, model: str) -> ModelsConfig:
        """Every role on one model, keeping the rest of each role's spec.

        Built from `roles()` rather than from the fields, so a role the file left
        out keeps its `ModelRoles` default: grounding stays at `max_tokens=256`
        (DECISIONS #7a) instead of quietly becoming as expensive as extraction
        because a model string was overridden on the command line.
        """
        roles = self.roles()
        return self.model_copy(
            update={
                role: getattr(roles, role).model_copy(update={"model": model})
                for role in ("extract", "ground", "infer")
            }
        )


class StagesConfig(_Strict):
    """Which implementation fills each of the thirteen stages."""

    loader: StageSpec | None = None
    chunker: StageSpec | None = None
    router: StageSpec | None = None
    # The one stage with no identity function, so the one a config must name.
    extractor: StageSpec
    grounder: StageSpec | None = None
    normalizer: StageSpec | None = None
    resolver: StageSpec | None = None
    corroborator: StageSpec | None = None
    # Left out, `odke run` scores with `evidence` (`run.build.DEFAULTS`).
    scorer: StageSpec | None = None
    gate: StageSpec | None = None
    # One sink or several; none writes nothing.
    sink: tuple[StageSpec, ...] = ()
    constrainer: StageSpec | None = None
    inferrer: StageSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def _old_gate_key(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "validator" not in value:
            return value
        if "gate" in value:
            raise ValueError("`validator` is the old name of `gate`; give `gate` alone")
        # 3: past pydantic's `model_validate`, to the line that called it.
        deprecated("stages.validator", "stages.gate", stacklevel=3)
        return {("gate" if key == "validator" else key): item for key, item in value.items()}

    @model_validator(mode="before")
    @classmethod
    def _one_sink_or_many(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "sink" in value:
            sink = value["sink"]
            if sink is None:
                sink = []
            elif not isinstance(sink, list | tuple):
                sink = [sink]
            return {**value, "sink": sink}
        return value


class RunConfig(_Strict):
    """A whole run: inputs, ontology, models, the thirteen stages, and bootstrap.

    Relative paths resolve against `base_dir`, which `load_config` sets to the
    config file's own directory, so a config runs the same from any working
    directory.
    """

    ontology: str
    inputs: tuple[InputSpec, ...] = Field(min_length=1)
    # Directories put on `sys.path` before a `package.module:Name` stage is imported.
    pythonpath: tuple[str, ...] = ()
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    stages: StagesConfig
    # Apply the ontology's constraints through the sink before the first write.
    bootstrap: bool = False
    # Add each edge's inverse or symmetric partner (DECISIONS #28). Left out, it
    # is on exactly when the ontology declares one; `false` turns it off.
    inverses: bool | None = None

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else self._base_dir / candidate

    def with_model(self, model: str) -> RunConfig:
        """This config with every model role on `model`: what `odke run --model` applies.

        `base_dir` comes along because `model_copy` carries private attributes. A
        copy that resolved its paths against the working directory instead of the
        config file would break the promise that a config runs the same from anywhere.
        """
        return self.model_copy(update={"models": self.models.with_model(model)})

    def with_widen(self) -> RunConfig:
        """This config with the model grounder's widen-and-retry on: what `--widen` applies."""
        grounder = self.stages.grounder
        if grounder is None or grounder.use != "llm":
            raise ConfigError("--widen needs the model grounder: stages.grounder: llm")
        spec = grounder.model_copy(update={"options": {**grounder.options, "widen": True}})
        return self.model_copy(update={"stages": self.stages.model_copy(update={"grounder": spec})})


def load_config(path: str | Path) -> RunConfig:
    """A config from a YAML (`.yaml`, `.yml`) or JSON file, paths relative to it."""
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"{source}: no such file")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        data = _parse_yaml(text, str(source))
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{source}: line {exc.lineno} column {exc.colno}: {exc.msg}"
            ) from None
    return parse_config(data, base_dir=source.parent, where=str(source))


def parse_config(data: Any, *, base_dir: str | Path = ".", where: str | None = None) -> RunConfig:
    """A config from an already-parsed mapping, with every problem named by its dotted path."""
    prefix = f"{where}: " if where else ""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{prefix}the top level must be a mapping")
    try:
        config = RunConfig.model_validate(dict(data))
    except ValidationError as exc:
        problems = tuple(_format(error) for error in exc.errors())
        if len(problems) == 1:
            raise ConfigError(prefix + problems[0], problems=problems) from None
        lines = [f"{prefix}{len(problems)} problems", *(f"  {p}" for p in problems)]
        raise ConfigError("\n".join(lines), problems=problems) from None
    config._base_dir = Path(base_dir).resolve()
    return config


def _parse_yaml(text: str, where: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            'reading a YAML config needs PyYAML. Run: pip install "openodke[yaml]", '
            "or write the same keys as JSON"
        ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        at = f"line {mark.line + 1} column {mark.column + 1}: " if mark else ""
        raise ConfigError(f"{where}: {at}{getattr(exc, 'problem', None) or exc}") from None


_KNOWN_KEYS = sorted(
    {
        *RunConfig.model_fields,
        *StagesConfig.model_fields,
        *ModelsConfig.model_fields,
        *InputSpec.model_fields,
        *ModelSpec.model_fields,
    }
)


def _format(error: Mapping[str, Any]) -> str:
    loc = [part for part in error["loc"] if part != "options"]
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
    path = path or "(top level)"
    if error["type"] == "extra_forbidden":
        close = difflib.get_close_matches(str(loc[-1]), _KNOWN_KEYS, n=1)
        return f"{path}: unknown key" + (f" — did you mean {close[0]!r}?" if close else "")
    return f"{path}: {error['msg']}"


__all__ = [
    "STAGES",
    "ConfigError",
    "InputSpec",
    "ModelsConfig",
    "RunConfig",
    "StageSpec",
    "StagesConfig",
    "load_config",
    "parse_config",
]
