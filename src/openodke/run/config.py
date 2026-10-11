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
      cache: .odke-cache                         # answer a repeated call from disk
      budget: {usd: 1.50, calls: 2000}           # stop cleanly here, keeping what is done
      limits: {anthropic: 8}                     # calls in flight per provider, all stages
    stages:
      chunker: {use: sentence, max_words: 120}
      extractor: hybrid                           # the one required stage
      grounder: llm
      gate: verdict
      sink: {use: jsonl, directory: out}
    bootstrap: false
    coverage: true                   # what extraction left behind; no model
    reextract: {windows: 3}          # hand those gaps back; off unless named
    store_lookup: neo4j              # resolve against the store; off unless named
    manifest: run.manifest.json      # the run manifest here too; see below
    batch_size: 500                  # stream: documents a micro-batch; off unless named
    tenant: acme                     # key the store's writes and reads by tenant

A stage is a built-in's short name, or `package.module:Name` for your own, and
either may take options: `{use: name, option: value, ...}`. A stage left out is
the pass-through from `openodke.stages` (DECISIONS #20) — except the scorer,
which `odke run` fills with `evidence` so that a written fact's confidence
reflects what the grounder found; `scorer: passthrough` opts out.

`store_lookup` resolves each batch against what the store already holds, with
the native resolver (DECISIONS #31): `neo4j` reads the run's Neo4j sink's
store, or one named by its own `uri`, and `package.module:Name` is a
`StoreLookup` of your own.

Every run writes a manifest (`openodke.manifest`): into each JSONL sink's
`manifest.json`, and to `manifest`, relative to this file, when it is named.
A run with neither writes it beside this file as `<name>.manifest.json`.

`batch_size` streams the run (#158, DECISIONS #45): that many documents are
loaded, run through every stage and written, then the next, so the run's
memory is one micro-batch's. Left out, the run is one batch. It is part of the
config a run manifest hashes, so a replay of a streamed run streams.

`tenant` keys everything the run writes to, and reads from, its store by that
tenant (#159, DECISIONS #44): two tenants' identical facts are two facts, and
a lookup, a merge or a retraction for one never reaches the other.

`gate` was `validator` in 0.2 (DECISIONS #26). The old key still works, with a
warning, until 2.0.0 (DECISIONS #48).
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from openodke._renamed import deprecated
from openodke.llm.base import ModelSpec
from openodke.llm.budget import Budget
from openodke.llm.roles import ModelRoles
from openodke.manifest import RunManifest, read_manifest
from openodke.tenants import tenant_name

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
    """`ModelRoles` by job, plus recorded responses, a cost meter, a response cache
    and a budget.

    `replay` maps a role to a file of recorded responses — a cassette object
    (`openodke.llm.ReplayClient`) or a list of match entries
    (`openodke.llm.RecordedClient`) — so a run needs no key and no network. `meter`
    wraps every model client in a `CostMeter` and puts the report in the graph's
    stats. `cache` names a directory of answers (`openodke.llm.cache`): a call
    asked before is answered from it, and every new answer is kept there.
    `budget` is the most the run may spend (`openodke.llm.budget`): `usd`,
    `calls`, `input_tokens`, `output_tokens`, each optional. `limits` caps the
    calls in flight to each provider it names, across every stage and the whole
    process (`openodke.llm.limits`).
    """

    extract: ModelSpec | None = None
    ground: ModelSpec | None = None
    infer: ModelSpec | None = None
    replay: dict[Role, str] = Field(default_factory=dict)
    meter: bool = False
    cache: str | None = None
    budget: Budget | None = None
    limits: dict[str, Annotated[int, Field(ge=1)]] = Field(default_factory=dict)

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

    def canonical(self) -> dict[str, Any]:
        """The block resolved: each role the spec `ModelRoles` gives it, the budget's limits."""
        data = self.model_dump(mode="json")
        roles = self.roles()
        for role in ("extract", "ground", "infer"):
            data[role] = getattr(roles, role).model_dump(mode="json")
        data["budget"] = self.budget.limits if self.budget is not None else None
        return data

    def with_budget(self, *, usd: float | None = None, calls: int | None = None) -> ModelsConfig:
        """These limits over the block's own budget; a limit given as None is kept as it was."""
        given = {k: v for k, v in (("usd", usd), ("calls", calls)) if v is not None}
        if not given:
            return self
        current = self.budget.model_dump() if self.budget is not None else {}
        return self.model_copy(update={"budget": Budget(**{**current, **given})})

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

    @property
    def validator(self) -> StageSpec | None:
        """The 0.2 name of `gate`; reading it warns until 2.0.0 (DECISIONS #26, #48)."""
        deprecated("StagesConfig.validator", "StagesConfig.gate")
        return self.gate

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


class ReextractConfig(_Strict):
    """Hand the coverage report's gaps back to the extractor (#102): `reextract:` in a config.

    `true` takes the defaults; a mapping sets `windows`, the most windows asked
    per document. The extractor stage must have a `reextract` method.
    """

    windows: int = Field(default=3, ge=1)


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
    # Count what extraction left behind in each document (`openodke.coverage`). Free.
    coverage: bool = True
    # Hand those gaps back to the extractor and ground what returns. Off unless named.
    reextract: ReextractConfig | None = None
    # Resolve against what the store already holds (DECISIONS #31). Off unless named.
    store_lookup: StageSpec | None = None
    # Where the run manifest is written, besides each JSONL sink's manifest.json.
    manifest: str | None = None
    # Documents a micro-batch: the run streams (#158). Left out, it is one batch.
    batch_size: int | None = Field(default=None, ge=1)
    # The tenant the store keys this run's writes and reads by (#159). None: no tenant.
    tenant: str | None = None

    @field_validator("tenant")
    @classmethod
    def _tenant(cls, value: str | None) -> str | None:
        return tenant_name(value)

    @model_validator(mode="before")
    @classmethod
    def _reextract_switch(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and isinstance(value.get("reextract"), bool):
            return {**value, "reextract": {} if value["reextract"] else None}
        return value

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)
    # The file this config was read from, when it was read from one.
    _source: Path | None = PrivateAttr(default=None)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def source(self) -> Path | None:
        """The file this config was read from; None for one built from a mapping."""
        return self._source

    def canonical(self) -> dict[str, Any]:
        """The config resolved: a mapping `parse_config` reads back as this run.

        Every key is filled in with its default, and every model role is the
        spec `ModelRoles` resolves it to, so a config that leaves a key or a
        role to its default and one that names that default are one config.
        Stages and inputs are in the shape a file gives them,
        `{use: name, option: ...}`, and a stage left out stays out. What a run
        manifest records and hashes (`openodke.manifest`).
        """
        data = self.model_dump(mode="json", exclude={"stages", "inputs", "store_lookup"})
        data["models"] = self.models.canonical()
        data["inputs"] = [
            {"path": item.path, **({"loader": _given(item.loader)} if item.loader else {})}
            for item in self.inputs
        ]
        stages: dict[str, Any] = {}
        for name in STAGES:
            spec = getattr(self.stages, name)
            if name == "sink":
                stages[name] = [_given(sink) for sink in spec]
            elif spec is not None:
                stages[name] = _given(spec)
        data["stages"] = stages
        data["store_lookup"] = _given(self.store_lookup) if self.store_lookup else None
        return data

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

    def with_cache(self, directory: str | Path) -> RunConfig:
        """This config answering from the cache in `directory`: what `--cache` applies.

        Resolved against the working directory, as a path given on the command line is.
        """
        absolute = str(Path(directory).expanduser().resolve())
        return self.model_copy(
            update={"models": self.models.model_copy(update={"cache": absolute})}
        )

    def with_budget(self, *, usd: float | None = None, calls: int | None = None) -> RunConfig:
        """This config with these limits on top of its own: what `--budget-usd` and
        `--budget-calls` apply."""
        return self.model_copy(update={"models": self.models.with_budget(usd=usd, calls=calls)})

    def with_batch_size(self, size: int | None) -> RunConfig:
        """This config streamed in micro-batches of `size`: what `--batch-size` applies.

        None keeps the config's own.
        """
        if size is None:
            return self
        if size < 1:
            raise ConfigError("--batch-size: at least 1")
        return self.model_copy(update={"batch_size": size})

    def with_tenant(self, tenant: str | None) -> RunConfig:
        """This config written and read as `tenant`: what `--tenant` applies. None keeps its own."""
        if tenant is None:
            return self
        try:
            return self.model_copy(update={"tenant": tenant_name(tenant)})
        except ValueError as exc:
            raise ConfigError(f"--tenant: {exc}") from None

    def with_widen(self) -> RunConfig:
        """This config with the model grounder's widen-and-retry on: what `--widen` applies."""
        grounder = self.stages.grounder
        if grounder is None or grounder.use != "llm":
            raise ConfigError("--widen needs the model grounder: stages.grounder: llm")
        spec = grounder.model_copy(update={"options": {**grounder.options, "widen": True}})
        return self.model_copy(update={"stages": self.stages.model_copy(update={"grounder": spec})})


def _given(spec: StageSpec) -> dict[str, Any]:
    """A stage as a config file gives it: `{use: name, option: value, ...}`."""
    plain: dict[str, Any] = json.loads(json.dumps(spec.options, default=str))
    return {"use": spec.use, **plain}


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
    config = parse_config(data, base_dir=source.parent, where=str(source))
    config._source = source.resolve()
    return config


def from_manifest(path: str | Path) -> tuple[RunConfig, RunManifest]:
    """The config an `odke run` manifest recorded, to run it again, and the manifest.

    `path` is the `manifest.json`, or a directory holding one. The config is
    the one the run resolved, its paths relative to the directory the run read
    it from. A manifest of `odke validate` or `odke ground` records that
    command's options instead, and is refused here.
    """
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"--from-manifest: {exc}") from None
    if manifest.command != "run" or manifest.base_dir is None:
        raise ConfigError(
            f"--from-manifest: {path} records odke {manifest.command}, not odke run; "
            "run that command again with the options under its `config`"
        )
    config = parse_config(manifest.config, base_dir=manifest.base_dir, where=str(path))
    if manifest.config_file is not None:
        config._source = Path(manifest.base_dir) / manifest.config_file
    return config, manifest


def load_models(path: str | Path) -> tuple[ModelsConfig, Path]:
    """The `models` block of a config file alone, and the directory its paths resolve from.

    What `odke ground` reads: a file that holds only a `models` block, or a whole
    run config, whose other keys are left to `odke run`. A key that belongs to
    neither is an error with a suggestion, as in `load_config`.
    """
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
    if not isinstance(data, Mapping) or not isinstance(data.get("models"), Mapping):
        raise ConfigError(f"{source}: no `models` block to read")
    for key in data:
        if key not in RunConfig.model_fields:
            close = difflib.get_close_matches(str(key), list(RunConfig.model_fields), n=1)
            hint = f" — did you mean {close[0]!r}?" if close else ""
            raise ConfigError(f"{source}: {key}: unknown key{hint}")
    try:
        models = ModelsConfig.model_validate(dict(data["models"]))
    except ValidationError as exc:
        problems = tuple(_format({**e, "loc": ("models", *e["loc"])}) for e in exc.errors())
        raise ConfigError(f"{source}: {problems[0]}", problems=problems) from None
    return models, source.parent.resolve()


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
        *ReextractConfig.model_fields,
        *ModelSpec.model_fields,
        *Budget.model_fields,
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
    "ReextractConfig",
    "RunConfig",
    "StageSpec",
    "StagesConfig",
    "from_manifest",
    "load_config",
    "load_models",
    "parse_config",
]
