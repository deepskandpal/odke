"""`odke run`: the whole pipeline from one config file.

Three modules, in the order a run goes through them. `config` reads and checks
the file (`RunConfig`, `load_config`). `build` turns it into stage objects —
short names for the built-ins, `package.module:Name` for your own, the
pass-through for anything left out. `execute` bootstraps the store, loads the
inputs, runs the pipeline, copies every stage's counts into
`KnowledgeGraph.stats`, and writes — or, on a dry run, says what it would have
written.

Everything the CLI does is reachable from here: `execute(load_config(path))`.
"""

from openodke.run.build import BUILTINS, Built, build, with_path_ids
from openodke.run.config import (
    STAGES,
    ConfigError,
    InputSpec,
    ModelsConfig,
    RunConfig,
    StagesConfig,
    StageSpec,
    load_config,
    parse_config,
)
from openodke.run.execute import RunResult, collect_stats, execute, run_built

__all__ = [
    "BUILTINS",
    "STAGES",
    "Built",
    "ConfigError",
    "InputSpec",
    "ModelsConfig",
    "RunConfig",
    "RunResult",
    "StageSpec",
    "StagesConfig",
    "build",
    "collect_stats",
    "execute",
    "load_config",
    "parse_config",
    "run_built",
    "with_path_ids",
]
