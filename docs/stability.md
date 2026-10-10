# Stability

From 1.0, openodke follows [semantic versioning](https://semver.org/). What it
covers, how long a deprecated name lives and what "provisional" allows are
written once, in [DECISIONS #48](decisions.md#48). This page says which parts of
the package are which.

## Public

A name is public when a public module lists it in `__all__`. A module is public
unless a part of its path starts with `_`. The `odke` command is public by its
commands, flags and exit statuses; `openodke.cli` is not a Python API. The run
config is public by its keys and the short names `use:` takes. A test pins all
of it, so none of it changes by accident.

## Provisional

Public and documented, but may change in a minor release, with a CHANGELOG
line. Each module says so at the top of its docstring.

| Module | Why |
|---|---|
| `openodke.corroborate.judge`, and the resolver's `normalize_batch` and `embed` | The pair judge and batch normalisation are off by default until their calibration card ([DECISIONS #34](decisions.md#34), [#43](decisions.md#43)) |
| `openodke.eval.equivalence` | The fact-equivalence judge's lenient score is unvalidated until its calibration card ([DECISIONS #39](decisions.md#39)) |
| `openodke.eval.datasets` | The benchmark adapters follow the datasets' own releases |
| `openodke.interop.langchain`, `.langextract`, `.graphrag` | They read other libraries' objects, which change with those libraries |
| `openodke.reextract` | The re-extract hook is off by default, with one extractor behind it |

The OpenTelemetry spans' names and attributes are provisional too, and so is
which version of a [registered prompt](models.md#prompts) a stage sends. The
keys themselves are never edited or removed.

## Formats

Each format a reader outside Python depends on names its version, and its JSON
Schema ships in the package. A minor version adds optional fields; a reader
refuses another major version by name.

| Format | Version field | Now | Schema |
|---|---|---|---|
| [Triples input](inputs.md) | `schema_version`, on each row, optional | 1.0 | `openodke/interop/triples.schema.json` |
| [Run report](validator.md#the-report), `ValidationReport` | `schema_version` | 1.0 | `openodke/validation_report.schema.json` |
| [Eval report](evaluation.md) | `schema_version` | 1.2 | `openodke/eval/eval_report.schema.json` |
| [Run manifest](run.md#the-run-manifest) | `manifest_version` | 1 | |

## Deprecated

Each still works, warns with a `DeprecationWarning` that names what to use, and
is removed in 2.0.0. Python shows such a warning only in a script's own code
and under a test runner; `python -W default::DeprecationWarning` shows every
one. `odke run` prints a deprecated config key's warning itself.

| Deprecated | Use |
|---|---|
| `openodke.VerdictValidator`, `openodke.validators.VerdictValidator` | `openodke.VerdictGate` |
| `openodke.stages.Validator`, `openodke.pipeline.Validator` | `openodke.Gate` |
| `openodke.stages.PassThroughValidator`, `openodke.pipeline.PassThroughValidator` | `openodke.stages.PassThroughGate` |
| `Pipeline(validator=...)`, `Pipeline.validator` | `Pipeline(gate=...)`, `Pipeline.gate` |
| `Built.pipeline(validator=...)`, `StagesConfig.validator` | `gate` |
| `validator:` under `stages` in a run config | `gate:` |
| `openodke.sinks.neo4j.cardinality_scope(predicate)` | `predicate.scope_keys` |

`openodke.Validator` named the gate in 0.2 and is the Validator, the whole
layer, from 1.0 ([DECISIONS #26](decisions.md#26)).
