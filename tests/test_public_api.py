"""The public surface, pinned, and every old name warning (DECISIONS #48).

A name in a public module's `__all__`, a command or flag of `odke`, and a key of
the run config are promises under semver. Each is held here to a deliberate
list, `public_api.json` beside this file, so a change to any of them is a change
to that file in the same diff, made on purpose and seen in review:

- adding a name, a flag or a key is a minor release, with a CHANGELOG line;
- removing or renaming one is a major release, after a minor that kept the old
  spelling working with a `DeprecationWarning` (listed in `OLD_NAMES` or
  `OLD_SPELLINGS` below, and tested to warn);
- a provisional module (`PROVISIONAL`) may change in a minor release, with a
  CHANGELOG line, and says so in its docstring.

A module is public unless a part of its dotted path starts with `_`, or it is
one of `PRIVATE`. A public module says what it exports in `__all__`; what it
leaves out is private, whatever its spelling.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import re
import typing
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import typer
from pydantic import BaseModel

import openodke

PIN = Path(__file__).with_name("public_api.json")
REMOVAL = "2.0.0"

# Modules with no Python promise: `odke`'s own code. The command is public, and
# pinned below by its commands and flags, not by the functions that serve them.
PRIVATE = frozenset({"openodke.cli", "openodke.cli.main"})

# Public, outside the semver promise until the reason is gone (DECISIONS #48).
PROVISIONAL = {
    "openodke.corroborate.judge": "the pair judge: off by default until its calibration card",
    "openodke.eval.equivalence": "the fact-equivalence judge: unvalidated until label set F's card",
    "openodke.eval.datasets": "benchmark adapters: they follow the datasets' own releases",
    "openodke.eval.datasets.redocred": "a benchmark adapter",
    "openodke.eval.datasets.text2kgbench": "a benchmark adapter",
    "openodke.eval.datasets.trex": "a benchmark adapter",
    "openodke.interop.graphrag": "reads neo4j-graphrag's objects, which change with it",
    "openodke.interop.langchain": "reads LangChain's objects, which change with it",
    "openodke.interop.langextract": "reads LangExtract's objects, which change with it",
    "openodke.reextract": "the re-extract hook: off by default, with one extractor behind it",
}

# Old names a module still serves, each the new object, with a warning:
# (module, old name, what the warning says to use).
OLD_NAMES = [
    ("openodke", "VerdictValidator", "openodke.VerdictGate"),
    ("openodke.validators", "VerdictValidator", "openodke.gate.VerdictGate"),
    ("openodke.stages", "Validator", "openodke.stages.Gate"),
    ("openodke.stages", "PassThroughValidator", "openodke.stages.PassThroughGate"),
    ("openodke.pipeline", "Validator", "openodke.stages.Gate"),
    ("openodke.pipeline", "PassThroughValidator", "openodke.stages.PassThroughGate"),
]


# --------------------------------------------------------------------------- #
# The surface, as it is
# --------------------------------------------------------------------------- #


def public_modules() -> Iterator[str]:
    yield "openodke"
    for found in pkgutil.walk_packages(openodke.__path__, "openodke."):
        if found.name in PRIVATE or any(p.startswith("_") for p in found.name.split(".")):
            continue
        yield found.name


def python_surface() -> dict[str, list[str]]:
    """Each public module's `__all__`, sorted."""
    surface: dict[str, list[str]] = {}
    for name in public_modules():
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None)
        assert exported is not None, f"{name} has no __all__: say what it exports (DECISIONS #48)"
        surface[name] = sorted(exported)
    return surface


def cli_surface() -> dict[str, list[str]]:
    """Every `odke` command, by its path, with its arguments and every spelling of its flags."""
    from openodke.cli.main import app

    surface: dict[str, list[str]] = {}

    def walk(command: Any, path: tuple[str, ...]) -> None:
        params: list[str] = []
        for param in command.params:
            if getattr(param, "param_type_name", "") == "option":
                params.extend(param.opts)
                params.extend(param.secondary_opts)
            else:
                params.append(f"<{param.name}>")
        surface[" ".join(("odke", *path))] = sorted(params)
        for name, sub in (getattr(command, "commands", None) or {}).items():
            walk(sub, (*path, name))

    walk(typer.main.get_command(app), ())
    return surface


def _models(annotation: Any) -> Iterator[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
    for arg in typing.get_args(annotation):
        yield from _models(arg)


def config_surface() -> dict[str, list[str]]:
    """The run config's keys, dotted, and the short names each stage and sink takes."""
    from openodke.run import RunConfig
    from openodke.run.build import BUILTINS, SINKS

    keys: list[str] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            key = prefix + (field.alias or name)
            keys.append(key)
            for sub in _models(field.annotation):
                walk(sub, key + ".")

    walk(RunConfig, "")
    short = {f"use.{stage}": sorted(names) for stage, names in BUILTINS.items()}
    return {"keys": sorted(set(keys)), **short, "use.sink": sorted(SINKS)}


def surface() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return {"python": python_surface(), "cli": cli_surface(), "config": config_surface()}


# --------------------------------------------------------------------------- #
# The surface, as it was promised
# --------------------------------------------------------------------------- #


def _changes(kind: str, pinned: dict[str, list[str]], found: dict[str, list[str]]) -> list[str]:
    lines = []
    for key in sorted(pinned.keys() - found.keys()):
        lines.append(f"  {kind} {key!r} is gone, and public_api.json promises it")
    for key in sorted(found.keys() - pinned.keys()):
        lines.append(f"  {kind} {key!r} is new, and not in public_api.json: {found[key]}")
    for key in sorted(pinned.keys() & found.keys()):
        added = sorted(set(found[key]) - set(pinned[key]))
        removed = sorted(set(pinned[key]) - set(found[key]))
        if added:
            lines.append(f"  {kind} {key!r} has {added}, which public_api.json does not promise")
        if removed:
            lines.append(f"  {kind} {key!r} lost {removed}, which public_api.json promises")
    return lines


HOW = (
    "\nThe public surface is a semver promise (DECISIONS #48). If this change is "
    "deliberate, edit tests/public_api.json to match and add a CHANGELOG line: an "
    "addition is a minor release; a removal or a rename keeps the old spelling "
    "working with a DeprecationWarning until 2.0.0 (see OLD_NAMES in this file)."
)


@pytest.fixture(scope="module")
def pinned() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(PIN.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def found() -> dict[str, Any]:
    return surface()


@pytest.mark.parametrize(
    ("part", "kind"), [("python", "module"), ("cli", "command"), ("config", "config")]
)
def test_the_public_surface_is_the_one_promised(
    pinned: dict[str, Any], found: dict[str, Any], part: str, kind: str
) -> None:
    changes = _changes(kind, pinned[part], found[part])
    assert not changes, "the public surface changed:\n" + "\n".join(changes) + HOW


def test_every_exported_name_exists() -> None:
    for name in public_modules():
        module = importlib.import_module(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            missing = [n for n in module.__all__ if not hasattr(module, n)]
        assert not missing, f"{name}.__all__ names what it does not have: {missing}"


def test_a_name_that_warns_is_an_old_name_on_the_list() -> None:
    """A deprecation is a decision: it is listed here, and so tested to warn properly."""
    listed = {(module, old) for module, old, _ in OLD_NAMES}
    for name in public_modules():
        module = importlib.import_module(name)
        for exported in module.__all__:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                try:
                    getattr(module, exported)
                except DeprecationWarning:
                    assert (name, exported) in listed, f"{name}.{exported} warns but is not listed"


def test_provisional_modules_are_public_and_say_so() -> None:
    public = set(public_modules())
    for name in PROVISIONAL:
        assert name in public, f"{name} is listed as provisional but is not a public module"
        doc = importlib.import_module(name).__doc__ or ""
        assert "Provisional (DECISIONS #48)" in doc, f"{name}'s docstring does not say so"


# --------------------------------------------------------------------------- #
# Every old spelling warns, names the new one, and says when it goes
# --------------------------------------------------------------------------- #


def _config(tmp_path: Path, **stages: str) -> Any:
    from openodke.run import parse_config

    (tmp_path / "ontology.json").write_text('{"name": "x"}', encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Ada.", encoding="utf-8")
    data = {
        "ontology": "ontology.json",
        "inputs": [{"path": "notes.txt"}],
        "stages": {"extractor": "pattern", **stages},
    }
    return parse_config(data, base_dir=tmp_path)


def _old_gate_config(tmp_path: Path) -> None:
    _config(tmp_path, validator="verdict")


def _built_pipeline(tmp_path: Path) -> None:
    from openodke.run import build
    from openodke.stages import PassThroughGate

    build(_config(tmp_path)).pipeline(validator=PassThroughGate())


def _pipeline(**stages: Any) -> Any:
    from openodke import Ontology, PatternExtractor, Pipeline

    return Pipeline(Ontology(), PatternExtractor(), **stages)


def _pipeline_with_validator(tmp_path: Path) -> None:
    from openodke.stages import PassThroughGate

    _pipeline(validator=PassThroughGate())


def _read_pipeline_validator(tmp_path: Path) -> None:
    pipeline = _pipeline()
    pipeline.validator  # noqa: B018


def _set_pipeline_validator(tmp_path: Path) -> None:
    from openodke.stages import PassThroughGate

    _pipeline().validator = PassThroughGate()


def _stages_config_validator(tmp_path: Path) -> None:
    _config(tmp_path, gate="verdict").stages.validator  # noqa: B018


def _cardinality_scope(tmp_path: Path) -> None:
    from openodke import Predicate
    from openodke.sinks.neo4j import cardinality_scope

    cardinality_scope(Predicate(name="p"))


# Old spellings that are not module attributes: (old, new, a call that uses it).
OLD_SPELLINGS: list[tuple[str, str, Callable[[Path], None]]] = [
    ("Pipeline(validator=...)", "Pipeline(gate=...)", _pipeline_with_validator),
    ("Pipeline.validator", "Pipeline.gate", _read_pipeline_validator),
    ("Pipeline.validator", "Pipeline.gate", _set_pipeline_validator),
    ("Built.pipeline(validator=...)", "Built.pipeline(gate=...)", _built_pipeline),
    ("stages.validator", "stages.gate", _old_gate_config),
    ("StagesConfig.validator", "StagesConfig.gate", _stages_config_validator),
    (
        "openodke.sinks.neo4j.cardinality_scope(predicate)",
        "predicate.scope_keys",
        _cardinality_scope,
    ),
]


def _says(old: str, new: str) -> str:
    return (
        rf"^{re.escape(old)} is deprecated: use {re.escape(new)}\. "
        rf"The old name is removed in {re.escape(REMOVAL)} \(DECISIONS #48\)\.$"
    )


@pytest.mark.parametrize(
    ("module", "old", "new"), OLD_NAMES, ids=[f"{m}.{o}" for m, o, _ in OLD_NAMES]
)
def test_every_old_name_warns_with_its_new_name_and_its_removal(
    module: str, old: str, new: str
) -> None:
    found = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=_says(f"{module}.{old}", new)) as record:
        value = getattr(found, old)
    assert len(record) == 1
    target, _, attr = new.rpartition(".")
    assert value is getattr(importlib.import_module(target), attr)


@pytest.mark.parametrize(
    ("old", "new", "use"),
    OLD_SPELLINGS,
    ids=[f"{o}-{i}" for i, (o, _, _) in enumerate(OLD_SPELLINGS)],
)
def test_every_old_spelling_warns_with_its_new_name_and_its_removal(
    old: str, new: str, use: Callable[[Path], None], tmp_path: Path
) -> None:
    with pytest.warns(DeprecationWarning, match=_says(old, new)):
        use(tmp_path)
