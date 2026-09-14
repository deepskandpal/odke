"""Getting a schema in, and saying what was wrong with it when it fails.

The error messages here decide whether a user gets to a second run. A pydantic
traceback names the right key in the wrong language, forty lines down; what a
user needs is one line per problem, each starting with the dotted path to the
key that caused it — `predicates.employer.cardinality` — and ending with what
was found there. Syntax errors carry a line and column instead, because that is
where the editor's cursor needs to go.

`Ontology.from_json` and friends are thin wrappers over this module. It never
imports `odke.ontology` at module level: the model classes arrive as `cls`, so
the package can import this file first and a subclass loads as itself.
"""

from __future__ import annotations

import difflib
import json
import warnings
from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from odke.ontology import Ontology

OntologyT = TypeVar("OntologyT", bound="Ontology")
Source = str | PathLike[str]


class OntologyLoadError(ValueError):
    """A schema that could not become an `Ontology`, explained one problem per line.

    `problems` holds the same lines individually, for a tool that wants to
    render them itself.
    """

    def __init__(self, message: str, *, problems: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.problems = problems or (message,)


class OntologyImportWarning(UserWarning):
    """What an importer run with `strict=False` could not carry over, one problem per line.

    The same lines `OntologyLoadError` would have raised with, so choosing to
    load a large public ontology anyway never means choosing not to be told.
    """

    def __init__(self, message: str, *, problems: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.problems = problems or (message,)


def report_problems(
    problems: Sequence[str], *, strict: bool, where: str | None, stacklevel: int = 3
) -> None:
    """Raise everything an importer could not map, or with `strict=False` warn about it once."""
    if not problems:
        return
    listed = tuple(problems)
    message = _headline(where, listed)
    if strict:
        raise OntologyLoadError(message, problems=listed)
    warnings.warn(OntologyImportWarning(message, problems=listed), stacklevel=stacklevel)


def load_dict(
    cls: type[OntologyT], data: Any, *, strict: bool, where: str | None = None
) -> OntologyT:
    if not isinstance(data, Mapping):
        raise OntologyLoadError(
            _prefix(where) + f"the top level must be a mapping of name/types/predicates, "
            f"not {_kind(data)}"
        )
    try:
        ontology = cls.model_validate(dict(data))
    except ValidationError as exc:
        problems = tuple(_format(error) for error in exc.errors())
        # `from None`: the pydantic traceback is the thing being replaced.
        raise OntologyLoadError(_headline(where, problems), problems=problems) from None
    if strict:
        # Errors block loading and warnings do not: a warning is a schema that
        # works, and refusing it would push people to strict=False for good.
        errors = tuple(
            f"{d.path}: {d.message} [{d.code}]"
            for d in ontology.validate()
            if d.severity == "error"
        )
        if errors:
            raise OntologyLoadError(_headline(where, errors), problems=errors)
    return ontology


def load_json(cls: type[OntologyT], source: Source, *, strict: bool) -> OntologyT:
    text, where = _read(source)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OntologyLoadError(
            _prefix(where) + f"line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from None
    return load_dict(cls, data, strict=strict, where=where)


def load_yaml(cls: type[OntologyT], source: Source, *, strict: bool) -> OntologyT:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError('PyYAML is not installed. Run: pip install "odke[yaml]"') from exc
    text, where = _read(source)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        at = f"line {mark.line + 1} column {mark.column + 1}: " if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise OntologyLoadError(_prefix(where) + at + str(problem)) from None
    return load_dict(cls, data, strict=strict, where=where)


def _read(source: Source) -> tuple[str, str | None]:
    """The document text and, when it came from a file, the name to blame.

    A `str` is ambiguous. It is the document itself when it looks like one — it
    has a newline, or opens the way JSON or YAML opens — and a path otherwise.
    Checked in that order so a large document is never handed to the
    filesystem as a file name.
    """
    if isinstance(source, str) and (
        "\n" in source or source.lstrip().startswith(("{", "[", "---", "#", "%"))
    ):
        return source, None
    path = Path(fspath(source))
    if not path.is_file():
        raise OntologyLoadError(f"{path}: no such file")
    return path.read_text(encoding="utf-8"), str(path)


def _format(error: Mapping[str, Any]) -> str:
    loc = tuple(error["loc"])
    path = _dotted(loc) or "(top level)"
    if error["type"] == "extra_forbidden":
        return f"{path}: unknown key{_suggest(loc)}"
    found = error.get("input")
    shown = f" (got {found!r})" if isinstance(found, str | int | float | bool | None) else ""
    return f"{path}: {error['msg']}{shown}"


def _suggest(loc: tuple[Any, ...]) -> str:
    """'did you mean' for a misspelt key, or the full list when nothing is close."""
    owner = _model_at(loc[:-1])
    if owner is None:
        return ""
    fields = list(owner.model_fields)
    close = difflib.get_close_matches(str(loc[-1]), fields, n=1)
    if close:
        return f" — did you mean {close[0]!r}?"
    return f" — {owner.__name__} accepts: {', '.join(fields)}"


def _model_at(parts: tuple[Any, ...]) -> type[BaseModel] | None:
    # Imported here: this module is imported by the package that defines them.
    from odke.ontology import EntityType, Ontology, Predicate, Qualifier

    if not parts:
        return Ontology
    if parts[0] == "types" and len(parts) == 2:
        return EntityType
    if parts[0] == "predicates":
        if len(parts) == 2:
            return Predicate
        if len(parts) == 4 and parts[2] == "qualifiers":
            return Qualifier
    return None


def _dotted(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _headline(where: str | None, problems: tuple[str, ...]) -> str:
    if len(problems) == 1:
        return _prefix(where) + problems[0]
    lines = [_prefix(where) + f"{len(problems)} problems"]
    return "\n".join(lines + [f"  {p}" for p in problems])


def _prefix(where: str | None) -> str:
    return f"{where}: " if where else ""


def _kind(value: Any) -> str:
    if value is None:
        return "an empty document"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "a string"
    return type(value).__name__


__all__ = [
    "OntologyImportWarning",
    "OntologyLoadError",
    "load_dict",
    "load_json",
    "load_yaml",
    "report_problems",
]
