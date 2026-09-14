"""Pydantic models to an ontology: the path for users who already have the schema.

A Python-first user has written their entities once, as models, with field
descriptions already in them. Asking them to write the same thing again as JSON
is how the two copies drift. So a model becomes an `EntityType`, a field becomes
a `Predicate`, and the annotation says the rest:

- a field typed as another model passed in is an **edge** to that type;
- `list[X]`, `tuple[X, ...]`, `set[X]` are `cardinality="multi"`;
- `X | None`, or a default, is not `required`;
- a field description becomes the predicate description, a title its label;
- a model subclassing another model passed in names it as a parent, and does
  not redeclare the fields it inherited.

What cannot be mapped is reported by `Model.field` and the whole conversion
fails. A dropped field is a predicate the extractor is silently never asked
about, which is exactly the bug that gets blamed on the model.
"""

from __future__ import annotations

import inspect
from collections.abc import MutableSequence, MutableSet, Sequence
from collections.abc import Set as AbstractSet
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from types import NoneType, UnionType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel

from openodke.ontology.load import OntologyLoadError

if TYPE_CHECKING:
    from openodke.ontology import Ontology, Predicate

_LITERAL_RANGES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    Decimal: "number",
    bool: "boolean",
    date: "date",
    datetime: "datetime",
    UUID: "string",
}
_MANY = (list, set, frozenset, tuple, Sequence, MutableSequence, AbstractSet, MutableSet)


class _Unsupported(Exception):
    """An annotation with no ontology range; the message says what was found."""


def ontology_from_models(
    cls: type[Ontology], models: tuple[Any, ...], *, name: str, version: str
) -> Ontology:
    from openodke.ontology import EntityType, Predicate

    problems: list[str] = []
    passed = [m for m in models if isinstance(m, type) and issubclass(m, BaseModel)]
    problems += [f"{m!r}: not a pydantic model class" for m in models if m not in passed]
    by_name: dict[str, type[BaseModel]] = {}
    for model in passed:
        if by_name.setdefault(model.__name__, model) is not model:
            problems.append(f"{model.__name__}: two models passed with this name")

    types: dict[str, EntityType] = {}
    predicates: dict[str, Predicate] = {}
    declared_on: dict[str, str] = {}
    for model in passed:
        if not model.__pydantic_complete__:
            try:
                model.model_rebuild()
            except Exception as exc:  # pydantic raises its own error per failure kind
                problems.append(f"{model.__name__}: annotations do not resolve ({exc})")
                continue
        ancestors = [b for b in model.__mro__[1:] if b in passed]
        doc = model.__dict__.get("__doc__")
        types[model.__name__] = EntityType(
            name=model.__name__,
            description=inspect.cleandoc(doc) if doc else None,
            parents=tuple(b.__name__ for b in model.__bases__ if b in passed),
        )
        for field_name, field in model.model_fields.items():
            if any(field_name in a.model_fields for a in ancestors):
                continue  # declared on a parent type, inherited through lineage()
            where = f"{model.__name__}.{field_name}"
            try:
                range_, multi, optional = _read_annotation(field.annotation, passed)
            except _Unsupported as exc:
                problems.append(f"{where}: {exc}")
                continue
            candidate = Predicate(
                name=field_name,
                label=field.title,
                description=field.description,
                domain=(model.__name__,),
                range=range_,
                cardinality="multi" if multi else "single",
                required=field.is_required() and not optional,
                examples=tuple(str(e) for e in field.examples or ()),
            )
            existing = predicates.get(field_name)
            if existing is None:
                predicates[field_name] = candidate
                declared_on[field_name] = model.__name__
            elif (existing.range, existing.cardinality) != (range_, candidate.cardinality):
                problems.append(
                    f"{where}: {range_}/{candidate.cardinality} conflicts with "
                    f"{declared_on[field_name]}.{field_name} "
                    f"({existing.range}/{existing.cardinality}) — predicates are shared by "
                    "name across an ontology, so rename one of the fields"
                )
            else:
                predicates[field_name] = _merge(existing, candidate)

    if problems:
        head = problems[0] if len(problems) == 1 else f"{len(problems)} problems"
        lines = [head] if len(problems) == 1 else [head, *(f"  {p}" for p in problems)]
        raise OntologyLoadError("\n".join(lines), problems=tuple(problems))
    return cls(name=name, version=version, types=types, predicates=predicates)


def _merge(existing: Predicate, candidate: Predicate) -> Predicate:
    # One predicate, several domains. `required` is per predicate, not per
    # domain, so it holds only if every model that declares the field agrees.
    return existing.model_copy(
        update={
            "domain": existing.domain + candidate.domain,
            "required": existing.required and candidate.required,
            "label": existing.label or candidate.label,
            "description": existing.description or candidate.description,
            "examples": existing.examples or candidate.examples,
        }
    )


def _read_annotation(annotation: Any, passed: list[type[BaseModel]]) -> tuple[str, bool, bool]:
    """`(range, multi, optional)` for one field, or `_Unsupported` saying why not."""
    shown = _show(annotation)
    optional = False
    if get_origin(annotation) in (Union, UnionType):
        members = [a for a in get_args(annotation) if a is not NoneType]
        optional = len(members) < len(get_args(annotation))
        if len(members) != 1:
            raise _Unsupported(
                f"{shown} is a union — an ontology range is one type; "
                "split it into two fields or pick one"
            )
        annotation = members[0]
    annotation = _strip_annotated(annotation)
    multi = False
    origin = get_origin(annotation)
    if origin in _MANY or annotation in _MANY:
        args = get_args(annotation)
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            args = args[:1]
        if len(args) != 1:
            raise _Unsupported(f"{shown} has no single item type — use list[X] or tuple[X, ...]")
        annotation, multi = _strip_annotated(args[0]), True
    return _range_of(annotation, shown, passed), multi, optional


def _range_of(annotation: Any, shown: str, passed: list[type[BaseModel]]) -> str:
    if annotation in passed:
        return str(annotation.__name__)
    if annotation in _LITERAL_RANGES:
        return _LITERAL_RANGES[annotation]
    values: tuple[Any, ...] = ()
    if get_origin(annotation) is Literal:
        values = get_args(annotation)
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        values = tuple(member.value for member in annotation)
    kinds = {_LITERAL_RANGES.get(type(v)) for v in values}
    if values and len(kinds) == 1 and None not in kinds:
        return kinds.pop() or "string"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        raise _Unsupported(
            f"{shown} is a model that was not passed to from_pydantic — pass it too, "
            "so the field becomes an edge to it"
        )
    raise _Unsupported(
        f"{shown} has no ontology range — use {', '.join(sorted(_names()))}, "
        "an enum or Literal of one of those, a model passed in, or a list of any of them"
    )


def _strip_annotated(annotation: Any) -> Any:
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _names() -> set[str]:
    return {t.__name__ for t in _LITERAL_RANGES}


def _show(annotation: Any) -> str:
    if isinstance(annotation, type) and not get_args(annotation):
        return annotation.__name__
    return repr(annotation).replace("typing.", "")


__all__ = ["ontology_from_models"]
