"""A reader keeps what it knows of a newer minor version (DECISIONS #48, #52).

A minor release may add an optional field anywhere in a format. A reader from an
older minor cannot know that field, so it leaves it out and reads the rest,
rather than refusing what a newer openodke wrote (release audit finding 18 and
call F). A field it does not know in a document of its own version, or an older
one, is still refused: there it is a typo, or a field from nowhere.

The run manifest's version is one number, so its reader keeps what it knows of
any manifest (`openodke.manifest.read_manifest`). The triples input and the eval
report are `major.minor`: their readers leave unknown fields only where the
document names a later minor of the same major (`newer_minor`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, get_args, get_origin

from pydantic import BaseModel


def newer_minor(version: object, reads: str) -> bool:
    """True when `version` is a later `major.minor` of the major `reads` names."""
    if not isinstance(version, str):
        return False
    major, _, minor = version.partition(".")
    own_major, _, own_minor = reads.partition(".")
    return (
        major == own_major
        and minor.isdigit()
        and own_minor.isdigit()
        and int(minor) > int(own_minor)
    )


def known(model: type[BaseModel], data: Any) -> Any:
    """`data` with only the fields `model` knows, in it and in every model it holds.

    A plain mapping the model holds (`dict[str, Any]`, a recorded config) is kept
    whole: it has no fields to know.
    """
    if not isinstance(data, Mapping):
        return data
    fields = model.model_fields
    return {name: within(fields[name].annotation, data[name]) for name in fields if name in data}


def within(annotation: Any, value: Any) -> Any:
    """`value` as `annotation` holds it: a model, a mapping or list of them, or one or None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return known(annotation, value)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is dict and len(args) == 2 and isinstance(value, Mapping):
        return {key: within(args[1], item) for key, item in value.items()}
    if origin in (list, tuple, Sequence) and isinstance(value, list | tuple):
        items = [arg for arg in args if arg is not Ellipsis]
        if len(items) == 1:
            return [within(items[0], item) for item in value]
        return value
    models = [arg for arg in args if isinstance(arg, type) and issubclass(arg, BaseModel)]
    return within(models[0], value) if len(models) == 1 else value
