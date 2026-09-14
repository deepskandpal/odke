"""What every file loader shares: reading a source without changing a character."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Literal

from openodke.stages import Source

Modality = Literal["unstructured", "semi_structured", "structured"]


class MissingExtraError(ImportError):
    """A loader needs an optional dependency that is not installed.

    A class of its own so `DirectoryLoader` can skip the one file that needs the
    extra and keep walking, without also swallowing an `ImportError` that is a
    real bug.
    """


class MissingExtraWarning(UserWarning):
    """`DirectoryLoader` skipped a file because its loader's extra is not installed."""


def import_extra(module: str, *, extra: str, package: str, reading: str) -> Any:
    """Import a loader's optional dependency, or say which extra provides it.

    Called when a file is read, never when a module is imported, so the base
    install can name every loader and a directory walk needs only the extras for
    the files it actually meets.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f'reading {reading} needs {package}. Run: pip install "openodke[{extra}]"'
        ) from exc


def read_text(source: Source, encoding: str) -> tuple[str, str | None]:
    """The source's text, and its URI when it has one.

    A path is read as bytes and decoded, never opened in text mode: text mode
    turns CRLF into LF, and every offset after the first Windows line ending
    would then point short of the file the caller actually has.
    """
    if isinstance(source, bytes | bytearray | memoryview):
        return bytes(source).decode(encoding), None
    if isinstance(source, str | os.PathLike):
        path = Path(source)
        return path.read_bytes().decode(encoding), path.resolve().as_uri()
    raise TypeError(f"expected a path or bytes, got {type(source).__name__}")


def read_bytes(source: Source) -> tuple[bytes, str | None]:
    """The source's bytes, and its URI when it has one, for formats that are not text."""
    if isinstance(source, bytes | bytearray | memoryview):
        return bytes(source), None
    if isinstance(source, str | os.PathLike):
        path = Path(source)
        return path.read_bytes(), path.resolve().as_uri()
    raise TypeError(f"expected a path or bytes, got {type(source).__name__}")


__all__ = [
    "MissingExtraError",
    "MissingExtraWarning",
    "Modality",
    "import_extra",
    "read_bytes",
    "read_text",
]
