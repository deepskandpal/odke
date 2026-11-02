"""What every file loader shares: reading a source without changing a character."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from openodke.stages import Source
from openodke.types import Document

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


def path_parts(doc: Document, base: Path) -> tuple[str, str] | None:
    """A document's file, relative to `base`, and the `#L<line>` or `#<row>` of its record.

    None for a document read from no file. A file outside `base` keeps its
    absolute path.
    """
    if not doc.uri or not doc.uri.startswith("file:"):
        return None
    path = Path(unquote(urlsplit(doc.uri).path))
    try:
        name = path.relative_to(base).as_posix()
    except ValueError:
        name = path.as_posix()
    line, row = doc.metadata.get("line"), doc.metadata.get("row_index")
    if isinstance(line, int):
        return name, f"#L{line}"
    if isinstance(row, int):
        return name, f"#{row}"
    return name, ""


def with_path_ids(docs: Iterable[Document], base: Path) -> list[Document]:
    """The documents re-keyed by source path: `texts/a.txt`, `corpus/register.csv#L2`.

    A file's path relative to `base`, plus `#L<line>` for a record with a
    source line or `#<row>` for one without. A name two documents share gets
    `~2`, `~3` after the first. A document read from no file keeps its id.
    """
    seen: dict[str, int] = {}
    out: list[Document] = []
    for doc in docs:
        parts = path_parts(doc, base)
        if parts is None:
            out.append(doc)
            continue
        name = "".join(parts)
        count = seen[name] = seen.get(name, 0) + 1
        out.append(doc.model_copy(update={"id": name if count == 1 else f"{name}~{count}"}))
    return out


__all__ = [
    "MissingExtraError",
    "MissingExtraWarning",
    "Modality",
    "import_extra",
    "read_bytes",
    "read_text",
]
