"""What every file loader shares: reading a source without changing a character."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from odke.stages import Source

Modality = Literal["unstructured", "semi_structured", "structured"]


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


__all__ = ["Modality", "read_text"]
