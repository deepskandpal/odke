"""Offset-keeping text helpers shared by the loaders and the pattern extractor."""

from __future__ import annotations

import re
from collections.abc import Iterator

_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)?")


def iter_lines(text: str, start: int = 0, end: int | None = None) -> Iterator[tuple[int, int]]:
    """`(start, end)` of every line in `text[start:end]`, line break excluded.

    Not `str.splitlines`: that also breaks on form feeds and Unicode separators,
    which a Markdown table or a `Key: value` block never does, and it returns
    strings rather than the offsets a span needs.
    """
    stop = len(text) if end is None else end
    pos = start
    while pos < stop:
        match = _LINE.match(text, pos, stop)
        if match is None:  # pragma: no cover - the pattern matches the empty string
            break
        line_end = match.end()
        while line_end > pos and text[line_end - 1] in "\r\n":
            line_end -= 1
        yield pos, line_end
        pos = match.end()


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Narrow `[start, end)` past whitespace on both sides, without copying."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
