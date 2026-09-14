"""One record, rendered to text that a span can point into.

A row from a CSV or a JSON array has no prose to cite, but a fact from it still
needs evidence a grounder can check (DECISIONS #3). So a record becomes a
document whose `text` is one `path: value` line per leaf, rendered the same way
every time — same record, same text, same offsets — with the record itself in
`metadata["row"]` and the offsets of every rendered value in
`metadata["fields"]`. The pattern extractor reads typed values from the row and
cites the rendered cell; it never parses its own rendering back.

Paths are the part of JSONPath that records need: `name`, `address.city`,
`tags[0]`, `["key.with.dots"]`, and `tags[*]` in a mapping.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from openodke.types import Document, SourceTier

# One step of a path: a key, a list index, or None for the `[*]` wildcard.
Step = str | int | None

_BARE_KEY = re.compile(r'[^\s.\[\]":$](?:[^.\[\]":\r\n]*[^\s.\[\]":])?')
_DECODER = json.JSONDecoder()


def format_path(steps: Sequence[Step]) -> str:
    """Steps to the canonical path string — the key `metadata["fields"]` uses."""
    out: list[str] = []
    for step in steps:
        if step is None:
            out.append("[*]")
        elif isinstance(step, int):
            out.append(f"[{step}]")
        elif _BARE_KEY.fullmatch(step):
            out.append(f".{step}" if out else step)
        else:
            out.append(f"[{json.dumps(step, ensure_ascii=False)}]")
    return "".join(out)


def parse_path(path: str) -> list[Step]:
    """A path string to steps. A leading `$` or `$.` is accepted and ignored."""
    text = path.strip()
    if text.startswith("$"):
        text = text[1:].removeprefix(".")
    steps: list[Step] = []
    i = 0
    try:
        while i < len(text):
            step: Step
            if text[i] == "[":
                if text.startswith('["', i):
                    step, close = _DECODER.raw_decode(text, i + 1)
                else:
                    close = text.index("]", i)
                    inner = text[i + 1 : close].strip()
                    step = None if inner == "*" else int(inner)
                if text[close] != "]":
                    raise ValueError(path)
                i = close + 1
            else:
                if text[i] == ".":
                    if not steps:
                        raise ValueError(path)
                    i += 1
                stops = [s for s in (text.find(".", i), text.find("[", i)) if s != -1]
                end = min(stops, default=len(text))
                if end == i:
                    raise ValueError(path)
                step, i = text[i:end], end
            steps.append(step)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"not a record path: {path!r}") from exc
    if not steps:
        raise ValueError(f"not a record path: {path!r}")
    return steps


def leaves(record: Mapping[str, Any]) -> Iterator[tuple[list[Step], Any]]:
    """Every leaf of a record with its steps, in the record's own order.

    Empty containers are leaves, so nothing in the row goes unrendered.
    """

    def walk(node: Any, steps: list[Step]) -> Iterator[tuple[list[Step], Any]]:
        if isinstance(node, Mapping) and node:
            for key, child in node.items():
                yield from walk(child, [*steps, str(key)])
        elif isinstance(node, list | tuple) and node:
            for index, child in enumerate(node):
                yield from walk(child, [*steps, index])
        elif steps:
            yield steps, node

    yield from walk(record, [])


def resolve_path(record: Any, steps: Sequence[Step]) -> Iterator[tuple[list[Step], Any]]:
    """Every `(concrete steps, value)` a path reaches, wildcards expanded in order."""

    def walk(node: Any, rest: Sequence[Step], done: list[Step]) -> Iterator[tuple[list[Step], Any]]:
        if not rest:
            yield done, node
            return
        head, tail = rest[0], rest[1:]
        if head is None:
            pairs: list[tuple[Step, Any]] = []
            if isinstance(node, Mapping):
                pairs = [(str(k), v) for k, v in node.items()]
            elif isinstance(node, list | tuple):
                pairs = list(enumerate(node))
            for step, child in pairs:
                yield from walk(child, tail, [*done, step])
        elif isinstance(head, int):
            if isinstance(node, list | tuple) and 0 <= head < len(node):
                yield from walk(node[head], tail, [*done, head])
        elif isinstance(node, Mapping) and head in node:
            yield from walk(node[head], tail, [*done, head])

    yield from walk(record, steps, [])


def render_value(value: Any) -> str:
    """A leaf as it appears in the rendering. Never empty, so a span always has width.

    A string is shown bare unless that would be ambiguous — empty, padded, or
    spanning lines — in which case it is shown as a JSON string literal.
    """
    if isinstance(value, str):
        plain = value and value == value.strip() and "\n" not in value and "\r" not in value
        return value if plain else json.dumps(value, ensure_ascii=False)
    if value is None or isinstance(value, bool | int | float):
        return json.dumps(value)
    if isinstance(value, Mapping):
        return "{}"
    if isinstance(value, list | tuple):
        return "[]"
    # Dates, decimals, whatever a Parquet column holds.
    return render_value(str(value))


def render_record(record: Mapping[str, Any]) -> tuple[str, dict[str, list[int]]]:
    """The record's text, and `path -> [start, end]` of each rendered value in it."""
    lines: list[str] = []
    fields: dict[str, list[int]] = {}
    pos = 0
    for steps, value in leaves(record):
        path, shown = format_path(steps), render_value(value)
        start = pos + len(path) + 2
        fields[path] = [start, start + len(shown)]
        line = f"{path}: {shown}"
        lines.append(line)
        pos += len(line) + 1
    return "\n".join(lines), fields


def record_document(
    record: Mapping[str, Any],
    *,
    uri: str | None = None,
    tier: SourceTier = SourceTier.UNVERIFIED,
    row_index: int | None = None,
    line: int | None = None,
    title: str | None = None,
) -> Document:
    """A structured `Document` for one record: rendered text, row and offsets in metadata.

    Public so that rows from somewhere other than a file — a database cursor, a
    DataFrame — become documents the pattern extractor reads the same way.
    """
    text, fields = render_record(record)
    metadata: dict[str, Any] = {"row": dict(record), "fields": fields}
    if row_index is not None:
        metadata["row_index"] = row_index
    if line is not None:
        metadata["line"] = line
    return Document(
        text=text, uri=uri, title=title, modality="structured", tier=tier, metadata=metadata
    )


__all__ = [
    "Step",
    "format_path",
    "leaves",
    "parse_path",
    "record_document",
    "render_record",
    "render_value",
    "resolve_path",
]
