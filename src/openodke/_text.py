"""Text helpers: offset-keeping ones for the loaders and the pattern extractor, and one
that takes the password out of a URL in any message."""

from __future__ import annotations

import re
from collections.abc import Iterator

_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)?")

REDACTED = "<redacted>"
# `scheme://user:password@`: the password runs from the colon to the last `@` of the
# authority, as `urlsplit` reads it, so a raw `@` in a password is taken out too.
_USERINFO = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<user>[^/?#@:\s]*):[^/?#\s]*@")


def without_password(text: str) -> str:
    """`text` with the password of every URL in it replaced, the user kept.

    `bolt://neo4j:<redacted>@db`. For a config value (the run manifest), and for a
    message that quotes a URL: an error, a warning, a dry run's plan. A secret in a
    message reaches the manifest's `failed` list, the logs and stderr (release audit
    finding 9), so it is taken out where the message is made, and again where a
    document's failure is put in words (`openodke._batch.reason`).
    """
    return _USERINFO.sub(rf"\g<scheme>\g<user>:{REDACTED}@", text)


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
