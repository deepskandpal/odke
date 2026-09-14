"""Plain text and Markdown: the file as written is the document."""

from __future__ import annotations

import re
from typing import Any

from openodke._text import iter_lines
from openodke.loaders.base import Modality, read_text
from openodke.stages import Source
from openodke.types import Document, SourceTier

_BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_ATX = re.compile(r" {0,3}(#{1,6})(?=[ \t]|$)")
_ATX_CLOSE = re.compile(r"[ \t]+#+[ \t]*$")


class TextLoader:
    """Plain text: one file, one document, every character kept."""

    def __init__(
        self, *, tier: SourceTier = SourceTier.UNVERIFIED, encoding: str = "utf-8"
    ) -> None:
        self.tier = tier
        self.encoding = encoding

    def load(self, source: Source) -> list[Document]:
        text, uri = read_text(source, self.encoding)
        return [Document(text=text, uri=uri, modality="unstructured", tier=self.tier)]


class MarkdownLoader:
    """Markdown as written: the raw file is the text, so offsets are file offsets.

    Nothing is rendered or stripped, so a span found in a chunk resolves against
    the file the caller has. The heading outline rides in `metadata["headings"]`
    with the offsets of each heading's text, and `heading_path` puts a chunk back
    under its section without the chunk carrying more than its offsets
    (DECISIONS #19). The first level-one heading is the title.

    `unstructured` by default. Markdown that is mostly tables and `Key: value`
    blocks is better loaded as `semi_structured`, which sends it to the pattern
    extractor as well as the model.
    """

    def __init__(
        self,
        *,
        tier: SourceTier = SourceTier.UNVERIFIED,
        encoding: str = "utf-8",
        modality: Modality = "unstructured",
    ) -> None:
        self.tier = tier
        self.encoding = encoding
        self.modality: Modality = modality

    def load(self, source: Source) -> list[Document]:
        text, uri = read_text(source, self.encoding)
        headings = markdown_headings(text)
        title = next((h["text"] for h in headings if h["level"] == 1), None)
        return [
            Document(
                text=text,
                uri=uri,
                title=title,
                modality=self.modality,
                tier=self.tier,
                metadata={"headings": headings},
            )
        ]


def markdown_headings(text: str) -> list[dict[str, Any]]:
    """ATX headings outside fenced code, as `{"level", "text", "start", "end"}`.

    Plain dicts rather than a model: `Document.metadata` is what a sink writes
    back, and it has to survive JSON.
    """
    headings: list[dict[str, Any]] = []
    fence: str | None = None
    for start, end in iter_lines(text):
        line = text[start:end]
        lead = 1 if start == 0 and line.startswith(_BOM) else 0
        opener = _FENCE.match(line, lead)
        if fence is not None:
            marker = opener.group(1) if opener else ""
            closes = marker[:1] == fence[0] and len(marker) >= len(fence)
            if opener and closes and not line[opener.end() :].strip():
                fence = None
            continue
        if opener:
            fence = opener.group(1)
            continue
        atx = _ATX.match(line, lead)
        if atx is None:
            continue
        body_start, body_end = atx.end(), len(line)
        closing = _ATX_CLOSE.search(line, body_start)
        if closing:
            body_end = closing.start()
        while body_start < body_end and line[body_start].isspace():
            body_start += 1
        while body_end > body_start and line[body_end - 1].isspace():
            body_end -= 1
        if body_start < body_end:
            headings.append(
                {
                    "level": len(atx.group(1)),
                    "text": line[body_start:body_end],
                    "start": start + body_start,
                    "end": start + body_end,
                }
            )
    return headings


def heading_path(doc: Document, offset: int) -> tuple[str, ...]:
    """The headings in force at `offset`, outermost first: `("Guide", "Install")`."""
    trail: list[tuple[int, str]] = []
    for heading in doc.metadata.get("headings", ()):
        if heading["start"] > offset:
            break
        while trail and trail[-1][0] >= heading["level"]:
            trail.pop()
        trail.append((heading["level"], heading["text"]))
    return tuple(text for _, text in trail)


__all__ = ["MarkdownLoader", "TextLoader", "heading_path", "markdown_headings"]
