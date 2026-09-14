"""Tracing extracted text back to the file it was extracted from.

A text or Markdown document's `text` is the file, so a span's offsets are already
file offsets. HTML, PDF and DOCX are not like that: what sits on disk is markup,
a content stream or a zip of XML — nothing a span could point into and nothing a
grounder could check a quote against. So for those formats `Document.text` is the
*extracted* text, every `Span` indexes it exactly as DECISIONS #3 requires, and a
separate map in `metadata["source_map"]` records where each piece of that text
came from. The grounder never reads the map. A person getting from a fact back to
the original file does.

The map is plain JSON, because `Document.metadata` has to be:

    {
        "format": "html",                       # or "pdf", "docx"
        "units": [{"path": "/html[1]/body[1]/p[1]"}, ...],
        "segments": [[start, end, unit, source_start, source_end], ...],
    }

`units` are the places in the source a format can name:

- **HTML** `{"path": "/html[1]/body[1]/div[2]/p[1]"}` — the element holding the
  text, as an XPath over the tags as written (an implied `html` or `body` is not
  invented). Source offsets are characters into the decoded markup, one
  coordinate space for the whole file.
- **PDF** `{"page": 3}`, one-based, plus `"label"` when the file gives the page
  a printed label other than its number. Source offsets are characters into
  that page's `page.extract_text()`, which depends on the pypdf version the map
  records under `"extractor"`.
- **DOCX** `{"paragraph": 4}` for `document.paragraphs[4]`, or
  `{"table": 0, "row": 1, "cell": 2, "paragraph": 0}` for
  `document.tables[0].rows[1].cells[2].paragraphs[0]`. A table nested in a cell
  adds `"inner"`, the same shape relative to that cell. Source offsets are
  characters into that paragraph's `text`.

A segment says `doc.text[start:end]` came from `[source_start, source_end)` of
`units[unit]`. Segments are sorted and never overlap. Where the two lengths are
equal the correspondence is character for character; where they differ — `&amp;`
became `&`, a newline and its indentation became one space — the segment is
indivisible, and any part of it maps to the whole source range. Characters in no
segment were written by the loader rather than read: the blank line between two
blocks, the `|` of a rendered table. They have no source.

`source_locations` does the lookup.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from odke.types import Document

# `(is_header, cell)` per cell, per row.
Rows = Sequence[Sequence[tuple[bool, "SourceMapBuilder"]]]


class SourceMapBuilder:
    """Extracted text and its map, built together so they cannot disagree."""

    def __init__(self, format: str = "", **extra: Any) -> None:
        self.format = format
        self.extra = extra
        self.length = 0
        # Newlines at the very end of the text, so a block break never doubles one.
        self.trailing_newlines = 0
        self._parts: list[str] = []
        self._units: list[dict[str, Any]] = []
        self._unit_ids: dict[str, int] = {}
        self._segments: list[list[int]] = []

    @property
    def text(self) -> str:
        if len(self._parts) > 1:
            self._parts = ["".join(self._parts)]
        return self._parts[0] if self._parts else ""

    def unit(self, locator: Mapping[str, Any]) -> int:
        key = json.dumps(locator, sort_keys=True)
        if key not in self._unit_ids:
            self._unit_ids[key] = len(self._units)
            self._units.append(dict(locator))
        return self._unit_ids[key]

    def insert(self, text: str) -> None:
        """Text the loader writes itself: it has no source."""
        self._append(text)

    def add(self, text: str, unit: int, source_start: int, source_end: int | None = None) -> None:
        """Text read from `[source_start, source_end)` of `unit`; the end defaults to linear."""
        if not text:
            return
        source_end = source_start + len(text) if source_end is None else source_end
        start = self.length
        self._append(text)
        if self._segments:
            last = self._segments[-1]
            linear = (
                last[1] - last[0] == last[4] - last[3] and len(text) == source_end - source_start
            )
            if linear and last[1] == start and last[2] == unit and last[4] == source_start:
                last[1], last[4] = self.length, source_end
                return
        self._segments.append([start, self.length, unit, source_start, source_end])

    def runs(self) -> Iterator[tuple[str, dict[str, Any] | None, int, int]]:
        """`(text, locator, source_start, source_end)` in order; inserted text has no locator."""
        text, pos = self.text, 0
        for start, end, unit, source_start, source_end in self._segments:
            if start > pos:
                yield text[pos:start], None, 0, 0
            yield text[start:end], self._units[unit], source_start, source_end
            pos = end
        if pos < len(text):
            yield text[pos:], None, 0, 0

    def extend(self, other: SourceMapBuilder, *, escape_pipes: bool = False) -> None:
        """Append another builder's text and map, optionally as a pipe-table cell."""
        for text, locator, source_start, source_end in other.runs():
            if locator is None:
                self.insert(text.replace("|", "\\|") if escape_pipes else text)
                continue
            unit = self.unit(locator)
            if not escape_pipes or "|" not in text:
                self.add(text, unit, source_start, source_end)
            elif len(text) != source_end - source_start:
                # Indivisible, so it is a decoded `&#124;`: the escape goes in front.
                self.insert("\\")
                self.add(text, unit, source_start, source_end)
            else:
                pos = 0
                for i, char in enumerate(text):
                    if char == "|":
                        self.add(text[pos:i], unit, source_start + pos, source_start + i)
                        self.insert("\\")
                        pos = i
                self.add(text[pos:], unit, source_start + pos, source_end)

    def build(self) -> dict[str, Any]:
        return {
            "format": self.format,
            **self.extra,
            "units": [dict(u) for u in self._units],
            "segments": [list(s) for s in self._segments],
        }

    def _append(self, text: str) -> None:
        if not text:
            return
        self._parts.append(text)
        self.length += len(text)
        body = text.rstrip("\n")
        tail = len(text) - len(body)
        self.trailing_newlines = self.trailing_newlines + tail if not body else tail


def write_table(out: SourceMapBuilder, rows: Rows) -> bool:
    """Render table rows in a shape the pattern extractor reads; True if it can read it.

    Two shapes, both already understood by `odke.extract.PatternExtractor`, so a
    table in a web page or a Word file needs no format-specific extraction code:

    - every row a header cell and a data cell (an infobox, a spec sheet; a lone
      header cell may title a group) becomes `Key: value` lines;
    - anything else becomes a pipe table whose first row is the header.

    Cells are one line each — the caller flattens them — and a `|` inside a cell
    is escaped so it cannot split one.
    """
    kept = [row for row in rows if any(cell.length for _, cell in row)]
    if not kept:
        return False
    pairs = any(len(row) == 2 for row in kept) and all(
        (len(row) == 2 and row[0][0] and not row[1][0]) or (len(row) == 1 and row[0][0])
        for row in kept
    )
    for i, row in enumerate(kept):
        if i:
            out.insert("\n")
        if pairs:
            out.extend(row[0][1])
            if len(row) == 2:
                out.insert(": ")
                out.extend(row[1][1])
            continue
        out.insert("| ")
        for j, (_, cell) in enumerate(row):
            if j:
                out.insert(" | ")
            out.extend(cell, escape_pipes=True)
        out.insert(" |")
        if i == 0:
            out.insert("\n| " + " | ".join(["---"] * len(row)) + " |")
    return pairs or len(kept) > 1


def source_locations(doc: Document, start: int, end: int) -> list[dict[str, Any]]:
    """Where `doc.text[start:end]` came from, one entry per source unit, in text order.

    Each entry is the unit's locator plus `start`/`end` (the part of the range it
    covers) and `source_start`/`source_end`. For an evidence span, pass
    `span.start, span.end`. Characters the loader inserted are skipped, so a
    range made only of them yields nothing. Raises `ValueError` for a document
    whose loader kept no map — for text and Markdown the offsets already are
    file offsets — or for a range outside the text.
    """
    source_map = doc.metadata.get("source_map")
    if not isinstance(source_map, Mapping):
        raise ValueError(f"document {doc.id!r} has no source map")
    if not 0 <= start <= end <= len(doc.text):
        raise ValueError(f"[{start}, {end}) is not a range in a text of length {len(doc.text)}")
    units, segments = source_map["units"], source_map["segments"]
    found: list[dict[str, Any]] = []
    i = bisect_right(segments, start, key=lambda segment: segment[1])
    while i < len(segments) and segments[i][0] < end:
        seg_start, seg_end, unit, source_start, source_end = segments[i]
        i += 1
        lo, hi = max(start, seg_start), min(end, seg_end)
        if lo >= hi:
            continue
        if seg_end - seg_start == source_end - source_start:
            source_start, source_end = (
                source_start + lo - seg_start,
                source_start + hi - seg_start,
            )
        last = found[-1] if found else None
        if last is not None and last["_unit"] == unit:
            last["end"], last["source_end"] = hi, source_end
            continue
        found.append(
            {
                **units[unit],
                "_unit": unit,
                "start": lo,
                "end": hi,
                "source_start": source_start,
                "source_end": source_end,
            }
        )
    for entry in found:
        del entry["_unit"]
    return found


__all__ = ["SourceMapBuilder", "source_locations", "write_table"]
