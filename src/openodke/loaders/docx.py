"""Word documents: paragraphs and tables in document order, with a map back to each paragraph.

A .docx is a zip of XML; nothing in it is text a span could point into. So
`Document.text` is the extracted text, every span indexes that (DECISIONS #3), and
`metadata["source_map"]` (see `openodke.loaders.sourcemap`) names the paragraph each
run of it came from — `document.paragraphs[i]`, or a paragraph inside
`document.tables[t].rows[r].cells[c]` — with character offsets into that
paragraph's `text`. `source_locations(doc, span.start, span.end)` resolves a
fact's evidence to those, so a person can find it in Word.

The text is shaped like Markdown, as the HTML loader's is, so the rest of the
pipeline reads it with no Word-specific code:

- Body paragraphs are separated by a blank line, the chunker's paragraph break. A
  line break inside a paragraph stays a newline.
- A paragraph styled `Heading 1` to `Heading 9` is written `# Heading` (at most
  six marks, as Markdown allows) and recorded in `metadata["headings"]` in
  `MarkdownLoader`'s shape, so `heading_path` works unchanged.
- A table becomes a pipe table whose first row is the header, which
  `PatternExtractor` reads, so a document holding one is `semi_structured` unless
  `modality` says otherwise. A cell's paragraphs are joined by a space; a merged
  cell's text appears once, in its first grid slot, so columns still line up; a
  table nested in a cell is read into that cell.

The title is the core `title` property, else the first `Title`-styled paragraph,
else the first level-one heading. A `Title` paragraph is text, not a heading.

**Known limits.** Headers, footers, footnotes, comments, text boxes and content
controls are not read: python-docx does not expose them in document order.
Heading levels come from style names, so a heading style renamed or localised is
read as a plain paragraph.

Needs `pip install "openodke[docx]"`; python-docx is imported when a file is read.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from typing import Any

from openodke.loaders.base import Modality, import_extra, read_bytes
from openodke.loaders.sourcemap import SourceMapBuilder, write_table
from openodke.stages import Source
from openodke.types import Document, SourceTier

_HEADING_STYLE = re.compile(r"heading ([1-9])", re.IGNORECASE)
# Same length in, same length out, so a flattened cell keeps a linear map.
_CELL_LINE = str.maketrans("\n\r\t\v\f", "     ")

# Wraps a locator relative to a cell into one relative to the document.
Locate = Callable[[dict[str, Any]], dict[str, Any]]


class DocxLoader:
    """A .docx file to one document, with headings and a source map in `metadata`."""

    def __init__(
        self, *, tier: SourceTier = SourceTier.UNVERIFIED, modality: Modality | None = None
    ) -> None:
        self.tier = tier
        self.modality = modality

    def load(self, source: Source) -> list[Document]:
        docx: Any = import_extra(
            "docx", extra="docx", package="python-docx", reading="Word documents"
        )
        data, uri = read_bytes(source)
        document = docx.Document(io.BytesIO(data))
        out = SourceMapBuilder("docx")
        headings: list[dict[str, Any]] = []
        styled_title: str | None = None
        structured = False
        paragraphs = tables = 0
        for block in document.iter_inner_content():
            if _is_table(block):
                rows = _rows(block, _document_level, tables)
                tables += 1
                if any(cell.length for row in rows for _, cell in row):
                    if out.length:
                        out.insert("\n\n")
                    structured |= write_table(out, rows)
                continue
            index, paragraphs = paragraphs, paragraphs + 1
            text: str = block.text
            if not text.strip():
                continue
            style = block.style.name if block.style is not None else ""
            if styled_title is None and style == "Title":
                styled_title = text.strip()
            heading = _HEADING_STYLE.fullmatch(style or "")
            if out.length:
                out.insert("\n\n")
            if heading:
                out.insert("#" * min(int(heading.group(1)), 6) + " ")
            start = out.length
            out.add(text, out.unit({"paragraph": index}), 0)
            if heading:
                headings.append(
                    {
                        "level": int(heading.group(1)),
                        "text": text.strip(),
                        "start": start + len(text) - len(text.lstrip()),
                        "end": start + len(text.rstrip()),
                    }
                )
        title = (
            (document.core_properties.title or "").strip()
            or styled_title
            or next((h["text"] for h in headings if h["level"] == 1), None)
        )
        modality: Modality = self.modality or ("semi_structured" if structured else "unstructured")
        return [
            Document(
                text=out.text,
                uri=uri,
                title=title,
                modality=modality,
                tier=self.tier,
                metadata={"headings": headings, "source_map": out.build()},
            )
        ]


def _is_table(block: Any) -> bool:
    return hasattr(block, "rows")


def _document_level(locator: dict[str, Any]) -> dict[str, Any]:
    return locator


def _in_cell(locate: Locate, table: int, row: int, cell: int) -> Locate:
    return lambda locator: locate({"table": table, "row": row, "cell": cell, **locator})


def _nested(locate: Locate) -> Locate:
    return lambda locator: locate({"inner": locator})


def _rows(table: Any, locate: Locate, index: int) -> list[list[tuple[bool, SourceMapBuilder]]]:
    rows: list[list[tuple[bool, SourceMapBuilder]]] = []
    # python-docx repeats a merged cell in every grid slot it covers. The element
    # is held, not just its id, so no id can be reused while the table is read.
    seen: dict[int, Any] = {}
    for r, row in enumerate(table.rows):
        cells: list[tuple[bool, SourceMapBuilder]] = []
        for c, cell in enumerate(row.cells):
            content = SourceMapBuilder()
            element = cell._tc
            if id(element) not in seen:
                seen[id(element)] = element
                _fill(content, cell, _in_cell(locate, index, r, c))
            cells.append((False, content))
        rows.append(cells)
    return rows


def _fill(out: SourceMapBuilder, cell: Any, locate: Locate) -> None:
    """A cell's paragraphs and nested tables, in order, on one line."""
    paragraphs = tables = 0
    for block in cell.iter_inner_content():
        if _is_table(block):
            for row in _rows(block, _nested(locate), tables):
                for _, inner in row:
                    if inner.length:
                        if out.length:
                            out.insert(" ")
                        out.extend(inner)
            tables += 1
            continue
        index, paragraphs = paragraphs, paragraphs + 1
        text: str = block.text
        if not text.strip():
            continue
        if out.length:
            out.insert(" ")
        out.add(text.translate(_CELL_LINE), out.unit(locate({"paragraph": index})), 0, len(text))


__all__ = ["DocxLoader"]
