"""HTML: the page as a reader sees it, with a map back to the markup.

The markup is not the text. Offsets into `<p>AT&amp;T</p>` would put quotes among
tags and entities, and a grounder could check none of them. So `Document.text` is
the extracted text and every span indexes it, exactly as DECISIONS #3 requires.
Where each piece of it sits in the markup — the element's path and character
offsets into the decoded file — rides in `metadata["source_map"]` (see
`odke.loaders.sourcemap`), and `source_locations(doc, span.start, span.end)`
turns a fact's evidence back into the markup it came from. The grounder checks
the text; the map is for the person who has to find the fact in the page.

The text is shaped like Markdown, on purpose, because the rest of the pipeline
already reads Markdown:

- Blocks end lines, and paragraphs, headings, lists and tables are separated by a
  blank line, so the chunker sees the page's own paragraphs.
- Headings are written `## Heading` and recorded in `metadata["headings"]` in
  `MarkdownLoader`'s shape, so `heading_path` works unchanged.
- A table becomes a pipe table, or `Key: value` lines when every row is a header
  cell and a data cell (an infobox). A definition list becomes `Term: definition`
  lines. Those are the shapes `PatternExtractor` reads, so a table on a page is
  reachable with no HTML-specific extraction code, and a page holding one is
  `semi_structured` unless `modality` says otherwise.
- List items start `- `.

Whitespace is collapsed the way a browser collapses it — ASCII whitespace only,
since a no-break space is content — except inside `pre`. Entities are decoded.
`script`, `style`, `noscript` and `template` are dropped; so are `nav` and
`footer` when `strip_boilerplate` is set, which it is not by default, because a
footer is sometimes the only place a page states who published it. The `<title>`
is the title, else the first `h1`; it is not part of the text.

Standard library only (`html.parser`), so this runs on the base install. That
parser does not build a tree the way a browser does. The loader closes what HTML
lets authors leave open — `p`, `li`, `td`, `tr`, `dt`, `dd` — which keeps paths
right on ordinary pages; on pathological markup a path is a best effort. The
offsets are not: they come from the parser's own position in the markup.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from odke.loaders.base import Modality, read_text
from odke.loaders.sourcemap import SourceMapBuilder, write_table
from odke.stages import Source
from odke.types import Document, SourceTier

_ASCII_WHITESPACE = " \t\n\r\f"
_WHITESPACE = re.compile(r"[ \t\n\r\f]+")
_BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"

_DROPPED = frozenset({"script", "style", "noscript", "template"})
_BOILERPLATE = frozenset({"nav", "footer"})
_BOILERPLATE_ROLES = frozenset({"navigation", "contentinfo"})
_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param"}
    | {"source", "track", "wbr"}
)
_HEADINGS = {f"h{n}": n for n in range(1, 7)}
# Newlines a block puts around itself; two is a paragraph break to the chunker.
_BLOCKS = {
    **dict.fromkeys(
        ("p", "blockquote", "section", "article", "header", "footer", "nav", "aside", "main"),
        2,
    ),
    **dict.fromkeys(("figure", "form", "fieldset", "details", "address", "hgroup", "dl"), 2),
    **dict.fromkeys(("ul", "ol", "menu", "dialog", "search"), 2),
    **dict.fromkeys(("div", "figcaption", "summary", "legend", "center", "option"), 1),
    **dict.fromkeys(("html", "body"), 1),
}
_STRUCTURE = frozenset({"table", "tr", "td", "th", "caption", "dt", "dd", "li", "pre"})
_BREAKING = frozenset(_BLOCKS) | frozenset(_HEADINGS) | _STRUCTURE

# An open `p` is closed by any of these, per the HTML specification.
_CLOSES_P = frozenset(
    {"address", "article", "aside", "blockquote", "details", "dialog", "div", "dl"}
    | {"fieldset", "figcaption", "figure", "footer", "form", "header", "hgroup", "hr"}
    | {"main", "menu", "nav", "ol", "p", "pre", "search", "section", "table", "ul"}
    | set(_HEADINGS)
)
_TABLE_SCOPE = frozenset({"table", "td", "th", "caption", "template", "html"})
# Opening the key closes an open element among the targets, searching no further
# out than a boundary.
_IMPLIED: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "li": (frozenset({"li"}), frozenset({"ul", "ol", "menu"}) | _TABLE_SCOPE),
    "dt": (frozenset({"dt", "dd"}), frozenset({"dl"}) | _TABLE_SCOPE),
    "dd": (frozenset({"dt", "dd"}), frozenset({"dl"}) | _TABLE_SCOPE),
    "tr": (frozenset({"tr"}), frozenset({"table"})),
    "td": (frozenset({"td", "th"}), frozenset({"tr", "table"})),
    "th": (frozenset({"td", "th"}), frozenset({"tr", "table"})),
    **dict.fromkeys(
        ("thead", "tbody", "tfoot"),
        (frozenset({"thead", "tbody", "tfoot"}), frozenset({"table"})),
    ),
}
_P_SCOPE = _TABLE_SCOPE | {"button", "body"}
_TABLE_PARTS = frozenset({"table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption"})


@dataclass
class _Frame:
    tag: str
    path: str
    dropped: bool
    role: str | None = None
    level: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    heading_start: int | None = None


@dataclass
class _Table:
    rows: list[list[tuple[bool, SourceMapBuilder]]] = field(default_factory=list)


class _Renderer(HTMLParser):
    """One pass over the markup, writing text and map as it goes."""

    def __init__(self, markup: str, *, strip_boilerplate: bool) -> None:
        # Character references are decoded here rather than by the parser, which
        # would otherwise hand over decoded data with no way to tell its length.
        super().__init__(convert_charrefs=False)
        self.markup = markup
        self.strip_boilerplate = strip_boilerplate
        self.line_starts = [0] + [m.end() for m in re.finditer("\n", markup)]
        self.main = SourceMapBuilder("html")
        self.sinks = [self.main]
        self.stack = [_Frame("", "", dropped=False)]
        self.breaks = 0  # newlines owed before the next text
        self.lead = ""  # markup-free syntax owed before the next text: "- ", "## ", ": "
        # A collapsed run of whitespace owed before the next text: its source
        # range, or None for a space the loader inserts itself.
        self.space: tuple[int, int, int] | tuple[None, int, int] | None = None
        self.inline = 0  # open cells, terms, definitions, headings: one line each
        self.pre = 0
        self.tables: list[_Table] = []
        self.in_term = False
        self.term_text = False
        self.key_open = False  # a term has text and its definition has not begun
        self.headings: list[dict[str, Any]] = []
        self.title: str | None = None
        self.title_parts: list[str] | None = None
        self.structured = False

    def render(self) -> None:
        self.feed(self.markup)
        self.close()
        self._pop_to(1)

    @property
    def out(self) -> SourceMapBuilder:
        return self.sinks[-1]

    # ------------------------------------------------------------------ #
    # Parser events
    # ------------------------------------------------------------------ #

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._implied_close(tag)
        parent = self.stack[-1]
        if tag in _VOID:
            if not parent.dropped:
                if tag == "br":
                    self._line_break()
                elif tag == "hr":
                    self._breaking(2, lead="")
            return
        count = parent.counts[tag] = parent.counts.get(tag, 0) + 1
        role = dict(attrs).get("role")
        dropped = (
            parent.dropped
            or tag in _DROPPED
            or (self.strip_boilerplate and (tag in _BOILERPLATE or role in _BOILERPLATE_ROLES))
        )
        frame = _Frame(tag, f"{parent.path}/{tag}[{count}]", dropped)
        self.stack.append(frame)
        if not dropped:
            self._open(frame)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            boundary = _DROPPED
        else:
            boundary = _DROPPED | (frozenset({"table"}) if tag in _TABLE_PARTS else _TABLE_SCOPE)
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self._pop_to(i)
                return
            if self.stack[i].tag in boundary:
                break
        if tag == "br" and not self.stack[-1].dropped:
            self._line_break()  # browsers read a stray </br> as <br>

    def handle_data(self, data: str) -> None:
        start = self._position()
        if start == 0 and data.startswith(_BOM):
            data, start = data[1:], 1
        self._text(data, start, start + len(data))

    def handle_entityref(self, name: str) -> None:
        start = self._position()
        self._reference(start, start + 1 + len(name))

    def handle_charref(self, name: str) -> None:
        start = self._position()
        self._reference(start, start + 2 + len(name))

    # ------------------------------------------------------------------ #
    # Elements
    # ------------------------------------------------------------------ #

    def _implied_close(self, tag: str) -> None:
        if tag in _CLOSES_P:
            self._close_open(frozenset({"p"}), _P_SCOPE)
        if tag in _HEADINGS and self.stack[-1].tag in _HEADINGS:
            self._pop_to(len(self.stack) - 1)
        if tag in _IMPLIED:
            self._close_open(*_IMPLIED[tag])

    def _close_open(self, targets: frozenset[str], boundaries: frozenset[str]) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag in targets:
                self._pop_to(i)
                return
            # A dropped element is a wall: `<p>` inside `<noscript>` must not
            # close a paragraph outside it and let the hidden text escape.
            if self.stack[i].tag in boundaries or self.stack[i].tag in _DROPPED:
                return

    def _pop_to(self, index: int) -> None:
        while len(self.stack) > index:
            frame = self.stack.pop()
            if not frame.dropped and frame.role is not None:
                self._close(frame)

    def _open(self, frame: _Frame) -> None:
        tag = frame.tag
        if tag == "title":
            if self.title is None and self.title_parts is None and "/svg[" not in frame.path:
                frame.role, self.title_parts = "title", []
            else:
                frame.dropped = True  # an SVG's tooltip, or a second title
            return
        if self.inline:
            if tag in _BREAKING:
                self._soft_break()
            return
        if tag in _HEADINGS:
            frame.role, frame.level = "heading", _HEADINGS[tag]
            self._breaking(2, lead="#" * frame.level + " ")
            self.inline += 1
        elif tag == "table":
            frame.role = "table"
            self._breaking(2, lead="")
            self.tables.append(_Table())
        elif tag == "tr" and self.tables:
            frame.role = "row"
            self.tables[-1].rows.append([])
        elif tag in ("td", "th") and self.tables:
            frame.role = "cell"
            if not self.tables[-1].rows:
                self.tables[-1].rows.append([])
            self.sinks.append(SourceMapBuilder())
            self.space = None
            self.inline += 1
        elif tag == "caption":
            frame.role = "line"
            self._breaking(1, lead="")
            self.inline += 1
        elif tag == "dt":
            frame.role = "term"
            self._breaking(1, lead="")
            self.in_term, self.term_text = True, False
            self.inline += 1
        elif tag == "dd":
            frame.role = "line"
            if self.key_open:
                self.breaks, self.lead, self.space = 0, ": ", None
                self.structured = True
            else:
                self._breaking(1, lead="")
            self.key_open = False
            self.inline += 1
        elif tag == "li":
            frame.role = "item"
            self._breaking(1, lead="- ")
        elif tag == "pre":
            frame.role = "pre"
            self._breaking(2, lead="")
            self.pre += 1
        elif tag in _BLOCKS:
            frame.role, frame.level = "block", _BLOCKS[tag]
            self._breaking(frame.level)

    def _close(self, frame: _Frame) -> None:
        role = frame.role
        if role == "title":
            text = _WHITESPACE.sub(" ", "".join(self.title_parts or ())).strip(_ASCII_WHITESPACE)
            self.title, self.title_parts = text or None, None
        elif role == "heading":
            self.inline -= 1
            start = frame.heading_start
            if start is not None:
                text = self.main.text[start:]
                self.headings.append(
                    {"level": frame.level, "text": text, "start": start, "end": self.main.length}
                )
            self._breaking(2, lead="")
        elif role == "cell":
            self.inline -= 1
            cell = self.sinks.pop()
            self.tables[-1].rows[-1].append((frame.tag == "th", cell))
            self.space = None
        elif role == "table":
            table = self.tables.pop()
            self.lead, self.space = "", None
            if any(cell.length for row in table.rows for _, cell in row):
                # The rows were written to their cells, not here, so a caption
                # may have used the break the table opened with.
                self.breaks = max(self.breaks, 1)
                self._flush_breaks()
                self.structured |= write_table(self.main, table.rows)
            self._breaking(2)
        elif role == "term":
            self.inline -= 1
            self.in_term = False
            self._breaking(1, lead="")
            self.key_open = self.term_text
        elif role == "line":
            self.inline -= 1
            self._breaking(1, lead="")
        elif role == "item":
            self._breaking(1, lead="")
        elif role == "pre":
            self.pre -= 1
            self._breaking(2, lead="")
        elif role == "block":
            self._breaking(frame.level)

    # ------------------------------------------------------------------ #
    # Text
    # ------------------------------------------------------------------ #

    def _position(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def _reference(self, start: int, end: int) -> None:
        if self.markup.startswith(";", end):
            end += 1
        raw = self.markup[start:end]
        decoded = html.unescape(raw)
        if decoded == raw:
            self._text(raw, start, end)
        elif self.stack[-1].dropped:
            return
        elif self.title_parts is not None and self.stack[-1].role == "title":
            self.title_parts.append(decoded)
        elif not decoded.strip(_ASCII_WHITESPACE) and not (self.pre and not self.inline):
            self._pending_space(start, end)
        else:
            self._emit(decoded, start, end)

    def _text(self, data: str, start: int, end: int) -> None:
        frame = self.stack[-1]
        if frame.dropped or not data:
            return
        if self.title_parts is not None and frame.role == "title":
            self.title_parts.append(data)
            return
        if self.pre and not self.inline:
            self._emit(data, start, end)
            return
        pos = 0
        for run in _WHITESPACE.finditer(data):
            if run.start() > pos:
                self._emit(data[pos : run.start()], start + pos, start + run.start())
            self._pending_space(start + run.start(), start + run.end())
            pos = run.end()
        if pos < len(data):
            self._emit(data[pos:], start + pos, end)

    def _pending_space(self, start: int, end: int) -> None:
        if self.space is None:
            self.space = (self._unit(), start, end)

    def _soft_break(self) -> None:
        if self.space is None:
            self.space = (None, 0, 0)

    def _line_break(self) -> None:
        if self.inline:
            self._soft_break()
        elif self.pre:
            self.out.insert("\n")
        else:
            self.breaks, self.space = min(self.breaks + 1, 2), None

    def _breaking(self, level: int, *, lead: str | None = None) -> None:
        if self.inline:
            self._soft_break()
            return
        self.breaks, self.space = max(self.breaks, level), None
        if lead is not None:
            self.lead = lead
        self.key_open = False

    def _flush_breaks(self) -> None:
        if self.out is not self.main:
            return  # a cell is one line; the breaks are owed to the page
        if self.breaks and self.out.length:
            self.out.insert("\n" * max(0, self.breaks - self.out.trailing_newlines))
        self.breaks = 0

    def _unit(self) -> int:
        return self.out.unit({"path": self.stack[-1].path or "/"})

    def _emit(self, text: str, start: int, end: int) -> None:
        out = self.out
        self._flush_breaks()
        if self.lead:
            out.insert(self.lead)
            self.lead, self.space = "", None
        elif self.space is not None:
            unit, space_start, space_end = self.space
            if out.length and not out.trailing_newlines:
                if unit is None:
                    out.insert(" ")
                else:
                    out.add(" ", unit, space_start, space_end)
            self.space = None
        heading = next((f for f in reversed(self.stack) if f.role == "heading"), None)
        if heading is not None and heading.heading_start is None:
            heading.heading_start = out.length
        if self.in_term:
            self.term_text = True
        else:
            self.key_open = False
        out.add(text, self._unit(), start, end)


class HtmlLoader:
    """HTML to Markdown-shaped text, with headings and a source map in `metadata`.

    `modality` left as None is `semi_structured` when the page held a table or a
    definition list the pattern extractor can read, and `unstructured` otherwise.
    `encoding` decodes the file; source offsets are into the decoded markup.
    """

    def __init__(
        self,
        *,
        tier: SourceTier = SourceTier.UNVERIFIED,
        encoding: str = "utf-8",
        modality: Modality | None = None,
        strip_boilerplate: bool = False,
    ) -> None:
        self.tier = tier
        self.encoding = encoding
        self.modality = modality
        self.strip_boilerplate = strip_boilerplate

    def load(self, source: Source) -> list[Document]:
        markup, uri = read_text(source, self.encoding)
        renderer = _Renderer(markup, strip_boilerplate=self.strip_boilerplate)
        renderer.render()
        title = renderer.title or next(
            (h["text"] for h in renderer.headings if h["level"] == 1), None
        )
        modality: Modality = self.modality or (
            "semi_structured" if renderer.structured else "unstructured"
        )
        return [
            Document(
                text=renderer.main.text,
                uri=uri,
                title=title,
                modality=modality,
                tier=self.tier,
                metadata={"headings": renderer.headings, "source_map": renderer.main.build()},
            )
        ]


__all__ = ["HtmlLoader"]
