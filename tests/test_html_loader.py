"""HTML loads as the page a reader sees, and every character of it traces back to the markup."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from odke import Document, Fact, Loader, Ontology, SentenceChunker, SourceTier
from odke.extract import PatternExtractor
from odke.loaders import (
    DirectoryLoader,
    HtmlLoader,
    TextLoader,
    heading_path,
    source_locations,
)

# Real-world shape: CRLF line endings, an unclosed <p>, an unclosed <tr>, inline
# tags mid-word-run, entities, <br>, a script whose body looks like markup, and
# the hidden elements a naive tag-stripper leaks.
PAGE = (
    '<!DOCTYPE html>\r\n<html lang="en">\r\n<head>\r\n  <meta charset="utf-8">\r\n'
    "  <title>Ada Lovelace &mdash; Biography</title>\r\n"
    "  <style>body { color: red; }</style>\r\n"
    '  <script>var x = "<p>not text</p>"; if (a < b && c) {}</script>\r\n'
    "</head>\r\n<body>\r\n"
    '  <nav><a href="/">Home</a> | <a href="/people">People</a></nav>\r\n'
    "  <h1>Ada <em>King</em>, Countess of Lovelace</h1>\r\n"
    "  <p>Ada   Lovelace\r\n     wrote the <b>first</b> published program for Babbage&rsquo;s\r\n"
    "     Analytical&nbsp;Engine. AT&amp;T did not exist.<br>She died in 1852.\r\n"
    "  <p>A second paragraph with an unclosed tag.\r\n"
    "  <noscript><p>Enable JavaScript</p></noscript>\r\n"
    "  <h2>Staff</h2>\r\n"
    '  <table class="wikitable">\r\n    <caption>People we know</caption>\r\n'
    "    <thead><tr><th>Name</th><th>Born</th><th>Employer</th><th>Note</th></tr></thead>\r\n"
    "    <tbody>\r\n"
    '      <tr><td>Ada Lovelace</td><td>1815-12-10</td><td><a href="#">Analytical Engines</a>'
    " Ltd</td><td></td>\r\n"
    "      <tr><td>Alan <i>Turing</i></td><td>1912-06-23<br></td><td></td><td>A | B</td></tr>\r\n"
    "    </tbody>\r\n  </table>\r\n"
    "  <dl><dt>Name</dt><dd>Grace Hopper</dd>\r\n  <dt>Employer</dt> <dd>US&#32;Navy</dd></dl>\r\n"
    "  <ul><li>First item<li>Second &lt;item&gt;</ul>\r\n"
    "  <template><p>hidden</p></template>\r\n"
    "  <pre>  code   stays\r\n    as is</pre>\r\n"
    '  <footer role="contentinfo">Published by Example Press</footer>\r\n'
    "</body>\r\n</html>\r\n"
)

TEXT = (
    "Home | People\n\n"
    "# Ada King, Countess of Lovelace\n\n"
    "Ada Lovelace wrote the first published program for Babbage\N{RIGHT SINGLE QUOTATION MARK}s"
    " Analytical\N{NO-BREAK SPACE}Engine. AT&T did not exist.\n"
    "She died in 1852.\n\n"
    "A second paragraph with an unclosed tag.\n\n"
    "## Staff\n\n"
    "People we know\n"
    "| Name | Born | Employer | Note |\n"
    "| --- | --- | --- | --- |\n"
    "| Ada Lovelace | 1815-12-10 | Analytical Engines Ltd |  |\n"
    "| Alan Turing | 1912-06-23 |  | A \\| B |\n\n"
    "Name: Grace Hopper\n"
    "Employer: US Navy\n\n"
    "- First item\n"
    "- Second <item>\n\n"
    "  code   stays\r\n    as is\n\n"
    "Published by Example Press"
)

# What the loader may write that the markup did not say: layout, never content.
LAYOUT = set("\n #|-:\\")


def _load(markup: str, **options: Any) -> Document:
    (doc,) = HtmlLoader(**options).load(markup.encode())
    return doc


def _assert_map_is_exact(doc: Document, markup: str) -> None:
    """Each segment is what the markup says at its offsets; everything else is layout."""
    source_map = doc.metadata["source_map"]
    assert source_map["format"] == "html"
    cursor = 0
    for start, end, unit, source_start, source_end in source_map["segments"]:
        assert cursor <= start < end <= len(doc.text)
        assert set(doc.text[cursor:start]) <= LAYOUT, doc.text[cursor:start]
        text, raw = doc.text[start:end], markup[source_start:source_end]
        if end - start == source_end - source_start:
            for char, source in zip(text, raw, strict=True):
                assert char == source or (char == " " and source in " \t\r\n\f"), (text, raw)
        else:
            decoded = html.unescape(raw)
            assert decoded == text or (text == " " and not decoded.strip(" \t\r\n\f")), (text, raw)
        assert source_map["units"][unit]["path"].startswith("/")
        cursor = end
    assert set(doc.text[cursor:]) <= LAYOUT


def _triples(facts: list[Fact]) -> set[tuple[str, str, str]]:
    return {
        (
            f.subject.key,
            f.predicate,
            f.object_entity.key if f.object_entity else str(f.object_value),
        )
        for f in facts
    }


def test_a_messy_page_becomes_the_text_a_reader_sees() -> None:
    doc = _load(PAGE)
    assert doc.text == TEXT
    assert doc.title == "Ada Lovelace \N{EM DASH} Biography"
    # A table and a definition list: the pattern extractor has work here.
    assert (doc.modality, doc.tier, doc.uri) == ("semi_structured", SourceTier.UNVERIFIED, None)
    for hidden in ("not text", "color", "Enable JavaScript", "hidden", "Biography"):
        assert hidden not in doc.text


def test_every_character_is_read_from_the_markup_or_is_layout() -> None:
    _assert_map_is_exact(_load(PAGE), PAGE)


def test_a_span_resolves_back_into_the_raw_markup() -> None:
    doc = _load(PAGE)

    start = doc.text.index("first published")
    first, rest = source_locations(doc, start, start + len("first published"))
    assert first["path"] == "/html[1]/body[1]/p[1]/b[1]"
    assert rest["path"] == "/html[1]/body[1]/p[1]"
    assert (first["start"], rest["end"]) == (start, start + len("first published"))
    assert PAGE[first["source_start"] : rest["source_end"]] == "first</b> published"

    # An entity is indivisible: any part of it maps to all of it.
    start = doc.text.index("AT&T")
    (where,) = source_locations(doc, start, start + len("AT&T"))
    assert PAGE[where["source_start"] : where["source_end"]] == "AT&amp;T"
    (amp,) = source_locations(doc, start + 2, start + 3)
    assert PAGE[amp["source_start"] : amp["source_end"]] == "&amp;"

    # Offsets are into the markup as it is, CRLF and all.
    start = doc.text.index("She died")
    (where,) = source_locations(doc, start, start + len("She died in 1852."))
    assert PAGE[where["source_start"] : where["source_end"]] == "She died in 1852."

    # A cell comes back to its element; the escape the loader added has no source.
    start = doc.text.index("A \\| B")
    (cell,) = source_locations(doc, start, start + len("A \\| B"))
    assert cell["path"] == "/html[1]/body[1]/table[1]/tbody[1]/tr[2]/td[4]"
    assert PAGE[cell["source_start"] : cell["source_end"]] == "A | B"

    start = doc.text.index("## Staff")
    assert source_locations(doc, start, start + 3) == []
    assert source_locations(doc, start, start) == []


def test_headings_are_recorded_with_offsets_like_markdown() -> None:
    doc = _load(PAGE)
    headings = doc.metadata["headings"]
    assert [(h["level"], h["text"]) for h in headings] == [
        (1, "Ada King, Countess of Lovelace"),
        (2, "Staff"),
    ]
    assert all(doc.text[h["start"] : h["end"]] == h["text"] for h in headings)
    assert heading_path(doc, doc.text.index("Ada Lovelace |")) == (
        "Ada King, Countess of Lovelace",
        "Staff",
    )


def test_a_table_is_reachable_by_the_pattern_extractor_and_its_cells_resolve(
    people: Ontology,
) -> None:
    doc = _load(PAGE)
    extractor = PatternExtractor(documents=[doc])
    chunks = list(SentenceChunker(max_words=40).chunk(doc))
    assert len(chunks) > 1
    facts = [fact for chunk in chunks for fact in extractor.extract(chunk, people)]
    assert _triples(facts) == {
        ("Person:ada lovelace", "full_name", "Ada Lovelace"),
        ("Person:ada lovelace", "birth_date", "1815-12-10"),
        ("Person:ada lovelace", "employer", "Company:analytical engines ltd"),
        ("Person:alan turing", "full_name", "Alan Turing"),
        ("Person:alan turing", "birth_date", "1912-06-23"),
        ("Person:grace hopper", "full_name", "Grace Hopper"),
        ("Person:grace hopper", "employer", "Company:us navy"),
    }
    for fact in facts:
        (evidence,) = fact.evidence
        assert evidence.span is not None and evidence.span.is_faithful(doc)

    born = next(f for f in facts if f.object_value == "1912-06-23")
    assert born.evidence[0].span is not None
    (cell,) = source_locations(doc, born.evidence[0].span.start, born.evidence[0].span.end)
    assert cell["path"].endswith("/tr[2]/td[2]")
    assert PAGE[cell["source_start"] : cell["source_end"]] == "1912-06-23"

    # `US&#32;Navy`: the space is the character reference, not an invention.
    navy = next(f for f in facts if f.object_entity and f.object_entity.label == "US Navy")
    assert navy.evidence[0].span is not None
    locations = source_locations(doc, navy.evidence[0].span.start, navy.evidence[0].span.end)
    markup = PAGE[locations[0]["source_start"] : locations[-1]["source_end"]]
    assert markup == "US&#32;Navy"


def test_an_infobox_becomes_key_value_lines_named_by_its_heading(people: Ontology) -> None:
    page = (
        "<h2>Grace Hopper</h2><table>"
        "<tr><th colspan=2>Personal details</th></tr>"
        "<tr><th>Born</th><td>1906-12-09</td></tr>"
        "<tr><th>Employer</th><td>US <b>Navy</b></td></tr>"
        "<tr><td></td><td></td></tr>"
        "</table>"
    )
    doc = _load(page)
    assert doc.text == ("## Grace Hopper\n\nPersonal details\nBorn: 1906-12-09\nEmployer: US Navy")
    assert doc.modality == "semi_structured"
    facts = list(
        PatternExtractor(documents=[doc]).extract(next(SentenceChunker().chunk(doc)), people)
    )
    assert _triples(facts) == {
        ("Person:grace hopper", "birth_date", "1906-12-09"),
        ("Person:grace hopper", "employer", "Company:us navy"),
    }
    _assert_map_is_exact(doc, page)


def test_boilerplate_is_kept_unless_asked_for() -> None:
    doc = _load(PAGE)
    assert "Home | People" in doc.text and "Example Press" in doc.text
    stripped = _load(PAGE, strip_boilerplate=True)
    assert "Home" not in stripped.text and "Example Press" not in stripped.text
    assert stripped.text.startswith("# Ada King")
    by_role = _load('<div role="navigation">Menu</div><p>Body.</p>', strip_boilerplate=True)
    assert by_role.text == "Body."
    _assert_map_is_exact(stripped, PAGE)


@pytest.mark.parametrize(
    ("markup", "text"),
    [
        # A byte-order mark is not text; a whitespace reference collapses like whitespace.
        ("\N{ZERO WIDTH NO-BREAK SPACE}<p>a&#10;&#10; b</p>", "a b"),
        # Self-closing and stray </br> are one break each, and two make a paragraph.
        ("one<br/>two</br>three<br><br>four", "one\ntwo\nthree\n\nfour"),
        # Blocks inside a cell stay on the cell's line; a nested table is spaces.
        (
            "<table><tr><th>A</th><th>B</th></tr><tr><td><p>x</p><p>y</p></td>"
            "<td><table><tr><td>in</td><td>ner</td></tr></table></td></tr></table>",
            "| A | B |\n| --- | --- |\n| x y | in ner |",
        ),
        # A decoded pipe is escaped too; a lone row is still a header; an empty
        # table leaves nothing behind.
        (
            "<table><tr><td>&#124;</td></tr></table><table><tr><td> </td></tr></table>",
            "| \\| |\n| --- |",
        ),
        # A term with no definition, a definition with no term.
        (
            "<dl><dt>Alone</dt><dt>Key</dt><dd>value</dd><dd>orphan</dd></dl>",
            "Alone\nKey: value\norphan",
        ),
        # An empty item leaves no bullet behind; text after a list is not an item.
        ("<ul><li></li><li>kept</li></ul>after", "- kept\n\nafter"),
        ("<p>unclosed <b>bold<p>next", "unclosed bold\n\nnext"),
        ("plain &T &amp text", "plain &T & text"),
        ("", ""),
    ],
)
def test_edge_cases_render_deterministically(markup: str, text: str) -> None:
    doc = _load(markup)
    assert doc.text == text
    _assert_map_is_exact(doc, markup)


def test_title_modality_and_sources() -> None:
    page = "<svg><title>Close</title></svg><h1>Heading &amp; more</h1><p>Prose.</p>"
    doc = _load(page)
    # No <title>: the first h1 is the title, and an SVG tooltip never is.
    assert doc.title == "Heading & more"
    assert doc.modality == "unstructured"
    assert _load(page, modality="semi_structured").modality == "semi_structured"
    assert _load("<title>  </title><p>x</p>").title is None


def test_a_directory_reads_html_and_htm(tmp_path: Path) -> None:
    (tmp_path / "a.html").write_bytes(b"<title>A</title><p>Alpha.</p>")
    (tmp_path / "b.HTM").write_bytes(b"<p>Beta &copy;</p>")
    docs = list(DirectoryLoader(tier=SourceTier.COMMUNITY).load(tmp_path))
    assert [(d.title, d.text, d.tier) for d in docs] == [
        ("A", "Alpha.", SourceTier.COMMUNITY),
        (None, "Beta \N{COPYRIGHT SIGN}", SourceTier.COMMUNITY),
    ]
    assert docs[0].uri == (tmp_path / "a.html").resolve().as_uri()
    assert isinstance(HtmlLoader(), Loader)


def test_source_locations_refuses_what_it_cannot_answer() -> None:
    (plain,) = TextLoader().load(b"no map here")
    with pytest.raises(ValueError, match="no source map"):
        source_locations(plain, 0, 2)
    doc = _load("<p>short</p>")
    with pytest.raises(ValueError, match="not a range"):
        source_locations(doc, 3, 99)
    with pytest.raises(ValueError, match="not a range"):
        source_locations(doc, 3, 2)


_PIECES = st.sampled_from(
    [
        "<p>", "</p>", "<div>", "</div>", "<b>", "</b>", "<br>", "<br/>", "<h2>", "</h2>",
        "<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>", "<th>", "</th>",
        "<dl>", "<dt>", "</dt>", "<dd>", "</dd>", "<ul>", "<li>", "</li>", "<pre>", "</pre>",
        "<script>", "</script>", "<noscript>", "</noscript>", "<title>", "</title>",
        "&amp;", "&nbsp;", "&#124;", "&#10;", "&bogus;", "&", "<", ">", "|", ":", "#",
        " ", "  \r\n ", "\t", "word", "Name", "x.y", "\N{LATIN SMALL LETTER E WITH ACUTE}",
    ]
)  # fmt: skip


@settings(max_examples=300, deadline=None)
@given(st.lists(_PIECES, max_size=40))
def test_any_markup_keeps_the_map_exact(pieces: list[str]) -> None:
    markup = "".join(pieces)
    doc = _load(markup)
    _assert_map_is_exact(doc, markup)
    assert all(doc.text[h["start"] : h["end"]] == h["text"] for h in doc.metadata["headings"])
