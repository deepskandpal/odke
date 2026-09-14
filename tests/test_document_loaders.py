"""PDF and Word files: extracted text every span indexes, and a map back to page and paragraph."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest

from openodke import Document, Fact, Loader, Ontology, SentenceChunker, SourceTier, Span
from openodke.extract import PatternExtractor
from openodke.loaders import (
    DirectoryLoader,
    DocxLoader,
    MissingExtraError,
    MissingExtraWarning,
    PdfLoader,
    heading_path,
    source_locations,
)

PAGES = [
    ["Ada Lovelace was born in 1815.", "She wrote notes (the first program)."],
    ["Page two continues the story.", "An exam-", "ple of a hyphen at a line end."],
    [],  # a scanned page: no text layer at all
]


def _pdf(pages: list[list[str]], *, title: str | None = None, labelled: bool = False) -> bytes:
    """A small valid PDF, one Helvetica line per string, with its xref table computed.

    Written out by hand rather than by a library, so the fixture depends on
    nothing the loader under test does not, and its content is readable here.
    """
    objects: dict[int, bytes] = {
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    }
    kids: list[int] = []
    number = 4
    for lines in pages:
        ops = [b"BT /F1 12 Tf 14 TL 72 720 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(b"(" + escaped.encode("latin-1") + b") Tj T*")
        ops.append(b"ET")
        stream = b"\n".join(ops)
        objects[number] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        objects[number + 1] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % number
        )
        kids.append(number + 1)
        number += 2
    # Front matter numbered i, ii...; the body restarts at 1.
    labels = b" /PageLabels << /Nums [0 << /S /r >> 1 << /S /D >>] >>" if labelled else b""
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R%s >>" % labels
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % kid for kid in kids),
        len(kids),
    )
    info = b""
    if title is not None:
        objects[number] = b"<< /Title (%s) >>" % title.encode("latin-1")
        info = b" /Info %d 0 R" % number
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for key in sorted(objects):
        offsets[key] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (key, objects[key])
    xref, size = len(out), max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % offsets[key] for key in range(1, size))
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (size, info, xref)
    return bytes(out)


def _page_texts(data: bytes) -> list[str]:
    """The oracle: what pypdf itself extracts, page by page."""
    pypdf = pytest.importorskip("pypdf")
    return [page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(data)).pages]


def _span(doc: Document, quote: str) -> Span:
    start = doc.text.index(quote)
    span = Span(doc_id=doc.id, start=start, end=start + len(quote), quote=quote)
    assert span.is_faithful(doc)
    return span


def _where(doc: Document, span: Span) -> list[dict[str, Any]]:
    return source_locations(doc, span.start, span.end)


def _triples(facts: list[Fact]) -> set[tuple[str, str, str]]:
    return {
        (
            f.subject.key,
            f.predicate,
            f.object_entity.key if f.object_entity else str(f.object_value),
        )
        for f in facts
    }


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


def test_a_multi_page_pdf_is_one_document_whose_spans_resolve_to_pages(tmp_path: Path) -> None:
    data = _pdf(PAGES, title="Ada's notes")
    pages = _page_texts(data)
    path = tmp_path / "notes.pdf"
    path.write_bytes(data)
    (doc,) = PdfLoader(tier=SourceTier.CURATED).load(path)

    # The text is each page's extracted text, verbatim, joined by a blank line;
    # the page with no text layer adds nothing, not even a separator.
    assert doc.text == pages[0] + "\n\n" + pages[1]
    assert (doc.title, doc.uri) == ("Ada's notes", path.resolve().as_uri())
    assert (doc.modality, doc.tier) == ("unstructured", SourceTier.CURATED)
    source_map = doc.metadata["source_map"]
    assert source_map["format"] == "pdf"
    assert source_map["extractor"].startswith("pypdf ")
    assert source_map["units"] == [{"page": 1}, {"page": 2}]

    # A citation needs the page, and the offset into that page's text.
    (where,) = _where(doc, _span(doc, "born in 1815"))
    assert where["page"] == 1
    assert pages[0][where["source_start"] : where["source_end"]] == "born in 1815"
    (where,) = _where(doc, _span(doc, "continues the story"))
    assert where["page"] == 2
    assert pages[1][where["source_start"] : where["source_end"]] == "continues the story"

    # A span across the page break names both pages; the separator has no source.
    start = doc.text.index("program)")
    end = doc.text.index("Page two") + len("Page two")
    first, second = source_locations(doc, start, end)
    assert (first["page"], second["page"]) == (1, 2)
    assert pages[0][first["source_start"] :].startswith("program)")
    assert pages[1][: second["source_end"]] == "Page two"

    # Not de-hyphenated: the join cannot be told from a real hyphen.
    assert "exam-" in doc.text and "example" not in doc.text

    # Every chunk can say which page it starts on.
    chunks = list(SentenceChunker(max_words=6).chunk(doc))
    assert [source_locations(doc, c.start, c.end)[0]["page"] for c in chunks] == [1, 1, 2, 2]


def test_per_page_documents_carry_their_page_and_a_viewer_fragment(tmp_path: Path) -> None:
    data = _pdf(PAGES, labelled=True)
    pages = _page_texts(data)
    path = tmp_path / "book.pdf"
    path.write_bytes(data)
    docs = PdfLoader(per_page=True).load(path)

    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    assert [d.text for d in docs] == pages
    assert [d.uri for d in docs] == [f"{path.resolve().as_uri()}#page={n}" for n in (1, 2, 3)]
    # A scanned page is still a document, so it shows up rather than vanishing.
    assert docs[2].text == "" and docs[2].metadata["source_map"]["segments"] == []
    # Printed labels ride along when they differ from the page number.
    (front,) = _where(docs[0], _span(docs[0], "Ada Lovelace"))
    assert (front["page"], front["label"]) == (1, "i")
    (body,) = _where(docs[1], _span(docs[1], "Page two"))
    assert (body["page"], body["label"], body["source_start"]) == (2, "1", 0)


def test_pdf_from_bytes_with_its_own_separator() -> None:
    data = _pdf([["One."], ["Two."]])
    pages = _page_texts(data)
    (doc,) = PdfLoader(page_separator="\n\f\n", modality="semi_structured").load(data)
    assert doc.text == pages[0] + "\n\f\n" + pages[1]
    assert (doc.uri, doc.title, doc.modality) == (None, None, "semi_structured")
    span = _span(doc, "Two.")
    on_page = pages[1].index("Two.")
    assert _where(doc, span) == [
        {
            "page": 2,
            "start": span.start,
            "end": span.end,
            "source_start": on_page,
            "source_end": on_page + len("Two."),
        }
    ]


def test_pdf_without_pypdf_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(MissingExtraError, match=r'pip install "openodke\[pdf\]"'):
        PdfLoader().load(b"%PDF-1.4")


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def _handbook() -> Any:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Staff handbook", style="Title")
    document.add_heading("People", level=1)
    intro = document.add_paragraph("Ada Lovelace wrote the first program.")
    intro.add_run().add_break()
    intro.add_run("It ran on\tpaper.")
    document.add_paragraph("")
    table = document.add_table(rows=3, cols=4)
    for column, header in enumerate(["Name", "Born", "Employer", "Note"]):
        table.cell(0, column).text = header
    table.cell(1, 0).text = "Ada Lovelace"
    table.cell(1, 1).text = "1815-12-10"
    table.cell(1, 2).merge(table.cell(1, 3)).text = "Analytical Engines"
    table.cell(2, 0).text = "Alan Turing"
    table.cell(2, 1).text = "1912-06-23"
    note = table.cell(2, 3)
    note.text = "NPL | Manchester"
    note.add_paragraph("second\nline")
    note.add_table(rows=1, cols=1).cell(0, 0).text = "nested"
    document.add_heading("Details", level=2)
    document.add_paragraph("Name: Grace Hopper")
    return document


HANDBOOK = (
    "Staff handbook\n\n"
    "# People\n\n"
    "Ada Lovelace wrote the first program.\nIt ran on\tpaper.\n\n"
    "| Name | Born | Employer | Note |\n"
    "| --- | --- | --- | --- |\n"
    "| Ada Lovelace | 1815-12-10 | Analytical Engines |  |\n"
    "| Alan Turing | 1912-06-23 |  | NPL \\| Manchester second line nested |\n\n"
    "## Details\n\n"
    "Name: Grace Hopper"
)


def _save(document: Any, path: Path | None = None) -> bytes:
    buffer = io.BytesIO()
    document.save(buffer)
    if path is not None:
        path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def _paragraph(document: Any, locator: dict[str, Any]) -> str:
    """Follow a locator through python-docx, the way a person would."""
    container = document
    while "table" in locator:
        container = container.tables[locator["table"]].rows[locator["row"]].cells[locator["cell"]]
        if "inner" not in locator:
            break
        locator = locator["inner"]
    return str(container.paragraphs[locator["paragraph"]].text)


def test_a_word_document_keeps_paragraphs_tables_and_headings_in_order(tmp_path: Path) -> None:
    path = tmp_path / "handbook.docx"
    _save(_handbook(), path)
    (doc,) = DocxLoader(tier=SourceTier.AUTHORITATIVE).load(path)
    assert doc.text == HANDBOOK
    # No core title set: the Title-styled paragraph is the title, and still text.
    assert (doc.title, doc.uri) == ("Staff handbook", path.resolve().as_uri())
    assert (doc.modality, doc.tier) == ("semi_structured", SourceTier.AUTHORITATIVE)
    headings = doc.metadata["headings"]
    assert [(h["level"], h["text"]) for h in headings] == [(1, "People"), (2, "Details")]
    assert all(doc.text[h["start"] : h["end"]] == h["text"] for h in headings)
    assert heading_path(doc, doc.text.index("Grace")) == ("People", "Details")


def test_docx_spans_resolve_to_paragraphs_and_cells(tmp_path: Path) -> None:
    document = _handbook()
    (doc,) = DocxLoader().load(_save(document))
    reopened = pytest.importorskip("docx").Document(io.BytesIO(_save(document)))

    def resolves(quote: str, locator: dict[str, Any], source: str | None = None) -> None:
        (where,) = _where(doc, _span(doc, quote))
        found = {k: v for k, v in where.items() if k not in ("start", "end")}
        assert found.pop("source_start") >= 0
        end = found.pop("source_end")
        assert found == locator
        text = _paragraph(reopened, locator)
        assert text[where["source_start"] : end] == (source or quote)

    resolves("first program", {"paragraph": 2})
    # Paragraph 3 is empty: skipped in the text, still counted in the index.
    resolves("Details", {"paragraph": 4})
    resolves("1912-06-23", {"table": 0, "row": 2, "cell": 1, "paragraph": 0})
    # A merged cell is read once, from its first grid slot.
    resolves("Analytical Engines", {"table": 0, "row": 1, "cell": 2, "paragraph": 0})
    resolves(
        "NPL \\| Manchester", {"table": 0, "row": 2, "cell": 3, "paragraph": 0}, "NPL | Manchester"
    )
    # A line break inside a cell became a space; the map is still character for character.
    resolves("second line", {"table": 0, "row": 2, "cell": 3, "paragraph": 1}, "second\nline")
    resolves(
        "nested",
        {
            "table": 0,
            "row": 2,
            "cell": 3,
            "inner": {"table": 0, "row": 0, "cell": 0, "paragraph": 0},
        },
    )

    # Every segment, not just the ones above, is what python-docx says is there.
    source_map = doc.metadata["source_map"]
    for start, end, unit, source_start, source_end in source_map["segments"]:
        assert end - start == source_end - source_start
        text = _paragraph(reopened, source_map["units"][unit])[source_start:source_end]
        assert doc.text[start:end] in (text, text.translate(str.maketrans("\n\t", "  ")))


def test_a_docx_table_is_reachable_by_the_pattern_extractor(people: Ontology) -> None:
    (doc,) = DocxLoader().load(_save(_handbook()))
    extractor = PatternExtractor(documents=[doc])
    facts = [f for chunk in SentenceChunker().chunk(doc) for f in extractor.extract(chunk, people)]
    assert _triples(facts) == {
        ("Person:ada lovelace", "full_name", "Ada Lovelace"),
        ("Person:ada lovelace", "birth_date", "1815-12-10"),
        ("Person:ada lovelace", "employer", "Company:analytical engines"),
        ("Person:alan turing", "full_name", "Alan Turing"),
        ("Person:alan turing", "birth_date", "1912-06-23"),
        ("Person:grace hopper", "full_name", "Grace Hopper"),
    }
    for fact in facts:
        (evidence,) = fact.evidence
        assert evidence.span is not None and evidence.span.is_faithful(doc)
        assert source_locations(doc, evidence.span.start, evidence.span.end)


def test_docx_core_title_deep_headings_and_prose_only() -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.core_properties.title = "  Core title  "
    document.add_paragraph("Styled", style="Title")
    document.add_heading("Very deep", level=8)
    document.add_paragraph("  padded heading  ", style="Heading 1")
    (doc,) = DocxLoader().load(_save(document))
    assert doc.title == "Core title"
    assert doc.modality == "unstructured"
    assert doc.text == "Styled\n\n###### Very deep\n\n#   padded heading  "
    assert [(h["level"], h["text"]) for h in doc.metadata["headings"]] == [
        (8, "Very deep"),
        (1, "padded heading"),
    ]
    assert all(doc.text[h["start"] : h["end"]] == h["text"] for h in doc.metadata["headings"])
    assert DocxLoader(modality="semi_structured").load(_save(document))[0].modality == (
        "semi_structured"
    )


def test_docx_without_python_docx_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "docx", None)
    with pytest.raises(MissingExtraError, match=r'pip install "openodke\[docx\]"'):
        DocxLoader().load(b"PK")


# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #


def test_a_directory_reads_pdf_and_docx(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(_pdf([["Alpha."]]))
    _save(_handbook(), tmp_path / "b.DOCX")
    docs = list(DirectoryLoader().load(tmp_path))
    assert [d.metadata["source_map"]["format"] for d in docs] == ["pdf", "docx"]
    assert docs[1].title == "Staff handbook"


def test_a_missing_extra_skips_that_file_and_keeps_walking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_text("Alpha.", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 not really")
    (tmp_path / "c.docx").write_bytes(b"PK not really")
    (tmp_path / "d.md").write_text("# Delta", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setitem(sys.modules, "docx", None)

    with pytest.warns(MissingExtraWarning) as caught:
        docs = list(DirectoryLoader().load(tmp_path))
    assert [d.text for d in docs] == ["Alpha.", "# Delta"]
    messages = [str(w.message) for w in caught]
    assert len(messages) == 2
    assert "b.pdf" in messages[0] and "openodke[pdf]" in messages[0]
    assert "c.docx" in messages[1] and "openodke[docx]" in messages[1]

    with pytest.raises(MissingExtraError, match=r"openodke\[pdf\]"):
        list(DirectoryLoader(missing_extras="raise").load(tmp_path))
    with pytest.raises(ValueError, match="missing_extras"):
        DirectoryLoader(missing_extras="ignore")  # type: ignore[arg-type]


def test_both_loaders_satisfy_the_protocol() -> None:
    for loader in (PdfLoader(), DocxLoader()):
        assert isinstance(loader, Loader)
