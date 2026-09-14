"""Loaders keep every character where the file had it."""

from __future__ import annotations

from pathlib import Path

import pytest

from openodke import Loader, SentenceChunker, SourceTier
from openodke.loaders import (
    DirectoryLoader,
    MarkdownLoader,
    TextLoader,
    heading_path,
    markdown_headings,
)

MARKDOWN = (
    "# Guide\r\n\r\nIntro text.\r\n\r\n## Install ##\r\n\r\n"
    "```bash\r\n# not a heading\r\n```\r\n\r\n"
    "### From source\r\nSteps.\r\n\r\n## Use\r\n#hashtag is not a heading\r\n"
)


def test_text_is_decoded_from_bytes_so_crlf_offsets_match_the_file(tmp_path: Path) -> None:
    raw = "Ada wrote notes.\r\nZoë read them.\r\n\r\nThe end.\r\n".encode()
    path = tmp_path / "notes.txt"
    path.write_bytes(raw)
    (doc,) = TextLoader(tier=SourceTier.CURATED).load(path)
    # Text mode would have turned every CRLF into LF and shifted the offsets.
    assert doc.text.encode("utf-8") == raw
    assert doc.uri == path.resolve().as_uri()
    assert (doc.modality, doc.tier, doc.title) == ("unstructured", SourceTier.CURATED, None)
    chunks = list(SentenceChunker(max_words=3).chunk(doc))
    assert len(chunks) == 3
    assert all(raw.decode("utf-8")[c.start : c.end] == c.text for c in chunks)


def test_paths_strings_and_bytes_are_all_sources(tmp_path: Path) -> None:
    (from_bytes,) = TextLoader().load(b"plain")
    assert from_bytes.text == "plain" and from_bytes.uri is None
    path = tmp_path / "a.txt"
    path.write_bytes("caf\N{LATIN SMALL LETTER E WITH ACUTE}".encode("latin-1"))
    (from_str,) = TextLoader(encoding="latin-1").load(str(path))
    assert from_str.text == "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    with pytest.raises(UnicodeDecodeError):
        TextLoader().load(path)
    with pytest.raises(TypeError, match="path or bytes"):
        TextLoader().load(42)


def test_markdown_keeps_the_raw_file_and_its_outline(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_bytes(MARKDOWN.encode())
    (doc,) = MarkdownLoader(tier=SourceTier.AUTHORITATIVE).load(path)
    assert doc.text == MARKDOWN
    assert doc.title == "Guide"
    assert (doc.modality, doc.tier) == ("unstructured", SourceTier.AUTHORITATIVE)
    headings = doc.metadata["headings"]
    # Fenced code and a hashtag are not headings; closing hashes are not text.
    assert [(h["level"], h["text"]) for h in headings] == [
        (1, "Guide"),
        (2, "Install"),
        (3, "From source"),
        (2, "Use"),
    ]
    assert all(doc.text[h["start"] : h["end"]] == h["text"] for h in headings)


def test_heading_path_puts_an_offset_back_under_its_section() -> None:
    (doc,) = MarkdownLoader().load(MARKDOWN.encode())
    assert heading_path(doc, doc.text.index("Steps.")) == ("Guide", "Install", "From source")
    assert heading_path(doc, doc.text.index("#hashtag")) == ("Guide", "Use")
    assert heading_path(doc, doc.text.index("Intro")) == ("Guide",)
    assert heading_path(doc, 0) == ()


def test_markdown_edge_cases() -> None:
    bom = "\N{ZERO WIDTH NO-BREAK SPACE}"
    assert markdown_headings(bom + "# Title\n") == [
        {"level": 1, "text": "Title", "start": 3, "end": 8}
    ]
    # A tilde fence is closed only by a tilde fence at least as long.
    text = "~~~~\n# inside\n~~~\n# still inside\n~~~~\n## Out\n#\n####### seven\n"
    assert [h["text"] for h in markdown_headings(text)] == ["Out"]
    (untitled,) = MarkdownLoader(modality="semi_structured").load(b"## Only a section\n")
    assert untitled.title is None
    assert untitled.modality == "semi_structured"


def test_a_directory_routes_each_file_by_its_suffix(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# B\n\nBody.", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.MD").write_text("C.", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")

    docs = list(DirectoryLoader(tier=SourceTier.COMMUNITY).load(tmp_path))
    assert [(d.uri or "").rsplit("/", 1)[-1] for d in docs] == ["a.txt", "b.md", "c.MD"]
    assert [d.title for d in docs] == [None, "B", None]
    assert {d.tier for d in docs} == {SourceTier.COMMUNITY}

    assert [d.title for d in DirectoryLoader("*.md").load(tmp_path)] == ["B"]
    assert len(list(DirectoryLoader().load(tmp_path / "a.txt"))) == 1
    assert list(DirectoryLoader(loaders={}).load(tmp_path)) == []
    with pytest.raises(FileNotFoundError):
        list(DirectoryLoader().load(tmp_path / "missing"))


def test_every_loader_satisfies_the_protocol() -> None:
    for loader in (TextLoader(), MarkdownLoader(), DirectoryLoader()):
        assert isinstance(loader, Loader)
