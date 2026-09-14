"""A directory of mixed files to documents, each file read by its suffix's loader."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

from odke.loaders.html import HtmlLoader
from odke.loaders.structured import CsvLoader, JsonlLoader, JsonLoader, ParquetLoader, TsvLoader
from odke.loaders.text import MarkdownLoader, TextLoader
from odke.stages import Loader, Source
from odke.types import Document, SourceTier


def default_loaders(
    *, tier: SourceTier = SourceTier.UNVERIFIED, encoding: str | None = None
) -> dict[str, Loader]:
    """The suffix table `DirectoryLoader` starts from. Lower-case, dot included.

    `encoding` left as None keeps each loader's own default: plain UTF-8 for
    text, whose offsets are file offsets, and UTF-8 with any byte-order mark
    stripped for records, whose offsets are into the rendering.
    """
    text_encoding = encoding or "utf-8"
    record_encoding = encoding or "utf-8-sig"
    text = TextLoader(tier=tier, encoding=text_encoding)
    markdown = MarkdownLoader(tier=tier, encoding=text_encoding)
    jsonl = JsonlLoader(tier=tier, encoding=record_encoding)
    html = HtmlLoader(tier=tier, encoding=text_encoding)
    return {
        ".txt": text,
        ".text": text,
        ".md": markdown,
        ".markdown": markdown,
        ".html": html,
        ".htm": html,
        ".csv": CsvLoader(tier=tier, encoding=record_encoding),
        ".tsv": TsvLoader(tier=tier, encoding=record_encoding),
        ".json": JsonLoader(tier=tier, encoding=record_encoding),
        ".jsonl": jsonl,
        ".ndjson": jsonl,
        # Claimed so a directory walk says "install the extra" rather than
        # silently skipping the file.
        ".parquet": ParquetLoader(tier=tier),
    }


class DirectoryLoader:
    """Walks a directory — or takes one file — and hands each file to its loader.

    Modality comes from the loader that claims the suffix, so a mixed directory
    routes itself: Markdown and text arrive unstructured and go to the model,
    records arrive structured and never cost a call. A file no loader claims is
    skipped rather than guessed at; reading a PDF as UTF-8 would produce a
    document, and every fact from it would be noise.

    Files are visited in sorted order, so two runs over one directory produce
    documents in the same order.
    """

    def __init__(
        self,
        pattern: str = "**/*",
        *,
        tier: SourceTier = SourceTier.UNVERIFIED,
        encoding: str | None = None,
        loaders: Mapping[str, Loader] | None = None,
    ) -> None:
        self.pattern = pattern
        table = default_loaders(tier=tier, encoding=encoding) if loaders is None else loaders
        self.loaders = {suffix.lower(): loader for suffix, loader in table.items()}

    def load(self, source: Source) -> Iterator[Document]:
        root = Path(source)
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = sorted(p for p in root.glob(self.pattern) if p.is_file())
        else:
            raise FileNotFoundError(f"no such file or directory: {root}")
        for path in paths:
            loader = self.loaders.get(path.suffix.lower())
            if loader is not None:
                yield from loader.load(path)


__all__ = ["DirectoryLoader", "default_loaders"]
