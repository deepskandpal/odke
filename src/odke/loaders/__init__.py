"""Loaders: bytes to documents, with modality and tier attached.

Every loader keeps offsets honest. A text or Markdown document's `text` is the
file exactly as it is on disk, decoded rather than read in text mode, which
would turn CRLF into LF and shift every offset after the first Windows line
ending. A structured record's `text` is a deterministic rendering of the record,
with the record itself in `metadata`, so a pattern extractor reads typed values
and a grounder still has characters to check (`odke.loaders.records`).

All of them satisfy `odke.stages.Loader` without inheriting from it, and all of
them run on the base install except Parquet, which needs `odke[parquet]`.
"""

from odke.loaders.base import Modality, read_text
from odke.loaders.directory import DirectoryLoader, default_loaders
from odke.loaders.records import record_document
from odke.loaders.structured import (
    CsvLoader,
    JsonlLoader,
    JsonLoader,
    ParquetLoader,
    RecordsLoader,
    TsvLoader,
)
from odke.loaders.text import MarkdownLoader, TextLoader, heading_path, markdown_headings

__all__ = [
    "CsvLoader",
    "DirectoryLoader",
    "JsonLoader",
    "JsonlLoader",
    "MarkdownLoader",
    "Modality",
    "ParquetLoader",
    "RecordsLoader",
    "TextLoader",
    "TsvLoader",
    "default_loaders",
    "heading_path",
    "markdown_headings",
    "read_text",
    "record_document",
]
