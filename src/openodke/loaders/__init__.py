"""Loaders: bytes to documents, with modality and tier attached.

Every loader keeps offsets honest. A text or Markdown document's `text` is the
file exactly as it is on disk, decoded rather than read in text mode, which
would turn CRLF into LF and shift every offset after the first Windows line
ending. A structured record's `text` is a deterministic rendering of the record,
with the record itself in `metadata`, so a pattern extractor reads typed values
and a grounder still has characters to check (`openodke.loaders.records`).

HTML, PDF and Word files are neither: what is on disk is not text a span could
point into. Their documents' `text` is the extracted text — shaped like Markdown
where the format has structure to keep — and `metadata["source_map"]` traces
every piece of it back to the element, page or paragraph it came from
(`openodke.loaders.sourcemap`). Spans still index `text`; `source_locations` is how
a person gets from one back to the file.

All of them satisfy `openodke.stages.Loader` without inheriting from it, and all of
them run on the base install except Parquet, PDF and DOCX, which need
`openodke[parquet]`, `openodke[pdf]` and `openodke[docx]` and say so when they meet a file.
"""

from openodke.loaders.base import MissingExtraError, MissingExtraWarning, Modality, read_text
from openodke.loaders.directory import DirectoryLoader, default_loaders
from openodke.loaders.docx import DocxLoader
from openodke.loaders.html import HtmlLoader
from openodke.loaders.pdf import PdfLoader
from openodke.loaders.records import record_document
from openodke.loaders.sourcemap import source_locations
from openodke.loaders.structured import (
    CsvLoader,
    JsonlLoader,
    JsonLoader,
    ParquetLoader,
    RecordsLoader,
    TsvLoader,
)
from openodke.loaders.text import MarkdownLoader, TextLoader, heading_path, markdown_headings

__all__ = [
    "CsvLoader",
    "DirectoryLoader",
    "DocxLoader",
    "HtmlLoader",
    "JsonLoader",
    "JsonlLoader",
    "MarkdownLoader",
    "MissingExtraError",
    "MissingExtraWarning",
    "Modality",
    "ParquetLoader",
    "PdfLoader",
    "RecordsLoader",
    "TextLoader",
    "TsvLoader",
    "default_loaders",
    "heading_path",
    "markdown_headings",
    "read_text",
    "record_document",
    "source_locations",
]
