"""Loaders: bytes to documents, with modality and tier attached.

Every loader keeps offsets honest. A text or Markdown document's `text` is the
file exactly as it is on disk, decoded rather than read in text mode, which
would turn CRLF into LF and shift every offset after the first Windows line
ending.

All of them satisfy `odke.stages.Loader` without inheriting from it, and all of
them run on the base install.
"""

from odke.loaders.base import Modality, read_text
from odke.loaders.directory import DirectoryLoader, default_loaders
from odke.loaders.text import MarkdownLoader, TextLoader, heading_path, markdown_headings

__all__ = [
    "DirectoryLoader",
    "MarkdownLoader",
    "Modality",
    "TextLoader",
    "default_loaders",
    "heading_path",
    "markdown_headings",
    "read_text",
]
