"""PDF: the text pypdf extracts, with a map back to the page each piece came from.

A PDF is a content stream of positioned glyphs; nothing in the file is text a span
could point into. So `Document.text` is the extracted text and every span indexes
that, exactly as DECISIONS #3 requires. `metadata["source_map"]` (see
`odke.loaders.sourcemap`) records which page each run of the text came from and
where in that page's extracted text, so `source_locations(doc, span.start,
span.end)` answers the question a citation asks — which page — and the grounder
never has to know pages exist.

**pypdf, not pdfplumber.** pypdf is pure Python with no dependencies of its own,
and it was already the PDF reader in `[docs]`. pdfplumber would add bounding
boxes, at the price of pdfminer.six and several times the extraction time. A page
number and a character index into that page's text is what a citation needs, so
the lighter library wins; a caller who needs coordinates can re-open the page the
map names.

**Pages.** One document per file by default, so a sentence running over a page
break is still one sentence to the chunker. Pages are joined by `page_separator`,
a blank line by default, which the chunker takes as a paragraph break; a page
with no text adds no separator. `per_page=True` makes one document per page
instead, with `metadata["page"]` and a `#page=N` fragment on the URI, which PDF
viewers open at that page.

**No de-hyphenation.** The map could survive it — the dropped `-` and newline
would simply be source characters with no text — but telling `exam-\\nple` (one
word) from `well-\\nknown` (two) needs a dictionary, and a wrong join writes a
word into `text` that is not in the file. A quote nobody can find by searching
the PDF is worse than a hyphen.

**Known limits.** A scanned page has no text layer and extracts as nothing: it
needs OCR, which this does not do. With `per_page` it is still a document, with
empty text, so it shows up rather than vanishing. Multi-column layouts come out in
pypdf's reading order, which is only approximately the author's. Extraction
changes between pypdf versions, so the map records the version that produced its
offsets under `"extractor"`.

Needs `pip install "odke[pdf]"`; pypdf is imported when a file is read.
"""

from __future__ import annotations

import io
from typing import Any

from odke.loaders.base import Modality, import_extra, read_bytes
from odke.loaders.sourcemap import SourceMapBuilder
from odke.stages import Source
from odke.types import Document, SourceTier


class PdfLoader:
    """A PDF to one document, or one per page, with pages in the source map.

    `unstructured` by default: a table in a PDF extracts as runs of words, not as
    rows the pattern extractor could read.
    """

    def __init__(
        self,
        *,
        tier: SourceTier = SourceTier.UNVERIFIED,
        per_page: bool = False,
        page_separator: str = "\n\n",
        modality: Modality = "unstructured",
    ) -> None:
        self.tier = tier
        self.per_page = per_page
        self.page_separator = page_separator
        self.modality: Modality = modality

    def load(self, source: Source) -> list[Document]:
        pypdf: Any = import_extra("pypdf", extra="pdf", package="pypdf", reading="PDF")
        data, uri = read_bytes(source)
        reader = pypdf.PdfReader(io.BytesIO(data))
        extractor = f"pypdf {pypdf.__version__}"
        labels = _labels(reader)
        title = _title(reader)
        pages = [
            (number, page.extract_text() or "", labels.get(number))
            for number, page in enumerate(reader.pages, 1)
        ]
        if not self.per_page:
            return [self._document(pages, uri, title, extractor, {})]
        return [
            self._document(
                [page],
                f"{uri}#page={page[0]}" if uri else None,
                title,
                extractor,
                {"page": page[0]},
            )
            for page in pages
        ]

    def _document(
        self,
        pages: list[tuple[int, str, str | None]],
        uri: str | None,
        title: str | None,
        extractor: str,
        metadata: dict[str, Any],
    ) -> Document:
        out = SourceMapBuilder("pdf", extractor=extractor)
        for number, text, label in pages:
            if not text:
                continue
            if out.length:
                out.insert(self.page_separator)
            locator: dict[str, Any] = {"page": number}
            if label is not None and label != str(number):
                locator["label"] = label
            out.add(text, out.unit(locator), 0)
        return Document(
            text=out.text,
            uri=uri,
            title=title,
            modality=self.modality,
            tier=self.tier,
            metadata={**metadata, "source_map": out.build()},
        )


def _labels(reader: Any) -> dict[int, str]:
    try:
        return {number: str(label) for number, label in enumerate(reader.page_labels, 1)}
    except Exception:  # a malformed label tree must not cost the file its text
        return {}


def _title(reader: Any) -> str | None:
    info = reader.metadata
    value = info.title if info is not None else None
    return str(value).strip() or None if value else None


__all__ = ["PdfLoader"]
